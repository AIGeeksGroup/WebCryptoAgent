from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
from openai import OpenAI

from .core.llm import build_l1_bundle, run_l2_reasoner
from .core.market import build_indicator_signal
from .utils.indicators_calc import compute_indicators

Ticker_API = ""
REQ_TIMEOUT = 15


@dataclass
class ModelConfig:
    provider: str
    model: str
    api_key: str
    base_url: str


@dataclass
class ModelBacktestResult:
    model_key: str
    provider: str
    model: str
    symbol: str
    start: datetime
    end: datetime
    bars: int
    decisions: int
    trades: int
    win_rate: float
    total_return: float
    cagr: float
    max_drawdown: float
    sharpe: float
    avg_trade_return: float
    median_trade_return: float
    equity_end: float
    initial_equity: float
    memory_trades: int
    llm_fallbacks: int
    memory_enabled: bool


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        if key not in os.environ:
            os.environ[key] = value


def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _interval_to_minutes(interval: str) -> int:
    interval = interval.strip().lower()
    if interval.endswith("m"):
        return int(interval[:-1])
    if interval.endswith("h"):
        return int(interval[:-1]) * 60
    if interval.endswith("d"):
        return int(interval[:-1]) * 24 * 60
    raise ValueError(f"Unsupported interval: {interval}")


def _extract_base_coin(symbol: str) -> str:
    upper = symbol.upper()
    for suffix in ("USDC.E", "USDT", "USDC", "USD"):
        if upper.endswith(suffix):
            return symbol[: -len(suffix)]
    return symbol


def _load_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename = {
        "iso": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Vol",
        "trades": "Trades",
    }
    df = df.rename(columns=rename)
    if "Open" in df.columns:
        df["open"] = df["Open"]
    if "High" in df.columns:
        df["high"] = df["High"]
    if "Low" in df.columns:
        df["low"] = df["Low"]
    if "Close" in df.columns:
        df["close"] = df["Close"]
    if "Trades" in df.columns:
        df["trade"] = df["Trades"]
    elif "trades" in df.columns:
        df["trade"] = df["trades"]
    if "Date" not in df.columns and "ts" in df.columns:
        df["Date"] = pd.to_datetime(df["ts"], unit="s", utc=True, errors="coerce")
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    return df


def _ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" in df.columns:
        return df
    if "iso" in df.columns:
        df["Date"] = pd.to_datetime(df["iso"], utc=True, errors="coerce")
        return df
    if "ts" in df.columns:
        df["Date"] = pd.to_datetime(df["ts"], unit="s", utc=True, errors="coerce")
        return df
    return df


def _ensure_window(
    df: pd.DataFrame,
    start: datetime,
    end: datetime,
    symbol: str,
) -> Tuple[pd.DataFrame, datetime, datetime]:
    if df.empty or "Date" not in df.columns:
        raise ValueError(f"{symbol}: missing Date column or empty dataframe")
    min_date = df["Date"].min()
    max_date = df["Date"].max()
    if end > max_date:
        end = max_date
    if start < min_date:
        print(f"[WARN] {symbol}: data starts {min_date.date()} < requested {start.date()}; using available range.")
        start = min_date
    df = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()
    return df, start, end


def _fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> List[List[Any]]:
    bars: List[List[Any]] = []
    while start_ms < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": 1000,
        }
        resp = requests.get(Ticker_API, params=params, timeout=REQ_TIMEOUT)
        if resp.status_code != 200:
            raise RuntimeError(f"Ticker {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        if not data:
            break
        bars.extend(data)
        start_ms = int(data[-1][0]) + 1
        time.sleep(0.2)
    return bars


def _save_ohlcv_csv(path: Path, bars: List[List[Any]]) -> None:
    rows = []
    for b in bars:
        ts = int(b[0]) // 1000
        iso = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        rows.append(
            {
                "ts": ts,
                "iso": iso,
                "open": b[1],
                "high": b[2],
                "low": b[3],
                "close": b[4],
                "volume": b[5],
                "trades": b[8],
            }
        )
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _ensure_ohlcv_file(
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
    data_dir: Path,
    fetch: bool,
) -> Path:
    path = data_dir / f"{symbol}_{interval}.csv"
    if path.exists():
        df = _load_ohlcv(path)
        df = _ensure_date_column(df)
        if not df.empty:
            min_date = df["Date"].min()
            max_date = df["Date"].max()
            if min_date <= start and max_date >= end:
                return path
    if not fetch:
        raise FileNotFoundError(f"{symbol}: missing coverage for {start.date()} → {end.date()} in {path}")

    print(f"[INFO] Fetching {symbol} {interval} data from ticker...")
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    bars = _fetch_klines(symbol, interval, start_ms, end_ms)
    if not bars:
        raise RuntimeError(f"{symbol}: ticker returned no bars for {start.date()} → {end.date()}")
    _save_ohlcv_csv(path, bars)
    print(f"[INFO] Saved {len(bars)} rows → {path}")
    return path


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _build_features(row: pd.Series, score: float, confidence: float) -> Dict[str, float]:
    rsi = row.get("rsi")
    macd_hist = row.get("macd_hist")
    vol_z = row.get("vol_z")
    ema_fast = row.get("ema_fast")
    ema_slow = row.get("ema_slow")

    features: Dict[str, float] = {
        "signal_score": float(score),
        "signal_confidence": float(confidence),
    }

    if pd.notna(rsi):
        features["rsi_norm"] = _clamp((float(rsi) - 50.0) / 25.0, -1.0, 1.0)
    if pd.notna(macd_hist):
        features["macd_hist_sign"] = 1.0 if macd_hist > 0 else -1.0 if macd_hist < 0 else 0.0
    if pd.notna(vol_z):
        features["vol_z_norm"] = _clamp(float(vol_z) / 3.0, -1.0, 1.0)
    if pd.notna(ema_fast) and pd.notna(ema_slow):
        features["trend_sign"] = 1.0 if ema_fast > ema_slow else -1.0 if ema_fast < ema_slow else 0.0

    return features


def _extract_experience_features(exp: Dict[str, Any]) -> Dict[str, float]:
    features: Dict[str, float] = {}
    memory = exp.get("memory") or {}
    mem_features = memory.get("features")
    if isinstance(mem_features, dict):
        for key, val in mem_features.items():
            try:
                features[key] = float(val)
            except (TypeError, ValueError):
                continue

    if "signal_score" not in features:
        signal = exp.get("signal") or {}
        p_up = signal.get("p_up")
        if p_up is None:
            p_up = exp.get("p_up")
        if p_up is not None:
            try:
                features["signal_score"] = _clamp(float(p_up) * 2.0 - 1.0, -1.0, 1.0)
            except (TypeError, ValueError):
                pass

    if "signal_confidence" not in features:
        signal = exp.get("signal") or {}
        conf = signal.get("confidence", exp.get("confidence"))
        if conf is not None:
            try:
                features["signal_confidence"] = _clamp(float(conf), 0.0, 1.0)
            except (TypeError, ValueError):
                pass

    sentiment = (
        exp.get("bundle_snapshot", {})
        .get("aggregates", {})
        .get("sentiment", {})
        .get("level")
    )
    if sentiment is not None:
        try:
            features.setdefault("sentiment_level", _clamp(float(sentiment), -1.0, 1.0))
        except (TypeError, ValueError):
            pass

    risk_score = exp.get("risk_metrics", {}).get("risk_score")
    if risk_score is None:
        risk_score = exp.get("risk_score")
    if risk_score is not None:
        try:
            features.setdefault("risk_score", _clamp(float(risk_score), 0.0, 1.0))
        except (TypeError, ValueError):
            pass

    return features


def _similarity(current: Dict[str, float], past: Dict[str, float]) -> Optional[float]:
    keys = set(current) & set(past)
    if not keys:
        return None
    dist = 0.0
    for key in keys:
        dist += (current[key] - past[key]) ** 2
    dist = math.sqrt(dist / len(keys))
    return 1.0 / (1.0 + dist)


def _select_memory(
    current_features: Dict[str, float],
    experiences: List[Dict[str, Any]],
    top_k: int,
    min_similarity: float,
    symbol: str,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for exp in experiences:
        exp_symbol = exp.get("symbol") or exp.get("coin")
        if exp_symbol and exp_symbol != symbol:
            continue
        exp_features = _extract_experience_features(exp)
        sim = _similarity(current_features, exp_features)
        if sim is None or sim < min_similarity:
            continue
        scored.append((sim, exp))

    if not scored:
        for exp in experiences:
            exp_features = _extract_experience_features(exp)
            sim = _similarity(current_features, exp_features)
            if sim is None or sim < min_similarity:
                continue
            scored.append((sim, exp))

    if not scored:
        return [], []

    scored.sort(key=lambda item: item[0], reverse=True)
    picked = scored[: max(1, top_k)]
    summaries = []
    ids: List[str] = []
    for sim, exp in picked:
        exp_id = str(exp.get("id", ""))
        ids.append(exp_id)
        summaries.append(
            {
                "id": exp_id,
                "action": exp.get("action") or (exp.get("signal") or {}).get("action"),
                "reward": exp.get("reward") or exp.get("outcome", {}).get("return"),
                "features": exp.get("memory", {}).get("features") or _extract_experience_features(exp),
                "similarity": round(sim, 4),
                "timestamp": exp.get("closed_at") or exp.get("created_at"),
            }
        )
    return summaries, ids


def _format_num(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(val):
        return "n/a"
    return f"{val:.{digits}f}"


def _indicator_context(row: pd.Series, score: float, confidence: float) -> Dict[str, Any]:
    close = row.get("Close")
    ema_fast = row.get("ema_fast")
    ema_slow = row.get("ema_slow")
    rsi = row.get("rsi")
    macd_hist = row.get("macd_hist")
    vol_z = row.get("vol_z")
    atr = row.get("atr")
    bb_width = row.get("bb_width")
    pdh = row.get("PDH")
    pdl = row.get("PDL")

    summary = (
        "signal "
        f"score={_format_num(score, 2)} conf={_format_num(confidence, 2)} | "
        f"ema_fast={_format_num(ema_fast, 4)} ema_slow={_format_num(ema_slow, 4)} | "
        f"rsi={_format_num(rsi, 2)} macd_hist={_format_num(macd_hist, 4)} | "
        f"vol_z={_format_num(vol_z, 2)} atr={_format_num(atr, 4)} bb_width={_format_num(bb_width, 4)} | "
        f"pdh={_format_num(pdh, 4)} pdl={_format_num(pdl, 4)}"
    )

    numeric: Dict[str, float] = {}
    for key in [
        "ema_fast",
        "ema_slow",
        "rsi",
        "macd",
        "macd_signal",
        "macd_hist",
        "atr",
        "bb_mid",
        "bb_upper",
        "bb_lower",
        "bb_width",
        "PDH",
        "PDL",
        "vol_z",
    ]:
        val = row.get(key)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            continue
        try:
            numeric[key] = float(val)
        except (TypeError, ValueError):
            continue

    if close is not None and atr is not None:
        try:
            close_val = float(close)
            atr_val = float(atr)
            if close_val > 0:
                numeric["atr_bps"] = (atr_val / close_val) * 10000.0
        except (TypeError, ValueError):
            pass

    if close is not None and row.get("bb_upper") is not None and row.get("bb_lower") is not None:
        try:
            close_val = float(close)
            upper = float(row.get("bb_upper"))
            lower = float(row.get("bb_lower"))
            if upper != lower:
                numeric["boll_pctB_20"] = (close_val - lower) / (upper - lower)
        except (TypeError, ValueError):
            pass

    return {"summary": summary, "numeric": numeric}


def _indicator_docs(
    coin: str,
    row: pd.Series,
    score: float,
    confidence: float,
    signal_label: str,
    window: str,
    price_change_pct: Optional[float],
) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    docs.append(
        {
            "source": "indicator_snapshot",
            "url": f"memory://indicator/{coin}",
            "title": f"{coin} indicator snapshot",
            "text": f"{coin} indicator signal={signal_label} score={score:.2f} conf={confidence:.2f}",
            "coins": [coin],
            "hash": f"indicator-{coin}-{uuid.uuid4().hex}",
            "category": "market",
            "meta": {
                "signal_score": score,
                "signal_confidence": confidence,
                "signal_label": signal_label,
                "window": window,
            },
        }
    )

    atr = row.get("atr")
    vol_z = row.get("vol_z")
    bb_width = row.get("bb_width")
    ema_fast = row.get("ema_fast")
    ema_slow = row.get("ema_slow")

    def _as_float(val: Any) -> Optional[float]:
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    atr_val = _as_float(atr)
    vol_val = _as_float(vol_z)
    bb_val = _as_float(bb_width)
    trend_penalty = 0.0
    if _as_float(ema_fast) is not None and _as_float(ema_slow) is not None:
        trend_penalty = 0.2 if float(ema_fast) < float(ema_slow) else 0.0

    components = []
    if atr_val is not None:
        components.append(min(1.0, max(0.0, atr_val / 150)))
    if vol_val is not None:
        components.append(min(1.0, abs(vol_val) / 3))
    if bb_val is not None:
        components.append(min(1.0, bb_val / 0.15))
    base_score = sum(components) / len(components) if components else 0.0
    risk_score = _clamp(base_score + trend_penalty, 0.0, 1.0)

    docs.append(
        {
            "source": "pool_risk_model",
            "url": f"memory://risk/{coin}",
            "title": f"{coin} risk snapshot",
            "text": f"{coin} risk score={risk_score:.2f}",
            "coins": [coin],
            "hash": f"risk-{coin}-{uuid.uuid4().hex}",
            "category": "risk",
            "meta": {"risk_score": risk_score},
        }
    )

    if price_change_pct is not None:
        docs.append(
            {
                "source": "price_page",
                "url": f"memory://price/{coin}",
                "title": f"{coin} 24h price change",
                "text": f"{coin} 24h change {price_change_pct:.2f}%",
                "coins": [coin],
                "hash": f"price-{coin}-{uuid.uuid4().hex}",
                "category": "market",
                "meta": {"price_change_pct": price_change_pct},
            }
        )
    return docs


def _max_drawdown(equity: Iterable[float]) -> float:
    peak = -math.inf
    max_dd = 0.0
    for val in equity:
        peak = max(peak, val)
        if peak > 0:
            max_dd = min(max_dd, (val / peak) - 1.0)
    return abs(max_dd)


def _sharpe(bar_returns: List[float], bars_per_year: int) -> float:
    if not bar_returns:
        return 0.0
    mean = sum(bar_returns) / len(bar_returns)
    variance = sum((r - mean) ** 2 for r in bar_returns) / len(bar_returns)
    std = math.sqrt(variance)
    if std == 0:
        return 0.0
    return (mean / std) * math.sqrt(bars_per_year)


def _safe_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())


def _safe_model_key(provider: str, model: str) -> str:
    return _safe_key(f"{provider}_{model}")


def _collect_models() -> List[ModelConfig]:
    models: List[ModelConfig] = []
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        models.append(
            ModelConfig(
                provider="openai",
                model=os.environ.get("OPENAI_MODEL", "gpt-5.2"),
                api_key=openai_key,
                base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            )
        )
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        models.append(
            ModelConfig(
                provider="gemini",
                model=os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview"),
                api_key=gemini_key,
                base_url=os.environ.get(
                    "GEMINI_BASE_URL",
                    "https://generativelanguage.googleapis.com/v1beta/openai/",
                ),
            )
        )
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
    if deepseek_key:
        models.append(
            ModelConfig(
                provider="deepseek",
                model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
                api_key=deepseek_key,
                base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            )
        )
    qwen_key = os.environ.get("QWEN_API_KEY")
    if qwen_key:
        models.append(
            ModelConfig(
                provider="qwen",
                model=os.environ.get("QWEN_MODEL", "qwen-max"),
                api_key=qwen_key,
                base_url=os.environ.get(
                    "QWEN_BASE_URL",
                    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                ),
            )
        )
    return models


def _load_experiences(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            agent = (item.get("agent") or "").lower()
            if agent in {"news", "social"}:
                continue
            source = (item.get("source") or "").lower()
            if source and not source.startswith("backtest"):
                continue
            items.append(item)
    return items


def _append_experiences(path: Path, records: List[Dict[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run_model_backtest(
    df: pd.DataFrame,
    symbol: str,
    model_cfg: ModelConfig,
    client: OpenAI,
    interval: str,
    decision_interval: int,
    fee_bps: float,
    initial_equity: float,
    default_size_pct: float,
    experiences: List[Dict[str, Any]],
    memory_top_k: int,
    memory_min_similarity: float,
    llm_sleep: float,
    use_memory: bool,
) -> Tuple[ModelBacktestResult, pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
    if len(df) < 3:
        raise ValueError(f"{symbol}: insufficient data ({len(df)} rows)")

    df = df.sort_values("Date").reset_index(drop=True)
    coin = _extract_base_coin(symbol)
    fee_rate = fee_bps / 10000.0
    decision_interval = max(1, decision_interval)
    default_size_pct = _clamp(default_size_pct, 0.0, 1.0)
    cash = float(initial_equity)
    position_qty = 0.0
    avg_entry = 0.0
    entry_time = None
    entry_index = None
    entry_features: Optional[Dict[str, float]] = None
    entry_memory_ids: List[str] = []

    equity_curve: List[Dict[str, Any]] = []
    trades: List[Dict[str, Any]] = []
    new_experiences: List[Dict[str, Any]] = []
    bar_returns: List[float] = []
    memory_hits = 0
    llm_fallbacks = 0
    decisions = 0
    last_action = "HOLD"
    last_p_up = 0.5
    memory_pool = list(experiences) if use_memory else []

    interval_minutes = _interval_to_minutes(interval)
    bars_per_24h = max(1, int(24 * 60 / interval_minutes))

    for i in range(len(df) - 1):
        row = df.iloc[i]
        next_row = df.iloc[i + 1]
        prev_equity = cash + position_qty * float(row["Close"])

        decision = (i % decision_interval == 0)
        memory_context: List[Dict[str, Any]] = []
        memory_ids: List[str] = []
        signal_meta = None
        score = 0.0
        confidence = 0.0
        features: Optional[Dict[str, float]] = None

        if decision:
            signal_meta = build_indicator_signal(row)
            score = float(signal_meta.get("signal_score") or 0.0)
            confidence = float(signal_meta.get("signal_confidence") or 0.0)
            features = _build_features(row, score, confidence)
            if use_memory and memory_pool:
                memory_context, memory_ids = _select_memory(
                    features,
                    memory_pool,
                    top_k=memory_top_k,
                    min_similarity=memory_min_similarity,
                    symbol=symbol,
                )
                if memory_context:
                    memory_hits += 1

            change_pct = None
            if i - bars_per_24h >= 0:
                prev_close = df.iloc[i - bars_per_24h]["Close"]
                try:
                    prev_close_val = float(prev_close)
                    close_val = float(row["Close"])
                    if prev_close_val > 0:
                        change_pct = (close_val / prev_close_val - 1.0) * 100.0
                except (TypeError, ValueError):
                    change_pct = None

            indicator_context = _indicator_context(row, score, confidence)
            docs = _indicator_docs(
                coin,
                row,
                score,
                confidence,
                signal_meta.get("signal_label", "neutral"),
                interval,
                change_pct,
            )
            bundle = build_l1_bundle(
                coin=coin,
                window=interval,
                window_start=str(row["Date"]),
                window_end=str(next_row["Date"]),
                docs=docs,
                client=client,
                model=model_cfg.model,
            )
            wallet = {
                "cash_usd": round(cash, 2),
                "position_btc": round(position_qty, 8),
                "budget_cap_usd": round(prev_equity, 2),
            }
            signal = run_l2_reasoner(
                client,
                model_cfg.model,
                bundle,
                chart_b64=None,
                wallet=wallet,
                last_price_usd=float(row["Close"]),
                experiences=memory_context if use_memory else None,
                indicator_context=indicator_context,
            )
            decisions += 1
            last_action = signal.get("action", "HOLD")
            last_p_up = float(signal.get("p_up", 0.5) or 0.5)
            reasoning = signal.get("reasoning", "")
            if isinstance(reasoning, str) and reasoning.startswith("Fallback"):
                llm_fallbacks += 1

            open_next = float(next_row["Open"])
            size_usd = float(signal.get("size_usd") or 0.0)
            size_btc = float(signal.get("size_btc") or 0.0)

            if last_action == "BUY":
                if size_usd <= 0:
                    size_usd = max(0.0, min(cash, prev_equity * default_size_pct))
                size_usd = min(size_usd, cash)
                buy_qty = (size_usd / open_next) if open_next > 0 else 0.0
                if buy_qty > 0:
                    fee = buy_qty * open_next * fee_rate
                    cash -= buy_qty * open_next + fee
                    if position_qty > 0:
                        avg_entry = (avg_entry * position_qty + open_next * buy_qty) / (position_qty + buy_qty)
                    else:
                        avg_entry = open_next
                        entry_time = next_row["Date"]
                        entry_index = i + 1
                        entry_features = features
                        entry_memory_ids = memory_ids
                    position_qty += buy_qty

            elif last_action in {"SELL", "HOLD"}:
                if position_qty > 0:
                    if last_action == "HOLD":
                        sell_qty = position_qty
                    else:
                        if size_btc > 0:
                            sell_qty = min(size_btc, position_qty)
                        elif size_usd > 0:
                            sell_qty = min(size_usd / open_next, position_qty)
                        else:
                            sell_qty = position_qty
                    if sell_qty > 0:
                        fee = sell_qty * open_next * fee_rate
                        cash += sell_qty * open_next - fee
                        pnl = (open_next - avg_entry) * sell_qty
                        ret = (pnl / (avg_entry * sell_qty)) if avg_entry > 0 else 0.0
                        exp_id = uuid.uuid4().hex
                        trades.append(
                            {
                                "model": model_cfg.model,
                                "symbol": symbol,
                                "side": "long",
                                "entry_time": entry_time,
                                "exit_time": next_row["Date"],
                                "entry_price": avg_entry,
                                "exit_price": open_next,
                                "size_qty": sell_qty,
                                "return_pct": ret * 100.0,
                                "bars_held": (i + 1 - (entry_index or i + 1)),
                                "memory_matches": len(entry_memory_ids),
                                "memory_ids": ";".join(entry_memory_ids),
                            }
                        )
                        new_experiences.append(
                            {
                                "id": exp_id,
                                "coin": symbol,
                                "symbol": symbol,
                                "action": "BUY",
                                "reward": ret,
                                "created_at": entry_time.isoformat() if entry_time else None,
                                "closed_at": next_row["Date"].isoformat(),
                                "source": "backtest",
                                "agent": f"llm_{model_cfg.provider}",
                                "memory": {
                                    "features": entry_features or {},
                                    "memory_ids": entry_memory_ids,
                                },
                                "outcome": {
                                    "return": ret,
                                    "entry_price": avg_entry,
                                    "exit_price": open_next,
                                },
                            }
                        )
                        if use_memory:
                            memory_pool.append(new_experiences[-1])
                        position_qty -= sell_qty
                        if position_qty <= 1e-12:
                            position_qty = 0.0
                            avg_entry = 0.0
                            entry_time = None
                            entry_index = None
                            entry_features = None
                            entry_memory_ids = []

            if llm_sleep > 0:
                time.sleep(llm_sleep)

        close_next = float(next_row["Close"])
        equity = cash + position_qty * close_next
        bar_returns.append((equity / prev_equity) - 1.0 if prev_equity else 0.0)
        equity_curve.append(
            {
                "Date": next_row["Date"],
                "equity": equity,
                "cash": cash,
                "position_qty": position_qty,
                "action": last_action,
                "p_up": last_p_up,
                "memory_matches": len(memory_ids),
            }
        )

    if position_qty > 0:
        last_row = df.iloc[-1]
        exit_price = float(last_row["Close"])
        fee = position_qty * exit_price * fee_rate
        cash += position_qty * exit_price - fee
        pnl = (exit_price - avg_entry) * position_qty
        ret = (pnl / (avg_entry * position_qty)) if avg_entry > 0 else 0.0
        exp_id = uuid.uuid4().hex
        trades.append(
            {
                "model": model_cfg.model,
                "symbol": symbol,
                "side": "long",
                "entry_time": entry_time,
                "exit_time": last_row["Date"],
                "entry_price": avg_entry,
                "exit_price": exit_price,
                "size_qty": position_qty,
                "return_pct": ret * 100.0,
                "bars_held": len(df) - 1 - (entry_index or len(df) - 1),
                "memory_matches": len(entry_memory_ids),
                "memory_ids": ";".join(entry_memory_ids),
            }
        )
        new_experiences.append(
            {
                "id": exp_id,
                "coin": symbol,
                "symbol": symbol,
                "action": "BUY",
                "reward": ret,
                "created_at": entry_time.isoformat() if entry_time else None,
                "closed_at": last_row["Date"].isoformat(),
                "source": "backtest",
                "agent": f"llm_{model_cfg.provider}",
                "memory": {
                    "features": entry_features or {},
                    "memory_ids": entry_memory_ids,
                },
                "outcome": {
                    "return": ret,
                    "entry_price": avg_entry,
                    "exit_price": exit_price,
                },
            }
        )
        if use_memory:
            memory_pool.append(new_experiences[-1])
        position_qty = 0.0
        avg_entry = 0.0
        entry_time = None
        entry_index = None
        entry_features = None
        entry_memory_ids = []
        if equity_curve:
            equity_curve[-1]["equity"] = cash
            equity_curve[-1]["cash"] = cash
            equity_curve[-1]["position_qty"] = 0.0

    trade_returns = [t["return_pct"] / 100.0 for t in trades]
    wins = sum(1 for r in trade_returns if r > 0)
    win_rate = wins / len(trade_returns) if trade_returns else 0.0
    avg_trade = sum(trade_returns) / len(trade_returns) if trade_returns else 0.0
    median_trade = sorted(trade_returns)[len(trade_returns) // 2] if trade_returns else 0.0
    equity_end = cash
    total_return = (equity_end / initial_equity) - 1.0

    start = df["Date"].iloc[0]
    end = df["Date"].iloc[-1]
    days = max(1, int((end - start).total_seconds() / 86400))
    cagr = ((equity_end / initial_equity) ** (365 / days) - 1.0) if days > 0 else 0.0

    bars_per_year = int(365 * 24 * 60 / _interval_to_minutes(interval))
    summary = ModelBacktestResult(
        model_key=_safe_model_key(model_cfg.provider, model_cfg.model),
        provider=model_cfg.provider,
        model=model_cfg.model,
        symbol=symbol,
        start=start,
        end=end,
        bars=len(df),
        decisions=decisions,
        trades=len(trades),
        win_rate=win_rate,
        total_return=total_return,
        cagr=cagr,
        max_drawdown=_max_drawdown([e["equity"] for e in equity_curve]),
        sharpe=_sharpe(bar_returns, bars_per_year=bars_per_year),
        avg_trade_return=avg_trade,
        median_trade_return=median_trade,
        equity_end=equity_end,
        initial_equity=initial_equity,
        memory_trades=memory_hits,
        llm_fallbacks=llm_fallbacks,
        memory_enabled=use_memory,
    )
    return summary, pd.DataFrame(trades), pd.DataFrame(equity_curve), new_experiences


def _write_outputs(
    out_dir: Path,
    result_key: str,
    summary: ModelBacktestResult,
    trades: pd.DataFrame,
    equity: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"{result_key}_summary.json"
    trades_path = out_dir / f"{result_key}_trades.csv"
    equity_path = out_dir / f"{result_key}_equity.csv"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary.__dict__, fh, ensure_ascii=False, indent=2, default=str)
    trades.to_csv(trades_path, index=False)
    equity.to_csv(equity_path, index=False)


def _plot_equity_curves(
    out_dir: Path,
    curves: Dict[str, pd.DataFrame],
    initial_equity: float,
    title: str,
    filename: str = "equity_compare.png",
) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable: {exc}")
        return None

    fig, ax = plt.subplots(figsize=(11, 6))
    for label, df in curves.items():
        if df.empty:
            continue
        tmp = df.copy()
        tmp["Date"] = pd.to_datetime(tmp["Date"], utc=True, errors="coerce")
        tmp = tmp.dropna(subset=["Date"])
        tmp["cum_return"] = (tmp["equity"] / initial_equity) - 1.0
        ax.plot(tmp["Date"], tmp["cum_return"], label=label)
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel("Cumulative Return")
    ax.set_xlabel("Date")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    default_output = Path(__file__).resolve().parents[2] / "results" / "backtests"
    default_experience_log = Path(__file__).resolve().parents[1] / "data" / "experiences.jsonl"
    default_data_dir = Path(__file__).resolve().parents[1] / "data" / "ohlcv"
    default_env = Path(__file__).resolve().parents[2] / ".env.backtest"

    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument(
        "--symbols",
        type=str,
        default="",
        help="Comma-separated symbols to backtest (overrides --symbol)",
    )
    parser.add_argument("--interval", type=str, default="15m")
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--start", type=str, default="")
    parser.add_argument("--end", type=str, default="")
    parser.add_argument("--data-dir", type=str, default=str(default_data_dir))
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--decision-interval", type=int, default=96)
    parser.add_argument("--fee-bps", type=float, default=10.0)
    parser.add_argument("--initial-equity", type=float, default=10000.0)
    parser.add_argument("--default-size-pct", type=float, default=0.2)
    parser.add_argument("--memory-top-k", type=int, default=3)
    parser.add_argument("--memory-min-similarity", type=float, default=0.15)
    parser.add_argument("--memory-append", action="store_true")
    parser.add_argument("--disable-memory", action="store_true")
    parser.add_argument("--llm-sleep", type=float, default=0.1)
    parser.add_argument("--output-dir", type=str, default=str(default_output))
    parser.add_argument("--env-file", type=str, default=str(default_env))
    args = parser.parse_args()

    _load_env_file(Path(args.env_file))
    models = _collect_models()
    if not models:
        print("[ERROR] No LLM API keys found in environment.")
        return

    end = _parse_iso(args.end) if args.end else datetime.now(timezone.utc)
    start = _parse_iso(args.start) if args.start else end - timedelta(days=args.lookback_days)
    data_dir = Path(args.data_dir)

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        symbols = [args.symbol]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / run_id
    experience_log = Path(default_experience_log)
    base_experiences = _load_experiences(experience_log)

    summaries: List[Dict[str, Any]] = []
    curves_by_symbol: Dict[str, Dict[str, pd.DataFrame]] = {}
    new_records: List[Dict[str, Any]] = []

    use_memory = not args.disable_memory
    llm_timeout = float(os.environ.get("LLM_TIMEOUT", "60"))
    llm_client_retries = int(os.environ.get("LLM_CLIENT_RETRIES", "0"))

    for symbol in symbols:
        try:
            _ensure_ohlcv_file(
                symbol=symbol,
                interval=args.interval,
                start=start,
                end=end,
                data_dir=data_dir,
                fetch=args.fetch,
            )
        except Exception as exc:
            print(f"[WARN] {symbol}: {exc}")
            continue

        ohlcv_path = data_dir / f"{symbol}_{args.interval}.csv"
        df_raw = _load_ohlcv(ohlcv_path)
        df_ind = compute_indicators(df_raw, by_ticker=False)
        df_ind = _ensure_date_column(df_ind)
        df_window, start_ts, end_ts = _ensure_window(df_ind, start, end, symbol)
        if df_window.empty:
            print(f"[WARN] {symbol}: no data in requested window.")
            continue

        for model_cfg in models:
            client = OpenAI(
                api_key=model_cfg.api_key,
                base_url=model_cfg.base_url,
                timeout=llm_timeout,
                max_retries=llm_client_retries,
            )
            model_key = _safe_model_key(model_cfg.provider, model_cfg.model)
            result_key = _safe_key(f"{model_key}_{symbol}")
            summary, trades, equity, new_exps = run_model_backtest(
                df_window,
                symbol,
                model_cfg,
                client,
                interval=args.interval,
                decision_interval=args.decision_interval,
                fee_bps=args.fee_bps,
                initial_equity=args.initial_equity,
                default_size_pct=args.default_size_pct,
                experiences=list(base_experiences),
                memory_top_k=args.memory_top_k,
                memory_min_similarity=args.memory_min_similarity,
                llm_sleep=args.llm_sleep,
                use_memory=use_memory,
            )
            _write_outputs(out_dir, result_key, summary, trades, equity)
            summaries.append(summary.__dict__)
            curves_by_symbol.setdefault(symbol, {})[model_key] = equity
            new_records.extend(new_exps)

            print(
                f"{model_key} {symbol} {summary.start.date()} → {summary.end.date()} | "
                f"trades={summary.trades} return={summary.total_return:.2%} "
                f"mdd={summary.max_drawdown:.2%} sharpe={summary.sharpe:.2f}"
            )

    if not summaries:
        print("[ERROR] No backtest results generated.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary_all.json").open("w", encoding="utf-8") as fh:
        json.dump(summaries, fh, ensure_ascii=False, indent=2, default=str)

    table = pd.DataFrame(summaries)
    table_path = out_dir / "model_comparison.csv"
    table.to_csv(table_path, index=False)
    table_md = out_dir / "model_comparison.md"
    try:
        table.to_markdown(table_md, index=False)
    except Exception as exc:
        print(f"[WARN] Failed to write markdown table: {exc}")

    plot_paths: List[Path] = []
    for symbol, curves in curves_by_symbol.items():
        if not curves:
            continue
        suffix = _safe_key(symbol)
        filename = "equity_compare.png"
        if len(curves_by_symbol) > 1:
            filename = f"equity_compare_{suffix}.png"
        plot_path = _plot_equity_curves(
            out_dir,
            curves,
            initial_equity=args.initial_equity,
            title=f"Cumulative Return ({symbol})",
            filename=filename,
        )
        if plot_path:
            plot_paths.append(plot_path)

    print(f"Saved results to {out_dir}")
    for path in plot_paths:
        print(f"Saved equity comparison chart to {path}")
    if args.memory_append:
        _append_experiences(experience_log, new_records)
        print(f"Appended {len(new_records)} experiences to {experience_log}")


if __name__ == "__main__":
    main()
