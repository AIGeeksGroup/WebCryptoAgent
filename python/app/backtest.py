from __future__ import annotations

import argparse
import json
import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from .core.market import build_indicator_signal
from .utils.indicators_calc import compute_indicators


@dataclass
class BacktestResult:
    symbol: str
    start: datetime
    end: datetime
    bars: int
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


def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _resolve_data_dir(path_str: str | None) -> Path:
    if path_str:
        return Path(path_str)
    return Path(__file__).resolve().parents[2] / "data" / "ohlcv"


def _list_csv_symbols(data_dir: Path) -> List[str]:
    symbols = []
    for path in sorted(data_dir.glob("*_15m.csv")):
        symbols.append(path.stem.replace("_15m", ""))
    return symbols


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
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


def _append_experiences(path: Path, records: List[Dict[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


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


def _ensure_window(
    df: pd.DataFrame,
    start: Optional[datetime],
    end: Optional[datetime],
    lookback_days: int,
    symbol: str,
) -> Tuple[pd.DataFrame, datetime, datetime]:
    if df.empty or "Date" not in df.columns:
        raise ValueError(f"{symbol}: missing Date column or empty dataframe")
    min_date = df["Date"].min()
    max_date = df["Date"].max()
    if end is None:
        end = max_date
    if start is None:
        start = end - timedelta(days=lookback_days)
    if end > max_date:
        end = max_date
    if start < min_date:
        print(f"[WARN] {symbol}: data starts {min_date.date()} < requested {start.date()}; using available range.")
        start = min_date
    df = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()
    return df, start, end


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


def _add_signal_columns(df: pd.DataFrame) -> pd.DataFrame:
    def _signal(row: pd.Series) -> pd.Series:
        meta = build_indicator_signal(row)
        return pd.Series(
            {
                "signal_score": meta.get("signal_score", 0.0),
                "signal_label": meta.get("signal_label", "neutral"),
                "signal_confidence": meta.get("signal_confidence", 0.0),
            }
        )

    signals = df.apply(_signal, axis=1)
    return pd.concat([df, signals], axis=1)


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


def _experience_direction(exp: Dict[str, Any]) -> int:
    action = exp.get("action") or (exp.get("signal") or {}).get("action")
    if not action:
        return 0
    action = str(action).upper()
    if action == "BUY":
        return 1
    if action == "SELL":
        return -1
    return 0


def _experience_reward(exp: Dict[str, Any]) -> Optional[float]:
    reward = exp.get("reward")
    if reward is None:
        reward = exp.get("outcome", {}).get("pnl")
    if reward is None:
        reward = exp.get("outcome", {}).get("return")
    if reward is None:
        reward = exp.get("result")
    if reward is None:
        return None
    try:
        return float(reward)
    except (TypeError, ValueError):
        return None


def _similarity(current: Dict[str, float], past: Dict[str, float]) -> Optional[float]:
    keys = set(current) & set(past)
    if not keys:
        return None
    dist = 0.0
    for key in keys:
        dist += (current[key] - past[key]) ** 2
    dist = math.sqrt(dist / len(keys))
    return 1.0 / (1.0 + dist)


def _memory_bias(
    current_features: Dict[str, float],
    experiences: List[Dict[str, Any]],
    top_k: int,
    reward_scale: float,
    min_similarity: float,
    symbol: str,
) -> Tuple[float, List[str]]:
    scored: List[Tuple[float, float, str]] = []
    for exp in experiences:
        exp_symbol = exp.get("symbol") or exp.get("coin")
        if exp_symbol and exp_symbol != symbol:
            continue
        exp_features = _extract_experience_features(exp)
        sim = _similarity(current_features, exp_features)
        if sim is None or sim < min_similarity:
            continue
        reward = _experience_reward(exp)
        if reward is None:
            continue
        direction = _experience_direction(exp)
        if direction == 0:
            continue
        norm = _clamp(reward / reward_scale, -1.0, 1.0)
        scored.append((sim, direction * norm, str(exp.get("id", ""))))

    if not scored:
        for exp in experiences:
            exp_features = _extract_experience_features(exp)
            sim = _similarity(current_features, exp_features)
            if sim is None or sim < min_similarity:
                continue
            reward = _experience_reward(exp)
            if reward is None:
                continue
            direction = _experience_direction(exp)
            if direction == 0:
                continue
            norm = _clamp(reward / reward_scale, -1.0, 1.0)
            scored.append((sim, direction * norm, str(exp.get("id", ""))))

    if not scored:
        return 0.0, []

    scored.sort(key=lambda item: item[0], reverse=True)
    top = scored[: max(1, top_k)]
    total_weight = sum(sim for sim, _, _ in top) or 1.0
    bias = sum(sim * val for sim, val, _ in top) / total_weight
    return _clamp(bias, -1.0, 1.0), [exp_id for _, _, exp_id in top if exp_id]


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


def run_backtest(
    df: pd.DataFrame,
    symbol: str,
    threshold: float,
    min_confidence: float,
    fee_bps: float,
    experiences: Optional[List[Dict[str, Any]]] = None,
    initial_equity: float = 10000.0,
    memory_weight: float = 0.35,
    memory_top_k: int = 3,
    memory_reward_scale: float = 0.05,
    memory_min_similarity: float = 0.15,
) -> Tuple[BacktestResult, pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
    if len(df) < 3:
        raise ValueError(f"{symbol}: insufficient data ({len(df)} rows)")

    df = df.sort_values("Date").reset_index(drop=True)
    df = _add_signal_columns(df)

    fee_per_side = fee_bps / 10000.0
    position = 0
    entry_price = 0.0
    entry_time = None
    entry_index = None
    entry_features: Optional[Dict[str, float]] = None
    entry_memory_bias = 0.0
    entry_memory_ids: List[str] = []
    equity = float(initial_equity)
    equity_curve = []
    bar_returns: List[float] = []
    trades: List[Dict[str, Any]] = []
    new_experiences: List[Dict[str, Any]] = []
    memory_hits = 0
    memory_pool = experiences if experiences is not None else []

    for i in range(len(df) - 1):
        row = df.iloc[i]
        next_row = df.iloc[i + 1]

        score = float(row.get("signal_score") or 0.0)
        confidence = float(row.get("signal_confidence") or 0.0)
        features = _build_features(row, score, confidence)
        bias = 0.0
        similar_ids: List[str] = []
        if memory_pool:
            bias, similar_ids = _memory_bias(
                features,
                memory_pool,
                top_k=memory_top_k,
                reward_scale=memory_reward_scale,
                min_similarity=memory_min_similarity,
                symbol=symbol,
            )
        adjusted_score = _clamp(score + (memory_weight * bias), -1.0, 1.0)
        adjusted_confidence = _clamp(
            confidence + abs(bias) * 0.2,
            0.0,
            1.0,
        )
        desired = 0
        if adjusted_confidence >= min_confidence:
            if adjusted_score >= threshold:
                desired = 1
            elif adjusted_score <= -threshold:
                desired = -1
        if similar_ids and desired != 0 and position == 0:
            memory_hits += 1

        open_next = float(next_row["Open"])
        close_next = float(next_row["Close"])
        prev_equity = equity

        if desired != position:
            if position != 0:
                exit_price = open_next
                exit_time = next_row["Date"]
                gross = (exit_price - entry_price) / entry_price * position
                net = gross - (2 * fee_per_side)
                exp_id = uuid.uuid4().hex
                trades.append(
                    {
                        "symbol": symbol,
                        "side": "long" if position > 0 else "short",
                        "entry_time": entry_time,
                        "exit_time": exit_time,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "return_pct": net * 100.0,
                        "bars_held": i + 1 - (entry_index or 0),
                        "memory_bias": entry_memory_bias,
                        "memory_matches": len(entry_memory_ids),
                        "memory_ids": ";".join(entry_memory_ids),
                    }
                )
                new_experiences.append(
                    {
                        "id": exp_id,
                        "coin": symbol,
                        "symbol": symbol,
                        "action": "BUY" if position > 0 else "SELL",
                        "reward": net,
                        "created_at": entry_time.isoformat() if entry_time else None,
                        "closed_at": exit_time.isoformat(),
                        "source": "backtest",
                        "memory": {
                            "features": entry_features or {},
                            "memory_bias": entry_memory_bias,
                            "memory_ids": entry_memory_ids,
                        },
                        "outcome": {
                            "return": net,
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                        },
                    }
                )
                memory_pool.append(new_experiences[-1])
                equity *= 1 - fee_per_side

            if desired != 0:
                entry_price = open_next
                entry_time = next_row["Date"]
                entry_index = i + 1
                entry_features = features
                entry_memory_bias = bias
                entry_memory_ids = similar_ids
                equity *= 1 - fee_per_side

        bar_ret = 0.0
        if desired != 0:
            bar_ret = (close_next - open_next) / open_next * desired
        equity *= 1 + bar_ret

        bar_returns.append((equity / prev_equity) - 1.0)
        equity_curve.append(
            {
                "Date": next_row["Date"],
                "equity": equity,
                "position": desired,
                "signal_score": score,
                "signal_confidence": confidence,
                "memory_bias": bias,
                "memory_matches": len(similar_ids),
            }
        )
        position = desired

    if position != 0:
        last_row = df.iloc[-1]
        exit_price = float(last_row["Close"])
        exit_time = last_row["Date"]
        gross = (exit_price - entry_price) / entry_price * position
        net = gross - (2 * fee_per_side)
        exp_id = uuid.uuid4().hex
        trades.append(
            {
                "symbol": symbol,
                "side": "long" if position > 0 else "short",
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "return_pct": net * 100.0,
                "bars_held": len(df) - 1 - (entry_index or 0),
                "memory_bias": entry_memory_bias,
                "memory_matches": len(entry_memory_ids),
                "memory_ids": ";".join(entry_memory_ids),
            }
        )
        new_experiences.append(
            {
                "id": exp_id,
                "coin": symbol,
                "symbol": symbol,
                "action": "BUY" if position > 0 else "SELL",
                "reward": net,
                "created_at": entry_time.isoformat() if entry_time else None,
                "closed_at": exit_time.isoformat(),
                "source": "backtest",
                "memory": {
                    "features": entry_features or {},
                    "memory_bias": entry_memory_bias,
                    "memory_ids": entry_memory_ids,
                },
                "outcome": {
                    "return": net,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                },
            }
        )
        memory_pool.append(new_experiences[-1])
        equity *= 1 - fee_per_side

    trade_returns = [t["return_pct"] / 100.0 for t in trades]
    wins = sum(1 for r in trade_returns if r > 0)
    win_rate = wins / len(trade_returns) if trade_returns else 0.0
    avg_trade = sum(trade_returns) / len(trade_returns) if trade_returns else 0.0
    median_trade = (
        sorted(trade_returns)[len(trade_returns) // 2] if trade_returns else 0.0
    )
    total_return = (equity / initial_equity) - 1.0

    start = df["Date"].iloc[0]
    end = df["Date"].iloc[-1]
    days = max(1, int((end - start).total_seconds() / 86400))
    cagr = ((equity / initial_equity) ** (365 / days) - 1.0) if days > 0 else 0.0

    mdd = _max_drawdown([e["equity"] for e in equity_curve])
    sharpe = _sharpe(bar_returns, bars_per_year=96 * 365)

    summary = BacktestResult(
        symbol=symbol,
        start=start,
        end=end,
        bars=len(df),
        trades=len(trades),
        win_rate=win_rate,
        total_return=total_return,
        cagr=cagr,
        max_drawdown=mdd,
        sharpe=sharpe,
        avg_trade_return=avg_trade,
        median_trade_return=median_trade,
        equity_end=equity,
        initial_equity=initial_equity,
        memory_trades=memory_hits,
    )
    return summary, pd.DataFrame(trades), pd.DataFrame(equity_curve), new_experiences


def _write_outputs(
    out_dir: Path,
    summary: BacktestResult,
    trades: pd.DataFrame,
    equity: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"{summary.symbol}_summary.json"
    trades_path = out_dir / f"{summary.symbol}_trades.csv"
    equity_path = out_dir / f"{summary.symbol}_equity.csv"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary.__dict__, fh, ensure_ascii=False, indent=2, default=str)
    trades.to_csv(trades_path, index=False)
    equity.to_csv(equity_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    default_output = Path(__file__).resolve().parents[2] / "results" / "backtests"
    default_experience_log = Path(__file__).resolve().parents[1] / "data" / "experiences.jsonl"
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing *_15m.csv OHLCV files",
    )
    parser.add_argument("--symbols", type=str, default="")
    parser.add_argument("--start", type=str, default="")
    parser.add_argument("--end", type=str, default="")
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--signal-threshold", type=float, default=0.35)
    parser.add_argument("--min-confidence", type=float, default=0.35)
    parser.add_argument("--fee-bps", type=float, default=10.0)
    parser.add_argument("--initial-equity", type=float, default=10000.0)
    parser.add_argument("--output-dir", type=str, default=str(default_output))
    parser.add_argument(
        "--experience-log",
        type=str,
        default=str(default_experience_log),
        help="JSONL file used as contextual memory for similarity matching",
    )
    parser.add_argument("--memory-weight", type=float, default=0.35)
    parser.add_argument("--memory-top-k", type=int, default=3)
    parser.add_argument("--memory-reward-scale", type=float, default=0.05)
    parser.add_argument("--memory-min-similarity", type=float, default=0.15)
    parser.add_argument("--memory-append", action="store_true")
    args = parser.parse_args()

    data_dir = _resolve_data_dir(args.data_dir)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        symbols = _list_csv_symbols(data_dir)

    start = _parse_iso(args.start) if args.start else None
    end = _parse_iso(args.end) if args.end else None

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / run_id
    experience_log = Path(args.experience_log)
    experiences = _load_experiences(experience_log)
    summaries: List[Dict[str, Any]] = []
    new_records: List[Dict[str, Any]] = []

    for symbol in symbols:
        path = data_dir / f"{symbol}_15m.csv"
        if not path.exists():
            print(f"[WARN] Missing OHLCV file for {symbol}: {path}")
            continue
        df_raw = _load_ohlcv(path)
        df_ind = compute_indicators(df_raw, by_ticker=False)
        df_ind = _ensure_date_column(df_ind)
        df_window, start_ts, end_ts = _ensure_window(
            df_ind, start, end, args.lookback_days, symbol
        )
        if df_window.empty:
            print(f"[WARN] {symbol}: no rows in requested window.")
            continue
        summary, trades, equity, new_exps = run_backtest(
            df_window,
            symbol,
            threshold=args.signal_threshold,
            min_confidence=args.min_confidence,
            fee_bps=args.fee_bps,
            experiences=experiences,
            initial_equity=args.initial_equity,
            memory_weight=args.memory_weight,
            memory_top_k=args.memory_top_k,
            memory_reward_scale=args.memory_reward_scale,
            memory_min_similarity=args.memory_min_similarity,
        )
        new_records.extend(new_exps)
        _write_outputs(out_dir, summary, trades, equity)
        summaries.append(summary.__dict__)

        print(
            f"{symbol} {summary.start.date()} → {summary.end.date()} | "
            f"trades={summary.trades} return={summary.total_return:.2%} "
            f"mdd={summary.max_drawdown:.2%} sharpe={summary.sharpe:.2f}"
        )

    if summaries:
        with (out_dir / "summary_all.json").open("w", encoding="utf-8") as fh:
            json.dump(summaries, fh, ensure_ascii=False, indent=2, default=str)
        print(f"Saved results to {out_dir}")
        if args.memory_append:
            _append_experiences(experience_log, new_records)
            print(f"Appended {len(new_records)} experiences to {experience_log}")
    else:
        print("No backtest results generated.")


if __name__ == "__main__":
    main()
