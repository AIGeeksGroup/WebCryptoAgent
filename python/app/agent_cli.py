from __future__ import annotations

import argparse
import base64
import copy
import logging
import math
import os
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from openai import OpenAI
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

from .core.browser import driver_config
from .core.config import ts
from .core.experience import ExperienceStore
from .core.io_utils import ensure_dir, write_jsonl_line
from .core.llm import build_l1_bundle, run_l2_reasoner
from .core.market import build_indicator_signal
from .core.signals import post_signal_to_node
from .core.scrapper import scrape_price_page
from .data.ovhclv_to_indicators import generate_indicator_csv
from .data.pool_quotes import fetch_pool_quote

MAX_TRADE_USD = 5.0
FALLBACK_BNB = 0.001
API_ENDPOINT = os.environ.get("NODE_SIGNAL_URL", "")

POOL_CONFIG = {
    "ETH": {
        "symbol": "WETHUSDT",
        "pool_address": "put address here for your pool",
        "token_in_address": "in token address",  # WETH
        "token_out_address": "out token address",  # USDT
        "token_in_decimals": 18,
        "token_out_decimals": 6,
        "fee_tier": 500,  # 0.05%
        "chain": "polygon",
        "rpc_url": "wherever",
        "sample_amount_in": 0.01,
    },
    "POL": {
        "symbol": "POLUSDT",
        "pool_address": "put address here for your pool",
        "token_in_address": "in token address",  # 
        "token_out_address": "out token address",  # 
        "token_in_decimals": 18,
        "token_out_decimals": 6,
        "fee_tier": 3000,  # 0.3%
        "chain": "polygon",
        "rpc_url": "wherever",
        "sample_amount_in": 1.0,
    },
    "BTC": {
        "symbol": "WBTCUSDC.e",
        "pool_address": "put address here for your pool",
        "token_in_address": "in token address",  # 
        "token_out_address": "out token address",  # 
        "token_in_decimals": 8,
        "token_out_decimals": 6,
        "fee_tier": 500,
        "chain": "polygon",
        "rpc_url": "wherever",
        "sample_amount_in": 0.01,
    },
}

OHLCV_INTERVAL = "15m"
OHLCV_DIR = Path(__file__).resolve().parents[2] / "data" / "ohlcv"

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
else:
    logging.getLogger().setLevel(logging.INFO)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _ensure_indicator_csv(symbol: str) -> Optional[Path]:
    csv_name = f"{symbol}_{OHLCV_INTERVAL}_indicators.csv"
    path = OHLCV_DIR / csv_name
    if path.exists():
        return path
    logging.info("Indicator CSV missing for %s; attempting generation", symbol)
    try:
        generate_indicator_csv(symbol)
    except FileNotFoundError as exc:
        logging.error("Base OHLCV CSV missing for %s: %s", symbol, exc)
        return None
    except Exception as exc:
        logging.error("Failed to generate indicator CSV for %s: %s", symbol, exc)
        return None
    return path if path.exists() else None


def _latest_indicator_row(symbol: str) -> Tuple[Optional[pd.Series], Optional[Path]]:
    path = _ensure_indicator_csv(symbol)
    if not path:
        return None, None
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        logging.error("Failed to read indicator CSV for %s: %s", symbol, exc)
        return None, path
    if df.empty:
        logging.warning("Indicator CSV %s is empty", path)
        return None, path
    return df.iloc[-1], path


def _get_indicator_context(coin: str) -> Tuple[Optional[pd.Series], Optional[Path], Optional[Dict[str, Any]]]:
    cfg = POOL_CONFIG.get(coin)
    if not cfg:
        logging.warning("No pool configuration for %s", coin)
        return None, None, None
    row, path = _latest_indicator_row(cfg["symbol"])
    return row, path, cfg


def _indicator_snapshot_doc(
    coin: str,
    row: Optional[pd.Series] = None,
    path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    if row is None or path is None:
        row, path, _ = _get_indicator_context(coin)
    if row is None or path is None:
        logging.warning("Unable to provide indicator snapshot for %s", coin)
        return None
    date = row.get("Date", "unknown")
    close = row.get("Close", "n/a")
    ema_fast = row.get("ema_fast", "n/a")
    ema_slow = row.get("ema_slow", "n/a")
    rsi = row.get("rsi", row.get("rsi14", "n/a"))
    macd = row.get("macd", "n/a")
    macd_hist = row.get("macd_hist", "n/a")
    vol_z = row.get("vol_z", "n/a")
    signal_meta = build_indicator_signal(row)
    signal_label = signal_meta.get("signal_label", "neutral")
    signal_score = signal_meta.get("signal_score", 0.0)
    text = (
        f"{coin} snapshot {date}: close={close}, ema_fast={ema_fast}, "
        f"ema_slow={ema_slow}, rsi={rsi}, macd={macd}, macd_hist={macd_hist}, "
        f"vol_z={vol_z}, signal={signal_label}, signal_score={signal_score}."
    )
    return {
        "source": "indicator_snapshot",
        "url": str(path),
        "title": f"{coin} indicator snapshot @ {date}",
        "text": text,
        "coins": [coin],
        "hash": f"indicator-{coin}-{ts()}",
        "category": "market",
        "meta": signal_meta,
    }


def _risk_snapshot_doc(
    coin: str,
    row: Optional[pd.Series] = None,
    path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    if row is None or path is None:
        row, path, _ = _get_indicator_context(coin)
    if row is None or path is None:
        logging.warning("Unable to provide risk snapshot for %s", coin)
        return None

    def _as_float(key: str) -> float:
        val = row.get(key)
        try:
            return float(val)
        except (TypeError, ValueError):
            return float("nan")

    close = _as_float("Close")
    atr = _as_float("atr")
    vol_z = _as_float("vol_z")
    bb_width = _as_float("bb_width")
    ema_fast = _as_float("ema_fast")
    ema_slow = _as_float("ema_slow")
    rsi = _as_float("rsi")

    atr_bps = float("nan")
    if close and not math.isnan(close) and close != 0 and not math.isnan(atr):
        atr_bps = (atr / close) * 10000

    components = []
    if not math.isnan(atr_bps):
        components.append(min(1.0, max(0.0, atr_bps / 150)))
    if not math.isnan(vol_z):
        components.append(min(1.0, abs(vol_z) / 3))
    if not math.isnan(bb_width):
        components.append(min(1.0, bb_width / 0.15))
    trend_penalty = 0.2 if (not math.isnan(ema_fast) and not math.isnan(ema_slow) and ema_fast < ema_slow) else 0.0
    rsi_penalty = 0.0
    if not math.isnan(rsi):
        rsi_penalty = min(0.2, abs(rsi - 50.0) / 250.0)
    base_score = sum(components) / len(components) if components else 0.0
    score = clamp(base_score + trend_penalty + rsi_penalty, 0.0, 1.0)
    if score < 0.33:
        level = "low"
    elif score < 0.66:
        level = "medium"
    else:
        level = "high"

    text_parts = [
        f"{coin} risk {row.get('Date', 'unknown')}: level={level} score={score:.2f}",
    ]
    if not math.isnan(atr_bps):
        text_parts.append(f"ATR={atr_bps:.0f} bps")
    if not math.isnan(vol_z):
        text_parts.append(f"vol_z={vol_z:.2f}")
    if not math.isnan(bb_width):
        text_parts.append(f"bb_width={bb_width:.3f}")
    if not math.isnan(ema_fast) and not math.isnan(ema_slow):
        text_parts.append(f"trend={'bearish' if ema_fast < ema_slow else 'bullish'}")
    text = " | ".join(text_parts)

    return {
        "source": "pool_risk_model",
        "url": str(path),
        "title": f"{coin} pool risk snapshot",
        "text": text,
        "coins": [coin],
        "hash": f"risk-{coin}-{ts()}",
        "category": "risk",
        "meta": {
            "risk_score": score,
            "risk_level": level,
            "atr_bps": atr_bps if not math.isnan(atr_bps) else None,
            "vol_z": vol_z if not math.isnan(vol_z) else None,
            "bb_width": bb_width if not math.isnan(bb_width) else None,
        },
    }


def _pool_liquidity_doc(coin: str, quote: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not quote:
        return None
    entry_slip = float(quote.get("entry_slippage_bps", 0.0) or 0.0)
    exit_slip = float(quote.get("exit_slippage_bps", 0.0) or 0.0)
    gas_open = float(quote.get("gas_open_usd", 0.0) or 0.0)
    text = (
        f"{coin} pool {quote.get('pool_address')} price={quote.get('mid_price')} "
        f"fee={quote.get('fee_bps')}bps entry_slip={entry_slip:.1f}bps "
        f"exit_slip={exit_slip:.1f}bps gas_open={gas_open:.2f}USD"
    )
    return {
        "source": "pool_quote",
        "url": quote.get("pool_address"),
        "title": f"{coin} pool liquidity snapshot",
        "text": text,
        "coins": [coin],
        "hash": f"pool-{coin}-{ts()}",
        "category": "liquidity",
        "meta": quote,
    }


def _action_to_direction(action: str) -> str:
    if action == "BUY":
        return "long"
    if action == "SELL":
        return "short"
    return "flat"


def _expected_edge_bps(action: str, p_up: float, p_down: float, tp_bps: float, sl_bps: float) -> float:
    if action == "BUY":
        return (p_up * tp_bps) - ((1 - p_up) * sl_bps)
    if action == "SELL":
        return (p_down * tp_bps) - ((1 - p_down) * sl_bps)
    return 0.0


def _compute_round_trip_costs(quote: Dict[str, Any], size_usd: float) -> Dict[str, float]:
    size = max(float(size_usd), 1e-6)
    fee_rt = 2 * float(quote.get("fee_bps", 0.0) or 0.0)
    slip_entry = float(quote.get("entry_slippage_bps", 0.0) or 0.0)
    slip_exit = float(quote.get("exit_slippage_bps", 0.0) or 0.0)
    gas_open = float(quote.get("gas_open_usd", 0.0) or 0.0)
    gas_close = float(quote.get("gas_close_usd", 0.0) or 0.0)
    gas_rt = ((gas_open + gas_close) / size) * 10000
    mev_buffer = float(quote.get("mev_buffer_bps", 0.0) or 0.0)
    borrow = float(quote.get("borrow_bps", 0.0) or 0.0)
    basis = float(quote.get("basis_bps", 0.0) or 0.0)
    total = fee_rt + slip_entry + slip_exit + gas_rt + mev_buffer + borrow + abs(basis)
    return {
        "fee_bps_rt": fee_rt,
        "slippage_entry_bps": slip_entry,
        "slippage_exit_bps": slip_exit,
        "gas_bps_rt": gas_rt,
        "mev_buffer_bps": mev_buffer,
        "borrow_bps": borrow,
        "basis_bps": basis,
        "total_bps": total,
    }


def _evaluate_trade_gate(
    coin: str,
    l2: Dict[str, Any],
    action: str,
    size_usd: float,
    quote: Optional[Dict[str, Any]],
    risk_doc: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    direction = _action_to_direction(action)
    if direction == "flat":
        return {
            "allowed": False,
            "reason": "flat_action",
            "direction": direction,
            "edge_bps": 0.0,
            "net_edge_bps": 0.0,
            "margin_bps": 0.0,
            "costs": None,
            "quote": quote,
        }
    if not quote:
        return {
            "allowed": False,
            "reason": "missing_quote",
            "direction": direction,
            "edge_bps": 0.0,
            "net_edge_bps": 0.0,
            "margin_bps": 0.0,
            "costs": None,
            "quote": None,
        }

    p_up = float(l2.get("p_up", 0.0) or 0.0)
    p_down = float(l2.get("p_down", 0.0) or (1.0 - p_up))
    tp_bps = float(l2.get("tp_bps", 100) or 100)
    sl_bps = float(l2.get("sl_bps", 70) or 70)

    edge_bps = _expected_edge_bps(action, p_up, p_down, tp_bps, sl_bps)
    costs = _compute_round_trip_costs(quote, size_usd)

    risk_score = None
    if risk_doc:
        risk_score = risk_doc.get("meta", {}).get("risk_score")
    risk_score = float(risk_score) if risk_score is not None else 0.5
    margin_bps = 10.0 + risk_score * 20.0

    net_edge = edge_bps - (costs["total_bps"] + margin_bps)
    allowed = net_edge > 0 and edge_bps > 0

    reason = "net_edge_positive" if allowed else "insufficient_edge"
    return {
        "allowed": allowed,
        "reason": reason,
        "direction": direction,
        "edge_bps": edge_bps,
        "net_edge_bps": net_edge,
        "margin_bps": margin_bps,
        "costs": costs,
        "quote": quote,
    }


def build_confidence_breakdown(confidence: float) -> Tuple[Dict[str, float], float]:
    c = clamp(float(confidence), 0.0, 1.0)
    base = round(c * 0.55, 3)
    src = round(c * 0.18, 3)
    vol = round(c * 0.12, 3)
    tech = round(c * 0.20, 3)
    pen = round(-min(0.05, (1 - c) * 0.10), 3)
    conf = base + src + vol + tech + pen
    breakdown = {
        "base_sentiment": base,
        "source_agreement": src,
        "volume_confirmation": vol,
        "technical_alignment": tech,
        "uncertainty_penalty": pen,
    }
    return breakdown, clamp(conf, 0.0, 1.0)


AGENT_SPECS: List[Tuple[str, Set[str]]] = [
    ("chart", {"market", "liquidity"}),
    ("risk", {"risk"}),
]

AGENT_EVIDENCE_SOURCES = {
    "chart": ["indicator_snapshot", "binance_price_page", "pool_quote"],
    "risk": ["pool_risk_model"],
}


def evaluate_l2_action(l2: Dict[str, Any]) -> Tuple[str, float, float, float]:
    action = (l2.get("action") or "HOLD").upper()
    p_up = clamp(float(l2.get("p_up", 0.0)), 0.0, 1.0)
    p_down_raw = l2.get("p_down")
    p_down = clamp(float(p_down_raw if p_down_raw is not None else (1.0 - p_up)), 0.0, 1.0)
    if action == "SELL":
        confidence = p_down
    elif action == "BUY":
        confidence = p_up
    else:
        confidence = max(p_up, p_down)
    return action, confidence, p_up, p_down


def make_agent_bundle(
    base_bundle: Dict[str, Any], categories: Set[str], agent_name: str
) -> Optional[Dict[str, Any]]:
    docs = [
        copy.deepcopy(d) for d in base_bundle.get("docs", []) if d.get("category") in categories
    ]
    if not docs:
        return None

    agent_bundle = copy.deepcopy(base_bundle)
    agent_bundle["docs"] = docs

    meta = dict(agent_bundle.get("meta", {}))
    meta["agent"] = agent_name
    meta["sources"] = sorted({d.get("source", "unknown") for d in docs}) or meta.get(
        "sources", []
    )
    
    agent_bundle["meta"] = meta

    by_source: Dict[str, int] = {}
    evidence: List[Dict[str, Any]] = []
    used_pairs: Set[Tuple[str, str]] = set()
    weighted_sum = 0.0
    weight_total = 0.0

    for doc in docs:
        src = doc.get("source", "unknown")
        by_source[src] = by_source.get(src, 0) + 1
        sentiment = doc.get("llm_sentiment", {})
        conf = clamp(float(sentiment.get("confidence", 0.0)), 0.0, 1.0)
        score = float(sentiment.get("s", 0.0))
        weight_total += conf
        weighted_sum += conf * score
        key = (src, doc.get("url"))
        if len(evidence) < 3 and key not in used_pairs:
            used_pairs.add(key)
            evidence.append({"src": src, "url": doc.get("url"), "hash": doc.get("hash")})

    level = (weighted_sum / weight_total) if weight_total > 0 else 0.0
    sentiment_block = {
        "level": level,
        "ewm6": level,
        "ewm24": level,
        "surprise": 0.0,
        "event_flags": {"listing": 0, "upgrade": 0, "hack": 0, "regulatory": 0},
    }

    aggregates = {
        "doc_count": len(docs),
        "by_source": by_source,
        "sentiment": sentiment_block,
        "evidence": evidence,
    }

    agent_bundle["aggregates"] = aggregates
    return agent_bundle


def _summarize_statuses(trade_outcomes: List[Dict[str, Any]]) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for outcome in trade_outcomes:
        status = outcome.get("status", "unknown")
        summary[status] = summary.get(status, 0) + 1
    return summary


def _normalize_window(args) -> Tuple[str, str]:
    now = datetime.now(timezone.utc)
    if args.window == "1h":
        start = now - timedelta(hours=1)
    elif args.window == "15m":
        start = now - timedelta(minutes=15)
    else:
        start = now - timedelta(hours=1)
    return start.isoformat(), now.isoformat()


def _chart_to_data_url(path: Optional[str]) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def run_analysis_cycle(args, client, options, experience_store: ExperienceStore) -> Dict[str, Any]:
    cycle_start = time.time()
    requested_coins = [c.strip().upper() for c in args.coins.split(",") if c.strip()]
    invalid = [c for c in requested_coins if c not in POOL_CONFIG]
    if invalid:
        logging.warning("Unsupported coins requested (ignored): %s", ", ".join(invalid))
    coins_keep = [c for c in requested_coins if c in POOL_CONFIG]
    coins_label = ", ".join(coins_keep) if coins_keep else "none"
    if not coins_keep:
        logging.warning("No supported coins selected; aborting analysis cycle.")
        return {
            "result_dir": None,
            "trade_outcomes": [],
            "status_counts": {},
            "started_at": cycle_start,
            "finished_at": cycle_start,
            "duration": 0.0,
        }
    logging.info("Starting analysis cycle for coins: %s", coins_label)

    current_time = time.strftime("%Y%m%d_%H_%M_%S", time.localtime())
    result_dir = os.path.join(args.output_dir, current_time)
    os.makedirs(result_dir, exist_ok=True)
    market_out = os.path.join(result_dir, "events", "market", "market.jsonl")
    risk_out = os.path.join(result_dir, "events", "risk", "risk.jsonl")
    liquidity_out = os.path.join(result_dir, "events", "liquidity", "liquidity.jsonl")
    bundle_dir = os.path.join(result_dir, "bundles")

    for path in [market_out, risk_out, liquidity_out]:
        ensure_dir(os.path.dirname(path))
    ensure_dir(bundle_dir)

    driver_sources = None
    all_docs: List[Dict[str, Any]] = []
    coin_chart_path: Dict[str, str] = {}
    indicator_context: Dict[str, Dict[str, Any]] = {}
    risk_snapshots: Dict[str, Dict[str, Any]] = {}

    for coin in coins_keep:
        row, path, cfg = _get_indicator_context(coin)
        indicator_context[coin] = {"row": row, "path": path, "config": cfg}

    skip_scrape = getattr(args, "skip_scrape", False)

    if skip_scrape:
        logging.info("Skip scrape flag enabled; using indicator snapshots instead.")
        for coin in coins_keep:
            ctx = indicator_context.get(coin, {})
            doc = _indicator_snapshot_doc(coin, ctx.get("row"), ctx.get("path"))
            if doc:
                all_docs.append(doc)
                write_jsonl_line(market_out, doc)
    else:
        try:
            service = Service(ChromeDriverManager().install())
            driver_sources = webdriver.Chrome(service=service, options=options)
            driver_sources.set_window_size(args.window_width, args.window_height)

            for coin in coins_keep:
                try:
                    stats = scrape_binance_price_page(driver_sources, coin, result_dir, args.window)
                    if isinstance(stats, dict):
                        text = (
                            f"Price {stats.get('price_str', 'N/A')} | "
                            f"Change24h {stats.get('price_change_pct', 'N/A')}% | "
                            f"High24h {stats.get('high_24h_str', 'N/A')} | "
                            f"Low24h {stats.get('low_24h_str', 'N/A')} | "
                            f"Vol24h {stats.get('volume_24h_str', 'N/A')}"
                        )
                        doc = {
                            "source": "binance_price_page",
                            "url": stats.get("url"),
                            "title": f"{coin}  spot",
                            "text": text,
                            "coins": [coin],
                            "hash": f"bn-{coin}-{ts()}",
                            "category": "market",
                            "meta": stats,
                        }
                        all_docs.append(doc)
                        write_jsonl_line(market_out, stats)
                        if stats.get("screenshot"):
                            coin_chart_path[coin] = stats["screenshot"]
                except Exception as exc:
                    logging.error("Error scraping  for %s: %s", coin, exc)
                    continue

        finally:
            if driver_sources:
                driver_sources.quit()

    for coin in coins_keep:
        ctx = indicator_context.get(coin, {})
        row = ctx.get("row")
        path = ctx.get("path")
        cfg = ctx.get("config") or {}
        symbol_for_ind = cfg.get("symbol")
        indicator_ctx_for_l2 = None
        if symbol_for_ind:
            indicator_ctx_for_l2 = build_indicator_context(symbol_for_ind, OHLCV_INTERVAL)

        if not skip_scrape:
            indicator_doc = _indicator_snapshot_doc(coin, row, path)
            if indicator_doc:
                all_docs.append(indicator_doc)
                write_jsonl_line(market_out, indicator_doc)

        risk_doc = _risk_snapshot_doc(coin, row, path)
        if risk_doc:
            risk_snapshots[coin] = risk_doc
            all_docs.append(risk_doc)
            write_jsonl_line(risk_out, risk_doc)

        symbol = cfg.get("symbol")
        pool_address = cfg.get("pool_address")
        symbol = cfg.get("symbol")
        chain = cfg.get("chain", "polygon")
        fee_bps_default = cfg.get("fee_tier", 500) / 100  # 500 -> 5 bps, 3000 -> 30 bps

        trade_quote = None
        if pool_address and symbol:
            trade_quote = fetch_pool_quote(
                coin=coin,
                pool_address=pool_address,
                symbol=symbol,
                size_usd=50,
                indicator_row=row,
                risk_score=risk_doc.get("meta", {}).get("risk_score") if risk_doc else None,
                fee_bps_default=fee_bps_default,
                chain=chain,
                rpc_url=cfg.get("rpc_url"),
                token_in=cfg.get("token_in_address"),
                token_out=cfg.get("token_out_address"),
                token_in_decimals=cfg.get("token_in_decimals", 18),
                token_out_decimals=cfg.get("token_out_decimals", 6),
                sample_amount_in=cfg.get("sample_amount_in", 0.01),
            )
            pool_doc = _pool_liquidity_doc(coin, trade_quote)
            if pool_doc:
                all_docs.append(pool_doc)
                write_jsonl_line(liquidity_out, pool_doc)

    trade_outcomes: List[Dict[str, Any]] = []
    window_start, window_end = _normalize_window(args)

    for coin in coins_keep:
        docs_coin = [d for d in all_docs if coin in d.get("coins", [])]
        docs_all = docs_coin
        if not docs_all:
            logging.warning("No docs found for %s", coin)
            for agent_name, _ in AGENT_SPECS:
                trade_outcomes.append(
                    {
                        "coin": coin,
                        "agent": agent_name,
                        "action": "HOLD",
                        "status": "no_data",
                        "reason": "No documents collected for this interval",
                    }
                )
            continue

        try:
            bundle = build_l1_bundle(
                coin,
                args.window,
                window_start,
                window_end,
                docs_all,
                client,
                args.api_model,
            )

            aggregates = bundle.setdefault("aggregates", {})
            meta = bundle.setdefault("meta", {})
            srcs = set(meta.get("sources", []))
            if docs_coin:
                srcs.update(d.get("source") for d in docs_coin if d.get("source"))
            meta["sources"] = sorted(srcs) if srcs else meta.get("sources", [])

            base_l1_file = os.path.join(bundle_dir, f"{coin}_{window_end}_l1.jsonl")
            write_jsonl_line(base_l1_file, bundle)

            wallet = {
                "cash_usd": 20.0,
                "position_btc": 0.0,
                "budget_cap_usd": 20.0,
            }

            chart_path = coin_chart_path.get(coin)
            chart_b64 = _chart_to_data_url(chart_path)
            last_price_usd = None
            l2_global = run_l2_reasoner(
                client,
                args.api_model,
                bundle,
                chart_b64,
                wallet,
                last_price_usd,
                experiences=None,  # we skip per-agent experiences to avoid extra calls / complexity
                indicator_context=indicator_ctx_for_l2,
            )
            global_action, global_confidence, global_p_up, global_p_down = evaluate_l2_action(l2_global)
            for agent_name, categories in AGENT_SPECS:
                try:
                    agent_bundle = make_agent_bundle(bundle, categories, agent_name)
                    if not agent_bundle:
                        trade_outcomes.append(
                            {
                                "coin": coin,
                                "agent": agent_name,
                                "action": "HOLD",
                                "status": "no_data",
                                "reason": f"No {agent_name} evidence available",
                            }
                        )
                        continue

                    agent_l1_file = os.path.join(
                        bundle_dir, f"{coin}_{agent_name}_{window_end}_l1.jsonl"
                    )
                    write_jsonl_line(agent_l1_file, agent_bundle)

                    recent_experiences_raw = experience_store.fetch_recent(
                        coin=coin,
                        agent=agent_name,
                        limit=getattr(args, "experience_replay_limit", 3),
                        require_reward=True,
                    )
                    
                    agent_l2_file = os.path.join(
                        bundle_dir, f"{coin}_{agent_name}_{window_end}_l2.jsonl"
                    )
                    write_jsonl_line(agent_l2_file, l2_global)

                    action, confidence, p_up, p_down = global_action, global_confidence, global_p_up, global_p_down

                    if action == "HOLD":
                        logging.info(
                            "[HOLD] %s (%s agent): action=%s conf=%.2f",
                            coin,
                            agent_name,
                            action,
                            confidence,
                        )
                        trade_outcomes.append(
                            {
                                "coin": coin,
                                "agent": agent_name,
                                "action": action,
                                "confidence": round(confidence, 3),
                                "status": "skipped",
                                "reason": "Model chose HOLD",
                            }
                        )
                        continue

                    size_usd = float(l2_global.get("size_usd", 1.0) or 1.0)
                    size_usd = min(max(size_usd, 0.0), MAX_TRADE_USD)
                    if size_usd <= 0.0:
                        logging.info(
                            "[SKIP] %s (%s agent): Non-positive size from L2 (%.2f)",
                            coin,
                            agent_name,
                            size_usd,
                        )
                        trade_outcomes.append(
                            {
                                "coin": coin,
                                "agent": agent_name,
                                "action": action,
                                "confidence": round(confidence, 3),
                                "status": "skipped",
                                "reason": "Non-positive trade size from L2",
                            }
                        )
                        continue

                    ctx = indicator_context.get(coin, {})
                    row = ctx.get("row")
                    cfg = ctx.get("config") or {}
                    risk_doc = risk_snapshots.get(coin)

                    pool_address = cfg.get("pool_address")
                    symbol = cfg.get("symbol")
                    fee_bps_default = cfg.get("fee_bps", 30)
                    chain = cfg.get("chain", "bsc")

                    trade_quote = None
                    if pool_address and symbol:
                        trade_quote = fetch_pool_quote(
                            coin=coin,
                            pool_address=pool_address,
                            symbol=symbol,
                            size_usd=size_usd,
                            indicator_row=row,
                            risk_score=risk_doc.get("meta", {}).get("risk_score") if risk_doc else None,
                            fee_bps_default=fee_bps_default,
                            chain=chain,
                        )

                    gate = _evaluate_trade_gate(
                        coin=coin,
                        l2=l2_global,
                        action=action,
                        size_usd=size_usd,
                        quote=trade_quote,
                        risk_doc=risk_doc,
                    )

                    if not gate["allowed"]:
                        logging.info(
                            "[GATE] %s (%s agent): blocked reason=%s edge=%.1f net=%.1f cost=%.1f margin=%.1f",
                            coin,
                            agent_name,
                            gate["reason"],
                            gate["edge_bps"],
                            gate["net_edge_bps"],
                            gate["costs"]["total_bps"] if gate["costs"] else 0.0,
                            gate["margin_bps"],
                        )
                        trade_outcomes.append(
                            {
                                "coin": coin,
                                "agent": agent_name,
                                "action": action,
                                "direction": gate["direction"],
                                "confidence": round(confidence, 3),
                                "status": "gated",
                                "reason": gate["reason"],
                                "edge_bps": gate["edge_bps"],
                                "net_edge_bps": gate["net_edge_bps"],
                                "costs": gate["costs"],
                            }
                        )
                        continue

                    sources = AGENT_EVIDENCE_SOURCES.get(
                        agent_name, [f"{agent_name}_agent", "bundle"]
                    )
                    breakdown, _ = build_confidence_breakdown(confidence)
                    amount_bnb = FALLBACK_BNB
                    direction = gate["direction"]
                    execution_meta = {
                        "quote": gate["quote"],
                        "costs": gate["costs"],
                        "edge_bps": gate["edge_bps"],
                        "net_edge_bps": gate["net_edge_bps"],
                        "margin_bps": gate["margin_bps"],
                    }

                    signal_for_node = {
                        "action": action,
                        "direction": direction,
                        "coin": coin,
                        "token_address": cfg.get("token_address", pool_address),
                        "pool_address": pool_address,
                        "chain": chain,
                        "confidence": round(confidence, 3),
                        "confidence_breakdown": breakdown,
                        "evidence_sources": sources,
                        "reasoning": l2_global.get(
                            "reasoning", f"{agent_name.title()} agent reasoning"
                        ),
                        "amount_usd": size_usd,
                        "amount_eth": amount_bnb,
                        "agent": agent_name,
                        "p_up": p_up,
                        "p_down": p_down,
                        "execution": execution_meta,
                        "risk_metrics": risk_doc.get("meta") if risk_doc else {},
                        "l2_payload": l2_global,
                    }

                    experience_id = uuid.uuid4().hex
                    signal_for_node["experience_id"] = experience_id

                    used_experience_ids = [
                        exp.get("id")
                        for exp in recent_experiences_raw
                        if isinstance(exp, dict) and exp.get("id")
                    ]

                    experience_record = {
                        "id": experience_id,
                        "coin": coin,
                        "agent": agent_name,
                        "action": action,
                        "confidence": round(confidence, 3),
                        "p_up": p_up,
                        "p_down": p_down,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "bundle_snapshot": {
                            "aggregates": agent_bundle.get("aggregates"),
                            "meta": {
                                "window": agent_bundle.get("meta", {}).get("window"),
                                "sources": agent_bundle.get("meta", {}).get("sources"),
                            },
                        },
                        "paths": {
                            "base_l1": base_l1_file,
                            "agent_l1": agent_l1_file,
                            "agent_l2": agent_l2_file,
                            "chart_path": chart_path,
                        },
                        "wallet": wallet,
                        "signal": signal_for_node.copy(),
                        "cycle_dir": result_dir,
                        "context_experience_ids": used_experience_ids,
                    }
                    experience_store.record_open_experience(experience_record)

                    if not API_ENDPOINT:
                        logging.info(
                            "[DRY-RUN] %s (%s agent): direction=%s conf=%.2f net_edge=%.1f",
                            coin,
                            agent_name,
                            direction,
                            confidence,
                            execution_meta["net_edge_bps"],
                        )
                        trade_outcomes.append(
                            {
                                "coin": coin,
                                "agent": agent_name,
                                "action": action,
                                "direction": direction,
                                "confidence": round(confidence, 3),
                                "status": "dry_run",
                                "signal": signal_for_node,
                                "experience_id": experience_id,
                                "used_experience_ids": used_experience_ids,
                                "costs": execution_meta["costs"],
                                "edge_bps": execution_meta["edge_bps"],
                                "net_edge_bps": execution_meta["net_edge_bps"],
                            }
                        )
                        continue

                    try:
                        resp = post_signal_to_node(signal_for_node, url=API_ENDPOINT)
                        processed = bool(resp and resp.get("processed"))
                        experience_store.update_experience(
                            experience_id, {"node_response": resp, "node_processed": processed}
                        )
                        logging.info(
                            "[TRADE] %s %s agent=%s conf=%.2f native=%s -> %s",
                            coin,
                            action,
                            agent_name,
                            confidence,
                            amount_bnb,
                            resp,
                        )
                        trade_outcomes.append(
                            {
                                "coin": coin,
                                "agent": agent_name,
                                "action": action,
                                "direction": direction,
                                "confidence": round(confidence, 3),
                                "status": "sent",
                                "processed": processed,
                                "response": resp,
                                "experience_id": experience_id,
                                "used_experience_ids": used_experience_ids,
                                "costs": execution_meta["costs"],
                                "edge_bps": execution_meta["edge_bps"],
                                "net_edge_bps": execution_meta["net_edge_bps"],
                            }
                        )
                    except Exception as send_err:
                        logging.error("Error posting %s signal for %s agent: %s", coin, agent_name, send_err)
                        experience_store.update_experience(
                            experience_id,
                            {"status": "error", "error": str(send_err)},
                        )
                        trade_outcomes.append(
                            {
                                "coin": coin,
                                "agent": agent_name,
                                "action": action,
                                "direction": direction,
                                "confidence": round(confidence, 3),
                                "status": "error",
                                "error": str(send_err),
                                "experience_id": experience_id,
                                "used_experience_ids": used_experience_ids,
                                "costs": execution_meta["costs"],
                                "edge_bps": execution_meta["edge_bps"],
                                "net_edge_bps": execution_meta["net_edge_bps"],
                            }
                        )

                except Exception as agent_exc:
                    logging.error("Agent %s failed for %s: %s", agent_name, coin, agent_exc)
                    trade_outcomes.append(
                        {
                            "coin": coin,
                            "agent": agent_name,
                            "action": "HOLD",
                            "status": "error",
                            "reason": str(agent_exc),
                        }                    
                    )
                    continue

        except Exception as exc:
            logging.error("Error processing %s: %s", coin, exc)
            for agent_name, _ in AGENT_SPECS:
                trade_outcomes.append(
                    {
                        "coin": coin,
                        "agent": agent_name,
                        "action": "HOLD",
                        "status": "error",
                        "reason": str(exc),
                    }
                )
            continue

    status_counts = _summarize_statuses(trade_outcomes)
    cycle_end = time.time()
    logging.info("[COMPLETED] Results saved to %s", result_dir)
    logging.info("Outcome summary: %s", status_counts)

    return {
        "result_dir": result_dir,
        "trade_outcomes": trade_outcomes,
        "status_counts": status_counts,
        "started_at": cycle_start,
        "finished_at": cycle_end,
        "duration": cycle_end - cycle_start,
    }

def build_indicator_context(symbol, interval):
    path = f"python/data/ohlcv/{symbol}_{interval}.csv"
    if not os.path.exists(path):
        return {"numeric": {}, "summary": f"No local indicator data found for {symbol} {interval}"}
    
    df = pd.read_csv(path)
    if df.empty:
        return {"numeric": {}, "summary": f"Empty OHLCV data for {symbol} {interval}"}
    latest = df.iloc[-1].to_dict()
    ema_fast = latest.get("EMA_6", None)
    ema_slow = latest.get("EMA_24", None)
    rsi = latest.get("RSI", None)
    macd = latest.get("MACD", None)
    atr = latest.get("ATR", None)
    trend = None
    if ema_fast is not None and ema_slow is not None:
        trend = "bullish" if ema_fast > ema_slow else "bearish"

    overbought = rsi > 70 if rsi is not None else False
    oversold = rsi < 30 if rsi is not None else False

    summary = (
        f"{symbol} ({interval}) indicators → "
        f"Trend: {trend or 'n/a'}; RSI={rsi:.1f} ({'overbought' if overbought else 'oversold' if oversold else 'neutral'}); "
        f"MACD={macd:.3f}, ATR={atr:.4f}, "
        f"Close={latest.get('close', 'n/a')}, "
        f"EMA6={ema_fast}, EMA24={ema_slow}."
    )

    return {"numeric": latest, "summary": summary}
    

def run_agent_loop(args, client, options, experience_store: ExperienceStore) -> None:
    interval = max(int(args.agent_interval), 60)
    completion_delay = max(int(args.agent_completion_delay), 10)
    failure_delay = max(int(args.agent_failure_delay), 60)
    max_cycles = int(args.agent_max_cycles) if getattr(args, "agent_max_cycles", 0) else 0
    max_cycles = max_cycles if max_cycles > 0 else None

    logging.info(
        "Agent loop configured: interval=%ss completion_delay=%ss failure_delay=%ss max_cycles=%s",
        interval,
        completion_delay,
        failure_delay,
        max_cycles if max_cycles is not None else "infinite",
    )

    cycles = 0
    next_run_ts = time.time()

    try:
        while True:
            now = time.time()
            if now < next_run_ts:
                sleep_for = next_run_ts - now
                logging.info("Sleeping %.0f seconds before next cycle", sleep_for)
                time.sleep(sleep_for)

            try:
                result = run_analysis_cycle(args, client, options, experience_store)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                logging.exception("Analysis cycle failed: %s", exc)
                next_run_ts = time.time() + failure_delay
                cycles += 1
                if max_cycles is not None and cycles >= max_cycles:
                    logging.info("Max cycles reached after failure; exiting agent loop.")
                    break
                continue

            cycles += 1

            trade_outcomes = result.get("trade_outcomes", [])
            processed = any(
                outcome.get("processed")
                for outcome in trade_outcomes
                if outcome.get("status") == "sent"
            )

            status_counts = result.get("status_counts", {})
            logging.info("Cycle %s summary: %s", cycles, status_counts)

            if processed:   
                next_run_ts = time.time() + completion_delay
                logging.info(
                    "Execution completed on Node. Next analysis cycle in %s seconds.",
                    completion_delay,
                )
            else:
                sleep_for = max(0.0, interval - result.get("duration", 0.0))
                next_run_ts = time.time() + sleep_for
                logging.info(
                    "No completed execution reported. Next cycle in %.0f seconds.",
                    sleep_for,
                )

            if max_cycles is not None and cycles >= max_cycles:
                logging.info("Reached configured max cycles (%s). Stopping agent loop.", max_cycles)
                break

    except KeyboardInterrupt:
        logging.info("Agent loop interrupted by user.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["summarizer", "agent"],
        default="summarizer",
        help="summarizer = scrape & bundle (L1); agent = continuous loop",
    )
    parser.add_argument(
        "--coins",
        default="BTC,ETH,POL",
        help="Comma list of coins to keep (subset of BTC,ETH,POL)",
    )
    parser.add_argument("--window", default="1h", choices=["15m", "1h"])
    parser.add_argument(
    "--api_key",
    default=None,
    type=str,
    help="OpenAI API key (or set OPENAI_API_KEY env var)",
    )
    parser.add_argument("--api_model", default="gpt-5.1", type=str, help="API model name")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--download_dir", type=str, default="downloads")
    parser.add_argument("--text_only", action="store_true")
    parser.add_argument("--headless", action="store_true", help="Run Selenium in headless mode")
    parser.add_argument(
        "--skip-scrape",
        action="store_true",
        help="Skip Selenium scraping and rely on local indicator snapshots for docs",
    )
    parser.add_argument("--force_device_scale", action="store_true")
    parser.add_argument("--window_width", type=int, default=1024)
    parser.add_argument("--window_height", type=int, default=768)
    parser.add_argument(
        "--experience_log",
        type=str,
        default=os.path.join("data", "experiences.jsonl"),
        help="Path to JSONL file where trade experiences are stored.",
    )
    parser.add_argument(
        "--experience_replay_limit",
        type=int,
        default=3,
        help="Number of past experiences to inject into each agent prompt.",
    )
    parser.add_argument(
        "--agent_interval",
        type=int,
        default=3600,
        help="Seconds between analysis cycles in agent mode",
    )
    parser.add_argument(
        "--agent_completion_delay",
        type=int,
        default=60,
        help="Seconds to wait before rerunning after a completed execution",
    )
    parser.add_argument(
        "--agent_failure_delay",
        type=int,
        default=300,
        help="Seconds to wait before retrying after a failure",
    )
    parser.add_argument(
        "--agent_max_cycles",
        type=int,
        default=0,
        help="Optional maximum number of cycles in agent mode (0 = run forever)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("No API key provided. Pass --api_key or set OPENAI_API_KEY.")

    client = OpenAI(api_key=api_key)
    options = driver_config(args)
    experience_store = ExperienceStore(args.experience_log)

    if args.mode == "summarizer":
        run_analysis_cycle(args, client, options, experience_store)
        return

    if args.mode == "agent":
        run_agent_loop(args, client, options, experience_store)
        return


if __name__ == "__main__":
    main()
    print("End of process")
