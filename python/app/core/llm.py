from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

MAX_LLM_DOCS = 8
MAX_SUMMARY_CHARS = 1500
SYNTHETIC_SOURCES = {
    "indicator_snapshot",
    "pool_risk_model",
    "pool_quote",
    "binance_price_page",
    "liquidity_quote",
}

SUMMARIZER_SYS = (
    'You are a crypto market signal summarizer. '
    'Return ONLY JSON: {"s":number in [-1,1],"confidence":number in [0,1],"tags":[string],"summary":"short"}. '
    "Be conservative; if uncertain, keep s near 0 and confidence low."
)

L2_SYSTEM_PROMPT = (
    "You are a crypto trading L2 reasoner. "
    "Your job is to interpret signal sentiment, chart context, and 15-minute technical indicators "
    "to decide the most probable trading stance: BUY (long), SELL (short), or HOLD (flat). "
    "You must always output exactly one action. "
    "You are also given a summary of indicators (EMA, RSI, MACD, ATR, Bollinger, PDH/PDL). "
    "Use these to justify your reasoning alongside signal sentiment and wallet data.\n"
    "Return STRICT JSON with this schema:\n"
    "{"
    '  "coin":"BTC|ETH|SOL|...",'
    '  "action":"BUY|SELL|HOLD",'
    '  "size_usd": number,'
    '  "size_btc": number,'
    '  "p_up": number,'
    '  "tp_bps": integer,'
    '  "sl_bps": integer,'
    '  "timeframe":"15m|1h",'
    '  "reasoning": "short string",'
    '  "evidence_sources": ["sentiment","indicators","chart"]'
    "}\n"
    "Guidelines:\n"
    "- Treat BUY as long, SELL as short, HOLD as flat.\n"
    "- Use indicator summary to confirm or contradict sentiment.\n"
    "- Prefer conservative sizing (≤50% available funds).\n"
    "- Return ONLY JSON. No explanations outside JSON."
)

_TEMPERATURE_RESTRICTED_PREFIXES = (
    "gpt-5",
    "gpt-4.1",
    "gpt-4o",
    "gpt-4o-mini",
    "o4",
    "o1",
)

_RATE_LIMIT_RE = re.compile(r"in ([0-9.]+)s|in ([0-9]+)ms", re.IGNORECASE)


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "rate limit" in msg or "rate_limit" in msg or "429" in msg


def _retry_after_seconds(exc: Exception) -> Optional[float]:
    match = _RATE_LIMIT_RE.search(str(exc))
    if not match:
        return None
    if match.group(1):
        try:
            return float(match.group(1))
        except ValueError:
            return None
    if match.group(2):
        try:
            return float(match.group(2)) / 1000.0
        except ValueError:
            return None
    return None


def _select_temperature(model: str, desired: float) -> Optional[float]:
    """
    Some Chat Completions models (e.g. gpt-4.1 family) only support the default temperature.
    Return None to omit the parameter when the model is temperature-locked.
    """
    model = model or ""
    if any(model.startswith(prefix) for prefix in _TEMPERATURE_RESTRICTED_PREFIXES):
        return None
    return desired


def summarize_doc_with_llm(openai_client, model: str, doc: Dict[str, Any]) -> Dict[str, Any]:
    """Ask OpenAI to summarise a document with sentiment scoring."""
    payload = {
        "source": doc.get("source"),
        "url": doc.get("url"),
        "title": doc.get("title"),
        "text": (doc.get("text") or "")[:MAX_SUMMARY_CHARS],
        "coins": doc.get("coins"),
    }
    try:
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": SUMMARIZER_SYS},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "response_format": {"type": "json_object"},
        }
        temp = _select_temperature(model, 0.2)
        if temp is not None:
            kwargs["temperature"] = temp
        resp = openai_client.chat.completions.create(**kwargs)
        out = json.loads(resp.choices[0].message.content)
        s = float(max(-1.0, min(1.0, out.get("s", 0.0))))
        c = float(max(0.0, min(1.0, out.get("confidence", 0.0))))
        return {
            "s": s,
            "confidence": c,
            "tags": out.get("tags", []),
            "summary": out.get("summary", ""),
        }
    except Exception as exc:
        logging.error("Summarizer error: %s", exc)
        return {"s": 0.0, "confidence": 0.0, "tags": [], "summary": ""}


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _local_summary(doc: Dict[str, Any]) -> Dict[str, Any]:
    src = (doc.get("source") or "").lower()
    meta = doc.get("meta") or {}
    if src == "pool_risk_model":
        risk = float(meta.get("risk_score") or 0.5)
        sentiment = _clamp(0.4 - risk, -1.0, 1.0)
        return {
            "s": sentiment,
            "confidence": 0.6,
            "tags": ["risk_model"],
            "summary": f"Pool risk score={risk:.2f}",
        }
    if src == "binance_price_page":
        change = meta.get("price_change_pct")
        try:
            change_val = float(change)
        except (TypeError, ValueError):
            change_val = 0.0
        sentiment = _clamp(change_val / 50.0, -1.0, 1.0)
        return {
            "s": sentiment,
            "confidence": 0.5,
            "tags": ["price_change"],
            "summary": f"24h change {change_val:.2f}%",
        }
    if src == "indicator_snapshot":
        signal_score = meta.get("signal_score")
        if signal_score is not None:
            try:
                score = float(signal_score)
            except (TypeError, ValueError):
                score = 0.0
            score = _clamp(score, -1.0, 1.0)
            try:
                conf = float(meta.get("signal_confidence", 0.4))
            except (TypeError, ValueError):
                conf = 0.4
            conf = _clamp(conf, 0.0, 1.0)
            label = meta.get("signal_label")
            if not label:
                label = "bullish" if score > 0.2 else "bearish" if score < -0.2 else "neutral"
            return {
                "s": score,
                "confidence": conf,
                "tags": ["indicator_signal", label],
                "summary": f"Indicator signal {label} score={score:.2f}",
            }
        return {
            "s": 0.0,
            "confidence": 0.2,
            "tags": ["indicator_snapshot"],
            "summary": "Structured indicator snapshot",
        }
    if src == "pool_quote":
        return {
            "s": 0.0,
            "confidence": 0.2,
            "tags": [src],
            "summary": "Structured liquidity snapshot",
        }
    return {
        "s": 0.0,
        "confidence": 0.1,
        "tags": ["auto"],
        "summary": "Auto summary unavailable",
    }


def _requires_llm(doc: Dict[str, Any]) -> bool:
    src = (doc.get("source") or "").lower()
    return src not in SYNTHETIC_SOURCES


def build_l1_bundle(
    coin: str,
    window: str,
    window_start: str,
    window_end: str,
    docs: List[Dict[str, Any]],
    client,
    model: str,
) -> Dict[str, Any]:
    docs_out: List[Dict[str, Any]] = []
    llm_budget = MAX_LLM_DOCS
    for d in docs:
        if _requires_llm(d):
            if llm_budget > 0:
                sentiment = summarize_doc_with_llm(client, model, d)
                llm_budget -= 1
            else:
                sentiment = _local_summary(d)
        else:
            sentiment = _local_summary(d)
        d_out = dict(d)
        d_out["full_text"] = (d_out.pop("text") or "")[:10000]
        d_out["llm_sentiment"] = sentiment
        d_out["scroll_done"] = True
        docs_out.append(d_out)

    weights = [max(0.0, min(1.0, doc["llm_sentiment"]["confidence"])) for doc in docs_out]
    weighted = [doc["llm_sentiment"]["s"] * w for doc, w in zip(docs_out, weights)]
    weight_sum = sum(weights)
    level = (sum(weighted) / weight_sum) if weight_sum > 0 else 0.0

    ewm6 = ewm24 = level
    surprise = level - ewm24

    by_source: Dict[str, int] = {}
    for doc in docs_out:
        src = doc.get("source", "unknown")
        by_source[src] = by_source.get(src, 0) + 1

    evidence: List[Dict[str, Any]] = []
    seen = set()
    for doc in docs_out:
        if len(evidence) >= 3:
            break
        key = (doc.get("source"), doc.get("url"))
        if key in seen:
            continue
        seen.add(key)
        evidence.append(
            {"src": doc.get("source"), "url": doc.get("url"), "hash": doc.get("hash")}
        )

    bundle = {
        "meta": {
            "coin": coin,
            "window": window,
            "window_start": window_start,
            "window_end": window_end,
            "sources": [
                "binance_price_page",
                "indicator_snapshot",
                "pool_quote",
                "pool_risk_model",
            ],
            "scrape_config": {
                "scroll_to_end": True,
                "max_scrolls": 30,
                "headless": True,
            },
            "hygiene": {"dedup": "sha256(url|title|text)", "language": "en"},
        },
        "docs": docs_out,
        "aggregates": {
            "doc_count": len(docs_out),
            "by_source": by_source,
            "sentiment": {
                "level": level,
                "ewm6": ewm6,
                "ewm24": ewm24,
                "surprise": surprise,
                "event_flags": {
                    "listing": 0,
                    "upgrade": 0,
                    "hack": 0,
                    "regulatory": 0,
                },
            },
            "evidence": evidence,
        },
        "handoff_tasks_for_L2": {
            "questions": [
                "Confirm EMA trend agrees with MACD histogram sign.",
                "Check RSI regime (rsi > 60 bullish, rsi < 40 bearish).",
                "Verify volatility (ATR/vol_z) supports tp_bps vs sl_bps.",
            ],
            "required_numeric_features": [
                "ret_1h",
                "ret_4h",
                "ret_8h",
                "ema12",
                "ema26",
                "macd",
                "macd_signal",
                "rsi14",
                "boll_pctB_20",
                "atr14_bps",
                "vol_z20",
                "btc_corr24",
                "beta_btc24",
            ],
            "cost_assumptions": {"round_trip_cost_bps": 20},
            "output_expectation": {
                "json": ["p_up", "action", "tp_bps", "sl_bps", "rationales"],
                "caution": "Calibrate p_up; trade only if p_up >= p_break_even + margin.",
            },
        },
        "audit": {"popups_closed": True, "errors": [], "deduped_count": 0},
    }
    return bundle


def build_l2_payload(
    bundle: Dict[str, Any],
    wallet: Dict[str, float],
    last_price_usd: Optional[float] = None,
    experiences: Optional[List[Dict[str, Any]]] = None,
    indicator_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {
        "meta": bundle.get("meta", {}),
        "sentiment": bundle.get("aggregates", {}),
        "wallet": wallet,
        "last_price_usd": last_price_usd,
        "cost_assumptions": bundle.get("handoff_tasks_for_L2", {}).get(
            "cost_assumptions", {"round_trip_cost_bps": 20}
        ),
    }

    # add past experiences if any
    if experiences:
        payload["past_experiences"] = experiences

    # add indicator context (numeric + summary)
    if indicator_context:
        payload["indicators"] = indicator_context.get("numeric", {})
        payload["indicator_summary"] = indicator_context.get("summary", "")

    return payload


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def validate_and_clip_signal(sig: Dict[str, Any], wallet: Dict[str, float]) -> Dict[str, Any]:
    action = sig.get("action", "HOLD")
    size_usd = float(sig.get("size_usd", 0.0) or 0.0)
    size_btc = float(sig.get("size_btc", 0.0) or 0.0)
    p_up = float(_clip(sig.get("p_up", 0.5) or 0.5, 0.0, 1.0))
    tp_bps = int(_clip(int(sig.get("tp_bps", 100)), 50, 300))
    sl_bps = int(_clip(int(sig.get("sl_bps", 70)), 30, 300))

    cash = max(0.0, float(wallet.get("cash_usd", 0.0)))
    position = max(0.0, float(wallet.get("position_btc", 0.0)))
    cap = max(0.0, float(wallet.get("budget_cap_usd", cash)))

    if action == "BUY":
        size_usd = _clip(size_usd, 0.0, min(cash, cap))
        size_btc = 0.0
    elif action == "SELL":
        size_btc = _clip(size_btc, 0.0, position)
        size_usd = 0.0
    else:
        action = "HOLD"
        size_usd = 0.0
        size_btc = 0.0

    return {
        "coin": sig.get("coin"),
        "action": action,
        "size_usd": round(size_usd, 2),
        "size_btc": round(size_btc, 8),
        "p_up": round(p_up, 4),
        "tp_bps": tp_bps,
        "sl_bps": sl_bps,
        "timeframe": sig.get("timeframe", "1h"),
        "reasoning": (sig.get("reasoning") or "")[:400],
        "evidence_sources": sig.get("evidence_sources", [])[:8],
    }


def run_l2_reasoner(
    openai_client,
    model,
    bundle,
    chart_b64,
    wallet,
    last_price_usd=None,
    experiences=None,
    indicator_context=None,
):
    payload = build_l2_payload(
        bundle,
        wallet,
        last_price_usd,
        experiences,
        indicator_context,
    )
    max_retries = int(os.environ.get("LLM_RATE_LIMIT_RETRY_MAX", "3"))
    base_sleep = float(os.environ.get("LLM_RATE_LIMIT_SLEEP", "1.0"))
    attempt = 0
    while True:
        try:
            content: List[Dict[str, Any]] = [
                {
                    "type": "text",
                    "text": json.dumps(payload, ensure_ascii=False),
                }
            ]
            if chart_b64:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{chart_b64}"},
                    }
                )
            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": L2_SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                "response_format": {"type": "json_object"},
            }

            temp = _select_temperature(model, 0.2)
            if temp is not None:
                kwargs["temperature"] = temp

            resp = openai_client.chat.completions.create(**kwargs)
            raw = json.loads(resp.choices[0].message.content)
            if isinstance(raw, list):
                picked = None
                for item in raw:
                    if isinstance(item, dict):
                        picked = item
                        break
                if picked is None:
                    raise ValueError("L2 response list missing object payload")
                raw = picked
            break
        except Exception as exc:
            if _is_rate_limit_error(exc) and attempt < max_retries:
                attempt += 1
                retry_after = _retry_after_seconds(exc)
                delay = max(base_sleep, retry_after or base_sleep)
                logging.warning(
                    "L2 rate limit hit; retrying in %.2fs (attempt %s/%s)",
                    delay,
                    attempt,
                    max_retries,
                )
                time.sleep(delay)
                continue
            logging.error("L2 reasoner error: %s", exc)
            coin = bundle.get("meta", {}).get("coin", "BTC")
            sentiment = bundle.get("aggregates", {}).get("sentiment", {})
            level = float(sentiment.get("level", 0.0))

            if level > 0.1:
                fallback_action = "BUY"
                p_up = 0.6
            elif level < -0.1:
                fallback_action = "SELL"
                p_up = 0.4  # lower than 0.5 to reflect bearish
            else:
                fallback_action = "HOLD"
                p_up = 0.5

            raw = {
                "coin": coin,
                "action": fallback_action,
                "size_usd": 5,  # small probe size
                "size_btc": 0,
                "p_up": p_up,
                "tp_bps": 120,
                "sl_bps": 80,
                "timeframe": bundle.get("meta", {}).get("window", "1h"),
                "reasoning": f"Fallback {fallback_action} based on sentiment level={level:.2f} due to L2 error.",
                "evidence_sources": ["sentiment"],
            }
            break

    return validate_and_clip_signal(raw, wallet)
