from __future__ import annotations

import math
from typing import Any, Dict, Optional


def _to_float(value: Any) -> Optional[float]:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(num):
        return None
    return num


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def build_indicator_signal(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a directional signal from indicator columns on a row-like mapping.
    Expected keys: ema_fast, ema_slow, rsi, macd_hist, vol_z.
    """
    ema_fast = _to_float(row.get("ema_fast"))
    ema_slow = _to_float(row.get("ema_slow"))
    rsi = _to_float(row.get("rsi"))
    macd_hist = _to_float(row.get("macd_hist"))
    vol_z = _to_float(row.get("vol_z"))

    trend = None
    trend_score = 0.0
    if ema_fast is not None and ema_slow is not None:
        if ema_fast == ema_slow:
            trend = "flat"
            trend_score = 0.0
        else:
            trend = "bullish" if ema_fast > ema_slow else "bearish"
            trend_score = 1.0 if ema_fast > ema_slow else -1.0

    rsi_score = 0.0
    if rsi is not None:
        rsi_score = _clamp((rsi - 50.0) / 25.0, -1.0, 1.0)

    macd_score = 0.0
    if macd_hist is not None:
        if macd_hist > 0:
            macd_score = 1.0
        elif macd_hist < 0:
            macd_score = -1.0

    components: Dict[str, float] = {}
    weighted: list[tuple[float, float]] = []
    if trend_score != 0.0:
        components["trend"] = trend_score
        weighted.append((trend_score, 0.5))
    if rsi is not None:
        components["rsi"] = rsi_score
        weighted.append((rsi_score, 0.3))
    if macd_score != 0.0:
        components["macd"] = macd_score
        weighted.append((macd_score, 0.2))

    if weighted:
        total_weight = sum(weight for _, weight in weighted)
        score = sum(score * weight for score, weight in weighted) / total_weight
    else:
        score = 0.0
    score = _clamp(score, -1.0, 1.0)

    confidence = 0.3 + 0.1 * len(weighted)
    if vol_z is not None and abs(vol_z) >= 1.0:
        confidence += 0.1
    confidence = _clamp(confidence, 0.2, 0.7)

    if score > 0.2:
        label = "bullish"
    elif score < -0.2:
        label = "bearish"
    else:
        label = "neutral"

    return {
        "signal_score": round(score, 3),
        "signal_label": label,
        "signal_confidence": round(confidence, 3),
        "trend": trend,
        "rsi": rsi,
        "macd_hist": macd_hist,
        "vol_z": vol_z,
        "components": components,
    }
