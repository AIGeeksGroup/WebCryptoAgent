
from __future__ import annotations
import pandas as pd
from typing import Dict, Any

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def _rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/length, adjust=False).mean()
    roll_down = down.ewm(alpha=1/length, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = _ema(series, fast)
    ema_slow = _ema(series, slow)
    macd = ema_fast - ema_slow
    macd_signal = _ema(macd, signal)
    hist = macd - macd_signal
    return macd, macd_signal, hist

def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    tr = _true_range(high, low, close)
    return tr.ewm(alpha=1/length, adjust=False).mean()

def _bollinger(series: pd.Series, length: int = 20, stds: float = 2.0):
    ma = series.rolling(length).mean()
    sd = series.rolling(length).std(ddof=0)
    upper = ma + stds * sd
    lower = ma - stds * sd
    width = (upper - lower) / (ma + 1e-12)
    return ma, upper, lower, width

def _prev_day_high_low(df: pd.DataFrame) -> pd.DataFrame:
    # expects DateTimeIndex or a "Date" column that can be parsed
    if "Date" in df.columns:
        df = df.set_index("Date")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index must be a DatetimeIndex or include a 'Date' column.")
    # Daily highs/lows
    daily = df.resample("1D").agg({"High": "max", "Low": "min"}).rename(
        columns={"High": "DH", "Low": "DL"}
    )
    # Shift to represent PREVIOUS day values
    daily[["PDH", "PDL"]] = daily[["DH", "DL"]].shift(1)
    # Map back to intraday
    out = df.join(daily[["PDH", "PDL"]], how="left")
    # forward-fill within each day to have PDH/PDL available for all intraday rows
    out["PDH"] = out["PDH"].ffill()
    out["PDL"] = out["PDL"].ffill()
    return out

def compute_indicators(
    df: pd.DataFrame,
    params: Dict[str, Any] | None = None,
    by_ticker: bool = True
) -> pd.DataFrame:
    """
    Compute I2 indicators on OHLCV.
    Required columns: ["Date","Open","High","Low","Close","Vol"] and optionally "Ticker".

    Returns original columns +:
      - ema_fast, ema_slow
      - rsi
      - macd, macd_signal, macd_hist
      - atr
      - bb_mid, bb_upper, bb_lower, bb_width
      - PDH, PDL (previous-day high/low)
      - vol_z (rolling z-score of volume)
    """
    if params is None:
        params = {}
    p = {
        "ema_fast": params.get("ema_fast", 12),
        "ema_slow": params.get("ema_slow", 26),
        "rsi_len": params.get("rsi_len", 14),
        "macd_fast": params.get("macd_fast", 12),
        "macd_slow": params.get("macd_slow", 26),
        "macd_signal": params.get("macd_signal", 9),
        "atr_len": params.get("atr_len", 14),
        "bb_len": params.get("bb_len", 20),
        "bb_stds": params.get("bb_stds", 2.0),
        "vol_lookback": params.get("vol_lookback", 30),
    }

    work = df.copy()
    if "Date" in work.columns:
        work["Date"] = pd.to_datetime(work["Date"], utc=True, errors="coerce")
    cols_needed = {"open","high","low","close","trade"}
    if not cols_needed.issubset(set(work.columns)):
        missing = cols_needed - set(work.columns)
        raise ValueError(f"Missing columns: {missing}")
    # Ensure sorting
    sort_cols = ["Date"]
    if "Ticker" in work.columns:
        sort_cols = ["Ticker","Date"]
    work = work.sort_values(sort_cols).reset_index(drop=True)

    def _calc(group: pd.DataFrame) -> pd.DataFrame:
        g = group.copy()
        # EMA
        g["ema_fast"] = _ema(g["Close"], p["ema_fast"])
        g["ema_slow"] = _ema(g["Close"], p["ema_slow"])
        # RSI
        g["rsi"] = _rsi(g["Close"], p["rsi_len"])
        # MACD
        macd, macd_signal, macd_hist = _macd(g["Close"], p["macd_fast"], p["macd_slow"], p["macd_signal"])
        g["macd"] = macd
        g["macd_signal"] = macd_signal
        g["macd_hist"] = macd_hist
        # ATR
        g["atr"] = _atr(g["High"], g["Low"], g["Close"], p["atr_len"])
        # Bollinger
        bb_mid, bb_upper, bb_lower, bb_width = _bollinger(g["Close"], p["bb_len"], p["bb_stds"])
        g["bb_mid"] = bb_mid
        g["bb_upper"] = bb_upper
        g["bb_lower"] = bb_lower
        g["bb_width"] = bb_width
        # PDH/PDL
        g = _prev_day_high_low(g)
        # Volume z-score
        roll = g["Vol"].rolling(p["vol_lookback"])
        g["vol_mean"] = roll.mean()
        g["vol_std"] = roll.std(ddof=0)
        g["vol_z"] = (g["Vol"] - g["vol_mean"]) / (g["vol_std"] + 1e-12)
        return g

    if by_ticker and "Ticker" in work.columns:
        out = work.groupby("Ticker", group_keys=False).apply(_calc)
    else:
        out = _calc(work)

    return out.reset_index(drop=True)
