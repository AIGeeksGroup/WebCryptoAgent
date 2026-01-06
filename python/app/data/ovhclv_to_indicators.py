from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

try:
    from ..utils.indicators_calc import compute_indicators
except ImportError:  # script execution without package context
    project_root = Path(__file__).resolve().parents[3]
    python_root = project_root / "python"
    if str(python_root) not in sys.path:
        sys.path.insert(0, str(python_root))
    from app.utils.indicators_calc import compute_indicators

SYMBOLS = ["ETHUSDC", "POLUSDC", "WBTCUSDT"]

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "ohlcv"


def generate_indicator_csv(symbol: str) -> None:
    input_path = DATA_DIR / f"{symbol}_15m.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Input OHLCV file not found: {input_path}")

    df = pd.read_csv(input_path)
    df = df.rename(
        columns={
            "iso": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Vol",  # actual traded volume
            "trades": "Trades",  # optional, maintained for consistency
        }
    )
    df["Ticker"] = symbol

    df_indicators = compute_indicators(df, by_ticker=True)

    output_path = DATA_DIR / f"{symbol}_15m_indicators.csv"
    df_indicators.to_csv(output_path, index=False)
    print(f"{symbol} computed successfully.")


def main(symbols: list[str]) -> None:
    for symbol in symbols:
        generate_indicator_csv(symbol)


if __name__ == "__main__":
    main(SYMBOLS)
