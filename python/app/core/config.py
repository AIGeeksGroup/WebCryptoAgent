from __future__ import annotations

import os
import time

from .io_utils import ensure_path

TRADE_PAIRS = {
    "BTC": {"pair": "BTC_USDT", "api_symbol": "BTCUSDT"},
    "ETH": {"pair": "ETH_USDT", "api_symbol": "ETHUSDT"},
    "POL": {"pair": "POL_USDT", "api_symbol": "POLUSDT"},
}


def ts() -> str:
    """Compact UTC timestamp used to tag artefacts."""
    return time.strftime("%Y%m%d%H%M%S", time.gmtime())


def save_screenshot(driver, path: str, sleep: float = 0.0) -> str:
    if sleep > 0:
        time.sleep(sleep)
    ensure_path(path)
    driver.save_screenshot(path)
    return os.path.abspath(path)
