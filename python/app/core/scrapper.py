from __future__ import annotations

import logging
import os
import time
from typing import Dict

import requests
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

from .config import TRADE_PAIRS, save_screenshot


def close_popups(driver) -> None:
    """Best-effort dismissal of common modal overlays."""
    try:
        driver.switch_to.default_content()
        body = driver.find_element(By.TAG_NAME, "body")
        body.send_keys(Keys.ESCAPE)
    except Exception:

"""

Uncomment here and get a place to scrape the data to the folder data/ohlcv


TICKER_API = Find some source

"""
def _fetch_ticker(symbol: str) -> Dict[str, str]:
    try:
        resp = requests.get(TICKER_API, params={"symbol": symbol}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return {
            "price_str": data.get("lastPrice"),
            "price_change_pct": data.get("priceChangePercent"),
            "high_24h_str": data.get("highPrice"),
            "low_24h_str": data.get("lowPrice"),
            "volume_24h_str": data.get("quoteVolume"),
            "open_price": data.get("openPrice"),
            "close_time": data.get("closeTime"),
            "raw": data,
        }
    except Exception as exc:
        logging.warning("Unable to fetch ticker for %s: %s", symbol, exc)
        return {}


def scrape_price_page(
    driver,
    coin_sym: str,
    out_dir: str,
    window_label: str = "1h",
) -> Dict[str, str]:
    pair_info = TRADE_PAIRS.get(coin_sym)
    if not pair_info:
        raise ValueError(f"No trade pair configured for {coin_sym}")
    pair = pair_info["pair"]
    api_symbol = pair_info["api_symbol"]
    url = f"url"
    driver.get(url)
    time.sleep(4)
    close_popups(driver)

    chart_png = os.path.join(out_dir, f"market/{coin_sym}_chart_{window_label}.png")
    save_screenshot(driver, chart_png, sleep=0.5)

    stats = _fetch_ticker(api_symbol)
    stats["source"] = "price_page"
    stats["url"] = url
    stats["window"] = window_label
    stats["screenshot"] = chart_png
    stats["pair"] = pair
    stats["api_symbol"] = api_symbol
    return stats
