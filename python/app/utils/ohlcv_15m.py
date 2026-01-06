import time
import csv
import pathlib
import requests
from datetime import datetime, timedelta, timezone

OUT_DIR = "data/ohlcv"
SYMBOLS = ["WBTCUSDT","POLUSDC","ETHUSDC"]
INTERVAL = "15m"
LOOKBACK_DAYS = 7
REQ_TIMEOUT = 15

ticker_API = ""

def ensure_dir(path: pathlib.Path):
    path.mkdir(parents=True, exist_ok=True)

def fetch_klines(symbol: str, interval: str, start_time: int, end_time: int):
    """Fetch klines from online between start_time and end_time (ms)."""
    all_bars = []
    while start_time < end_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time,
            "endTime": end_time,
            "limit": 1000
        }
        r = requests.get(ticker_API, params=params, timeout=REQ_TIMEOUT)
        if r.status_code != 200:
            print(f"Error {r.status_code}: {r.text}")
            break
        data = r.json()
        if not data:
            break
        all_bars.extend(data)
        last_ts = data[-1][0]
        start_time = last_ts + 1
        time.sleep(0.25)  # be polite to API
    return all_bars

def save_csv(symbol: str, bars):
    out_dir = pathlib.Path(OUT_DIR)
    ensure_dir(out_dir)
    out_path = out_dir / f"{symbol}_{INTERVAL}.csv"
    write_header = not out_path.exists()

    with out_path.open("a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["ts", "iso", "open", "high", "low", "close", "volume", "trades"])
        for b in bars:
            ts = b[0] // 1000
            iso = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            w.writerow([
                ts, iso, b[1], b[2], b[3], b[4], b[5], b[8]
            ])
    print(f"Appended {len(bars)} rows → {out_path}")

def main():
    end_time = int(time.time() * 1000)
    start_time = int((datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)).timestamp() * 1000)

    for sym in SYMBOLS:
        print(f"\n=== {sym} ({INTERVAL}) ===")
        bars = fetch_klines(sym, INTERVAL, start_time, end_time)
        print(f"Fetched {len(bars)} bars")
        save_csv(sym, bars)

if __name__ == "__main__":
    main()
