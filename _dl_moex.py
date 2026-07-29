"""Data loader — MOEX ISS API (replaces T-Bank server API).

Why MOEX:
  - No token required (free, public)
  - 2+ years of history (vs T-Bank sandbox: 3 days)
  - Real MOEX volumes
  - Verified 1:1 identical to T-Bank API (avg diff: 0.0000%)

Cache: data_cache/moex_{days}d.json
"""
import json
import os
import urllib.request
import time
from typing import List, Dict, Tuple
from datetime import datetime, timedelta, timezone


TICKERS = ['SBER', 'GAZP', 'LKOH', 'GMKN', 'VTBR', 'MGNT', 'TATN', 'MTSS', 'NVTK', 'PLZL', 'ROSN']
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_cache')
USER_AGENT = 'ai-trader-evolution/1.0'
INTERVAL = 10  # 10min candles (MOEX ISS supports: 1, 10, 60, 24, D, W, M, Q)


def fetch_moex_candles(ticker: str, from_date: str, to_date: str = None) -> List[Dict]:
    """Fetch 10min candles from MOEX ISS for given ticker."""
    if to_date is None:
        to_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    all_candles = []
    start = 0

    while True:
        url = (f"https://iss.moex.com/iss/engines/stock/markets/shares/securities/{ticker}"
               f"/candles.json?interval={INTERVAL}&from={from_date}&till={to_date}"
               f"&start={start}&iss.meta=off&iss.only=candles"
               f"&candles.columns=begin,end,open,high,low,close,volume")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            resp = urllib.request.urlopen(req, timeout=30)
            data = json.loads(resp.read())
            candles = data.get("candles", {}).get("data", [])

            if not candles:
                break

            for c in candles:
                # MOEX: [begin, end, open, high, low, close, volume]
                # Convert time string to ms timestamp (for compatibility with backtest)
                begin_str = c[0]  # '2026-07-29 10:00:00' (MSK)
                dt = datetime.strptime(begin_str, "%Y-%m-%d %H:%M:%S")
                # MOEX time is MSK = UTC+3
                msk_tz = timezone(timedelta(hours=3))
                dt = dt.replace(tzinfo=msk_tz)
                ts_ms = int(dt.timestamp() * 1000)

                all_candles.append({
                    "time": ts_ms,
                    "time_str": begin_str,  # keep original for debugging
                    "open": float(c[2]),
                    "high": float(c[3]),
                    "low": float(c[4]),
                    "close": float(c[5]),
                    "volume": int(c[6]),
                })

            if len(candles) < 500:
                break
            start += 500
            time.sleep(0.05)  # be nice to MOEX
        except Exception as e:
            print(f"  {ticker}: fetch error at start={start}: {e}")
            break

    return all_candles


def load_candles(ticker: str, days: int = 7) -> List[Dict]:
    """Load candles for one ticker — uses MOEX cache or fetches fresh."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"moex_{days}d_{ticker}.json")

    # Cache for 1 day
    if os.path.exists(cache_file):
        age = time.time() - os.path.getmtime(cache_file)
        if age < 86400:
            with open(cache_file) as f:
                return json.load(f)

    from_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    candles = fetch_moex_candles(ticker, from_date)
    if candles:
        with open(cache_file, "w") as f:
            json.dump(candles, f)
    return candles


def load_all_tickers(days: int = 7) -> Dict[str, List[Dict]]:
    """Load candles for all 11 tickers from MOEX."""
    data = {}
    print(f"Loading {days}-day MOEX 10min candles for {len(TICKERS)} tickers...")
    for ticker in TICKERS:
        candles = load_candles(ticker, days)
        if candles:
            data[ticker] = candles
            print(f"  {ticker}: {len(candles)} candles")
        else:
            print(f"  {ticker}: FAILED")
    print(f"Loaded {len(data)}/{len(TICKERS)} tickers, total {sum(len(c) for c in data.values())} candles")
    return data


def split_data(data: Dict[str, List[Dict]], train_pct: float = 0.6, val_pct: float = 0.2) -> Tuple[Dict, Dict, Dict]:
    """Split data into train/val/test sets chronologically."""
    train, val, test = {}, {}, {}
    for ticker, candles in data.items():
        n = len(candles)
        train_end = int(n * train_pct)
        val_end = int(n * (train_pct + val_pct))
        train[ticker] = candles[:train_end]
        val[ticker] = candles[train_end:val_end]
        test[ticker] = candles[val_end:]
    return train, val, test


if __name__ == "__main__":
    data = load_all_tickers(days=7)
    train, val, test = split_data(data)
    print()
    print(f"Train: {sum(len(c) for c in train.values())} candles")
    print(f"Val:   {sum(len(c) for c in val.values())} candles")
    print(f"Test:  {sum(len(c) for c in test.values())} candles")
