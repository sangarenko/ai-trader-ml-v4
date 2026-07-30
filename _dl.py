"""Data loader — fetch candle data from server or load from cache.

Cache: /home/z/backtest/fresh_{TICKER}.json (3 days, 5min candles)
Server: http://2.26.122.152:3002/api/candles?ticker=X&days=N&interval=5min
"""
import json
import os
import urllib.request
from typing import List, Dict, Tuple
from datetime import datetime


TICKERS = ['SBER', 'GAZP', 'LKOH', 'GMKN', 'VTBR', 'MGNT', 'TATN', 'MTSS', 'NVTK', 'PLZL', 'ROSN']
CACHE_DIR = '/home/z/backtest'  # server cache
LOCAL_CACHE = '/home/z/my-project/training/data_cache'


def fetch_candles_server(ticker: str, days: int = 7, server: str = 'http://2.26.122.152:3002') -> List[Dict]:
    """Fetch candles from server API."""
    url = f'{server}/api/candles?ticker={ticker}&days={days}&interval=5min'
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = json.loads(r.read())
        return [
            {
                'time': int(datetime.fromisoformat(c[0].replace('Z', '+00:00')).timestamp() * 1000),
                'open': c[1], 'high': c[2], 'low': c[3], 'close': c[4], 'volume': c[5]
            }
            for c in data.get('candles', [])
        ]
    except Exception as e:
        print(f'  fetch {ticker} failed: {e}')
        return []


def _normalize_candles(raw) -> List[Dict]:
    """Convert raw candle arrays [time, o, h, l, c, v] to dicts."""
    if not raw:
        return []
    if isinstance(raw[0], dict):
        return raw  # already dicts
    # Convert arrays to dicts
    result = []
    for c in raw:
        if len(c) >= 6:
            t = c[0]
            if isinstance(t, str):
                t = int(datetime.fromisoformat(t.replace('Z', '+00:00')).timestamp() * 1000)
            result.append({
                'time': t, 'open': c[1], 'high': c[2], 'low': c[3], 'close': c[4], 'volume': c[5]
            })
    return result


def load_candles(ticker: str, days: int = 3) -> List[Dict]:
    """Load candles — try local cache, then server cache, then fetch."""
    os.makedirs(LOCAL_CACHE, exist_ok=True)
    cache_file = f'{LOCAL_CACHE}/{ticker}_{days}d.json'

    # 1. Local cache
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            raw = json.load(f)
        return _normalize_candles(raw)

    # 2. Server cache — try week_ (7-day) first, then fresh_ (3-day)
    for prefix in ['week_', 'fresh_']:
        server_cache = f'{CACHE_DIR}/{prefix}{ticker}.json'
        if os.path.exists(server_cache):
            with open(server_cache) as f:
                data = json.load(f)
            candles = _normalize_candles(data.get('candles', []))
            with open(cache_file, 'w') as f:
                json.dump(candles, f)
            return candles

    # 3. Fetch from server API
    candles = fetch_candles_server(ticker, days)
    if candles:
        with open(cache_file, 'w') as f:
            json.dump(candles, f)
    return candles


def load_all_tickers(days: int = 3) -> Dict[str, List[Dict]]:
    """Load candles for all 11 tickers."""
    data = {}
    print(f'Loading {days}-day candles for {len(TICKERS)} tickers...')
    for ticker in TICKERS:
        candles = load_candles(ticker, days)
        if candles:
            data[ticker] = candles
            print(f'  {ticker}: {len(candles)} candles')
        else:
            print(f'  {ticker}: FAILED')
    print(f'Loaded {len(data)}/{len(TICKERS)} tickers, total {sum(len(c) for c in data.values())} candles')
    return data


def split_data(data: Dict[str, List[Dict]], train_pct: float = 0.6, val_pct: float = 0.2) -> Tuple[Dict, Dict, Dict]:
    """Split data into train/val/test sets.

    Args:
        data: {ticker: [candles]}
        train_pct: 0.6 (60% train)
        val_pct: 0.2 (20% validation)

    Returns:
        (train_data, val_data, test_data)
    """
    train, val, test = {}, {}, {}
    for ticker, candles in data.items():
        n = len(candles)
        train_end = int(n * train_pct)
        val_end = int(n * (train_pct + val_pct))
        train[ticker] = candles[:train_end]
        val[ticker] = candles[train_end:val_end]
        test[ticker] = candles[val_end:]
    return train, val, test


if __name__ == '__main__':
    # Test data loading
    data = load_all_tickers(days=3)
    train, val, test = split_data(data)
    print()
    print(f'Train: {sum(len(c) for c in train.values())} candles')
    print(f'Val:   {sum(len(c) for c in val.values())} candles')
    print(f'Test:  {sum(len(c) for c in test.values())} candles')
