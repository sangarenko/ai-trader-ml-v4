#!/usr/bin/env python3
"""Multi-timeframe data pipeline for ML.

Downloads candles from MOEX ISS API at multiple timeframes:
  - 5min  — entry signals (точка входа)
  - 15min — short-term trend context
  - 1hour — medium-term trend (regime detection)
  - 1day  — global trend (не торговать против)

Aligns all timeframes to 5min grid (forward-fill higher TFs).
Output: numpy arrays ready for ML.

Cache: data_cache/mtf_{ticker}_{days}d.npz
"""
import os
import json
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple

import numpy as np

TICKERS = ["SBER", "GAZP", "LKOH", "GMKN", "VTBR", "MGNT", "TATN", "MTSS", "NVTK", "PLZL", "ROSN"]
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")
USER_AGENT = "ai-trader-ml/1.0"

# MOEX ISS intervals: 1, 10, 60, 24, D, W, M
# We want: 5min (use 1min × 5 aggregation), 15min (1min × 15), 1hour (60), 1day (24)
# Actually MOEX supports interval=10 for 10min. For 5min we need 1min and aggregate.
# But 1min returns max 500 per request → 1 day = ~480 candles → need many requests.
# Simpler: use 10min as base (close to 5min), 60min for 1hour, 24 for daily.

TF_CONFIG = {
    "5min":  {"interval": 10, "minutes": 10,  "label": "10min→5min proxy"},  # MOEX 10min as proxy
    "15min": {"interval": 10, "minutes": 10,  "label": "10min (use as 15min proxy)"},
    "1hour": {"interval": 60, "minutes": 60,  "label": "1hour"},
    "1day":  {"interval": 24, "minutes": 1440, "label": "daily"},
}


def fetch_moex_candles(ticker: str, interval: int, from_date: str, to_date: str = None) -> List[Dict]:
    """Fetch candles from MOEX ISS."""
    if to_date is None:
        to_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    all_candles = []
    start = 0
    
    while True:
        url = (f"https://iss.moex.com/iss/engines/stock/markets/shares/securities/{ticker}"
               f"/candles.json?interval={interval}&from={from_date}&till={to_date}"
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
                dt = datetime.strptime(c[0], "%Y-%m-%d %H:%M:%S")
                msk_tz = timezone(timedelta(hours=3))
                dt = dt.replace(tzinfo=msk_tz)
                ts_ms = int(dt.timestamp() * 1000)
                
                all_candles.append({
                    "time": ts_ms,
                    "open": float(c[2]),
                    "high": float(c[3]),
                    "low": float(c[4]),
                    "close": float(c[5]),
                    "volume": int(c[6]),
                })
            
            if len(candles) < 500:
                break
            start += 500
            time.sleep(0.05)
        except Exception as e:
            print(f"  {ticker}: fetch error: {e}")
            break
    
    return all_candles


def download_multi_timeframe(ticker: str, days: int = 180) -> Dict[str, np.ndarray]:
    """Download all timeframes for a ticker. Returns dict of numpy arrays."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"mtf_{ticker}_{days}d.npz")
    
    # Check cache (1 day TTL)
    if os.path.exists(cache_path):
        age = time.time() - os.path.getmtime(cache_path)
        if age < 86400:
            print(f"  {ticker}: cached (age {age/3600:.1f}h)")
            return dict(np.load(cache_path))
    
    from_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    
    result = {}
    for tf_name, config in TF_CONFIG.items():
        candles = fetch_moex_candles(ticker, config["interval"], from_date)
        if not candles:
            print(f"  {ticker} {tf_name}: 0 candles (FAIL)")
            continue
        
        # Convert to numpy arrays
        arr = np.array([
            [c["time"], c["open"], c["high"], c["low"], c["close"], c["volume"]]
            for c in candles
        ], dtype=np.float64)
        
        result[f"{tf_name}_time"] = arr[:, 0]
        result[f"{tf_name}_open"] = arr[:, 1]
        result[f"{tf_name}_high"] = arr[:, 2]
        result[f"{tf_name}_low"] = arr[:, 3]
        result[f"{tf_name}_close"] = arr[:, 4]
        result[f"{tf_name}_volume"] = arr[:, 5]
        
        print(f"  {ticker} {tf_name}: {len(candles)} candles")
    
    # Save cache
    np.savez(cache_path, **result)
    return result


def align_timeframes(data: Dict[str, np.ndarray], base_tf: str = "5min") -> Dict[str, np.ndarray]:
    """Align all timeframes to the base timeframe grid.
    
    For each 5min candle, attach the latest 15min/1hour/1day candle values.
    Uses forward-fill (last known value from higher TF).
    """
    base_time = data[f"{base_tf}_time"]
    n = len(base_time)
    
    aligned = {}
    # Copy base TF
    for key in ["open", "high", "low", "close", "volume"]:
        aligned[f"{base_tf}_{key}"] = data[f"{base_tf}_{key}"]
    aligned["time"] = base_time
    
    # For each higher TF, forward-fill onto base grid
    for tf in ["15min", "1hour", "1day"]:
        tf_time = data.get(f"{tf}_time")
        if tf_time is None:
            # Use base TF as fallback
            for key in ["open", "high", "low", "close", "volume"]:
                aligned[f"{tf}_{key}"] = data[f"{base_tf}_{key}"]
            continue
        
        for key in ["open", "high", "low", "close", "volume"]:
            tf_vals = data[f"{tf}_{key}"]
            # For each base candle, find the latest TF candle with time <= base time
            # Use searchsorted for efficiency
            indices = np.searchsorted(tf_time, base_time, side="right") - 1
            indices = np.clip(indices, 0, len(tf_vals) - 1)
            aligned[f"{tf}_{key}"] = tf_vals[indices]
    
    return aligned


if __name__ == "__main__":
    # Test: download SBER multi-timeframe
    print("=== Multi-timeframe data pipeline test ===")
    data = download_multi_timeframe("SBER", days=30)
    print(f"\nDownloaded {len(data)} arrays")
    for key in sorted(data.keys()):
        print(f"  {key}: shape={data[key].shape}")
    
    # Test alignment
    print("\n=== Alignment test ===")
    aligned = align_timeframes(data)
    print(f"Aligned {len(aligned)} arrays")
    for key in sorted(aligned.keys()):
        print(f"  {key}: shape={aligned[key].shape}")
