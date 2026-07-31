#!/usr/bin/env python3
"""Feature engineering + labeling for ML model.

Features (40+):
  - Price-based: returns (1, 5, 10, 30 candles), log returns
  - SMA ratios: sma5/sma14, sma14/sma20, sma20/sma50
  - RSI: rsi14, rsi2 (Connors)
  - Bollinger: %b (position within bands), bandwidth
  - MACD: histogram, line, signal
  - ATR: volatility (atr/price)
  - Volume: volume ratio, OBV slope
  - Stochastic: %K, %D
  - Higher TF context: 1h return, 1h RSI, 1d return, 1d trend
  - Time: hour of day, day of week

Labels (target):
  - forward_return_30min = (close[t+6] - close[t]) / close[t]  (6 candles × 5min = 30min)
  - label = 1 if forward_return > threshold (e.g., 0.1%) else 0
  - Also: label_short = 1 if forward_return < -threshold

Output: X (features matrix), y_long (binary), y_short (binary)
"""
import numpy as np
from typing import Dict, Tuple


def compute_features(aligned: Dict[str, np.ndarray]) -> Tuple[np.ndarray, list]:
    """Compute all features from aligned multi-timeframe data.
    
    Returns: (X, feature_names)
    """
    close5 = aligned["5min_close"]
    high5 = aligned["5min_high"]
    low5 = aligned["5min_low"]
    open5 = aligned["5min_open"]
    vol5 = aligned["5min_volume"]
    time5 = aligned["time"]
    
    n = len(close5)
    features = {}
    
    # === Returns ===
    features["ret_1"] = np.diff(close5, prepend=close5[0]) / close5
    features["ret_5"] = (close5 - np.roll(close5, 5)) / np.roll(close5, 5)
    features["ret_10"] = (close5 - np.roll(close5, 10)) / np.roll(close5, 10)
    features["ret_30"] = (close5 - np.roll(close5, 30)) / np.roll(close5, 30)
    features["ret_5_log"] = np.log(close5 / np.roll(close5, 5))
    
    # === SMA ratios ===
    sma5 = np.convolve(close5, np.ones(5)/5, mode="same")
    sma14 = np.convolve(close5, np.ones(14)/14, mode="same")
    sma20 = np.convolve(close5, np.ones(20)/20, mode="same")
    sma50 = np.convolve(close5, np.ones(50)/50, mode="same")
    
    features["sma5_sma14"] = sma5 / (sma14 + 1e-10)
    features["sma14_sma20"] = sma14 / (sma20 + 1e-10)
    features["sma20_sma50"] = sma20 / (sma50 + 1e-10)
    features["price_sma20"] = close5 / (sma20 + 1e-10)
    features["price_sma50"] = close5 / (sma50 + 1e-10)
    
    # === RSI ===
    deltas = np.diff(close5, prepend=close5[0])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    # RSI(14) — rolling mean
    def rolling_mean(arr, window):
        ret = np.cumsum(arr)
        ret[window:] = ret[window:] - ret[:-window]
        return ret / window
    
    avg_gain = rolling_mean(gains, 14)
    avg_loss = rolling_mean(losses, 14)
    rs = avg_gain / (avg_loss + 1e-10)
    features["rsi14"] = 100 - 100 / (1 + rs)
    
    # RSI(2) — Connors
    avg_gain2 = rolling_mean(gains, 2)
    avg_loss2 = rolling_mean(losses, 2)
    rs2 = avg_gain2 / (avg_loss2 + 1e-10)
    features["rsi2"] = 100 - 100 / (1 + rs2)
    
    # === Bollinger Bands ===
    sma20_safe = sma20.copy()
    std20 = np.array([np.std(close5[max(0, i-19):i+1]) for i in range(n)])
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_width = (4 * std20) / (sma20 + 1e-10)
    
    features["bb_pct_b"] = (close5 - bb_lower) / (4 * std20 + 1e-10)  # 0=lower, 1=upper, 0.5=middle
    features["bb_width"] = bb_width
    features["price_bb_upper"] = close5 / (bb_upper + 1e-10)
    features["price_bb_lower"] = close5 / (bb_lower + 1e-10)
    
    # === MACD ===
    def ema(arr, period):
        k = 2 / (period + 1)
        result = np.zeros_like(arr)
        result[0] = arr[0]
        for i in range(1, len(arr)):
            result[i] = arr[i] * k + result[i-1] * (1 - k)
        return result
    
    ema12 = ema(close5, 12)
    ema26 = ema(close5, 26)
    macd_line = ema12 - ema26
    macd_signal = ema(macd_line, 9)
    features["macd_hist"] = macd_line - macd_signal
    features["macd_line"] = macd_line / (close5 + 1e-10)  # normalize
    features["macd_signal"] = macd_signal / (close5 + 1e-10)
    
    # === ATR (volatility) ===
    tr = np.maximum(high5 - low5, np.maximum(
        np.abs(high5 - np.roll(close5, 1)),
        np.abs(low5 - np.roll(close5, 1))
    ))
    atr14 = rolling_mean(tr, 14)
    features["atr_pct"] = atr14 / (close5 + 1e-10)  # ATR as % of price
    
    # === Volume ===
    vol_avg20 = rolling_mean(vol5.astype(float), 20)
    features["vol_ratio"] = vol5 / (vol_avg20 + 1e-10)
    
    # OBV slope (simplified)
    obv = np.cumsum(np.where(deltas > 0, vol5, -vol5))
    features["obv_slope"] = (obv - np.roll(obv, 10)) / (np.roll(obv, 10) + 1e-10)
    
    # === Stochastic ===
    hh14 = np.array([np.max(high5[max(0, i-13):i+1]) for i in range(n)])
    ll14 = np.array([np.min(low5[max(0, i-13):i+1]) for i in range(n)])
    features["stoch_k"] = (close5 - ll14) / (hh14 - ll14 + 1e-10) * 100
    
    # === Higher timeframe context ===
    # 1hour
    if "1hour_close" in aligned:
        close1h = aligned["1hour_close"]
        features["1h_ret"] = (close1h - np.roll(close1h, 1)) / (np.roll(close1h, 1) + 1e-10)
        sma1h = np.convolve(close1h, np.ones(10)/10, mode="same")
        features["1h_trend"] = close1h / (sma1h + 1e-10)  # >1 = uptrend
        
        deltas1h = np.diff(close1h, prepend=close1h[0])
        gains1h = np.where(deltas1h > 0, deltas1h, 0)
        losses1h = np.where(deltas1h < 0, -deltas1h, 0)
        avg_g1h = rolling_mean(gains1h, 14)
        avg_l1h = rolling_mean(losses1h, 14)
        features["1h_rsi"] = 100 - 100 / (1 + avg_g1h / (avg_l1h + 1e-10))
    
    # 1day
    if "1day_close" in aligned:
        close1d = aligned["1day_close"]
        features["1d_ret"] = (close1d - np.roll(close1d, 1)) / (np.roll(close1d, 1) + 1e-10)
        sma1d = np.convolve(close1d, np.ones(5)/5, mode="same")
        features["1d_trend"] = close1d / (sma1d + 1e-10)
    
    # === Time features ===
    ts_seconds = time5 / 1000
    hours = (ts_seconds // 3600 % 24).astype(float)
    dow = (ts_seconds // 86400 % 7).astype(float)
    features["hour"] = hours / 24.0  # normalized 0-1
    features["day_of_week"] = dow / 7.0
    
    # === ADX (simplified trend strength) ===
    up_moves = np.where(deltas > 0, 1, 0)
    down_moves = np.where(deltas < 0, 1, 0)
    adx = np.abs(rolling_mean(up_moves, 14) - rolling_mean(down_moves, 14)) * 100
    features["adx"] = adx
    
    # Fix NaN/inf
    feature_names = sorted(features.keys())
    X = np.column_stack([features[name] for name in feature_names])
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
    
    # Zero out first 50 rows (not enough history)
    X[:50] = 0
    
    return X, feature_names


def compute_labels(aligned: Dict[str, np.ndarray], horizon: int = 6, threshold: float = 0.001) -> Tuple[np.ndarray, np.ndarray]:
    """Compute labels: will price go up/down by threshold in next `horizon` candles?
    
    Args:
        aligned: multi-timeframe data
        horizon: number of 5min candles to look forward (6 = 30 min)
        threshold: minimum return to count as "up" (0.001 = 0.1%)
    
    Returns: (y_long, y_short) — binary arrays
    """
    close = aligned["5min_close"]
    n = len(close)
    
    # Forward return
    forward_close = np.roll(close, -horizon)
    forward_close[-horizon:] = close[-1]  # can't see future → use last price
    forward_return = (forward_close - close) / (close + 1e-10)
    
    y_long = (forward_return > threshold).astype(int)
    y_short = (forward_return < -threshold).astype(int)
    
    # Zero out last `horizon` candles (can't label future)
    y_long[-horizon:] = 0
    y_short[-horizon:] = 0
    
    return y_long, y_short


if __name__ == "__main__":
    # Test
    from ml_data_pipeline import download_multi_timeframe, align_timeframes
    
    print("=== Feature engineering test ===")
    data = download_multi_timeframe("SBER", days=30)
    aligned = align_timeframes(data)
    
    X, names = compute_features(aligned)
    y_long, y_short = compute_labels(aligned)
    
    print(f"\nFeatures: {X.shape} ({len(names)} features)")
    print(f"Labels: long={y_long.sum()} ({y_long.mean()*100:.1f}%), short={y_short.sum()} ({y_short.mean()*100:.1f}%)")
    print(f"\nFeature names:")
    for name in names:
        print(f"  {name}")
