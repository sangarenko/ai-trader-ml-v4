#!/usr/bin/env python3
"""Feature pipeline v4 — clean, causal, ~22 features (no duplicates).

This is the v4 feature pipeline for the ML trading system. Compared to the
legacy ``ml_features.py`` (which had ~30 features with many duplicates — RSI14
+ RSI2, SMA5/SMA14/SMA20/SMA50 absolute + ratios, macd_line + macd_signal +
macd_hist), this module:

  * keeps only non-redundant indicators
  * adds cross-asset features (market_breadth, sber_gazp_corr)
  * adds volatility regime features (vol_regime, trend_strength)
  * is strictly CAUSAL — no look-ahead bias anywhere
    (uses cumsum-based trailing SMA / rolling mean, slicing for returns,
    forward-fill alignment for higher timeframes via align_timeframes)

Features (22 total, grouped by family):

  Returns (4):
      ret_1, ret_5, ret_10, ret_30                # 1/5/10/30 bar % returns

  Momentum (1):
      rsi14                                       # Wilder RSI(14) — drops RSI2

  Trend / SMA ratios (3):
      sma5_sma14, sma14_sma20, sma20_sma50        # ratios only — no absolute SMAs

  Bollinger (2):
      bb_pct_b, bb_width                          # %B + bandwidth

  MACD (1):
      macd_hist                                   # histogram captures line+signal

  Volatility / range (3):
      atr_pct, stoch_k, vol_ratio                 # ATR%, stochastic %K, vol ratio

  Higher-timeframe context (2):
      1h_ret, 1d_ret                              # aligned 1-hour, 1-day returns

  Time (2):
      hour, day_of_week                           # normalized MSK hour / weekday

  Cross-asset (2) [NEW — require all_tickers_data]:
      market_breadth                              # % of tickers with positive ret_5
      sber_gazp_corr                              # 20-bar rolling SBER-GAZP return corr

  Volatility regime (2) [NEW]:
      vol_regime                                  # ATR percentile rank (0-1) over last 100 bars
      trend_strength                              # ADX / 100 (normalized 0-1)

Signature:
    compute_features_v4(aligned, all_tickers_data=None) -> (X, feature_names)

Args:
    aligned: dict of multi-timeframe arrays for ONE ticker (same format as
        ml_data_pipeline.align_timeframes() output — keys like "5min_close",
        "1hour_close", "1day_close", "time", ...). The "5min_*" arrays are
        actually 10-min candles (MOEX ISS interval=10), but we keep the v1
        naming for compatibility.
    all_tickers_data: optional dict {ticker: aligned_data} for cross-asset
        features. If None or missing SBER/GAZP, cross-asset features are set
        to neutral defaults (0.5 / 0.0).

Returns:
    X (np.ndarray of shape (n_bars, n_features)), feature_names (list[str])
"""
import numpy as np
from typing import Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- #
# Causal helpers
# --------------------------------------------------------------------------- #
def _causal_sma(arr: np.ndarray, w: int) -> np.ndarray:
    """Trailing SMA via cumsum. Strictly causal: SMA[i] = mean(arr[i-w+1 .. i]).

    For the first ``w-1`` bars the trailing window is partial, so we use the
    cumulative mean up to that index instead.
    """
    n = len(arr)
    out = np.empty(n, dtype=float)
    c = np.cumsum(arr, dtype=float)
    # For i < w: cumulative mean = c[i] / (i+1)
    counts = np.minimum(np.arange(1, n + 1), w)
    out[:w] = c[:w] / counts[:w]
    if n > w:
        out[w:] = (c[w:] - c[:-w]) / w
    return out


def _causal_rolling_mean(arr: np.ndarray, w: int) -> np.ndarray:
    """Trailing rolling mean via cumsum. Same as _causal_sma but kept separate
    for clarity (used for RSI gain/loss averages, ATR, vol_avg)."""
    return _causal_sma(arr, w)


def _causal_rolling_std(arr: np.ndarray, w: int) -> np.ndarray:
    """Trailing population std via cumsum / cumsum-of-squares. Causal.

    Uses population std (ddof=0). For the warm-up period (i < w-1) we use the
    partial-window std to avoid zero variance early on.
    """
    n = len(arr)
    c1 = np.cumsum(arr, dtype=float)
    c2 = np.cumsum(arr * arr, dtype=float)
    out = np.zeros(n)
    counts = np.minimum(np.arange(1, n + 1), w)
    # Partial-window variance for i < w
    mean_partial = c1[:w] / counts[:w]
    var_partial = c2[:w] / counts[:w] - mean_partial ** 2
    out[:w] = np.sqrt(np.clip(var_partial, 0, None))
    if n > w:
        mean_full = (c1[w:] - c1[:-w]) / w
        c2_full = (c2[w:] - c2[:-w]) / w
        var_full = c2_full - mean_full ** 2
        out[w:] = np.sqrt(np.clip(var_full, 0, None))
    return out


def _causal_ema(arr: np.ndarray, period: int) -> np.ndarray:
    """EMA via sequential loop. Strictly causal (uses only arr[0..i])."""
    k = 2.0 / (period + 1)
    out = np.empty_like(arr, dtype=float)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = arr[i] * k + out[i - 1] * (1 - k)
    return out


def _causal_returns(close: np.ndarray) -> np.ndarray:
    """ret_1 = (close[i] - close[i-1]) / close[i-1]. ret_1[0] = 0 (no prior bar)."""
    r = np.zeros(len(close))
    if len(close) > 1:
        r[1:] = (close[1:] - close[:-1]) / (close[:-1] + 1e-10)
    return r


def _causal_ret_n(close: np.ndarray, n: int) -> np.ndarray:
    """n-bar trailing return. ret_n[i<0] = 0 (warm-up)."""
    r = np.zeros(len(close))
    if len(close) > n:
        r[n:] = (close[n:] - close[:-n]) / (close[:-n] + 1e-10)
    return r


# --------------------------------------------------------------------------- #
# Cross-asset helpers
# --------------------------------------------------------------------------- #
def _align_to_base_grid(src_time: np.ndarray, src_vals: np.ndarray,
                        base_time: np.ndarray) -> np.ndarray:
    """Forward-fill ``src_vals`` onto ``base_time`` grid (strictly causal).

    For each base timestamp ``t`` we pick the latest src bar whose own
    timestamp is <= ``t``. No future src bar is ever used.
    """
    if src_time is None or len(src_time) == 0:
        return np.full(len(base_time), src_vals[-1] if len(src_vals) else 0.0)
    idx = np.searchsorted(src_time, base_time, side="right") - 1
    idx = np.clip(idx, 0, len(src_vals) - 1)
    return src_vals[idx]


def _compute_market_breadth(all_tickers_data: Optional[Dict[str, Dict[str, np.ndarray]]],
                             base_time: np.ndarray,
                             n: int) -> np.ndarray:
    """% of tickers with positive ret_5 at each base timestamp.

    Causal: ret_5 at time t only uses close[t-5 .. t] of each ticker.

    Returns neutral 0.5 if no ticker data is available.
    """
    breadth = np.full(n, 0.5)
    if not all_tickers_data:
        return breadth

    pos_count = np.zeros(n)
    total = 0
    for ticker, aligned in all_tickers_data.items():
        close = aligned.get("5min_close")
        if close is None or len(close) < 6:
            continue
        # ret_5 on the ticker's own grid (causal)
        ret5 = _causal_ret_n(close, 5)
        t = aligned.get("time")
        if t is None or len(t) != len(close):
            # Same grid as base assumed — direct index alignment
            if len(close) == n:
                pos_count += (ret5 > 0).astype(float)
                total += 1
        else:
            # Forward-fill ret5 onto base grid
            ret5_aligned = _align_to_base_grid(t, ret5, base_time)
            pos_count += (ret5_aligned > 0).astype(float)
            total += 1

    if total > 0:
        breadth = pos_count / total
    return breadth


def _compute_sber_gazp_corr(all_tickers_data: Optional[Dict[str, Dict[str, np.ndarray]]],
                            aligned: Dict[str, np.ndarray],
                            n: int,
                            window: int = 20) -> np.ndarray:
    """Rolling 20-bar correlation of SBER vs GAZP 1-bar returns.

    Causal: corr[i] uses only returns up to and including bar i.

    Returns zeros if SBER or GAZP data is unavailable.
    """
    corr = np.zeros(n)

    # Resolve SBER and GAZP aligned data — current ticker may itself be SBER/GAZP
    sber = None
    gazp = None
    if all_tickers_data is not None:
        sber = all_tickers_data.get("SBER")
        gazp = all_tickers_data.get("GAZP")
    # If current aligned is SBER or GAZP, use it directly
    if sber is None and aligned.get("_ticker") == "SBER":
        sber = aligned
    if gazp is None and aligned.get("_ticker") == "GAZP":
        gazp = aligned

    if sber is None or gazp is None:
        return corr

    sber_close = sber.get("5min_close")
    gazp_close = gazp.get("5min_close")
    if sber_close is None or gazp_close is None:
        return corr

    # Compute 1-bar returns on each ticker's own grid (causal)
    sber_r = _causal_returns(sber_close)
    gazp_r = _causal_returns(gazp_close)

    # Align returns to the base grid by index (assuming both SBER/GAZP share the
    # MOEX trading grid — true when downloaded with same `days` parameter)
    m = min(len(sber_r), len(gazp_r), n)
    if m < window + 1:
        return corr

    sber_r = sber_r[:m]
    gazp_r = gazp_r[:m]

    # Cumulative sums for rolling correlation (causal trailing window)
    c_x = np.cumsum(sber_r)
    c_y = np.cumsum(gazp_r)
    c_xy = np.cumsum(sber_r * gazp_r)
    c_x2 = np.cumsum(sber_r ** 2)
    c_y2 = np.cumsum(gazp_r ** 2)

    # Trailing-window sums at index i (window covers [i-window+1, i])
    end = np.arange(m)
    start = np.maximum(0, end - window + 1)
    # Use prefix-sum diff: sum[i-window+1..i] = c[i] - c[i-window]
    # Handle warm-up by using searchsorted trick: for i < window-1 use full prefix
    for i in range(m):
        if i >= window - 1:
            if i >= window:
                sx = c_x[i] - c_x[i - window]
                sy = c_y[i] - c_y[i - window]
                sxy = c_xy[i] - c_xy[i - window]
                sx2 = c_x2[i] - c_x2[i - window]
                sy2 = c_y2[i] - c_y2[i - window]
            else:
                sx = c_x[i]
                sy = c_y[i]
                sxy = c_xy[i]
                sx2 = c_x2[i]
                sy2 = c_y2[i]
            w_eff = i + 1 if i < window else window
            mean_x = sx / w_eff
            mean_y = sy / w_eff
            cov = sxy / w_eff - mean_x * mean_y
            var_x = sx2 / w_eff - mean_x ** 2
            var_y = sy2 / w_eff - mean_y ** 2
            denom = np.sqrt(var_x * var_y)
            if denom > 1e-12:
                corr[i] = np.clip(cov / denom, -1.0, 1.0)
    return corr


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def compute_features_v4(aligned: Dict[str, np.ndarray],
                       all_tickers_data: Optional[Dict[str, Dict[str, np.ndarray]]] = None
                       ) -> Tuple[np.ndarray, List[str]]:
    """Compute v4 features from aligned multi-timeframe data for ONE ticker.

    Args:
        aligned: dict with keys like "5min_close", "5min_high", "5min_low",
            "5min_open", "5min_volume", "time", "1hour_close", "1day_close".
        all_tickers_data: optional {ticker: aligned_data} for cross-asset features.

    Returns:
        X (np.ndarray shape (n_bars, ~22)), feature_names (list[str])
    """
    close5 = aligned["5min_close"]
    high5 = aligned["5min_high"]
    low5 = aligned["5min_low"]
    vol5 = aligned["5min_volume"].astype(float)
    time5 = aligned["time"]

    n = len(close5)
    features: Dict[str, np.ndarray] = {}

    # ===================== Returns (causal, slicing-based) ====================
    features["ret_1"] = _causal_returns(close5)
    features["ret_5"] = _causal_ret_n(close5, 5)
    features["ret_10"] = _causal_ret_n(close5, 10)
    features["ret_30"] = _causal_ret_n(close5, 30)

    # ===================== SMA ratios (causal cumsum) ========================
    sma5 = _causal_sma(close5, 5)
    sma14 = _causal_sma(close5, 14)
    sma20 = _causal_sma(close5, 20)
    sma50 = _causal_sma(close5, 50)
    features["sma5_sma14"] = sma5 / (sma14 + 1e-10)
    features["sma14_sma20"] = sma14 / (sma20 + 1e-10)
    features["sma20_sma50"] = sma20 / (sma50 + 1e-10)

    # ===================== RSI(14) — drops RSI2 (duplicate) =================
    deltas = np.diff(close5, prepend=close5[0])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = _causal_rolling_mean(gains, 14)
    avg_loss = _causal_rolling_mean(losses, 14)
    rs = avg_gain / (avg_loss + 1e-10)
    features["rsi14"] = 100.0 - 100.0 / (1.0 + rs)

    # ===================== Bollinger Bands (causal) ==========================
    std20 = _causal_rolling_std(close5, 20)
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    features["bb_pct_b"] = (close5 - bb_lower) / (4 * std20 + 1e-10)
    features["bb_width"] = (4 * std20) / (sma20 + 1e-10)

    # ===================== MACD histogram only ==============================
    ema12 = _causal_ema(close5, 12)
    ema26 = _causal_ema(close5, 26)
    macd_line = ema12 - ema26
    macd_signal = _causal_ema(macd_line, 9)
    features["macd_hist"] = macd_line - macd_signal

    # ===================== ATR% (causal) ====================================
    # True Range: max(H-L, |H-prev_close|, |L-prev_close|). prev_close[i] = close5[i-1].
    prev_close = np.empty(n)
    prev_close[0] = close5[0]
    prev_close[1:] = close5[:-1]
    tr = np.maximum(high5 - low5, np.maximum(
        np.abs(high5 - prev_close), np.abs(low5 - prev_close)
    ))
    atr14 = _causal_rolling_mean(tr, 14)
    features["atr_pct"] = atr14 / (close5 + 1e-10)

    # ===================== Stochastic %K (causal) ===========================
    # hh14[i] = max(high[i-13..i]); ll14[i] = min(low[i-13..i])
    hh14 = np.array([
        high5[max(0, i - 13):i + 1].max() if i >= 0 else high5[0]
        for i in range(n)
    ])
    ll14 = np.array([
        low5[max(0, i - 13):i + 1].min() if i >= 0 else low5[0]
        for i in range(n)
    ])
    features["stoch_k"] = (close5 - ll14) / (hh14 - ll14 + 1e-10) * 100.0

    # ===================== Volume ratio (causal) ===========================
    vol_avg20 = _causal_rolling_mean(vol5, 20)
    features["vol_ratio"] = vol5 / (vol_avg20 + 1e-10)

    # ===================== Higher timeframe context (causal) ================
    # These arrays are already forward-filled by align_timeframes() so they
    # never look into the future.
    if "1hour_close" in aligned:
        close1h = aligned["1hour_close"]
        # 1-bar return on the 1h grid, already forward-filled onto the 5min grid
        prev1h = np.empty(len(close1h))
        prev1h[0] = close1h[0]
        prev1h[1:] = close1h[:-1]
        features["1h_ret"] = (close1h - prev1h) / (prev1h + 1e-10)
    else:
        features["1h_ret"] = np.zeros(n)

    if "1day_close" in aligned:
        close1d = aligned["1day_close"]
        prev1d = np.empty(len(close1d))
        prev1d[0] = close1d[0]
        prev1d[1:] = close1d[:-1]
        features["1d_ret"] = (close1d - prev1d) / (prev1d + 1e-10)
    else:
        features["1d_ret"] = np.zeros(n)

    # ===================== Time features (MSK) ==============================
    # time5 is epoch-ms. MSK = UTC+3. Day-of-week: epoch day 0 = Thursday.
    ts_seconds = time5 / 1000.0
    hours_msk = ((ts_seconds // 3600 + 3) % 24).astype(float)
    dow = ((ts_seconds // 86400 + 4) % 7).astype(float)  # 0=Thursday
    features["hour"] = hours_msk / 24.0
    features["day_of_week"] = dow / 7.0

    # ===================== Vol regime: ATR percentile rank ==================
    # vol_regime[i] = fraction of atr_pct[i-99..i] values that are <= atr_pct[i].
    # Strictly causal (only past + current bar). Window=100.
    atr_pct = features["atr_pct"]
    vol_regime = np.full(n, 0.5)
    W = 100
    for i in range(n):
        lo = max(0, i - W + 1)
        window = atr_pct[lo:i + 1]
        # rank: how many are <= current, normalized by window length
        vol_regime[i] = np.searchsorted(np.sort(window), atr_pct[i]) / len(window)
    features["vol_regime"] = vol_regime

    # ===================== Trend strength (ADX / 100) ======================
    # Simplified directional-movement proxy: |P(up) - P(down)| over trailing 14 bars.
    # Already normalized to 0..100 in v1 — divide by 100 for 0..1.
    up_moves = (deltas > 0).astype(float)
    down_moves = (deltas < 0).astype(float)
    adx_raw = np.abs(
        _causal_rolling_mean(up_moves, 14) - _causal_rolling_mean(down_moves, 14)
    ) * 100.0
    features["trend_strength"] = adx_raw / 100.0

    # ===================== Cross-asset (NEW) ================================
    features["market_breadth"] = _compute_market_breadth(
        all_tickers_data, time5, n
    )
    features["sber_gazp_corr"] = _compute_sber_gazp_corr(
        all_tickers_data, aligned, n, window=20
    )

    # ===================== Assemble, sanitize, warm-up =====================
    feature_names = sorted(features.keys())
    X = np.column_stack([features[name] for name in feature_names])
    # Replace NaN/inf with 0, then clip to a sane range to avoid XGBoost explosions
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -10.0, 10.0)
    # Zero out first 50 bars (not enough history for SMA50 / stable vol_regime)
    if n >= 50:
        X[:50] = 0.0
    else:
        X[:] = 0.0

    return X, feature_names


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from ml_data_pipeline import download_multi_timeframe, align_timeframes

    print("=== features_v4 self-test ===")
    data = download_multi_timeframe("SBER", days=30)
    aligned = align_timeframes(data)
    X, names = compute_features_v4(aligned)
    print(f"\nFeatures: X.shape={X.shape} ({len(names)} features)")
    print("\nFeature names:")
    for nm in names:
        print(f"  {nm}")

    # Quick sanity check: any NaN / inf?
    print(f"\nNaN count: {np.isnan(X).sum()}, Inf count: {np.isinf(X).sum()}")
    print(f"Min/max per column (first 5):")
    for i, nm in enumerate(names[:5]):
        col = X[50:, i]  # skip warm-up
        print(f"  {nm:20s}  min={col.min():.4f}  max={col.max():.4f}  mean={col.mean():.4f}")
