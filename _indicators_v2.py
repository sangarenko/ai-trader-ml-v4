#!/usr/bin/env python3
"""Extended indicators v2 — adds all indicators needed for 22 strategies.

Adds to compute_indicators_extended:
  - rsi2 (Connors RSI period 2)
  - zscore (Z-score of price vs SMA20)
  - sma50 (50-period SMA)
  - supertrend_flip (+1 cross up, -1 cross down, 0 none)
  - is_squeeze (BB bandwidth at 125-candle low)
  - atr_upper / atr_lower (SMA20 ± 2*ATR)
  - ha_green / ha_red / ha_close (Heikin-Ashi)
  - dt_upper / dt_lower (Dual Thrust levels)
  - ao_cross (Awesome Oscillator cross)
  - golden_cross / death_cross (SMA50/SMA200)
  - or_high / or_low (Opening Range)
"""
import math
from typing import List, Dict


def compute_indicators_v2(candles: List[Dict], idx: int) -> Dict:
    """All indicators needed for 22 strategies."""
    if idx < 50:
        return None
    
    n = idx + 1
    closes = [c['close'] for c in candles[max(0, idx-199):idx+1]]
    highs = [c['high'] for c in candles[max(0, idx-199):idx+1]]
    lows = [c['low'] for c in candles[max(0, idx-199):idx+1]]
    volumes = [c['volume'] for c in candles[max(0, idx-199):idx+1]]
    
    # === Existing indicators ===
    sma5 = sum(closes[-5:]) / 5
    sma14 = sum(closes[-14:]) / 14
    sma20 = sum(closes[-20:]) / 20
    sma50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else sma20
    
    # RSI(14)
    gains, losses = 0, 0
    for i in range(1, min(15, len(closes))):
        ch = closes[i] - closes[i-1]
        if ch > 0: gains += ch
        else: losses -= ch
    rsi = 100 if losses == 0 else (0 if gains == 0 else 100 - 100 / (1 + gains / losses))
    
    # RSI(2) — Connors
    if len(closes) >= 3:
        g2, l2 = 0, 0
        for i in range(len(closes)-2, len(closes)):
            if i > 0:
                ch = closes[i] - closes[i-1]
                if ch > 0: g2 += ch
                else: l2 -= ch
        rsi2 = 100 if l2 == 0 else (0 if g2 == 0 else 100 - 100 / (1 + g2 / l2))
    else:
        rsi2 = 50
    
    last3 = closes[-3:]
    all_up = last3[0] < last3[1] < last3[2]
    all_down = last3[0] > last3[1] > last3[2]
    
    # Bollinger Bands
    bb_std = math.sqrt(sum((c - sma20) ** 2 for c in closes[-20:]) / 20)
    bb_upper = sma20 + 2 * bb_std
    bb_lower = sma20 - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / sma20 if sma20 > 0 else 0
    
    # Z-score
    zscore = (closes[-1] - sma20) / bb_std if bb_std > 0 else 0
    
    # ATR(20)
    trs = []
    for i in range(1, min(21, len(closes))):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        trs.append(tr)
    atr = sum(trs) / len(trs) if trs else 0
    atr_upper = sma20 + 2 * atr
    atr_lower = sma20 - 2 * atr
    
    # MACD (simplified)
    def ema(vals, period):
        k = 2 / (period + 1)
        e = vals[0]
        for v in vals[1:]: e = v * k + e * (1 - k)
        return e
    macd_line = ema(closes[-12:], 12) - ema(closes[-26:], 26) if len(closes) >= 26 else 0
    macd_signal = macd_line * 0.8
    macd_hist = macd_line - macd_signal
    
    # ADX (simplified)
    up_moves = sum(1 for i in range(-13, 0) if i >= -len(closes)+1 and closes[i] > closes[i-1])
    down_moves = sum(1 for i in range(-13, 0) if i >= -len(closes)+1 and closes[i] < closes[i-1])
    adx = abs(up_moves - down_moves) / 14 * 100
    
    # Donchian
    donchian_upper = max(highs[-20:]) if len(highs) >= 20 else max(highs)
    donchian_lower = min(lows[-20:]) if len(lows) >= 20 else min(lows)
    
    # VWAP
    tp = [(h + l + c) / 3 for h, l, c in zip(highs[-20:], lows[-20:], closes[-20:])]
    vol_sum = sum(volumes[-20:])
    vwap = sum(t * v for t, v in zip(tp, volumes[-20:])) / vol_sum if vol_sum > 0 else closes[-1]
    
    # ROC
    roc = (closes[-1] - closes[-11]) / closes[-11] * 100 if len(closes) >= 11 and closes[-11] > 0 else 0
    
    # Volume ratio
    vol_avg = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else sum(volumes) / len(volumes)
    vol_ratio = volumes[-1] / vol_avg if vol_avg > 0 else 1
    
    # Stochastic
    if len(highs) >= 14:
        hh = max(highs[-14:]); ll = min(lows[-14:])
        stoch_k = (closes[-1] - ll) / (hh - ll) * 100 if hh > ll else 50
    else:
        stoch_k = 50
    
    # === NEW indicators ===
    
    # Supertrend flip (simplified: price crosses ATR-based line)
    mid = (highs[-1] + lows[-1]) / 2
    st_line = mid - 3 * atr
    prev_mid = (highs[-2] + lows[-2]) / 2 if len(highs) >= 2 else mid
    prev_st_line = prev_mid - 3 * atr
    supertrend_flip = 0
    if len(closes) >= 2:
        if closes[-2] < prev_st_line and closes[-1] > st_line: supertrend_flip = 1
        elif closes[-2] > prev_st_line and closes[-1] < st_line: supertrend_flip = -1
    
    # BB squeeze (bandwidth at 125-candle low)
    is_squeeze = False
    if len(closes) >= 125:
        min_bw = float('inf')
        for j in range(idx - 124, idx + 1):
            c20 = closes[j-19:j+1] if j >= 19 else closes[:j+1]
            if len(c20) >= 20:
                s20 = sum(c20) / len(c20)
                sd = math.sqrt(sum((c - s20) ** 2 for c in c20) / len(c20))
                bw = (s20 + 2*sd - (s20 - 2*sd)) / s20 if s20 > 0 else 0
                if bw < min_bw: min_bw = bw
        is_squeeze = bb_width <= min_bw * 1.1 if min_bw != float('inf') else False
    
    # Heikin-Ashi
    ha_close = (candles[idx].open + highs[-1] + lows[-1] + closes[-1]) / 4
    ha_open = (candles[idx-1].open + candles[idx-1].close) / 2 if idx >= 1 else ha_close
    ha_green = ha_close > ha_open
    ha_red = ha_close < ha_open
    
    # Dual Thrust (5-candle lookback)
    if len(closes) >= 5:
        hh5 = max(highs[-5:]); lc5 = min(closes[-5:])
        hc5 = max(closes[-5:]); ll5 = min(lows[-5:])
        dt_range = max(hh5 - lc5, hc5 - ll5)
        dt_upper = candles[idx].open + 0.5 * dt_range
        dt_lower = candles[idx].open - 0.5 * dt_range
    else:
        dt_upper = dt_lower = closes[-1]
    
    # Awesome Oscillator cross
    if len(closes) >= 34:
        medians = [(h + l) / 2 for h, l in zip(highs[-34:], lows[-34:])]
        ao = sum(medians[-5:]) / 5 - sum(medians) / 34
        prev_meds = [(h + l) / 2 for h, l in zip(highs[-35:-1], lows[-35:-1])]
        prev_ao = sum(prev_meds[-5:]) / 5 - sum(prev_meds) / 34 if len(prev_meds) >= 34 else 0
        ao_cross = 0
        if prev_ao < 0 and ao > 0: ao_cross = 1
        elif prev_ao > 0 and ao < 0: ao_cross = -1
    else:
        ao_cross = 0
    
    # Golden / Death cross (SMA50 / SMA200)
    golden_cross = death_cross = False
    if len(closes) >= 200:
        sma200 = sum(closes[-200:]) / 200
        prev_sma50 = sum(closes[-51:-1]) / 50
        prev_sma200 = sum(closes[-201:-1]) / 200
        golden_cross = prev_sma50 <= prev_sma200 and sma50 > sma200
        death_cross = prev_sma50 >= prev_sma200 and sma50 < sma200
    
    # Opening Range (first candle of day)
    or_high = or_low = closes[-1]
    if idx >= 1:
        import datetime
        cur_day = datetime.datetime.fromtimestamp(candles[idx]['time'] / 1000 if isinstance(candles[idx].get('time'), (int, float)) else 0).strftime('%Y-%m-%d')
        day_start = idx
        while day_start > 0:
            prev_day = datetime.datetime.fromtimestamp(candles[day_start-1].get('time', 0) / 1000 if isinstance(candles[day_start-1].get('time'), (int, float)) else 0).strftime('%Y-%m-%d')
            if prev_day != cur_day: break
            day_start -= 1
        if idx > day_start:
            or_high = candles[day_start]['high']
            or_low = candles[day_start]['low']
    
    return {
        # existing
        'sma5': sma5, 'sma14': sma14, 'sma20': sma20, 'sma50': sma50,
        'rsi': rsi, 'rsi2': rsi2, 'cur': closes[-1],
        'allUp': all_up, 'allDown': all_down,
        'bb_upper': bb_upper, 'bb_lower': bb_lower, 'bb_width': bb_width,
        'zscore': zscore,
        'macd_line': macd_line, 'macd_signal': macd_signal, 'macd_hist': macd_hist,
        'atr': atr, 'atr_upper': atr_upper, 'atr_lower': atr_lower,
        'adx': adx,
        'donchian_upper': donchian_upper, 'donchian_lower': donchian_lower,
        'vwap': vwap, 'roc': roc, 'vol_ratio': vol_ratio,
        'stoch_k': stoch_k,
        # new
        'supertrend_flip': supertrend_flip,
        'is_squeeze': is_squeeze,
        'ha_green': ha_green, 'ha_red': ha_red, 'ha_close': ha_close,
        'dt_upper': dt_upper, 'dt_lower': dt_lower,
        'ao_cross': ao_cross,
        'golden_cross': golden_cross, 'death_cross': death_cross,
        'or_high': or_high, 'or_low': or_low,
    }
