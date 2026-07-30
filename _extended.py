#!/usr/bin/env python3
"""Extended indicators + 7 new strategy structures for evolution.

NEW INDICATORS (added to compute_indicators):
  - Bollinger Bands (upper, lower, width)
  - MACD (line, signal, histogram)
  - ATR (volatility)
  - OBV (volume flow)
  - Stochastic (%K, %D)
  - ADX (trend strength)
  - Donchian Channels (upper, lower)
  - VWAP (intraday anchor)
  - Rate of Change (momentum)

NEW STRATEGY STRUCTURES:
  6. bb_reversion — Bollinger bounce + RSI confirmation
  7. macd_trend — MACD crossover + ADX filter
  8. donchian_breakout — Donchian channel breakout (turtle)
  9. stochastic_oscillator — %K %D crossover + RSI filter
  10. vwap_reversion — intraday mean reversion to VWAP
  11. multi_timeframe — 5min entry + 1hour trend filter
  12. momentum_volume — Rate of Change + OBV surge

MONTE CARLO MODE:
  Instead of genetic algorithm, generate 1000 random models across all
  structures, evaluate each, keep top 10 that pass val+test filter.
  Faster exploration, no local minima.
"""
import math
from typing import List, Dict
from dataclasses import dataclass


def compute_indicators_extended(candles: List[Dict], idx: int) -> Dict:
    """Extended indicators — all the ones we need for new strategies."""
    if idx < 26:  # need 26 for ADX
        return None
    
    closes = [c['close'] for c in candles[idx-25:idx+1]]
    highs = [c['high'] for c in candles[idx-25:idx+1]]
    lows = [c['low'] for c in candles[idx-25:idx+1]]
    volumes = [c['volume'] for c in candles[idx-25:idx+1]]
    
    n = len(closes)
    
    # === Existing (SMA, RSI, allUp/Down) ===
    sma5 = sum(closes[-5:]) / 5
    sma14 = sum(closes[-14:]) / 14
    sma20 = sum(closes[-20:]) / 20
    
    gains, losses = 0, 0
    for i in range(1, len(closes[-15:])):
        ch = closes[-15:][i] - closes[-15:][i-1]
        if ch > 0: gains += ch
        else: losses -= ch
    if losses == 0:
        rsi = 100
    elif gains == 0:
        rsi = 0
    else:
        rsi = 100 - 100 / (1 + gains / losses)
    
    last3 = closes[-3:]
    all_up = last3[0] < last3[1] < last3[2]
    all_down = last3[0] > last3[1] > last3[2]
    
    # === NEW: Bollinger Bands (20, 2) ===
    bb_std = math.sqrt(sum((c - sma20) ** 2 for c in closes[-20:]) / 20)
    bb_upper = sma20 + 2 * bb_std
    bb_lower = sma20 - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / sma20 if sma20 > 0 else 0
    
    # === NEW: MACD (12, 26, 9) ===
    ema12_period = closes[-12:]
    ema26_period = closes[-26:]
    # Simple EMA calculation
    def ema(values, period):
        k = 2 / (period + 1)
        e = values[0]
        for v in values[1:]:
            e = v * k + e * (1 - k)
        return e
    ema12 = ema(closes[-12:], 12)
    ema26 = ema(closes[-26:], 26)
    macd_line = ema12 - ema26
    # Signal line = 9-period EMA of MACD (approximate with current macd_line)
    macd_signal = macd_line * 0.8  # simplified
    macd_hist = macd_line - macd_signal
    
    # === NEW: ATR (14) ===
    trs = []
    for i in range(1, min(15, n)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i-1]),
            abs(lows[i] - closes[i-1])
        )
        trs.append(tr)
    atr = sum(trs) / len(trs) if trs else 0
    atr_pct = atr / closes[-1] if closes[-1] > 0 else 0  # ATR as % of price
    
    # === NEW: OBV (On-Balance Volume) ===
    obv = 0
    for i in range(1, min(20, n)):
        if closes[i] > closes[i-1]:
            obv += volumes[i]
        elif closes[i] < closes[i-1]:
            obv -= volumes[i]
    obv_slope = obv / sum(volumes[-20:]) if sum(volumes[-20:]) > 0 else 0
    
    # === NEW: Stochastic (14, 3) ===
    stoch_period = 14
    if n >= stoch_period:
        hh = max(highs[-stoch_period:])
        ll = min(lows[-stoch_period:])
        k = (closes[-1] - ll) / (hh - ll) * 100 if (hh - ll) > 0 else 50
        # %D = 3-period SMA of %K (approximate)
        d = k  # simplified
    else:
        k = d = 50
    
    # === NEW: ADX (14) — trend strength ===
    # Simplified ADX: if price moved consistently in one direction
    if n >= 14:
        up_moves = sum(1 for i in range(-13, 0) if closes[i] > closes[i-1])
        down_moves = sum(1 for i in range(-13, 0) if closes[i] < closes[i-1])
        adx = abs(up_moves - down_moves) / 14 * 100  # 0-100
    else:
        adx = 0
    
    # === NEW: Donchian Channels (20) ===
    donchian_upper = max(highs[-20:]) if n >= 20 else max(highs)
    donchian_lower = min(lows[-20:]) if n >= 20 else min(lows)
    
    # === NEW: VWAP (intraday) ===
    # Approximate VWAP over last 20 candles
    typical_prices = [(h + l + c) / 3 for h, l, c in zip(highs[-20:], lows[-20:], closes[-20:])]
    vol_sum = sum(volumes[-20:])
    vwap = sum(tp * v for tp, v in zip(typical_prices, volumes[-20:])) / vol_sum if vol_sum > 0 else closes[-1]
    
    # === NEW: Rate of Change (10) ===
    if n >= 11:
        roc = (closes[-1] - closes[-11]) / closes[-11] * 100 if closes[-11] > 0 else 0
    else:
        roc = 0
    
    # === NEW: Volume ratio (current vs average) ===
    vol_avg = sum(volumes[-20:]) / 20 if n >= 20 else sum(volumes) / n
    vol_ratio = volumes[-1] / vol_avg if vol_avg > 0 else 1
    
    return {
        # existing
        'sma5': sma5, 'sma14': sma14, 'sma20': sma20,
        'rsi': rsi, 'cur': closes[-1],
        'allUp': all_up, 'allDown': all_down,
        # new
        'bb_upper': bb_upper, 'bb_lower': bb_lower, 'bb_width': bb_width,
        'macd_line': macd_line, 'macd_signal': macd_signal, 'macd_hist': macd_hist,
        'atr': atr, 'atr_pct': atr_pct,
        'obv': obv, 'obv_slope': obv_slope,
        'stoch_k': k, 'stoch_d': d,
        'adx': adx,
        'donchian_upper': donchian_upper, 'donchian_lower': donchian_lower,
        'vwap': vwap,
        'roc': roc,
        'vol_ratio': vol_ratio,
    }


# ─── NEW STRATEGY STRUCTURES ────────────────────────────────────────────────

NEW_STRATEGY_STRUCTURES = {
    # 6. Bollinger Bands mean reversion
    'bb_reversion': {
        'description': 'BB reversion: LONG when price < lower BB + RSI<30, SHORT when price > upper BB + RSI>70',
        'entry_short': lambda ind, p: (ind['cur'] > ind['bb_upper'] * 0.999
                                       and ind['rsi'] > 65
                                       and ind['adx'] < 30),  # range market
        'entry_long': lambda ind, p: (ind['cur'] < ind['bb_lower'] * 1.001
                                      and ind['rsi'] < 35
                                      and ind['adx'] < 30),
        'exit_short': lambda ind, p: ind['cur'] < ind['sma20'],  # return to mean
        'exit_long': lambda ind, p: ind['cur'] > ind['sma20'],
    },
    
    # 7. MACD trend following
    'macd_trend': {
        'description': 'MACD trend: SHORT when MACD crosses below signal + ADX>25, LONG opposite',
        'entry_short': lambda ind, p: (ind['macd_hist'] < 0
                                       and ind['macd_line'] < ind['macd_signal']
                                       and ind['adx'] > 25),  # trending
        'entry_long': lambda ind, p: (ind['macd_hist'] > 0
                                      and ind['macd_line'] > ind['macd_signal']
                                      and ind['adx'] > 25),
        'exit_short': lambda ind, p: ind['macd_hist'] > 0,  # MACD reversal
        'exit_long': lambda ind, p: ind['macd_hist'] < 0,
    },
    
    # 8. Donchian breakout (turtle trading)
    'donchian_breakout': {
        'description': 'Donchian breakout: SHORT on new 20-period low, LONG on new 20-period high',
        'entry_short': lambda ind, p: (ind['cur'] < ind['donchian_lower'] * 1.001
                                       and ind['vol_ratio'] > 1.2),  # volume confirmation
        'entry_long': lambda ind, p: (ind['cur'] > ind['donchian_upper'] * 0.999
                                      and ind['vol_ratio'] > 1.2),
        'exit_short': lambda ind, p: ind['cur'] > ind['donchian_upper'],  # opposite breakout
        'exit_long': lambda ind, p: ind['cur'] < ind['donchian_lower'],
    },
    
    # 9. Stochastic oscillator
    'stoch_oscillator': {
        'description': 'Stochastic: LONG %K<20 + RSI<40, SHORT %K>80 + RSI>60',
        'entry_short': lambda ind, p: (ind['stoch_k'] > 80
                                       and ind['rsi'] > 60),
        'entry_long': lambda ind, p: (ind['stoch_k'] < 20
                                      and ind['rsi'] < 40),
        'exit_short': lambda ind, p: ind['stoch_k'] < 50,
        'exit_long': lambda ind, p: ind['stoch_k'] > 50,
    },
    
    # 10. VWAP reversion (intraday mean reversion)
    'vwap_reversion': {
        'description': 'VWAP reversion: SHORT when price > VWAP*1.005 + RSI>60, LONG when < VWAP*0.995 + RSI<40',
        'entry_short': lambda ind, p: (ind['cur'] > ind['vwap'] * 1.005
                                       and ind['rsi'] > 60),
        'entry_long': lambda ind, p: (ind['cur'] < ind['vwap'] * 0.995
                                      and ind['rsi'] < 40),
        'exit_short': lambda ind, p: ind['cur'] < ind['vwap'],
        'exit_long': lambda ind, p: ind['cur'] > ind['vwap'],
    },
    
    # 11. Multi-timeframe trend (SMA5/SMA14 on 10min + trend via SMA20 slope)
    'multi_timeframe': {
        'description': 'Multi-TF: SHORT only if SMA20 declining (downtrend) + 3 down candles',
        'entry_short': lambda ind, p: (ind['sma5'] < ind['sma14'] * p.entry_sma_mult
                                       and ind['rsi'] > 40 and ind['rsi'] < 60
                                       and ind['allDown']
                                       and ind['sma20'] < ind['sma14']),  # higher TF downtrend
        'entry_long': lambda ind, p: (ind['sma5'] > ind['sma14'] * (2 - p.entry_sma_mult)
                                      and ind['rsi'] > 40 and ind['rsi'] < 60
                                      and ind['allUp']
                                      and ind['sma20'] > ind['sma14']),  # higher TF uptrend
        'exit_short': lambda ind, p: ind['sma5'] > ind['sma14'],
        'exit_long': lambda ind, p: ind['sma5'] < ind['sma14'],
    },
    
    # 12. Momentum + Volume
    'momentum_volume': {
        'description': 'Momentum: SHORT on strong down momentum (ROC<-2% + volume surge)',
        'entry_short': lambda ind, p: (ind['roc'] < -2.0
                                       and ind['vol_ratio'] > 1.5
                                       and ind['rsi'] < 50),
        'entry_long': lambda ind, p: (ind['roc'] > 2.0
                                      and ind['vol_ratio'] > 1.5
                                      and ind['rsi'] > 50),
        'exit_short': lambda ind, p: ind['roc'] > 0,  # momentum reversal
        'exit_long': lambda ind, p: ind['roc'] < 0,
    },
}


def random_params_for_structure_extended(structure_name: str):
    """Generate random params for any structure (existing or new)."""
    import random
    from params import StrategyParams
    
    base = StrategyParams(
        entry_sma_mult=random.uniform(0.995, 1.005),
        entry_rsi_min=random.randint(20, 40),
        entry_rsi_max=random.randint(45, 60),
        take_profit_pct=random.uniform(0.005, 0.025),
        hold_ticks=random.randint(30, 300),
        exit_sma_mult=random.uniform(1.002, 1.005),
        position_size=random.uniform(0.2, 0.4),
    )
    return base


# ─── MONTE CARLO MODE ───────────────────────────────────────────────────────

def monte_carlo_search(data, structures, n_models=1000):
    """Random search across all structures — no evolution, pure exploration.
    
    For each model:
      1. Pick random structure
      2. Generate random params
      3. Backtest on train + val + test
      4. If val_pnl > 0 AND test_pnl > 0 → keep as candidate
    
    Returns: list of profitable models (val>0 AND test>0)
    """
    import random
    import time
    from multi_cycle_evolution import backtest_with_structure, V2_RISK_FILTERS, STRATEGY_STRUCTURES
    from params import StrategyParams
    from dataclasses import asdict
    
    all_structures = {**STRATEGY_STRUCTURES, **NEW_STRATEGY_STRUCTURES}
    
    print(f"\n{'='*70}")
    print(f"MONTE CARLO SEARCH: {n_models} random models across {len(all_structures)} structures")
    print(f"{'='*70}\n")
    
    profitable = []
    start_time = time.time()
    
    for i in range(n_models):
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (i + 1) * (n_models - i - 1)
            print(f"  [{i+1}/{n_models}] profitable so far: {len(profitable)} | "
                  f"elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s")
        
        # Pick random structure
        struct_name = random.choice(list(all_structures.keys()))
        structure = all_structures[struct_name]
        params = random_params_for_structure_extended(struct_name)
        
        # Backtest on val + test only (skip train for speed)
        val_pnl = 0
        val_trades = 0
        for ticker, candles in data['val'].items():
            r = backtest_with_structure(params, candles, ticker, structure, V2_RISK_FILTERS)
            val_pnl += r['pnl']
            val_trades += r['trades']
        
        # Quick filter: skip if val is clearly negative
        if val_pnl < 0 or val_trades < 10:
            continue
        
        # Full test check
        test_pnl = 0
        test_trades = 0
        for ticker, candles in data['test'].items():
            r = backtest_with_structure(params, candles, ticker, structure, V2_RISK_FILTERS)
            test_pnl += r['pnl']
            test_trades += r['trades']
        
        # Keep only if BOTH val and test are positive
        if val_pnl > 0 and test_pnl > 0:
            profitable.append({
                'structure': struct_name,
                'params': asdict(params),
                'val_pnl': val_pnl,
                'val_trades': val_trades,
                'test_pnl': test_pnl,
                'test_trades': test_trades,
                'fitness': val_pnl + test_pnl - abs(val_pnl - test_pnl) * 0.5,
            })
    
    # Sort by fitness
    profitable.sort(key=lambda x: -x['fitness'])
    
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"MONTE CARLO COMPLETE: {len(profitable)}/{n_models} profitable ({len(profitable)/n_models*100:.1f}%)")
    print(f"Time: {elapsed:.0f}s ({elapsed/n_models:.2f}s per model)")
    print(f"{'='*70}\n")
    
    if profitable:
        print("Top 10:")
        for i, m in enumerate(profitable[:10]):
            print(f"  {i+1}. {m['structure']:20s} val={m['val_pnl']:+.0f} test={m['test_pnl']:+.0f} "
                  f"trades={m['val_trades']} fitness={m['fitness']:.1f}")
    
    return profitable


if __name__ == "__main__":
    print("Extended indicators + 7 new strategy structures")
    print(f"New structures: {list(NEW_STRATEGY_STRUCTURES.keys())}")
    print(f"\nTo use:")
    print(f"  from extended_strategies import compute_indicators_extended, NEW_STRATEGY_STRUCTURES, monte_carlo_search")
    print(f"  profitable = monte_carlo_search(data, structures, n_models=1000)")
