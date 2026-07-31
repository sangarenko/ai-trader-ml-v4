#!/usr/bin/env python3
"""Strategy collection — all strategies to test, with descriptions.

Categories:
  A. WISEPLAT (from GitHub, проверенные сообществом)
  B. Evolution winners (from our Monte Carlo)
  C. Classic (from trading literature)
  D. Experimental (наши идеи)

Each strategy has:
  - structure: entry/exit lambdas (used by backtest_with_structure)
  - description: human-readable
  - source: where it came from
  - params_hint: recommended param ranges
"""
import math
from typing import Dict


# ─── A. WISEPLAT strategies (from GitHub) ───────────────────────────────────

WISEPLAT_STRATEGIES = {
    'wiseplat_triple_sma': {
        'description': 'WISEPLAT Strategy 04 (177% backtest): LONG on SMA9 crossover SMA30 + SMA60<SMA30 (trend reversal up)',
        'source': 'https://github.com/WISEPLAT/backtrader_moexalgo/blob/master/StrategyExamplesMoexAlgo/04%20-%20Offline%20Backtest%20Indicators.py',
        'params_hint': 'Uses fixed periods (9/30/60), only position_size configurable',
        # LONG only: SMA9 crosses above SMA30, AND SMA60 < SMA30 (bottom reversal)
        # Exit: RSI < 30 (oversold — exit on fear)
        # NOTE: original uses 20-period RSI, we use 14
        'entry_short': lambda ind, p: False,  # long-only
        'entry_long': lambda ind, p: (
            # Need SMA30, SMA60 — we have sma14, sma20. Approximate:
            # SMA9 crossover SMA20 from below + SMA20 < SMA14 (bottom turn)
            # Actually we need more SMAs. Use what we have:
            ind['sma5'] > ind['sma14'] * 1.001  # SMA5 crossed above SMA14
            and ind['sma20'] < ind['sma14']     # SMA20 below SMA14 (downtrend, reversal)
            and ind['rsi'] > 30 and ind['rsi'] < 55  # not overbought
        ),
        'exit_long': lambda ind, p: ind['rsi'] < 30,  # exit on oversold (RSI<30)
        'exit_short': lambda ind, p: False,
    },
    
    'wiseplat_rsi_sma': {
        'description': 'WISEPLAT Strategy 01: RSI + SMA crossover, multi-ticker',
        'source': 'https://github.com/WISEPLAT/Hackathon-MOEX-How-To-Guide',
        'params_hint': 'Standard RSI 14, SMA 8/16',
        'entry_short': lambda ind, p: (
            ind['sma5'] < ind['sma14'] * 0.999  # SMA fast below slow
            and ind['rsi'] > 60  # overbought
        ),
        'entry_long': lambda ind, p: (
            ind['sma5'] > ind['sma14'] * 1.001  # SMA fast above slow
            and ind['rsi'] < 40  # oversold
        ),
        'exit_short': lambda ind, p: ind['sma5'] > ind['sma14'],  # SMA cross back
        'exit_long': lambda ind, p: ind['sma5'] < ind['sma14'],
    },
}


# ─── B. Evolution winners (from our Monte Carlo) ────────────────────────────

EVOLUTION_WINNERS = {
    'multi_timeframe': {
        'description': 'Monte Carlo WINNER: SHORT only if SMA20 declining (HT downtrend) + 3 down candles + RSI 30-55',
        'source': 'Our Monte Carlo batch1+2, top-5 models (val +220-350, test +180-379)',
        'params_hint': 'entry_sma_mult: 0.995-1.005, entry_rsi_min: 20-35, entry_rsi_max: 45-59, hold_ticks: 38-275',
        'entry_short': lambda ind, p: (
            ind['sma5'] < ind['sma14'] * p.entry_sma_mult
            and p.entry_rsi_min <= ind['rsi'] <= p.entry_rsi_max
            and ind['allDown']
            and ind['sma20'] < ind['sma14']  # HT downtrend
        ),
        'entry_long': lambda ind, p: (
            ind['sma5'] > ind['sma14'] * (2 - p.entry_sma_mult)
            and 40 <= ind['rsi'] <= 60
            and ind['allUp']
            and ind['sma20'] > ind['sma14']  # HT uptrend
        ),
        'exit_short': lambda ind, p: ind['sma5'] > ind['sma14'],
        'exit_long': lambda ind, p: ind['sma5'] < ind['sma14'],
    },
}


# ─── C. Classic strategies (from trading literature) ────────────────────────

CLASSIC_STRATEGIES = {
    'turtle_donchian': {
        'description': 'Turtle Trading: LONG on 20-day high breakout, SHORT on 20-day low, exit on 10-day opposite',
        'source': 'Richard Dennis turtles (1980s), original Donchian channel breakout',
        'params_hint': 'Donchian period 20, exit period 10',
        'entry_short': lambda ind, p: (
            ind['cur'] < ind['donchian_lower'] * 1.001  # new low
            and ind['vol_ratio'] > 1.0  # volume confirmation
        ),
        'entry_long': lambda ind, p: (
            ind['cur'] > ind['donchian_upper'] * 0.999  # new high
            and ind['vol_ratio'] > 1.0
        ),
        'exit_short': lambda ind, p: ind['cur'] > ind['donchian_upper'],  # opposite breakout
        'exit_long': lambda ind, p: ind['cur'] < ind['donchian_lower'],
    },
    
    'rsi_extremes': {
        'description': 'RSI extremes: LONG RSI<25 (oversold), SHORT RSI>75 (overbought), exit at RSI 50',
        'source': 'Classic mean-reversion, Welles Wilder RSI (1978)',
        'params_hint': 'RSI period 14, entry thresholds 25/75, exit at 50',
        'entry_short': lambda ind, p: ind['rsi'] > 75,
        'entry_long': lambda ind, p: ind['rsi'] < 25,
        'exit_short': lambda ind, p: ind['rsi'] < 50,
        'exit_long': lambda ind, p: ind['rsi'] > 50,
    },
    
    'bollinger_bounce': {
        'description': 'Bollinger bounce: LONG price<lower BB + RSI<30, SHORT price>upper BB + RSI>70',
        'source': 'John Bollinger (1980s), classic mean-reversion',
        'params_hint': 'BB(20, 2), RSI 14',
        'entry_short': lambda ind, p: (
            ind['cur'] > ind['bb_upper'] * 0.999
            and ind['rsi'] > 70
        ),
        'entry_long': lambda ind, p: (
            ind['cur'] < ind['bb_lower'] * 1.001
            and ind['rsi'] < 30
        ),
        'exit_short': lambda ind, p: ind['cur'] < ind['sma20'],
        'exit_long': lambda ind, p: ind['cur'] > ind['sma20'],
    },
    
    'macd_trend': {
        'description': 'MACD trend follow: LONG MACD>signal + ADX>25, SHORT opposite',
        'source': 'Gerald Appel MACD (1979), trend-following classic',
        'params_hint': 'MACD(12,26,9), ADX threshold 25',
        'entry_short': lambda ind, p: (
            ind['macd_hist'] < 0
            and ind['macd_line'] < ind['macd_signal']
            and ind['adx'] > 25
        ),
        'entry_long': lambda ind, p: (
            ind['macd_hist'] > 0
            and ind['macd_line'] > ind['macd_signal']
            and ind['adx'] > 25
        ),
        'exit_short': lambda ind, p: ind['macd_hist'] > 0,
        'exit_long': lambda ind, p: ind['macd_hist'] < 0,
    },
    
    'vwap_reversion': {
        'description': 'VWAP reversion: SHORT price>VWAP*1.005+RSI>60, LONG price<VWAP*0.995+RSI<40',
        'source': 'Institutional VWAP strategies, intraday mean-reversion',
        'params_hint': 'VWAP over 20 candles, RSI 14',
        'entry_short': lambda ind, p: (
            ind['cur'] > ind['vwap'] * 1.005
            and ind['rsi'] > 60
        ),
        'entry_long': lambda ind, p: (
            ind['cur'] < ind['vwap'] * 0.995
            and ind['rsi'] < 40
        ),
        'exit_short': lambda ind, p: ind['cur'] < ind['vwap'],
        'exit_long': lambda ind, p: ind['cur'] > ind['vwap'],
    },
}


# ─── D. Experimental strategies (наши идеи) ─────────────────────────────────

EXPERIMENTAL_STRATEGIES = {
    'momentum_volume': {
        'description': 'Momentum + Volume: SHORT on strong down momentum (ROC<-2% + volume surge + RSI<50)',
        'source': 'Our idea: momentum + volume confirmation',
        'params_hint': 'ROC threshold -2%, volume ratio >1.5',
        'entry_short': lambda ind, p: (
            ind['roc'] < -2.0
            and ind['vol_ratio'] > 1.5
            and ind['rsi'] < 50
        ),
        'entry_long': lambda ind, p: (
            ind['roc'] > 2.0
            and ind['vol_ratio'] > 1.5
            and ind['rsi'] > 50
        ),
        'exit_short': lambda ind, p: ind['roc'] > 0,
        'exit_long': lambda ind, p: ind['roc'] < 0,
    },
    
    'stoch_oscillator': {
        'description': 'Stochastic: LONG %K<20 + RSI<40, SHORT %K>80 + RSI>60',
        'source': 'George Lane stochastic oscillator (1950s)',
        'params_hint': 'Stoch(14,3), RSI 14',
        'entry_short': lambda ind, p: (
            ind['stoch_k'] > 80
            and ind['rsi'] > 60
        ),
        'entry_long': lambda ind, p: (
            ind['stoch_k'] < 20
            and ind['rsi'] < 40
        ),
        'exit_short': lambda ind, p: ind['stoch_k'] < 50,
        'exit_long': lambda ind, p: ind['stoch_k'] > 50,
    },
    
    'multi_timeframe': {  # also in evolution winners
        'description': 'Multi-TF: SHORT only if SMA20 declining + 3 down candles',
        'source': 'Our Monte Carlo winner',
        'entry_short': lambda ind, p: (
            ind['sma5'] < ind['sma14'] * p.entry_sma_mult
            and p.entry_rsi_min <= ind['rsi'] <= p.entry_rsi_max
            and ind['allDown']
            and ind['sma20'] < ind['sma14']
        ),
        'entry_long': lambda ind, p: (
            ind['sma5'] > ind['sma14'] * (2 - p.entry_sma_mult)
            and 40 <= ind['rsi'] <= 60
            and ind['allUp']
            and ind['sma20'] > ind['sma14']
        ),
        'exit_short': lambda ind, p: ind['sma5'] > ind['sma14'],
        'exit_long': lambda ind, p: ind['sma5'] < ind['sma14'],
    },
    
    'bb_reversion_adx': {
        'description': 'BB reversion + ADX filter: only trade in range markets (ADX<30)',
        'source': 'Our idea: BB + regime filter',
        'entry_short': lambda ind, p: (
            ind['cur'] > ind['bb_upper'] * 0.999
            and ind['rsi'] > 65
            and ind['adx'] < 30  # range market
        ),
        'entry_long': lambda ind, p: (
            ind['cur'] < ind['bb_lower'] * 1.001
            and ind['rsi'] < 35
            and ind['adx'] < 30
        ),
        'exit_short': lambda ind, p: ind['cur'] < ind['sma20'],
        'exit_long': lambda ind, p: ind['cur'] > ind['sma20'],
    },
}


# ─── All strategies combined ────────────────────────────────────────────────

ALL_STRATEGIES = {
    **WISEPLAT_STRATEGIES,
    **EVOLUTION_WINNERS,
    **CLASSIC_STRATEGIES,
    **EXPERIMENTAL_STRATEGIES,
}


def list_strategies():
    """Print all strategies with descriptions."""
    categories = [
        ('A. WISEPLAT (GitHub)', WISEPLAT_STRATEGIES),
        ('B. Evolution Winners', EVOLUTION_WINNERS),
        ('C. Classic', CLASSIC_STRATEGIES),
        ('D. Experimental', EXPERIMENTAL_STRATEGIES),
    ]
    
    for cat_name, strategies in categories:
        print(f"\n{'='*70}")
        print(f"  {cat_name}")
        print(f"{'='*70}")
        for name, s in strategies.items():
            print(f"\n  {name}:")
            print(f"    {s['description']}")
            print(f"    source: {s.get('source', '?')}")
            print(f"    params: {s.get('params_hint', '?')}")


if __name__ == "__main__":
    list_strategies()
    print(f"\n\nTotal strategies: {len(ALL_STRATEGIES)}")
