#!/usr/bin/env python3
"""All 22 strategies for backtest — Python implementations.

Each strategy has:
  - entry_short / entry_long / exit_short / exit_long: lambda(ind, params) -> bool
  - random_params(): generate random params for this strategy
  - description: human-readable

This file is imported by monte_carlo_runner.py to test all strategies.
"""
import random
import math
from params import StrategyParams


# ─── Existing strategies (from multi_cycle_evolution + extended) ────────────

EXISTING = {
    'v2_short': {
        'description': 'V2: SHORT on 3 down candles + RSI 30-55',
        'entry_short': lambda i, p: (i['sma5'] < i['sma14'] * p.entry_sma_mult and p.entry_rsi_min <= i['rsi'] <= p.entry_rsi_max and i['allDown']),
        'entry_long': lambda i, p: (i['sma5'] > i['sma14'] * (2 - p.entry_sma_mult) and 25 <= i['rsi'] <= 40 and i['allUp']),
        'exit_short': lambda i, p: (i['sma5'] > i['sma14'] * p.exit_sma_mult and i['rsi'] > 65),
        'exit_long': lambda i, p: (i['sma5'] < i['sma14'] * (2 - p.exit_sma_mult) and i['rsi'] < 35),
    },
    'multi_timeframe': {
        'description': 'Multi-TF: SHORT only if SMA20 declining + 3 down candles',
        'entry_short': lambda i, p: (i['sma5'] < i['sma14'] * p.entry_sma_mult and p.entry_rsi_min <= i['rsi'] <= p.entry_rsi_max and i['allDown'] and i.get('sma20', 0) < i['sma14']),
        'entry_long': lambda i, p: (i['sma5'] > i['sma14'] * (2 - p.entry_sma_mult) and 40 <= i['rsi'] <= 60 and i['allUp'] and i.get('sma20', 0) > i['sma14']),
        'exit_short': lambda i, p: i['sma5'] > i['sma14'],
        'exit_long': lambda i, p: i['sma5'] < i['sma14'],
    },
    'v2_inverted': {
        'description': 'V2-inverted: LONG on 3 down candles (buy the dip)',
        'entry_short': lambda i, p: (i['sma5'] > i['sma14'] * (2 - p.entry_sma_mult) and 60 <= i['rsi'] <= 80 and i['allUp']),
        'entry_long': lambda i, p: (i['sma5'] < i['sma14'] * p.entry_sma_mult and 20 <= i['rsi'] <= 40 and i['allDown']),
        'exit_short': lambda i, p: i['rsi'] < 40,
        'exit_long': lambda i, p: i['rsi'] > 60,
    },
    'mean_reversion': {
        'description': 'Mean-reversion: LONG RSI<25, SHORT RSI>75',
        'entry_short': lambda i, p: i['rsi'] > 75 - (55 - p.entry_rsi_max),
        'entry_long': lambda i, p: i['rsi'] < 25 + (p.entry_rsi_min - 30),
        'exit_short': lambda i, p: i['rsi'] < 50,
        'exit_long': lambda i, p: i['rsi'] > 50,
    },
    'trend_follow': {
        'description': 'Trend-follow: SHORT on 3 up (breakout), LONG on 3 down',
        'entry_short': lambda i, p: (i['sma5'] > i['sma14'] * (1 + (1 - p.entry_sma_mult)) and i['rsi'] > 60 and i['allUp']),
        'entry_long': lambda i, p: (i['sma5'] < i['sma14'] * p.entry_sma_mult and i['rsi'] < 40 and i['allDown']),
        'exit_short': lambda i, p: i['sma5'] < i['sma14'],
        'exit_long': lambda i, p: i['sma5'] > i['sma14'],
    },
    'random_hold_short': {
        'description': 'Random-hold SHORT: enter on 3 down, exit after random hold',
        'entry_short': lambda i, p: (i['sma5'] < i['sma14'] * p.entry_sma_mult and p.entry_rsi_min <= i['rsi'] <= p.entry_rsi_max and i['allDown']),
        'entry_long': lambda i, p: (i['sma5'] > i['sma14'] * (2 - p.entry_sma_mult) and 25 <= i['rsi'] <= 40 and i['allUp']),
        'exit_short': lambda i, p: False,
        'exit_long': lambda i, p: False,
    },
    'bb_reversion': {
        'description': 'BB reversion + ADX filter: range markets only',
        'entry_short': lambda i, p: (i.get('cur', 0) > i.get('bb_upper', 0) * 0.999 and i['rsi'] > 65 and i.get('adx', 0) < 30),
        'entry_long': lambda i, p: (i.get('cur', 0) < i.get('bb_lower', 0) * 1.001 and i['rsi'] < 35 and i.get('adx', 0) < 30),
        'exit_short': lambda i, p: i.get('cur', 0) < i.get('sma20', 0),
        'exit_long': lambda i, p: i.get('cur', 0) > i.get('sma20', 0),
    },
    'macd_trend': {
        'description': 'MACD trend: SHORT MACD<signal + ADX>25',
        'entry_short': lambda i, p: (i.get('macd_hist', 0) < 0 and i.get('macd_line', 0) < i.get('macd_signal', 0) and i.get('adx', 0) > 25),
        'entry_long': lambda i, p: (i.get('macd_hist', 0) > 0 and i.get('macd_line', 0) > i.get('macd_signal', 0) and i.get('adx', 0) > 25),
        'exit_short': lambda i, p: i.get('macd_hist', 0) > 0,
        'exit_long': lambda i, p: i.get('macd_hist', 0) < 0,
    },
    'donchian_breakout': {
        'description': 'Donchian breakout: SHORT on new 20-period low + volume',
        'entry_short': lambda i, p: (i.get('cur', 0) < i.get('donchian_lower', 0) * 1.001 and i.get('vol_ratio', 0) > 1.2),
        'entry_long': lambda i, p: (i.get('cur', 0) > i.get('donchian_upper', 0) * 0.999 and i.get('vol_ratio', 0) > 1.2),
        'exit_short': lambda i, p: i.get('cur', 0) > i.get('donchian_upper', 0),
        'exit_long': lambda i, p: i.get('cur', 0) < i.get('donchian_lower', 0),
    },
    'stoch_oscillator': {
        'description': 'Stochastic: LONG %K<20+RSI<40, SHORT %K>80+RSI>60',
        'entry_short': lambda i, p: (i.get('stoch_k', 50) > 80 and i['rsi'] > 60),
        'entry_long': lambda i, p: (i.get('stoch_k', 50) < 20 and i['rsi'] < 40),
        'exit_short': lambda i, p: i.get('stoch_k', 50) < 50,
        'exit_long': lambda i, p: i.get('stoch_k', 50) > 50,
    },
    'vwap_reversion': {
        'description': 'VWAP reversion: SHORT price>VWAP*1.005+RSI>60',
        'entry_short': lambda i, p: (i.get('cur', 0) > i.get('vwap', 0) * 1.005 and i['rsi'] > 60),
        'entry_long': lambda i, p: (i.get('cur', 0) < i.get('vwap', 0) * 0.995 and i['rsi'] < 40),
        'exit_short': lambda i, p: i.get('cur', 0) < i.get('vwap', 0),
        'exit_long': lambda i, p: i.get('cur', 0) > i.get('vwap', 0),
    },
    'momentum_volume': {
        'description': 'Momentum: SHORT ROC<-2% + volume surge',
        'entry_short': lambda i, p: (i.get('roc', 0) < -2.0 and i.get('vol_ratio', 0) > 1.5 and i['rsi'] < 50),
        'entry_long': lambda i, p: (i.get('roc', 0) > 2.0 and i.get('vol_ratio', 0) > 1.5 and i['rsi'] > 50),
        'exit_short': lambda i, p: i.get('roc', 0) > 0,
        'exit_long': lambda i, p: i.get('roc', 0) < 0,
    },
}


# ─── 10 NEW strategies from research ────────────────────────────────────────

NEW_FROM_RESEARCH = {
    'connors_rsi2': {
        'description': 'Connors RSI(2): LONG RSI2<10 + price>SMA50, 77% win rate',
        'entry_short': lambda i, p: (i.get('rsi2', 50) > 90 and i.get('cur', 0) < i.get('sma50', 0)),
        'entry_long': lambda i, p: (i.get('rsi2', 50) < 10 and i.get('cur', 0) > i.get('sma50', 0)),
        'exit_short': lambda i, p: i.get('rsi2', 50) < 35 or i.get('cur', 0) < i.get('sma5', 0),
        'exit_long': lambda i, p: i.get('rsi2', 50) > 65 or i.get('cur', 0) > i.get('sma5', 0),
    },
    'zscore_reversion': {
        'description': 'Z-Score reversion: LONG z<-2, SHORT z>+2, 131% return',
        'entry_short': lambda i, p: i.get('zscore', 0) > 2.0,
        'entry_long': lambda i, p: i.get('zscore', 0) < -2.0,
        'exit_short': lambda i, p: i.get('zscore', 0) < 0.5,
        'exit_long': lambda i, p: i.get('zscore', 0) > -0.5,
    },
    'supertrend': {
        'description': 'Supertrend ATR(10,3): 67% accuracy trend follow',
        'entry_short': lambda i, p: i.get('supertrend_flip', 0) < 0,
        'entry_long': lambda i, p: i.get('supertrend_flip', 0) > 0,
        'exit_short': lambda i, p: i.get('supertrend_flip', 0) > 0,
        'exit_long': lambda i, p: i.get('supertrend_flip', 0) < 0,
    },
    'bollinger_squeeze': {
        'description': 'BB squeeze breakout: low volatility then breakout, R:R 2:1+',
        'entry_short': lambda i, p: (i.get('is_squeeze', False) and i.get('cur', 0) < i.get('bb_lower', 0)),
        'entry_long': lambda i, p: (i.get('is_squeeze', False) and i.get('cur', 0) > i.get('bb_upper', 0)),
        'exit_short': lambda i, p: i.get('cur', 0) > i.get('sma20', 0),
        'exit_long': lambda i, p: i.get('cur', 0) < i.get('sma20', 0),
    },
    'atr_bands': {
        'description': 'ATR bands mean reversion: 33yr backtest profitable',
        'entry_short': lambda i, p: i.get('cur', 0) > i.get('atr_upper', 0),
        'entry_long': lambda i, p: i.get('cur', 0) < i.get('atr_lower', 0),
        'exit_short': lambda i, p: i.get('cur', 0) < i.get('sma20', 0),
        'exit_long': lambda i, p: i.get('cur', 0) > i.get('sma20', 0),
    },
    'heikin_ashi': {
        'description': 'Heikin-Ashi+SMA50: noise-filtered trend, DD 29% vs 52%',
        'entry_short': lambda i, p: (i.get('ha_red', False) and i.get('ha_close', 0) < i.get('sma50', 0)),
        'entry_long': lambda i, p: (i.get('ha_green', False) and i.get('ha_close', 0) > i.get('sma50', 0)),
        'exit_short': lambda i, p: i.get('ha_green', False),
        'exit_long': lambda i, p: i.get('ha_red', False),
    },
    'dual_thrust': {
        'description': 'Dual Thrust breakout: intraday classic',
        'entry_short': lambda i, p: i.get('cur', 0) < i.get('dt_lower', 0),
        'entry_long': lambda i, p: i.get('cur', 0) > i.get('dt_upper', 0),
        'exit_short': lambda i, p: i.get('cur', 0) > i.get('dt_upper', 0),
        'exit_long': lambda i, p: i.get('cur', 0) < i.get('dt_lower', 0),
    },
    'awesome_oscillator': {
        'description': 'AO+MACD: momentum confirmation, fewer whipsaws',
        'entry_short': lambda i, p: (i.get('ao_cross', 0) < 0 and i.get('macd_hist', 0) < 0),
        'entry_long': lambda i, p: (i.get('ao_cross', 0) > 0 and i.get('macd_hist', 0) > 0),
        'exit_short': lambda i, p: i.get('ao_cross', 0) > 0,
        'exit_long': lambda i, p: i.get('ao_cross', 0) < 0,
    },
    'golden_cross': {
        'description': 'Golden Cross 50/200 SMA: $100k→$7.2M/66yr',
        'entry_short': lambda i, p: False,  # long-only
        'entry_long': lambda i, p: i.get('golden_cross', False),
        'exit_short': lambda i, p: False,
        'exit_long': lambda i, p: i.get('death_cross', False),
    },
    'orb': {
        'description': 'Opening Range Breakout 5-min: Sharpe 2.81',
        'entry_short': lambda i, p: i.get('cur', 0) < i.get('or_low', 0),
        'entry_long': lambda i, p: i.get('cur', 0) > i.get('or_high', 0),
        'exit_short': lambda i, p: i.get('cur', 0) > i.get('or_high', 0),
        'exit_long': lambda i, p: i.get('cur', 0) < i.get('or_low', 0),
    },
}


# Combine all
ALL_STRATEGIES = {**EXISTING, **NEW_FROM_RESEARCH}


def random_params(structure_name: str) -> StrategyParams:
    """Generate random params appropriate for the strategy structure."""
    if structure_name == 'connors_rsi2':
        return StrategyParams(
            entry_sma_mult=0.999, entry_rsi_min=5, entry_rsi_max=15,
            take_profit_pct=random.uniform(0.01, 0.03),
            hold_ticks=random.randint(30, 180),
            exit_sma_mult=1.003, position_size=random.uniform(0.2, 0.4),
        )
    elif structure_name == 'zscore_reversion':
        return StrategyParams(
            entry_sma_mult=0.999, entry_rsi_min=20, entry_rsi_max=80,
            take_profit_pct=random.uniform(0.01, 0.03),
            hold_ticks=random.randint(30, 180),
            exit_sma_mult=1.003, position_size=random.uniform(0.2, 0.4),
        )
    elif structure_name in ('supertrend', 'dual_thrust', 'orb', 'golden_cross'):
        return StrategyParams(
            entry_sma_mult=random.uniform(0.995, 1.005),
            entry_rsi_min=random.randint(20, 40),
            entry_rsi_max=random.randint(45, 60),
            take_profit_pct=random.uniform(0.01, 0.04),
            hold_ticks=random.randint(60, 600),
            exit_sma_mult=random.uniform(1.0, 1.005),
            position_size=random.uniform(0.2, 0.4),
        )
    elif structure_name in ('bollinger_squeeze', 'atr_bands', 'heikin_ashi', 'awesome_oscillator'):
        return StrategyParams(
            entry_sma_mult=random.uniform(0.995, 1.005),
            entry_rsi_min=random.randint(20, 40),
            entry_rsi_max=random.randint(45, 60),
            take_profit_pct=random.uniform(0.01, 0.03),
            hold_ticks=random.randint(60, 300),
            exit_sma_mult=random.uniform(1.0, 1.005),
            position_size=random.uniform(0.2, 0.4),
        )
    else:
        # Default for V2-based strategies
        return StrategyParams(
            entry_sma_mult=random.uniform(0.995, 1.005),
            entry_rsi_min=random.randint(20, 40),
            entry_rsi_max=random.randint(45, 60),
            take_profit_pct=random.uniform(0.005, 0.025),
            hold_ticks=random.randint(30, 300),
            exit_sma_mult=random.uniform(1.002, 1.005),
            position_size=random.uniform(0.2, 0.4),
        )
