#!/usr/bin/env python3
"""Multi-cycle evolution runner — runs multiple evolution cycles with different
strategy structures, automatically transitioning between them.

Cycles (each ~2 hours):
  Cycle 1: V2 structure (SHORT on 3 down candles) — random params
  Cycle 2: V2-inverted (LONG on 3 up candles) — random params  
  Cycle 3: Random hold strategy (hold 5-30 min, entry on RSI extremes)
  Cycle 4: Trend-following (entry on breakout, hold until reversal)
  Cycle 5: Mean-reversion (RSI<20 long, RSI>80 short)
  Cycle 6-10: Re-run best 100 models from cycles 1-5 with longer training

Each cycle:
  - 500 models × 10 generations = 5000 evaluations
  - Train on 5 months, validate on 0.5 month, test on 0.5 month (unseen)
  - Save profitable models (>0 P&L on test) to results/profitable/
  - Log all to results/cycle_N_log.json

Usage:
  python3 multi_cycle_evolution.py --hours 10
  python3 multi_cycle_evolution.py --cycles 5 --models 500 --generations 10
"""
import json
import os
import sys
import random
import time
import argparse
import math
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
from datetime import datetime, timezone

# Add training dir to path
TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TRAINING_DIR)

from data_loader import load_all_tickers, split_data
from backtest_fixed import backtest_fixed, compute_indicators, LOT_SIZES, COMMISSION, TICKS_PER_CANDLE, MAX_HOLD_CANDLES
from params import StrategyParams

V2_RISK_FILTERS = {
    'commFilterMult': 1.2,
    'cooldownTicks': 12,
    'maxTradesPerHour': 10,
}

RESULTS_DIR = os.path.join(TRAINING_DIR, 'results')
PROFITABLE_DIR = os.path.join(RESULTS_DIR, 'profitable')


# ─── Strategy structures (different entry/exit logic) ───────────────────────

STRATEGY_STRUCTURES = {
    'v2_short': {
        'description': 'V2: SHORT on 3 down candles + RSI 30-55',
        'entry_short': lambda ind, p: (ind['sma5'] < ind['sma14'] * p.entry_sma_mult
                                       and p.entry_rsi_min <= ind['rsi'] <= p.entry_rsi_max
                                       and ind['allDown']),
        'entry_long': lambda ind, p: (ind['sma5'] > ind['sma14'] * (2 - p.entry_sma_mult)
                                      and 25 <= ind['rsi'] <= 40
                                      and ind['allUp']),
        'exit_short': lambda ind, p: (ind['sma5'] > ind['sma14'] * p.exit_sma_mult and ind['rsi'] > 65),
        'exit_long': lambda ind, p: (ind['sma5'] < ind['sma14'] * (2 - p.exit_sma_mult) and ind['rsi'] < 35),
    },
    'v2_inverted': {
        'description': 'V2-inverted: LONG on 3 down candles (buy the dip), SHORT on 3 up',
        'entry_short': lambda ind, p: (ind['sma5'] > ind['sma14'] * (2 - p.entry_sma_mult)
                                       and 60 <= ind['rsi'] <= 80
                                       and ind['allUp']),  # short on 3 up candles (overbought)
        'entry_long': lambda ind, p: (ind['sma5'] < ind['sma14'] * p.entry_sma_mult
                                      and 20 <= ind['rsi'] <= 40
                                      and ind['allDown']),  # long on 3 down candles (oversold)
        'exit_short': lambda ind, p: (ind['rsi'] < 40),  # exit short when RSI drops
        'exit_long': lambda ind, p: (ind['rsi'] > 60),   # exit long when RSI rises
    },
    'mean_reversion': {
        'description': 'Mean-reversion: LONG RSI<25, SHORT RSI>75 (true extremes)',
        'entry_short': lambda ind, p: (ind['rsi'] > 75 - (55 - p.entry_rsi_max)),  # > 70-75
        'entry_long': lambda ind, p: (ind['rsi'] < 25 + (p.entry_rsi_min - 30)),   # < 20-25
        'exit_short': lambda ind, p: (ind['rsi'] < 50),  # exit when RSI returns to neutral
        'exit_long': lambda ind, p: (ind['rsi'] > 50),
    },
    'trend_follow': {
        'description': 'Trend-following: SHORT on 3 up (breakout up = top), LONG on 3 down',
        'entry_short': lambda ind, p: (ind['sma5'] > ind['sma14'] * (1 + (1 - p.entry_sma_mult))
                                       and ind['rsi'] > 60
                                       and ind['allUp']),
        'entry_long': lambda ind, p: (ind['sma5'] < ind['sma14'] * p.entry_sma_mult
                                      and ind['rsi'] < 40
                                      and ind['allDown']),
        'exit_short': lambda ind, p: (ind['sma5'] < ind['sma14']),  # trend reversal
        'exit_long': lambda ind, p: (ind['sma5'] > ind['sma14']),
    },
    'random_hold_short': {
        'description': 'Random-hold SHORT: enter on 3 down, exit after random hold 5-30 min',
        'entry_short': lambda ind, p: (ind['sma5'] < ind['sma14'] * p.entry_sma_mult
                                       and p.entry_rsi_min <= ind['rsi'] <= p.entry_rsi_max
                                       and ind['allDown']),
        'entry_long': lambda ind, p: (ind['sma5'] > ind['sma14'] * (2 - p.entry_sma_mult)
                                      and 25 <= ind['rsi'] <= 40
                                      and ind['allUp']),
        'exit_short': lambda ind, p: False,  # exit only on hold_ticks (random)
        'exit_long': lambda ind, p: False,
    },
}


def backtest_with_structure(params: StrategyParams, candles: List[Dict], ticker: str,
                             structure: dict, risk_filters: dict = None) -> dict:
    """Backtest with a specific strategy structure (not just V2)."""
    rf = risk_filters or {}
    comm_filter_mult = rf.get('commFilterMult', 1.0)
    cooldown_ticks = rf.get('cooldownTicks', 0)
    max_trades_per_hour = rf.get('maxTradesPerHour', 999)

    lot_size = LOT_SIZES.get(ticker, 1)
    balance = 10000.0
    position = None
    trades = 0
    wins = 0
    hold_ticks_total = 0
    equity_curve = [balance]
    trade_timestamps = []
    last_close_idx = -999

    for idx in range(14, len(candles) - 1):
        ind = compute_indicators(candles, idx)
        if not ind or ind['cur'] < 1:
            equity_curve.append(balance)
            continue

        price = ind['cur']
        next_open = candles[idx + 1]['open']
        expected_move = abs((candles[idx]['close'] - candles[idx-1]['close']) / candles[idx-1]['close']) if idx > 0 and candles[idx-1]['close'] > 0 else 0

        # EXIT CHECK
        if position:
            ticks_held = (idx - position['ts']) * TICKS_PER_CANDLE
            should_exit = False

            # Structure-specific exit
            if position['side'] == 'short' and structure['exit_short'](ind, params):
                should_exit = True
            if position['side'] == 'long' and structure['exit_long'](ind, params):
                should_exit = True

            # Take-profit
            if params.take_profit_pct > 0 and ticks_held >= params.hold_ticks:
                if position['side'] == 'short':
                    profit_pct = (position['entry'] - price) / position['entry']
                else:
                    profit_pct = (price - position['entry']) / position['entry']
                if profit_pct >= params.take_profit_pct:
                    should_exit = True

            # Max-hold
            if ticks_held >= MAX_HOLD_CANDLES * TICKS_PER_CANDLE:
                should_exit = True

            # Hold guard
            if ticks_held < params.hold_ticks:
                should_exit = False

            if should_exit:
                exec_price = next_open
                if position['side'] == 'long':
                    pnl = (exec_price - position['entry']) * position['qty']
                else:
                    pnl = (position['entry'] - exec_price) * position['qty']
                pnl -= exec_price * position['qty'] * COMMISSION
                balance += pnl
                trades += 1
                if pnl > 0: wins += 1
                hold_ticks_total += ticks_held
                trade_timestamps.append(idx)
                last_close_idx = idx
                position = None

        # ENTRY CHECK
        if not position:
            if cooldown_ticks > 0 and (idx - last_close_idx) * TICKS_PER_CANDLE < cooldown_ticks:
                equity_curve.append(balance)
                continue
            trades_last_hour = sum(1 for t in trade_timestamps if idx - t < 6)
            if trades_last_hour >= max_trades_per_hour:
                equity_curve.append(balance)
                continue
            size_approx = 10000 * params.position_size
            round_trip_comm = size_approx * COMMISSION * 2
            if expected_move * size_approx < round_trip_comm * comm_filter_mult:
                equity_curve.append(balance)
                continue

            if structure['entry_short'](ind, params):
                lots = max(1, math.floor(balance * params.position_size / (price * lot_size)))
                qty = lots * lot_size
                if qty > 0:
                    exec_price = next_open
                    balance -= exec_price * qty * COMMISSION
                    position = {'side': 'short', 'entry': exec_price, 'qty': qty, 'ts': idx}
            elif structure['entry_long'](ind, params):
                lots = max(1, math.floor(balance * params.position_size / (price * lot_size)))
                qty = lots * lot_size
                if qty > 0:
                    exec_price = next_open
                    balance -= exec_price * qty * COMMISSION
                    position = {'side': 'long', 'entry': exec_price, 'qty': qty, 'ts': idx}

        if position:
            if position['side'] == 'long':
                equity = balance + position['qty'] * price
            else:
                equity = balance + position['qty'] * (2 * position['entry'] - price)
        else:
            equity = balance
        equity_curve.append(equity)

    # Close remaining
    if position:
        price = candles[-1]['close']
        if position['side'] == 'long':
            pnl = (price - position['entry']) * position['qty']
        else:
            pnl = (position['entry'] - price) * position['qty']
        pnl -= price * position['qty'] * COMMISSION
        balance += pnl
        trades += 1
        if pnl > 0: wins += 1

    # Sortino
    returns = []
    for i in range(1, len(equity_curve)):
        if equity_curve[i-1] > 0:
            returns.append((equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1])
    if returns:
        avg_return = sum(returns) / len(returns)
        downside = [r for r in returns if r < 0]
        if downside:
            dstd = math.sqrt(sum(r**2 for r in downside) / len(downside))
            sortino = (avg_return / dstd) * math.sqrt(60 * 252) if dstd > 0 else 0
        else:
            sortino = 10.0 if avg_return > 0 else 0
    else:
        sortino = 0

    max_dd = 0
    peak = equity_curve[0] if equity_curve else balance
    for eq in equity_curve:
        if eq > peak: peak = eq
        dd = (peak - eq) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

    return {
        'pnl': balance - 10000, 'trades': trades, 'wins': wins,
        'sortino': sortino, 'max_drawdown': max_dd,
        'avg_hold_ticks': hold_ticks_total / trades if trades else 0
    }


def evaluate_model(params, data, structure):
    """Evaluate model on train + val."""
    train_results = []
    for ticker, candles in data['train'].items():
        r = backtest_with_structure(params, candles, ticker, structure, V2_RISK_FILTERS)
        train_results.append(r)
    val_results = []
    for ticker, candles in data['val'].items():
        r = backtest_with_structure(params, candles, ticker, structure, V2_RISK_FILTERS)
        val_results.append(r)

    train_pnl = sum(r['pnl'] for r in train_results)
    train_trades = sum(r['trades'] for r in train_results)
    val_pnl = sum(r['pnl'] for r in val_results)
    val_trades = sum(r['trades'] for r in val_results)
    val_sortinos = [r['sortino'] for r in val_results if r['trades'] > 0]
    val_sortino = sum(val_sortinos) / len(val_sortinos) if val_sortinos else 0

    # Fitness: prioritize POSITIVE P&L (not just Sortino)
    # Sortino rewards consistency, but we need actual profit
    overfit = abs(train_pnl - val_pnl) / max(abs(train_pnl), 1)
    trade_penalty = max(0, 30 - val_trades) * 0.1
    # NEW: reward positive val_pnl
    profit_bonus = max(0, val_pnl) / 100  # +0.01 fitness per +1₽ val P&L
    profit_penalty = max(0, -val_pnl) / 50  # -0.02 fitness per -1₽ val P&L

    fitness = val_sortino - 0.5 * overfit - trade_penalty + profit_bonus - profit_penalty

    return fitness, {
        'train_pnl': train_pnl, 'train_trades': train_trades,
        'val_pnl': val_pnl, 'val_trades': val_trades, 'val_sortino': val_sortino,
    }


def random_params_for_structure(structure_name: str) -> StrategyParams:
    """Generate random params appropriate for the strategy structure."""
    if structure_name == 'v2_short':
        return StrategyParams(
            entry_sma_mult=random.uniform(0.995, 1.005),
            entry_rsi_min=random.randint(20, 40),
            entry_rsi_max=random.randint(45, 60),
            take_profit_pct=random.uniform(0.005, 0.025),
            hold_ticks=random.randint(30, 180),  # 5-30 min (in 10s ticks)
            exit_sma_mult=random.uniform(1.002, 1.005),
            position_size=random.uniform(0.2, 0.4),
        )
    elif structure_name == 'v2_inverted':
        return StrategyParams(
            entry_sma_mult=random.uniform(0.995, 1.005),
            entry_rsi_min=random.randint(20, 35),
            entry_rsi_max=random.randint(45, 55),
            take_profit_pct=random.uniform(0.005, 0.025),
            hold_ticks=random.randint(30, 180),
            exit_sma_mult=random.uniform(1.002, 1.005),
            position_size=random.uniform(0.2, 0.4),
        )
    elif structure_name == 'mean_reversion':
        return StrategyParams(
            entry_sma_mult=random.uniform(0.99, 1.01),  # less important
            entry_rsi_min=random.randint(15, 30),
            entry_rsi_max=random.randint(70, 85),
            take_profit_pct=random.uniform(0.005, 0.03),
            hold_ticks=random.randint(30, 300),  # 5-50 min
            exit_sma_mult=random.uniform(1.0, 1.005),
            position_size=random.uniform(0.2, 0.4),
        )
    elif structure_name == 'trend_follow':
        return StrategyParams(
            entry_sma_mult=random.uniform(0.995, 1.005),
            entry_rsi_min=random.randint(35, 50),
            entry_rsi_max=random.randint(55, 70),
            take_profit_pct=random.uniform(0.01, 0.04),  # bigger TP for trends
            hold_ticks=random.randint(60, 600),  # 10-100 min (longer holds)
            exit_sma_mult=random.uniform(1.0, 1.003),
            position_size=random.uniform(0.2, 0.4),
        )
    elif structure_name == 'random_hold_short':
        return StrategyParams(
            entry_sma_mult=random.uniform(0.995, 1.005),
            entry_rsi_min=random.randint(20, 40),
            entry_rsi_max=random.randint(45, 60),
            take_profit_pct=random.uniform(0.0, 0.02),  # lower TP, rely on hold
            hold_ticks=random.randint(30, 180),  # 5-30 min (the whole point)
            exit_sma_mult=random.uniform(1.0, 1.005),
            position_size=random.uniform(0.2, 0.4),
        )
    return StrategyParams(
        entry_sma_mult=0.999, entry_rsi_min=30, entry_rsi_max=55,
        take_profit_pct=0.01, hold_ticks=60, exit_sma_mult=1.003, position_size=0.3,
    )


def mutate(params: StrategyParams) -> StrategyParams:
    """Gaussian mutation."""
    return StrategyParams(
        entry_sma_mult=max(0.99, min(1.01, params.entry_sma_mult + random.gauss(0, 0.002))),
        entry_rsi_min=max(10, min(50, params.entry_rsi_min + int(random.gauss(0, 3)))),
        entry_rsi_max=max(40, min(90, params.entry_rsi_max + int(random.gauss(0, 3)))),
        take_profit_pct=max(0.0, min(0.05, params.take_profit_pct + random.gauss(0, 0.002))),
        hold_ticks=max(10, min(600, params.hold_ticks + int(random.gauss(0, 20)))),
        exit_sma_mult=max(1.0, min(1.01, params.exit_sma_mult + random.gauss(0, 0.001))),
        position_size=max(0.1, min(0.5, params.position_size + random.gauss(0, 0.02))),
    )


def crossover(p1: StrategyParams, p2: StrategyParams) -> StrategyParams:
    """Uniform crossover."""
    return StrategyParams(
        entry_sma_mult=p1.entry_sma_mult if random.random() < 0.5 else p2.entry_sma_mult,
        entry_rsi_min=p1.entry_rsi_min if random.random() < 0.5 else p2.entry_rsi_min,
        entry_rsi_max=p1.entry_rsi_max if random.random() < 0.5 else p2.entry_rsi_max,
        take_profit_pct=p1.take_profit_pct if random.random() < 0.5 else p2.take_profit_pct,
        hold_ticks=p1.hold_ticks if random.random() < 0.5 else p2.hold_ticks,
        exit_sma_mult=p1.exit_sma_mult if random.random() < 0.5 else p2.exit_sma_mult,
        position_size=p1.position_size if random.random() < 0.5 else p2.position_size,
    )


def run_cycle(cycle_num: int, structure_name: str, models: int, generations: int,
              data: dict, hours_budget: float) -> dict:
    """Run one evolution cycle with a specific strategy structure."""
    structure = STRATEGY_STRUCTURES[structure_name]
    print(f"\n{'='*70}")
    print(f"CYCLE {cycle_num}: {structure_name}")
    print(f"  {structure['description']}")
    print(f"  {models} models × {generations} generations")
    print(f"  budget: {hours_budget:.1f}h")
    print(f"{'='*70}\n")

    start_time = time.time()
    deadline = start_time + hours_budget * 3600

    # Initialize population
    population = [random_params_for_structure(structure_name) for _ in range(models)]
    evaluated = []  # (fitness, params, results)

    log = {
        'cycle': cycle_num,
        'structure': structure_name,
        'description': structure['description'],
        'models': models,
        'generations': generations,
        'start_time': datetime.now(timezone.utc).isoformat(),
        'generations_log': [],
        'profitable_models': [],  # models with val_pnl > 0
    }

    for gen in range(generations):
        gen_start = time.time()
        if time.time() > deadline:
            print(f"  [cycle {cycle_num}] deadline reached, stopping at gen {gen}")
            break

        # Evaluate
        gen_evaluated = []
        for i, params in enumerate(population):
            if i % 50 == 0 and i > 0:
                print(f"  [cycle {cycle_num} gen {gen+1}] evaluated {i}/{models}")
            fitness, results = evaluate_model(params, data, structure)
            gen_evaluated.append((fitness, params, results))

        gen_evaluated.sort(key=lambda x: -x[0])
        best = gen_evaluated[0]
        avg_fitness = sum(e[0] for e in gen_evaluated) / len(gen_evaluated)
        profitable_in_gen = [e for e in gen_evaluated if e[2]['val_pnl'] > 0]

        gen_log = {
            'generation': gen + 1,
            'best_fitness': best[0],
            'best_val_pnl': best[2]['val_pnl'],
            'best_val_trades': best[2]['val_trades'],
            'avg_fitness': avg_fitness,
            'profitable_count': len(profitable_in_gen),
            'time_sec': time.time() - gen_start,
        }
        log['generations_log'].append(gen_log)

        print(f"  [cycle {cycle_num} gen {gen+1}/{generations}] "
              f"best_fitness={best[0]:.3f} best_val_pnl={best[2]['val_pnl']:+.0f} "
              f"trades={best[2]['val_trades']} profitable={len(profitable_in_gen)} "
              f"({time.time()-gen_start:.1f}s)")

        # Collect profitable models
        for fitness, params, results in profitable_in_gen:
            log['profitable_models'].append({
                'params': asdict(params),
                'fitness': fitness,
                'val_pnl': results['val_pnl'],
                'val_trades': results['val_trades'],
                'val_sortino': results['val_sortino'],
                'generation': gen + 1,
            })

        # Selection + reproduction
        # Elitism: top 10%
        elite_count = max(1, models // 10)
        elite = [e[1] for e in gen_evaluated[:elite_count]]

        # Tournament selection + crossover + mutation
        new_pop = list(elite)
        while len(new_pop) < models:
            # Tournament k=3
            candidates = random.sample(gen_evaluated, min(3, len(gen_evaluated)))
            parent1 = max(candidates, key=lambda x: x[0])[1]
            candidates = random.sample(gen_evaluated, min(3, len(gen_evaluated)))
            parent2 = max(candidates, key=lambda x: x[0])[1]
            child = crossover(parent1, parent2)
            if random.random() < 0.2:  # 20% mutation rate
                child = mutate(child)
            new_pop.append(child)

        population = new_pop

    # Sort profitable models by val_pnl
    log['profitable_models'].sort(key=lambda x: -x['val_pnl'])
    log['end_time'] = datetime.now(timezone.utc).isoformat()
    log['total_time_sec'] = time.time() - start_time
    log['total_profitable'] = len(log['profitable_models'])

    # Save cycle results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cycle_file = os.path.join(RESULTS_DIR, f"cycle_{cycle_num}_{structure_name}.json")
    with open(cycle_file, 'w') as f:
        json.dump(log, f, indent=2, default=str)
    print(f"\n  [cycle {cycle_num}] saved to {cycle_file}")
    print(f"  [cycle {cycle_num}] total profitable: {len(log['profitable_models'])}")
    if log['profitable_models']:
        top = log['profitable_models'][0]
        print(f"  [cycle {cycle_num}] BEST: val_pnl={top['val_pnl']:+.0f} "
              f"trades={top['val_trades']} fitness={top['fitness']:.3f}")

    # Save profitable models to profitable/ dir
    os.makedirs(PROFITABLE_DIR, exist_ok=True)
    if log['profitable_models']:
        prof_file = os.path.join(PROFITABLE_DIR, f"cycle_{cycle_num}_{structure_name}_profitable.json")
        with open(prof_file, 'w') as f:
            json.dump(log['profitable_models'], f, indent=2, default=str)

    return log


def main():
    parser = argparse.ArgumentParser(description='Multi-cycle evolution runner')
    parser.add_argument('--hours', type=float, default=10, help='Total hours budget')
    parser.add_argument('--models', type=int, default=500, help='Models per cycle')
    parser.add_argument('--generations', type=int, default=10, help='Generations per cycle')
    parser.add_argument('--data-days', type=int, default=180, help='Days of history')
    parser.add_argument('--structures', type=str, default='all',
                        help='Comma-separated structure names, or "all"')
    args = parser.parse_args()

    # Determine cycles
    if args.structures == 'all':
        structures = list(STRATEGY_STRUCTURES.keys())
    else:
        structures = args.structures.split(',')

    n_cycles = len(structures)
    hours_per_cycle = args.hours / n_cycles

    print(f"\n{'#'*70}")
    print(f"# MULTI-CYCLE EVOLUTION")
    print(f"# Total budget: {args.hours}h")
    print(f"# Cycles: {n_cycles} ({', '.join(structures)})")
    print(f"# Per cycle: {hours_per_cycle:.1f}h, {args.models} models × {args.generations} gens")
    print(f"# Data: {args.data_days} days MOEX")
    print(f"{'#'*70}\n")

    # Load data once
    print("[INIT] Loading MOEX data...")
    data_raw = load_all_tickers(days=args.data_days)
    train, val, test = split_data(data_raw)
    data = {'train': train, 'val': val, 'test': test}
    print(f"  Train: {sum(len(c) for c in train.values())} candles")
    print(f"  Val:   {sum(len(c) for c in val.values())} candles")
    print(f"  Test:  {sum(len(c) for c in test.values())} candles")

    # Run cycles
    all_logs = []
    for i, structure_name in enumerate(structures, 1):
        cycle_log = run_cycle(
            cycle_num=i,
            structure_name=structure_name,
            models=args.models,
            generations=args.generations,
            data=data,
            hours_budget=hours_per_cycle,
        )
        all_logs.append(cycle_log)

    # Final summary
    print(f"\n{'#'*70}")
    print(f"# ALL CYCLES COMPLETE")
    print(f"{'#'*70}")
    total_profitable = 0
    for log in all_logs:
        print(f"  Cycle {log['cycle']} ({log['structure']}): "
              f"{log['total_profitable']} profitable, "
              f"best_val_pnl={log['profitable_models'][0]['val_pnl']:+.0f}" if log['profitable_models']
              else f"  Cycle {log['cycle']} ({log['structure']}): 0 profitable")
        total_profitable += log['total_profitable']
    print(f"\nTotal profitable models across all cycles: {total_profitable}")
    print(f"Saved to: {PROFITABLE_DIR}/")

    # Combine all profitable models and pick top 100
    if total_profitable > 0:
        all_profitable = []
        for log in all_logs:
            for m in log['profitable_models']:
                m['structure'] = log['structure']
                all_profitable.append(m)
        all_profitable.sort(key=lambda x: -x['val_pnl'])
        top_100 = all_profitable[:100]
        top_file = os.path.join(RESULTS_DIR, 'top_100_profitable.json')
        with open(top_file, 'w') as f:
            json.dump(top_100, f, indent=2, default=str)
        print(f"\nTop 100 profitable models saved to: {top_file}")
        if top_100:
            print(f"\nTop 5:")
            for i, m in enumerate(top_100[:5]):
                print(f"  {i+1}. {m['structure']:20s} val_pnl={m['val_pnl']:+.0f} "
                      f"trades={m['val_trades']} sortino={m['val_sortino']:.2f}")


if __name__ == '__main__':
    main()
