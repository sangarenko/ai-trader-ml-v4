#!/usr/bin/env python3
"""Monte Carlo runner — parallel random search across all strategy structures.

Usage:
  python3 monte_carlo_runner.py --models 2000 --seed 42 --tag batch1
  python3 monte_carlo_runner.py --models 2000 --seed 43 --tag batch2
"""
import os
import sys
import json
import time
import random
import argparse
from dataclasses import asdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_all_tickers, split_data
from multi_cycle_evolution import backtest_with_structure, V2_RISK_FILTERS, STRATEGY_STRUCTURES
from extended_strategies import compute_indicators_extended, NEW_STRATEGY_STRUCTURES, random_params_for_structure_extended
from params import StrategyParams

# Monkey-patch: make backtest_with_structure use extended indicators
import multi_cycle_evolution as mce
original_backtest = mce.backtest_with_structure

def backtest_with_extended(params, candles, ticker, structure, risk_filters=None):
    """Backtest using extended indicators (BB, MACD, ATR, etc.)."""
    import math
    rf = risk_filters or {}
    comm_filter_mult = rf.get('commFilterMult', 1.0)
    cooldown_ticks = rf.get('cooldownTicks', 0)
    max_trades_per_hour = rf.get('maxTradesPerHour', 999)
    
    from multi_cycle_evolution import LOT_SIZES, COMMISSION, TICKS_PER_CANDLE, MAX_HOLD_CANDLES
    
    lot_size = LOT_SIZES.get(ticker, 1)
    balance = 10000.0
    position = None
    trades = 0
    wins = 0
    hold_ticks_total = 0
    equity_curve = [balance]
    trade_timestamps = []
    last_close_idx = -999
    
    for idx in range(26, len(candles) - 1):  # 26 for ADX
        ind = compute_indicators_extended(candles, idx)
        if not ind or ind['cur'] < 1:
            equity_curve.append(balance)
            continue
        
        price = ind['cur']
        next_open = candles[idx + 1]['open']
        expected_move = abs((candles[idx]['close'] - candles[idx-1]['close']) / candles[idx-1]['close']) if idx > 0 and candles[idx-1]['close'] > 0 else 0
        
        # EXIT
        if position:
            ticks_held = (idx - position['ts']) * TICKS_PER_CANDLE
            should_exit = False
            try:
                if position['side'] == 'short' and structure['exit_short'](ind, params):
                    should_exit = True
                if position['side'] == 'long' and structure['exit_long'](ind, params):
                    should_exit = True
            except Exception:
                should_exit = False
            
            if params.take_profit_pct > 0 and ticks_held >= params.hold_ticks:
                if position['side'] == 'short':
                    profit_pct = (position['entry'] - price) / position['entry']
                else:
                    profit_pct = (price - position['entry']) / position['entry']
                if profit_pct >= params.take_profit_pct:
                    should_exit = True
            
            if ticks_held >= MAX_HOLD_CANDLES * TICKS_PER_CANDLE:
                should_exit = True
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
        
        # ENTRY
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
            
            try:
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
            except Exception:
                pass
        
        if position:
            if position['side'] == 'long':
                equity = balance + position['qty'] * price
            else:
                equity = balance + position['qty'] * (2 * position['entry'] - price)
        else:
            equity = balance
        equity_curve.append(equity)
    
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tag', type=str, default='batch1')
    parser.add_argument('--data-days', type=int, default=180)
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    print(f"\n{'='*70}")
    print(f"MONTE CARLO BATCH: {args.tag}")
    print(f"  models: {args.models}")
    print(f"  seed: {args.seed}")
    print(f"  data: {args.data_days} days MOEX")
    print(f"{'='*70}\n")
    
    # Load data
    print("[1/3] Loading data...")
    data_raw = load_all_tickers(days=args.data_days)
    train, val, test = split_data(data_raw)
    data = {'train': train, 'val': val, 'test': test}
    print(f"  Train: {sum(len(c) for c in train.values())} candles")
    print(f"  Val:   {sum(len(c) for c in val.values())} candles")
    print(f"  Test:  {sum(len(c) for c in test.values())} candles")
    
    # All structures
    all_structures = {**STRATEGY_STRUCTURES, **NEW_STRATEGY_STRUCTURES}
    print(f"\n[2/3] Structures: {len(all_structures)}")
    for name, s in all_structures.items():
        print(f"  - {name}: {s['description'][:60]}")
    
    # Monte Carlo search
    print(f"\n[3/3] Running Monte Carlo ({args.models} models)...")
    profitable = []
    start_time = time.time()
    results_log = []
    
    for i in range(args.models):
        if (i + 1) % 25 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (i + 1) * (args.models - i - 1)
            print(f"  [{i+1}/{args.models}] profitable: {len(profitable)} | "
                  f"elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s | "
                  f"structures tried: {len(set(r['structure'] for r in results_log[-25:]))}")
        
        struct_name = random.choice(list(all_structures.keys()))
        structure = all_structures[struct_name]
        params = random_params_for_structure_extended(struct_name)
        
        # Val check first (fast filter)
        val_pnl = 0
        val_trades = 0
        val_sortinos = []
        for ticker, candles in data['val'].items():
            r = backtest_with_extended(params, candles, ticker, structure, V2_RISK_FILTERS)
            val_pnl += r['pnl']
            val_trades += r['trades']
            if r['trades'] > 0:
                val_sortinos.append(r['sortino'])
        
        if val_pnl < 0 or val_trades < 10:
            results_log.append({
                'structure': struct_name, 'val_pnl': val_pnl, 'val_trades': val_trades,
                'test_pnl': None, 'status': 'filtered_val'
            })
            continue
        
        # Test check
        test_pnl = 0
        test_trades = 0
        test_sortinos = []
        for ticker, candles in data['test'].items():
            r = backtest_with_extended(params, candles, ticker, structure, V2_RISK_FILTERS)
            test_pnl += r['pnl']
            test_trades += r['trades']
            if r['trades'] > 0:
                test_sortinos.append(r['sortino'])
        
        val_sortino = sum(val_sortinos) / len(val_sortinos) if val_sortinos else 0
        test_sortino = sum(test_sortinos) / len(test_sortinos) if test_sortinos else 0
        
        # Keep if BOTH val and test positive
        if val_pnl > 0 and test_pnl > 0:
            # Also check train (must not be hugely negative)
            train_pnl = 0
            train_trades = 0
            for ticker, candles in data['train'].items():
                r = backtest_with_extended(params, candles, ticker, structure, V2_RISK_FILTERS)
                train_pnl += r['pnl']
                train_trades += r['trades']
            
            fitness = val_pnl + test_pnl - abs(val_pnl - test_pnl) * 0.5 + min(train_pnl, 0) * 0.1
            
            profitable.append({
                'structure': struct_name,
                'params': asdict(params),
                'train_pnl': train_pnl,
                'train_trades': train_trades,
                'val_pnl': val_pnl,
                'val_trades': val_trades,
                'val_sortino': val_sortino,
                'test_pnl': test_pnl,
                'test_trades': test_trades,
                'test_sortino': test_sortino,
                'fitness': fitness,
            })
            print(f"  [{i+1}/{args.models}] PROFITABLE: {struct_name} val={val_pnl:+.0f} test={test_pnl:+.0f} fitness={fitness:.1f}")
        
        results_log.append({
            'structure': struct_name, 'val_pnl': val_pnl, 'val_trades': val_trades,
            'test_pnl': test_pnl, 'test_trades': test_trades, 'status': 'evaluated'
        })
    
    # Sort by fitness
    profitable.sort(key=lambda x: -x['fitness'])
    
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"BATCH {args.tag} COMPLETE")
    print(f"  Total models: {args.models}")
    print(f"  Profitable (val>0 AND test>0): {len(profitable)} ({len(profitable)/args.models*100:.1f}%)")
    print(f"  Time: {elapsed:.0f}s ({elapsed/args.models:.2f}s per model)")
    print(f"{'='*70}\n")
    
    # Structure breakdown
    struct_counts = {}
    for r in results_log:
        struct_counts[r['structure']] = struct_counts.get(r['structure'], 0) + 1
    print("Models tried per structure:")
    for s, c in sorted(struct_counts.items(), key=lambda x: -x[1]):
        prof_count = sum(1 for p in profitable if p['structure'] == s)
        print(f"  {s:25s}: {c:4d} tried, {prof_count} profitable")
    
    if profitable:
        print(f"\nTop 20 profitable models:")
        for i, m in enumerate(profitable[:20]):
            print(f"  {i+1:2d}. {m['structure']:22s} train={m['train_pnl']:+7.0f} val={m['val_pnl']:+7.0f} test={m['test_pnl']:+7.0f} "
                  f"trades={m['val_trades']:3d} fitness={m['fitness']:.1f}")
    
    # Save results
    results_dir = "/root/ai-trader-evolution/training/results"
    os.makedirs(results_dir, exist_ok=True)
    
    output = {
        'tag': args.tag,
        'seed': args.seed,
        'models': args.models,
        'data_days': args.data_days,
        'profitable_count': len(profitable),
        'total_time_sec': elapsed,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'profitable_models': profitable,
        'structure_stats': struct_counts,
    }
    
    output_file = os.path.join(results_dir, f"monte_carlo_{args.tag}.json")
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")
    
    # Also save just profitable models (for easy loading)
    if profitable:
        prof_file = os.path.join(results_dir, f"monte_carlo_{args.tag}_profitable.json")
        with open(prof_file, 'w') as f:
            json.dump(profitable, f, indent=2, default=str)
        print(f"Profitable models: {prof_file}")


if __name__ == '__main__':
    main()
