#!/usr/bin/env python3
"""Fast Monte Carlo — 1M models sweep with ALL results recorded.

This replaces monte_carlo_runner.py with the fast vectorized engine.
Records EVERY model (profitable + unprofitable) for ML training.

Output: results/all_models_1m.parquet
  Columns: strategy, entry_sma_mult, rsi_min, rsi_max, take_profit,
           hold_ticks, exit_sma_mult, position_size,
           train_pnl, val_pnl, test_pnl, trades, win_rate, sortino,
           profitable (0/1)

Usage:
  python3 fast_monte_carlo.py --models 1000000 --data-days 180
"""
import os
import sys
import json
import time
import random
import argparse
import numpy as np
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fast_backtest import precompute_indicators, vectorized_backtest
from data_loader import load_all_tickers, split_data


# All 22 strategies
STRATEGIES = [
    "multi_timeframe", "v2_short", "v2_inverted", "mean_reversion",
    "trend_follow", "random_hold_short", "bb_reversion", "macd_trend",
    "donchian_breakout", "stoch_oscillator", "vwap_reversion", "momentum_volume",
    "connors_rsi2", "zscore_reversion", "supertrend", "bollinger_squeeze",
    "atr_bands", "heikin_ashi", "dual_thrust", "awesome_oscillator",
    "golden_cross", "orb",
]


def random_params(strategy: str) -> dict:
    """Generate random params for a strategy."""
    return {
        "entry_sma_mult": random.uniform(0.995, 1.005),
        "entry_rsi_min": random.randint(20, 40),
        "entry_rsi_max": random.randint(45, 60),
        "take_profit_pct": random.uniform(0.005, 0.025),
        "hold_ticks": random.randint(30, 300),
        "exit_sma_mult": random.uniform(1.002, 1.005),
        "position_size": random.uniform(0.2, 0.4),
    }


def run_fast_monte_carlo(models: int, data_days: int, output_tag: str = "1m"):
    """Run fast Monte Carlo sweep. Records ALL results for ML."""
    print(f"\n{'='*70}")
    print(f"FAST MONTE CARLO: {models:,} models × {len(STRATEGIES)} strategies")
    print(f"Data: {data_days} days MOEX")
    print(f"{'='*70}\n")
    
    # 1. Load data
    print("[1/4] Loading data...")
    data_raw = load_all_tickers(days=data_days)
    train, val, test = split_data(data_raw)
    
    # 2. Pre-compute indicators for each ticker × each split
    print("\n[2/4] Pre-computing indicators...")
    t0 = time.time()
    
    ind_train = {}
    ind_val = {}
    ind_test = {}
    closes_train = {}
    closes_val = {}
    closes_test = {}
    
    for ticker, candles in train.items():
        if not candles:
            continue
        closes = np.array([c["close"] for c in candles])
        highs = np.array([c["high"] for c in candles])
        lows = np.array([c["low"] for c in candles])
        vols = np.array([c["volume"] for c in candles], dtype=float)
        
        ind_train[ticker] = precompute_indicators(closes, highs, lows, vols)
        closes_train[ticker] = closes
    
    for ticker, candles in val.items():
        if not candles:
            continue
        closes = np.array([c["close"] for c in candles])
        highs = np.array([c["high"] for c in candles])
        lows = np.array([c["low"] for c in candles])
        vols = np.array([c["volume"] for c in candles], dtype=float)
        
        ind_val[ticker] = precompute_indicators(closes, highs, lows, vols)
        closes_val[ticker] = closes
    
    for ticker, candles in test.items():
        if not candles:
            continue
        closes = np.array([c["close"] for c in candles])
        highs = np.array([c["high"] for c in candles])
        lows = np.array([c["low"] for c in candles])
        vols = np.array([c["volume"] for c in candles], dtype=float)
        
        ind_test[ticker] = precompute_indicators(closes, highs, lows, vols)
        closes_test[ticker] = closes
    
    t1 = time.time()
    print(f"  Pre-computed indicators for {len(ind_train)} tickers in {t1-t0:.1f}s")
    
    # 3. Run sweep
    print(f"\n[3/4] Running {models:,} models...")
    
    all_results = []
    profitable_count = 0
    start_time = time.time()
    
    tickers = list(ind_train.keys())
    
    for i in range(models):
        if (i + 1) % 10000 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (models - i - 1) / rate
            print(f"  [{i+1:,}/{models:,}] profitable: {profitable_count:,} | "
                  f"rate: {rate:.0f}/sec | ETA: {eta/60:.0f}min")
        
        strategy = random.choice(STRATEGIES)
        params = random_params(strategy)
        
        # Evaluate on train + val + test (aggregate across tickers)
        train_pnl = 0
        train_trades = 0
        val_pnl = 0
        val_trades = 0
        test_pnl = 0
        test_trades = 0
        total_wins = 0
        total_sortino = 0
        ticker_count = 0
        
        for ticker in tickers:
            if ticker not in ind_val or ticker not in ind_test:
                continue
            
            r_train = vectorized_backtest(ind_train[ticker], closes_train[ticker], strategy, params)
            r_val = vectorized_backtest(ind_val[ticker], closes_val[ticker], strategy, params)
            r_test = vectorized_backtest(ind_test[ticker], closes_test[ticker], strategy, params)
            
            train_pnl += r_train["pnl"]
            train_trades += r_train["trades"]
            val_pnl += r_val["pnl"]
            val_trades += r_val["trades"]
            test_pnl += r_test["pnl"]
            test_trades += r_test["trades"]
            total_wins += r_train["wins"]
            total_sortino += r_train["sortino"]
            ticker_count += 1
        
        win_rate = total_wins / train_trades * 100 if train_trades > 0 else 0
        avg_sortino = total_sortino / ticker_count if ticker_count > 0 else 0
        is_profitable = 1 if (val_pnl > 0 and test_pnl > 0) else 0
        
        if is_profitable:
            profitable_count += 1
        
        # Record EVERY model (for ML training)
        all_results.append({
            "strategy": strategy,
            "entry_sma_mult": params["entry_sma_mult"],
            "entry_rsi_min": params["entry_rsi_min"],
            "entry_rsi_max": params["entry_rsi_max"],
            "take_profit_pct": params["take_profit_pct"],
            "hold_ticks": params["hold_ticks"],
            "exit_sma_mult": params["exit_sma_mult"],
            "position_size": params["position_size"],
            "train_pnl": round(train_pnl, 2),
            "val_pnl": round(val_pnl, 2),
            "test_pnl": round(test_pnl, 2),
            "trades": train_trades,
            "win_rate": round(win_rate, 1),
            "sortino": round(avg_sortino, 2),
            "profitable": is_profitable,
        })
    
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"COMPLETE: {models:,} models in {elapsed/60:.1f} min ({models/elapsed:.0f} models/sec)")
    print(f"Profitable: {profitable_count:,} ({profitable_count/models*100:.1f}%)")
    print(f"{'='*70}\n")
    
    # 4. Save results
    print("[4/4] Saving results...")
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save as JSON (can convert to parquet later)
    output_file = os.path.join(results_dir, f"all_models_{output_tag}.json")
    with open(output_file, "w") as f:
        json.dump({
            "tag": output_tag,
            "models": models,
            "data_days": data_days,
            "strategies": STRATEGIES,
            "profitable_count": profitable_count,
            "total_time_sec": elapsed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "models_data": all_results,
        }, f)
    print(f"  Saved to: {output_file}")
    
    # Also save profitable-only (for quick deployment)
    profitable_models = [r for r in all_results if r["profitable"] == 1]
    profitable_models.sort(key=lambda x: -(x["val_pnl"] + x["test_pnl"]))
    
    prof_file = os.path.join(results_dir, f"profitable_{output_tag}.json")
    with open(prof_file, "w") as f:
        json.dump(profitable_models, f, indent=2)
    print(f"  Profitable ({len(profitable_models)}): {prof_file}")
    
    # Print top 10
    if profitable_models:
        print(f"\nTop 10 profitable models:")
        for i, m in enumerate(profitable_models[:10], 1):
            print(f"  {i:2d}. {m['strategy']:22s} val={m['val_pnl']:+8.0f} test={m['test_pnl']:+8.0f} "
                  f"trades={m['trades']:4d} win={m['win_rate']:.0f}%")
    
    # Strategy breakdown
    print(f"\nStrategy breakdown:")
    strat_stats = {}
    for r in all_results:
        s = r["strategy"]
        if s not in strat_stats:
            strat_stats[s] = {"total": 0, "profitable": 0}
        strat_stats[s]["total"] += 1
        if r["profitable"]:
            strat_stats[s]["profitable"] += 1
    for s, stats in sorted(strat_stats.items(), key=lambda x: -x[1]["profitable"]):
        pct = stats["profitable"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {s:25s}: {stats['profitable']:4d}/{stats['total']:5d} profitable ({pct:.1f}%)")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=int, default=1000000, help="Number of models to evaluate")
    parser.add_argument("--data-days", type=int, default=180)
    parser.add_argument("--tag", type=str, default="1m")
    args = parser.parse_args()
    
    run_fast_monte_carlo(args.models, args.data_days, args.tag)
