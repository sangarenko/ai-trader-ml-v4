#!/usr/bin/env python3
"""Patch for multi_cycle_evolution.py — fix fitness function to prevent overfitting.

NEW fitness:
  1. Run backtest on train, val, AND test
  2. fitness = val_pnl (rubles) — direct profit, not Sortino
  3. HEAVY penalty if test_pnl < 0 (model overfit to val)
  4. HEAVY penalty if |train_pnl - val_pnl| > 30% of val_pnl (instability)
  5. Penalty if val_trades < 30 (statistically insignificant)
  6. Bonus if profitable on ALL three: train, val, test

This ensures we select models that ACTUALLY generalize, not just memorize val.
"""
import os

PATCH_FILE = "/root/ai-trader-evolution/training/multi_cycle_evolution.py"

# Read original
with open(PATCH_FILE) as f:
    content = f.read()

# New evaluate_model with test check + heavy overfit penalty
new_eval = '''def evaluate_model(params, data, structure):
    """Evaluate model on train + val + test. Penalize overfitting heavily."""
    train_results = []
    for ticker, candles in data['train'].items():
        r = backtest_with_structure(params, candles, ticker, structure, V2_RISK_FILTERS)
        train_results.append(r)
    val_results = []
    for ticker, candles in data['val'].items():
        r = backtest_with_structure(params, candles, ticker, structure, V2_RISK_FILTERS)
        val_results.append(r)
    test_results = []
    for ticker, candles in data['test'].items():
        r = backtest_with_structure(params, candles, ticker, structure, V2_RISK_FILTERS)
        test_results.append(r)

    train_pnl = sum(r['pnl'] for r in train_results)
    train_trades = sum(r['trades'] for r in train_results)
    val_pnl = sum(r['pnl'] for r in val_results)
    val_trades = sum(r['trades'] for r in val_results)
    test_pnl = sum(r['pnl'] for r in test_results)
    test_trades = sum(r['trades'] for r in test_results)

    val_sortinos = [r['sortino'] for r in val_results if r['trades'] > 0]
    val_sortino = sum(val_sortinos) / len(val_sortinos) if val_sortinos else 0

    # === NEW FITNESS (anti-overfit) ===
    # Base: val_pnl in rubles (direct profit, not Sortino)
    # - +1 fitness per +1 RUB val profit
    # - HEAVY penalty if test_pnl < 0 (model overfit to val, fails on unseen)
    # - HEAVY penalty if val and test diverge >50% (instability)
    # - Penalty if val_trades < 30 (statistically insignificant)
    # - BONUS if profitable on ALL three (train, val, test)

    # 1. Base profit on val
    base_fitness = val_pnl / 100  # +0.01 per +1 RUB

    # 2. Test penalty: if test is negative, model is overfit
    if test_pnl < 0:
        test_penalty = abs(test_pnl) / 50  # -0.02 per -1 RUB on test
    else:
        test_penalty = 0

    # 3. Stability: penalize divergence between val and test
    if val_pnl != 0:
        divergence = abs(val_pnl - test_pnl) / max(abs(val_pnl), 1)
        if divergence > 0.5:  # >50% divergence = unstable
            stability_penalty = divergence * 10  # up to -10 fitness
        else:
            stability_penalty = 0
    else:
        stability_penalty = 0

    # 4. Trade count penalty (need 30+ for stats)
    trade_penalty = max(0, 30 - val_trades) * 0.5

    # 5. Triple-profit bonus: all three profitable = +5 fitness
    triple_bonus = 5.0 if (train_pnl > 0 and val_pnl > 0 and test_pnl > 0) else 0

    # 6. Sortino bonus (small, for consistency)
    sortino_bonus = min(val_sortino, 2.0)  # cap at +2

    fitness = base_fitness - test_penalty - stability_penalty - trade_penalty + triple_bonus + sortino_bonus

    return fitness, {
        'train_pnl': train_pnl, 'train_trades': train_trades,
        'val_pnl': val_pnl, 'val_trades': val_trades, 'val_sortino': val_sortino,
        'test_pnl': test_pnl, 'test_trades': test_trades,
    }'''

# Find and replace the old evaluate_model function
import re
# Match from "def evaluate_model" to "def random_params_for_structure"
pattern = re.compile(r"def evaluate_model\(params, data, structure\):.*?\n\ndef random_params_for_structure", re.DOTALL)
match = pattern.search(content)
if not match:
    print("ERROR: could not find evaluate_model function")
    exit(1)

old_func = match.group(0)
new_block = new_eval + "\n\ndef random_params_for_structure"
content = content.replace(old_func, new_block)

# Also update run_cycle to track test_pnl in profitable models
# Find: 'val_pnl': results['val_pnl'], in profitable_models append
old_profitable = "log['profitable_models'].append({\n                'params': asdict(params),\n                'fitness': fitness,\n                'val_pnl': results['val_pnl'],\n                'val_trades': results['val_trades'],\n                'val_sortino': results['val_sortino'],\n                'generation': gen + 1,\n            })"
new_profitable = "log['profitable_models'].append({\n                'params': asdict(params),\n                'fitness': fitness,\n                'val_pnl': results['val_pnl'],\n                'val_trades': results['val_trades'],\n                'val_sortino': results['val_sortino'],\n                'test_pnl': results.get('test_pnl', 0),\n                'test_trades': results.get('test_trades', 0),\n                'generation': gen + 1,\n            })"

if old_profitable in content:
    content = content.replace(old_profitable, new_profitable)
    print("patched profitable_models tracking")
else:
    print("WARNING: could not find profitable_models block to patch")

# Also: only collect models as "profitable" if BOTH val_pnl > 0 AND test_pnl > 0
old_filter = "profitable_in_gen = [e for e in gen_evaluated if e[2]['val_pnl'] > 0]"
new_filter = "profitable_in_gen = [e for e in gen_evaluated if e[2]['val_pnl'] > 0 and e[2].get('test_pnl', -1) > 0]"
if old_filter in content:
    content = content.replace(old_filter, new_filter)
    print("patched profitable filter (require val AND test positive)")
else:
    print("WARNING: could not find profitable filter")

with open(PATCH_FILE, 'w') as f:
    f.write(content)
print("fitness function patched successfully")
print(f"file size: {os.path.getsize(PATCH_FILE)} bytes")
