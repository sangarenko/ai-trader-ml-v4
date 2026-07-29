#!/usr/bin/env python3
"""Evolution engine — genetic algorithm for SniperTrend strategy optimization.

Architecture:
  1. Population: 200 models (including V2 anchor)
  2. Generations: 50
  3. Fitness: Sortino ratio on validation data
  4. Selection: Tournament k=3
  5. Crossover: Uniform (0.7 probability)
  6. Mutation: Gaussian (0.05 rate)
  7. Elitism: Top 5% (10 models) preserved

Usage:
  python3 evolution.py                          # full run (50 min)
  python3 evolution.py --models 200 --generations 50
  python3 evolution.py --batch 50 --generation 1  # batch mode
  python3 evolution.py --resume results/checkpoint_gen_25.json
"""
import json
import os
import sys
import random
import argparse
import time
from typing import List, Tuple, Dict

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from params import (
    StrategyParams, V2_PARAMS, random_params, mutate, crossover, tournament_select
)
from backtest import backtest, BacktestResult
from data_loader import load_all_tickers, split_data

# V2 risk filters (mirror live RiskManager) — applied to ALL backtests
# so evolution optimizes under the same constraints as live trading
V2_RISK_FILTERS = {
    'commFilterMult': 1.2,     # skip entry if expected_move < comm × 1.2
    'cooldownTicks': 12,        # wait 12 candles (1 hour) after close
    'maxTradesPerHour': 10,     # max 10 trades per hour
}


RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def evaluate_model(params: StrategyParams, data: Dict[str, List]) -> Tuple[float, BacktestResult, BacktestResult]:
    """Evaluate model on train and validation data.

    Returns: (fitness, train_result, val_result)
    """
    # Backtest on train
    train_results = []
    for ticker, candles in data['train'].items():
        r = backtest(params, candles, ticker, risk_filters=V2_RISK_FILTERS)
        train_results.append(r)

    # Backtest on validation
    val_results = []
    for ticker, candles in data['val'].items():
        r = backtest(params, candles, ticker, risk_filters=V2_RISK_FILTERS)
        val_results.append(r)

    # Aggregate
    train_pnl = sum(r.pnl for r in train_results)
    train_trades = sum(r.trades for r in train_results)
    train_sortinos = [r.sortino for r in train_results if r.trades > 0]
    train_sortino = sum(train_sortinos) / len(train_sortinos) if train_sortinos else 0

    val_pnl = sum(r.pnl for r in val_results)
    val_trades = sum(r.trades for r in val_results)
    val_sortinos = [r.sortino for r in val_results if r.trades > 0]
    val_sortino = sum(val_sortinos) / len(val_sortinos) if val_sortinos else 0

    # Fitness = Sortino on val - overfit penalty - trade penalty
    overfit = abs(train_pnl - val_pnl) / max(abs(train_pnl), 1)
    trade_penalty = max(0, 30 - val_trades) * 0.1  # penalize <30 trades
    fitness = val_sortino - 0.5 * overfit - trade_penalty

    train_result = BacktestResult(
        pnl=train_pnl, trades=train_trades,
        wins=sum(r.wins for r in train_results),
        sortino=train_sortino,
        max_drawdown=max((r.max_drawdown for r in train_results), default=0),
        avg_hold_ticks=sum(r.avg_hold_ticks for r in train_results) / len(train_results) if train_results else 0,
    )
    val_result = BacktestResult(
        pnl=val_pnl, trades=val_trades,
        wins=sum(r.wins for r in val_results),
        sortino=val_sortino,
        max_drawdown=max((r.max_drawdown for r in val_results), default=0),
        avg_hold_ticks=sum(r.avg_hold_ticks for r in val_results) / len(val_results) if val_results else 0,
    )

    return fitness, train_result, val_result


def run_evolution(models: int = 200, generations: int = 50, data_days: int = 7,
                  batch_size: int = None, resume_from: str = None, seed: int = None,
                  run_id: int = 0):
    """Run genetic algorithm evolution.

    Args:
        seed: Random seed for reproducibility (None = random)
        run_id: Run identifier for multi-run mode (0, 1, 2...)
    """
    if seed is not None:
        random.seed(seed)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Load data
    print('=' * 70)
    print(f'EVOLUTION: {models} models × {generations} generations')
    print('=' * 70)
    print()
    print('[1/4] Loading candle data...')
    data_raw = load_all_tickers(days=data_days)
    train, val, test = split_data(data_raw)
    data = {'train': train, 'val': val, 'test': test}
    print(f'  Train: {sum(len(c) for c in train.values())} candles')
    print(f'  Val:   {sum(len(c) for c in val.values())} candles')
    print(f'  Test:  {sum(len(c) for c in test.values())} candles')
    print()

    # 2. Initialize population
    print('[2/4] Initializing population...')
    if resume_from:
        with open(resume_from) as f:
            checkpoint = json.load(f)
        population = [StrategyParams(**p) for p in checkpoint['population']]
        start_gen = checkpoint['generation'] + 1
        print(f'  Resumed from gen {start_gen} ({len(population)} models)')
    else:
        population = [V2_PARAMS]  # V2 as anchor
        for _ in range(models - 1):
            population.append(random_params())
        start_gen = 0
        print(f'  {len(population)} models (V2 anchor + {models-1} random)')

    # Evaluate V2 baseline
    print()
    print('  Evaluating V2 anchor...')
    v2_fitness, v2_train, v2_val = evaluate_model(V2_PARAMS, data)
    print(f'  V2: fitness={v2_fitness:.3f}, val_pnl={v2_val.pnl:+.0f}₽, val_trades={v2_val.trades}, val_sortino={v2_val.sortino:.2f}')
    print()

    # 3. Evolution loop
    print('[3/4] Running evolution...')
    print()
    log = {'v2_baseline': {'fitness': v2_fitness, 'val_pnl': v2_val.pnl, 'val_trades': v2_val.trades},
           'generations': []}

    for gen in range(start_gen, generations):
        gen_start = time.time()
        print(f'--- Generation {gen+1}/{generations} ---')

        # Evaluate population
        fitnesses = []
        for i, params in enumerate(population):
            fitness, train_r, val_r = evaluate_model(params, data)
            fitnesses.append(fitness)
            if (i + 1) % 50 == 0:
                print(f'  Evaluated {i+1}/{len(population)}...')

        # Sort by fitness
        sorted_pop = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)
        best_params, best_fitness = sorted_pop[0]
        avg_fitness = sum(fitnesses) / len(fitnesses)

        gen_time = time.time() - gen_start
        print(f'  Best: fitness={best_fitness:.3f}')
        print(f'  Avg:  fitness={avg_fitness:.3f}')
        print(f'  V2:   fitness={v2_fitness:.3f}')
        print(f'  Time: {gen_time:.1f}s')
        print()

        # Log
        log['generations'].append({
            'generation': gen + 1,
            'best_fitness': best_fitness,
            'avg_fitness': avg_fitness,
            'best_params': best_params.to_dict(),
            'time_sec': gen_time,
        })

        # Checkpoint every 5 generations
        if (gen + 1) % 5 == 0:
            checkpoint = {
                'generation': gen,
                'population': [p.to_dict() for p in population],
                'fitnesses': fitnesses,
            }
            with open(f'{RESULTS_DIR}/checkpoint_gen_{gen+1}.json', 'w') as f:
                json.dump(checkpoint, f, indent=2)

        # Batch mode — stop after 1 generation if batch_size specified
        if batch_size and gen + 1 >= batch_size:
            print(f'Batch mode: stopping after {batch_size} generation(s)')
            break

        # Create next generation
        new_population = []

        # Elitism: top 5% (10 models)
        elite_count = max(1, len(population) // 20)
        for params, _ in sorted_pop[:elite_count]:
            new_population.append(params)

        # Fill rest with crossover + mutation
        while len(new_population) < models:
            parent1 = tournament_select(population, fitnesses, k=3)
            parent2 = tournament_select(population, fitnesses, k=3)
            if random.random() < 0.7:  # crossover
                child = crossover(parent1, parent2)
            else:
                child = parent1
            child = mutate(child, rate=0.05)
            new_population.append(child)

        population = new_population

    # 4. Final evaluation on test set
    print('[4/4] Final evaluation on test set (unseen data)...')
    print()

    # Get top 10 models
    final_fitnesses = []
    for params in population:
        fitness, _, _ = evaluate_model(params, data)
        final_fitnesses.append(fitness)

    sorted_final = sorted(zip(population, final_fitnesses), key=lambda x: x[1], reverse=True)
    top_models = sorted_final[:10]

    # Evaluate top models on test set
    results = []
    for i, (params, fitness) in enumerate(top_models):
        test_results = []
        for ticker, candles in data['test'].items():
            r = backtest(params, candles, ticker, risk_filters=V2_RISK_FILTERS)
            test_results.append(r)
        test_pnl = sum(r.pnl for r in test_results)
        test_trades = sum(r.trades for r in test_results)
        test_sortinos = [r.sortino for r in test_results if r.trades > 0]
        test_sortino = sum(test_sortinos) / len(test_sortinos) if test_sortinos else 0

        # V2 on test
        v2_test_results = []
        for ticker, candles in data['test'].items():
            r = backtest(V2_PARAMS, candles, risk_filters=V2_RISK_FILTERS)
            v2_test_results.append(r)
        v2_test_pnl = sum(r.pnl for r in v2_test_results)
        v2_test_trades = sum(r.trades for r in v2_test_results)

        results.append({
            'rank': i + 1,
            'params': params.to_dict(),
            'fitness': fitness,
            'test_pnl': test_pnl,
            'test_trades': test_trades,
            'test_sortino': test_sortino,
            'beats_v2_test': test_pnl > v2_test_pnl,
        })
        print(f'  #{i+1}: fitness={fitness:.3f}, test_pnl={test_pnl:+.0f}₽, test_trades={test_trades}, '
              f'sortino={test_sortino:.2f} {"✓ beats V2" if test_pnl > v2_test_pnl else "✗ worse than V2"}')

    print()
    print(f'V2 on test: pnl={v2_test_pnl:+.0f}₽, trades={v2_test_trades}')

    # Save results (with run_id for multi-run mode)
    suffix = f'_run{run_id}' if run_id > 0 else ''
    with open(f'{RESULTS_DIR}/top_models{suffix}.json', 'w') as f:
        json.dump(results, f, indent=2)

    with open(f'{RESULTS_DIR}/evolution_log{suffix}.json', 'w') as f:
        json.dump(log, f, indent=2)

    # V2 comparison
    v2_comparison = {
        'run_id': run_id,
        'seed': seed,
        'v2_test_pnl': v2_test_pnl,
        'v2_test_trades': v2_test_trades,
        'best_model_test_pnl': results[0]['test_pnl'] if results else 0,
        'best_model_beats_v2': results[0]['beats_v2_test'] if results else False,
    }
    with open(f'{RESULTS_DIR}/v2_comparison{suffix}.json', 'w') as f:
        json.dump(v2_comparison, f, indent=2)

    print()
    print('=' * 70)
    print(f'EVOLUTION COMPLETE (run {run_id})')
    print('=' * 70)
    print(f'Results saved to: {RESULTS_DIR}/')
    print(f'  - top_models{suffix}.json (top 10)')
    print(f'  - evolution_log{suffix}.json (full history)')
    print(f'  - v2_comparison{suffix}.json')
    if results and results[0]['beats_v2_test']:
        print(f'  ✓ Best model BEATS V2 on test: {results[0]["test_pnl"]:+.0f}₽ vs V2 {v2_test_pnl:+.0f}₽')
    else:
        print(f'  ✗ No model beats V2 on test — V2 is optimal')

    return results


def run_multi(models: int = 200, generations: int = 1000, data_days: int = 7,
              num_runs: int = 2):
    """Run multiple independent evolution runs, pick best across all."""
    print('=' * 70)
    print(f'MULTI-RUN EVOLUTION: {num_runs} independent runs')
    print(f'  {models} models × {generations} generations each')
    print(f'  Data: {data_days} days')
    print('=' * 70)

    all_results = []
    for run_id in range(1, num_runs + 1):
        print(f'\n{"#" * 70}')
        print(f'# RUN {run_id}/{num_runs}')
        print(f'{"#" * 70}\n')

        results = run_evolution(
            models=models,
            generations=generations,
            data_days=data_days,
            seed=42 + run_id,  # different seed per run
            run_id=run_id,
        )
        all_results.extend([(run_id, r) for r in results])

    # Pick best across all runs
    print('\n' + '=' * 70)
    print('MULTI-RUN SUMMARY — Best models across all runs')
    print('=' * 70)

    # Sort all results by fitness
    all_results.sort(key=lambda x: x[1]['fitness'], reverse=True)

    top_10 = all_results[:10]
    for i, (run_id, r) in enumerate(top_10):
        print(f'  #{i+1} (run {run_id}): fitness={r["fitness"]:.3f}, test_pnl={r["test_pnl"]:+.0f}₽, '
              f'trades={r["test_trades"]}, beats_v2={r["beats_v2_test"]}')

    # Save combined results
    combined = [
        {'rank': i+1, 'run_id': run_id, **r}
        for i, (run_id, r) in enumerate(top_10)
    ]
    with open(f'{RESULTS_DIR}/best_overall.json', 'w') as f:
        json.dump(combined, f, indent=2)

    print(f'\nBest overall saved to: {RESULTS_DIR}/best_overall.json')
    return combined


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evolve SniperTrend strategies')
    parser.add_argument('--models', type=int, default=200, help='Population size (default: 200)')
    parser.add_argument('--generations', type=int, default=1000, help='Generations (default: 1000)')
    parser.add_argument('--data-days', type=int, default=7, help='Days of candle data (default: 7)')
    parser.add_argument('--batch', type=int, help='Batch mode: stop after N generations')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint file')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--runs', type=int, default=1, help='Number of independent runs (default: 1)')
    parser.add_argument('--run-id', type=int, default=0, help='Run identifier (default: 0)')
    args = parser.parse_args()

    if args.runs > 1:
        # Multi-run mode: run N independent evolutions, pick best
        run_multi(
            models=args.models,
            generations=args.generations,
            data_days=args.data_days,
            num_runs=args.runs,
        )
    else:
        # Single run
        run_evolution(
            models=args.models,
            generations=args.generations,
            data_days=args.data_days,
            batch_size=args.batch,
            resume_from=args.resume,
            seed=args.seed,
            run_id=args.run_id,
        )
