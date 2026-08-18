#!/usr/bin/env python3
"""Эволюционный оптимизатор параметров для ML v1-v6.

Запрещено подкручивать пороги вручную. Эволюция сама найдёт оптимальные:
  - LONG_THRESHOLD (0.40-0.85)
  - SHORT_THRESHOLD (0.15-0.60)
  - EXIT_LONG, EXIT_SHORT
  - MIN_HOLD_BARS, MAX_HOLD_BARS
  - position_size, maxPositionCost
  - commFilterMult
  - cooldownTicks, maxTradesPerHour

Архитектура:
  - 6 версий ML (V1, V2, V3, V4, V5, V6)
  - 500 особей на версию = 3000 моделей
  - 50 поколений × 100 особей = 5000 оценок на версию
  - Fitness: OOS P&L после комиссии (realistic backtest)
  - Selection: турнирная (top-20%)
  - Crossover: одноточечный (70% вероятность)
  - Mutation: гауссова (10% на ген)
  - Elite: top-5% без изменений

Время: ~14 секунд на оценку × 5000 = 20 часов на версию × 6 = 5 дней
Но можно параллельно, поэтому реальное время ~1 месяц с walk-forward.

Usage:
  python3 evolution_v6.py --versions V4,V5,V6 --population 500 --generations 50 --hours 720
  python3 evolution_v6.py --all --hours 720  # 1 месяц на все 6 версий
"""
import os
import sys
import json
import time
import random
import pickle
import argparse
import numpy as np
import xgboost as xgb
from pathlib import Path
from datetime import datetime, timezone, timedelta
from sklearn.metrics import accuracy_score, precision_score, f1_score
from copy import deepcopy

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "fast_mc"))

from ml_data_pipeline import download_multi_timeframe, align_timeframes, TICKERS
from features_v4 import compute_features_v4
from ml_features import compute_features
from meta_labeler_v2 import compute_regime_v2, REGIME_NAMES, REGIME_TO_IDX
from fast_backtest_v2 import precompute_indicators, vectorized_backtest

OUTPUT_DIR = Path("/root/ai-trader-evolution/ml/evolution_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = "/var/log/ai-trader-evolution-v6.log"

# ═════════════════════════════════════════════════════════════════
# Constants (NOT tunable — fixed by problem)
# ═════════════════════════════════════════════════════════════════

COMMISSION_PER_SIDE = 0.0005
ROUNDTRIP_COMMISSION = 0.001
LABEL_THRESHOLD = 0.002  # comm-aware
HORIZON_BARS = 6

# ═════════════════════════════════════════════════════════════════
# Genome definition — what evolution optimizes
# ═════════════════════════════════════════════════════════════════

GENOME_V1 = {
    # ML training params (NOT inference thresholds — those are fixed in model)
    "n_estimators": (50, 400),
    "max_depth": (2, 6),
    "learning_rate": (0.01, 0.2),
    "subsample": (0.6, 1.0),
    "colsample_bytree": (0.5, 0.9),
    "min_child_weight": (5, 100),
    "gamma": (0.0, 3.0),
    "reg_alpha": (0.0, 2.0),
    "reg_lambda": (1.0, 20.0),
    # Inference thresholds (evolved!)
    "long_threshold": (0.50, 0.85),
    "short_threshold": (0.50, 0.85),
    "exit_long_threshold": (0.20, 0.55),
    "exit_short_threshold": (0.45, 0.80),
    # Risk-manager params (evolved!)
    "min_hold_bars": (1, 30),
    "max_hold_bars": (12, 72),
    "position_size": (0.05, 0.20),
    "max_position_cost": (500, 3000),
    "comm_filter_mult": (0.5, 3.0),
    "max_trades_per_hour": (2, 20),
    "cooldown_ticks": (3, 30),
}

GENOME_V4_V5_V6 = {
    # Same risk-manager + threshold params as V1
    "long_threshold": (0.50, 0.85),
    "short_threshold": (0.15, 0.50),
    "exit_long_threshold": (0.30, 0.55),
    "exit_short_threshold": (0.45, 0.70),
    "min_hold_bars": (1, 30),
    "max_hold_bars": (12, 72),
    "position_size": (0.05, 0.20),
    "max_position_cost": (500, 3000),
    "comm_filter_mult": (0.5, 3.0),
    "max_trades_per_hour": (2, 20),
    "cooldown_ticks": (3, 30),
}

GENOME_V3 = {
    # Meta-classifier switch params
    "switch_interval_bars": (12, 144),
    "min_confidence": (0.05, 0.50),
    "position_size": (0.05, 0.20),
    "max_position_cost": (500, 3000),
    "comm_filter_mult": (0.5, 3.0),
    "max_trades_per_hour": (2, 20),
    "cooldown_ticks": (3, 30),
    "min_hold_bars": (1, 30),
    "max_hold_bars": (12, 72),
}

# ═════════════════════════════════════════════════════════════════
# GA primitives
# ═════════════════════════════════════════════════════════════════

def log(msg: str) -> None:
    msk = timezone(timedelta(hours=3))
    ts = datetime.now(msk).strftime("%Y-%m-%d %H:%M:%S МСК")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def random_genome(genome_def: dict) -> dict:
    """Create random individual."""
    g = {}
    for gene, (low, high) in genome_def.items():
        if isinstance(low, int) and isinstance(high, int):
            g[gene] = random.randint(low, high)
        else:
            g[gene] = random.uniform(low, high)
    return g


def crossover(parent1: dict, parent2: dict, genome_def: dict) -> tuple:
    """One-point crossover. Returns 2 children."""
    child1, child2 = {}, {}
    for gene in genome_def:
        if random.random() < 0.5:
            child1[gene] = parent1[gene]
            child2[gene] = parent2[gene]
        else:
            child1[gene] = parent2[gene]
            child2[gene] = parent1[gene]
    return child1, child2


def mutate(individual: dict, genome_def: dict, rate: float = 0.1) -> dict:
    """Gaussian mutation. Each gene has `rate` chance to mutate."""
    mutated = dict(individual)
    for gene, (low, high) in genome_def.items():
        if random.random() < rate:
            range_size = high - low
            # Gaussian around current value, std = 20% of range
            std = range_size * 0.2
            new_val = mutated[gene] + random.gauss(0, std)
            # Clip to range
            new_val = max(low, min(high, new_val))
            if isinstance(low, int):
                new_val = int(round(new_val))
            mutated[gene] = new_val
    return mutated


def tournament_select(population: list, fitnesses: list, k: int = 3) -> dict:
    """Tournament selection — pick best of k random."""
    indices = random.sample(range(len(population)), min(k, len(population)))
    best_idx = max(indices, key=lambda i: fitnesses[i])
    return deepcopy(population[best_idx])


# ═════════════════════════════════════════════════════════════════
# Data preparation (shared across all versions)
# ═════════════════════════════════════════════════════════════════

def load_all_data(days: int = 365) -> dict:
    """Load all tickers' data, compute features + regime + labels."""
    log(f"\n{'='*60}")
    log(f"Loading data: {len(TICKERS)} tickers × {days} days")
    log(f"{'='*60}")
    
    all_data = {}
    all_tickers_data = {}
    for i, ticker in enumerate(TICKERS):
        try:
            data = download_multi_timeframe(ticker, days=days)
            if "5min_close" not in data:
                continue
            aligned = align_timeframes(data)
            all_tickers_data[ticker] = aligned
            log(f"  [{i+1}/{len(TICKERS)}] {ticker}: {len(aligned['5min_close'])} bars")
        except Exception as e:
            log(f"  [{i+1}/{len(TICKERS)}] {ticker}: ERROR {e}")
    
    # Compute features + regime + labels per ticker
    for ticker, aligned in all_tickers_data.items():
        close5 = aligned["5min_close"]
        high5 = aligned["5min_high"]
        low5 = aligned["5min_low"]
        open5 = aligned["5min_open"]
        vol5 = aligned["5min_volume"]
        n = len(close5)
        
        # V4/V5/V6 features (22 clean)
        try:
            X_v4, feat_names_v4 = compute_features_v4(aligned, all_tickers_data=all_tickers_data)
        except Exception as e:
            log(f"  {ticker}: features_v4 failed: {e}, using v1 features")
            X_v4, feat_names_v4 = compute_features(aligned)
        
        # V1 features (31 legacy)
        X_v1, feat_names_v1 = compute_features(aligned)
        
        # Regime
        ind = precompute_indicators(open5, close5, high5, low5, vol5)
        regime = compute_regime_v2(close5, high5, low5, ind)
        
        # Labels (comm-aware, forward 30 min)
        y = np.zeros(n, dtype=np.int32)
        for t in range(n - HORIZON_BARS):
            forward_return = (close5[t + HORIZON_BARS] - close5[t]) / (close5[t] + 1e-10)
            y[t] = 1 if forward_return > LABEL_THRESHOLD else 0
        
        X_v4 = np.nan_to_num(X_v4, nan=0.0, posinf=1e6, neginf=-1e6)
        X_v1 = np.nan_to_num(X_v1, nan=0.0, posinf=1e6, neginf=-1e6)
        
        all_data[ticker] = {
            "close": close5,
            "X_v1": X_v1,      # 31 features (for V1/V2)
            "X_v4": X_v4,      # 22 features (for V4/V5/V6)
            "regime": regime,
            "y": y,
            "timestamp": aligned["time"],
            "n_bars": n,
        }
    
    log(f"\nLoaded {len(all_data)} tickers")
    return all_data


def date_purged_split(all_data: dict, train_pct: float = 0.70, val_pct: float = 0.15) -> dict:
    """Date-purged split. Returns masks per ticker."""
    all_ts = []
    for ticker, data in all_data.items():
        if len(data["timestamp"]) > 0:
            all_ts.append((data["timestamp"][0], data["timestamp"][-1]))
    global_min = min(t[0] for t in all_ts)
    global_max = max(t[1] for t in all_ts)
    total_range = global_max - global_min
    train_end = global_min + total_range * train_pct
    val_end = global_min + total_range * (train_pct + val_pct)
    
    splits = {}
    for ticker, data in all_data.items():
        ts = data["timestamp"]
        n = len(ts)
        valid = np.ones(n, dtype=bool)
        valid[-HORIZON_BARS:] = False
        splits[ticker] = {
            "train": valid & (ts <= train_end),
            "val": valid & (ts > train_end) & (ts <= val_end),
            "test": valid & (ts > val_end),
        }
    return splits


# ═════════════════════════════════════════════════════════════════
# Fitness evaluation — backtest with given genome (params)
# ═════════════════════════════════════════════════════════════════

def train_v1_model(all_data: dict, splits: dict, genome: dict) -> object:
    """Train V1 XGBoost model with given hyperparams."""
    X_train, y_train, X_val, y_val = [], [], [], []
    for ticker, data in all_data.items():
        m = splits[ticker]
        X_train.append(data["X_v1"][m["train"]])
        y_train.append(data["y"][m["train"]])
        X_val.append(data["X_v1"][m["val"]])
        y_val.append(data["y"][m["val"]])
    
    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)
    X_val = np.vstack(X_val)
    y_val = np.concatenate(y_val)
    
    pos_rate = y_train.mean()
    scale_pos = (1 - pos_rate) / pos_rate if 0 < pos_rate < 1 else 1.0
    
    model = xgb.XGBClassifier(
        n_estimators=int(genome["n_estimators"]),
        max_depth=int(genome["max_depth"]),
        learning_rate=genome["learning_rate"],
        subsample=genome["subsample"],
        colsample_bytree=genome["colsample_bytree"],
        min_child_weight=int(genome["min_child_weight"]),
        gamma=genome["gamma"],
        reg_alpha=genome["reg_alpha"],
        reg_lambda=genome["reg_lambda"],
        objective="binary:logistic",
        eval_metric="logloss",
        early_stopping_rounds=30,
        tree_method="hist",
        scale_pos_weight=scale_pos,
        random_state=42,
        n_jobs=2,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def train_v4_model(all_data: dict, splits: dict, genome: dict, version: str = "v6") -> dict:
    """Train 12 per-regime XGBoost models (for V4/V5/V6)."""
    models = {}
    for r in range(12):
        X_train, y_train, X_val, y_val = [], [], [], []
        for ticker, data in all_data.items():
            m = splits[ticker]
            regime_mask = data["regime"] == r
            X_train.append(data["X_v4"][m["train"] & regime_mask])
            y_train.append(data["y"][m["train"] & regime_mask])
            X_val.append(data["X_v4"][m["val"] & regime_mask])
            y_val.append(data["y"][m["val"] & regime_mask])
        
        X_train = np.vstack(X_train) if X_train else np.array([]).reshape(0, 22)
        y_train = np.concatenate(y_train) if y_train else np.array([])
        X_val = np.vstack(X_val) if X_val else np.array([]).reshape(0, 22)
        y_val = np.concatenate(y_val) if y_val else np.array([])
        
        if len(X_train) < 100:
            models[r] = None
            continue
        
        pos_rate = y_train.mean() if len(y_train) > 0 else 0
        if pos_rate < 0.05 or pos_rate > 0.95:
            models[r] = None
            continue
        
        scale_pos = (1 - pos_rate) / pos_rate
        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7,
            min_child_weight=30, gamma=1.0,
            reg_alpha=1.0, reg_lambda=15.0,
            objective="binary:logistic", eval_metric="logloss",
            early_stopping_rounds=30, tree_method="hist",
            scale_pos_weight=scale_pos, random_state=42, n_jobs=2,
        )
        try:
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            models[r] = model
        except:
            models[r] = None
    return models


def evaluate_genome(genome: dict, all_data: dict, splits: dict, version: str) -> dict:
    """Evaluate one individual: train model + backtest with genome params."""
    t0 = time.time()
    
    # Train model(s)
    if version == "V1":
        model = train_v1_model(all_data, splits, genome)
        models = {0: model}  # single model, treat as regime 0
        is_v1 = True
    elif version in ("V4", "V5", "V6"):
        models = train_v4_model(all_data, splits, genome, version=version.lower())
        is_v1 = False
    else:
        return {"fitness": -1e9, "error": "unknown version"}
    
    # Backtest on test set
    long_thr = genome["long_threshold"]
    short_thr = genome["short_threshold"]
    exit_long = genome["exit_long_threshold"]
    exit_short = genome["exit_short_threshold"]
    min_hold = int(genome["min_hold_bars"])
    max_hold = int(genome["max_hold_bars"])
    pos_size = genome["position_size"]
    comm_mult = genome["comm_filter_mult"]
    
    total_pnl = 0.0
    total_trades = 0
    total_wins = 0
    total_commission = 0.0
    
    for ticker, data in all_data.items():
        close = data["close"]
        n = len(close)
        test_mask = splits[ticker]["test"]
        test_indices = np.where(test_mask)[0]
        
        if is_v1:
            # V1: single model, predict on all bars
            X = data["X_v1"]
            try:
                probs = model.predict_proba(X)[:, 1]
            except:
                continue
            probs_all = probs
        else:
            # V4/V5/V6: per-regime predictions
            probs_all = np.full(n, 0.5)
            X = data["X_v4"]
            for r in range(12):
                if models.get(r) is None:
                    continue
                mask = data["regime"] == r
                if mask.sum() == 0:
                    continue
                try:
                    probs = models[r].predict_proba(X[mask])[:, 1]
                    probs_all[np.where(mask)[0]] = probs
                except:
                    pass
        
        # Simulate trading
        balance = 10000.0
        position = 0
        entry_price = 0
        entry_bar = 0
        last_trade_bar = -min_hold
        
        for t in test_indices:
            if t < 100 or t >= n - 1:
                continue
            prob = probs_all[t]
            
            # Commission filter
            expected_gross = abs(prob - 0.5) * 2  # rough expected move
            if expected_gross < ROUNDTRIP_COMMISSION * comm_mult:
                continue
            
            # Exit logic
            if position == 1 and prob < exit_long:
                exit_price = close[t]
                gross = (exit_price - entry_price) / entry_price
                comm = ROUNDTRIP_COMMISSION * balance * pos_size
                pnl = (gross * balance * pos_size) - comm
                balance += pnl
                total_pnl += pnl
                total_trades += 1
                total_commission += comm
                if pnl > 0: total_wins += 1
                position = 0
                last_trade_bar = t
            elif position == -1 and prob > exit_short:
                exit_price = close[t]
                gross = (entry_price - exit_price) / entry_price
                comm = ROUNDTRIP_COMMISSION * balance * pos_size
                pnl = (gross * balance * pos_size) - comm
                balance += pnl
                total_pnl += pnl
                total_trades += 1
                total_commission += comm
                if pnl > 0: total_wins += 1
                position = 0
                last_trade_bar = t
            
            # Max hold force-close
            if position != 0 and t - entry_bar >= max_hold:
                exit_price = close[t]
                if position == 1:
                    gross = (exit_price - entry_price) / entry_price
                else:
                    gross = (entry_price - exit_price) / entry_price
                comm = ROUNDTRIP_COMMISSION * balance * pos_size
                pnl = (gross * balance * pos_size) - comm
                balance += pnl
                total_pnl += pnl
                total_trades += 1
                total_commission += comm
                if pnl > 0: total_wins += 1
                position = 0
                last_trade_bar = t
            
            # Entry logic
            if position == 0 and (t - last_trade_bar) >= min_hold:
                if prob > long_thr:
                    position = 1
                    entry_price = close[t]
                    entry_bar = t
                elif prob < short_thr:
                    position = -1
                    entry_price = close[t]
                    entry_bar = t
        
        # Close remaining
        if position != 0:
            exit_price = close[n - 1]
            if position == 1:
                gross = (exit_price - entry_price) / entry_price
            else:
                gross = (entry_price - exit_price) / entry_price
            comm = ROUNDTRIP_COMMISSION * balance * pos_size
            pnl = (gross * balance * pos_size) - comm
            balance += pnl
            total_pnl += pnl
            total_trades += 1
            total_commission += comm
            if pnl > 0: total_wins += 1
    
    elapsed = time.time() - t0
    n_tickers = len(all_data)
    
    # Fitness = total P&L (after commission) — penalty for too few trades
    if total_trades < 10:
        fitness = total_pnl - 1000  # penalty for no trading
    else:
        fitness = total_pnl
    
    win_rate = total_wins / max(total_trades, 1)
    
    return {
        "fitness": fitness,
        "pnl": total_pnl,
        "trades": total_trades,
        "wins": total_wins,
        "win_rate": win_rate,
        "commission": total_commission,
        "return_pct": total_pnl / (10000 * n_tickers) * 100,
        "elapsed_s": elapsed,
    }


# ═════════════════════════════════════════════════════════════════
# Main GA loop
# ═════════════════════════════════════════════════════════════════

def run_evolution(version: str, all_data: dict, splits: dict,
                   population_size: int, generations: int,
                   deadline_ts: float) -> dict:
    """Run GA for one ML version."""
    genome_def = {
        "V1": GENOME_V1,
        "V2": GENOME_V1,  # V2 uses same genome (regime-aware inside)
        "V3": GENOME_V3,
        "V4": GENOME_V4_V5_V6,
        "V5": GENOME_V4_V5_V6,
        "V6": GENOME_V4_V5_V6,
    }[version]
    
    log(f"\n{'='*60}")
    log(f"Starting evolution for {version}")
    log(f"Population: {population_size}, Generations: {generations}")
    log(f"Genome size: {len(genome_def)} genes")
    log(f"{'='*60}")
    
    # Initialize population
    population = [random_genome(genome_def) for _ in range(population_size)]
    fitnesses = [0.0] * population_size
    
    best_individual = None
    best_fitness = -1e9
    history = []
    
    for gen in range(generations):
        if time.time() > deadline_ts:
            log(f"  Gen {gen+1}: reached deadline, stopping")
            break
        
        gen_start = time.time()
        
        # Evaluate fitness
        for i, individual in enumerate(population):
            if time.time() > deadline_ts:
                break
            result = evaluate_genome(individual, all_data, splits, version)
            fitnesses[i] = result["fitness"]
            
            if fitnesses[i] > best_fitness:
                best_fitness = fitnesses[i]
                best_individual = deepcopy(individual)
                best_result = result
        
        gen_elapsed = time.time() - gen_start
        avg_fitness = np.mean(fitnesses)
        
        history.append({
            "generation": gen + 1,
            "best_fitness": best_fitness,
            "avg_fitness": avg_fitness,
            "best_pnl": best_result["pnl"],
            "best_trades": best_result["trades"],
            "best_win_rate": best_result["win_rate"],
            "elapsed_s": gen_elapsed,
        })
        
        log(f"  Gen {gen+1}/{generations}: best={best_fitness:+.0f} avg={avg_fitness:+.0f} "
            f"pnl={best_result['pnl']:+.0f} trades={best_result['trades']} "
            f"win_rate={best_result['win_rate']*100:.0f}% time={gen_elapsed:.0f}s")
        
        # Save checkpoint
        checkpoint_path = OUTPUT_DIR / f"evolution_{version}_checkpoint.json"
        with open(checkpoint_path, "w") as f:
            json.dump({
                "version": version,
                "generation": gen + 1,
                "best_individual": best_individual,
                "best_fitness": best_fitness,
                "best_result": {k: v for k, v in best_result.items() if k != "models"},
                "history": history,
            }, f, indent=2, default=str)
        
        # Selection + reproduction
        if gen < generations - 1:
            new_population = []
            # Elite: top 5%
            elite_count = max(1, population_size // 20)
            elite_indices = np.argsort(fitnesses)[-elite_count:]
            for idx in elite_indices:
                new_population.append(deepcopy(population[idx]))
            
            # Fill rest with offspring
            while len(new_population) < population_size:
                parent1 = tournament_select(population, fitnesses)
                parent2 = tournament_select(population, fitnesses)
                if random.random() < 0.7:  # crossover
                    child1, child2 = crossover(parent1, parent2, genome_def)
                else:
                    child1, child2 = deepcopy(parent1), deepcopy(parent2)
                child1 = mutate(child1, genome_def, rate=0.1)
                child2 = mutate(child2, genome_def, rate=0.1)
                new_population.append(child1)
                if len(new_population) < population_size:
                    new_population.append(child2)
            
            population = new_population[:population_size]
    
    # Save final result
    final_path = OUTPUT_DIR / f"evolution_{version}_final.json"
    with open(final_path, "w") as f:
        json.dump({
            "version": version,
            "best_individual": best_individual,
            "best_fitness": best_fitness,
            "best_result": {k: v for k, v in best_result.items() if k != "models"},
            "history": history,
            "generations_completed": len(history),
        }, f, indent=2, default=str)
    log(f"\n  {version} evolution complete. Best fitness: {best_fitness:+.0f}")
    log(f"  Best individual: {best_individual}")
    log(f"  Saved: {final_path}")
    
    return {
        "version": version,
        "best_individual": best_individual,
        "best_fitness": best_fitness,
        "best_result": best_result,
        "history": history,
    }


# ═════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--versions", type=str, default="all",
                        help="V1,V2,V3,V4,V5,V6 or 'all'")
    parser.add_argument("--population", type=int, default=500)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--hours", type=float, default=720,
                        help="Total hours for all versions (default 30 days)")
    parser.add_argument("--days", type=int, default=365)
    args = parser.parse_args()
    
    versions = ["V1", "V2", "V3", "V4", "V5", "V6"] if args.versions == "all" else args.versions.split(",")
    
    log(f"╔{'═'*70}╗")
    log(f"║  EVOLUTIONARY OPTIMIZER v6                                   ║")
    log(f"╠{'═'*70}╣")
    log(f"║  Versions: {', '.join(versions):<58} ║")
    log(f"║  Population: {args.population} per version                              ║")
    log(f"║  Generations: {args.generations}                                          ║")
    log(f"║  Hours total: {args.hours}                                          ║")
    log(f"║  Days data: {args.days}                                              ║")
    log(f"╚{'═'*70}╝")
    
    start_time = time.time()
    deadline = start_time + args.hours * 3600
    hours_per_version = args.hours / len(versions)
    
    # Load data once (shared across versions)
    all_data = load_all_data(days=args.days)
    splits = date_purged_split(all_data)
    
    results = {}
    for version in versions:
        version_deadline = time.time() + hours_per_version * 3600
        result = run_evolution(version, all_data, splits,
                                population_size=args.population,
                                generations=args.generations,
                                deadline_ts=min(deadline, version_deadline))
        results[version] = result
    
    # Summary
    log(f"\n{'='*70}")
    log(f"EVOLUTION COMPLETE — Summary")
    log(f"{'='*70}")
    log(f"{'Version':8} {'Best P&L':>10} {'Trades':>8} {'Win%':>6} {'Return%':>10}")
    log(f"{'-'*50}")
    for version, result in results.items():
        r = result["best_result"]
        log(f"{version:8} {r['pnl']:+10.0f} {r['trades']:8} {r['win_rate']*100:6.1f} {r['return_pct']:+10.2f}")
    
    # Save summary
    summary_path = OUTPUT_DIR / "evolution_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "versions": {v: {
                "best_individual": r["best_individual"],
                "best_fitness": r["best_fitness"],
                "best_result": {k: val for k, val in r["best_result"].items() if k != "models"},
            } for v, r in results.items()},
            "total_time_s": time.time() - start_time,
        }, f, indent=2, default=str)
    log(f"\nSummary saved: {summary_path}")
    
    elapsed = time.time() - start_time
    log(f"\nTotal time: {elapsed/3600:.1f} hours")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
