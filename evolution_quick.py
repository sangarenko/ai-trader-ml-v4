#!/usr/bin/env python3
"""Эволюционный оптимизатор v2 — быстрый, без временных ограничений.

Изменения по фидбеку пользователя:
  1. Убраны min_hold_bars, max_hold_bars, cooldown_ticks, maxTradesPerHour
     — "по времени ставить неправильно", пусть торгует когда видит сигнал
  2. Больше параметров (25+ вместо 11)
  3. Быстрый прогон — 5 часов максимум
  4. Боты учатся на 5-мин свечах (это уже было)
  5. Метод отсеивания: 500 случайных → топ-50 → топ-3 → среднее

Архитектура:
  - 6 версий ML (V1-V6)
  - 500 случайных параметров на версию
  - Stage 1: backtest на 7 днях (быстро), отсеять bottom 90%
  - Stage 2: backtest топ-50 на 30 днях
  - Stage 3: топ-3 → усреднить параметры → финальная модель
  - 5 часов максимум

Usage:
  python3 evolution_quick.py --versions all --hours 5
  python3 evolution_quick.py --versions V4,V5,V6 --hours 5
"""
import os
import sys
import json
import time
import random
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
from fast_backtest_v2 import precompute_indicators

OUTPUT_DIR = Path("/root/ai-trader-evolution/ml/evolution_quick_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = "/var/log/ai-trader-evolution-quick.log"

# Fixed (NOT evolved)
COMMISSION_PER_SIDE = 0.0005
ROUNDTRIP_COMMISSION = 0.001
LABEL_THRESHOLD = 0.002
HORIZON_BARS = 6  # 30 min forward (6 × 5min candles)

# ═════════════════════════════════════════════════════════════════
# Genome — расширенный, без временных ограничений
# ═════════════════════════════════════════════════════════════════

GENOME_V1 = {
    # === ML Training Hyperparams (15) ===
    "n_estimators": (50, 500),
    "max_depth": (2, 8),
    "learning_rate": (0.01, 0.3),
    "subsample": (0.5, 1.0),
    "colsample_bytree": (0.4, 0.95),
    "colsample_bylevel": (0.4, 0.95),
    "min_child_weight": (1, 200),
    "gamma": (0.0, 5.0),
    "reg_alpha": (0.0, 5.0),
    "reg_lambda": (0.1, 30.0),
    "max_delta_step": (0.0, 5.0),
    "scale_pos_weight_mult": (0.5, 3.0),  # multiplier on computed scale_pos
    "base_score": (0.3, 0.7),
    "early_stopping_rounds": (20, 100),
    "grow_policy": ["lossguide", "depthwise"],  # categorical
    
    # === Inference Thresholds (4) ===
    "long_threshold": (0.50, 0.90),
    "short_threshold": (0.10, 0.50),
    "exit_long_threshold": (0.20, 0.55),
    "exit_short_threshold": (0.45, 0.80),
    
    # === Position Sizing (3) ===
    "position_size": (0.02, 0.25),
    "max_position_cost": (300, 5000),
    "kelly_fraction": (0.0, 1.0),  # 0=fixed, 0.5=half-kelly, 1=full-kelly
    
    # === Risk Manager (3) ===
    "comm_filter_mult": (0.5, 4.0),
    "stop_loss_pct": (0.005, 0.05),  # 0.5% to 5% stop-loss
    "take_profit_pct": (0.005, 0.05),  # 0.5% to 5% take-profit
}

GENOME_V4_V5_V6 = {
    # === ML Training Hyperparams (10) ===
    "n_estimators": (50, 400),
    "max_depth": (2, 6),
    "learning_rate": (0.01, 0.2),
    "subsample": (0.5, 1.0),
    "colsample_bytree": (0.4, 0.95),
    "min_child_weight": (5, 150),
    "gamma": (0.0, 4.0),
    "reg_alpha": (0.0, 3.0),
    "reg_lambda": (0.5, 25.0),
    "scale_pos_weight_mult": (0.5, 3.0),
    
    # === Inference Thresholds (4) ===
    "long_threshold": (0.50, 0.90),
    "short_threshold": (0.10, 0.50),
    "exit_long_threshold": (0.25, 0.55),
    "exit_short_threshold": (0.45, 0.75),
    
    # === Position Sizing (3) ===
    "position_size": (0.02, 0.25),
    "max_position_cost": (300, 5000),
    "kelly_fraction": (0.0, 1.0),
    
    # === Risk Manager (3) ===
    "comm_filter_mult": (0.5, 4.0),
    "stop_loss_pct": (0.005, 0.05),
    "take_profit_pct": (0.005, 0.05),
}

GENOME_V3 = {
    # === Meta-classifier params (5) ===
    "switch_interval_bars": (6, 144),
    "min_confidence": (0.05, 0.50),
    "top_k_strategies": (1, 5),
    "strategy_pool_size": (5, 22),
    "feature_subset": ["all", "price_only", "indicator_only"],
    
    # === Inference Thresholds (2) ===
    "min_prob_to_trade": (0.05, 0.50),
    "exit_threshold": (0.05, 0.40),
    
    # === Position Sizing (3) ===
    "position_size": (0.02, 0.25),
    "max_position_cost": (300, 5000),
    "kelly_fraction": (0.0, 1.0),
    
    # === Risk Manager (3) ===
    "comm_filter_mult": (0.5, 4.0),
    "stop_loss_pct": (0.005, 0.05),
    "take_profit_pct": (0.005, 0.05),
}

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
    for gene, bounds in genome_def.items():
        if isinstance(bounds, list):
            # categorical
            g[gene] = random.choice(bounds)
        else:
            low, high = bounds
            if isinstance(low, int) and isinstance(high, int):
                g[gene] = random.randint(low, high)
            else:
                g[gene] = random.uniform(low, high)
    return g


def mutate(individual: dict, genome_def: dict, rate: float = 0.15) -> dict:
    """Mutate individual."""
    mutated = dict(individual)
    for gene, bounds in genome_def.items():
        if random.random() < rate:
            if isinstance(bounds, list):
                mutated[gene] = random.choice(bounds)
            else:
                low, high = bounds
                range_size = high - low
                std = range_size * 0.15
                new_val = mutated[gene] + random.gauss(0, std)
                new_val = max(low, min(high, new_val))
                if isinstance(low, int):
                    new_val = int(round(new_val))
                mutated[gene] = new_val
    return mutated


# ═════════════════════════════════════════════════════════════════
# Data preparation
# ═════════════════════════════════════════════════════════════════

def load_all_data(days: int = 60) -> dict:
    """Load all tickers' data — fast (60 days = ~17000 bars/ticker)."""
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
    
    for ticker, aligned in all_tickers_data.items():
        close5 = aligned["5min_close"]
        high5 = aligned["5min_high"]
        low5 = aligned["5min_low"]
        open5 = aligned["5min_open"]
        vol5 = aligned["5min_volume"]
        n = len(close5)
        
        try:
            X_v4, _ = compute_features_v4(aligned, all_tickers_data=all_tickers_data)
        except:
            X_v4, _ = compute_features(aligned)
        
        X_v1, _ = compute_features(aligned)
        ind = precompute_indicators(open5, close5, high5, low5, vol5)
        regime = compute_regime_v2(close5, high5, low5, ind)
        
        y = np.zeros(n, dtype=np.int32)
        for t in range(n - HORIZON_BARS):
            forward_return = (close5[t + HORIZON_BARS] - close5[t]) / (close5[t] + 1e-10)
            y[t] = 1 if forward_return > LABEL_THRESHOLD else 0
        
        X_v4 = np.nan_to_num(X_v4, nan=0.0, posinf=1e6, neginf=-1e6)
        X_v1 = np.nan_to_num(X_v1, nan=0.0, posinf=1e6, neginf=-1e6)
        
        all_data[ticker] = {
            "close": close5, "high": high5, "low": low5,
            "X_v1": X_v1, "X_v4": X_v4,
            "regime": regime, "y": y,
            "timestamp": aligned["time"], "n_bars": n,
        }
    
    log(f"Loaded {len(all_data)} tickers")
    return all_data


def date_split(all_data: dict, train_pct: float = 0.60, val_pct: float = 0.20) -> dict:
    """Date-purged split: 60% train / 20% val / 20% test."""
    all_ts = [(d["timestamp"][0], d["timestamp"][-1]) for d in all_data.values() if len(d["timestamp"]) > 0]
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
            "test_7d": valid & (ts > val_end) & (ts > (global_max - 7*86400*1000)),
            "test_30d": valid & (ts > val_end) & (ts > (global_max - 30*86400*1000)),
        }
    return splits


# ═════════════════════════════════════════════════════════════════
# Model training
# ═════════════════════════════════════════════════════════════════

def train_v1_model(all_data, splits, genome):
    """Train V1 single XGBoost model."""
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
    
    if len(X_train) < 100 or len(X_val) < 10:
        return None
    
    pos_rate = y_train.mean()
    if pos_rate < 0.05 or pos_rate > 0.95:
        return None
    scale_pos = ((1 - pos_rate) / pos_rate) * genome["scale_pos_weight_mult"]
    
    grow = genome.get("grow_policy", "depthwise")
    
    model = xgb.XGBClassifier(
        n_estimators=int(genome["n_estimators"]),
        max_depth=int(genome["max_depth"]),
        learning_rate=genome["learning_rate"],
        subsample=genome["subsample"],
        colsample_bytree=genome["colsample_bytree"],
        colsample_bylevel=genome.get("colsample_bylevel", 1.0),
        min_child_weight=int(genome["min_child_weight"]),
        gamma=genome["gamma"],
        reg_alpha=genome["reg_alpha"],
        reg_lambda=genome["reg_lambda"],
        max_delta_step=genome.get("max_delta_step", 0),
        base_score=genome.get("base_score", 0.5),
        objective="binary:logistic",
        eval_metric="logloss",
        early_stopping_rounds=int(genome.get("early_stopping_rounds", 30)),
        tree_method="hist",
        grow_policy=grow,
        scale_pos_weight=scale_pos,
        random_state=42,
        n_jobs=2,
    )
    try:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return model
    except Exception as e:
        return None


def train_v4_models(all_data, splits, genome):
    """Train 12 per-regime models (V4/V5/V6)."""
    models = {}
    for r in range(12):
        X_train, y_train, X_val, y_val = [], [], [], []
        for ticker, data in all_data.items():
            m = splits[ticker]
            rm = data["regime"] == r
            X_train.append(data["X_v4"][m["train"] & rm])
            y_train.append(data["y"][m["train"] & rm])
            X_val.append(data["X_v4"][m["val"] & rm])
            y_val.append(data["y"][m["val"] & rm])
        
        X_train = np.vstack(X_train) if X_train else np.array([]).reshape(0, 22)
        y_train = np.concatenate(y_train) if y_train else np.array([])
        X_val = np.vstack(X_val) if X_val else np.array([]).reshape(0, 22)
        y_val = np.concatenate(y_val) if y_val else np.array([])
        
        if len(X_train) < 50:
            models[r] = None
            continue
        
        pos_rate = y_train.mean()
        if pos_rate < 0.05 or pos_rate > 0.95:
            models[r] = None
            continue
        scale_pos = ((1 - pos_rate) / pos_rate) * genome["scale_pos_weight_mult"]
        
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
        try:
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            models[r] = model
        except:
            models[r] = None
    return models


# ═════════════════════════════════════════════════════════════════
# Backtest evaluation
# ═════════════════════════════════════════════════════════════════

def evaluate_genome(genome, all_data, splits, version, test_key="test_7d"):
    """Backtest with genome params on test period."""
    t0 = time.time()
    
    if version == "V1":
        model = train_v1_model(all_data, splits, genome)
        if model is None:
            return {"fitness": -1e9, "error": "train_failed"}
        models = {0: model}
        is_v1 = True
    elif version in ("V4", "V5", "V6"):
        models = train_v4_models(all_data, splits, genome)
        is_v1 = False
        if all(m is None for m in models.values()):
            return {"fitness": -1e9, "error": "all_regimes_failed"}
    else:
        return {"fitness": -1e9, "error": "unknown_version"}
    
    long_thr = genome["long_threshold"]
    short_thr = genome["short_threshold"]
    exit_long = genome["exit_long_threshold"]
    exit_short = genome["exit_short_threshold"]
    pos_size = genome["position_size"]
    comm_mult = genome["comm_filter_mult"]
    kelly_frac = genome["kelly_fraction"]
    stop_loss = genome["stop_loss_pct"]
    take_profit = genome["take_profit_pct"]
    
    total_pnl = 0.0
    total_trades = 0
    total_wins = 0
    
    for ticker, data in all_data.items():
        close = data["close"]
        n = len(close)
        test_mask = splits[ticker][test_key]
        test_indices = np.where(test_mask)[0]
        
        if is_v1:
            X = data["X_v1"]
            try:
                probs_all = model.predict_proba(X)[:, 1]
            except:
                continue
        else:
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
        
        balance = 10000.0
        position = 0
        entry_price = 0
        entry_bar = 0
        
        for t in test_indices:
            if t < 100 or t >= n - 1:
                continue
            prob = probs_all[t]
            
            # Kelly position sizing
            edge = abs(prob - 0.5) * 2  # 0 to 1
            actual_pos_size = pos_size * (1 - kelly_frac) + kelly_frac * edge * 0.5
            
            # Commission filter
            expected_gross = edge * 0.005  # rough expected move
            if expected_gross < ROUNDTRIP_COMMISSION * comm_mult:
                continue
            
            # Exit logic
            if position == 1:
                price_change = (close[t] - entry_price) / entry_price
                if prob < exit_long or price_change <= -stop_loss or price_change >= take_profit:
                    gross = (close[t] - entry_price) / entry_price
                    comm = ROUNDTRIP_COMMISSION * balance * actual_pos_size
                    pnl = gross * balance * actual_pos_size - comm
                    balance += pnl
                    total_pnl += pnl
                    total_trades += 1
                    if pnl > 0: total_wins += 1
                    position = 0
            elif position == -1:
                price_change = (entry_price - close[t]) / entry_price
                if prob > exit_short or price_change <= -stop_loss or price_change >= take_profit:
                    gross = (entry_price - close[t]) / entry_price
                    comm = ROUNDTRIP_COMMISSION * balance * actual_pos_size
                    pnl = gross * balance * actual_pos_size - comm
                    balance += pnl
                    total_pnl += pnl
                    total_trades += 1
                    if pnl > 0: total_wins += 1
                    position = 0
            
            # Entry (no min_hold — trade when signal)
            if position == 0:
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
            if position == 1:
                gross = (close[n-1] - entry_price) / entry_price
            else:
                gross = (entry_price - close[n-1]) / entry_price
            pnl = gross * balance * actual_pos_size - ROUNDTRIP_COMMISSION * balance * actual_pos_size
            balance += pnl
            total_pnl += pnl
            total_trades += 1
            if pnl > 0: total_wins += 1
    
    elapsed = time.time() - t0
    n_tickers = len(all_data)
    
    if total_trades < 5:
        fitness = total_pnl - 500  # penalty for not trading
    else:
        fitness = total_pnl
    
    return {
        "fitness": fitness,
        "pnl": total_pnl,
        "trades": total_trades,
        "wins": total_wins,
        "win_rate": total_wins / max(total_trades, 1),
        "return_pct": total_pnl / (10000 * n_tickers) * 100,
        "elapsed_s": elapsed,
    }


# ═════════════════════════════════════════════════════════════════
# Stage 1: Random search (500 individuals, 7-day backtest)
# ═════════════════════════════════════════════════════════════════

def stage1_random_search(version, all_data, splits, n_individuals, deadline):
    """Generate N random individuals, evaluate on 7-day backtest."""
    genome_def = {
        "V1": GENOME_V1,
        "V2": GENOME_V1,
        "V3": GENOME_V3,
        "V4": GENOME_V4_V5_V6,
        "V5": GENOME_V4_V5_V6,
        "V6": GENOME_V4_V5_V6,
    }[version]
    
    log(f"\n{'='*60}")
    log(f"Stage 1: {version} — {n_individuals} random individuals, 7-day backtest")
    log(f"{'='*60}")
    
    results = []
    for i in range(n_individuals):
        if time.time() > deadline:
            log(f"  [{i+1}/{n_individuals}] reached deadline, stopping")
            break
        
        individual = random_genome(genome_def)
        result = evaluate_genome(individual, all_data, splits, version, test_key="test_7d")
        result["individual"] = individual
        results.append(result)
        
        if (i+1) % 10 == 0:
            elapsed = time.time() - (deadline - 5*3600)  # rough
            avg_pnl = np.mean([r["pnl"] for r in results[-10:]])
            best_pnl = max(r["pnl"] for r in results)
            log(f"  [{i+1}/{n_individuals}] avg_pnl={avg_pnl:+.0f} best_pnl={best_pnl:+.0f}")
    
    # Sort by fitness, take top 50
    results.sort(key=lambda r: r["fitness"], reverse=True)
    top_50 = results[:50]
    
    log(f"\n  Stage 1 done. Evaluated {len(results)} individuals")
    log(f"  Top-1: P&L={top_50[0]['pnl']:+.0f} trades={top_50[0]['trades']} win_rate={top_50[0]['win_rate']*100:.0f}%")
    log(f"  Top-50 P&L range: {top_50[-1]['pnl']:+.0f} to {top_50[0]['pnl']:+.0f}")
    
    return top_50


# ═════════════════════════════════════════════════════════════════
# Stage 2: Full backtest top-50 (30-day backtest)
# ═════════════════════════════════════════════════════════════════

def stage2_full_backtest(version, top_individuals, all_data, splits):
    """Re-evaluate top-50 on 30-day backtest."""
    log(f"\n{'='*60}")
    log(f"Stage 2: {version} — top-{len(top_individuals)} on 30-day backtest")
    log(f"{'='*60}")
    
    results = []
    for i, ind_result in enumerate(top_individuals):
        individual = ind_result["individual"]
        result = evaluate_genome(individual, all_data, splits, version, test_key="test_30d")
        result["individual"] = individual
        results.append(result)
        
        if (i+1) % 10 == 0:
            log(f"  [{i+1}/{len(top_individuals)}] pnl={result['pnl']:+.0f} trades={result['trades']}")
    
    results.sort(key=lambda r: r["fitness"], reverse=True)
    
    log(f"\n  Stage 2 done.")
    log(f"  Top-3:")
    for i, r in enumerate(results[:3]):
        log(f"    {i+1}. P&L={r['pnl']:+.0f} trades={r['trades']} win_rate={r['win_rate']*100:.0f}%")
    
    return results[:3]


# ═════════════════════════════════════════════════════════════════
# Stage 3: Average top-3 → final genome
# ═════════════════════════════════════════════════════════════════

def stage3_average_top3(top3, genome_def):
    """Average numerical params of top-3, keep categorical from top-1."""
    final = {}
    for gene, bounds in genome_def.items():
        if isinstance(bounds, list):
            # categorical — take from top-1
            final[gene] = top3[0]["individual"][gene]
        else:
            # numerical — average
            values = [r["individual"][gene] for r in top3]
            low, high = bounds
            avg = sum(values) / len(values)
            if isinstance(low, int):
                avg = int(round(avg))
            final[gene] = max(low, min(high, avg))
    
    log(f"\n  Final averaged genome: {final}")
    return final


# ═════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--versions", type=str, default="all")
    parser.add_argument("--hours", type=float, default=5.0)
    parser.add_argument("--population", type=int, default=500)
    parser.add_argument("--days", type=int, default=60)
    args = parser.parse_args()
    
    versions = ["V1", "V2", "V3", "V4", "V5", "V6"] if args.versions == "all" else args.versions.split(",")
    
    log(f"╔{'═'*70}╗")
    log(f"║  EVOLUTION QUICK (5-hour) — V2                                ║")
    log(f"╠{'═'*70}╣")
    log(f"║  Versions: {', '.join(versions):<55} ║")
    log(f"║  Population: {args.population} (Stage 1) → 50 (Stage 2) → 3 (Stage 3)   ║")
    log(f"║  Hours: {args.hours}                                                  ║")
    log(f"║  Days data: {args.days}                                              ║")
    log(f"║  Stage 1: 7-day backtest (fast filter)                          ║")
    log(f"║  Stage 2: 30-day backtest (full evaluation)                     ║")
    log(f"║  Stage 3: average top-3 → final genome                           ║")
    log(f"╚{'═'*70}╝")
    
    start_time = time.time()
    deadline = start_time + args.hours * 3600
    hours_per_version = args.hours / len(versions)
    
    all_data = load_all_data(days=args.days)
    splits = date_split(all_data)
    
    all_results = {}
    genome_defs = {
        "V1": GENOME_V1, "V2": GENOME_V1, "V3": GENOME_V3,
        "V4": GENOME_V4_V5_V6, "V5": GENOME_V4_V5_V6, "V6": GENOME_V4_V5_V6,
    }
    
    for vi, version in enumerate(versions):
        if time.time() > deadline:
            log(f"\nReached global deadline, skipping {version}")
            break
        
        version_deadline = min(deadline, time.time() + hours_per_version * 3600)
        # Adjust population if time is short
        remaining_time = version_deadline - time.time()
        estimated_per_eval = 15  # seconds
        max_evals = int(remaining_time / estimated_per_eval)
        actual_population = min(args.population, max_evals)
        log(f"\n  {version}: {actual_population} individuals (time-limited)")
        
        # Stage 1
        top_50 = stage1_random_search(version, all_data, splits, actual_population, version_deadline)
        
        if time.time() > version_deadline:
            log(f"  {version}: out of time after Stage 1")
            all_results[version] = {"top_50": top_50[:3], "final": top_50[0]["individual"] if top_50 else None}
            continue
        
        # Stage 2
        top_3 = stage2_full_backtest(version, top_50, all_data, splits)
        
        # Stage 3: average
        final_genome = stage3_average_top3(top_3, genome_defs[version])
        
        all_results[version] = {
            "top_3": top_3,
            "final_genome": final_genome,
        }
        
        # Save checkpoint
        checkpoint_path = OUTPUT_DIR / f"evolution_{version}_quick.json"
        with open(checkpoint_path, "w") as f:
            json.dump({
                "version": version,
                "top_3": [{k: v for k, v in r.items() if k != "individual"} for r in top_3],
                "top_3_genomes": [r["individual"] for r in top_3],
                "final_genome": final_genome,
            }, f, indent=2, default=str)
        log(f"  Saved: {checkpoint_path}")
    
    # Summary
    log(f"\n{'='*70}")
    log(f"EVOLUTION QUICK COMPLETE")
    log(f"{'='*70}")
    log(f"{'Version':8} {'Final P&L':>12} {'Trades':>8} {'Win%':>6} {'Return%':>10}")
    log(f"{'-'*50}")
    for version, res in all_results.items():
        if "top_3" in res and res["top_3"]:
            best = res["top_3"][0]
            log(f"{version:8} {best['pnl']:+12.0f} {best['trades']:8} {best['win_rate']*100:6.1f} {best['return_pct']:+10.2f}")
    
    # Save final summary
    summary_path = OUTPUT_DIR / "evolution_quick_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "versions": {v: {
                "final_genome": r.get("final_genome"),
                "top_3_pnl": [t["pnl"] for t in r.get("top_3", [])],
            } for v, r in all_results.items()},
            "total_time_s": time.time() - start_time,
        }, f, indent=2, default=str)
    log(f"\nSummary: {summary_path}")
    
    elapsed = time.time() - start_time
    log(f"\nTotal time: {elapsed/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
