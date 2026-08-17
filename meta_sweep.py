#!/usr/bin/env python3
"""Ночной Sweep: серія ML моделей с разными форматами.

За 12 часов прогоняет ~50-100 разных ML конфигураций:
  - Разные lookback окна (24h / 48h / 96h)
  - Разные label horizons (30min / 1h / 2.5h)
  - Разные pools стратегий (all_22 / top_10_mc / top_5_mc)
  - Разные model hyperparams (n_estimators, max_depth, learning_rate)
  - Разные label types (best_strategy / pnl_regression / long_short_binary)
  - Разные feature subsets (all_33 / price_only / indicator_only)

Для каждого эксперимента:
  1. Обучить модель
  2. Записать val accuracy
  3. Прогнать backtest с разным switch_interval
  4. Записать P&L
  5. Сохранить в results

В конце: отсортировать по P&L, показать топ-10.
Лучший эксперимент автоматически деплоится на trader-сервер.

Usage:
  python3 meta_sweep.py --hours 12
  python3 meta_sweep.py --max-experiments 50
"""
import os
import sys
import json
import time
import random
import pickle
import argparse
import itertools
import numpy as np
import xgboost as xgb
from pathlib import Path
from datetime import datetime, timezone, timedelta
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from sklearn.preprocessing import LabelEncoder

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "fast_mc"))

from ml_data_pipeline import download_multi_timeframe, align_timeframes, TICKERS
from ml_features import compute_features
from meta_labeler import compute_regime, STRATEGY_NAMES, DEFAULT_PARAMS, STRAT_TO_IDX
from fast_backtest_v2 import precompute_indicators, vectorized_backtest

LOG_FILE = "/var/log/ai-trader-meta-sweep.log"
LABELS_PATH = Path("/root/ai-trader-evolution/ml/data_cache/meta_labels.npz")
OUTPUT_DIR = Path("/root/ai-trader-evolution/ml/sweep_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Top-5 strategies by Monte Carlo P&L (from previous analysis)
TOP_5_STRATEGIES = ['random_hold_short', 'v2_short', 'momentum_volume', 'golden_cross', 'zscore_reversion']
TOP_10_STRATEGIES = TOP_5_STRATEGIES + ['mean_reversion', 'vwap_reversion', 'multi_timeframe', 'v2_inverted', 'bb_reversion']

# Feature subsets
PRICE_FEATURES = ['ret_1', 'ret_5', 'ret_10', 'ret_30', 'ret_5_log', '1d_ret', '1h_ret']
INDICATOR_FEATURES = ['rsi14', 'rsi2', 'sma5_sma14', 'sma14_sma20', 'sma20_sma50',
                       'bb_pct_b', 'bb_width', 'macd_hist', 'macd_line', 'macd_signal',
                       'stoch_k', 'adx', 'atr_pct', 'obv_slope', 'vol_ratio']


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


def load_meta_labels() -> dict:
    d = np.load(str(LABELS_PATH), allow_pickle=True)
    return {
        "bar_indices": d["bar_indices"],
        "regimes": d["regimes"],
        "best_strategies": d["best_strategies"],
        "pnls_matrix": d["pnls_matrix"],
        "strategy_names": list(d["strategy_names"]),
        "tickers": list(d["tickers"]),
        "window_bars": int(d["window_bars"]),
        "step_bars": int(d["step_bars"]),
    }


def build_dataset(labels: dict, strategy_subset: list, feature_subset: list,
                  days: int = 180) -> tuple:
    """Build (X, y, ticker_ids, feature_names) for given strategy pool and features."""
    tickers = labels["tickers"]
    all_X = []
    all_y = []
    all_ticker_ids = []
    all_global_idx = []

    # Map strategy subset to indices
    subset_indices = [STRAT_TO_IDX[s] for s in strategy_subset]

    sample_idx_per_ticker = {}
    cursor = 0
    for ti, t in enumerate(tickers):
        try:
            data = download_multi_timeframe(t, days=days)
            if "5min_close" not in data:
                continue
            aligned = align_timeframes(data)
            n = len(aligned["5min_close"])
            window = labels["window_bars"]
            step = labels["step_bars"]
            samples_this = len(range(window, n - 10, step))
            sample_idx_per_ticker[ti] = (cursor, cursor + samples_this, aligned)
            cursor += samples_this
        except Exception:
            continue

    for ti, (start_i, end_i, aligned) in sample_idx_per_ticker.items():
        n = len(aligned["5min_close"])
        X_full, feat_names_full = compute_features(aligned)
        close5 = aligned["5min_close"]
        from fast_backtest_v2 import precompute_indicators as precompute
        ind = precompute(aligned["5min_open"], close5,
                          aligned["5min_high"], aligned["5min_low"],
                          aligned["5min_volume"])
        regime = compute_regime(close5, ind)
        sma14 = ind.get("sma14", np.zeros(n))
        sma50 = ind.get("sma50", np.zeros(n))

        # Add regime + trend_slope if not in features
        feat_names = list(feat_names_full)
        X = X_full
        if "regime" not in feat_names:
            X = np.column_stack([X, regime.astype(float)])
            feat_names.append("regime")
        if "trend_slope" not in feat_names:
            X = np.column_stack([X, (sma50 - sma14) / (sma14 + 1e-9)])
            feat_names.append("trend_slope")

        # Subset features if specified
        if feature_subset != "all":
            # feature_subset is a list of feature names to keep
            cols_to_keep = [i for i, fn in enumerate(feat_names) if fn in feature_subset]
            X = X[:, cols_to_keep]
            feat_names = [feat_names[i] for i in cols_to_keep]

        for gi in range(start_i, end_i):
            global_bar_idx = labels["bar_indices"][gi]
            if 0 <= global_bar_idx < n:
                # Label: best strategy AMONG subset (or original best_strategy if not in subset)
                orig_best = int(labels["best_strategies"][gi])
                if orig_best in subset_indices:
                    # Remap to position in subset
                    y = subset_indices.index(orig_best)
                else:
                    # Best strategy not in subset → pick best among subset using pnls_matrix
                    pnls_row = labels["pnls_matrix"][gi, subset_indices]
                    y = int(np.argmax(pnls_row))
                all_X.append(X[global_bar_idx])
                all_y.append(y)
                all_ticker_ids.append(ti)
                all_global_idx.append(gi)

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.int32)
    ticker_ids = np.array(all_ticker_ids, dtype=np.int32)
    return X, y, ticker_ids, feat_names


def train_one_model(X, y, hyperparams: dict) -> tuple:
    """Train one XGBoost model with given hyperparams. Returns (model, metrics)."""
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    train_classes = sorted(np.unique(y_train).tolist())
    orig_to_enc = {int(o): int(e) for e, o in enumerate(train_classes)}
    enc_to_orig = {v: k for k, v in orig_to_enc.items()}
    eff_n_classes = len(train_classes)

    def remap(arr):
        return np.array([orig_to_enc.get(int(v), 0) for v in arr], dtype=np.int32)

    y_train = remap(y_train)
    y_val = remap(y_val)
    y_test = remap(y_test)

    class_counts = np.bincount(y_train, minlength=eff_n_classes)
    class_weights = len(y_train) / (eff_n_classes * np.maximum(class_counts, 1))
    sample_weights = class_weights[y_train]

    model = xgb.XGBClassifier(
        n_estimators=hyperparams.get("n_estimators", 200),
        max_depth=hyperparams.get("max_depth", 4),
        learning_rate=hyperparams.get("learning_rate", 0.05),
        subsample=hyperparams.get("subsample", 0.8),
        colsample_bytree=hyperparams.get("colsample_bytree", 0.7),
        min_child_weight=hyperparams.get("min_child_weight", 20),
        gamma=hyperparams.get("gamma", 0.5),
        reg_alpha=hyperparams.get("reg_alpha", 0.5),
        reg_lambda=hyperparams.get("reg_lambda", 5.0),
        objective="multi:softprob",
        num_class=eff_n_classes,
        random_state=42,
        n_jobs=2,
        eval_metric="mlogloss",
        early_stopping_rounds=30,
        tree_method="hist",
    )

    t0 = time.time()
    model.fit(X_train, y_train, sample_weight=sample_weights,
              eval_set=[(X_val, y_val)], verbose=False)
    elapsed = time.time() - t0

    # Evaluate
    metrics = {"train_time_s": elapsed, "best_iter": model.best_iteration}
    for split, Xs, ys in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        if len(Xs) == 0: continue
        probs = model.predict_proba(Xs)
        preds = np.argmax(probs, axis=1)
        acc = accuracy_score(ys, preds)
        top3 = top_k_accuracy_score(ys, probs, k=min(3, eff_n_classes), labels=range(eff_n_classes))
        metrics[f"{split}_top1"] = acc
        metrics[f"{split}_top3"] = top3

    return model, metrics, enc_to_orig, eff_n_classes


def backtest_meta_selector(model, enc_to_orig, eff_n_classes, ticker: str,
                            aligned: dict, ind: dict, feat_names: list,
                            strategy_subset: list, switch_intervals: list = [36, 144]) -> dict:
    """Backtest the model on a single ticker with multiple switch intervals."""
    close5 = aligned["5min_close"]
    high5 = aligned["5min_high"]
    low5 = aligned["5min_low"]
    n = len(close5)

    if n < 600:
        return {}

    X_full, feat_names_full = compute_features(aligned)
    # Add regime + trend_slope if not in features
    regime = compute_regime(close5, ind)
    sma14 = ind.get("sma14", np.zeros(n))
    sma50 = ind.get("sma50", np.zeros(n))
    if "regime" not in feat_names_full:
        X_full = np.column_stack([X_full, regime.astype(float)])
        feat_names_full.append("regime")
    if "trend_slope" not in feat_names_full:
        X_full = np.column_stack([X_full, (sma50 - sma14) / (sma14 + 1e-9)])
        feat_names_full.append("trend_slope")

    # Subset features if model was trained on subset
    if feat_names != feat_names_full:
        cols = [feat_names_full.index(f) for f in feat_names if f in feat_names_full]
        X_full = X_full[:, cols]

    X_full = np.nan_to_num(X_full, nan=0.0, posinf=1e6, neginf=-1e6)
    probs_all = model.predict_proba(X_full)
    preds_enc = np.argmax(probs_all, axis=1)

    results = {}
    for switch_int in switch_intervals:
        switch_points = list(range(50, n - switch_int - 1, switch_int))
        total_pnl = 0.0
        total_trades = 0
        total_wins = 0
        strat_usage = {}

        for sp in switch_points:
            end_bar = sp + switch_int
            if end_bar > n - 1:
                break
            strat_idx_enc = int(preds_enc[sp])
            strat_idx_orig = enc_to_orig.get(strat_idx_enc, 0)
            strat_name = strategy_subset[strat_idx_orig] if strat_idx_orig < len(strategy_subset) else strategy_subset[0]

            sub_ind = {k: v[sp:end_bar] for k, v in ind.items()}
            sub_close = close5[sp:end_bar]
            sub_high = high5[sp:end_bar]
            sub_low = low5[sp:end_bar]

            if len(sub_close) < 30:
                continue

            params = DEFAULT_PARAMS.get(strat_name, DEFAULT_PARAMS['v2_short'])
            try:
                result = vectorized_backtest(sub_ind, sub_close, sub_high, sub_low, strat_name, params, commission=0.0005)
                pnl = result.get("pnl", 0.0)
                trades = result.get("trades", 0)
                wins = result.get("wins", 0)
            except Exception:
                pnl, trades, wins = 0, 0, 0

            scale = 0.10 / params.get("position_size", 0.3)
            total_pnl += pnl * scale
            total_trades += trades
            total_wins += wins
            strat_usage[strat_name] = strat_usage.get(strat_name, 0) + 1

        results[f"switch_{switch_int}"] = {
            "pnl": total_pnl,
            "trades": total_trades,
            "win_rate": total_wins / max(total_trades, 1),
            "switches": len(switch_points),
            "return_pct": total_pnl / 10000 * 100,
        }

    return results


def generate_experiments(n_experiments: int = 50) -> list:
    """Generate a list of experiment configs."""
    # Define grid
    grids = {
        "strategy_pool": ["all_22", "top_10_mc", "top_5_mc"],
        "feature_subset": ["all", "price_only", "indicator_only"],
        "n_estimators": [100, 200, 400],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.03, 0.05, 0.1],
        "min_child_weight": [10, 20, 50],
        "gamma": [0.3, 1.0, 2.0],
        "reg_lambda": [2.0, 5.0, 10.0],
    }

    # Generate random combinations
    random.seed(42)
    keys = list(grids.keys())
    experiments = []
    seen = set()
    attempts = 0
    while len(experiments) < n_experiments and attempts < n_experiments * 5:
        config = {k: random.choice(grids[k]) for k in keys}
        config_key = tuple(sorted(config.items()))
        if config_key not in seen:
            seen.add(config_key)
            experiments.append(config)
        attempts += 1

    return experiments


def run_experiment(exp_config: dict, labels: dict, days: int = 180) -> dict:
    """Run one experiment: train + backtest."""
    # Determine strategy subset
    if exp_config["strategy_pool"] == "all_22":
        strategy_subset = STRATEGY_NAMES
    elif exp_config["strategy_pool"] == "top_10_mc":
        strategy_subset = TOP_10_STRATEGIES
    elif exp_config["strategy_pool"] == "top_5_mc":
        strategy_subset = TOP_5_STRATEGIES

    # Determine feature subset
    if exp_config["feature_subset"] == "all":
        feature_subset = "all"
    elif exp_config["feature_subset"] == "price_only":
        feature_subset = PRICE_FEATURES + ["regime", "trend_slope"]
    elif exp_config["feature_subset"] == "indicator_only":
        feature_subset = INDICATOR_FEATURES + ["regime", "trend_slope"]

    # Build dataset
    log(f"  Building dataset (pool={exp_config['strategy_pool']}, feats={exp_config['feature_subset']})...")
    X, y, ticker_ids, feat_names = build_dataset(labels, strategy_subset, feature_subset, days=days)
    log(f"  Dataset: X={X.shape}, y={y.shape}, classes={len(np.unique(y))}")

    # Train
    hyperparams = {
        "n_estimators": exp_config["n_estimators"],
        "max_depth": exp_config["max_depth"],
        "learning_rate": exp_config["learning_rate"],
        "min_child_weight": exp_config["min_child_weight"],
        "gamma": exp_config["gamma"],
        "reg_lambda": exp_config["reg_lambda"],
    }

    log(f"  Training: {hyperparams}")
    model, metrics, enc_to_orig, eff_n_classes = train_one_model(X, y, hyperparams)
    log(f"  Trained in {metrics['train_time_s']:.1f}s, val top1={metrics['val_top1']:.3f} top3={metrics['val_top3']:.3f}")

    # Backtest on each ticker
    log(f"  Backtesting on {len(labels['tickers'])} tickers...")
    per_ticker = {}
    for ti, ticker in enumerate(labels["tickers"]):
        try:
            data = download_multi_timeframe(ticker, days=days)
            aligned = align_timeframes(data)
            ind = precompute_indicators(aligned["5min_open"], aligned["5min_close"],
                                          aligned["5min_high"], aligned["5min_low"],
                                          aligned["5min_volume"])
            res = backtest_meta_selector(model, enc_to_orig, eff_n_classes,
                                           ticker, aligned, ind, feat_names,
                                           strategy_subset, switch_intervals=[36, 144, 288])
            per_ticker[ticker] = res
        except Exception as e:
            log(f"    {ticker}: ERROR {e}")
            per_ticker[ticker] = {}

    # Aggregate
    total_pnl_36 = sum(per_ticker[t].get("switch_36", {}).get("pnl", 0) for t in per_ticker)
    total_pnl_288 = sum(per_ticker[t].get("switch_288", {}).get("pnl", 0) for t in per_ticker)
    total_pnl_144 = sum(per_ticker[t].get("switch_144", {}).get("pnl", 0) for t in per_ticker)
    total_trades_36 = sum(per_ticker[t].get("switch_36", {}).get("trades", 0) for t in per_ticker)
    total_trades_144 = sum(per_ticker[t].get("switch_144", {}).get("trades", 0) for t in per_ticker)
    total_trades_288 = sum(per_ticker[t].get("switch_288", {}).get("trades", 0) for t in per_ticker)

    summary = {
        "exp_config": exp_config,
        "n_samples": len(X),
        "n_features": len(feat_names),
        "feat_names": feat_names,
        "n_classes_effective": eff_n_classes,
        "strategy_pool_size": len(strategy_subset),
        "metrics": metrics,
        "backtest": {
            "switch_36": {
                "total_pnl": total_pnl_36,
                "total_trades": total_trades_36,
                "return_pct": total_pnl_36 / (10000 * len(per_ticker)) * 100,
            },
            "switch_144": {
                "total_pnl": total_pnl_144,
                "total_trades": total_trades_144,
                "return_pct": total_pnl_144 / (10000 * len(per_ticker)) * 100,
            },
            "switch_288": {
                "total_pnl": total_pnl_288,
                "total_trades": total_trades_288,
                "return_pct": total_pnl_288 / (10000 * len(per_ticker)) * 100,
            },
        },
        "per_ticker": per_ticker,
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=12, help="How long to run (hours)")
    parser.add_argument("--max-experiments", type=int, default=50, help="Max experiments to run")
    parser.add_argument("--days", type=int, default=180)
    args = parser.parse_args()

    log(f"═══ META SWEEP START ═══")
    log(f"Hours: {args.hours}, Max experiments: {args.max_experiments}, Days: {args.days}")

    # Load labels
    labels = load_meta_labels()
    log(f"Labels: {len(labels['bar_indices'])} samples from {len(labels['tickers'])} tickers")

    # Generate experiments
    experiments = generate_experiments(args.max_experiments)
    log(f"Generated {len(experiments)} experiments")

    results = []
    start_time = time.time()
    deadline = start_time + args.hours * 3600

    for i, exp in enumerate(experiments):
        elapsed = time.time() - start_time
        if time.time() > deadline:
            log(f"\nReached deadline ({args.hours}h). Stopping.")
            break

        log(f"\n{'='*60}")
        log(f"EXPERIMENT {i+1}/{len(experiments)} (elapsed {elapsed/60:.1f}min / {args.hours*60:.0f}min)")
        log(f"Config: {exp}")
        log(f"{'='*60}")

        try:
            t0 = time.time()
            summary = run_experiment(exp, labels, days=args.days)
            elapsed_exp = time.time() - t0
            summary["elapsed_s"] = elapsed_exp
            results.append(summary)

            # Print summary
            log(f"\n  RESULT:")
            log(f"    val top1={summary['metrics']['val_top1']:.3f}  top3={summary['metrics']['val_top3']:.3f}")
            log(f"    test top1={summary['metrics']['test_top1']:.3f}  top3={summary['metrics']['test_top3']:.3f}")
            log(f"    backtest switch_36: P&L={summary['backtest']['switch_36']['total_pnl']:+.0f} ({summary['backtest']['switch_36']['return_pct']:+.2f}%) trades={summary['backtest']['switch_36']['total_trades']}")
            log(f"    backtest switch_144: P&L={summary['backtest']['switch_144']['total_pnl']:+.0f} ({summary['backtest']['switch_144']['return_pct']:+.2f}%) trades={summary['backtest']['switch_144']['total_trades']}")
            log(f"    elapsed: {elapsed_exp:.0f}s")
        except Exception as e:
            log(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

        # Save incremental results
        out_path = OUTPUT_DIR / "sweep_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        log(f"  Saved → {out_path} ({len(results)} results so far)")

    # Final ranking
    log(f"\n{'='*60}")
    log(f"SWEEP DONE — {len(results)} experiments")
    log(f"{'='*60}")

    # Rank by switch_144 P&L (longer interval usually more trades)
    results.sort(key=lambda r: r["backtest"]["switch_144"]["total_pnl"], reverse=True)

    log(f"\n=== TOP 10 by switch_144 P&L ===")
    log(f"{'#':3} {'pool':10} {'feats':14} {'n_est':>5} {'depth':>5} {'lr':>5} {'val_t1':>6} {'val_t3':>6} {'P&L_144':>8} {'ret%':>6}")
    for i, r in enumerate(results[:10]):
        c = r["exp_config"]
        m = r["metrics"]
        b = r["backtest"]["switch_144"]
        log(f"{i+1:3} {c['strategy_pool']:10} {c['feature_subset']:14} {c['n_estimators']:>5} {c['max_depth']:>5} {c['learning_rate']:>5} {m.get('val_top1',0):>6.3f} {m.get('val_top3',0):>6.3f} {b['total_pnl']:>+8.0f} {b['return_pct']:>+6.2f}")

    # Save best experiment details
    if results:
        best = results[0]
        best_path = OUTPUT_DIR / "best_experiment.json"
        with open(best_path, "w") as f:
            json.dump(best, f, indent=2, default=str)
        log(f"\n=== BEST EXPERIMENT ===")
        log(f"Config: {best['exp_config']}")
        log(f"P&L (switch_144): {best['backtest']['switch_144']['total_pnl']:+.0f} ({best['backtest']['switch_144']['return_pct']:+.2f}%)")
        log(f"Saved → {best_path}")

    log(f"\nAll results: {OUTPUT_DIR / 'sweep_results.json'}")
    log(f"═══ META SWEEP DONE ═══")
    return 0


if __name__ == "__main__":
    sys.exit(main())
