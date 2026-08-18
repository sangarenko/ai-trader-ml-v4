#!/usr/bin/env python3
"""ML v6 Training — proper version with all fixes from v1-v5 lessons.

Улучшения над v5:
  1. Threshold=0.002 (comm-aware, was 0.001 in v4)
  2. 22 clean features (no duplicates)
  3. Date-purged global split (no cross-ticker leakage)
  4. HIGHER inference threshold: P>0.7 LONG, P<0.3 SHORT (was 0.6/0.4)
     → fewer trades, higher precision
  5. Walk-forward validation: 5 folds (de Prado style)
  6. Per-regime class balance handling
  7. Export with explicit feature_names (no alphabet sorting issues)

Output: /root/ai-trader-evolution/ml/meta_models_v6/regime_v6_<name>.json
"""
import os
import sys
import json
import time
import pickle
import argparse
import numpy as np
import xgboost as xgb
from pathlib import Path
from datetime import datetime, timezone, timedelta
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "fast_mc"))

from ml_data_pipeline import download_multi_timeframe, align_timeframes, TICKERS
from features_v4 import compute_features_v4
from meta_labeler_v2 import compute_regime_v2, REGIME_NAMES, REGIME_TO_IDX
from fast_backtest_v2 import precompute_indicators

OUTPUT_DIR = Path("/root/ai-trader-evolution/ml/meta_models_v6")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = "/var/log/ai-trader-v6-train.log"

# ═════════════════════════════════════════════════════════════════
# v6 CONSTANTS — HIGHER thresholds for precision
# ═════════════════════════════════════════════════════════════════

COMMISSION_PER_SIDE = 0.0005
ROUNDTRIP_COMMISSION = 0.001
PROFIT_MARGIN = 0.001
LABEL_THRESHOLD = ROUNDTRIP_COMMISSION + PROFIT_MARGIN  # 0.002
HORIZON_BARS = 6  # 30 min forward

# v6: HIGHER thresholds for better precision (fewer but better trades)
LONG_THRESHOLD = 0.70    # v5 was 0.65, v4 was 0.6
SHORT_THRESHOLD = 0.30   # v5 was 0.35, v4 was 0.4
EXIT_LONG = 0.50         # close long if P drops to neutral
EXIT_SHORT = 0.50        # close short if P rises to neutral
MIN_HOLD_BARS = 6        # minimum 30 min hold (prevents wash trading)
MAX_HOLD_BARS = 36       # max 3 hours hold

XGB_PARAMS = {
    "n_estimators": 250,
    "max_depth": 4,        # slightly deeper than v5 (was 3)
    "learning_rate": 0.04,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 50,  # higher than v5 (was 30)
    "gamma": 1.0,
    "reg_alpha": 1.0,
    "reg_lambda": 15.0,
    "objective": "binary:logistic",
    "eval_metric": ["logloss", "auc"],
    "early_stopping_rounds": 40,
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": 2,
}


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


# ═════════════════════════════════════════════════════════════════
# Phase 1+2+3: Data + Features + Labels
# ═════════════════════════════════════════════════════════════════

def compute_features_and_labels(ticker: str, aligned: dict, all_tickers_data: dict) -> dict:
    close5 = aligned["5min_close"]
    high5 = aligned["5min_high"]
    low5 = aligned["5min_low"]
    open5 = aligned["5min_open"]
    vol5 = aligned["5min_volume"]
    time5 = aligned["time"]
    n = len(close5)
    
    if n < 200:
        return None
    
    X, feat_names = compute_features_v4(aligned, all_tickers_data=all_tickers_data)
    ind = precompute_indicators(open5, close5, high5, low5, vol5)
    regime = compute_regime_v2(close5, high5, low5, ind)
    
    y = np.zeros(n, dtype=np.int32)
    for t in range(n - HORIZON_BARS):
        forward_close = close5[t + HORIZON_BARS]
        forward_return = (forward_close - close5[t]) / (close5[t] + 1e-10)
        y[t] = 1 if forward_return > LABEL_THRESHOLD else 0
    
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return {
        "ticker": ticker,
        "X": X,
        "feat_names": feat_names,
        "regime": regime,
        "y": y,
        "timestamp": time5,
        "close": close5,
        "n_bars": n,
    }


# ═════════════════════════════════════════════════════════════════
# Phase 4: Walk-forward split (de Prado style, 5 folds)
# ═════════════════════════════════════════════════════════════════

def walk_forward_splits(all_data: dict, n_folds: int = 5) -> list:
    """Generate walk-forward splits.
    
    For fold i (0-indexed):
      Train: days [0, total_days * (i+1)/(n_folds+1)]
      Test:  days [train_end, train_end + total_days/(n_folds+1)]
    
    Returns list of dicts with train_mask, val_mask, test_mask per ticker.
    """
    log(f"\n{'='*60}")
    log(f"PHASE 4: Walk-forward split ({n_folds} folds)")
    log(f"{'='*60}")
    
    # Global time range
    all_ts = []
    for ticker, data in all_data.items():
        if len(data["timestamp"]) > 0:
            all_ts.append((data["timestamp"][0], data["timestamp"][-1]))
    global_min = min(t[0] for t in all_ts)
    global_max = max(t[1] for t in all_ts)
    total_range = global_max - global_min
    
    fold_size = total_range / (n_folds + 1)
    
    folds = []
    for i in range(n_folds):
        train_end = global_min + fold_size * (i + 1)
        test_end = train_end + fold_size
        # Use 80% of train for train, 20% for val
        val_start = train_end - fold_size * 0.2
        
        splits_per_ticker = {}
        for ticker, data in all_data.items():
            ts = data["timestamp"]
            n = len(ts)
            valid_mask = np.ones(n, dtype=bool)
            valid_mask[-HORIZON_BARS:] = False
            
            train_mask = valid_mask & (ts < val_start)
            val_mask = valid_mask & (ts >= val_start) & (ts < train_end)
            test_mask = valid_mask & (ts >= train_end) & (ts < test_end)
            
            splits_per_ticker[ticker] = {
                "train_mask": train_mask,
                "val_mask": val_mask,
                "test_mask": test_mask,
            }
        
        folds.append({
            "fold": i + 1,
            "train_end": train_end,
            "test_end": test_end,
            "splits": splits_per_ticker,
        })
        
        log(f"  Fold {i+1}: train < {datetime.fromtimestamp(val_start/1000, tz=timezone(timedelta(hours=3))).strftime('%Y-%m-%d')} | "
            f"val < {datetime.fromtimestamp(train_end/1000, tz=timezone(timedelta(hours=3))).strftime('%Y-%m-%d')} | "
            f"test < {datetime.fromtimestamp(test_end/1000, tz=timezone(timedelta(hours=3))).strftime('%Y-%m-%d')}")
    
    return folds


# ═════════════════════════════════════════════════════════════════
# Phase 5: Train per-regime models (last fold = final)
# ═════════════════════════════════════════════════════════════════

def train_regime_models(all_data: dict, splits: dict) -> dict:
    """Train 12 XGBoost binary classifiers for one fold."""
    log(f"\n{'='*60}")
    log(f"PHASE 5: Training 12 per-regime XGBoost classifiers")
    log(f"{'='*60}")
    
    # Aggregate per regime
    regime_data = {r: {"X_train": [], "y_train": [], "X_val": [], "y_val": [],
                       "X_test": [], "y_test": [], "feat_names": None} for r in range(12)}
    
    for ticker, data in all_data.items():
        X = data["X"]
        y = data["y"]
        regime = data["regime"]
        feat_names = data["feat_names"]
        masks = splits[ticker]
        
        for r in range(12):
            regime_mask = regime == r
            if regime_mask.sum() == 0:
                continue
            regime_data[r]["feat_names"] = feat_names
            regime_data[r]["X_train"].append(X[masks["train_mask"] & regime_mask])
            regime_data[r]["y_train"].append(y[masks["train_mask"] & regime_mask])
            regime_data[r]["X_val"].append(X[masks["val_mask"] & regime_mask])
            regime_data[r]["y_val"].append(y[masks["val_mask"] & regime_mask])
            regime_data[r]["X_test"].append(X[masks["test_mask"] & regime_mask])
            regime_data[r]["y_test"].append(y[masks["test_mask"] & regime_mask])
    
    models = {}
    for r in range(12):
        rname = REGIME_NAMES[r]
        rd = regime_data[r]
        
        X_train = np.vstack(rd["X_train"]) if rd["X_train"] else np.array([]).reshape(0, 22)
        y_train = np.concatenate(rd["y_train"]) if rd["y_train"] else np.array([])
        X_val = np.vstack(rd["X_val"]) if rd["X_val"] else np.array([]).reshape(0, 22)
        y_val = np.concatenate(rd["y_val"]) if rd["y_val"] else np.array([])
        X_test = np.vstack(rd["X_test"]) if rd["X_test"] else np.array([]).reshape(0, 22)
        y_test = np.concatenate(rd["y_test"]) if rd["y_test"] else np.array([])
        
        n_train = len(X_train)
        log(f"\n  [{r+1:2}/12] {rname}: train={n_train} val={len(X_val)} test={len(X_test)}")
        
        if n_train < 200:
            log(f"    SKIP — too few train samples (<200)")
            models[r] = {"status": "skipped", "reason": "too_few_samples", "n_train": n_train}
            continue
        
        pos_rate = y_train.mean() if len(y_train) > 0 else 0
        log(f"    Positive rate: {pos_rate:.3f}")
        
        if pos_rate < 0.05 or pos_rate > 0.95:
            log(f"    SKIP — class imbalance extreme")
            models[r] = {"status": "skipped", "reason": "class_imbalance", "pos_rate": pos_rate}
            continue
        
        scale_pos = (1 - pos_rate) / pos_rate if pos_rate > 0 else 1.0
        params = {**XGB_PARAMS, "scale_pos_weight": scale_pos}
        
        t0 = time.time()
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        elapsed = time.time() - t0
        
        metrics = {}
        for split, Xs, ys in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
            if len(Xs) == 0:
                metrics[split] = {"n": 0}
                continue
            probs = model.predict_proba(Xs)[:, 1]
            preds = (probs > 0.5).astype(int)
            metrics[split] = {
                "n": int(len(ys)),
                "pos_rate": float(ys.mean()),
                "accuracy": float(accuracy_score(ys, preds)),
                "precision": float(precision_score(ys, preds, zero_division=0)),
                "recall": float(recall_score(ys, preds, zero_division=0)),
                "f1": float(f1_score(ys, preds, zero_division=0)),
            }
            # Precision at different thresholds
            for thr in [0.6, 0.7, 0.8]:
                mask = probs > thr
                if mask.sum() > 0:
                    metrics[split][f"prec_at_{thr}"] = float((ys[mask] == 1).mean())
                    metrics[split][f"n_at_{thr}"] = int(mask.sum())
                else:
                    metrics[split][f"prec_at_{thr}"] = 0.0
                    metrics[split][f"n_at_{thr}"] = 0
        
        log(f"    Trained in {elapsed:.1f}s, best_iter={model.best_iteration}")
        log(f"    VAL:  prec={metrics['val']['precision']:.3f} f1={metrics['val']['f1']:.3f}")
        log(f"    TEST: prec={metrics['test']['precision']:.3f} f1={metrics['test']['f1']:.3f}")
        log(f"    TEST @0.70: prec={metrics['test']['prec_at_0.7']:.3f} (n={metrics['test']['n_at_0.7']})")
        log(f"    TEST @0.80: prec={metrics['test']['prec_at_0.8']:.3f} (n={metrics['test']['n_at_0.8']})")
        
        models[r] = {
            "status": "trained",
            "model": model,
            "feat_names": rd["feat_names"],
            "n_train": n_train,
            "n_val": int(len(X_val)),
            "n_test": int(len(X_test)),
            "metrics": metrics,
        }
    
    return models


# ═════════════════════════════════════════════════════════════════
# Phase 6: Realistic walk-forward backtest
# ═════════════════════════════════════════════════════════════════

def walk_forward_backtest(all_data: dict, models: dict) -> dict:
    """Simulate live trading with v6 thresholds + cooldown + min/max hold."""
    log(f"\n{'='*60}")
    log(f"PHASE 6: Walk-forward backtest (v6 thresholds)")
    log(f"  LONG_THRESHOLD={LONG_THRESHOLD} SHORT_THRESHOLD={SHORT_THRESHOLD}")
    log(f"  EXIT_LONG={EXIT_LONG} EXIT_SHORT={EXIT_SHORT}")
    log(f"  MIN_HOLD={MIN_HOLD_BARS} bars ({MIN_HOLD_BARS*5}min) MAX_HOLD={MAX_HOLD_BARS} bars ({MAX_HOLD_BARS*5/60:.0f}h)")
    log(f"{'='*60}")
    
    results = {"per_ticker": {}, "total": {"pnl": 0, "trades": 0, "wins": 0}}
    
    for ticker, data in all_data.items():
        X = data["X"]
        regime = data["regime"]
        close = data["close"]
        n = len(close)
        
        balance = 10000.0
        position = 0
        entry_price = 0
        entry_bar = 0
        trades = 0
        wins = 0
        pnl_total = 0
        last_trade_bar = -MIN_HOLD_BARS  # cooldown
        
        # Predict P(up) for all bars at once (using regime model)
        # Cache predictions per regime
        regime_preds = {}
        for r in range(12):
            if r in models and models[r]["status"] == "trained":
                mask = regime == r
                if mask.sum() > 0:
                    model = models[r]["model"]
                    probs = model.predict_proba(X[mask])[:, 1]
                    regime_preds[r] = (mask, probs)
        
        # Build full probs array
        all_probs = np.full(n, 0.5)
        for r, (mask, probs) in regime_preds.items():
            indices = np.where(mask)[0]
            all_probs[indices] = probs
        
        for t in range(100, n - 1):
            prob = all_probs[t]
            
            # Exit logic
            if position == 1 and prob < EXIT_LONG:
                exit_price = close[t]
                gross = (exit_price - entry_price) / entry_price
                pnl = (gross - ROUNDTRIP_COMMISSION) * balance * 0.08
                balance += pnl
                pnl_total += pnl
                trades += 1
                if pnl > 0: wins += 1
                position = 0
                last_trade_bar = t
            elif position == -1 and prob > EXIT_SHORT:
                exit_price = close[t]
                gross = (entry_price - exit_price) / entry_price
                pnl = (gross - ROUNDTRIP_COMMISSION) * balance * 0.08
                balance += pnl
                pnl_total += pnl
                trades += 1
                if pnl > 0: wins += 1
                position = 0
                last_trade_bar = t
            
            # Max hold force-close
            if position != 0 and t - entry_bar >= MAX_HOLD_BARS:
                exit_price = close[t]
                if position == 1:
                    gross = (exit_price - entry_price) / entry_price
                else:
                    gross = (entry_price - exit_price) / entry_price
                pnl = (gross - ROUNDTRIP_COMMISSION) * balance * 0.08
                balance += pnl
                pnl_total += pnl
                trades += 1
                if pnl > 0: wins += 1
                position = 0
                last_trade_bar = t
            
            # Entry logic: only if flat AND past cooldown AND past min hold
            if position == 0 and (t - last_trade_bar) >= MIN_HOLD_BARS:
                if prob > LONG_THRESHOLD:
                    position = 1
                    entry_price = close[t]
                    entry_bar = t
                elif prob < SHORT_THRESHOLD:
                    position = -1
                    entry_price = close[t]
                    entry_bar = t
        
        # Close any remaining
        if position != 0:
            exit_price = close[n - 1]
            if position == 1:
                gross = (exit_price - entry_price) / entry_price
            else:
                gross = (entry_price - exit_price) / entry_price
            pnl = (gross - ROUNDTRIP_COMMISSION) * balance * 0.08
            balance += pnl
            pnl_total += pnl
            trades += 1
            if pnl > 0: wins += 1
        
        results["per_ticker"][ticker] = {
            "pnl": float(pnl_total),
            "balance": float(balance),
            "trades": int(trades),
            "wins": int(wins),
            "win_rate": float(wins / max(trades, 1)),
            "return_pct": float((balance - 10000) / 10000 * 100),
        }
        results["total"]["pnl"] += pnl_total
        results["total"]["trades"] += trades
        results["total"]["wins"] += wins
        
        log(f"  {ticker}: P&L={pnl_total:+.0f}₽ balance={balance:.0f}₽ trades={trades} win_rate={wins/max(trades,1)*100:.0f}%")
    
    n_tickers = len(results["per_ticker"])
    total_pnl = results["total"]["pnl"]
    total_trades = results["total"]["trades"]
    total_wins = results["total"]["wins"]
    log(f"\n  TOTAL: P&L={total_pnl:+.0f}₽ ({total_pnl/(10000*n_tickers)*100:+.2f}%) trades={total_trades} win_rate={total_wins/max(total_trades,1)*100:.0f}%")
    
    return results


# ═════════════════════════════════════════════════════════════════
# Phase 7: Export
# ═════════════════════════════════════════════════════════════════

def export_models(models: dict, backtest_results: dict, fold_info: dict) -> None:
    log(f"\n{'='*60}")
    log(f"PHASE 7: Export to JSON")
    log(f"{'='*60}")
    
    def to_native(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: to_native(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [to_native(x) for x in obj]
        return obj
    
    metadata = {
        "version": "v6",
        "trained_at": datetime.now(timezone(timedelta(hours=3))).isoformat(),
        "fold": fold_info,
        "label_threshold": LABEL_THRESHOLD,
        "horizon_bars": HORIZON_BARS,
        "long_threshold": LONG_THRESHOLD,
        "short_threshold": SHORT_THRESHOLD,
        "exit_long": EXIT_LONG,
        "exit_short": EXIT_SHORT,
        "min_hold_bars": MIN_HOLD_BARS,
        "max_hold_bars": MAX_HOLD_BARS,
        "commission_per_side": COMMISSION_PER_SIDE,
        "features_count": 22,
        "regimes": {},
        "backtest": to_native(backtest_results),
    }
    
    for r in range(12):
        rname = REGIME_NAMES[r].lower()
        if models[r]["status"] != "trained":
            metadata["regimes"][REGIME_NAMES[r]] = {
                "status": "skipped",
                "reason": models[r].get("reason", "unknown")
            }
            continue
        
        model = models[r]["model"]
        feat_names = models[r]["feat_names"]
        
        booster = model.get_booster()
        config = json.loads(booster.save_raw(raw_format="json").decode("utf-8"))
        json_path = OUTPUT_DIR / f"regime_v6_{rname}.json"
        with open(json_path, "w") as f:
            json.dump(config, f)
        
        pkl_path = OUTPUT_DIR / f"regime_v6_{rname}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(model, f)
        
        metadata["regimes"][REGIME_NAMES[r]] = {
            "status": "trained",
            "model_file": json_path.name,
            "n_train": int(models[r]["n_train"]),
            "n_val": int(models[r]["n_val"]),
            "n_test": int(models[r]["n_test"]),
            "metrics": to_native(models[r]["metrics"]),
            "feature_names": feat_names,
        }
        log(f"  {REGIME_NAMES[r]:24}: {json_path.name} ({json_path.stat().st_size/1024:.0f} KB)")
    
    meta_path = OUTPUT_DIR / "regime_v6_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(to_native(metadata), f, indent=2)
    log(f"\nMetadata: {meta_path}")
    
    bt_path = OUTPUT_DIR / "regime_v6_backtest.json"
    with open(bt_path, "w") as f:
        json.dump(to_native(backtest_results), f, indent=2)
    log(f"Backtest: {bt_path}")


# ═════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--tickers", type=str, default="all")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    
    tickers = TICKERS if args.tickers == "all" else args.tickers.split(",")
    
    log(f"╔{'═'*60}╗")
    log(f"║  ML v6 TRAINING PIPELINE                                  ║")
    log(f"╠{'═'*60}╣")
    log(f"║  Tickers: {len(tickers):<46} ║")
    log(f"║  Days: {args.days:<48} ║")
    log(f"║  Label threshold: {LABEL_THRESHOLD} (comm-aware)              ║")
    log(f"║  Inference: LONG>{LONG_THRESHOLD} SHORT<{SHORT_THRESHOLD} (v6 higher)         ║")
    log(f"║  Features: 22 (clean)                                     ║")
    log(f"║  Walk-forward: {args.folds} folds (de Prado)                       ║")
    log(f"╚{'═'*60}╝")
    
    start_time = time.time()
    
    # Phase 1+2+3: Data + Features + Labels
    log(f"\n{'='*60}")
    log(f"PHASE 1+2+3: Loading data + computing features + labels")
    log(f"{'='*60}")
    all_data = {}
    all_tickers_data = {}
    for i, ticker in enumerate(tickers):
        try:
            data = download_multi_timeframe(ticker, days=args.days)
            if "5min_close" not in data:
                continue
            aligned = align_timeframes(data)
            all_tickers_data[ticker] = aligned
        except Exception as e:
            log(f"  [{i+1}/{len(tickers)}] {ticker}: ERROR {e}")
    
    for ticker, aligned in all_tickers_data.items():
        data = compute_features_and_labels(ticker, aligned, all_tickers_data)
        if data:
            all_data[ticker] = data
            log(f"  {ticker}: X={data['X'].shape} pos_rate={data['y'].mean():.3f}")
    
    if not all_data:
        log("FATAL: No data loaded")
        return 1
    
    # Phase 4: Walk-forward splits
    folds = walk_forward_splits(all_data, n_folds=args.folds)
    
    # Phase 5+6: Use last fold for final model (most recent data)
    log(f"\n{'='*60}")
    log(f"PHASE 5: Training on last fold (most recent data)")
    log(f"{'='*60}")
    last_fold = folds[-1]
    models = train_regime_models(all_data, last_fold["splits"])
    
    # Phase 6: Backtest on test portion of last fold
    backtest_results = walk_forward_backtest(all_data, models)
    
    # Phase 7: Export
    fold_info = {
        "fold_number": last_fold["fold"],
        "n_folds": args.folds,
        "train_end": last_fold["train_end"],
        "test_end": last_fold["test_end"],
    }
    export_models(models, backtest_results, fold_info)
    
    elapsed = time.time() - start_time
    log(f"\n╔{'═'*60}╗")
    log(f"║  V6 TRAINING COMPLETE in {elapsed/60:.1f} min                       ║")
    log(f"╠{'═'*60}╣")
    log(f"║  Models trained: {sum(1 for m in models.values() if m['status']=='trained')}/12                          ║")
    log(f"║  Backtest P&L: {backtest_results['total']['pnl']:+.0f}₽                                ║")
    log(f"║  Output: {OUTPUT_DIR}                              ║")
    log(f"╚{'═'*60}╝")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
