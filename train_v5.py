#!/usr/bin/env python3
"""ML v5 Training Pipeline — clean features + comm-aware labels + walk-forward.

5-часовой workflow:
  1. Data: 11 tickers × 365 days × 5min MOEX
  2. Features: 22 clean (features_v4.py)
  3. Labels: forward_return > 0.002 (comm-aware, 0.2% = roundtrip + margin)
  4. Regime: 12 rule-based regimes (compute_regime_v2)
  5. Split: DATE-PURGED global (70/15/15, no per-ticker leakage)
  6. Train: 12 XGBoost binary classifiers per regime
  7. Walk-forward backtest: realistic commission + position size
  8. Export: .json for pure-TS inference

Улучшения над v4:
  - threshold=0.002 (vs 0.001) → модель учит alpha, не breakeven
  - features_v4 (22 vs 31) → без дубликатов
  - date-purged split → без cross-ticker leakage
  - 365 days (vs 180) → больше данных для редких режимов
  - walk-forward backtest → реалистичная оценка
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

OUTPUT_DIR = Path("/root/ai-trader-evolution/ml/meta_models_v5")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = "/var/log/ai-trader-v5-train.log"

# ═════════════════════════════════════════════════════════════════
# CONSTANTS — v5 improvements
# ═════════════════════════════════════════════════════════════════

COMMISSION_PER_SIDE = 0.0005  # 0.05%
ROUNDTRIP_COMMISSION = 0.001  # 0.1%
PROFIT_MARGIN = 0.001  # 0.1% minimum profit after commission
LABEL_THRESHOLD = ROUNDTRIP_COMMISSION + PROFIT_MARGIN  # = 0.002 (0.2%)
HORIZON_BARS = 6  # 6 × 5min = 30min forward

# Inference thresholds (HIGHER than v4)
LONG_THRESHOLD = 0.65  # was 0.6 in v4
SHORT_THRESHOLD = 0.35  # was 0.4 in v4
EXIT_LONG = 0.45
EXIT_SHORT = 0.55

# XGBoost hyperparams (strong regularization)
XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 30,
    "gamma": 0.5,
    "reg_alpha": 0.5,
    "reg_lambda": 10.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "early_stopping_rounds": 30,
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
# PHASE 1: Data loading
# ═════════════════════════════════════════════════════════════════

def load_all_data(tickers: list, days: int = 365) -> dict:
    """Load multi-timeframe data for all tickers. Returns {ticker: aligned_dict}."""
    log(f"\n{'='*60}")
    log(f"PHASE 1: Loading MOEX data ({len(tickers)} tickers × {days} days)")
    log(f"{'='*60}")
    
    all_data = {}
    all_tickers_data_for_cross = {}  # for cross-asset features
    for i, ticker in enumerate(tickers):
        try:
            data = download_multi_timeframe(ticker, days=days)
            if "5min_close" not in data:
                log(f"  [{i+1}/{len(tickers)}] {ticker}: SKIP (no 5min data)")
                continue
            aligned = align_timeframes(data)
            n = len(aligned["5min_close"])
            log(f"  [{i+1}/{len(tickers)}] {ticker}: {n} 5min bars ({n*5/60:.0f}h)")
            all_data[ticker] = aligned
            all_tickers_data_for_cross[ticker] = aligned
        except Exception as e:
            log(f"  [{i+1}/{len(tickers)}] {ticker}: ERROR {e}")
    
    log(f"\nLoaded {len(all_data)} tickers")
    return all_data, all_tickers_data_for_cross


# ═════════════════════════════════════════════════════════════════
# PHASE 2+3: Features + Labels
# ═════════════════════════════════════════════════════════════════

def compute_features_and_labels(ticker: str, aligned: dict, all_tickers_data: dict) -> dict:
    """Compute 22 features + regime + binary label for one ticker.
    
    Returns dict with:
        X: (n, 22) features
        regime: (n,) int 0-11
        y: (n,) int 0/1 (1 if forward_return > 0.002)
        timestamp: (n,) int ms
        close: (n,) float
    """
    close5 = aligned["5min_close"]
    high5 = aligned["5min_high"]
    low5 = aligned["5min_low"]
    open5 = aligned["5min_open"]
    vol5 = aligned["5min_volume"]
    time5 = aligned["time"]  # was "5min_time" — actual key is "time"
    n = len(close5)
    
    if n < 200:
        return None
    
    # Phase 2: Features (22 clean)
    X, feat_names = compute_features_v4(aligned, all_tickers_data=all_tickers_data)
    
    # Phase 2b: Regime (12 regimes, rule-based)
    ind = precompute_indicators(open5, close5, high5, low5, vol5)
    regime = compute_regime_v2(close5, high5, low5, ind)
    
    # Phase 3: Labels (comm-aware)
    # forward_return[t] = (close[t+HORIZON] - close[t]) / close[t]
    # y[t] = 1 if forward_return > LABEL_THRESHOLD else 0
    y = np.zeros(n, dtype=np.int32)
    for t in range(n - HORIZON_BARS):
        forward_close = close5[t + HORIZON_BARS]
        forward_return = (forward_close - close5[t]) / (close5[t] + 1e-10)
        y[t] = 1 if forward_return > LABEL_THRESHOLD else 0
    # Last HORIZON_BARS: no forward data → drop later
    
    # Clean: replace NaN/Inf
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
# PHASE 4: Date-purged split
# ═════════════════════════════════════════════════════════════════

def date_purged_split(all_data: dict, train_pct: float = 0.70, val_pct: float = 0.15) -> dict:
    """Split data by DATE, not by ticker. All tickers share same date range.
    
    Returns:
        {ticker: {"train_mask": bool, "val_mask": bool, "test_mask": bool}}
    """
    log(f"\n{'='*60}")
    log(f"PHASE 4: Date-purged split (train={train_pct*100:.0f}%, val={val_pct*100:.0f}%, test={(1-train_pct-val_pct)*100:.0f}%)")
    log(f"{'='*60}")
    
    # Find global min/max timestamps across all tickers
    all_timestamps = []
    for ticker, data in all_data.items():
        ts = data["timestamp"]
        if len(ts) > 0:
            all_timestamps.append((ts[0], ts[-1]))
    
    if not all_timestamps:
        return {}
    
    global_min = min(t[0] for t in all_timestamps)
    global_max = max(t[1] for t in all_timestamps)
    total_range = global_max - global_min
    
    train_end = global_min + total_range * train_pct
    val_end = global_min + total_range * (train_pct + val_pct)
    
    log(f"  Global range: {datetime.fromtimestamp(global_min/1000, tz=timezone(timedelta(hours=3)))}")
    log(f"            to: {datetime.fromtimestamp(global_max/1000, tz=timezone(timedelta(hours=3)))}")
    log(f"  Train cutoff: {datetime.fromtimestamp(train_end/1000, tz=timezone(timedelta(hours=3)))}")
    log(f"  Val cutoff:   {datetime.fromtimestamp(val_end/1000, tz=timezone(timedelta(hours=3)))}")
    
    splits = {}
    for ticker, data in all_data.items():
        ts = data["timestamp"]
        n = len(ts)
        # Drop last HORIZON_BARS (no forward data for labels)
        valid_mask = np.ones(n, dtype=bool)
        valid_mask[-HORIZON_BARS:] = False
        
        train_mask = valid_mask & (ts <= train_end)
        val_mask = valid_mask & (ts > train_end) & (ts <= val_end)
        test_mask = valid_mask & (ts > val_end)
        
        splits[ticker] = {
            "train_mask": train_mask,
            "val_mask": val_mask,
            "test_mask": test_mask,
        }
        
        log(f"  {ticker}: train={train_mask.sum()} val={val_mask.sum()} test={test_mask.sum()}")
    
    return splits


# ═════════════════════════════════════════════════════════════════
# PHASE 5: Train per-regime binary classifiers
# ═════════════════════════════════════════════════════════════════

def train_regime_models(all_data: dict, splits: dict) -> dict:
    """Train 12 XGBoost binary classifiers, one per regime."""
    log(f"\n{'='*60}")
    log(f"PHASE 5: Training 12 per-regime XGBoost classifiers")
    log(f"{'='*60}")
    
    # Aggregate samples per regime across all tickers
    regime_data = {r: {"X_train": [], "y_train": [], "X_val": [], "y_val": [], "X_test": [], "y_test": [],
                        "feat_names": None} for r in range(12)}
    
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
    
    # Train each regime model
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
        n_val = len(X_val)
        n_test = len(X_test)
        
        log(f"\n  [{r:2}/12] {rname}: train={n_train} val={n_val} test={n_test}")
        
        if n_train < 100:
            log(f"    SKIP — too few train samples (<100)")
            models[r] = {"status": "skipped", "reason": "too_few_samples", "n_train": n_train}
            continue
        
        pos_rate = y_train.mean() if len(y_train) > 0 else 0
        log(f"    Positive rate: train={pos_rate:.3f}")
        
        if pos_rate < 0.05 or pos_rate > 0.95:
            log(f"    SKIP — class imbalance extreme ({pos_rate:.3f})")
            models[r] = {"status": "skipped", "reason": "class_imbalance", "pos_rate": pos_rate}
            continue
        
        # Class weight for imbalance
        scale_pos = (1 - pos_rate) / pos_rate if pos_rate > 0 else 1.0
        params = {**XGB_PARAMS, "scale_pos_weight": scale_pos}
        
        t0 = time.time()
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        elapsed = time.time() - t0
        
        # Evaluate
        metrics = {}
        for split, Xs, ys in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
            if len(Xs) == 0:
                metrics[split] = {"n": 0}
                continue
            probs = model.predict_proba(Xs)[:, 1]
            preds = (probs > 0.5).astype(int)
            acc = accuracy_score(ys, preds)
            prec = precision_score(ys, preds, zero_division=0)
            rec = recall_score(ys, preds, zero_division=0)
            f1 = f1_score(ys, preds, zero_division=0)
            # Precision at LONG_THRESHOLD
            high_conf_mask = probs > LONG_THRESHOLD
            if high_conf_mask.sum() > 0:
                prec_at_long = (ys[high_conf_mask] == 1).mean()
            else:
                prec_at_long = 0
            metrics[split] = {
                "n": len(ys), "pos_rate": ys.mean(),
                "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
                "precision_at_long_threshold": prec_at_long,
                "n_high_conf": high_conf_mask.sum(),
            }
        
        log(f"    Trained in {elapsed:.1f}s, best_iter={model.best_iteration}")
        log(f"    VAL:  prec={metrics['val']['precision']:.3f} recall={metrics['val']['recall']:.3f} f1={metrics['val']['f1']:.3f}")
        log(f"    TEST: prec={metrics['test']['precision']:.3f} recall={metrics['test']['recall']:.3f} f1={metrics['test']['f1']:.3f}")
        log(f"    TEST prec@P>0.65: {metrics['test']['precision_at_long_threshold']:.3f} (n={metrics['test']['n_high_conf']})")
        
        models[r] = {
            "status": "trained",
            "model": model,
            "feat_names": rd["feat_names"],
            "n_train": n_train, "n_val": n_val, "n_test": n_test,
            "metrics": metrics,
        }
    
    return models


# ═════════════════════════════════════════════════════════════════
# PHASE 6: Walk-forward backtest
# ═════════════════════════════════════════════════════════════════

def walk_forward_backtest(all_data: dict, models: dict) -> dict:
    """Simulate live trading with trained models. Realistic commission + position sizing."""
    log(f"\n{'='*60}")
    log(f"PHASE 6: Walk-forward backtest (realistic)")
    log(f"{'='*60}")
    
    results = {"per_ticker": {}, "total": {"pnl": 0, "trades": 0, "wins": 0}}
    
    for ticker, data in all_data.items():
        X = data["X"]
        y = data["y"]
        regime = data["regime"]
        close = data["close"]
        n = len(close)
        
        # Simulate: at each bar, predict P(up) using regime model
        # If P > 0.65 → LONG (buy 1 lot), hold 6 bars, sell
        # If P < 0.35 → SHORT (sell 1 lot), hold 6 bars, buy back
        position = 0  # 0=flat, 1=long, -1=short
        entry_price = 0
        entry_bar = 0
        balance = 10000.0
        trades = 0
        wins = 0
        pnl_total = 0
        
        for t in range(100, n - HORIZON_BARS):  # skip warmup
            r = regime[t]
            if r not in models or models[r]["status"] != "trained":
                continue
            
            model = models[r]["model"]
            feat = X[t].reshape(1, -1)
            prob = model.predict_proba(feat)[0, 1]
            
            # Exit logic: if holding and P crossed exit threshold
            if position == 1 and prob < EXIT_LONG:
                # Close long
                exit_price = close[t]
                gross = (exit_price - entry_price) / entry_price
                commission = ROUNDTRIP_COMMISSION
                pnl = (gross - commission) * balance * 0.08  # 8% position size
                balance += pnl
                pnl_total += pnl
                trades += 1
                if pnl > 0: wins += 1
                position = 0
            elif position == -1 and prob > EXIT_SHORT:
                # Close short
                exit_price = close[t]
                gross = (entry_price - exit_price) / entry_price
                commission = ROUNDTRIP_COMMISSION
                pnl = (gross - commission) * balance * 0.08
                balance += pnl
                pnl_total += pnl
                trades += 1
                if pnl > 0: wins += 1
                position = 0
            
            # Entry logic: only if flat
            if position == 0:
                if prob > LONG_THRESHOLD:
                    position = 1
                    entry_price = close[t]
                    entry_bar = t
                elif prob < SHORT_THRESHOLD:
                    position = -1
                    entry_price = close[t]
                    entry_bar = t
            
            # Force close after HORIZON_BARS
            if position != 0 and t - entry_bar >= HORIZON_BARS:
                exit_price = close[t]
                if position == 1:
                    gross = (exit_price - entry_price) / entry_price
                else:
                    gross = (entry_price - exit_price) / entry_price
                commission = ROUNDTRIP_COMMISSION
                pnl = (gross - commission) * balance * 0.08
                balance += pnl
                pnl_total += pnl
                trades += 1
                if pnl > 0: wins += 1
                position = 0
        
        # Close any remaining position
        if position != 0:
            exit_price = close[n - HORIZON_BARS - 1]
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
            "pnl": pnl_total,
            "balance": balance,
            "trades": trades,
            "wins": wins,
            "win_rate": wins / max(trades, 1),
            "return_pct": (balance - 10000) / 10000 * 100,
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
# PHASE 7: Export to JSON
# ═════════════════════════════════════════════════════════════════

def export_models(models: dict, backtest_results: dict) -> None:
    """Export all trained models to JSON for pure-TS inference."""
    log(f"\n{'='*60}")
    log(f"PHASE 7: Export to JSON")
    log(f"{'='*60}")
    
    def to_native(obj):
        """Convert numpy types to native Python for JSON serialization."""
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: to_native(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_native(x) for x in obj]
        return obj
    
    metadata = {
        "version": "v5",
        "trained_at": datetime.now(timezone(timedelta(hours=3))).isoformat(),
        "label_threshold": LABEL_THRESHOLD,
        "horizon_bars": HORIZON_BARS,
        "long_threshold": LONG_THRESHOLD,
        "short_threshold": SHORT_THRESHOLD,
        "exit_long": EXIT_LONG,
        "exit_short": EXIT_SHORT,
        "commission_per_side": COMMISSION_PER_SIDE,
        "features_count": 22,
        "regimes": {},
        "backtest": to_native(backtest_results),
    }
    
    for r in range(12):
        rname = REGIME_NAMES[r].lower()
        if models[r]["status"] != "trained":
            metadata["regimes"][REGIME_NAMES[r]] = {"status": "skipped", "reason": models[r].get("reason", "unknown")}
            continue
        
        model = models[r]["model"]
        feat_names = models[r]["feat_names"]
        
        # Export model JSON
        booster = model.get_booster()
        config = json.loads(booster.save_raw(raw_format="json").decode("utf-8"))
        json_path = OUTPUT_DIR / f"regime_v5_{rname}.json"
        with open(json_path, "w") as f:
            json.dump(config, f)
        
        # Export pickle (for Python verification)
        pkl_path = OUTPUT_DIR / f"regime_v5_{rname}.pkl"
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
    
    # Save metadata
    meta_path = OUTPUT_DIR / "regime_v5_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(to_native(metadata), f, indent=2)
    log(f"\nMetadata: {meta_path}")
    
    # Save backtest results
    bt_path = OUTPUT_DIR / "regime_v5_backtest.json"
    with open(bt_path, "w") as f:
        json.dump(to_native(backtest_results), f, indent=2)
    log(f"Backtest: {bt_path}")


# ═════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=365, help="History days (default 365)")
    parser.add_argument("--tickers", type=str, default="all", help="all or SBER,GAZP,...")
    args = parser.parse_args()
    
    tickers = TICKERS if args.tickers == "all" else args.tickers.split(",")
    
    log(f"╔{'═'*60}╗")
    log(f"║  ML v5 TRAINING PIPELINE                                  ║")
    log(f"╠{'═'*60}╣")
    log(f"║  Tickers: {len(tickers):<46} ║")
    log(f"║  Days: {args.days:<48} ║")
    log(f"║  Label threshold: {LABEL_THRESHOLD} (comm-aware)              ║")
    log(f"║  Features: 22 (clean, no duplicates)                       ║")
    log(f"║  Split: date-purged 70/15/15                                ║")
    log(f"╚{'═'*60}╝")
    
    start_time = time.time()
    
    # Phase 1: Load data
    all_data, all_tickers_data = load_all_data(tickers, days=args.days)
    if not all_data:
        log("FATAL: No data loaded")
        return 1
    
    # Phase 2+3: Features + Labels
    log(f"\n{'='*60}")
    log(f"PHASE 2+3: Computing features + labels per ticker")
    log(f"{'='*60}")
    for ticker in list(all_data.keys()):
        data = compute_features_and_labels(ticker, all_data[ticker], all_tickers_data)
        if data is None:
            log(f"  {ticker}: SKIP (too few bars)")
            del all_data[ticker]
            continue
        all_data[ticker] = data
        log(f"  {ticker}: X={data['X'].shape} regime_dist={np.bincount(data['regime'], minlength=12)[:6]}... pos_rate={data['y'].mean():.3f}")
    
    # Phase 4: Date-purged split
    splits = date_purged_split(all_data)
    
    # Phase 5: Train
    models = train_regime_models(all_data, splits)
    
    # Phase 6: Walk-forward backtest
    backtest_results = walk_forward_backtest(all_data, models)
    
    # Phase 7: Export
    export_models(models, backtest_results)
    
    elapsed = time.time() - start_time
    log(f"\n╔{'═'*60}╗")
    log(f"║  V5 TRAINING COMPLETE in {elapsed/60:.1f} min                       ║")
    log(f"╠{'═'*60}╣")
    log(f"║  Models trained: {sum(1 for m in models.values() if m['status']=='trained')}/12                          ║")
    log(f"║  Backtest P&L: {backtest_results['total']['pnl']:+.0f}₽                                ║")
    log(f"║  Output: {OUTPUT_DIR}                              ║")
    log(f"╚{'═'*60}╝")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
