#!/usr/bin/env python3
"""Train 12 binary XGBoost classifiers — one per market regime (ML v4).

For each of 12 regimes (STRONG_TREND_UP, MILD_TREND_UP, RANGE_TIGHT, RANGE_WIDE,
MILD_TREND_DOWN, STRONG_TREND_DOWN, CRASH, OVERSOLD_BOUNCE,
OVERBOUGHT_REVERSAL, BREAKOUT_UP, BREAKDOWN, HIGH_VOL_REGIME):

  1. Filter bars to that regime
  2. If < 100 samples: skip (rule-based fallback will handle inference)
  3. Compute binary label: y = 1 if forward_return > 0.001 else 0
     where forward_return = close[t+6] / close[t] - 1 (≈30min ahead)
  4. Train XGBoost binary classifier with strong regularization
     (max_depth=3, reg_lambda=10, min_child_weight=30)
  5. Chronological split: 70% train / 15% val / 15% test (per-ticker)
  6. Save .pkl + .json (via booster.save_raw(raw_format='json'))

Decision rule for inference:
    if P(up) > 0.6  → go LONG
    if P(up) < 0.4  → go SHORT
    otherwise       → FLAT

Outputs (in /root/ai-trader-evolution/ml/meta_models_v2/):
  regime_<name>.pkl          — full sklearn XGBClassifier (for backtest)
  regime_<name>.json         — XGBoost native JSON (for TS inference)
  regime_models_v4_metadata.json — per-regime metadata (n_samples, val_precision, val_f1, ...)

Usage:
  python3 train_regime_models_v4.py --days 180
  python3 train_regime_models_v4.py --days 180 --tickers SBER,GAZP
"""
import os
import sys
import json
import time
import pickle
import argparse
import traceback
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — make /root/ai-trader-evolution/ml + /root/ai-trader-evolution/fast_mc importable
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))                                # ml/
sys.path.insert(0, str(SCRIPT_DIR.parent / "fast_mc"))             # fast_mc/

import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from ml_data_pipeline import download_multi_timeframe, align_timeframes, TICKERS
from ml_features import compute_features
from meta_labeler_v2 import compute_regime_v2, REGIME_NAMES
from fast_backtest_v2 import precompute_indicators

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("/root/ai-trader-evolution/ml/meta_models_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = "/var/log/ai-trader-regime-train-v4.log"

# XGBoost params (per task spec) — strong regularization to avoid overfit
XGB_PARAMS = dict(
    n_estimators=150,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=30,
    gamma=0.5,
    reg_alpha=0.5,
    reg_lambda=10.0,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    n_jobs=2,
    early_stopping_rounds=30,
)

# Decision thresholds (documented in metadata, used by TS inference)
LONG_THRESHOLD = 0.6
SHORT_THRESHOLD = 0.4

# Label config
DEFAULT_HORIZON = 6       # 6 bars × 5min = 30min forward
DEFAULT_THRESHOLD = 0.001  # 0.1% minimum forward return to label y=1

# Skip regimes with fewer than this many total samples
MIN_SAMPLES_PER_REGIME = 100
# Require this many samples per split (train/val/test)
MIN_TRAIN_SAMPLES = 50
MIN_VAL_SAMPLES = 10
MIN_TEST_SAMPLES = 10
# Require at least this many positives AND negatives in train set
MIN_POS_NEG = 30
# Clip scale_pos_weight to avoid extreme values
SPW_MIN, SPW_MAX = 0.1, 10.0


def log(msg: str) -> None:
    """Print + append to log file with MSK timestamp."""
    msk = timezone(timedelta(hours=3))
    ts = datetime.now(msk).strftime("%Y-%m-%d %H:%M:%S МСК")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Label computation
# ---------------------------------------------------------------------------
def compute_binary_label(close5: np.ndarray,
                          horizon: int = DEFAULT_HORIZON,
                          threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    """Compute binary label y = 1 if forward_return > threshold else 0.

    forward_return = (close[t + horizon] - close[t]) / close[t]

    Last `horizon` bars have no forward close — set to -1 (sentinel for "drop").

    Args:
        close5: 5-min (actually 10-min in v1 pipeline, but "5min" naming kept) close prices
        horizon: forward window in bars
        threshold: minimum forward return to label as 1 (0.001 = 0.1%)

    Returns:
        int array of length n; values in {0, 1, -1}. -1 = drop (no future data).
    """
    n = len(close5)
    y = np.full(n, -1, dtype=np.int32)
    if n <= horizon:
        return y
    # Forward close: shift by `horizon`. Last `horizon` bars use placeholder (dropped).
    forward_close = np.empty(n)
    forward_close[:-horizon] = close5[horizon:]
    forward_close[-horizon:] = close5[-1]
    forward_return = (forward_close - close5) / (close5 + 1e-10)
    y = (forward_return > threshold).astype(np.int32)
    y[-horizon:] = -1  # mark as drop
    return y


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_one_regime_model(X_train, y_train, X_val, y_val, X_test, y_test,
                           regime_name: str) -> dict:
    """Train one XGBoost binary classifier for a single regime.

    Returns dict with 'model' and 'metrics', or None if too few pos/neg samples.
    """
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)

    if pos < MIN_POS_NEG or neg < MIN_POS_NEG:
        log(f"  [{regime_name}] SKIP: insufficient pos/neg (pos={pos}, neg={neg})")
        return None

    # scale_pos_weight handles class imbalance (long-biased market → more "up" labels)
    spw = neg / max(1, pos)
    spw = float(max(SPW_MIN, min(SPW_MAX, spw)))

    log(f"  [{regime_name}] train: n={len(y_train)}, pos={pos} ({pos*100/len(y_train):.1f}%), "
        f"neg={neg}, scale_pos_weight={spw:.3f}")
    log(f"  [{regime_name}] val:   n={len(y_val)}, pos={int(y_val.sum())} ({y_val.mean()*100:.1f}%)")
    log(f"  [{regime_name}] test:  n={len(y_test)}, pos={int(y_test.sum())} ({y_test.mean()*100:.1f}%)")

    model = xgb.XGBClassifier(
        **XGB_PARAMS,
        scale_pos_weight=spw,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    best_iter = getattr(model, "best_iteration", None)
    if best_iter is None:
        best_iter = XGB_PARAMS["n_estimators"]

    # --- Validation metrics ---
    y_pred_val = (model.predict_proba(X_val)[:, 1] > 0.5).astype(int)
    val_precision = precision_score(y_val, y_pred_val, zero_division=0)
    val_recall = recall_score(y_val, y_pred_val, zero_division=0)
    val_f1 = f1_score(y_val, y_pred_val, zero_division=0)
    val_acc = accuracy_score(y_val, y_pred_val)

    # --- Test metrics ---
    y_pred_test = (model.predict_proba(X_test)[:, 1] > 0.5).astype(int)
    test_precision = precision_score(y_test, y_pred_test, zero_division=0)
    test_recall = recall_score(y_test, y_pred_test, zero_division=0)
    test_f1 = f1_score(y_test, y_pred_test, zero_division=0)
    test_acc = accuracy_score(y_test, y_pred_test)

    # --- Precision @ 0.6 threshold (the actual LONG_THRESHOLD used in inference) ---
    y_proba_test = model.predict_proba(X_test)[:, 1]
    n_high_conf = int((y_proba_test > LONG_THRESHOLD).sum())
    if n_high_conf >= 10:
        prec_at_06 = precision_score(
            y_test[y_proba_test > LONG_THRESHOLD],
            y_pred_test[y_proba_test > LONG_THRESHOLD],
            zero_division=0,
        )
    else:
        prec_at_06 = None

    log(f"  [{regime_name}] VAL:  precision={val_precision*100:.1f}% recall={val_recall*100:.1f}% "
        f"f1={val_f1*100:.1f}% acc={val_acc*100:.1f}%")
    log(f"  [{regime_name}] TEST: precision={test_precision*100:.1f}% recall={test_recall*100:.1f}% "
        f"f1={test_f1*100:.1f}% acc={test_acc*100:.1f}%")
    if prec_at_06 is not None:
        log(f"  [{regime_name}] TEST precision@P>0.6: {prec_at_06*100:.1f}% (n={n_high_conf})")
    log(f"  [{regime_name}] best_iteration={best_iter}")

    return {
        "model": model,
        "metrics": {
            "n_train": int(len(y_train)),
            "train_pos": int(pos),
            "train_neg": int(neg),
            "scale_pos_weight": float(spw),
            "best_iteration": int(best_iter),
            # validation
            "val_precision": float(val_precision),
            "val_recall": float(val_recall),
            "val_f1": float(val_f1),
            "val_accuracy": float(val_acc),
            "val_n": int(len(y_val)),
            # test
            "test_precision": float(test_precision),
            "test_recall": float(test_recall),
            "test_f1": float(test_f1),
            "test_accuracy": float(test_acc),
            "test_n": int(len(y_test)),
            # high-confidence precision
            "test_precision_at_0.6": float(prec_at_06) if prec_at_06 is not None else None,
            "n_high_confidence_test": n_high_conf,
        },
    }


# ---------------------------------------------------------------------------
# Save model artifacts (.pkl + .json)
# ---------------------------------------------------------------------------
def save_model_files(model: xgb.XGBClassifier,
                     regime_name: str,
                     feature_names: list) -> tuple:
    """Save .pkl (sklearn model) and .json (XGBoost native JSON) for one regime.

    Returns (pkl_path, json_path).
    """
    safe = regime_name.lower()
    pkl_path = OUTPUT_DIR / f"regime_{safe}.pkl"
    json_path = OUTPUT_DIR / f"regime_{safe}.json"

    # .pkl — full XGBClassifier wrapped in a dict with metadata
    with open(pkl_path, "wb") as f:
        pickle.dump({
            "model": model,
            "feature_names": feature_names,
            "regime": regime_name,
            "long_threshold": LONG_THRESHOLD,
            "short_threshold": SHORT_THRESHOLD,
        }, f)

    # .json — XGBoost native JSON via booster.save_raw(raw_format='json')
    # This is the format XGBoost itself loads (full model spec including
    # feature_names, learner params, trees). The TS side parses this.
    booster = model.get_booster()
    raw = booster.save_raw(raw_format="json")
    with open(json_path, "wb") as f:
        f.write(raw)

    return pkl_path, json_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=180, help="History depth in days")
    parser.add_argument("--tickers", type=str, default="all",
                        help="Comma-separated tickers (default: all 11)")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON,
                        help=f"Forward horizon in bars (default {DEFAULT_HORIZON} = 30min)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Min forward return to label as up (default {DEFAULT_THRESHOLD})")
    parser.add_argument("--min-samples", type=int, default=MIN_SAMPLES_PER_REGIME,
                        help=f"Min samples per regime to train (default {MIN_SAMPLES_PER_REGIME})")
    args = parser.parse_args()

    tickers = TICKERS if args.tickers == "all" else args.tickers.split(",")
    min_samples = args.min_samples

    log("=" * 78)
    log("🚀 Regime-Aware ML Trainer V4 — 12 binary XGBoost classifiers (one per regime)")
    log(f"   days={args.days}, tickers={len(tickers)}, horizon={args.horizon} bars "
        f"(~{args.horizon*5}min), threshold={args.threshold}")
    log(f"   min_samples_per_regime={min_samples}, xgb max_depth=3, reg_lambda=10")
    log(f"   decision rule: P(up)>{LONG_THRESHOLD} → LONG, P(up)<{SHORT_THRESHOLD} → SHORT, else FLAT")
    log("=" * 78)

    # === Step 1: Load all data ===
    log("\n[Step 1/5] Loading multi-timeframe data for all tickers...")
    t0 = time.time()
    all_aligned = {}
    for i, ticker in enumerate(tickers):
        log(f"  [{i+1}/{len(tickers)}] {ticker}...")
        try:
            data = download_multi_timeframe(ticker, days=args.days)
            if "5min_close" not in data:
                log(f"    SKIP: no 5min data")
                continue
            aligned = align_timeframes(data)
            all_aligned[ticker] = aligned
        except Exception as e:
            log(f"    ERROR: {e}")
    log(f"  Loaded {len(all_aligned)}/{len(tickers)} tickers in {time.time()-t0:.1f}s")

    if not all_aligned:
        log("FATAL: no data loaded")
        return 1

    # === Step 2: Compute features + regime + binary label per ticker ===
    log("\n[Step 2/5] Computing features + regimes + binary labels per ticker...")
    t0 = time.time()
    ticker_arrays = []  # list of (ticker, X, regime, y, n)
    feature_names = None
    for i, (ticker, aligned) in enumerate(all_aligned.items()):
        log(f"  [{i+1}/{len(all_aligned)}] {ticker}: features + regime + label...")
        try:
            X, names = compute_features(aligned)
            if feature_names is None:
                feature_names = list(names)
            elif list(names) != feature_names:
                log(f"    ⚠️ feature mismatch: got {len(names)}, expected {len(feature_names)} — using intersection")
                common = sorted(set(names) & set(feature_names))
                if not common:
                    log(f"    SKIP: no common features with previous tickers")
                    continue
                old_idx = {n: j for j, n in enumerate(names)}
                new_idx = {n: j for j, n in enumerate(feature_names)}
                # Remap current ticker's X to common features order
                X_new = np.zeros((len(X), len(common)))
                for j, cn in enumerate(common):
                    X_new[:, j] = X[:, old_idx[cn]]
                X = X_new
                feature_names = common
                # Remap previous tickers' X to common features order
                new_arrays = []
                for t, X_t, reg_t, y_t, n_t in ticker_arrays:
                    X_t_new = np.zeros((len(X_t), len(common)))
                    # We didn't save the per-ticker feature_names order before.
                    # Since align_timeframes always fills higher TFs, previous tickers
                    # should have used the same feature_names as the first ticker.
                    # If we're here, something is off — but assume the first ticker's
                    # features matched `feature_names` (pre-shrink) and remap by position.
                    if X_t.shape[1] >= len(common):
                        # Best-effort: drop extra columns from the end
                        X_t_new = X_t[:, :len(common)]
                    else:
                        X_t_new[:, :X_t.shape[1]] = X_t
                    new_arrays.append((t, X_t_new, reg_t, y_t, n_t))
                ticker_arrays = new_arrays

            close5 = aligned["5min_close"]
            high5 = aligned["5min_high"]
            low5 = aligned["5min_low"]
            open5 = aligned["5min_open"]
            vol5 = aligned["5min_volume"]

            # Regime via meta_labeler_v2.compute_regime_v2 (uses precompute_indicators)
            ind = precompute_indicators(open5, close5, high5, low5, vol5)
            regime = compute_regime_v2(close5, high5, low5, ind)

            # Binary label: y = 1 if forward_return > threshold else 0; -1 = drop
            y = compute_binary_label(close5, horizon=args.horizon, threshold=args.threshold)

            ticker_arrays.append((ticker, X, regime, y, len(X)))

            # Stats
            valid = y >= 0
            log(f"    X={X.shape}, regimes computed, y positive_rate (valid bars)="
                f"{y[valid].sum()}/{valid.sum()} = {y[valid].mean() if valid.sum() else 0:.3f}")
            # Per-regime count for this ticker
            r_counts = {REGIME_NAMES[r]: int((regime[valid] == r).sum()) for r in range(len(REGIME_NAMES))}
            log(f"    regime counts: " + ", ".join(f"{k}={v}" for k, v in r_counts.items() if v > 0))
        except Exception as e:
            log(f"    ERROR: {e}")
            log(traceback.format_exc())
    log(f"  Step 2 done in {time.time()-t0:.1f}s")

    if not ticker_arrays:
        log("FATAL: no ticker arrays built")
        return 1

    # === Step 3: Concatenate + chronological split per ticker ===
    log("\n[Step 3/5] Building chronological train/val/test splits (per-ticker, 70/15/15)...")

    X = np.vstack([t[1] for t in ticker_arrays])
    regime = np.concatenate([t[2] for t in ticker_arrays])
    y = np.concatenate([t[3] for t in ticker_arrays])
    log(f"  Total bars: {len(X)}, features: {len(feature_names)}")
    valid = y >= 0
    log(f"  Valid (labelled) bars: {valid.sum()}/{len(y)}")
    log(f"  Positive (up) rate over all valid bars: {y[valid].sum()}/{valid.sum()} "
        f"= {y[valid].mean() if valid.sum() else 0:.4f}")

    # Chronological split per ticker — concatenate per-ticker train/val/test slices
    train_idx_list, val_idx_list, test_idx_list = [], [], []
    offset = 0
    for ticker, X_t, regime_t, y_t, n_t in ticker_arrays:
        train_end = int(n_t * 0.70)
        val_end = int(n_t * 0.85)
        train_idx_list.append(np.arange(offset, offset + train_end))
        val_idx_list.append(np.arange(offset + train_end, offset + val_end))
        test_idx_list.append(np.arange(offset + val_end, offset + n_t))
        offset += n_t
    train_idx = np.concatenate(train_idx_list)
    val_idx = np.concatenate(val_idx_list)
    test_idx = np.concatenate(test_idx_list)

    # Drop bars with y == -1 (no forward return)
    valid_train = y[train_idx] >= 0
    valid_val = y[val_idx] >= 0
    valid_test = y[test_idx] >= 0
    train_idx = train_idx[valid_train]
    val_idx = val_idx[valid_val]
    test_idx = test_idx[valid_test]
    log(f"  After dropping no-label bars:")
    log(f"    Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # === Step 4: Train one model per regime ===
    log("\n[Step 4/5] Training 12 regime-specific binary classifiers...")
    t0 = time.time()
    metadata = {}
    summary_rows = []
    n_trained, n_skipped = 0, 0

    for r_id, r_name in enumerate(REGIME_NAMES):
        log(f"\n  Regime {r_id}: {r_name}")

        # Filter by regime
        train_r_mask = regime[train_idx] == r_id
        val_r_mask = regime[val_idx] == r_id
        test_r_mask = regime[test_idx] == r_id

        X_tr = X[train_idx[train_r_mask]]
        y_tr = y[train_idx[train_r_mask]]
        X_va = X[val_idx[val_r_mask]]
        y_va = y[val_idx[val_r_mask]]
        X_te = X[test_idx[test_r_mask]]
        y_te = y[test_idx[test_r_mask]]

        n_total = int(len(X_tr) + len(X_va) + len(X_te))
        log(f"    n_samples={n_total} (train={len(X_tr)}, val={len(X_va)}, test={len(X_te)})")

        # Skip checks
        if n_total < min_samples:
            log(f"    SKIP: < {min_samples} samples — rule-based fallback will handle")
            metadata[r_name] = {
                "model_file": None,
                "pkl_file": None,
                "n_samples": n_total,
                "n_train": int(len(X_tr)),
                "n_val": int(len(X_va)),
                "n_test": int(len(X_te)),
                "val_precision": None,
                "val_recall": None,
                "val_f1": None,
                "val_accuracy": None,
                "test_precision": None,
                "test_recall": None,
                "test_f1": None,
                "test_accuracy": None,
                "best_iteration": None,
                "scale_pos_weight": None,
                "feature_names": feature_names,
                "skipped": True,
                "reason": f"insufficient_samples ({n_total} < {min_samples})",
            }
            summary_rows.append((r_name, n_total, None, None, None, None))
            n_skipped += 1
            continue

        if (len(X_tr) < MIN_TRAIN_SAMPLES or len(X_va) < MIN_VAL_SAMPLES
                or len(X_te) < MIN_TEST_SAMPLES):
            log(f"    SKIP: split too small (train={len(X_tr)}<{MIN_TRAIN_SAMPLES} OR "
                f"val={len(X_va)}<{MIN_VAL_SAMPLES} OR test={len(X_te)}<{MIN_TEST_SAMPLES})")
            metadata[r_name] = {
                "model_file": None,
                "pkl_file": None,
                "n_samples": n_total,
                "n_train": int(len(X_tr)),
                "n_val": int(len(X_va)),
                "n_test": int(len(X_te)),
                "val_precision": None, "val_recall": None, "val_f1": None, "val_accuracy": None,
                "test_precision": None, "test_recall": None, "test_f1": None, "test_accuracy": None,
                "best_iteration": None, "scale_pos_weight": None,
                "feature_names": feature_names,
                "skipped": True,
                "reason": "split_too_small",
            }
            summary_rows.append((r_name, n_total, None, None, None, None))
            n_skipped += 1
            continue

        # Train
        result = train_one_regime_model(X_tr, y_tr, X_va, y_va, X_te, y_te, r_name)
        if result is None:
            metadata[r_name] = {
                "model_file": None,
                "pkl_file": None,
                "n_samples": n_total,
                "n_train": int(len(X_tr)),
                "n_val": int(len(X_va)),
                "n_test": int(len(X_te)),
                "val_precision": None, "val_recall": None, "val_f1": None, "val_accuracy": None,
                "test_precision": None, "test_recall": None, "test_f1": None, "test_accuracy": None,
                "best_iteration": None, "scale_pos_weight": None,
                "feature_names": feature_names,
                "skipped": True,
                "reason": "too_few_pos_or_neg_in_train",
            }
            summary_rows.append((r_name, n_total, None, None, None, None))
            n_skipped += 1
            continue

        # Save model artifacts
        pkl_path, json_path = save_model_files(result["model"], r_name, feature_names)
        log(f"    Saved: {pkl_path.name} ({pkl_path.stat().st_size//1024} KB), "
            f"{json_path.name} ({json_path.stat().st_size//1024} KB)")

        m = result["metrics"]
        metadata[r_name] = {
            "model_file": json_path.name,
            "pkl_file": pkl_path.name,
            "n_samples": n_total,
            "n_train": m["n_train"],
            "n_val": m["val_n"],
            "n_test": m["test_n"],
            "train_positive_rate": float(m["train_pos"] / max(1, m["n_train"])),
            "val_precision": m["val_precision"],
            "val_recall": m["val_recall"],
            "val_f1": m["val_f1"],
            "val_accuracy": m["val_accuracy"],
            "test_precision": m["test_precision"],
            "test_recall": m["test_recall"],
            "test_f1": m["test_f1"],
            "test_accuracy": m["test_accuracy"],
            "best_iteration": m["best_iteration"],
            "scale_pos_weight": m["scale_pos_weight"],
            "test_precision_at_0.6": m["test_precision_at_0.6"],
            "n_high_confidence_test": m["n_high_confidence_test"],
            "feature_names": feature_names,
            "skipped": False,
            "reason": None,
        }
        summary_rows.append((
            r_name, n_total,
            m["val_precision"], m["val_f1"],
            m["test_precision"], m["test_f1"],
        ))
        n_trained += 1
    log(f"\n  Step 4 done in {time.time()-t0:.1f}s. Trained={n_trained}, Skipped={n_skipped}")

    # === Step 5: Save metadata ===
    log("\n[Step 5/5] Saving metadata...")
    meta_path = OUTPUT_DIR / "regime_models_v4_metadata.json"
    # Top-level: 12 regime names (per task spec) + a single "_meta" key for global info.
    full_meta = {**metadata}
    full_meta["_meta"] = {
        "version": "v4",
        "trained_at": datetime.now(timezone(timedelta(hours=3))).isoformat(),
        "days_of_history": args.days,
        "tickers": [t[0] for t in ticker_arrays],
        "horizon_bars": args.horizon,
        "horizon_minutes": args.horizon * 5,
        "threshold": args.threshold,
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "xgb_params": {**XGB_PARAMS},
        "n_regimes": len(REGIME_NAMES),
        "n_trained": n_trained,
        "n_skipped": n_skipped,
        "min_samples_per_regime": min_samples,
        "long_threshold": LONG_THRESHOLD,
        "short_threshold": SHORT_THRESHOLD,
        "decision_rule": "P(up) > 0.6 → LONG, P(up) < 0.4 → SHORT, else FLAT",
        "fallback": "if model_file is null (skipped), use rule-based strategy from regime_strategy_mapping.json",
        "regime_names": list(REGIME_NAMES),
    }
    with open(meta_path, "w") as f:
        json.dump(full_meta, f, indent=2, ensure_ascii=False)
    log(f"  Saved: {meta_path} ({meta_path.stat().st_size//1024} KB)")

    # Also save a flat "train_summary.json" for easy scanning
    summary_path = OUTPUT_DIR / "regime_models_v4_train_summary.json"
    summary_data = {
        "trained_at": full_meta["_meta"]["trained_at"],
        "days_of_history": args.days,
        "n_trained": n_trained,
        "n_skipped": n_skipped,
        "feature_names": feature_names,
        "long_threshold": LONG_THRESHOLD,
        "short_threshold": SHORT_THRESHOLD,
        "results": [
            {
                "regime": r_name,
                "n_samples": n_s,
                "val_precision": vp,
                "val_f1": vf,
                "test_precision": tp,
                "test_f1": tf,
                "model_file": metadata[r_name]["model_file"],
            }
            for (r_name, n_s, vp, vf, tp, tf) in summary_rows
        ],
    }
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    log(f"  Saved: {summary_path}")

    # === Final summary table ===
    log("\n" + "=" * 92)
    log("📊 SUMMARY — 12 Regime Models (binary: P(price up > 0.1% in next 30min))")
    log("=" * 92)
    header = f"{'Regime':<24} {'N':>7} {'ValP':>7} {'ValF1':>7} {'TestP':>7} {'TestF1':>7}  Status"
    log(header)
    log("-" * 92)
    for r_name, n_s, vp, vf, tp, tf in summary_rows:
        if vp is None:
            log(f"{r_name:<24} {n_s:>7}   ----    ----    ----    ----   SKIP")
        else:
            log(f"{r_name:<24} {n_s:>7} {vp:>7.3f} {vf:>7.3f} {tp:>7.3f} {tf:>7.3f}   OK")
    log("=" * 92)
    log(f"Trained: {n_trained}/12, Skipped: {n_skipped}/12")
    log(f"📁 Models dir:        {OUTPUT_DIR}")
    log(f"📁 Metadata JSON:     {meta_path}")
    log(f"📁 Train summary:     {summary_path}")
    log(f"📝 Log file:          {LOG_FILE}")
    log("=" * 92)
    log("✅ DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
