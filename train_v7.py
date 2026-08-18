#!/usr/bin/env python3
"""ML v7 Training — ensemble (XGBoost + LightGBM + CatBoost) per regime.

Improvements over V6:
  1. 36 features (features_v7.py): 22 v4 + 14 new (VWAP, volume profile,
     order flow, macro, intraday structure)
  2. 3 base models per regime = 36 models total (vs 12 in V6)
  3. Stacking: LogisticRegression meta-model over (xgb, lgb, cat) probabilities
  4. Final prediction = stacking output (data-driven weights, not average)
  5. Date-purged global split: 60% train / 20% val / 20% test (no leakage)
  6. Per-regime class balance handling (scale_pos_weight / is_unbalance / auto_class_weights)
  7. Export EVERY model to JSON for TS-side inference

Output: /root/ai-trader-evolution/ml/meta_models_v7/
  - regime_v7_<name>_xgb.json   (12 files)
  - regime_v7_<name>_lgb.json   (12 files)
  - regime_v7_<name>_cat.json   (12 files)
  - regime_v7_stacking.json     (1 file, per-regime logistic regression coeffs)
  - regime_v7_metadata.json     (1 file, full metrics + features + params)
"""
import os
import sys
import json
import time
import pickle
import argparse
import warnings
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from pathlib import Path
from datetime import datetime, timezone, timedelta
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "fast_mc"))

from ml_data_pipeline import download_multi_timeframe, align_timeframes, TICKERS
from features_v7 import compute_features_v7, fetch_macro_data
from meta_labeler_v2 import compute_regime_v2, REGIME_NAMES, REGIME_TO_IDX
from fast_backtest_v2 import precompute_indicators

OUTPUT_DIR = Path("/root/ai-trader-evolution/ml/meta_models_v7")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = "/var/log/ai-trader-v7-train.log"

# ═════════════════════════════════════════════════════════════════
# V7 CONSTANTS
# ═════════════════════════════════════════════════════════════════

COMMISSION_PER_SIDE = 0.0005
ROUNDTRIP_COMMISSION = 0.001
PROFIT_MARGIN = 0.001
LABEL_THRESHOLD = ROUNDTRIP_COMMISSION + PROFIT_MARGIN  # 0.002
HORIZON_BARS = 6  # 30 min forward (same as V6)

# Split ratios (date-purged)
TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
TEST_RATIO = 0.20

# Minimum samples per regime per split for training
MIN_TRAIN_SAMPLES = 200
MIN_VAL_SAMPLES = 50

# XGBoost — same as V6
XGB_PARAMS = {
    "n_estimators": 250,
    "max_depth": 4,
    "learning_rate": 0.04,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 50,
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

# LightGBM — V7 spec (n_estimators=200, max_depth=4, learning_rate=0.05)
# NB: removed early_stopping — it was exiting at iter 8 (model underfit),
# so output probabilities never crossed 0.5 (LGB scale_pos_weight only
# affects training, not inference). Now we train all 200 trees.
LGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "num_leaves": 15,            # <= 2^max_depth to limit complexity
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.7,
    "min_child_samples": 20,    # lower than 50 so leaves can specialize
    "reg_alpha": 1.0,
    "reg_lambda": 5.0,
    "objective": "binary",
    "metric": ["binary_logloss", "auc"],
    "random_state": 42,
    "n_jobs": 2,
    "verbose": -1,
}

# CatBoost — V7 spec
CAT_PARAMS = {
    "iterations": 200,
    "depth": 4,
    "learning_rate": 0.05,
    "l2_leaf_reg": 5.0,
    "rsm": 0.7,                  # column subsample
    "min_data_in_leaf": 50,
    "loss_function": "Logloss",
    "eval_metric": "Logloss",
    "random_seed": 42,
    "verbose": False,
    "allow_writing_files": False,
}

# Stacking meta-model (per regime): LogisticRegression on 3 base probs
# Higher C (less L2 regularization) so LR can produce output probabilities
# that span the full [0, 1] range — otherwise output is cramped near the
# prior (0.2-0.4) and never crosses 0.6 / 0.7 thresholds.
STACK_PARAMS = {
    "C": 10.0,
    "penalty": "l2",
    "solver": "lbfgs",
    "max_iter": 1000,
    "class_weight": "balanced",
    "random_state": 42,
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


def to_native(obj):
    """Recursively convert numpy types to native Python for JSON."""
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


# ═════════════════════════════════════════════════════════════════
# Phase 1+2+3: Data + Features + Labels
# ═════════════════════════════════════════════════════════════════

def compute_features_and_labels(ticker: str, aligned: dict, all_tickers_data: dict,
                                macro_data: dict) -> dict:
    close5 = aligned["5min_close"]
    high5 = aligned["5min_high"]
    low5 = aligned["5min_low"]
    open5 = aligned["5min_open"]
    vol5 = aligned["5min_volume"]
    time5 = aligned["time"]
    n = len(close5)

    if n < 200:
        return None

    X, feat_names = compute_features_v7(aligned, all_tickers_data=all_tickers_data,
                                        macro_data=macro_data)
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
# Phase 4: Date-purged global split (60/20/20)
# ═════════════════════════════════════════════════════════════════

def date_purged_splits(all_data: dict) -> dict:
    """Single global time-based split — no overlap across tickers."""
    log(f"\n{'='*60}")
    log(f"PHASE 4: Date-purged split (train={TRAIN_RATIO:.0%} val={VAL_RATIO:.0%} test={TEST_RATIO:.0%})")
    log(f"{'='*60}")

    all_ts = []
    for ticker, data in all_data.items():
        if len(data["timestamp"]) > 0:
            all_ts.append((data["timestamp"][0], data["timestamp"][-1]))
    if not all_ts:
        raise RuntimeError("No data for split")

    global_min = min(t[0] for t in all_ts)
    global_max = max(t[1] for t in all_ts)
    total_range = global_max - global_min

    train_end = global_min + total_range * TRAIN_RATIO
    val_end = train_end + total_range * VAL_RATIO
    # test_end = global_max (everything past val_end)

    splits_per_ticker = {}
    for ticker, data in all_data.items():
        ts = data["timestamp"]
        n = len(ts)
        # Last HORIZON_BARS bars have no label (forward return unknown)
        valid_mask = np.ones(n, dtype=bool)
        valid_mask[-HORIZON_BARS:] = False

        train_mask = valid_mask & (ts < train_end)
        val_mask = valid_mask & (ts >= train_end) & (ts < val_end)
        test_mask = valid_mask & (ts >= val_end)

        splits_per_ticker[ticker] = {
            "train_mask": train_mask,
            "val_mask": val_mask,
            "test_mask": test_mask,
        }

    def fmt(ms):
        return datetime.fromtimestamp(ms / 1000, tz=timezone(timedelta(hours=3))).strftime("%Y-%m-%d")

    log(f"  train: < {fmt(train_end)}  ({TRAIN_RATIO:.0%})")
    log(f"  val:   {fmt(train_end)} → {fmt(val_end)}  ({VAL_RATIO:.0%})")
    log(f"  test:  {fmt(val_end)} → {fmt(global_max)}  ({TEST_RATIO:.0%})")

    return {
        "train_end_ms": float(train_end),
        "val_end_ms": float(val_end),
        "global_min_ms": float(global_min),
        "global_max_ms": float(global_max),
        "splits": splits_per_ticker,
    }


# ═════════════════════════════════════════════════════════════════
# Phase 5: Train 3 base models per regime + stacking meta-model
# ═════════════════════════════════════════════════════════════════

def _aggregate_regime_data(all_data: dict, splits: dict, n_features: int) -> dict:
    """Pool bars per regime across tickers, then split by global masks."""
    regime_data = {
        r: {"X_train": [], "y_train": [], "X_val": [], "y_val": [],
            "X_test": [], "y_test": [], "feat_names": None}
        for r in range(12)
    }
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
    # Concatenate
    for r in range(12):
        rd = regime_data[r]
        rd["X_train"] = np.vstack(rd["X_train"]) if rd["X_train"] else np.empty((0, n_features), dtype=float)
        rd["y_train"] = np.concatenate(rd["y_train"]) if rd["y_train"] else np.empty(0, dtype=np.int32)
        rd["X_val"] = np.vstack(rd["X_val"]) if rd["X_val"] else np.empty((0, n_features), dtype=float)
        rd["y_val"] = np.concatenate(rd["y_val"]) if rd["y_val"] else np.empty(0, dtype=np.int32)
        rd["X_test"] = np.vstack(rd["X_test"]) if rd["X_test"] else np.empty((0, n_features), dtype=float)
        rd["y_test"] = np.concatenate(rd["y_test"]) if rd["y_test"] else np.empty(0, dtype=np.int32)
    return regime_data


def _train_xgb(X_tr, y_tr, X_val, y_val, scale_pos):
    params = {**XGB_PARAMS, "scale_pos_weight": scale_pos}
    m = xgb.XGBClassifier(**params)
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return m


def _train_lgb(X_tr, y_tr, X_val, y_val, pos_rate):
    # scale_pos_weight only affects training in LightGBM (loss re-weighting);
    # inference probabilities still reflect the natural prior. We keep it so
    # the model focuses on positive samples during fitting. No early stopping —
    # use all 200 trees for a more expressive model.
    params = {**LGB_PARAMS, "is_unbalance": False}
    if 0 < pos_rate < 1:
        params["scale_pos_weight"] = (1 - pos_rate) / pos_rate
    m = lgb.LGBMClassifier(**params)
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
    return m


def _train_cat(X_tr, y_tr, X_val, y_val):
    params = {**CAT_PARAMS, "auto_class_weights": "Balanced"}
    m = cb.CatBoostClassifier(**params)
    m.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True, verbose=False)
    return m


def _proba_xgb(m, X):
    return m.predict_proba(X)[:, 1]


def _proba_lgb(m, X):
    return m.predict_proba(X)[:, 1]


def _proba_cat(m, X):
    return m.predict_proba(X)[:, 1]


def _eval_metrics(y_true, probs):
    if len(y_true) == 0:
        return {"n": 0}
    preds = (probs > 0.5).astype(int)
    m = {
        "n": int(len(y_true)),
        "pos_rate": float(np.mean(y_true)),
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
    }
    for thr in (0.6, 0.7, 0.8):
        mask = probs > thr
        if mask.sum() > 0:
            m[f"prec_at_{thr}"] = float(np.mean(y_true[mask] == 1))
            m[f"n_at_{thr}"] = int(mask.sum())
        else:
            m[f"prec_at_{thr}"] = 0.0
            m[f"n_at_{thr}"] = 0
    return m


def train_regime_ensemble(regime_data: dict, n_features: int) -> dict:
    log(f"\n{'='*60}")
    log(f"PHASE 5: Training 36 models (3 per regime × 12 regimes) + stacking")
    log(f"{'='*60}")

    models = {}
    trained_count = 0
    skipped_count = 0
    total_t0 = time.time()

    for r in range(12):
        rname = REGIME_NAMES[r]
        rd = regime_data[r]
        n_train = len(rd["X_train"])
        n_val = len(rd["X_val"])
        n_test = len(rd["X_test"])

        log(f"\n  [{r+1:2}/12] {rname}: train={n_train} val={n_val} test={n_test}")

        if n_train < MIN_TRAIN_SAMPLES:
            log(f"    SKIP — too few train samples (<{MIN_TRAIN_SAMPLES})")
            models[r] = {"status": "skipped", "reason": "too_few_train", "n_train": n_train}
            skipped_count += 1
            continue

        pos_rate = float(rd["y_train"].mean())
        log(f"    pos_rate={pos_rate:.3f}")

        if pos_rate < 0.02 or pos_rate > 0.98:
            log(f"    SKIP — extreme class imbalance")
            models[r] = {"status": "skipped", "reason": "class_imbalance", "pos_rate": pos_rate}
            skipped_count += 1
            continue

        if n_val < MIN_VAL_SAMPLES:
            log(f"    SKIP — too few val samples (<{MIN_VAL_SAMPLES})")
            models[r] = {"status": "skipped", "reason": "too_few_val", "n_val": n_val}
            skipped_count += 1
            continue

        scale_pos = (1 - pos_rate) / max(pos_rate, 1e-6)

        # ---- Train XGBoost ----
        t0 = time.time()
        try:
            xgb_model = _train_xgb(rd["X_train"], rd["y_train"], rd["X_val"], rd["y_val"], scale_pos)
        except Exception as e:
            log(f"    XGB FAIL: {e}")
            models[r] = {"status": "failed", "reason": f"xgb: {e}"}
            skipped_count += 1
            continue
        xgb_t = time.time() - t0

        # ---- Train LightGBM ----
        t0 = time.time()
        try:
            lgb_model = _train_lgb(rd["X_train"], rd["y_train"], rd["X_val"], rd["y_val"], pos_rate)
        except Exception as e:
            log(f"    LGB FAIL: {e}")
            models[r] = {"status": "failed", "reason": f"lgb: {e}"}
            skipped_count += 1
            continue
        lgb_t = time.time() - t0

        # ---- Train CatBoost ----
        t0 = time.time()
        try:
            cat_model = _train_cat(rd["X_train"], rd["y_train"], rd["X_val"], rd["y_val"])
        except Exception as e:
            log(f"    CAT FAIL: {e}")
            models[r] = {"status": "failed", "reason": f"cat: {e}"}
            skipped_count += 1
            continue
        cat_t = time.time() - t0

        # ---- Stacking: LogisticRegression on VAL predictions (out-of-train) ----
        # Train base models on TRAIN. Generate predictions on VAL (out-of-sample
        # for base learners) and TEST. Fit LogisticRegression on VAL predictions,
        # then evaluate on TEST.
        p_xgb_val = _proba_xgb(xgb_model, rd["X_val"])
        p_lgb_val = _proba_lgb(lgb_model, rd["X_val"])
        p_cat_val = _proba_cat(cat_model, rd["X_val"])
        S_val = np.column_stack([p_xgb_val, p_lgb_val, p_cat_val])

        p_xgb_test = _proba_xgb(xgb_model, rd["X_test"]) if n_test > 0 else np.array([])
        p_lgb_test = _proba_lgb(lgb_model, rd["X_test"]) if n_test > 0 else np.array([])
        p_cat_test = _proba_cat(cat_model, rd["X_test"]) if n_test > 0 else np.array([])

        try:
            stacking = LogisticRegression(**STACK_PARAMS)
            stacking.fit(S_val, rd["y_val"])
        except Exception as e:
            log(f"    STACK FAIL: {e}")
            models[r] = {"status": "failed", "reason": f"stack: {e}"}
            skipped_count += 1
            continue

        # ---- Final predictions (stacking output) ----
        if n_test > 0:
            S_test = np.column_stack([p_xgb_test, p_lgb_test, p_cat_test])
            stacking_probs = stacking.predict_proba(S_test)[:, 1]
        else:
            stacking_probs = np.array([])

        # ---- Compute metrics per base model + stacking ----
        base_metrics = {}
        for name, probs_fn, model in [
            ("xgb", _proba_xgb, xgb_model),
            ("lgb", _proba_lgb, lgb_model),
            ("cat", _proba_cat, cat_model),
        ]:
            split_metrics = {}
            for sname, Xs, ys in [
                ("train", rd["X_train"], rd["y_train"]),
                ("val", rd["X_val"], rd["y_val"]),
                ("test", rd["X_test"], rd["y_test"]),
            ]:
                if len(Xs) == 0:
                    split_metrics[sname] = {"n": 0}
                    continue
                split_metrics[sname] = _eval_metrics(ys, probs_fn(model, Xs))
            base_metrics[name] = split_metrics

        # Stacking metrics on TEST
        stacking_metrics = {
            "train": {"n": 0},
            "val": {"n": 0},
            "test": {"n": 0},
        }
        # Train stacking performance (sanity)
        S_train = np.column_stack([
            _proba_xgb(xgb_model, rd["X_train"]),
            _proba_lgb(lgb_model, rd["X_train"]),
            _proba_cat(cat_model, rd["X_train"]),
        ])
        stacking_train_probs = stacking.predict_proba(S_train)[:, 1]
        stacking_metrics["train"] = _eval_metrics(rd["y_train"], stacking_train_probs)
        stacking_metrics["val"] = _eval_metrics(rd["y_val"], stacking.predict_proba(S_val)[:, 1])
        if n_test > 0:
            stacking_metrics["test"] = _eval_metrics(rd["y_test"], stacking_probs)
        else:
            stacking_metrics["test"] = {"n": 0}

        models[r] = {
            "status": "trained",
            "feat_names": rd["feat_names"],
            "models": {"xgb": xgb_model, "lgb": lgb_model, "cat": cat_model},
            "stacking": stacking,
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
            "pos_rate": pos_rate,
            "scale_pos_weight": float(scale_pos),
            "timing": {"xgb_sec": float(xgb_t), "lgb_sec": float(lgb_t), "cat_sec": float(cat_t)},
            "base_metrics": base_metrics,
            "stacking_metrics": stacking_metrics,
        }
        trained_count += 1

        # ---- Log per-regime summary ----
        log(f"    XGB  {xgb_t:5.1f}s | VAL prec={base_metrics['xgb']['val'].get('precision', 0):.3f} | TEST prec={base_metrics['xgb']['test'].get('precision', 0):.3f}")
        log(f"    LGB  {lgb_t:5.1f}s | VAL prec={base_metrics['lgb']['val'].get('precision', 0):.3f} | TEST prec={base_metrics['lgb']['test'].get('precision', 0):.3f}")
        log(f"    CAT  {cat_t:5.1f}s | VAL prec={base_metrics['cat']['val'].get('precision', 0):.3f} | TEST prec={base_metrics['cat']['test'].get('precision', 0):.3f}")
        if n_test > 0:
            ts = stacking_metrics["test"]
            log(f"    STK      | TEST prec@0.5={ts.get('precision', 0):.3f} "
                f"@0.6={ts.get('prec_at_0.6', 0):.3f} (n={ts.get('n_at_0.6', 0)}) "
                f"@0.7={ts.get('prec_at_0.7', 0):.3f} (n={ts.get('n_at_0.7', 0)}) "
                f"@0.8={ts.get('prec_at_0.8', 0):.3f} (n={ts.get('n_at_0.8', 0)})")

    elapsed = time.time() - total_t0
    log(f"\n  Trained {trained_count}/12 regimes in {elapsed/60:.1f} min "
        f"({trained_count*3} base models + {trained_count} stacking)")
    return models


# ═════════════════════════════════════════════════════════════════
# Phase 6: Export all models to JSON
# ═════════════════════════════════════════════════════════════════

def export_models(models: dict, split_info: dict, feature_names_master: list,
                  all_data: dict) -> dict:
    log(f"\n{'='*60}")
    log(f"PHASE 6: Export ALL models to JSON")
    log(f"{'='*60}")

    n_features = len(feature_names_master) if feature_names_master else 36
    stacking_export = {"version": "v7", "regimes": {}}
    metadata_regimes = {}

    exported = {"xgb": 0, "lgb": 0, "cat": 0}
    for r in range(12):
        rname = REGIME_NAMES[r]
        rname_lower = rname.lower()
        info = models[r]

        if info["status"] != "trained":
            metadata_regimes[rname] = {"status": info["status"], "reason": info.get("reason", "")}
            stacking_export["regimes"][rname] = {"status": "skipped"}
            continue

        xgb_model = info["models"]["xgb"]
        lgb_model = info["models"]["lgb"]
        cat_model = info["models"]["cat"]
        stacking = info["stacking"]
        feat_names = info["feat_names"]

        # ---- XGBoost JSON (booster.save_raw(raw_format="json")) ----
        booster = xgb_model.get_booster()
        xgb_raw = booster.save_raw(raw_format="json")
        xgb_config = json.loads(xgb_raw.decode("utf-8"))
        # Attach feature_names for TS-side matching (booster JSON doesn't always include them)
        xgb_config["_feature_names"] = list(feat_names)
        xgb_config["_version"] = "v7"
        xgb_config["_regime"] = rname
        xgb_path = OUTPUT_DIR / f"regime_v7_{rname_lower}_xgb.json"
        with open(xgb_path, "w") as f:
            json.dump(xgb_config, f)
        exported["xgb"] += 1

        # ---- LightGBM JSON (Booster.dump_model()) ----
        lgb_booster = lgb_model.booster_
        lgb_dump = lgb_booster.dump_model()
        # Attach feature_names (LightGBM dump uses feature_infos, but TS needs names)
        lgb_dump["_feature_names"] = list(feat_names)
        lgb_dump["_version"] = "v7"
        lgb_dump["_regime"] = rname
        lgb_path = OUTPUT_DIR / f"regime_v7_{rname_lower}_lgb.json"
        with open(lgb_path, "w") as f:
            json.dump(lgb_dump, f)
        exported["lgb"] += 1

        # ---- CatBoost JSON (save_model(format="json")) ----
        cat_tmp_path = OUTPUT_DIR / f".tmp_cat_{rname_lower}.json"
        cat_model.save_model(str(cat_tmp_path), format="json")
        with open(cat_tmp_path, "r") as f:
            cat_config = json.load(f)
        cat_config["_feature_names"] = list(feat_names)
        cat_config["_version"] = "v7"
        cat_config["_regime"] = rname
        cat_path = OUTPUT_DIR / f"regime_v7_{rname_lower}_cat.json"
        with open(cat_path, "w") as f:
            json.dump(cat_config, f)
        try:
            os.remove(cat_tmp_path)
        except Exception:
            pass
        exported["cat"] += 1

        # ---- Stacking coefficients (per regime) ----
        stacking_export["regimes"][rname] = {
            "status": "trained",
            "coef": stacking.coef_.tolist(),
            "intercept": float(stacking.intercept_[0]),
            "classes": stacking.classes_.tolist(),
            "n_train": int(info["n_train"]),
            "n_val": int(info["n_val"]),
            "n_test": int(info["n_test"]),
            "feature_names": ["xgb_prob", "lgb_prob", "cat_prob"],
        }

        metadata_regimes[rname] = {
            "status": "trained",
            "n_train": int(info["n_train"]),
            "n_val": int(info["n_val"]),
            "n_test": int(info["n_test"]),
            "pos_rate": float(info["pos_rate"]),
            "scale_pos_weight": float(info["scale_pos_weight"]),
            "files": {
                "xgb": xgb_path.name,
                "lgb": lgb_path.name,
                "cat": cat_path.name,
            },
            "feature_names": list(feat_names),
            "base_metrics": to_native(info["base_metrics"]),
            "stacking_metrics": to_native(info["stacking_metrics"]),
            "timing_sec": to_native(info["timing"]),
        }

        log(f"  {rname:24}: xgb={xgb_path.stat().st_size/1024:.0f}KB "
            f"lgb={lgb_path.stat().st_size/1024:.0f}KB "
            f"cat={cat_path.stat().st_size/1024:.0f}KB")

    # ---- Stacking JSON (per-regime coefficients) ----
    stacking_path = OUTPUT_DIR / "regime_v7_stacking.json"
    with open(stacking_path, "w") as f:
        json.dump(to_native(stacking_export), f, indent=2)
    log(f"  Stacking: {stacking_path.name} ({stacking_path.stat().st_size/1024:.1f}KB)")

    # ---- Metadata ----
    metadata = {
        "version": "v7",
        "trained_at": datetime.now(timezone(timedelta(hours=3))).isoformat(),
        "n_features": n_features,
        "feature_names_master": list(feature_names_master) if feature_names_master else [],
        "label_threshold": LABEL_THRESHOLD,
        "horizon_bars": HORIZON_BARS,
        "commission_per_side": COMMISSION_PER_SIDE,
        "roundtrip_commission": ROUNDTRIP_COMMISSION,
        "split": {
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
            "test_ratio": TEST_RATIO,
            "global_min_ms": split_info["global_min_ms"],
            "train_end_ms": split_info["train_end_ms"],
            "val_end_ms": split_info["val_end_ms"],
            "global_max_ms": split_info["global_max_ms"],
        },
        "tickers": list(all_data.keys()),
        "n_tickers": len(all_data),
        "n_bars_per_ticker": {t: int(d["n_bars"]) for t, d in all_data.items()},
        "params": {
            "xgb": {k: v for k, v in XGB_PARAMS.items() if k != "early_stopping_rounds"},
            "lgb": LGB_PARAMS,
            "cat": CAT_PARAMS,
            "stacking": STACK_PARAMS,
        },
        "regimes": metadata_regimes,
        "exported_counts": exported,
        "files": {
            "xgb": exported["xgb"],
            "lgb": exported["lgb"],
            "cat": exported["cat"],
            "stacking": 1,
            "metadata": 1,
            "total": exported["xgb"] + exported["lgb"] + exported["cat"] + 2,
        },
    }
    meta_path = OUTPUT_DIR / "regime_v7_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(to_native(metadata), f, indent=2)
    log(f"  Metadata: {meta_path.name}")

    log(f"\n  Total JSON files: {metadata['files']['total']}")
    log(f"    xgb={exported['xgb']}  lgb={exported['lgb']}  cat={exported['cat']}  "
        f"stacking=1  metadata=1")

    return metadata


# ═════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--tickers", type=str, default="all")
    args = parser.parse_args()

    tickers = TICKERS if args.tickers == "all" else args.tickers.split(",")

    log(f"╔{'═'*60}╗")
    log(f"║  ML v7 TRAINING — ENSEMBLE (3 models × 12 regimes)        ║")
    log(f"╠{'═'*60}╣")
    log(f"║  Tickers: {len(tickers):<46} ║")
    log(f"║  Days: {args.days:<48} ║")
    log(f"║  Features: 36 (features_v7.py)                              ║")
    log(f"║  Models: 12 regimes × 3 base = 36 + 12 stacking             ║")
    log(f"║  Split: 60/20/20 (date-purged global)                      ║")
    log(f"║  Label: forward_return > {LABEL_THRESHOLD} (comm-aware)               ║")
    log(f"║  Stacking: LogisticRegression on (xgb, lgb, cat)            ║")
    log(f"╚{'═'*60}╝")

    start_time = time.time()

    # ---------- Phase 1+2+3: Data + Features + Labels ----------
    log(f"\n{'='*60}")
    log(f"PHASE 1+2+3: Loading data + computing v7 features + labels")
    log(f"{'='*60}")

    all_tickers_aligned = {}
    for i, ticker in enumerate(tickers):
        try:
            data = download_multi_timeframe(ticker, days=args.days)
            if "5min_close" not in data:
                log(f"  [{i+1}/{len(tickers)}] {ticker}: NO 5min data")
                continue
            aligned = align_timeframes(data)
            all_tickers_aligned[ticker] = aligned
        except Exception as e:
            log(f"  [{i+1}/{len(tickers)}] {ticker}: ERROR {e}")

    if not all_tickers_aligned:
        log("FATAL: No data loaded")
        return 1

    # Fetch macro data ONCE for all tickers (cached)
    try:
        macro_data = fetch_macro_data(days=max(args.days, 60))
        # NB: don't use `arr or []` — ndarray truthiness is ambiguous
        def _safe_len(x):
            if x is None:
                return 0
            try:
                return len(x)
            except Exception:
                return 0
        log(f"  Macro data fetched: "
            f"USD/RUB={_safe_len(macro_data.get('usdrub_close'))} "
            f"Brent={_safe_len(macro_data.get('brent_close'))} "
            f"IMOEX={_safe_len(macro_data.get('imoex_close'))} "
            f"IMOEX_1h={_safe_len(macro_data.get('imoex_1h_close'))}")
    except Exception as e:
        log(f"  Macro fetch failed (using zeros): {e}")
        macro_data = {}

    all_data = {}
    feature_names_master = None
    for ticker, aligned in all_tickers_aligned.items():
        data = compute_features_and_labels(ticker, aligned, all_tickers_aligned, macro_data)
        if data is None:
            continue
        all_data[ticker] = data
        if feature_names_master is None and data["feat_names"]:
            feature_names_master = data["feat_names"]
        log(f"  {ticker}: X={data['X'].shape} pos_rate={data['y'].mean():.3f} "
            f"regimes={np.unique(data['regime']).tolist()}")

    if not all_data:
        log("FATAL: No data after features/labels")
        return 1

    # ---------- Phase 4: Date-purged splits ----------
    split_info = date_purged_splits(all_data)

    # ---------- Phase 5: Train ensemble per regime ----------
    if feature_names_master is None:
        feature_names_master = []
    n_features = len(feature_names_master)
    regime_data = _aggregate_regime_data(all_data, split_info["splits"], n_features)
    models = train_regime_ensemble(regime_data, n_features)

    # ---------- Phase 6: Export ----------
    metadata = export_models(models, split_info, feature_names_master, all_data)

    # ---------- Summary ----------
    elapsed = time.time() - start_time
    trained = sum(1 for m in models.values() if m["status"] == "trained")
    skipped = sum(1 for m in models.values() if m["status"] == "skipped")
    failed = sum(1 for m in models.values() if m["status"] == "failed")

    log(f"\n╔{'═'*60}╗")
    log(f"║  V7 TRAINING COMPLETE in {elapsed/60:.1f} min                      ║")
    log(f"╠{'═'*60}╣")
    log(f"║  Regimes trained: {trained}/12 (skipped={skipped} failed={failed})               ║")
    log(f"║  Base models:    {trained*3}/36                                      ║")
    log(f"║  Stacking models: {trained}/12                                       ║")
    log(f"║  JSON files:      {metadata['files']['total']}                                       ║")
    log(f"║  Output: {str(OUTPUT_DIR):<46} ║")
    log(f"╚{'═'*60}╝")

    # Print per-regime precision summary table
    log(f"\nPer-regime TEST precision @ thresholds (stacking output):")
    log(f"  {'Regime':<22} {'N':>6} {'P@0.5':>6} {'P@0.6':>6} {'N@0.6':>5} {'P@0.7':>6} {'N@0.7':>5} {'P@0.8':>6} {'N@0.8':>5}")
    for r in range(12):
        rname = REGIME_NAMES[r]
        info = models[r]
        if info["status"] != "trained":
            log(f"  {rname:<22} {info['status'].upper()}")
            continue
        ts = info["stacking_metrics"]["test"]
        log(f"  {rname:<22} {ts.get('n',0):>6} "
            f"{ts.get('precision',0):>6.3f} "
            f"{ts.get('prec_at_0.6',0):>6.3f} {ts.get('n_at_0.6',0):>5} "
            f"{ts.get('prec_at_0.7',0):>6.3f} {ts.get('n_at_0.7',0):>5} "
            f"{ts.get('prec_at_0.8',0):>6.3f} {ts.get('n_at_0.8',0):>5}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
