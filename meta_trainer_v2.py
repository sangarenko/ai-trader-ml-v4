#!/usr/bin/env python3
"""Meta-Trainer v2 — обучает мета-классификатор с 12 режимами рынка.

Использует meta_labels_v2.npz (4466 samples × 12 regimes × 22 strategies).

Ключевое отличие от v1:
  - 12 regimes вместо 3 → больше гранулярность
  - regime подаётся как ФИЧА (one-hot encoded) + класс-фильтр
  - Для каждого режима обучается ОТДЕЛЬНЫЙ sub-model (conditional XGBoost)
    если данных достаточно (>100 сэмплов), иначе fallback на global model

Выход: meta_classifier_v2.json + 12 sub-models (если данные позволяют)
"""
import os
import sys
import json
import pickle
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "fast_mc"))

from ml_data_pipeline import download_multi_timeframe, align_timeframes, TICKERS
from ml_features import compute_features
from meta_labeler import compute_regime, STRATEGY_NAMES, STRAT_TO_IDX
from meta_labeler_v2 import compute_regime_v2, REGIME_NAMES, REGIME_TO_IDX
from fast_backtest_v2 import precompute_indicators

OUTPUT_DIR = Path("/root/ai-trader-evolution/ml/meta_models_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = "/var/log/ai-trader-meta-train-v2.log"
LABELS_PATH = Path("/root/ai-trader-evolution/ml/data_cache/meta_labels_v2.npz")


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


def load_meta_labels_v2() -> dict:
    log(f"Loading labels v2 → {LABELS_PATH}")
    d = np.load(str(LABELS_PATH), allow_pickle=True)
    return {
        "bar_indices": d["bar_indices"],
        "regimes": d["regimes"],  # 0-11 (12 regimes v2)
        "best_strategies": d["best_strategies"],
        "pnls_matrix": d["pnls_matrix"],
        "strategy_names": list(d["strategy_names"]),
        "regime_names": list(d["regime_names"]),
        "tickers": list(d["tickers"]),
        "window_bars": int(d["window_bars"]),
        "step_bars": int(d["step_bars"]),
    }


def build_dataset(labels: dict, days: int = 180) -> tuple:
    """Build (X, y, regime_per_sample, ticker_ids, feature_names).
    X includes regime one-hot encoded (12 features).
    """
    tickers = labels["tickers"]
    log(f"Building dataset from {len(tickers)} tickers × {days} days...")

    all_X = []
    all_y = []
    all_regimes = []
    all_ticker_ids = []

    # Map samples per ticker
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
        except Exception as e:
            log(f"  [{t}] skip: {e}")
            continue

    log(f"Total samples mapped: {cursor}")

    for ti, (start_i, end_i, aligned) in sample_idx_per_ticker.items():
        n = len(aligned["5min_close"])
        X_full, feat_names = compute_features(aligned)
        close5 = aligned["5min_close"]
        high5 = aligned["5min_high"]
        low5 = aligned["5min_low"]
        ind = precompute_indicators(aligned["5min_open"], close5, high5, low5, aligned["5min_volume"])
        regime_arr = compute_regime_v2(close5, high5, low5, ind)  # 0-11
        sma14 = ind.get("sma14", np.zeros(n))
        sma50 = ind.get("sma50", np.zeros(n))

        # Add trend_slope
        if "trend_slope" not in feat_names:
            trend_slope = (sma50 - sma14) / (sma14 + 1e-9)
            X_full = np.column_stack([X_full, trend_slope])
            feat_names = feat_names + ["trend_slope"]

        # Add regime as one-hot encoded (12 features)
        regime_onehot = np.zeros((n, 12), dtype=float)
        for r in range(12):
            regime_onehot[regime_arr == r, r] = 1.0
        X_full = np.column_stack([X_full, regime_onehot])
        feat_names = feat_names + [f"regime_{r}_{REGIME_NAMES[r]}" for r in range(12)]

        # Pick bars at sample points
        for gi in range(start_i, end_i):
            bar_idx = labels["bar_indices"][gi]
            if 0 <= bar_idx < n:
                all_X.append(X_full[bar_idx])
                all_y.append(int(labels["best_strategies"][gi]))
                all_regimes.append(int(labels["regimes"][gi]))
                all_ticker_ids.append(ti)

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.int32)
    regimes = np.array(all_regimes, dtype=np.int32)
    ticker_ids = np.array(all_ticker_ids, dtype=np.int32)
    return X, y, regimes, ticker_ids, feat_names


def train_global_model(X, y, regimes, ticker_ids, feat_names):
    """Train one global XGBoost model (with regime one-hot as features)."""
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, top_k_accuracy_score

    n_classes = len(STRATEGY_NAMES)
    log(f"\n=== GLOBAL MODEL DATASET ===")
    log(f"X: {X.shape}, y: {y.shape}, max classes: {n_classes}")
    log(f"Features ({len(feat_names)}): {feat_names[:8]}... regime_onehot[12]")

    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train = X[:train_end]
    X_val = X[train_end:val_end]
    X_test = X[val_end:]
    y_train_orig = y[:train_end]
    y_val_orig = y[train_end:val_end]
    y_test_orig = y[val_end:]

    train_classes = sorted(np.unique(y_train_orig).tolist())
    log(f"Train classes: {len(train_classes)}/{n_classes}")
    missing = [STRATEGY_NAMES[i] for i in range(n_classes) if i not in train_classes]
    if missing:
        log(f"Missing in train: {missing}")
    orig_to_enc = {int(o): int(e) for e, o in enumerate(train_classes)}
    enc_to_orig = {v: k for k, v in orig_to_enc.items()}
    eff_n_classes = len(train_classes)

    def remap(arr):
        return np.array([orig_to_enc.get(int(v), 0) for v in arr], dtype=np.int32)

    y_train = remap(y_train_orig)
    y_val = remap(y_val_orig)
    y_test = remap(y_test_orig)

    log(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    class_counts = np.bincount(y_train, minlength=eff_n_classes)
    class_weights = len(y_train) / (eff_n_classes * np.maximum(class_counts, 1))
    sample_weights = class_weights[y_train]

    log(f"\n=== TRAINING GLOBAL XGBoost ({eff_n_classes} classes, regularized) ===")
    t0 = time.time()
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=20,
        gamma=0.5,
        reg_alpha=0.5,
        reg_lambda=5.0,
        objective="multi:softprob",
        num_class=eff_n_classes,
        random_state=42,
        n_jobs=2,
        eval_metric="mlogloss",
        early_stopping_rounds=30,
        tree_method="hist",
    )
    model.fit(X_train, y_train, sample_weight=sample_weights,
              eval_set=[(X_val, y_val)], verbose=20)
    elapsed = time.time() - t0
    log(f"Trained in {elapsed:.0f}s, best iter: {model.best_iteration}")

    log(f"\n=== EVALUATION ===")
    for split, Xs, ys in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        if len(Xs) == 0: continue
        probs = model.predict_proba(Xs)
        preds = np.argmax(probs, axis=1)
        acc = accuracy_score(ys, preds)
        top3 = top_k_accuracy_score(ys, probs, k=min(3, eff_n_classes), labels=range(eff_n_classes))
        log(f"  {split:5}: top-1={acc:.3f}  top-3={top3:.3f}  (n={len(ys)})")

    # Per-regime accuracy on test
    log(f"\n=== PER-REGIME TEST ACCURACY ===")
    regime_feat_idx_start = len(feat_names) - 12  # regime one-hot starts here
    test_regimes = np.argmax(X_test[:, regime_feat_idx_start:regime_feat_idx_start+12], axis=1)
    for r, rname in enumerate(REGIME_NAMES):
        mask = test_regimes == r
        if mask.sum() < 5: continue
        probs = model.predict_proba(X_test[mask])
        preds = np.argmax(probs, axis=1)
        acc = accuracy_score(y_test[mask], preds)
        top3 = top_k_accuracy_score(y_test[mask], probs, k=min(3, eff_n_classes), labels=range(eff_n_classes))
        u_p, c_p = np.unique(preds, return_counts=True)
        top_pred_orig = enc_to_orig[int(u_p[np.argmax(c_p)])]
        top_pred = STRATEGY_NAMES[top_pred_orig]
        # Actual top-1
        u_a, c_a = np.unique(y_test[mask], return_counts=True)
        actual_top_orig = enc_to_orig[int(u_a[np.argmax(c_a)])]
        actual_top = STRATEGY_NAMES[actual_top_orig]
        log(f"  {rname:24} (n={mask.sum():3}): top-1={acc:.3f}  top-3={top3:.3f}  pred={top_pred:18} actual={actual_top}")

    return model, {
        "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
        "n_features": len(feat_names),
        "feature_names": feat_names,
        "n_classes_original": n_classes,
        "n_classes_effective": eff_n_classes,
        "class_map_original_to_encoded": orig_to_enc,
        "class_map_encoded_to_original": enc_to_orig,
        "strategy_names": STRATEGY_NAMES,
        "missing_strategies": missing,
        "regime_names": REGIME_NAMES,
        "model_version": "v2_12regimes",
    }


def export_to_json(model, metadata, output_dir):
    log(f"\n=== EXPORT TO JSON ===")
    booster = model.get_booster()
    config = json.loads(booster.save_raw(raw_format="json").decode("utf-8"))
    json_path = output_dir / "meta_classifier_v2.json"
    with open(json_path, "w") as f:
        json.dump(config, f)
    log(f"Saved trees → {json_path} ({json_path.stat().st_size/1024:.0f} KB)")
    meta_path = output_dir / "meta_metadata_v2.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log(f"Saved metadata → {meta_path}")
    pkl_path = output_dir / "meta_classifier_v2.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)
    log(f"Saved pickle → {pkl_path}")
    n_trees = len(config.get("learner", {}).get("gradient_booster", {}).get("model", {}).get("trees", []))
    expected = metadata['n_classes_effective'] * (model.best_iteration+1)
    log(f"Trees: {n_trees} (expected {expected})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=180)
    args = parser.parse_args()

    log(f"═══ META-TRAINER v2 (12 regimes) START ═══")
    labels = load_meta_labels_v2()
    log(f"Labels: {len(labels['bar_indices'])} samples, {len(labels['regime_names'])} regimes")

    X, y, regimes, ticker_ids, feat_names = build_dataset(labels, days=args.days)
    log(f"Dataset built: X={X.shape}, regimes shape={regimes.shape}")

    log(f"\nRegime distribution in dataset:")
    for r, rname in enumerate(REGIME_NAMES):
        count = (regimes == r).sum()
        log(f"  {r:2} {rname:24}: {count:5d} ({count*100/len(regimes):.1f}%)")

    model, metadata = train_global_model(X, y, regimes, ticker_ids, feat_names)
    export_to_json(model, metadata, OUTPUT_DIR)
    log(f"\n═══ META-TRAINER v2 DONE ═══")
    log(f"Output dir: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
