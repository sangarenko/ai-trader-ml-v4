#!/usr/bin/env python3
"""Meta-Trainer: обучает XGBoost multi-class классификатор.

Вход:
  - meta_labels.npz (4466 сэмплов: bar_idx, regime, best_strategy, pnls_matrix)
  - multi_timeframe данные (для подсчёта market features на каждом баре)

Выход:
  - ml/meta_models/meta_classifier.json — экспорт XGBoost деревьев для TypeScript
  - ml/meta_models/meta_metadata.json — feature names, strategy names, accuracy
  - ml/meta_models/meta_classifier.pkl — pickle для Python-проверки

Идея:
  Для каждого бара из meta_labels вычисляем market features (RSI, SMA, ATR,
  ADX, returns, regime, seasonal), и обучаем классификатор предсказывать
  лучшую стратегию (one of 22) по этим фичам.

  Фичи (34 шт.):
    - Базовые 31 из ml_features.py
    - + regime (0/1/2 — TREND_UP/DOWN/RANGE)
    - + adx_value (raw)
    - + sma50_minus_sma14 (наклон тренда)

  Метрика: top-3 accuracy (если предсказанная стратегия в топ-3 — засчитывается).
  Это реалистично потому что среди 22 стратегий много похожих, и попасть в топ-3
  почти так же хорошо как в топ-1.
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
from ml_features import compute_features  # returns (X, feature_names)
from meta_labeler import compute_regime, STRATEGY_NAMES, STRAT_TO_IDX

OUTPUT_DIR = Path("/root/ai-trader-evolution/ml/meta_models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = "/var/log/ai-trader-meta-train.log"
LABELS_PATH = Path("/root/ai-trader-evolution/ml/data_cache/meta_labels.npz")


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
    """Load labels from meta_labeler output."""
    log(f"Loading labels → {LABELS_PATH}")
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


def build_dataset(labels: dict, days: int = 180) -> tuple:
    """Build (X, y, ticker_ids) dataset by computing features at each labeled bar.

    Returns:
        X: (n_samples, n_features) float32
        y: (n_samples,) int — best strategy index
        ticker_ids: (n_samples,) int — for stratified split
        feature_names: list[str]
    """
    tickers = labels["tickers"]
    bar_indices = labels["bar_indices"]
    # Track which sample belongs to which ticker (since we concatenated)
    sample_ticker_idx = []
    sample_bar_idx_in_ticker = []  # bar index WITHIN its ticker
    cursor = 0
    for ti, t in enumerate(tickers):
        # We don't know how many samples per ticker upfront from labels,
        # but we can count from pnls_matrix shape per ticker via split
        pass

    # Recompute features per ticker, then pick bars at the labeled indices
    # bar_indices in the .npz are GLOBAL (concatenated). We need to track per-ticker.
    # Easier: redo the per-ticker loop and collect (ticker_i, bar_i, sample_i)

    log(f"Building dataset from {len(tickers)} tickers × {days} days...")

    all_X = []
    all_y = []
    all_ticker_ids = []
    all_global_idx = []  # index into labels arrays

    # Recompute sample points per ticker to know which global indices belong where
    # We rely on the fact that meta_labeler processed tickers in same order
    # AND used same window/step. So bar_indices[start:end] = ticker[t] samples.

    sample_idx_per_ticker = {}
    cursor = 0
    for ti, t in enumerate(tickers):
        # Try to load this ticker's data to count samples
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

    # Now compute features for each labeled bar
    for ti, (start_i, end_i, aligned) in sample_idx_per_ticker.items():
        ticker_name = tickers[ti]
        n = len(aligned["5min_close"])

        # Compute features once for full ticker
        X_full, feat_names = compute_features(aligned)
        # Add regime + ADX + trend_slope as features
        close5 = aligned["5min_close"]
        ind_extra = {}
        # Compute regime using meta_labeler's compute_regime (same code)
        # We need ind dict though — recompute using precompute_indicators
        from fast_backtest_v2 import precompute_indicators
        ind = precompute_indicators(aligned["5min_open"], close5,
                                      aligned["5min_high"], aligned["5min_low"],
                                      aligned["5min_volume"])
        regime = compute_regime(close5, ind)
        adx = ind.get("adx", np.zeros(n))
        sma14 = ind.get("sma14", np.zeros(n))
        sma50 = ind.get("sma50", np.zeros(n))

        # Append extra features (avoid duplicates with existing feature names)
        existing_names = list(feat_names)
        extra_names_to_add = []
        extra_cols_to_add = []
        if "regime" not in existing_names:
            extra_names_to_add.append("regime")
            extra_cols_to_add.append(regime.astype(float))
        if "trend_slope" not in existing_names:
            extra_names_to_add.append("trend_slope")
            extra_cols_to_add.append((sma50 - sma14) / (sma14 + 1e-9))
        # 'adx' is already in feat_names from ml_features
        if extra_cols_to_add:
            X_full = np.column_stack([X_full] + extra_cols_to_add)
            feat_names = existing_names + extra_names_to_add

        # Pick bars at sample points
        for gi in range(start_i, end_i):
            global_bar_idx = labels["bar_indices"][gi]
            # This bar index was a sample point for THIS ticker
            if 0 <= global_bar_idx < n:
                all_X.append(X_full[global_bar_idx])
                all_y.append(int(labels["best_strategies"][gi]))
                all_ticker_ids.append(ti)
                all_global_idx.append(gi)

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.int32)
    ticker_ids = np.array(all_ticker_ids, dtype=np.int32)

    feature_names = list(feat_names)  # 'regime' and 'trend_slope' appended inside build_dataset if not present
    return X, y, ticker_ids, feature_names


def train_classifier(X, y, ticker_ids, feature_names):
    """Train XGBoost multi-class. Chronological split by ticker to avoid leakage.

    Returns: (model, metrics)
    """
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, top_k_accuracy_score, classification_report
    from sklearn.preprocessing import LabelEncoder

    n_classes = len(STRATEGY_NAMES)
    log(f"\n=== DATASET ===")
    log(f"X: {X.shape}, y: {y.shape}, max classes: {n_classes}")
    log(f"Feature names ({len(feature_names)}): {feature_names[:10]}...")

    # Clean NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    # Split: 70% train / 15% val / 15% test (chronological by sample order)
    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train = X[:train_end]
    X_val = X[train_end:val_end]
    X_test = X[val_end:]
    y_train_orig = y[:train_end]
    y_val_orig = y[train_end:val_end]
    y_test_orig = y[val_end:]

    # XGBoost requires classes 0..n-1 without gaps. Remap based on TRAIN classes.
    train_classes = sorted(np.unique(y_train_orig).tolist())
    log(f"Train classes: {len(train_classes)}/{n_classes}")
    missing_in_train = [STRATEGY_NAMES[i] for i in range(n_classes) if i not in train_classes]
    if missing_in_train:
        log(f"Missing in train (no samples): {missing_in_train}")

    # Build mapping: original_idx -> 0..k-1 (only for classes present in train)
    orig_to_enc = {int(orig): int(enc) for enc, orig in enumerate(train_classes)}
    enc_to_orig = {int(enc): int(orig) for orig, enc in orig_to_enc.items()}
    effective_n_classes = len(train_classes)

    # Map val/test — any class not in train gets clipped to nearest valid enc
    def remap(arr):
        out = np.zeros(len(arr), dtype=np.int32)
        for i, v in enumerate(arr):
            out[i] = orig_to_enc.get(int(v), 0)  # default 0 if unseen
        return out

    y_train = remap(y_train_orig)
    y_val = remap(y_val_orig)
    y_test = remap(y_test_orig)

    log(f"Effective classes (from train): {effective_n_classes}")

    log(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Class distribution check
    log(f"\nClass distribution (train):")
    u, c = np.unique(y_train, return_counts=True)
    for i, ci in zip(u, c):
        orig_i = enc_to_orig[int(i)]
        if ci > 10:
            log(f"  {STRATEGY_NAMES[orig_i]:20}: {ci:4d} ({ci*100/len(y_train):.1f}%)")

    # Compute class weights to handle imbalance
    class_counts = np.bincount(y_train, minlength=effective_n_classes)
    class_weights = len(y_train) / (effective_n_classes * np.maximum(class_counts, 1))
    sample_weights = class_weights[y_train]

    log(f"\n=== TRAINING XGBoost (multi-class, {effective_n_classes} classes, regularized) ===")
    t0 = time.time()
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=3,             # shallow trees to fight overfitting
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.6,
        min_child_weight=20,     # large min child weight
        gamma=1.0,               # high gamma
        reg_alpha=1.0,
        reg_lambda=10.0,         # strong L2
        objective="multi:softprob",
        num_class=effective_n_classes,
        random_state=42,
        n_jobs=2,
        eval_metric="mlogloss",
        early_stopping_rounds=30,
        tree_method="hist",
    )

    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=20,
    )
    elapsed = time.time() - t0
    log(f"Trained in {elapsed:.0f}s, best iter: {model.best_iteration}")

    # Evaluate
    log(f"\n=== EVALUATION ===")
    for split, Xs, ys in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        if len(Xs) == 0: continue
        probs = model.predict_proba(Xs)
        preds = np.argmax(probs, axis=1)
        acc = accuracy_score(ys, preds)
        top3 = top_k_accuracy_score(ys, probs, k=min(3, effective_n_classes), labels=range(effective_n_classes))
        top5 = top_k_accuracy_score(ys, probs, k=min(5, effective_n_classes), labels=range(effective_n_classes))
        log(f"  {split:5}: top-1={acc:.3f}  top-3={top3:.3f}  top-5={top5:.3f}  (n={len(ys)})")

    # Per-regime accuracy on test
    log(f"\n=== PER-REGIME TEST ACCURACY ===")
    # We need regime for test samples — we passed it as feature (index -3 = regime)
    regime_feature_idx = feature_names.index("regime")
    test_regimes = X_test[:, regime_feature_idx].astype(int)
    for r, rname in [(0, "RANGE"), (1, "TREND_UP"), (2, "TREND_DOWN")]:
        mask = test_regimes == r
        if mask.sum() == 0: continue
        probs = model.predict_proba(X_test[mask])
        preds = np.argmax(probs, axis=1)
        acc = accuracy_score(y_test[mask], preds)
        top3 = top_k_accuracy_score(y_test[mask], probs, k=min(3, effective_n_classes), labels=range(effective_n_classes))
        # Most predicted strategy (decode back to original idx)
        u_p, c_p = np.unique(preds, return_counts=True)
        top_pred_orig_idx = enc_to_orig[int(u_p[np.argmax(c_p)])]
        top_pred = STRATEGY_NAMES[top_pred_orig_idx]
        log(f"  {rname:10} (n={mask.sum():3}): top-1={acc:.3f}  top-3={top3:.3f}  most_pred={top_pred}")

    # Per-regime CONFUSION: actual best vs predicted
    log(f"\n=== PER-REGIME: actual top-3 vs predicted top-3 ===")
    for r, rname in [(0, "RANGE"), (1, "TREND_UP"), (2, "TREND_DOWN")]:
        mask = test_regimes == r
        if mask.sum() == 0: continue
        y_actual_enc = y_test[mask]
        probs = model.predict_proba(X_test[mask])
        preds_enc = np.argmax(probs, axis=1)
        log(f"  {rname} (n={mask.sum()}):")
        # Actual top-3 (decode)
        u_act, c_act = np.unique(y_actual_enc, return_counts=True)
        act_top3_enc = list(reversed(u_act[np.argsort(c_act)[-3:]].tolist()))
        act_names = [f"{STRATEGY_NAMES[enc_to_orig[int(i)]]}({c_act[np.where(u_act==i)[0][0]]})" for i in act_top3_enc]
        # Predicted top-3 (decode)
        u_pred, c_pred = np.unique(preds_enc, return_counts=True)
        pred_top3_enc = list(reversed(u_pred[np.argsort(c_pred)[-3:]].tolist()))
        pred_names = [f"{STRATEGY_NAMES[enc_to_orig[int(i)]]}({c_pred[np.where(u_pred==i)[0][0]]})" for i in pred_top3_enc]
        log(f"    actual:    {act_names}")
        log(f"    predicted: {pred_names}")

    return model, {
        "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "n_classes_original": n_classes,
        "n_classes_effective": effective_n_classes,
        "class_map_original_to_encoded": orig_to_enc,
        "class_map_encoded_to_original": enc_to_orig,
        "strategy_names": STRATEGY_NAMES,
        "missing_strategies": missing_in_train,
    }


def export_to_json(model, metadata, output_dir):
    """Export XGBoost trees to JSON for TypeScript inference."""
    log(f"\n=== EXPORT TO JSON ===")
    booster = model.get_booster()
    config = json.loads(booster.save_raw(raw_format="json").decode("utf-8"))

    # Save full model JSON
    json_path = output_dir / "meta_classifier.json"
    with open(json_path, "w") as f:
        json.dump(config, f)
    log(f"Saved trees → {json_path} ({json_path.stat().st_size / 1024:.0f} KB)")

    # Save metadata
    meta_path = output_dir / "meta_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log(f"Saved metadata → {meta_path}")

    # Save pickle for Python verification
    pkl_path = output_dir / "meta_classifier.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)
    log(f"Saved pickle → {pkl_path}")

    # Print tree stats
    n_trees = len(config.get("learner", {}).get("gradient_booster", {}).get("model", {}).get("trees", []))
    expected = metadata['n_classes_effective'] * (model.best_iteration+1)
    log(f"Trees: {n_trees} (expected {expected})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=180, help="История в днях (должно совпадать с labeler)")
    args = parser.parse_args()

    log(f"═══ META-TRAINER START ═══")

    labels = load_meta_labels()
    log(f"Labels: {len(labels['bar_indices'])} samples from {len(labels['tickers'])} tickers")
    log(f"Window: {labels['window_bars']} bars, Step: {labels['step_bars']} bars")

    X, y, ticker_ids, feat_names = build_dataset(labels, days=args.days)
    log(f"Dataset built: X={X.shape}, y={y.shape}")

    model, metadata = train_classifier(X, y, ticker_ids, feat_names)
    export_to_json(model, metadata, OUTPUT_DIR)

    log(f"\n═══ META-TRAINER DONE ═══")
    log(f"Output dir: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
