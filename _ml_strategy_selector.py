#!/usr/bin/env python3
"""ML model — learns from Monte Carlo results what makes a strategy profitable.

Input: all_models_1m.json (1M models with params + P&L + profitable label)
Output: trained XGBoost model that predicts P(profitable) for new params

Then: generate 100K new params, predict best ones, validate with backtest.

Usage:
  python3 ml_strategy_selector.py --input results/all_models_1m.json
"""
import os
import sys
import json
import numpy as np
import argparse
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_dataset(filepath: str):
    """Load Monte Carlo results as ML dataset.
    
    Returns: X (features), y (profitable=1/0), strategy_names, feature_names
    """
    with open(filepath) as f:
        data = json.load(f)
    
    models = data["models_data"]
    print(f"Loaded {len(models):,} models")
    
    # Encode strategy as integer
    strategies = sorted(set(m["strategy"] for m in models))
    strat_to_idx = {s: i for i, s in enumerate(strategies)}
    
    feature_names = [
        "strategy_encoded", "entry_sma_mult", "entry_rsi_min", "entry_rsi_max",
        "take_profit_pct", "hold_ticks", "exit_sma_mult", "position_size",
    ]
    
    X = np.array([
        [strat_to_idx[m["strategy"]], m["entry_sma_mult"], m["entry_rsi_min"],
         m["entry_rsi_max"], m["take_profit_pct"], m["hold_ticks"],
         m["exit_sma_mult"], m["position_size"]]
        for m in models
    ], dtype=float)
    
    y = np.array([m["profitable"] for m in models], dtype=int)
    
    # Also extract val_pnl and test_pnl for regression
    y_pnl = np.array([m["val_pnl"] + m["test_pnl"] for m in models], dtype=float)
    
    print(f"X shape: {X.shape}")
    print(f"Profitable: {y.sum():,} ({y.mean()*100:.1f}%)")
    print(f"Strategies: {len(strategies)}")
    
    return X, y, y_pnl, strategies, feature_names


def train_profitability_classifier(X, y, strategies, feature_names):
    """Train XGBoost to predict P(profitable)."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    
    try:
        import xgboost as xgb
        model = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=10, gamma=0.1,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=2,
            eval_metric="logloss",
        )
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, random_state=42
        )
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining: {len(X_train)} samples, Test: {len(X_test)}")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_proba)
    except:
        auc = 0
    
    print(f"\nClassifier metrics:")
    print(f"  Precision: {precision:.3f} (из предсказанных profitable, {precision*100:.1f}% реально)")
    print(f"  Recall:    {recall:.3f} (из реальных profitable, {recall*100:.1f}% найдено)")
    print(f"  F1:        {f1:.3f}")
    print(f"  AUC:       {auc:.3f}")
    
    # Probability thresholds
    print(f"\nProbability analysis:")
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        pred = (y_proba > thresh).astype(int)
        if pred.sum() > 0:
            actual = y_test[pred == 1].mean()
            print(f"  P>{thresh:.1f}: {pred.sum():5d} predictions, "
                  f"actual profitable: {actual*100:.1f}%")
    
    # Feature importance
    if hasattr(model, "feature_importances_"):
        print(f"\nFeature importance:")
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        for idx in indices:
            print(f"  {feature_names[idx]:25s}: {importances[idx]:.4f}")
    
    return model, strategies


def generate_optimal_params(model, strategies, n_generate=100000, top_n=100):
    """Use ML model to generate new optimal parameters.
    
    1. Generate 100K random param sets
    2. Predict P(profitable) for each
    3. Take top 100 with highest probability
    4. These are ML-guided parameter recommendations
    """
    print(f"\n{'='*60}")
    print(f"GENERATING OPTIMAL PARAMS via ML")
    print(f"{'='*60}")
    
    # Generate random params
    X_new = np.zeros((n_generate, 8))
    X_new[:, 0] = np.random.randint(0, len(strategies), n_generate)  # strategy
    X_new[:, 1] = np.random.uniform(0.995, 1.005, n_generate)  # entry_sma_mult
    X_new[:, 2] = np.random.randint(20, 41, n_generate)  # rsi_min
    X_new[:, 3] = np.random.randint(45, 61, n_generate)  # rsi_max
    X_new[:, 4] = np.random.uniform(0.005, 0.025, n_generate)  # take_profit
    X_new[:, 5] = np.random.randint(30, 301, n_generate)  # hold_ticks
    X_new[:, 6] = np.random.uniform(1.002, 1.005, n_generate)  # exit_sma_mult
    X_new[:, 7] = np.random.uniform(0.2, 0.4, n_generate)  # position_size
    
    # Predict
    y_proba = model.predict_proba(X_new)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_new)
    
    # Top N
    top_indices = np.argsort(y_proba)[::-1][:top_n]
    
    print(f"Generated {n_generate:,} candidates, selected top {top_n}")
    print(f"Top P(profitable): {y_proba[top_indices[0]]:.3f}")
    print(f"\nTop 20 ML-recommended parameter sets:")
    print(f"{'#':>3s} {'strategy':22s} {'sma_mult':>10s} {'rsi_min':>8s} {'rsi_max':>8s} "
          f"{'tp_pct':>8s} {'hold':>6s} {'exit':>8s} {'pos':>6s} {'P(prof)':>8s}")
    
    recommendations = []
    for rank, idx in enumerate(top_indices[:20], 1):
        strat_idx = int(X_new[idx, 0])
        strat_name = strategies[strat_idx]
        prob = y_proba[idx]
        
        params = {
            "strategy": strat_name,
            "entry_sma_mult": round(X_new[idx, 1], 5),
            "entry_rsi_min": int(X_new[idx, 2]),
            "entry_rsi_max": int(X_new[idx, 3]),
            "take_profit_pct": round(X_new[idx, 4], 5),
            "hold_ticks": int(X_new[idx, 5]),
            "exit_sma_mult": round(X_new[idx, 6], 5),
            "position_size": round(X_new[idx, 7], 3),
            "ml_probability": round(float(prob), 3),
        }
        recommendations.append(params)
        
        print(f"{rank:3d} {strat_name:22s} {params['entry_sma_mult']:10.5f} "
              f"{params['entry_rsi_min']:8d} {params['entry_rsi_max']:8d} "
              f"{params['take_profit_pct']:8.4f} {params['hold_ticks']:6d} "
              f"{params['exit_sma_mult']:8.5f} {params['position_size']:6.3f} "
              f"{prob:8.3f}")
    
    # Save recommendations
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    rec_file = os.path.join(results_dir, "ml_recommendations.json")
    with open(rec_file, "w") as f:
        json.dump(recommendations, f, indent=2)
    print(f"\nSaved {len(recommendations)} recommendations to: {rec_file}")
    
    return recommendations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to all_models JSON")
    parser.add_argument("--generate", type=int, default=100000, help="N new params to generate")
    parser.add_argument("--top", type=int, default=100, help="Top N to select")
    args = parser.parse_args()
    
    # 1. Load dataset
    print("=" * 60)
    print("ML STRATEGY SELECTOR")
    print("=" * 60)
    X, y, y_pnl, strategies, feature_names = load_dataset(args.input)
    
    # 2. Train classifier
    model, strategies = train_profitability_classifier(X, y, strategies, feature_names)
    
    # 3. Generate optimal params
    recommendations = generate_optimal_params(model, strategies, args.generate, args.top)
    
    # 4. Save model
    import pickle
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "ml_strategy_selector.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": model,
            "strategies": strategies,
            "feature_names": feature_names,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }, f)
    print(f"\nModel saved to: {model_path}")
    
    print(f"\n{'='*60}")
    print("DONE. Next steps:")
    print("1. Validate top recommendations with fast_backtest.py")
    print("2. Deploy best 10 to live trading")
    print("3. Retrain ML model weekly with new data")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
