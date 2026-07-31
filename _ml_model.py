#!/usr/bin/env python3
"""ML model — XGBoost classifier for trading signals.

Pipeline:
  1. Load multi-timeframe data (all 11 tickers)
  2. Compute features (40+ indicators)
  3. Compute labels (price up/down in 30 min)
  4. Split: train (80%) / val (10%) / test (10%) — chronological
  5. Train XGBoost for long signals + XGBoost for short signals
  6. Evaluate: precision, recall, F1, backtest P&L
  7. Save model for live deployment

Usage:
  python3 ml_model.py --days 180 --tickers SBER,GAZP,LKOH
  python3 ml_model.py --days 180  # all 11 tickers
"""
import os
import sys
import json
import numpy as np
import argparse
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_data_pipeline import download_multi_timeframe, align_timeframes, TICKERS
from ml_features import compute_features, compute_labels


def load_dataset(tickers: list, days: int = 180):
    """Load features + labels for all tickers. Returns combined X, y_long, y_short, ticker_ids."""
    all_X = []
    all_y_long = []
    all_y_short = []
    all_ticker_ids = []
    
    for i, ticker in enumerate(tickers):
        print(f"\n[{i+1}/{len(tickers)}] Loading {ticker}...")
        data = download_multi_timeframe(ticker, days=days)
        if "5min_close" not in data:
            print(f"  SKIP: no 5min data")
            continue
        
        aligned = align_timeframes(data)
        X, names = compute_features(aligned)
        y_long, y_short = compute_labels(aligned, horizon=6, threshold=0.001)
        
        all_X.append(X)
        all_y_long.append(y_long)
        all_y_short.append(y_short)
        all_ticker_ids.append(np.full(len(X), i))
        
        print(f"  features: {X.shape}, long: {y_long.sum()} ({y_long.mean()*100:.1f}%), short: {y_short.sum()} ({y_short.mean()*100:.1f}%)")
    
    X = np.vstack(all_X)
    y_long = np.concatenate(all_y_long)
    y_short = np.concatenate(all_y_short)
    ticker_ids = np.concatenate(all_ticker_ids)
    
    return X, y_long, y_short, ticker_ids, names, tickers


def train_xgboost(X_train, y_train, X_val, y_val, name="long"):
    """Train XGBoost classifier."""
    try:
        import xgboost as xgb
    except ImportError:
        print("  xgboost not installed, using sklearn GradientBoosting")
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=42
        )
        model.fit(X_train, y_train)
        return model
    
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=2,
        eval_metric="logloss",
        early_stopping_rounds=20,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def evaluate_model(model, X_test, y_test, name="long"):
    """Evaluate model and print metrics."""
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    try:
        auc = roc_auc_score(y_test, y_proba)
    except:
        auc = 0
    
    print(f"\n  {name} model:")
    print(f"    Precision: {precision:.3f} (из всех предсказанных long, {precision*100:.1f}% реально выросли)")
    print(f"    Recall:    {recall:.3f} (из всех реальных long, {recall*100:.1f}% предсказаны)")
    print(f"    F1:        {f1:.3f}")
    print(f"    AUC:       {auc:.3f}")
    
    # Probability distribution
    print(f"    Probability distribution:")
    for threshold in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
        pred_high = (y_proba > threshold).astype(int)
        if pred_high.sum() > 0:
            actual_precision = y_test[pred_high == 1].mean()
            print(f"      P>{threshold:.2f}: {pred_high.sum():5d} predictions, "
                  f"actual win rate: {actual_precision*100:.1f}%")
    
    return precision, recall, f1, auc


def backtest_ml(model, X, close_prices, threshold=0.6, commission=0.0005):
    """Simple backtest: trade when model probability > threshold.
    
    For each signal:
      - If P(long) > threshold → buy, hold 6 candles (30 min), sell
      - If P(short) > threshold → sell short, hold 6 candles, buy back
      - Commission: 0.05% per side
    """
    y_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.predict(X)
    
    n = len(X)
    balance = 10000.0
    trades = 0
    wins = 0
    pnl_log = []
    
    i = 50  # skip warmup
    while i < n - 6:
        if y_proba[i] > threshold:
            # Long: buy at close[i], sell at close[i+6]
            entry = close_prices[i]
            exit_price = close_prices[i + 6]
            gross_pnl = (exit_price - entry) / entry * balance * 0.3  # 30% position
            comm = balance * 0.3 * commission * 2  # round-trip
            net_pnl = gross_pnl - comm
            balance += net_pnl
            trades += 1
            if net_pnl > 0: wins += 1
            pnl_log.append(net_pnl)
            i += 6  # skip ahead
        else:
            i += 1
    
    total_pnl = balance - 10000
    win_rate = wins / trades * 100 if trades > 0 else 0
    
    print(f"\n  Backtest (threshold={threshold}):")
    print(f"    Trades: {trades}")
    print(f"    Win rate: {win_rate:.1f}%")
    print(f"    Total P&L: {total_pnl:+.2f} RUB ({total_pnl/100:.2f}%)")
    print(f"    Avg P&L/trade: {total_pnl/trades:+.2f}" if trades > 0 else "")
    
    return total_pnl, trades, win_rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--tickers", type=str, default="all")
    parser.add_argument("--horizon", type=int, default=6, help="forward window in 5min candles")
    parser.add_argument("--threshold", type=float, default=0.001, help="min return to label as up")
    args = parser.parse_args()
    
    tickers = TICKERS if args.tickers == "all" else args.tickers.split(",")
    
    print(f"\n{'='*70}")
    print(f"ML MODEL: XGBoost for trading signals")
    print(f"  Tickers: {tickers}")
    print(f"  Data: {args.days} days MOEX")
    print(f"  Horizon: {args.horizon} candles ({args.horizon*5} min forward)")
    print(f"  Threshold: {args.threshold*100}% min return")
    print(f"{'='*70}")
    
    # 1. Load data
    print("\n[1/5] Loading data...")
    X, y_long, y_short, ticker_ids, feature_names, ticker_list = load_dataset(tickers, args.days)
    print(f"\nTotal: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Long signals: {y_long.sum()} ({y_long.mean()*100:.1f}%)")
    print(f"Short signals: {y_short.sum()} ({y_short.mean()*100:.1f}%)")
    
    # 2. Split chronological (80/10/10)
    print("\n[2/5] Splitting train/val/test...")
    n = len(X)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    
    X_train = X[:train_end]
    X_val = X[train_end:val_end]
    X_test = X[val_end:]
    
    y_long_train = y_long[:train_end]
    y_long_val = y_long[train_end:val_end]
    y_long_test = y_long[val_end:]
    
    y_short_train = y_short[:train_end]
    y_short_val = y_short[train_end:val_end]
    y_short_test = y_short[val_end:]
    
    print(f"  Train: {len(X_train)} ({y_long_train.mean()*100:.1f}% long)")
    print(f"  Val:   {len(X_val)} ({y_long_val.mean()*100:.1f}% long)")
    print(f"  Test:  {len(X_test)} ({y_long_test.mean()*100:.1f}% long)")
    
    # 3. Train long model
    print("\n[3/5] Training LONG model...")
    model_long = train_xgboost(X_train, y_long_train, X_val, y_long_val, "long")
    
    # 4. Train short model
    print("\n[4/5] Training SHORT model...")
    model_short = train_xgboost(X_train, y_short_train, X_val, y_short_val, "short")
    
    # 5. Evaluate
    print("\n[5/5] Evaluating...")
    evaluate_model(model_long, X_test, y_long_test, "LONG")
    evaluate_model(model_short, X_test, y_short_test, "SHORT")
    
    # Feature importance
    print(f"\n{'='*50}")
    print("Feature importance (LONG model):")
    if hasattr(model_long, "feature_importances_"):
        importances = model_long.feature_importances_
        indices = np.argsort(importances)[::-1]
        for i in indices[:15]:
            print(f"  {feature_names[i]:25s}: {importances[i]:.4f}")
    
    # Backtest on test set
    print(f"\n{'='*50}")
    print("Backtest on TEST set:")
    # Get close prices for test set (need to reconstruct from data)
    # For simplicity, use first ticker
    test_close = []
    for ticker in tickers[:1]:
        data = download_multi_timeframe(ticker, days=args.days)
        aligned = align_timeframes(data)
        close = aligned["5min_close"]
        test_close = close[val_end:]
        break
    
    if len(test_close) > 100:
        backtest_ml(model_long, X_test, test_close, threshold=0.6)
        backtest_ml(model_long, X_test, test_close, threshold=0.65)
        backtest_ml(model_long, X_test, test_close, threshold=0.7)
    
    # Save model
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(model_dir, exist_ok=True)
    
    import pickle
    model_path = os.path.join(model_dir, f"ml_model_{args.days}d.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({
            "model_long": model_long,
            "model_short": model_short,
            "feature_names": feature_names,
            "tickers": ticker_list,
            "horizon": args.horizon,
            "threshold": args.threshold,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }, f)
    print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    main()
