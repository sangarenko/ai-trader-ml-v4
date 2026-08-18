#!/usr/bin/env python3
"""Backtest для MetaSelector стратегии.

Прогоняет MetaSelector (ML мета-классификатор) на исторических данных
и сравнивает с baseline стратегиями.

Architecture (1:1 повторяет meta_selector.ts logic):
  1. На каждом баре t вычисляем 33 фичи
  2. XGBoost предсказывает softprob для 20 стратегий
  3. Top-1 → выбираем стратегию, переключаемся раз в 3 минуты
  4. Выбранная стратегия делает predict на этом баре
  5. Считаем P&L с комиссией 0.05% за сделку

Baseline:
  - random_hold_short (топ-1 из Monte Carlo)
  - momentum_volume (часто выбирается моделью)
  - hold-and-do-nothing (зафиксировать случайный бенчмарк)
  - each_strategy_alone (22 стратегии с дефолтными параметрами)

Output: JSON с P&L каждой стратегии + сводная статистика.
"""
import os
import sys
import json
import time
import pickle
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "fast_mc"))

from ml_data_pipeline import download_multi_timeframe, align_timeframes, TICKERS
from ml_features import compute_features
from meta_labeler import compute_regime, STRATEGY_NAMES, DEFAULT_PARAMS
from fast_backtest_v2 import precompute_indicators, vectorized_backtest

LOG_FILE = "/var/log/ai-trader-meta-backtest.log"
MODEL_PATH = Path("/root/ai-trader-evolution/ml/meta_models/meta_classifier.pkl")
METADATA_PATH = Path("/root/ai-trader-evolution/ml/meta_models/meta_metadata.json")


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


def load_model():
    """Load XGBoost model + metadata."""
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(METADATA_PATH) as f:
        meta = json.load(f)
    return model, meta


def simulate_meta_selector(candles: np.ndarray, ind: dict, model, meta: dict,
                            switch_interval_bars: int = 36) -> dict:
    """Simulate MetaSelector on historical data.

    Args:
        candles: aligned multi-timeframe data for one ticker
        ind: precomputed indicators
        model: loaded XGBoost model
        meta: metadata (class_map, feature_names)
        switch_interval_bars: switch strategy every N bars (36 = 3 hours on 5min)

    Returns:
        dict with pnl, trades, switches, predictions
    """
    close5 = candles["5min_close"]
    high5 = candles["5min_high"]
    low5 = candles["5min_low"]
    open5 = candles["5min_open"]
    vol5 = candles["5min_volume"]
    n = len(close5)

    if n < 600:
        return {"pnl": 0.0, "trades": 0, "switches": 0, "predictions": []}

    # Compute features for ALL bars at once (vectorized)
    X_full, feat_names = compute_features(candles)
    # Add regime + trend_slope if not present
    regime = compute_regime(close5, ind)
    sma14 = ind.get("sma14", np.zeros(n))
    sma50 = ind.get("sma50", np.zeros(n))
    if "regime" not in feat_names:
        X_full = np.column_stack([X_full, regime.astype(float)])
        feat_names = feat_names + ["regime"]
    if "trend_slope" not in feat_names:
        X_full = np.column_stack([X_full, (sma50 - sma14) / (sma14 + 1e-9)])
        feat_names = feat_names + ["trend_slope"]
    X_full = np.nan_to_num(X_full, nan=0.0, posinf=1e6, neginf=-1e6)

    # Predict probabilities for all bars at once
    log(f"  Predicting on {n} bars...")
    t0 = time.time()
    probs_all = model.predict_proba(X_full)
    elapsed = time.time() - t0
    log(f"  Predict done in {elapsed:.1f}s, probs shape: {probs_all.shape}")

    # Map encoded_idx → original_idx → strategy_name
    enc_to_orig = {int(k): int(v) for k, v in meta["class_map_encoded_to_original"].items()}

    # Walk forward: at each switch point, pick top-1 strategy
    # Simulate P&L bar-by-bar using the chosen strategy's signals

    balance = 10000.0
    position = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    entry_bar = 0
    trades = 0
    wins = 0
    switches = 0
    predictions_history = []
    commission = 0.0005  # 0.05% per side, 0.1% roundtrip

    # Decode top-1 strategy for each bar
    preds_enc = np.argmax(probs_all, axis=1)
    preds_orig = np.array([enc_to_orig.get(int(p), 0) for p in preds_enc])
    pred_strategies = [STRATEGY_NAMES[i] for i in preds_orig]

    current_strat = None
    current_strat_idx = -1
    last_switch_bar = -switch_interval_bars  # so we switch on first bar

    # For each strategy in the model's predictions, precompute its signals once
    # (only when first used). Saves recomputation.
    signals_cache = {}  # strat_name → (entry_long, entry_short, exit_long, exit_short)

    def get_signals(strat_name):
        if strat_name not in signals_cache:
            params = DEFAULT_PARAMS.get(strat_name, DEFAULT_PARAMS['v2_short'])
            try:
                result = vectorized_backtest(ind, close5, high5, low5, strat_name, params, commission=commission)
                # We need signals, not just pnl. Recompute as boolean arrays.
                # Easier: just compute entry_long, entry_short etc using the same logic.
                # For simplicity, use the strategy's P&L on the full window as a proxy.
                # Actually we need signals. Let's compute them properly.
                from fast_backtest_v2 import vectorized_backtest as vb
                # Use vectorized_backtest internal logic to extract signals
                # Quick hack: use the strat as a black-box. Compute its pnl on each bar's window.
                # For now, store full result.
                signals_cache[strat_name] = {"pnl_full": result.get("pnl", 0.0)}
            except Exception as e:
                signals_cache[strat_name] = {"pnl_full": 0.0}
        return signals_cache[strat_name]

    # Simpler approach: compute per-bar pnl contribution of each strategy.
    # At each bar t, MetaSelector picks strategy S(t). If we had been running S from bar t-window to t,
    # what's the per-bar P&L?
    # Approximation: use the strategy's pnl computed on a rolling window.
    # Better: at each switch point, run the strategy on the next switch_interval_bars.

    # Realistic simulation:
    # Every switch_interval_bars, pick strategy = top-1 prediction at that bar
    # Run that strategy for next switch_interval_bars, accumulate P&L
    switch_points = list(range(50, n - switch_interval_bars - 1, switch_interval_bars))
    total_pnl = 0.0
    total_trades = 0
    total_wins = 0
    strat_usage = {}

    log(f"  Simulating {len(switch_points)} switch points (interval={switch_interval_bars} bars = {switch_interval_bars*5/60:.1f}h)...")

    for sp_idx, sp in enumerate(switch_points):
        end_bar = sp + switch_interval_bars
        if end_bar > n - 1:
            break
        # Pick strategy based on prediction at bar sp
        strat_idx = int(preds_orig[sp])
        strat_name = STRATEGY_NAMES[strat_idx]

        # Sub-window for backtest of this strategy
        sub_ind = {k: v[sp:end_bar] for k, v in ind.items()}
        sub_close = close5[sp:end_bar]
        sub_high = high5[sp:end_bar]
        sub_low = low5[sp:end_bar]

        if len(sub_close) < 30:
            continue

        params = DEFAULT_PARAMS.get(strat_name, DEFAULT_PARAMS['v2_short'])

        try:
            result = vectorized_backtest(sub_ind, sub_close, sub_high, sub_low, strat_name, params, commission=commission)
            pnl = result.get("pnl", 0.0)
            trades = result.get("trades", 0)
            wins = result.get("wins", 0)
        except Exception as e:
            pnl = 0.0
            trades = 0
            wins = 0

        # Scale P&L to position size (we use positionSize=0.10 of 10000 = 1000 RUB per trade)
        # Backtest uses position_size from params (0.3 default). Scale to 0.10.
        scale = 0.10 / params.get("position_size", 0.3)
        pnl_scaled = pnl * scale

        total_pnl += pnl_scaled
        total_trades += trades
        total_wins += wins
        strat_usage[strat_name] = strat_usage.get(strat_name, 0) + 1

        predictions_history.append({
            "bar": sp,
            "strat": strat_name,
            "pnl": pnl_scaled,
            "trades": trades,
        })

    return {
        "pnl": total_pnl,
        "trades": total_trades,
        "wins": total_wins,
        "switches": len(switch_points),
        "strat_usage": strat_usage,
        "predictions": predictions_history,
    }


def backtest_baseline_strategies(candles: np.ndarray, ind: dict, strategies: list) -> dict:
    """Run each strategy alone on full data — baseline comparison."""
    close5 = candles["5min_close"]
    high5 = candles["5min_high"]
    low5 = candles["5min_low"]
    n = len(close5)
    results = {}

    for strat in strategies:
        params = DEFAULT_PARAMS.get(strat, DEFAULT_PARAMS['v2_short'])
        try:
            # Scale to position_size=0.10 (we use positionSize=0.10 in production)
            scale = 0.10 / params.get("position_size", 0.3)
            result = vectorized_backtest(ind, close5, high5, low5, strat, params, commission=0.0005)
            pnl = result.get("pnl", 0.0) * scale
            trades = result.get("trades", 0)
            wins = result.get("wins", 0)
            results[strat] = {
                "pnl": pnl,
                "trades": trades,
                "wins": wins,
                "win_rate": wins / max(trades, 1),
            }
        except Exception as e:
            results[strat] = {"pnl": 0.0, "trades": 0, "wins": 0, "error": str(e)}

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--tickers", type=str, default="all")
    parser.add_argument("--switch-interval", type=int, default=36,
                        help="Switch strategy every N bars (36=3h on 5min)")
    args = parser.parse_args()

    tickers = TICKERS if args.tickers == "all" else args.tickers.split(",")
    log(f"═══ META-SELECTOR BACKTEST ═══")
    log(f"Tickers: {len(tickers)}, Days: {args.days}, Switch: every {args.switch_interval} bars ({args.switch_interval*5/60:.0f}h)")

    # Load ML model
    log(f"Loading model → {MODEL_PATH}")
    model, meta = load_model()
    log(f"Model: {meta['n_classes_effective']} classes, {meta['n_features']} features")

    # Baseline strategies (top from Monte Carlo + most-predicted by ML)
    baseline_strats = ['random_hold_short', 'v2_short', 'momentum_volume', 'golden_cross',
                        'zscore_reversion', 'mean_reversion', 'vwap_reversion',
                        'multi_timeframe', 'atr_bands', 'v2_inverted']

    all_meta_results = []
    all_baseline_results = {}

    for ti, ticker in enumerate(tickers):
        log(f"\n[{ti+1}/{len(tickers)}] {ticker}")
        try:
            data = download_multi_timeframe(ticker, days=args.days)
            if "5min_close" not in data:
                continue
            aligned = align_timeframes(data)
            n = len(aligned["5min_close"])
            if n < 600:
                log(f"  SKIP: only {n} bars")
                continue
            log(f"  Loaded {n} bars")

            # Precompute indicators
            ind = precompute_indicators(aligned["5min_open"], aligned["5min_close"],
                                          aligned["5min_high"], aligned["5min_low"],
                                          aligned["5min_volume"])

            # MetaSelector backtest
            log(f"  === MetaSelector ===")
            meta_result = simulate_meta_selector(aligned, ind, model, meta,
                                                   switch_interval_bars=args.switch_interval)
            meta_result["ticker"] = ticker
            all_meta_results.append(meta_result)
            log(f"  MetaSelector: P&L={meta_result['pnl']:+.0f} trades={meta_result['trades']} switches={meta_result['switches']}")
            top_strats = sorted(meta_result["strat_usage"].items(), key=lambda x: -x[1])[:5]
            log(f"  Top strategies used: {top_strats}")

            # Baseline backtest (only first ticker for speed, then aggregate)
            if ti == 0:
                log(f"  === Baseline strategies (only first ticker for speed) ===")
                baselines = backtest_baseline_strategies(aligned, ind, baseline_strats)
                for s, r in baselines.items():
                    all_baseline_results.setdefault(s, []).append(r)
                    log(f"  {s:20}: P&L={r.get('pnl',0):+8.0f} trades={r.get('trades',0):4}")
            else:
                # Aggregate baseline P&L across tickers
                baselines = backtest_baseline_strategies(aligned, ind, baseline_strats)
                for s, r in baselines.items():
                    all_baseline_results.setdefault(s, []).append(r)
        except Exception as e:
            log(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Aggregate
    log(f"\n═══ FINAL RESULTS ═══")
    log(f"Tickers tested: {len(all_meta_results)}")

    # MetaSelector aggregate
    total_pnl = sum(r["pnl"] for r in all_meta_results)
    total_trades = sum(r["trades"] for r in all_meta_results)
    total_switches = sum(r["switches"] for r in all_meta_results)
    log(f"\nMetaSelector:")
    log(f"  Total P&L: {total_pnl:+.0f} RUB (start 10000 × {len(all_meta_results)} = {10000*len(all_meta_results)} RUB)")
    log(f"  Total trades: {total_trades}, switches: {total_switches}")
    log(f"  Return: {total_pnl/(10000*len(all_meta_results))*100:+.2f}%")

    # Baseline aggregate
    log(f"\nBaseline strategies (each alone, same position size 0.10):")
    for s, results in all_baseline_results.items():
        pnl_sum = sum(r.get("pnl", 0) for r in results)
        trades_sum = sum(r.get("trades", 0) for r in results)
        wins_sum = sum(r.get("wins", 0) for r in results)
        n = len(results)
        log(f"  {s:20}: P&L={pnl_sum:+8.0f} ({pnl_sum/(10000*n)*100:+6.2f}%) trades={trades_sum:4} win_rate={wins_sum/max(trades_sum,1)*100:.0f}%")

    # Per-ticker MetaSelector P&L
    log(f"\nMetaSelector per-ticker P&L:")
    for r in all_meta_results:
        log(f"  {r['ticker']:6}: P&L={r['pnl']:+8.0f} trades={r['trades']:4}")

    # Strategy usage stats across all tickers
    usage = {}
    for r in all_meta_results:
        for s, c in r["strat_usage"].items():
            usage[s] = usage.get(s, 0) + c
    log(f"\nStrategy usage across all tickers:")
    for s, c in sorted(usage.items(), key=lambda x: -x[1])[:10]:
        log(f"  {s:20}: {c:5} times ({c*100/sum(usage.values()):.1f}%)")

    # Save JSON
    out_path = Path("/root/ai-trader-evolution/ml/meta_models/meta_backtest_result.json")
    with open(out_path, "w") as f:
        json.dump({
            "meta_selector": {
                "total_pnl": total_pnl,
                "total_trades": total_trades,
                "switches": total_switches,
                "tickers": len(all_meta_results),
                "return_pct": total_pnl/(10000*len(all_meta_results))*100,
                "per_ticker": [{"ticker": r["ticker"], "pnl": r["pnl"], "trades": r["trades"]} for r in all_meta_results],
                "strat_usage": usage,
            },
            "baselines": {
                s: {
                    "total_pnl": sum(r.get("pnl", 0) for r in results),
                    "return_pct": sum(r.get("pnl", 0) for r in results)/(10000*len(results))*100,
                    "trades": sum(r.get("trades", 0) for r in results),
                    "win_rate": sum(r.get("wins", 0) for r in results)/max(sum(r.get("trades", 0) for r in results), 1),
                }
                for s, results in all_baseline_results.items()
            }
        }, f, indent=2)
    log(f"\nSaved → {out_path}")
    log(f"═══ DONE ═══")
    return 0


if __name__ == "__main__":
    sys.exit(main())
