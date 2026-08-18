#!/usr/bin/env python3
"""Backtest v4 regime-aware binary classifiers on 11 tickers × 180 days.

Compares v4 (regime-specific XGBoost binary classifier) against:
  - Buy & Hold per ticker
  - v2 meta-classifier (loaded from /root/ai-trader-evolution/ml/meta_models/meta_classifier.pkl)
    - Runs inference on each sampled bar (from meta_labels_v2.npz)
    - Picks strategy class → uses precomputed pnls_matrix[i, predicted_class] for P&L
  - Best single strategy from Monte Carlo: random_hold_short
    - Uses precomputed pnls_matrix[:, 5] for P&L

v4 simulation (per ticker, 180 days of 5min bars):
  For each bar t:
    1. Compute 31 features (same as training, ml_features.compute_features)
    2. Compute regime via meta_labeler_v2.compute_regime_v2
    3. If regime has trained model: load it, predict P(up) via predict_proba
       Else: use rule-based fallback (OVERSOLD_BOUNCE=LONG, OVERBOUGHT_REVERSAL=SHORT)
    4. Decision: P>0.6 → LONG, P<0.4 → SHORT, else FLAT
    5. Sequential position simulation (one position at a time):
       - If position open and (t - entry_bar) >= 6: close at close[t]
       - If no position and signal != FLAT: open at close[t]
       - P&L per trade = direction * (close[t+6]/close[t] - 1) * NOTIONAL - 2*0.05%*NOTIONAL
  Initial capital = 10000 RUB per ticker (NOT compound — each trade uses NOTIONAL).

Outputs:
  - /root/ai-trader-evolution/ml/meta_models_v2/v4_backtest_result.json
  - Stdout summary table

Usage:
  python3 backtest_v4.py
  python3 backtest_v4.py --days 180 --notional 10000
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

SCRIPT_DIR = Path("/root/ai-trader-evolution/ml")
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "fast_mc"))

from ml_data_pipeline import download_multi_timeframe, align_timeframes, TICKERS
from ml_features import compute_features
from meta_labeler_v2 import compute_regime_v2, REGIME_NAMES
from fast_backtest_v2 import precompute_indicators

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODELS_DIR = Path("/root/ai-trader-evolution/ml/meta_models_v2")
METADATA_PATH = MODELS_DIR / "regime_models_v4_metadata.json"
OUTPUT_PATH = MODELS_DIR / "v4_backtest_result.json"

V2_MODEL_PATH = Path("/root/ai-trader-evolution/ml/meta_models/meta_classifier.pkl")
V2_META_PATH = Path("/root/ai-trader-evolution/ml/meta_models/meta_metadata.json")
LABELS_NPZ_PATH = Path("/root/ai-trader-evolution/ml/data_cache/meta_labels_v2.npz")

LOG_FILE = "/var/log/ai-trader-v4-backtest.log"

# ---------------------------------------------------------------------------
# Config (from v4 training metadata)
# ---------------------------------------------------------------------------
LONG_THRESHOLD = 0.6
SHORT_THRESHOLD = 0.4
HORIZON_BARS = 6  # 30 min on 5min candles
COMMISSION_PER_SIDE = 0.0005  # 0.05% per side
NOTIONAL = 10000.0  # RUB per trade


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


# ---------------------------------------------------------------------------
# Load v4 models (one per regime)
# ---------------------------------------------------------------------------
def load_v4_models(metadata: dict) -> dict:
    """Load 10 trained XGBClassifier .pkl models keyed by regime_name.

    The .pkl files are dicts like:
        {"model": XGBClassifier, "feature_names": [...], "regime": str,
         "long_threshold": 0.6, "short_threshold": 0.4}
    """
    import xgboost as xgb  # noqa: F401 — ensure available
    models = {}
    for r_name in REGIME_NAMES:
        info = metadata.get(r_name, {})
        if info.get("skipped") or info.get("model_file") is None:
            continue
        pkl_path = MODELS_DIR / info["pkl_file"]
        if not pkl_path.exists():
            log(f"  WARN: {r_name} model file missing: {pkl_path}")
            continue
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)
        # Unwrap dict if needed
        if isinstance(obj, dict) and "model" in obj:
            m = obj["model"]
        else:
            m = obj
        models[r_name] = m
        log(f"  loaded {r_name}: {info['pkl_file']} "
            f"(n_samples={info.get('n_samples', '?')}, "
            f"test_precision={info.get('test_precision', '?'):.3f})")
    return models


# ---------------------------------------------------------------------------
# v4 backtest on one ticker
# ---------------------------------------------------------------------------
def backtest_v4_ticker(aligned: dict, models: dict, feature_names: list) -> dict:
    """Run v4 backtest on one ticker's aligned multi-timeframe data.

    Returns both full-period P&L AND out-of-sample (OOS) P&L computed on the
    last 15% of bars (matching the test slice used during v4 training, see
    train_regime_models_v4.py lines 421-427: train=70%, val=15%, test=15%).
    """
    close5 = aligned["5min_close"]
    high5 = aligned["5min_high"]
    low5 = aligned["5min_low"]
    open5 = aligned["5min_open"]
    vol5 = aligned["5min_volume"]
    n = len(close5)

    # 1. Compute 31 base features
    X, names = compute_features(aligned)
    name_to_idx = {n_: i for i, n_ in enumerate(names)}
    X_v4 = np.zeros((n, len(feature_names)), dtype=np.float32)
    for j, fname in enumerate(feature_names):
        if fname in name_to_idx:
            X_v4[:, j] = X[:, name_to_idx[fname]]
    # Replace NaN/Inf with 0
    X_v4 = np.nan_to_num(X_v4, nan=0.0, posinf=0.0, neginf=0.0)

    # 2. Compute regime per bar
    ind = precompute_indicators(open5, close5, high5, low5, vol5)
    regime = compute_regime_v2(close5, high5, low5, ind)

    # 3. Determine OOS test-slice boundaries (matches training split)
    test_start_idx = int(n * 0.85)  # last 15% = OOS test slice used during training

    # 4. Walk through bars, simulate sequential positions (6-bar hold, no overlap)
    pnl_total = 0.0
    pnl_oos = 0.0       # P&L only on bars within [test_start_idx, n)
    trades = 0
    trades_oos = 0
    wins = 0
    wins_oos = 0
    flat_count = 0
    per_regime = {r_name: {"trades": 0, "wins": 0, "pnl": 0.0} for r_name in REGIME_NAMES}
    long_count = 0
    short_count = 0
    position = None  # {'direction', 'entry_price', 'entry_bar', 'regime'}

    # Pre-compute model list per regime id (avoids dict lookup per bar)
    model_per_rid = [models.get(REGIME_NAMES[r]) for r in range(len(REGIME_NAMES))]

    for t in range(n - 1):
        if position is not None:
            # Check if HORIZON_BARS elapsed → close position
            if t - position["entry_bar"] >= HORIZON_BARS:
                exit_price = float(close5[t])
                gross = (exit_price / position["entry_price"] - 1.0) * position["direction"] * NOTIONAL
                commission = 2.0 * COMMISSION_PER_SIDE * NOTIONAL
                trade_pnl = gross - commission
                pnl_total += trade_pnl
                # OOS attribution: trade is OOS if entry bar is in test slice
                is_oos = position["entry_bar"] >= test_start_idx
                if is_oos:
                    pnl_oos += trade_pnl
                    trades_oos += 1
                    if trade_pnl > 0:
                        wins_oos += 1
                per_regime[position["regime"]]["pnl"] += trade_pnl
                if trade_pnl > 0:
                    wins += 1
                    per_regime[position["regime"]]["wins"] += 1
                position = None

        if position is None:
            # Look for new signal at bar t
            r_id = int(regime[t])
            r_name = REGIME_NAMES[r_id]
            model = model_per_rid[r_id]
            if model is None:
                # Rule-based fallback (matches MetaSelectorV4.ts)
                if r_name == "OVERSOLD_BOUNCE":
                    direction = 1
                elif r_name == "OVERBOUGHT_REVERSAL":
                    direction = -1
                else:
                    direction = 0
            else:
                p_up = float(model.predict_proba(X_v4[t:t + 1])[0, 1])
                if p_up > LONG_THRESHOLD:
                    direction = 1
                elif p_up < SHORT_THRESHOLD:
                    direction = -1
                else:
                    direction = 0

            if direction == 0:
                flat_count += 1
                continue

            # Enter at close[t]
            position = {
                "direction": direction,
                "entry_price": float(close5[t]),
                "entry_bar": t,
                "regime": r_name,
            }
            trades += 1
            per_regime[r_name]["trades"] += 1
            if direction == 1:
                long_count += 1
            else:
                short_count += 1

    # Close any open position at the last bar
    if position is not None:
        t = n - 1
        exit_price = float(close5[t])
        gross = (exit_price / position["entry_price"] - 1.0) * position["direction"] * NOTIONAL
        commission = 2.0 * COMMISSION_PER_SIDE * NOTIONAL
        trade_pnl = gross - commission
        pnl_total += trade_pnl
        is_oos = position["entry_bar"] >= test_start_idx
        if is_oos:
            pnl_oos += trade_pnl
            trades_oos += 1
            if trade_pnl > 0:
                wins_oos += 1
        per_regime[position["regime"]]["pnl"] += trade_pnl
        if trade_pnl > 0:
            wins += 1
            per_regime[position["regime"]]["wins"] += 1

    # Buy & Hold for this ticker
    bh_pnl = (float(close5[-1]) / float(close5[0]) - 1.0) * NOTIONAL - 2.0 * COMMISSION_PER_SIDE * NOTIONAL
    bh_return_pct = (float(close5[-1]) / float(close5[0]) - 1.0) * 100.0
    # Buy & Hold on test slice only (last 15%)
    bh_oos_pnl = (float(close5[-1]) / float(close5[test_start_idx]) - 1.0) * NOTIONAL - 2.0 * COMMISSION_PER_SIDE * NOTIONAL

    win_rate = float(wins / max(1, trades))
    win_rate_oos = float(wins_oos / max(1, trades_oos))

    return {
        "n_bars": int(n),
        "test_start_idx": int(test_start_idx),
        "pnl": float(pnl_total),
        "trades": int(trades),
        "wins": int(wins),
        "win_rate": float(win_rate),
        # OOS test-slice metrics (matches v4 training test split — last 15%)
        "pnl_oos": float(pnl_oos),
        "trades_oos": int(trades_oos),
        "wins_oos": int(wins_oos),
        "win_rate_oos": float(win_rate_oos),
        "buy_hold_oos_pnl": float(bh_oos_pnl),
        "long_trades": int(long_count),
        "short_trades": int(short_count),
        "flat_count": int(flat_count),
        "buy_hold_pnl": float(bh_pnl),
        "buy_hold_return_pct": float(bh_return_pct),
        "per_regime": {
            k: {
                "trades": int(v["trades"]),
                "wins": int(v["wins"]),
                "pnl": float(v["pnl"]),
                "win_rate": float(v["wins"] / max(1, v["trades"])),
            }
            for k, v in per_regime.items()
        },
    }


# ---------------------------------------------------------------------------
# v2 meta-classifier + random_hold_short baselines (using precomputed pnls_matrix)
# ---------------------------------------------------------------------------
def map_npz_samples_to_tickers(labels: dict, days: int) -> list:
    """Determine which npz samples belong to which ticker.

    meta_labeler_v2.py builds samples per-ticker sequentially using:
        samples_this = len(range(window, n - 10, step))
    """
    tickers = labels["tickers"]
    window = labels["window_bars"]
    step = labels["step_bars"]

    sample_ticker_map = []
    cursor = 0
    for ti, t in enumerate(tickers):
        try:
            data = download_multi_timeframe(t, days=days)
            if "5min_close" not in data:
                continue
            aligned = align_timeframes(data)
            n = len(aligned["5min_close"])
            samples_this = len(range(window, n - 10, step))
            sample_ticker_map.append({
                "ticker": t,
                "ticker_idx": ti,
                "sample_start": cursor,
                "sample_end": cursor + samples_this,
                "n_samples": samples_this,
                "aligned": aligned,
            })
            cursor += samples_this
        except Exception as e:
            log(f"  [map] {t} skip: {e}")
            continue
    return sample_ticker_map, cursor


def compute_v2_baseline_per_ticker(labels: dict, sample_ticker_map: list,
                                    v2_model, v2_meta: dict) -> dict:
    """For each ticker:
       - Compute 33 features (31 base + regime + trend_slope) at each sampled bar
       - Predict strategy class via v2_model
       - Map to original strategy index (since classes_ is 0..19 but strategies are 0..21)
       - Sum pnls_matrix[i, predicted_strategy] for each sample i in this ticker's slice
       Also: random_hold_short = pnls_matrix[:, 5] summed per ticker
    """
    strategy_names = labels["strategy_names"]
    rhs_idx = strategy_names.index("random_hold_short")

    v2_feature_names = v2_meta["feature_names"]
    class_map_enc_to_orig = v2_meta["class_map_encoded_to_original"]

    results = {}
    for entry in sample_ticker_map:
        t = entry["ticker"]
        s_start = entry["sample_start"]
        s_end = entry["sample_end"]
        if s_end <= s_start:
            results[t] = {"v2_pnl": 0.0, "v2_trades": 0, "rhs_pnl": 0.0, "rhs_trades": 0}
            continue

        aligned = entry["aligned"]
        close5 = aligned["5min_close"]
        high5 = aligned["5min_high"]
        low5 = aligned["5min_low"]
        open5 = aligned["5min_open"]
        vol5 = aligned["5min_volume"]
        n = len(close5)

        # Compute features at every bar
        X, names = compute_features(aligned)

        # Compute regime + trend_slope (the v1 meta-classifier used a single
        # integer regime feature, NOT one-hot)
        ind = precompute_indicators(open5, close5, high5, low5, vol5)
        regime = compute_regime_v2(close5, high5, low5, ind)
        sma14 = ind.get("sma14", np.zeros(n))
        sma50 = ind.get("sma50", np.zeros(n))
        trend_slope = (sma50 - sma14) / (sma14 + 1e-9)

        # Build 33-feature matrix in v2_meta["feature_names"] order
        name_to_idx = {nm: i for i, nm in enumerate(names)}
        X_v2 = np.zeros((n, len(v2_feature_names)), dtype=np.float32)
        for j, fname in enumerate(v2_feature_names):
            if fname in name_to_idx:
                X_v2[:, j] = X[:, name_to_idx[fname]]
            elif fname == "regime":
                X_v2[:, j] = regime
            elif fname == "trend_slope":
                X_v2[:, j] = trend_slope
        X_v2 = np.nan_to_num(X_v2, nan=0.0, posinf=0.0, neginf=0.0)

        # Sample at bar_indices for this ticker
        bar_idx_t = labels["bar_indices"][s_start:s_end]
        # Guard against out-of-range indices
        valid_mask = (bar_idx_t >= 0) & (bar_idx_t < n)
        bar_idx_valid = bar_idx_t[valid_mask]

        if len(bar_idx_valid) == 0:
            results[t] = {"v2_pnl": 0.0, "v2_trades": 0, "rhs_pnl": 0.0, "rhs_trades": 0}
            continue

        X_v2_samples = X_v2[bar_idx_valid]
        # Predict class (0..19 encoded)
        try:
            classes = v2_model.predict(X_v2_samples)
        except Exception as e:
            log(f"  [v2-predict] {t} ERROR: {e}")
            results[t] = {"v2_pnl": 0.0, "v2_trades": 0, "rhs_pnl": 0.0, "rhs_trades": 0}
            continue

        # Map encoded class → original strategy index → strategy name
        # class_map_enc_to_orig keys are strings of encoded indices
        pnls_t = labels["pnls_matrix"][s_start:s_end][valid_mask]  # (n_valid, 22)

        v2_pnl = 0.0
        v2_trades = 0
        v2_strat_usage = {}
        for i, c in enumerate(classes):
            c_int = int(c)
            c_str = str(c_int)
            orig_idx = int(class_map_enc_to_orig.get(c_str, c_int))
            if orig_idx < pnls_t.shape[1]:
                v2_pnl += float(pnls_t[i, orig_idx])
                v2_trades += 1
                s_name = strategy_names[orig_idx]
                v2_strat_usage[s_name] = v2_strat_usage.get(s_name, 0) + 1

        # random_hold_short: pnls_matrix[:, rhs_idx]
        rhs_pnl = float(pnls_t[:, rhs_idx].sum())
        rhs_trades = int((pnls_t[:, rhs_idx] != 0).sum())

        results[t] = {
            "v2_pnl": v2_pnl,
            "v2_trades": v2_trades,
            "v2_strat_usage": v2_strat_usage,
            "rhs_pnl": rhs_pnl,
            "rhs_trades": rhs_trades,
        }
        log(f"  {t}: v2_pnl={v2_pnl:+.2f} ({v2_trades} samples), rhs_pnl={rhs_pnl:+.2f}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    global NOTIONAL
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--notional", type=float, default=NOTIONAL)
    args = parser.parse_args()
    NOTIONAL = args.notional

    log("=" * 78)
    log(f"Backtest v4 Regime-Aware Binary Classifiers (180 days, 11 tickers)")
    log(f"  long_threshold={LONG_THRESHOLD}, short_threshold={SHORT_THRESHOLD}")
    log(f"  horizon={HORIZON_BARS} bars ({HORIZON_BARS * 5} min), commission={COMMISSION_PER_SIDE*100:.3f}%/side")
    log(f"  notional={NOTIONAL} RUB per trade")
    log("=" * 78)

    # ---- Load v4 metadata + models ----
    log("\n[Step 1/5] Loading v4 metadata + 10 regime models...")
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    feature_names = metadata["_meta"]["feature_names"]
    log(f"  v4 metadata: {len(feature_names)} features, {metadata['_meta']['n_trained']} trained regimes")

    models = load_v4_models(metadata)
    log(f"  Loaded {len(models)} regime models (10 expected; 2 skipped regimes use rule-based fallback)")

    # ---- Load v2 baseline (meta-classifier) ----
    log("\n[Step 2/5] Loading v2 meta-classifier + precomputed pnls_matrix...")
    with open(V2_MODEL_PATH, "rb") as f:
        v2_model = pickle.load(f)
    with open(V2_META_PATH) as f:
        v2_meta = json.load(f)
    log(f"  v2 model: {v2_model.n_features_in_} features, {len(v2_model.classes_)} classes")
    log(f"  v2 feature order: {v2_meta['feature_names'][:5]}... + regime, trend_slope")

    labels_npz = np.load(str(LABELS_NPZ_PATH), allow_pickle=True)
    labels = {
        "bar_indices": labels_npz["bar_indices"],
        "regimes": labels_npz["regimes"],
        "best_strategies": labels_npz["best_strategies"],
        "pnls_matrix": labels_npz["pnls_matrix"],
        "strategy_names": list(labels_npz["strategy_names"]),
        "regime_names": list(labels_npz["regime_names"]),
        "tickers": list(labels_npz["tickers"]),
        "window_bars": int(labels_npz["window_bars"]),
        "step_bars": int(labels_npz["step_bars"]),
    }
    log(f"  Loaded meta_labels_v2.npz: {len(labels['bar_indices'])} samples, "
        f"{len(labels['strategy_names'])} strategies, "
        f"window={labels['window_bars']}, step={labels['step_bars']}")

    # ---- Map npz samples to tickers (also reuses aligned data) ----
    log("\n[Step 3/5] Mapping npz samples to tickers (loading MTF data per ticker)...")
    sample_ticker_map, total_mapped = map_npz_samples_to_tickers(labels, args.days)
    log(f"  Mapped {total_mapped} samples across {len(sample_ticker_map)} tickers")

    # ---- Run v4 backtest per ticker ----
    log("\n[Step 4/5] Backtesting v4 per ticker...")
    v4_results = {}
    for entry in sample_ticker_map:
        t = entry["ticker"]
        aligned = entry["aligned"]
        n_bars = len(aligned["5min_close"])
        t0 = time.time()
        r = backtest_v4_ticker(aligned, models, feature_names)
        v4_results[t] = r
        log(f"  {t}: n_bars={n_bars} v4_pnl={r['pnl']:+.2f} trades={r['trades']} "
            f"win={r['win_rate']*100:.1f}% long={r['long_trades']} short={r['short_trades']} "
            f"flat={r['flat_count']} buy_hold={r['buy_hold_pnl']:+.2f} ({r['buy_hold_return_pct']:+.2f}%) "
            f"OOS[{r['test_start_idx']}]: pnl={r['pnl_oos']:+.2f} trd={r['trades_oos']} "
            f"win={r['win_rate_oos']*100:.1f}% bh_oos={r['buy_hold_oos_pnl']:+.2f} "
            f"[{time.time()-t0:.1f}s]")

    # ---- Compute v2 + random_hold_short per ticker (using npz) ----
    log("\n[Step 5/5] Computing v2 meta-classifier + random_hold_short per ticker (precomputed pnls_matrix)...")
    v2_results = compute_v2_baseline_per_ticker(labels, sample_ticker_map, v2_model, v2_meta)

    # ---- Aggregate ----
    total_v4_pnl = sum(r["pnl"] for r in v4_results.values())
    total_v4_trades = sum(r["trades"] for r in v4_results.values())
    total_v4_wins = sum(r["wins"] for r in v4_results.values())
    total_v4_win_rate = total_v4_wins / max(1, total_v4_trades)
    # OOS aggregation (test slice = last 15% of bars, matching v4 training split)
    total_v4_oos_pnl = sum(r.get("pnl_oos", 0.0) for r in v4_results.values())
    total_v4_oos_trades = sum(r.get("trades_oos", 0) for r in v4_results.values())
    total_v4_oos_wins = sum(r.get("wins_oos", 0) for r in v4_results.values())
    total_v4_oos_win_rate = total_v4_oos_wins / max(1, total_v4_oos_trades)
    total_bh_oos_pnl = sum(r.get("buy_hold_oos_pnl", 0.0) for r in v4_results.values())
    total_bh_pnl = sum(r["buy_hold_pnl"] for r in v4_results.values())
    total_v2_pnl = sum(r["v2_pnl"] for r in v2_results.values())
    total_v2_trades = sum(r["v2_trades"] for r in v2_results.values())
    total_rhs_pnl = sum(r["rhs_pnl"] for r in v2_results.values())
    total_rhs_trades = sum(r["rhs_trades"] for r in v2_results.values())

    # Per-regime aggregation across all tickers
    per_regime_agg = {r_name: {"trades": 0, "wins": 0, "pnl": 0.0} for r_name in REGIME_NAMES}
    for t, r in v4_results.items():
        for r_name, info in r["per_regime"].items():
            per_regime_agg[r_name]["trades"] += info["trades"]
            per_regime_agg[r_name]["wins"] += info["wins"]
            per_regime_agg[r_name]["pnl"] += info["pnl"]

    # ---- Print summary table ----
    log("\n" + "=" * 110)
    log("SUMMARY: v4 vs Buy&Hold vs v2 meta-classifier vs random_hold_short")
    log("=" * 110)
    header = f"{'Ticker':<7} | {'v4 P&L':>10} | {'v4 trd':>7} | {'v4 win%':>8} | {'Buy&Hold':>10} | {'v2 P&L':>10} | {'random_hold':>12}"
    log(header)
    log("-" * 110)
    for t in [e["ticker"] for e in sample_ticker_map]:
        v4 = v4_results.get(t, {})
        v2 = v2_results.get(t, {})
        v4_pnl = v4.get("pnl", 0.0)
        v4_trades = v4.get("trades", 0)
        v4_wr = v4.get("win_rate", 0.0) * 100.0
        bh_pnl = v4.get("buy_hold_pnl", 0.0)
        v2_pnl = v2.get("v2_pnl", 0.0)
        rhs_pnl = v2.get("rhs_pnl", 0.0)
        log(f"{t:<7} | {v4_pnl:>+10.2f} | {v4_trades:>7d} | {v4_wr:>7.1f}% | {bh_pnl:>+10.2f} | {v2_pnl:>+10.2f} | {rhs_pnl:>+12.2f}")
    log("-" * 110)
    log(f"{'TOTAL':<7} | {total_v4_pnl:>+10.2f} | {total_v4_trades:>7d} | {total_v4_win_rate*100:>7.1f}% | {total_bh_pnl:>+10.2f} | {total_v2_pnl:>+10.2f} | {total_rhs_pnl:>+12.2f}")
    log("=" * 110)
    log(f"  v4 vs Buy&Hold:        {total_v4_pnl - total_bh_pnl:>+10.2f} RUB  ({(total_v4_pnl - total_bh_pnl) / max(1.0, abs(total_bh_pnl)) * 100:>+6.2f}%)")
    log(f"  v4 vs v2 meta-classifier:  {total_v4_pnl - total_v2_pnl:>+10.2f} RUB")
    log(f"  v4 vs random_hold_short:   {total_v4_pnl - total_rhs_pnl:>+10.2f} RUB")

    log("\n" + "=" * 110)
    log("OUT-OF-SAMPLE (OOS) — v4 P&L on last 15% of bars (matches v4 training test slice)")
    log("=" * 110)
    log(f"  v4 OOS P&L:           {total_v4_oos_pnl:>+10.2f} RUB   ({total_v4_oos_trades} trades, {total_v4_oos_win_rate*100:.1f}% win)")
    log(f"  Buy&Hold OOS P&L:     {total_bh_oos_pnl:>+10.2f} RUB   (last 15% of bars)")
    log(f"  v4 OOS vs Buy&Hold OOS:   {total_v4_oos_pnl - total_bh_oos_pnl:>+10.2f} RUB")
    log("  NOTE: full-period v4 P&L above is IN-SAMPLE on training data — OOS slice is the honest metric.")
    log("        v2/random_hold numbers use precomputed pnls_matrix (sampled bars) and are not directly comparable to v4's full-frequency numbers.")

    log("\n" + "=" * 78)
    log("PER-REGIME WIN RATE (v4, all tickers aggregated)")
    log("=" * 78)
    log(f"  {'Regime':<24} {'Trades':>7} {'Wins':>5} {'Win%':>7} {'P&L (RUB)':>12}")
    log("  " + "-" * 70)
    for r_name in REGIME_NAMES:
        info = per_regime_agg[r_name]
        if info["trades"] == 0:
            continue
        wr = info["wins"] / info["trades"] * 100.0
        log(f"  {r_name:<24} {info['trades']:>7d} {info['wins']:>5d} {wr:>6.1f}% {info['pnl']:>+12.2f}")

    # ---- Save JSON ----
    result = {
        "_meta": {
            "version": "v4_backtest",
            "ran_at": datetime.now(timezone(timedelta(hours=3))).isoformat(),
            "days_of_history": args.days,
            "tickers": [e["ticker"] for e in sample_ticker_map],
            "tickers_count": len(sample_ticker_map),
            "config": {
                "long_threshold": LONG_THRESHOLD,
                "short_threshold": SHORT_THRESHOLD,
                "horizon_bars": HORIZON_BARS,
                "horizon_minutes": HORIZON_BARS * 5,
                "commission_per_side": COMMISSION_PER_SIDE,
                "notional_rub": NOTIONAL,
            },
            "v4_metadata": {
                "n_trained_regimes": metadata["_meta"]["n_trained"],
                "n_skipped_regimes": metadata["_meta"]["n_skipped"],
                "feature_names": feature_names,
            },
            "v2_metadata": {
                "n_features": v2_meta["n_features"],
                "n_classes_effective": v2_meta["n_classes_effective"],
            },
            "comparison_baseline": {
                "v2_model_path": str(V2_MODEL_PATH),
                "labels_npz_path": str(LABELS_NPZ_PATH),
                "random_hold_short_idx": labels["strategy_names"].index("random_hold_short"),
                "npz_window_bars": labels["window_bars"],
                "npz_step_bars": labels["step_bars"],
            },
        },
        "totals": {
            "v4_pnl": total_v4_pnl,
            "v4_trades": total_v4_trades,
            "v4_win_rate": total_v4_win_rate,
            "v4_oos_pnl": total_v4_oos_pnl,
            "v4_oos_trades": total_v4_oos_trades,
            "v4_oos_win_rate": total_v4_oos_win_rate,
            "buy_hold_pnl": total_bh_pnl,
            "buy_hold_oos_pnl": total_bh_oos_pnl,
            "v2_meta_classifier_pnl": total_v2_pnl,
            "v2_meta_classifier_trades": total_v2_trades,
            "random_hold_short_pnl": total_rhs_pnl,
            "random_hold_short_trades": total_rhs_trades,
            "v4_vs_buy_hold_delta": total_v4_pnl - total_bh_pnl,
            "v4_vs_v2_delta": total_v4_pnl - total_v2_pnl,
            "v4_vs_rhs_delta": total_v4_pnl - total_rhs_pnl,
            "v4_oos_vs_buy_hold_oos_delta": total_v4_oos_pnl - total_bh_oos_pnl,
        },
        "per_ticker": {
            t: {
                "v4": v4_results.get(t, {}),
                "v2_meta_classifier": {
                    "pnl": v2_results.get(t, {}).get("v2_pnl", 0.0),
                    "trades": v2_results.get(t, {}).get("v2_trades", 0),
                    "strat_usage": v2_results.get(t, {}).get("v2_strat_usage", {}),
                },
                "random_hold_short": {
                    "pnl": v2_results.get(t, {}).get("rhs_pnl", 0.0),
                    "trades": v2_results.get(t, {}).get("rhs_trades", 0),
                },
            }
            for t in [e["ticker"] for e in sample_ticker_map]
        },
        "per_regime_aggregated": {
            r_name: {
                "trades": per_regime_agg[r_name]["trades"],
                "wins": per_regime_agg[r_name]["wins"],
                "pnl": per_regime_agg[r_name]["pnl"],
                "win_rate": per_regime_agg[r_name]["wins"] / max(1, per_regime_agg[r_name]["trades"]),
            }
            for r_name in REGIME_NAMES
            if per_regime_agg[r_name]["trades"] > 0
        },
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    log(f"\nSaved result → {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size//1024 if OUTPUT_PATH.exists() else 0} KB)")

    log("\n✅ DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
