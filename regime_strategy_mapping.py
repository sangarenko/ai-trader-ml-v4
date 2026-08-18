#!/usr/bin/env python3
"""
regime_strategy_mapping.py
==========================

Build a hardcoded regime → strategy mapping from `meta_labels_v2.npz`.

For each of the 12 market regimes we:
  1. Slice pnls_matrix to the samples of that regime.
  2. Compute, for every one of the 22 strategies:
        - mean P&L across those samples
        - win rate = fraction of samples with P&L > 0
        - score   = mean_pnl * win_rate      (combines profitability + consistency)
  3. Pick the top-3 strategies by score (best + 2 fallbacks).

Outputs:
  * /root/ai-trader-evolution/ml/meta_models_v2/regime_strategy_mapping.json
  * A printed summary table.
  * A backtest comparison:
        - regime-filtered P&L  : run each bar with its regime's best strategy
        - always-run baselines:
              (a) best single strategy everywhere
              (b) all 22 strategies always active (naive deploy-all)

No ML training happens here — this is pure statistics over an existing
backtest matrix produced by meta_labeler_v2.py.

Run:
    python3 regime_strategy_mapping.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- paths ----
HERE        = Path(__file__).resolve().parent
NPZ_PATH    = HERE / "data_cache" / "meta_labels_v2.npz"
OUT_DIR     = HERE / "meta_models_v2"
OUT_JSON    = OUT_DIR / "regime_strategy_mapping.json"

# Reliability threshold: regimes with fewer samples than this are flagged
# as "low confidence" in the JSON output (mapping is still produced).
MIN_SAMPLES_RELIABLE = 30


# --------------------------------------------------------------- main ------
def build_mapping(data: np.lib.npyio.NpzFile) -> dict:
    """Compute regime → strategy mapping (top-3 by mean_pnl × win_rate)."""

    regimes        = data["regimes"]            # (N,) int
    pnls           = data["pnls_matrix"]         # (N, 22) float
    strategy_names = list(data["strategy_names"])
    regime_names   = list(data["regime_names"])

    n_samples, n_strat = pnls.shape
    assert len(regime_names) == 12
    assert len(strategy_names) == n_strat == 22

    mapping: dict[str, dict] = {}

    for r_idx, r_name in enumerate(regime_names):
        mask = regimes == r_idx
        n_r  = int(mask.sum())

        if n_r == 0:
            mapping[r_name] = {
                "best":      None,
                "fallback":  [],
                "mean_pnl":  0.0,
                "win_rate":  0.0,
                "score":     0.0,
                "n_samples": 0,
                "reliable":  False,
                "note":      "no samples observed",
            }
            continue

        pnls_r   = pnls[mask]                              # (n_r, 22)
        mean_pnl = pnls_r.mean(axis=0)                     # (22,)
        win_rate = (pnls_r > 0).mean(axis=0)               # (22,)  in [0,1]
        score    = mean_pnl * win_rate                     # (22,)  task formula

        # `score = mean_pnl * win_rate` has a degenerate case: any strategy
        # with win_rate = 0 gets score = 0 regardless of how negative its
        # mean is. With our data most regimes have NO profitable strategy,
        # so 0-win-rate strategies (e.g. macd_trend with mean=-229 RUB/bar)
        # would tie at score=0 and beat slightly-negative-mean strategies
        # with a few wins (e.g. bollinger_squeeze mean=-1.99, win=10.7%
        # → score=-0.21, which is *less than 0*).
        #
        # Fix: introduce an `effective_score` used ONLY for ranking:
        #   * if win_rate > 0 : effective_score = score  (task formula intact)
        #   * if win_rate = 0 : effective_score = mean_pnl (preserves the
        #     negative magnitude so a -229 mean strategy ranks below -1.99).
        # The `score` field in the JSON stays faithful to the task spec.
        effective_score = np.where(
            win_rate > 0,
            score,
            mean_pnl,        # strictly negative when mean_pnl < 0
        )

        # Lexicographic ranking (last key is primary):
        #   1. effective_score (desc)   — task formula with the 0-win fix
        #   2. mean_pnl        (desc)   — tiebreak: prefer higher expectancy
        #   3. win_rate        (desc)   — final tiebreak: prefer consistency
        order_desc = np.lexsort((-win_rate, -mean_pnl, -effective_score))
        top3 = order_desc[:3]

        # Diagnostic: how many strategies are "actually profitable" in this
        # regime? (mean_pnl > 0) — tells us whether the "best" is a real
        # winner or just the least-bad option.
        n_profitable = int((mean_pnl > 0).sum())
        selection_method = (
            "task_formula" if n_profitable > 0
            else "least_negative_mean"   # no profitable strategy; pick least-bad
        )

        best_idx   = int(top3[0])
        fb_indices = [int(i) for i in top3[1:3]]

        mapping[r_name] = {
            "best":      strategy_names[best_idx],
            "fallback":  [strategy_names[i] for i in fb_indices],
            "mean_pnl":  round(float(mean_pnl[best_idx]), 4),
            "win_rate":  round(float(win_rate[best_idx]), 4),
            "score":     round(float(score[best_idx]), 4),
            # effective_score is what actually drove the ranking; included
            # for transparency / debugging when it differs from `score`.
            "effective_score":    round(float(effective_score[best_idx]), 4),
            "selection_method":   selection_method,
            "n_samples":          n_r,
            "reliable":           n_r >= MIN_SAMPLES_RELIABLE,
            # Diagnostic: how many of the 22 strategies have positive mean
            # P&L in this regime? When 0, the "best" is just the least-bad
            # option and the regime should be considered "no-trade" or
            # "use global default" rather than committed to a regime-specific
            # strategy.
            "n_profitable_strategies": n_profitable,
            # Top-3 detail for traceability / debugging.
            "top3": [
                {
                    "strategy":       strategy_names[int(i)],
                    "mean_pnl":       round(float(mean_pnl[i]), 4),
                    "win_rate":       round(float(win_rate[i]), 4),
                    "score":          round(float(score[i]),    4),
                    "effective_score": round(float(effective_score[i]), 4),
                }
                for i in top3
            ],
        }

    return mapping


# --------------------------------------------------------------- backtest --
def backtest(data: np.lib.npyio.NpzFile, mapping: dict) -> dict:
    """
    Three numbers we want to compare:

      1. regime_filtered_total
            For every sample bar, run the BEST strategy assigned to that
            sample's regime (skip / zero-out other bars for other
            strategies). Sum of pnls[i, best_for_regime[regimes[i]]].

      2. always_run_best_single
            Pick ONE strategy that maximises sum of pnls across all bars
            and use it everywhere.  max_s  sum_i pnls[i, s].

      3. always_run_all_strategies
            Naive baseline: deploy all 22 strategies on every bar.
            Sum of pnls across the whole matrix (i.e. every strategy
            always active).

    Per-sample (per-bar) and per-regime breakdowns are also reported so we
    can see which regimes contribute most to the filtered total.
    """

    regimes        = data["regimes"]
    pnls           = data["pnls_matrix"]
    strategy_names = list(data["strategy_names"])
    regime_names   = list(data["regime_names"])

    n_samples, n_strat = pnls.shape

    # Index of the chosen best strategy per regime (looked up from mapping).
    strat_to_idx = {s: i for i, s in enumerate(strategy_names)}
    best_per_regime_idx = np.zeros(12, dtype=np.int64)
    for r_idx, r_name in enumerate(regime_names):
        best_name = mapping[r_name]["best"]
        best_per_regime_idx[r_idx] = (
            strat_to_idx[best_name] if best_name in strat_to_idx else -1
        )

    # 1. Regime-filtered: bar i uses the best strategy for regime[i].
    chosen_strat_per_sample = best_per_regime_idx[regimes]   # (N,)
    valid_mask              = chosen_strat_per_sample >= 0
    rows                    = np.arange(n_samples)[valid_mask]
    cols                    = chosen_strat_per_sample[valid_mask]
    regime_filtered_per_bar = np.zeros(n_samples, dtype=np.float64)
    regime_filtered_per_bar[rows] = pnls[rows, cols]
    regime_filtered_total   = float(regime_filtered_per_bar.sum())

    # Per-regime contribution to the filtered total (for the summary table).
    per_regime_contrib = {}
    for r_idx, r_name in enumerate(regime_names):
        m = (regimes == r_idx)
        per_regime_contrib[r_name] = {
            "n_samples":     int(m.sum()),
            "best_strategy": mapping[r_name]["best"],
            "filtered_pnl":  round(float(regime_filtered_per_bar[m].sum()), 2),
            "mean_pnl":      round(float(regime_filtered_per_bar[m].mean()) if m.sum() else 0.0, 4),
        }

    # 2. Always-run best single strategy.
    totals_per_strat   = pnls.sum(axis=0)                # (22,)
    best_single_idx    = int(np.argmax(totals_per_strat))
    always_run_best_total = float(totals_per_strat[best_single_idx])

    # 3. Naive: run ALL strategies on ALL bars.
    always_run_all_total = float(pnls.sum())

    # 4. Mean best achievable per bar (oracle: pick argmax per bar).
    #    Upper bound — impossible in practice but useful to compare.
    oracle_per_bar = pnls.max(axis=1)
    oracle_total   = float(oracle_per_bar.sum())

    return {
        "regime_filtered_total":     round(regime_filtered_total,    2),
        "always_run_best_single":    round(always_run_best_total,    2),
        "always_run_best_strategy":  strategy_names[best_single_idx],
        "always_run_all_strategies": round(always_run_all_total,     2),
        "oracle_per_bar_total":      round(oracle_total,             2),
        "n_samples":                 int(n_samples),
        "per_bar": {
            "regime_filtered": round(regime_filtered_total / n_samples, 4),
            "always_best":     round(always_run_best_total / n_samples, 4),
            "always_all":      round(always_run_all_total  / n_samples, 4),
            "oracle":          round(oracle_total          / n_samples, 4),
        },
        "per_regime": per_regime_contrib,
    }


# --------------------------------------------------------------- printing --
def print_summary(mapping: dict, backtest_result: dict) -> None:
    sep = "=" * 145
    print(sep)
    print("REGIME → STRATEGY MAPPING")
    print("  primary rank = effective_score :  win>0 → mean×win  |  win=0 → mean_pnl (preserves negative magnitude)")
    print("  tiebreaks: mean_pnl desc, then win_rate desc")
    print(sep)
    hdr = (
        f"{'#':>2}  {'Regime':<20}  {'Best strategy':<20} "
        f"{'Mean P&L':>10}  {'Win%':>6}  {'Score':>9}  {'EffScore':>9}  "
        f"{'N':>5}  {'Rel':>4}  {'#prof':>5}  {'Method':<22}  Fallbacks"
    )
    print(hdr)
    print("-" * 145)
    for i, (r_name, info) in enumerate(mapping.items()):
        best   = info["best"] or "—"
        meanp  = info["mean_pnl"]
        win    = info["win_rate"]
        score  = info["score"]
        eff    = info.get("effective_score", score)
        n      = info["n_samples"]
        rel    = "yes" if info["reliable"] else "NO"
        nprof  = info.get("n_profitable_strategies", "?")
        method = info.get("selection_method", "?")
        fb     = ", ".join(info["fallback"]) if info["fallback"] else "—"
        print(
            f"{i:>2}  {r_name:<20}  {best:<20} "
            f"{meanp:>+10.2f}  {win*100:>5.1f}%  {score:>+9.2f}  {eff:>+9.2f}  "
            f"{n:>5}  {rel:>4}  {nprof:>5}  {method:<22}  {fb}"
        )

    print()
    print(sep)
    print("BACKTEST COMPARISON")
    print(sep)
    print(f"  Total samples                : {backtest_result['n_samples']:>10d}")
    print(f"  Per-bar averages (RUB/bar):")
    for k, v in backtest_result["per_bar"].items():
        print(f"    {k:<18}            : {v:>+10.4f}")
    print()
    print(f"  Regime-filtered total P&L    : {backtest_result['regime_filtered_total']:>+12.2f} RUB")
    print(f"  Always-run best single strat : {backtest_result['always_run_best_single']:>+12.2f} RUB "
          f"({backtest_result['always_run_best_strategy']})")
    print(f"  Always-run ALL 22 strategies : {backtest_result['always_run_all_strategies']:>+12.2f} RUB")
    print(f"  Oracle (per-bar argmax)      : {backtest_result['oracle_per_bar_total']:>+12.2f} RUB  (upper bound)")
    print()

    # Edge of regime-filtered vs best single
    delta = backtest_result["regime_filtered_total"] - backtest_result["always_run_best_single"]
    base  = abs(backtest_result["always_run_best_single"]) or 1.0
    pct   = 100.0 * delta / base
    print(f"  Regime-filtered vs best single: {delta:>+12.2f} RUB  ({pct:>+7.2f}%)")

    print()
    print(sep)
    print("PER-REGIME CONTRIBUTION TO REGIME-FILTERED TOTAL")
    print(sep)
    print(f"  {'Regime':<20}  {'Best strategy':<20}  {'N':>5}  {'Filtered P&L':>14}  {'Mean/bar':>10}")
    print("  " + "-" * 90)
    for r_name, info in backtest_result["per_regime"].items():
        print(
            f"  {r_name:<20}  {info['best_strategy'] or '—':<20}  "
            f"{info['n_samples']:>5}  {info['filtered_pnl']:>+14.2f}  "
            f"{info['mean_pnl']:>+10.4f}"
        )
    print(sep)


# --------------------------------------------------------------- entry -----
def main() -> None:
    if not NPZ_PATH.exists():
        raise FileNotFoundError(f"meta_labels_v2.npz not found at {NPZ_PATH}")

    print(f"Loading {NPZ_PATH} ...")
    data = np.load(NPZ_PATH, allow_pickle=True)

    mapping = build_mapping(data)
    bt      = backtest(data, mapping)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_payload = {
        "meta": {
            "source":      str(NPZ_PATH),
            "window_bars": int(data["window_bars"]),
            "step_bars":   int(data["step_bars"]),
            "n_samples":   int(data["pnls_matrix"].shape[0]),
            "n_regimes":   12,
            "n_strategies": 22,
            "tickers":     list(data["tickers"]),
            "score_formula": "mean_pnl * win_rate",
            "min_samples_reliable": MIN_SAMPLES_RELIABLE,
        },
        "regimes":      mapping,
        "backtest":     bt,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out_payload, f, indent=2, ensure_ascii=False)
    print(f"Wrote {OUT_JSON}  ({OUT_JSON.stat().st_size} bytes)\n")

    print_summary(mapping, bt)


if __name__ == "__main__":
    main()
