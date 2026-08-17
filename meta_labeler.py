#!/usr/bin/env python3
"""Meta-Labeler: размечает каждый бар по лучшей стратегии.

Алгоритм:
  1. Загрузить multi-TF данные (11 тикеров, 180 дней, 5-мин свечи)
  2. Для каждого бара t (с шагом step_bars):
     - Взять окно [t - window_bars, t] (например 576 свечей = 48 часов)
     - Прогнать все 22 стратегии на этом окне
     - Лучшая = argmax(P&L) → label
  3. Сохранить разметку: (ticker, bar_idx, regime, best_strategy, pnl_per_strategy)

Выход: /root/ai-trader-evolution/ml/data_cache/meta_labels.npz

Это питание для meta_trainer.py который обучит ML-мета-классификатор.
"""
import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "fast_mc"))

from ml_data_pipeline import download_multi_timeframe, align_timeframes, TICKERS
from fast_backtest_v2 import precompute_indicators, vectorized_backtest
from all_22_strategies import ALL_STRATEGIES, random_params

OUTPUT_DIR = Path("/root/ai-trader-evolution/ml/data_cache")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = "/var/log/ai-trader-meta-labeler.log"
DEFAULT_PARAMS_FILE = Path("/root/ai-trader-evolution/ml/meta_default_params.json")


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


# Дефолтные параметры для каждой стратегии (взятые из Monte Carlo топов или разумные)
DEFAULT_PARAMS = {
    'v2_short':           {'entry_sma_mult': 0.999, 'entry_rsi_min': 30, 'entry_rsi_max': 55, 'take_profit_pct': 0.02, 'hold_ticks': 120, 'exit_sma_mult': 1.003, 'position_size': 0.3},
    'multi_timeframe':    {'entry_sma_mult': 0.999, 'entry_rsi_min': 30, 'entry_rsi_max': 55, 'take_profit_pct': 0.02, 'hold_ticks': 120, 'exit_sma_mult': 1.003, 'position_size': 0.3},
    'v2_inverted':        {'entry_sma_mult': 0.999, 'entry_rsi_min': 25, 'entry_rsi_max': 55, 'take_profit_pct': 0.02, 'hold_ticks': 120, 'exit_sma_mult': 1.003, 'position_size': 0.3},
    'mean_reversion':     {'entry_sma_mult': 0.999, 'entry_rsi_min': 25, 'entry_rsi_max': 75, 'take_profit_pct': 0.02, 'hold_ticks': 60,  'exit_sma_mult': 1.003, 'position_size': 0.3},
    'trend_follow':        {'entry_sma_mult': 0.999, 'entry_rsi_min': 30, 'entry_rsi_max': 60, 'take_profit_pct': 0.02, 'hold_ticks': 120, 'exit_sma_mult': 1.003, 'position_size': 0.3},
    'random_hold_short':   {'entry_sma_mult': 0.999, 'entry_rsi_min': 22, 'entry_rsi_max': 51, 'take_profit_pct': 0.02, 'hold_ticks': 108, 'exit_sma_mult': 1.003, 'position_size': 0.39},
    'bb_reversion':        {'entry_sma_mult': 0.999, 'entry_rsi_min': 20, 'entry_rsi_max': 55, 'take_profit_pct': 0.02, 'hold_ticks': 60,  'exit_sma_mult': 1.003, 'position_size': 0.3},
    'macd_trend':          {'entry_sma_mult': 0.999, 'entry_rsi_min': 25, 'entry_rsi_max': 60, 'take_profit_pct': 0.02, 'hold_ticks': 120, 'exit_sma_mult': 1.003, 'position_size': 0.3},
    'donchian_breakout':   {'entry_sma_mult': 0.999, 'entry_rsi_min': 25, 'entry_rsi_max': 60, 'take_profit_pct': 0.02, 'hold_ticks': 120, 'exit_sma_mult': 1.003, 'position_size': 0.3},
    'stoch_oscillator':    {'entry_sma_mult': 0.999, 'entry_rsi_min': 20, 'entry_rsi_max': 55, 'take_profit_pct': 0.02, 'hold_ticks': 60,  'exit_sma_mult': 1.003, 'position_size': 0.3},
    'vwap_reversion':      {'entry_sma_mult': 0.999, 'entry_rsi_min': 25, 'entry_rsi_max': 60, 'take_profit_pct': 0.02, 'hold_ticks': 60,  'exit_sma_mult': 1.003, 'position_size': 0.3},
    'momentum_volume':     {'entry_sma_mult': 0.999, 'entry_rsi_min': 25, 'entry_rsi_max': 60, 'take_profit_pct': 0.02, 'hold_ticks': 60,  'exit_sma_mult': 1.003, 'position_size': 0.3},
    'connors_rsi2':        {'entry_sma_mult': 0.999, 'entry_rsi_min': 10, 'entry_rsi_max': 15, 'take_profit_pct': 0.02, 'hold_ticks': 60,  'exit_sma_mult': 1.003, 'position_size': 0.3},
    'zscore_reversion':    {'entry_sma_mult': 0.999, 'entry_rsi_min': 20, 'entry_rsi_max': 80, 'take_profit_pct': 0.02, 'hold_ticks': 120, 'exit_sma_mult': 1.003, 'position_size': 0.3},
    'supertrend':          {'entry_sma_mult': 1.000, 'entry_rsi_min': 25, 'entry_rsi_max': 50, 'take_profit_pct': 0.02, 'hold_ticks': 120, 'exit_sma_mult': 1.003, 'position_size': 0.3},
    'bollinger_squeeze':   {'entry_sma_mult': 0.999, 'entry_rsi_min': 25, 'entry_rsi_max': 55, 'take_profit_pct': 0.02, 'hold_ticks': 120, 'exit_sma_mult': 1.003, 'position_size': 0.3},
    'atr_bands':            {'entry_sma_mult': 0.999, 'entry_rsi_min': 25, 'entry_rsi_max': 60, 'take_profit_pct': 0.02, 'hold_ticks': 120, 'exit_sma_mult': 1.003, 'position_size': 0.3},
    'heikin_ashi':         {'entry_sma_mult': 0.999, 'entry_rsi_min': 25, 'entry_rsi_max': 60, 'take_profit_pct': 0.02, 'hold_ticks': 120, 'exit_sma_mult': 1.003, 'position_size': 0.3},
    'dual_thrust':         {'entry_sma_mult': 0.999, 'entry_rsi_min': 25, 'entry_rsi_max': 60, 'take_profit_pct': 0.02, 'hold_ticks': 120, 'exit_sma_mult': 1.003, 'position_size': 0.3},
    'awesome_oscillator':  {'entry_sma_mult': 0.999, 'entry_rsi_min': 25, 'entry_rsi_max': 60, 'take_profit_pct': 0.02, 'hold_ticks': 120, 'exit_sma_mult': 1.003, 'position_size': 0.3},
    'golden_cross':        {'entry_sma_mult': 0.999, 'entry_rsi_min': 30, 'entry_rsi_max': 55, 'take_profit_pct': 0.02, 'hold_ticks': 240, 'exit_sma_mult': 1.003, 'position_size': 0.3},
    'orb':                  {'entry_sma_mult': 0.999, 'entry_rsi_min': 25, 'entry_rsi_max': 55, 'take_profit_pct': 0.02, 'hold_ticks': 60,  'exit_sma_mult': 1.003, 'position_size': 0.3},
}

STRATEGY_NAMES = list(ALL_STRATEGIES.keys())
STRAT_TO_IDX = {s: i for i, s in enumerate(STRATEGY_NAMES)}


def compute_regime(close5: np.ndarray, ind: dict) -> np.ndarray:
    """Разметка режима: 0=RANGE, 1=TREND_UP, 2=TREND_DOWN."""
    n = len(close5)
    sma14 = ind.get("sma14", np.zeros(n))
    sma20 = ind.get("sma20", np.zeros(n))
    sma50 = ind.get("sma50", np.zeros(n))
    adx = ind.get("adx", np.zeros(n))

    up_trend = (sma50 > sma20 * 0.999) & (sma20 > sma14 * 0.999) & (adx > 20)
    down_trend = (sma50 < sma20 * 1.001) & (sma20 < sma14 * 1.001) & (adx > 20)

    regime = np.zeros(n, dtype=np.int32)
    regime[up_trend] = 1
    regime[down_trend] = 2
    regime[:50] = 0  # warmup
    return regime


def label_window(ind: dict, closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
                 start: int, end: int, strategy: str, params: dict) -> float:
    """Прогнать backtest одной стратегии на окне [start, end). Возвращает P&L."""
    # Slice indicators + price arrays
    sub_ind = {k: v[start:end] for k, v in ind.items()}
    sub_closes = closes[start:end]
    sub_highs = highs[start:end]
    sub_lows = lows[start:end]

    if len(sub_closes) < 60:  # too short
        return 0.0

    try:
        result = vectorized_backtest(sub_ind, sub_closes, sub_highs, sub_lows,
                                      strategy, params, commission=0.0005)
        # result = {"pnl": float, "trades": int, "wins": int}
        return result.get("pnl", 0.0)
    except Exception as e:
        return 0.0


def label_all_bars(ticker: str, aligned: dict, window_bars: int = 576, step_bars: int = 36) -> dict:
    """Для каждого step-бара прогоняет все 22 стратегии на окне window_bars.

    Args:
        ticker: тикер
        aligned: данные из align_timeframes
        window_bars: длина lookback окна (576 = 48 часов на 5-мин)
        step_bars: шаг разметки (36 = 3 часа)

    Returns:
        dict с массивами: bar_indices, regimes, best_strategies, pnls (n_strategies × n_bars)
    """
    close5 = aligned["5min_close"]
    high5 = aligned["5min_high"]
    low5 = aligned["5min_low"]
    n = len(close5)

    if n < window_bars + 60:
        log(f"  [{ticker}] SKIP: only {n} bars (need {window_bars + 60})")
        return None

    # Precompute indicators once for full data
    ind = precompute_indicators(aligned["5min_open"], close5, high5, low5, aligned["5min_volume"])

    # Regime per bar
    regimes = compute_regime(close5, ind)

    # Sample points: start from window_bars (so window is fully formed), step every step_bars
    sample_points = list(range(window_bars, n - 10, step_bars))
    n_samples = len(sample_points)
    log(f"  [{ticker}] {n_samples} sample points (window={window_bars}, step={step_bars})")

    bar_indices = np.array(sample_points, dtype=np.int32)
    bar_regimes = np.zeros(n_samples, dtype=np.int32)
    best_strats = np.zeros(n_samples, dtype=np.int32)
    pnls_matrix = np.zeros((n_samples, len(STRATEGY_NAMES)), dtype=np.float32)

    t0 = time.time()
    for si, end in enumerate(sample_points):
        start = end - window_bars
        bar_regimes[si] = regimes[end]

        pnls = np.zeros(len(STRATEGY_NAMES), dtype=np.float32)
        for strat_i, strat in enumerate(STRATEGY_NAMES):
            params = DEFAULT_PARAMS.get(strat, DEFAULT_PARAMS['v2_short'])
            pnl = label_window(ind, close5, high5, low5, start, end, strat, params)
            pnls[strat_i] = pnl

        pnls_matrix[si] = pnls
        best_strats[si] = int(np.argmax(pnls))

        if si % 50 == 0 and si > 0:
            elapsed = time.time() - t0
            rate = si / elapsed if elapsed > 0 else 0
            eta = (n_samples - si) / rate if rate > 0 else 0
            log(f"  [{ticker}] {si}/{n_samples} ({si*100/n_samples:.0f}%) ETA={eta:.0f}s")

    elapsed = time.time() - t0
    log(f"  [{ticker}] DONE in {elapsed:.0f}s ({n_samples/elapsed:.1f} samples/s)")

    return {
        "ticker": ticker,
        "bar_indices": bar_indices,
        "regimes": bar_regimes,
        "best_strategies": best_strats,
        "pnls_matrix": pnls_matrix,
        "strategy_names": STRATEGY_NAMES,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=180, help="История в днях")
    parser.add_argument("--tickers", type=str, default="all", help="all или SBER,GAZP,...")
    parser.add_argument("--window", type=int, default=576, help="Lookback окно в 5-мин барах")
    parser.add_argument("--step", type=int, default=36, help="Шаг разметки в 5-мин барах")
    args = parser.parse_args()

    tickers = TICKERS if args.tickers == "all" else args.tickers.split(",")
    log(f"═══ META-LABELER START ═══")
    log(f"Tickers: {tickers} ({len(tickers)})")
    log(f"Days: {args.days}, Window: {args.window} bars ({args.window*5/60:.0f}h), Step: {args.step} bars ({args.step*5/60:.0f}h)")

    # Save default params for reference
    DEFAULT_PARAMS_FILE.write_text(json.dumps(DEFAULT_PARAMS, indent=2))
    log(f"Saved default params → {DEFAULT_PARAMS_FILE}")

    all_results = []
    for ti, ticker in enumerate(tickers):
        log(f"\n[{ti+1}/{len(tickers)}] Loading {ticker}...")
        try:
            data = download_multi_timeframe(ticker, days=args.days)
            if "5min_close" not in data:
                log(f"  SKIP: no 5min data for {ticker}")
                continue
            aligned = align_timeframes(data)
            if len(aligned["5min_close"]) < args.window + 60:
                log(f"  SKIP: too few bars ({len(aligned['5min_close'])})")
                continue
            log(f"  Loaded {len(aligned['5min_close'])} 5min bars")
            result = label_all_bars(ticker, aligned, window_bars=args.window, step_bars=args.step)
            if result:
                all_results.append(result)
        except Exception as e:
            log(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    if not all_results:
        log("No results! Check data.")
        return 1

    # Aggregate
    total_samples = sum(len(r["bar_indices"]) for r in all_results)
    log(f"\n═══ AGGREGATING {total_samples} samples from {len(all_results)} tickers ═══")

    # Concat all
    all_bar_idx = np.concatenate([r["bar_indices"] for r in all_results])
    all_regimes = np.concatenate([r["regimes"] for r in all_results])
    all_best = np.concatenate([r["best_strategies"] for r in all_results])
    all_pnls = np.vstack([r["pnls_matrix"] for r in all_results])

    # Stats
    log(f"\n=== STATS ===")
    log(f"Total samples: {len(all_best)}")
    log(f"Regime distribution:")
    for r, name in [(0, "RANGE"), (1, "TREND_UP"), (2, "TREND_DOWN")]:
        count = (all_regimes == r).sum()
        log(f"  {name:12}: {count:5d} ({count*100/len(all_regimes):.1f}%)")
    log(f"\nBest strategy distribution (top-10):")
    unique, counts = np.unique(all_best, return_counts=True)
    order = np.argsort(-counts)
    for i in order[:10]:
        log(f"  {STRATEGY_NAMES[i]:20}: {counts[i]:5d} ({counts[i]*100/len(all_best):.1f}%)")

    # Cross-tab: regime × best strategy
    log(f"\n=== Regime × Best Strategy (top-3 per regime) ===")
    for r, rname in [(0, "RANGE"), (1, "TREND_UP"), (2, "TREND_DOWN")]:
        mask = all_regimes == r
        if mask.sum() == 0: continue
        r_best = all_best[mask]
        u, c = np.unique(r_best, return_counts=True)
        o = np.argsort(-c)
        log(f"  {rname}:")
        for i in o[:3]:
            log(f"    {STRATEGY_NAMES[u[i]]:20}: {c[i]:4d} ({c[i]*100/mask.sum():.1f}%)")

    # Save .npz
    out_path = OUTPUT_DIR / "meta_labels.npz"
    np.savez_compressed(str(out_path),
        bar_indices=all_bar_idx,
        regimes=all_regimes,
        best_strategies=all_best,
        pnls_matrix=all_pnls,
        strategy_names=np.array(STRATEGY_NAMES),
        tickers=np.array([r["ticker"] for r in all_results]),
        window_bars=args.window,
        step_bars=args.step,
    )
    log(f"\nSaved → {out_path}")
    log(f"File size: {out_path.stat().st_size / 1024:.0f} KB")
    log(f"═══ META-LABELER DONE ═══")
    return 0


if __name__ == "__main__":
    sys.exit(main())
