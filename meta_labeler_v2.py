#!/usr/bin/env python3
"""Расширенный Meta-Labeler v2 — 12 режимов рынка вместо 3.

Таксономия:
  1. STRONG_TREND_UP     — ADX>30, SMA50>SMA20>SMA14 (сильный рост)
  2. MILD_TREND_UP       — ADX 20-30, восходящие SMA
  3. RANGE_TIGHT         — ADX<15, ATR<медианы (узкий боковик)
  4. RANGE_WIDE          — ADX<15, ATR>медианы (широкий боковик)
  5. MILD_TREND_DOWN     — ADX 20-30, нисходящие SMA
  6. STRONG_TREND_DOWN   — ADX>30, SMA50<SMA20<SMA14 (сильное падение)
  7. CRASH               — ret_30 < -1.5% за 30 мин
  8. OVERSOLD_BOUNCE     — RSI<25 + price > SMA5 (отскок от дна)
  9. OVERBOUGHT_REVERSAL — RSI>75 + price < SMA5 (разворот от пика)
  10. BREAKOUT_UP        — close > max(high[-20:-1]) (пробой вверх)
  11. BREAKDOWN          — close < min(low[-20:-1]) (пробой вниз)
  12. HIGH_VOL_REGIME    — ATR > 1.5× медианы (высокая волатильность)

Для каждого бара t (с шагом step_bars):
  1. Определить regime из 12
  2. Прогнать все 22 стратегии на lookback окне
  3. Лучшая = argmax(P&L) → label

Выход: /root/ai-trader-evolution/ml/data_cache/meta_labels_v2.npz
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
from all_22_strategies import ALL_STRATEGIES
from meta_labeler import DEFAULT_PARAMS, STRAT_TO_IDX

OUTPUT_DIR = Path("/root/ai-trader-evolution/ml/data_cache")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = "/var/log/ai-trader-meta-labeler-v2.log"

# 12 режимов
REGIME_NAMES = [
    "STRONG_TREND_UP",
    "MILD_TREND_UP",
    "RANGE_TIGHT",
    "RANGE_WIDE",
    "MILD_TREND_DOWN",
    "STRONG_TREND_DOWN",
    "CRASH",
    "OVERSOLD_BOUNCE",
    "OVERBOUGHT_REVERSAL",
    "BREAKOUT_UP",
    "BREAKDOWN",
    "HIGH_VOL_REGIME",
]
REGIME_TO_IDX = {name: i for i, name in enumerate(REGIME_NAMES)}
STRATEGY_NAMES = list(ALL_STRATEGIES.keys())


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


def compute_regime_v2(close5: np.ndarray, high5: np.ndarray, low5: np.ndarray,
                       ind: dict) -> np.ndarray:
    """12-классовая разметка режима каждого бара.

    Priority order (важные состояния сначала):
      1. CRASH (отдельно, не зависит от других)
      2. OVERSOLD_BOUNCE / OVERBOUGHT_REVERSAL (RSI-based, на отскоках)
      3. BREAKOUT_UP / BREAKDOWN (Donchian breakout)
      4. STRONG_TREND_UP / STRONG_TREND_DOWN (ADX>30)
      5. MILD_TREND_UP / MILD_TREND_DOWN (ADX 20-30)
      6. RANGE_TIGHT / RANGE_WIDE (ADX<15, по ATR)
      7. HIGH_VOL_REGIME (fallback, ATR>1.5× медианы)
    """
    n = len(close5)
    sma5 = ind.get("sma5", np.zeros(n))
    sma14 = ind.get("sma14", np.zeros(n))
    sma20 = ind.get("sma20", np.zeros(n))
    sma50 = ind.get("sma50", np.zeros(n))
    adx = ind.get("adx", np.zeros(n))
    rsi14 = ind.get("rsi14", np.full(n, 50))
    atr = ind.get("atr", np.zeros(n))

    regime = np.zeros(n, dtype=np.int32)  # default = STRONG_TREND_UP = 0? No, default RANGE_TIGHT=2

    # Compute 30-bar return (30 * 5min = 2.5h)
    ret_30 = np.zeros(n)
    for i in range(30, n):
        ret_30[i] = (close5[i] - close5[i - 30]) / (close5[i - 30] + 1e-10)

    # Donchian channel (20 bars)
    donchian_high = np.zeros(n)
    donchian_low = np.zeros(n)
    for i in range(20, n):
        donchian_high[i] = np.max(high5[i - 20:i])
        donchian_low[i] = np.min(low5[i - 20:i])

    # ATR median (rolling 100 bars)
    atr_median = np.zeros(n)
    for i in range(100, n):
        atr_median[i] = np.median(atr[i - 100:i])

    # Default: RANGE_TIGHT (most neutral)
    regime[:] = REGIME_TO_IDX["RANGE_TIGHT"]

    # Trend conditions
    up_aligned = (sma50 > sma20 * 0.999) & (sma20 > sma14 * 0.999)
    down_aligned = (sma50 < sma20 * 1.001) & (sma20 < sma14 * 1.001)

    # 4. STRONG_TREND_UP/DOWN (ADX > 30)
    strong_up = up_aligned & (adx > 30)
    strong_down = down_aligned & (adx > 30)
    regime[strong_up] = REGIME_TO_IDX["STRONG_TREND_UP"]
    regime[strong_down] = REGIME_TO_IDX["STRONG_TREND_DOWN"]

    # 5. MILD_TREND_UP/DOWN (ADX 20-30)
    mild_up = up_aligned & (adx > 20) & (adx <= 30)
    mild_down = down_aligned & (adx > 20) & (adx <= 30)
    regime[mild_up] = REGIME_TO_IDX["MILD_TREND_UP"]
    regime[mild_down] = REGIME_TO_IDX["MILD_TREND_DOWN"]

    # 6. RANGE_TIGHT / RANGE_WIDE (ADX < 15)
    range_mask = adx < 15
    wide_range = range_mask & (atr > atr_median) & (atr_median > 0)
    tight_range = range_mask & (atr <= atr_median) & (atr_median > 0)
    regime[wide_range] = REGIME_TO_IDX["RANGE_WIDE"]
    regime[tight_range] = REGIME_TO_IDX["RANGE_TIGHT"]

    # 7. CRASH (ret_30 < -1.5%)
    crash_mask = ret_30 < -0.015
    regime[crash_mask] = REGIME_TO_IDX["CRASH"]

    # 8. OVERSOLD_BOUNCE (RSI < 25 + price > SMA5)
    oversold = (rsi14 < 25) & (close5 > sma5)
    regime[oversold] = REGIME_TO_IDX["OVERSOLD_BOUNCE"]

    # 9. OVERBOUGHT_REVERSAL (RSI > 75 + price < SMA5)
    overbought = (rsi14 > 75) & (close5 < sma5)
    regime[overbought] = REGIME_TO_IDX["OVERBOUGHT_REVERSAL"]

    # 10. BREAKOUT_UP (close > donchian_high)
    breakout_up = (close5 > donchian_high) & (donchian_high > 0)
    regime[breakout_up] = REGIME_TO_IDX["BREAKOUT_UP"]

    # 11. BREAKDOWN (close < donchian_low)
    breakdown = (close5 < donchian_low) & (donchian_low > 0)
    regime[breakdown] = REGIME_TO_IDX["BREAKDOWN"]

    # 12. HIGH_VOL_REGIME (ATR > 1.5× median) — override if extreme volatility
    high_vol = (atr > 1.5 * atr_median) & (atr_median > 0)
    regime[high_vol] = REGIME_TO_IDX["HIGH_VOL_REGIME"]

    # Warmup: first 100 bars = RANGE_TIGHT
    regime[:100] = REGIME_TO_IDX["RANGE_TIGHT"]

    return regime


def label_all_bars(ticker: str, aligned: dict, window_bars: int = 576,
                    step_bars: int = 36) -> dict:
    """Для каждого step-бара прогоняет все 22 стратегии на lookback окне."""
    close5 = aligned["5min_close"]
    high5 = aligned["5min_high"]
    low5 = aligned["5min_low"]
    n = len(close5)

    if n < window_bars + 60:
        log(f"  [{ticker}] SKIP: only {n} bars")
        return None

    ind = precompute_indicators(aligned["5min_open"], close5, high5, low5, aligned["5min_volume"])
    regimes = compute_regime_v2(close5, high5, low5, ind)

    sample_points = list(range(window_bars, n - 10, step_bars))
    n_samples = len(sample_points)
    log(f"  [{ticker}] {n_samples} sample points")

    bar_indices = np.array(sample_points, dtype=np.int32)
    bar_regimes = np.zeros(n_samples, dtype=np.int32)
    best_strats = np.zeros(n_samples, dtype=np.int32)
    pnls_matrix = np.zeros((n_samples, len(STRATEGY_NAMES)), dtype=np.float32)

    t0 = time.time()
    for si, end in enumerate(sample_points):
        start = end - window_bars
        bar_regimes[si] = regimes[end]

        sub_ind = {k: v[start:end] for k, v in ind.items()}
        sub_closes = close5[start:end]
        sub_highs = high5[start:end]
        sub_lows = low5[start:end]

        pnls = np.zeros(len(STRATEGY_NAMES), dtype=np.float32)
        for strat_i, strat in enumerate(STRATEGY_NAMES):
            params = DEFAULT_PARAMS.get(strat, DEFAULT_PARAMS['v2_short'])
            try:
                result = vectorized_backtest(sub_ind, sub_closes, sub_highs, sub_lows,
                                              strat, params, commission=0.0005)
                pnls[strat_i] = result.get("pnl", 0.0)
            except:
                pnls[strat_i] = 0.0

        pnls_matrix[si] = pnls
        best_strats[si] = int(np.argmax(pnls))

        if si % 100 == 0 and si > 0:
            elapsed = time.time() - t0
            log(f"  [{ticker}] {si}/{n_samples} ({si*100/n_samples:.0f}%) ETA={(n_samples-si)/(si/elapsed if elapsed else 1):.0f}s")

    log(f"  [{ticker}] DONE in {time.time()-t0:.0f}s")
    return {
        "ticker": ticker,
        "bar_indices": bar_indices,
        "regimes": bar_regimes,
        "best_strategies": best_strats,
        "pnls_matrix": pnls_matrix,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--tickers", type=str, default="all")
    parser.add_argument("--window", type=int, default=576)
    parser.add_argument("--step", type=int, default=36)
    args = parser.parse_args()

    tickers = TICKERS if args.tickers == "all" else args.tickers.split(",")
    log(f"═══ META-LABELER v2 (12 regimes) START ═══")
    log(f"Tickers: {len(tickers)}, Days: {args.days}, Window: {args.window}, Step: {args.step}")
    log(f"Regimes: {len(REGIME_NAMES)}")

    all_results = []
    for ti, ticker in enumerate(tickers):
        log(f"\n[{ti+1}/{len(tickers)}] Loading {ticker}...")
        try:
            data = download_multi_timeframe(ticker, days=args.days)
            if "5min_close" not in data:
                continue
            aligned = align_timeframes(data)
            result = label_all_bars(ticker, aligned, args.window, args.step)
            if result:
                all_results.append(result)
        except Exception as e:
            log(f"  ERROR: {e}")

    if not all_results:
        log("No results!")
        return 1

    all_bar_idx = np.concatenate([r["bar_indices"] for r in all_results])
    all_regimes = np.concatenate([r["regimes"] for r in all_results])
    all_best = np.concatenate([r["best_strategies"] for r in all_results])
    all_pnls = np.vstack([r["pnls_matrix"] for r in all_results])

    log(f"\n=== STATS ({len(all_best)} samples) ===")
    log(f"\nRegime distribution:")
    for i, name in enumerate(REGIME_NAMES):
        count = (all_regimes == i).sum()
        log(f"  {i:2} {name:24}: {count:5d} ({count*100/len(all_regimes):.1f}%)")

    log(f"\nBest strategy distribution (top-10):")
    u, c = np.unique(all_best, return_counts=True)
    for i in np.argsort(-c)[:10]:
        log(f"  {STRATEGY_NAMES[i]:20}: {c[i]:5d} ({c[i]*100/len(all_best):.1f}%)")

    log(f"\n=== Regime × Best Strategy (top-3 per regime) ===")
    for r, rname in enumerate(REGIME_NAMES):
        mask = all_regimes == r
        if mask.sum() == 0: continue
        r_best = all_best[mask]
        u, c = np.unique(r_best, return_counts=True)
        o = np.argsort(-c)
        log(f"  {rname} (n={mask.sum()}):")
        for i in o[:3]:
            log(f"    {STRATEGY_NAMES[u[i]]:20}: {c[i]:4d} ({c[i]*100/mask.sum():.1f}%)")

    out_path = OUTPUT_DIR / "meta_labels_v2.npz"
    np.savez_compressed(str(out_path),
        bar_indices=all_bar_idx,
        regimes=all_regimes,
        best_strategies=all_best,
        pnls_matrix=all_pnls,
        strategy_names=np.array(STRATEGY_NAMES),
        regime_names=np.array(REGIME_NAMES),
        tickers=np.array([r["ticker"] for r in all_results]),
        window_bars=args.window,
        step_bars=args.step,
    )
    log(f"\nSaved → {out_path} ({out_path.stat().st_size/1024:.0f} KB)")
    log(f"═══ DONE ═══")
    return 0


if __name__ == "__main__":
    sys.exit(main())
