/**
 * meta_selector_v4.ts — ML orchestrator that ties the v4 ML stack together.
 *
 * This is the MAIN strategy file that runs on the trader server.
 * Replaces the v2/v3 meta_selector.ts (20-class softmax over strategy pool)
 * with a per-regime binary classifier: P(price up >0.1% in next 30 min).
 *
 * Architecture (every predict() call):
 *   1. detectRegime(candles)         → 12-regime classification (0..11)
 *   2. If regime has ML model (10/12 do):
 *        compute 31 features (port of ml_features.py)
 *        loadModel(regime_*.json)   → cached on first hit
 *        predict_proba(model, X)    → P(up) ∈ (0, 1)
 *        P > 0.6 → LONG (1)
 *        P < 0.4 → SHORT (2)
 *        else    → FLAT (0)
 *      If regime has NO model (OVERSOLD_BOUNCE / OVERBOUGHT_REVERSAL):
 *        OVERSOLD_BOUNCE     → LONG  (buy the dip)
 *        OVERBOUGHT_REVERSAL → SHORT (sell the top)
 *   3. Log every prediction
 *
 * Tested precision @ P>0.6: 78-87% per regime — VERY GOOD.
 *
 * Model files: /opt/ai-trader/data/regime_<name>.json (preferred)
 *              /opt/ai-trader/src/strategies/regime_<name>.json (fallback)
 *
 * Pure TS — no Python at runtime.
 *
 * Author: Agent 5 (meta-selector-v4)
 */
import { detectRegime, Candle } from './regime_detector'
import { loadModel, predict_proba } from './xgboost_binary_ts'
import * as fs from 'fs'
import * as path from 'path'

// ─── Trader-server strategy interface ─────────────────────────────────────────
//
// Inline definitions (compatible with /opt/ai-trader/src/core/types.ts Candle
// and /opt/ai-trader/src/strategies/base.ts IStrategy via structural typing).
// When deploying to the trader server, you MAY replace these with:
//
//   import { Candle } from '../core/types'
//   import { IStrategy, StrategyContext } from './base'
//
// The class will be structurally assignable to the trader server's IStrategy.
// Both interfaces have the same shape (name, description, predict(...)).

export interface StrategyContext {
  [key: string]: any
}

export interface IStrategy {
  name: string
  description: string
  predict(
    candles: Candle[],
    idx: number,
    hasPosition: boolean,
    stepsHeld?: number,
    ctx?: StrategyContext,
  ): number
}

// ─── Constants ─────────────────────────────────────────────────────────────────

/** Decision thresholds from v4 training metadata. */
const LONG_THRESHOLD = 0.6
const SHORT_THRESHOLD = 0.4

/** Minimum candle history needed (covers max lookback of 144 bars + 50 warmup). */
const MIN_CANDLES = 200

/**
 * Regime index → model file name (lowercase). `null` = no model, use fallback.
 *
 * Order MUST match REGIME_NAMES in regime_detector.ts:
 *   0 STRONG_TREND_UP, 1 MILD_TREND_UP, 2 RANGE_TIGHT, 3 RANGE_WIDE,
 *   4 MILD_TREND_DOWN, 5 STRONG_TREND_DOWN, 6 CRASH,
 *   7 OVERSOLD_BOUNCE (no model), 8 OVERBOUGHT_REVERSAL (no model),
 *   9 BREAKOUT_UP, 10 BREAKDOWN, 11 HIGH_VOL_REGIME
 */
const REGIME_MODEL_FILES: Array<string | null> = [
  'regime_strong_trend_up.json',     // 0 STRONG_TREND_UP
  'regime_mild_trend_up.json',       // 1 MILD_TREND_UP
  'regime_range_tight.json',         // 2 RANGE_TIGHT
  'regime_range_wide.json',          // 3 RANGE_WIDE
  'regime_mild_trend_down.json',     // 4 MILD_TREND_DOWN
  'regime_strong_trend_down.json',   // 5 STRONG_TREND_DOWN
  'regime_crash.json',               // 6 CRASH
  null,                              // 7 OVERSOLD_BOUNCE       (fallback: LONG)
  null,                              // 8 OVERBOUGHT_REVERSAL   (fallback: SHORT)
  'regime_breakout_up.json',         // 9 BREAKOUT_UP
  'regime_breakdown.json',           // 10 BREAKDOWN
  'regime_high_vol_regime.json',     // 11 HIGH_VOL_REGIME
]

/** Search paths for model files (tried in order, first hit wins). */
const MODEL_SEARCH_PATHS = [
  '/opt/ai-trader/data',
  '/opt/ai-trader/src/strategies',
  __dirname,                           // same dir as this file (Bun/Node)
  process.cwd(),                       // CWD fallback
]

// ─── Feature engineering (TS port of ml_features.py) ──────────────────────────
//
// CRITICAL: every formula below mirrors /root/ai-trader-evolution/ml/ml_features.py
// EXACTLY (see agent6_cache/ml_features.py).  The 31-feature vector is in
// alphabetical order by feature name (matches training metadata):
//
//   ['1d_ret', '1d_trend', '1h_ret', '1h_rsi', '1h_trend', 'adx', 'atr_pct',
//    'bb_pct_b', 'bb_width', 'day_of_week', 'hour', 'macd_hist', 'macd_line',
//    'macd_signal', 'obv_slope', 'price_bb_lower', 'price_bb_upper',
//    'price_sma20', 'price_sma50', 'ret_1', 'ret_10', 'ret_30', 'ret_5',
//    'ret_5_log', 'rsi14', 'rsi2', 'sma14_sma20', 'sma20_sma50', 'sma5_sma14',
//    'stoch_k', 'vol_ratio']
//
// IMPORTANT differences from regime_detector.ts / older meta_selector.ts:
//   - RSI uses SIMPLE ROLLING MEAN (not Wilder smoothing). ml_features.py line ~82.
//   - ATR uses SIMPLE ROLLING MEAN (not Wilder).              ml_features.py line ~126.
//   - ADX is SIMPLIFIED = |SMA(up_moves,14) - SMA(down_moves,14)| * 100
//     (NOT the full Wilder DI+/DI- formula).                  ml_features.py line ~173.
//   - OBV slope uses 10-bar window (NOT 5).                   ml_features.py line ~135.
//   - Time features use MSK timezone ((UTC_hour + 3) % 24).    ml_features.py line ~167.

const EPS = 1e-10

/**
 * Simple causal rolling mean via cumsum.
 *   result[i < w]  = cumsum[i] / w   (matches numpy rolling_mean with cumsum trick:
 *                                   warmup gets partial sum / w, NOT / (i+1))
 *   result[i >= w] = (cumsum[i] - cumsum[i-w]) / w
 *
 * Identical to the `rolling_mean()` helper in ml_features.py.
 */
function rollingMean(arr: number[], w: number): number[] {
  const n = arr.length
  const result = new Array<number>(n).fill(0)
  if (n === 0) return result
  const cumsum = new Array<number>(n)
  let s = 0
  for (let i = 0; i < n; i++) {
    s += arr[i]
    cumsum[i] = s
  }
  for (let i = 0; i < n; i++) {
    if (i < w) {
      // Python rolling_mean returns cumsum[i] / w (smaller than true mean during warmup).
      result[i] = cumsum[i] / w
    } else {
      result[i] = (cumsum[i] - cumsum[i - w]) / w
    }
  }
  return result
}

/**
 * Causal SMA used in ml_features.py `causal_sma()` — DIFFERENT from rolling_mean:
 *   result[i < w]  = cumsum[i] / (i + 1)   (true partial mean)
 *   result[i >= w] = (cumsum[i] - cumsum[i-w]) / w
 *
 * Used for sma5/14/20/50.
 */
function causalSMA(arr: number[], w: number): number[] {
  const n = arr.length
  const result = new Array<number>(n).fill(0)
  if (n === 0) return result
  const cumsum = new Array<number>(n)
  let s = 0
  for (let i = 0; i < n; i++) {
    s += arr[i]
    cumsum[i] = s
  }
  for (let i = 0; i < n; i++) {
    if (i < w) {
      result[i] = cumsum[i] / (i + 1)
    } else {
      result[i] = (cumsum[i] - cumsum[i - w]) / w
    }
  }
  return result
}

/** Standard EMA with smoothing factor k = 2/(period+1). Matches `ema()` in ml_features.py. */
function ema(arr: number[], period: number): number[] {
  const n = arr.length
  const result = new Array<number>(n).fill(0)
  if (n === 0) return result
  const k = 2 / (period + 1)
  result[0] = arr[0]
  for (let i = 1; i < n; i++) {
    result[i] = arr[i] * k + result[i - 1] * (1 - k)
  }
  return result
}

/**
 * Compute the 31-feature vector at the LAST bar of `candles`.
 *
 * Returns a fixed-length array of 31 numbers, in alphabetical feature order.
 * Mirrors `compute_features()` in ml_features.py exactly.
 *
 * Higher-timeframe features (1h_*, 1d_*) are approximated from 5min candles
 * using the same approach as the existing v2 meta_selector.ts, since the
 * trader server only provides a single 5min candle stream:
 *   - 1h_ret:  (close[i] - close[i-12]) / close[i-12]            (12 × 5min = 60min)
 *   - 1d_ret:  (close[i] - close[i-144]) / close[i-144]          (~1 trading day)
 *   - 1h_trend: close[i] / sma(close, 120)[i]                    (~10 hours of 5min)
 *   - 1d_trend: close[i] / sma(close, 144)[i]                    (~1 trading day)
 *   - 1h_rsi:   rsi14(close)[i]                                  (5min RSI as 1h RSI proxy)
 *
 * The first 50 rows of the training data were zeroed (X[:50] = 0), so the
 * model handles "not enough history" gracefully.  We mirror that by returning
 * all-zeros if n < 50.
 */
export function computeMetaV4Features(candles: Candle[]): number[] {
  const n = candles.length
  const N_FEATURES = 31
  if (n < 50) {
    return new Array<number>(N_FEATURES).fill(0)
  }

  const close = new Array<number>(n)
  const high = new Array<number>(n)
  const low = new Array<number>(n)
  const vol = new Array<number>(n)
  const time = new Array<number>(n)
  for (let i = 0; i < n; i++) {
    close[i] = candles[i].close
    high[i] = candles[i].high
    low[i] = candles[i].low
    vol[i] = candles[i].volume
    time[i] = candles[i].time
  }
  const idx = n - 1

  // ── Returns (causal, no wraparound) ──────────────────────────────────────────
  const ret1 = new Array<number>(n).fill(0)
  const ret5 = new Array<number>(n).fill(0)
  const ret10 = new Array<number>(n).fill(0)
  const ret30 = new Array<number>(n).fill(0)
  const ret5log = new Array<number>(n).fill(0)
  for (let i = 1; i < n; i++) {
    ret1[i] = (close[i] - close[i - 1]) / (close[i - 1] + EPS)
  }
  for (let i = 5; i < n; i++) {
    ret5[i] = (close[i] - close[i - 5]) / (close[i - 5] + EPS)
    ret5log[i] = Math.log(close[i] / (close[i - 5] + EPS))
  }
  for (let i = 10; i < n; i++) {
    ret10[i] = (close[i] - close[i - 10]) / (close[i - 10] + EPS)
  }
  for (let i = 30; i < n; i++) {
    ret30[i] = (close[i] - close[i - 30]) / (close[i - 30] + EPS)
  }

  // ── SMAs ───────────────────────────────────────────────────────────────────
  const sma5 = causalSMA(close, 5)
  const sma14 = causalSMA(close, 14)
  const sma20 = causalSMA(close, 20)
  const sma50 = causalSMA(close, 50)
  // Approx SMAs for higher-TF features
  const sma120 = causalSMA(close, 120)  // ~10 hours of 5min candles
  const sma144 = causalSMA(close, 144)  // ~1 trading day of 5min candles

  const sma5_sma14 = sma5.map((v, i) => v / (sma14[i] + EPS))
  const sma14_sma20 = sma14.map((v, i) => v / (sma20[i] + EPS))
  const sma20_sma50 = sma20.map((v, i) => v / (sma50[i] + EPS))
  const price_sma20 = close.map((v, i) => v / (sma20[i] + EPS))
  const price_sma50 = close.map((v, i) => v / (sma50[i] + EPS))

  // ── RSI (SIMPLE rolling mean of gains/losses, NOT Wilder) ─────────────────────
  const deltas = new Array<number>(n)
  deltas[0] = 0
  for (let i = 1; i < n; i++) {
    deltas[i] = close[i] - close[i - 1]
  }
  const gains = new Array<number>(n)
  const losses = new Array<number>(n)
  for (let i = 0; i < n; i++) {
    const d = deltas[i]
    gains[i] = d > 0 ? d : 0
    losses[i] = d < 0 ? -d : 0
  }
  const avgGain14 = rollingMean(gains, 14)
  const avgLoss14 = rollingMean(losses, 14)
  const rsi14 = new Array<number>(n)
  for (let i = 0; i < n; i++) {
    const rs = avgGain14[i] / (avgLoss14[i] + EPS)
    rsi14[i] = 100 - 100 / (1 + rs)
  }

  // RSI(2) — Connors
  const avgGain2 = rollingMean(gains, 2)
  const avgLoss2 = rollingMean(losses, 2)
  const rsi2 = new Array<number>(n)
  for (let i = 0; i < n; i++) {
    const rs = avgGain2[i] / (avgLoss2[i] + EPS)
    rsi2[i] = 100 - 100 / (1 + rs)
  }

  // ── Bollinger Bands ─────────────────────────────────────────────────────────
  // Python: std20[i] = np.std(close5[max(0,i-19):i+1], ddof=1) if i>=19 else 0
  //       = sample stddev over last 20 closes (inclusive of i)
  const std20 = new Array<number>(n).fill(0)
  for (let i = 19; i < n; i++) {
    let mean = 0
    for (let j = i - 19; j <= i; j++) mean += close[j]
    mean /= 20
    let sq = 0
    for (let j = i - 19; j <= i; j++) {
      const d = close[j] - mean
      sq += d * d
    }
    // ddof=1 → divide by (n-1)=19
    std20[i] = Math.sqrt(sq / 19)
  }
  const bbUpper = new Array<number>(n)
  const bbLower = new Array<number>(n)
  const bbWidth = new Array<number>(n)
  const bbPctB = new Array<number>(n)
  for (let i = 0; i < n; i++) {
    bbUpper[i] = sma20[i] + 2 * std20[i]
    bbLower[i] = sma20[i] - 2 * std20[i]
    bbWidth[i] = (4 * std20[i]) / (sma20[i] + EPS)
    bbPctB[i] = (close[i] - bbLower[i]) / (4 * std20[i] + EPS)
  }
  const price_bb_upper = close.map((v, i) => v / (bbUpper[i] + EPS))
  const price_bb_lower = close.map((v, i) => v / (bbLower[i] + EPS))

  // ── MACD ─────────────────────────────────────────────────────────────────────
  const ema12 = ema(close, 12)
  const ema26 = ema(close, 26)
  const macdLine = new Array<number>(n)
  for (let i = 0; i < n; i++) macdLine[i] = ema12[i] - ema26[i]
  const macdSignal = ema(macdLine, 9)
  const macdHist = new Array<number>(n)
  for (let i = 0; i < n; i++) macdHist[i] = macdLine[i] - macdSignal[i]
  // Normalize by close (matches ml_features.py)
  const macd_line = macdLine.map((v, i) => v / (close[i] + EPS))
  const macd_signal = macdSignal.map((v, i) => v / (close[i] + EPS))

  // ── ATR (SIMPLE rolling mean of TR, NOT Wilder) ──────────────────────────────
  // Python: tr = max(h-l, |h-prev_close|, |l-prev_close|) with np.roll(close, 1).
  // np.roll wraps (last→first), but first bar is in warmup. We use causal
  // prev_close (close[i-1] for i>0, close[0] for i=0) which matches Python
  // for i>=1 exactly.
  const tr = new Array<number>(n)
  for (let i = 0; i < n; i++) {
    const prevClose = i === 0 ? close[0] : close[i - 1]
    const a = high[i] - low[i]
    const b = Math.abs(high[i] - prevClose)
    const c = Math.abs(low[i] - prevClose)
    tr[i] = Math.max(a, Math.max(b, c))
  }
  const atr14 = rollingMean(tr, 14)
  const atr_pct = atr14.map((v, i) => v / (close[i] + EPS))

  // ── Volume ratio ─────────────────────────────────────────────────────────────
  const volAvg20 = rollingMean(vol, 20)
  const vol_ratio = vol.map((v, i) => v / (volAvg20[i] + EPS))

  // ── OBV slope (10-bar window, NOT 5) ─────────────────────────────────────────
  const obv = new Array<number>(n)
  obv[0] = 0
  for (let i = 1; i < n; i++) {
    obv[i] = obv[i - 1] + (deltas[i] > 0 ? vol[i] : -vol[i])
  }
  const obv_slope = new Array<number>(n).fill(0)
  for (let i = 10; i < n; i++) {
    obv_slope[i] = (obv[i] - obv[i - 10]) / (obv[i - 10] + EPS)
  }

  // ── Stochastic %K (14-bar high/low, INCLUSIVE of current bar) ────────────────
  // Python: hh14[i] = max(high[max(0,i-13):i+1]); ll14[i] = min(low[...])
  const stoch_k = new Array<number>(n)
  for (let i = 0; i < n; i++) {
    const start = Math.max(0, i - 13)
    let hh = -Infinity
    let ll = Infinity
    for (let j = start; j <= i; j++) {
      if (high[j] > hh) hh = high[j]
      if (low[j] < ll) ll = low[j]
    }
    stoch_k[i] = ((close[i] - ll) / (hh - ll + EPS)) * 100
  }

  // ── ADX (SIMPLIFIED — NOT Wilder) ───────────────────────────────────────────
  // Python:
  //   up_moves   = where(deltas > 0, 1, 0)
  //   down_moves = where(deltas < 0, 1, 0)
  //   adx = abs(rolling_mean(up_moves, 14) - rolling_mean(down_moves, 14)) * 100
  const up_moves = new Array<number>(n)
  const down_moves = new Array<number>(n)
  for (let i = 0; i < n; i++) {
    up_moves[i] = deltas[i] > 0 ? 1 : 0
    down_moves[i] = deltas[i] < 0 ? 1 : 0
  }
  const rm_up = rollingMean(up_moves, 14)
  const rm_dn = rollingMean(down_moves, 14)
  const adx = new Array<number>(n)
  for (let i = 0; i < n; i++) {
    adx[i] = Math.abs(rm_up[i] - rm_dn[i]) * 100
  }

  // ── Higher timeframe context (approximations from 5min candles) ──────────────
  // ml_features.py uses real 1hour and 1day candle series from the multi-TF
  // data pipeline.  The trader server only has 5min candles, so we approximate.
  // This matches the existing v2 meta_selector.ts approach (tested & deployed).
  const ret1h = idx >= 12 ? (close[idx] - close[idx - 12]) / (close[idx - 12] + EPS) : 0
  const ret1d = idx >= 144 ? (close[idx] - close[idx - 144]) / (close[idx - 144] + EPS) : 0
  // 1h_trend: close / sma10(close1h)  ≈ close / sma120(close5min)  [10 hours]
  const trend1h = close[idx] / (sma120[idx] + EPS)
  // 1d_trend: close / sma5(close1d)   ≈ close / sma144(close5min)  [~1 trading day]
  const trend1d = close[idx] / (sma144[idx] + EPS)
  // 1h_rsi: approximation — use 5min RSI14 (no real 1hour close series available)
  const rsi1h = rsi14[idx]

  // ── Time features (MSK timezone, UTC+3) ──────────────────────────────────────
  // ml_features.py:
  //   ts_seconds = time5 / 1000                    # time5 is in ms
  //   hours = (ts_seconds // 3600 + 3) % 24        # MSK = UTC+3
  //   dow   = (ts_seconds // 86400 + 4) % 7        # Sun=0...Sat=6 (matches JS getUTCDay)
  //   hour         = hours / 24.0
  //   day_of_week  = dow / 7.0
  //
  // regime_detector.ts Candle.time is "unix seconds (or any monotonic unit)".
  // The trader server's Candle.time is in MILLISECONDS (existing meta_selector.ts
  // does `new Date(ts)` directly, which expects ms).  We auto-detect: if ts > 1e12
  // (year 2001+), treat as ms; else seconds.
  const ts = time[idx]
  const tsMs = ts > 1e12 ? ts : ts * 1000
  const date = new Date(tsMs)
  const hour = ((date.getUTCHours() + 3) % 24) / 24.0
  const day_of_week = date.getUTCDay() / 7.0

  // ── Assemble feature vector in ALPHABETICAL order ────────────────────────────
  // (matches training metadata feature_names exactly)
  const features: Record<string, number> = {
    '1d_ret': ret1d,
    '1d_trend': trend1d,
    '1h_ret': ret1h,
    '1h_rsi': rsi1h,
    '1h_trend': trend1h,
    'adx': adx[idx],
    'atr_pct': atr_pct[idx],
    'bb_pct_b': bbPctB[idx],
    'bb_width': bbWidth[idx],
    'day_of_week': day_of_week,
    'hour': hour,
    'macd_hist': macdHist[idx],
    'macd_line': macd_line[idx],
    'macd_signal': macd_signal[idx],
    'obv_slope': obv_slope[idx],
    'price_bb_lower': price_bb_lower[idx],
    'price_bb_upper': price_bb_upper[idx],
    'price_sma20': price_sma20[idx],
    'price_sma50': price_sma50[idx],
    'ret_1': ret1[idx],
    'ret_10': ret10[idx],
    'ret_30': ret30[idx],
    'ret_5': ret5[idx],
    'ret_5_log': ret5log[idx],
    'rsi14': rsi14[idx],
    'rsi2': rsi2[idx],
    'sma14_sma20': sma14_sma20[idx],
    'sma20_sma50': sma20_sma50[idx],
    'sma5_sma14': sma5_sma14[idx],
    'stoch_k': stoch_k[idx],
    'vol_ratio': vol_ratio[idx],
  }

  // Build the vector in the exact order specified by the task / training metadata.
  const FEATURE_NAMES = [
    '1d_ret', '1d_trend', '1h_ret', '1h_rsi', '1h_trend', 'adx', 'atr_pct',
    'bb_pct_b', 'bb_width', 'day_of_week', 'hour', 'macd_hist', 'macd_line',
    'macd_signal', 'obv_slope', 'price_bb_lower', 'price_bb_upper',
    'price_sma20', 'price_sma50', 'ret_1', 'ret_10', 'ret_30', 'ret_5',
    'ret_5_log', 'rsi14', 'rsi2', 'sma14_sma20', 'sma20_sma50', 'sma5_sma14',
    'stoch_k', 'vol_ratio',
  ]
  const out = new Array<number>(FEATURE_NAMES.length)
  for (let i = 0; i < FEATURE_NAMES.length; i++) {
    let v = features[FEATURE_NAMES[i]]
    if (!Number.isFinite(v)) v = 0
    // Python: np.clip(X, -10, 10)
    if (v > 10) v = 10
    else if (v < -10) v = -10
    out[i] = v
  }
  return out
}

// ─── Model loader (with path search + caching) ───────────────────────────────

/**
 * Resolve the absolute path of a regime model file.
 *
 * Tries, in order:
 *   /opt/ai-trader/data/regime_<name>.json
 *   /opt/ai-trader/src/strategies/regime_<name>.json
 *   <__dirname>/regime_<name>.json
 *   <cwd>/regime_<name>.json
 *
 * Returns the first existing path, or null if none found.
 */
function resolveModelPath(modelFile: string): string | null {
  for (const dir of MODEL_SEARCH_PATHS) {
    const p = path.join(dir, modelFile)
    try {
      if (fs.existsSync(p)) return p
    } catch {
      // ignore — try next
    }
  }
  return null
}

/** Per-regime model cache: regimeIndex → loaded model (loaded lazily on first hit). */
const _modelByRegime = new Map<number, ReturnType<typeof loadModel> | null>()

/**
 * Get the (cached) ML model for a regime.
 *
 * Returns null if:
 *   - The regime has no model file (OVERSOLD_BOUNCE, OVERBOUGHT_REVERSAL)
 *   - The model file cannot be found in any search path
 *
 * On first call for a regime, the model is loaded from disk and cached.
 * Subsequent calls return the cached instance without disk I/O.
 */
function getRegimeModel(regimeIdx: number): ReturnType<typeof loadModel> | null {
  if (_modelByRegime.has(regimeIdx)) {
    return _modelByRegime.get(regimeIdx) ?? null
  }
  const modelFile = REGIME_MODEL_FILES[regimeIdx]
  if (!modelFile) {
    _modelByRegime.set(regimeIdx, null)
    return null
  }
  const modelPath = resolveModelPath(modelFile)
  if (!modelPath) {
    console.warn(
      `[MetaSelectorV4] WARN: model file not found: ${modelFile} ` +
        `(searched: ${MODEL_SEARCH_PATHS.join(', ')})`,
    )
    _modelByRegime.set(regimeIdx, null)
    return null
  }
  try {
    const model = loadModel(modelPath)
    _modelByRegime.set(regimeIdx, model)
    console.log(
      `[MetaSelectorV4] loaded ${modelFile}: nTrees=${model.nTrees} nFeatures=${model.nFeatures}`,
    )
    return model
  } catch (e: any) {
    console.error(`[MetaSelectorV4] ERROR loading ${modelFile}: ${e.message}`)
    _modelByRegime.set(regimeIdx, null)
    return null
  }
}

/** Clear all cached models (useful for hot-reload / testing). */
export function clearMetaV4ModelCache(): void {
  _modelByRegime.clear()
}

// ─── Strategy class ────────────────────────────────────────────────────────────

/**
 * MetaSelectorV4Strategy — the v4 ML orchestrator.
 *
 *   detectRegime → loadModel → predict_proba → LONG / SHORT / FLAT
 *
 * Fallback for regimes with no ML model (skipped during training due to
 * insufficient samples):
 *   OVERSOLD_BOUNCE     → LONG  (buy the dip — RSI<25 + close>SMA5)
 *   OVERBOUGHT_REVERSAL → SHORT (sell the top — RSI>75 + close<SMA5)
 */
export class MetaSelectorV4Strategy implements IStrategy {
  name = 'meta_selector_v4'
  description =
    'ML v4 orchestrator: detectRegime → per-regime binary XGBoost (P(up)) → LONG/SHORT/FLAT'

  /** Last prediction details — exposed for logging/monitoring. */
  lastRegimeIdx: number = -1
  lastRegimeName: string = ''
  lastProba: number = 0.5
  lastDecision: number = 0
  lastDecisionLabel: string = 'FLAT'
  lastModelUsed: string = ''
  lastError: string = ''

  predict(
    candles: Candle[],
    idx: number,
    _hasPosition: boolean,
    _stepsHeld?: number,
    _ctx?: StrategyContext,
  ): number {
    // Reset transient state
    this.lastError = ''

    // Need enough candle history.  detectRegime requires >100 bars; features
    // need >50; we require 200 for the higher-TF approximations (144-bar SMA).
    if (!candles || candles.length < MIN_CANDLES) {
      this.lastError = `insufficient_history (need >= ${MIN_CANDLES} candles, got ${candles?.length ?? 0})`
      // During warmup, stay flat.
      this.lastDecision = 0
      this.lastDecisionLabel = 'FLAT'
      return 0
    }

    // 1. Detect regime at the latest bar
    const regime = detectRegime(candles)
    this.lastRegimeIdx = regime.regime
    this.lastRegimeName = regime.regimeName

    // 2. If regime has no ML model → use rule-based fallback
    const model = getRegimeModel(regime.regime)
    if (!model) {
      let decision: number
      let label: string
      if (regime.regime === 7 /* OVERSOLD_BOUNCE */) {
        decision = 1 // LONG — buy the dip
        label = 'LONG'
      } else if (regime.regime === 8 /* OVERBOUGHT_REVERSAL */) {
        decision = 2 // SHORT — sell the top
        label = 'SHORT'
      } else {
        // Model file not found for a regime that should have one.  Stay flat
        // and log loudly so the operator notices.
        decision = 0
        label = 'FLAT'
        this.lastError = `model_missing_for_${regime.regimeName}`
      }
      this.lastProba = 0.5
      this.lastDecision = decision
      this.lastDecisionLabel = label
      this.lastModelUsed = '(fallback)'
      console.log(
        `[MetaSelectorV4] regime=${regime.regimeName} (idx=${regime.regime}) ` +
          `P(up)=N/A → ${label} [FALLBACK]`,
      )
      return decision
    }

    // 3. Has ML model → compute features + predict
    this.lastModelUsed = regime.regimeName
    let features: number[]
    try {
      features = computeMetaV4Features(candles)
    } catch (e: any) {
      this.lastError = `feature_error: ${e.message}`
      this.lastDecision = 0
      this.lastDecisionLabel = 'FLAT'
      console.error(`[MetaSelectorV4] feature computation failed: ${e.message}`)
      return 0
    }

    let proba: number
    try {
      proba = predict_proba(model, features)
    } catch (e: any) {
      this.lastError = `predict_error: ${e.message}`
      this.lastDecision = 0
      this.lastDecisionLabel = 'FLAT'
      console.error(`[MetaSelectorV4] predict_proba failed: ${e.message}`)
      return 0
    }

    // Guard against NaN/Inf
    if (!Number.isFinite(proba)) {
      proba = 0.5
    }
    this.lastProba = proba

    // 4. Decision thresholds (from v4 training metadata)
    let decision: number
    let label: string
    if (proba > LONG_THRESHOLD) {
      decision = 1 // LONG
      label = 'LONG'
    } else if (proba < SHORT_THRESHOLD) {
      decision = 2 // SHORT
      label = 'SHORT'
    } else {
      decision = 0 // FLAT
      label = 'FLAT'
    }
    this.lastDecision = decision
    this.lastDecisionLabel = label

    // 5. Log every prediction
    console.log(
      `[MetaSelectorV4] regime=${regime.regimeName} (idx=${regime.regime}) ` +
        `P(up)=${proba.toFixed(4)} → ${label}`,
    )

    return decision
  }
}

/** Factory — matches the trader server's strategy-instantiation pattern. */
export function createMetaSelectorV4Strategy(): MetaSelectorV4Strategy {
  return new MetaSelectorV4Strategy()
}

// ─── Self-test (run with: npx tsx meta_selector_v4.ts) ─────────────────────────
//
// Generates 250 synthetic 5min candles in a clear uptrend and prints:
//   - detected regime
//   - 31-feature vector (spot-check a few values)
//   - P(up) from each model that loads
//   - final decision
//
// Useful for verifying the wiring end-to-end without live market data.

// Detect "run as a script" (Node/tsx compatible).
// NOTE: We intentionally do NOT use `import.meta.main` (Bun-only) here so the
// file compiles cleanly under `tsc --moduleResolution node` (CommonJS mode).
// When run with `npx tsx meta_selector_v4.ts`, this branch fires.
const isMain: boolean =
  typeof require !== 'undefined' &&
  typeof require.main !== 'undefined' &&
  require.main === module

if (isMain) {
  console.log('=== MetaSelectorV4 — self-test ===\n')

  // Build 250 synthetic 5min candles: gentle uptrend with noise.
  const N = 250
  const candles: Candle[] = []
  let price = 100
  const t0 = Math.floor(Date.now() / 1000) // seconds
  for (let i = 0; i < N; i++) {
    // Strong upward drift on the last 80 bars → should trigger trend/breakout.
    const drift = i > 170 ? 0.4 : 0.05
    const noise = (Math.random() - 0.5) * 0.5
    const open = price
    const close = price + drift + noise
    const high = Math.max(open, close) + Math.random() * 0.25
    const low = Math.min(open, close) - Math.random() * 0.25
    const volume = 5000 + Math.random() * 2000
    candles.push({
      time: (t0 + i * 300) * 1000, // ms (trader-server convention)
      open,
      high,
      low,
      close,
      volume,
    })
    price = close
  }
  console.log(`Generated ${N} synthetic candles.`)

  // Detect regime
  const reg = detectRegime(candles)
  console.log(
    `\nDetectRegime:\n  regime=${reg.regime} (${reg.regimeName})\n  confidence=${reg.confidence.toFixed(3)}\n  adx=${reg.adx.toFixed(2)}  rsi=${reg.rsi.toFixed(2)}`,
  )

  // Compute features
  const feats = computeMetaV4Features(candles)
  console.log(
    `\nFeatures (n=${feats.length}): [${feats.slice(0, 8).map(v => v.toFixed(4)).join(', ')}, ...]`,
  )
  const FEATURE_NAMES = [
    '1d_ret', '1d_trend', '1h_ret', '1h_rsi', '1h_trend', 'adx', 'atr_pct',
    'bb_pct_b', 'bb_width', 'day_of_week', 'hour', 'macd_hist', 'macd_line',
    'macd_signal', 'obv_slope', 'price_bb_lower', 'price_bb_upper',
    'price_sma20', 'price_sma50', 'ret_1', 'ret_10', 'ret_30', 'ret_5',
    'ret_5_log', 'rsi14', 'rsi2', 'sma14_sma20', 'sma20_sma50', 'sma5_sma14',
    'stoch_k', 'vol_ratio',
  ]
  console.log('  All features (name=value):')
  for (let i = 0; i < FEATURE_NAMES.length; i++) {
    console.log(`    ${FEATURE_NAMES[i].padEnd(16)} = ${feats[i].toFixed(6)}`)
  }

  // Try to load a model for this regime
  console.log(`\nLoading model for regime ${reg.regime} (${reg.regimeName})...`)
  const model = getRegimeModel(reg.regime)
  if (model) {
    const p = predict_proba(model, feats)
    console.log(`  predict_proba → P(up) = ${p.toFixed(6)}`)
    console.log(
      `  decision: ${p > LONG_THRESHOLD ? 'LONG' : p < SHORT_THRESHOLD ? 'SHORT' : 'FLAT'}`,
    )
  } else {
    console.log('  (no model — fallback rule would apply)')
  }

  // Run the strategy end-to-end
  console.log('\nRunning MetaSelectorV4Strategy.predict()...')
  const strat = new MetaSelectorV4Strategy()
  const decision = strat.predict(candles, N - 1, false, 0, {})
  console.log(
    `\n  FINAL: regime=${strat.lastRegimeName} P(up)=${strat.lastProba.toFixed(4)} → ${strat.lastDecisionLabel} (code=${decision})`,
  )
  if (strat.lastError) {
    console.log(`  ERROR: ${strat.lastError}`)
  }

  // List all regime model files
  console.log('\nRegime model availability:')
  for (let r = 0; r < 12; r++) {
    const m = getRegimeModel(r)
    console.log(
      `  regime ${r} (${REGIME_MODEL_FILES[r] ?? 'NO MODEL'}): ${m ? `loaded (${m.nTrees} trees)` : 'NOT AVAILABLE'}`,
    )
  }

  console.log('\n=== self-test complete ===')
}
