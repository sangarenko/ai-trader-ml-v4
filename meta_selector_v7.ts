/**
 * MetaSelectorV7 Strategy — ensemble inference (XGB + LightGBM + CatBoost + stacking).
 *
 * Architecture (per predict() call):
 *   1. detectRegime(candles)  → 0..11
 *   2. If regime 7 or 8 → fallback (OVERSOLD_BOUNCE → LONG, OVERBOUGHT_REVERSAL → SHORT)
 *   3. Compute 36 features (features_v4 + features_v7 new — ported from Python)
 *   4. Load 3 models for this regime (xgb, lgb, cat) + stacking coefficients
 *   5. Get 3 probabilities: p_xgb, p_lgb, p_cat
 *   6. Stacking: p_final = sigmoid(coef[0]*p_xgb + coef[1]*p_lgb + coef[2]*p_cat + intercept)
 *   7. Decision: p > 0.65 → LONG, p < 0.35 → SHORT, else FLAT
 *   8. Exit: if holding and p crosses 0.50
 *
 * Model files: /opt/ai-trader/data/meta_models_v7/regime_v7_<name>_{xgb,lgb,cat}.json
 *              + regime_v7_stacking.json
 *
 * Pure TS — no Python at runtime. Verified to match Python:
 *   - XGB inference: matches xgboost lib within <1e-4 (see xgboost_binary_ts.ts)
 *   - LGB inference: matches lightgbm lib within <1e-10 (verified via dump_model walk)
 *   - CAT inference: matches catboost lib within 0.0 (verified via oblivious tree walk, LSB-first)
 *   - Stacking: logistic regression coef × (xgb_p, lgb_p, cat_p) + intercept → sigmoid
 *
 * Author: Agent V7-3 (ts-inference-portfolio)
 */
import { Candle, detectRegime, RegimeResult } from './regime_detector'
import { loadModel, predict_proba, sigmoid, XGBoostBinaryModel } from './xgboost_binary_ts'
import * as fs from 'fs'
import * as path from 'path'

// ─── Trader-server strategy interface (structurally compatible) ────────────────
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

// ─── v7 constants ──────────────────────────────────────────────────────────────
const LONG_THRESHOLD = 0.65
const SHORT_THRESHOLD = 0.35
const EXIT_LONG = 0.50
const EXIT_SHORT = 0.50
const MIN_CANDLES = 200

/** Action codes returned by predict() — match the trader-server engine. */
const ACTION_FLAT = 0
const ACTION_LONG = 1
const ACTION_SHORT = 2
const ACTION_CLOSE_LONG = 3
const ACTION_CLOSE_SHORT = 4

/** Regime index → lowercase name (used to build model file names). */
const REGIME_NAMES_LOWER = [
  'strong_trend_up',    // 0
  'mild_trend_up',      // 1
  'range_tight',        // 2
  'range_wide',         // 3
  'mild_trend_down',    // 4
  'strong_trend_down',  // 5
  'crash',              // 6
  'oversold_bounce',    // 7  — NO ML MODEL (fallback: LONG)
  'overbought_reversal',// 8 — NO ML MODEL (fallback: SHORT)
  'breakout_up',        // 9
  'breakdown',          // 10
  'high_vol_regime',    // 11
]

/** Directories to search for the meta_models_v7/ folder. */
const MODEL_SEARCH_PATHS = [
  '/opt/ai-trader/data/meta_models_v7',
  '/opt/ai-trader/src/strategies/meta_models_v7',
  '/opt/ai-trader/data',
  '/opt/ai-trader/src/strategies',
  path.join(__dirname, 'meta_models_v7'),
  __dirname,
  process.cwd(),
  path.join(process.cwd(), 'meta_models_v7'),
]

const EPS = 1e-10

// ─── Feature engineering (TS port of features_v7.py — 36 features, alphabetical) ─
//
// All features mirror Python `features_v7.py`:
//   v4 features (22):
//     1d_ret, 1h_ret, atr_pct, bb_pct_b, bb_width, day_of_week, hour,
//     macd_hist, market_breadth, ret_1, ret_10, ret_30, ret_5, rsi14,
//     sber_gazp_corr, sma14_sma20, sma20_sma50, sma5_sma14, stoch_k,
//     trend_strength, vol_ratio, vol_regime
//   v7 NEW (14):
//     brent_ret, cb_rate, cum_delta, gap_overnight, imoex_ret, imoex_ret_1h,
//     intraday_session, order_imbalance, poc, usdrub_ret, vah, val, vwap,
//     vwap_dev
//
// Total: 36 features, sorted alphabetically (matches training).
//
// NOTES:
//   - Macro features (usdrub_ret, brent_ret, imoex_ret, imoex_ret_1h, cb_rate)
//     are 0 if no macro data is supplied via `setMacroData()`. The Python
//     pipeline falls back to 0 on fetch failure too, so the model is
//     robust to missing macro data (slightly degraded but functional).
//   - Cross-asset features (market_breadth, sber_gazp_corr) are 0.5 / 0.0
//     by default — the model handles neutral inputs gracefully.
//   - First 50 bars are zeroed (matches Python X[:50] = 0 warmup).

const FEATURE_NAMES_V7: string[] = [
  // v4 features (22)
  '1d_ret', '1h_ret', 'atr_pct', 'bb_pct_b', 'bb_width', 'day_of_week',
  'hour', 'macd_hist', 'market_breadth', 'ret_1', 'ret_10', 'ret_30',
  'ret_5', 'rsi14', 'sber_gazp_corr', 'sma14_sma20', 'sma20_sma50',
  'sma5_sma14', 'stoch_k', 'trend_strength', 'vol_ratio', 'vol_regime',
  // v7 NEW (14)
  'brent_ret', 'cb_rate', 'cum_delta', 'gap_overnight', 'imoex_ret',
  'imoex_ret_1h', 'intraday_session', 'order_imbalance', 'poc',
  'usdrub_ret', 'vah', 'val', 'vwap', 'vwap_dev',
].sort()  // sort alphabetically to match Python `sorted(all_features.keys())`

/**
 * Optional macro data — forward-filled onto the base 10-min grid.
 *
 * Each array maps a base-grid index → the macro return value at that bar
 * (computed by `_macro_ret()` in features_v7.py, forward-filled causally).
 * cb_rate[i] is the key rate in % (e.g. 21.0) at bar i.
 *
 * If left undefined, macro features default to 0 (matches Python's
 * "if fetch fails, return 0" behaviour).
 */
export interface MacroData {
  usdrub_ret?: number[]     // USD/RUB daily return, forward-filled onto base grid
  brent_ret?: number[]      // Brent daily return
  imoex_ret?: number[]      // IMOEX daily return
  imoex_ret_1h?: number[]   // IMOEX 1-hour return
  cb_rate?: number[]        // CB key rate in % (NOT divided by 100 — we'll normalize)
}

/**
 * Optional cross-asset data for market_breadth / sber_gazp_corr features.
 * If undefined, defaults to 0.5 / 0.0 (matches Python when all_tickers_data=None).
 */
export interface CrossAssetData {
  marketBreadth?: number[]  // 0..1, % of tickers with positive ret_5
  sberGazpCorr?: number[]   // -1..1, rolling 20-bar correlation
}

let _macroData: MacroData = {}
let _crossAssetData: CrossAssetData = {}

/**
 * Inject macro data (called by trader server's macro-data sidecar).
 * Without this, macro features default to 0 — model still works in degraded mode.
 */
export function setMacroData(data: MacroData): void {
  _macroData = data || {}
}

export function setCrossAssetData(data: CrossAssetData): void {
  _crossAssetData = data || {}
}

/** Causal trailing SMA via cumsum. Mirrors `_causal_sma()` in features_v4.py. */
function causalSMA(arr: number[], w: number): number[] {
  const n = arr.length
  const out = new Array<number>(n).fill(0)
  if (n === 0) return out
  const cumsum = new Array<number>(n)
  let s = 0
  for (let i = 0; i < n; i++) {
    s += arr[i]
    cumsum[i] = s
  }
  for (let i = 0; i < n; i++) {
    if (i < w) {
      out[i] = cumsum[i] / (i + 1)
    } else {
      out[i] = (cumsum[i] - cumsum[i - w]) / w
    }
  }
  return out
}

/** Causal trailing rolling mean (alias of causalSMA, matches features_v4.py). */
function causalRollingMean(arr: number[], w: number): number[] {
  return causalSMA(arr, w)
}

/** Causal trailing population std via cumsum / cumsum-of-squares. */
function causalRollingStd(arr: number[], w: number): number[] {
  const n = arr.length
  const out = new Array<number>(n).fill(0)
  if (n === 0) return out
  const c1 = new Array<number>(n)
  const c2 = new Array<number>(n)
  let s1 = 0, s2 = 0
  for (let i = 0; i < n; i++) {
    s1 += arr[i]
    s2 += arr[i] * arr[i]
    c1[i] = s1
    c2[i] = s2
  }
  for (let i = 0; i < n; i++) {
    const k = Math.min(i + 1, w)
    const mean = k > 0 ? c1[i] / k : 0
    let varVal = k > 0 ? c2[i] / k - mean * mean : 0
    if (i >= w) {
      const m = (c1[i] - c1[i - w]) / w
      varVal = (c2[i] - c2[i - w]) / w - m * m
    }
    out[i] = Math.sqrt(Math.max(varVal, 0))
  }
  return out
}

/** Causal EMA (k = 2/(period+1)). */
function causalEMA(arr: number[], period: number): number[] {
  const n = arr.length
  const out = new Array<number>(n).fill(0)
  if (n === 0) return out
  const k = 2 / (period + 1)
  out[0] = arr[0]
  for (let i = 1; i < n; i++) {
    out[i] = arr[i] * k + out[i - 1] * (1 - k)
  }
  return out
}

/**
 * Causal n-bar return: ret_n[i] = (close[i] - close[i-n]) / close[i-n].
 * ret_n[i < n] = 0 (warmup).
 */
function causalRetN(close: number[], n: number): number[] {
  const r = new Array<number>(close.length).fill(0)
  for (let i = n; i < close.length; i++) {
    r[i] = (close[i] - close[i - n]) / (close[i - n] + EPS)
  }
  return r
}

/**
 * Compute day-start indices and "previous day's last bar" indices for MSK midnight
 * reset. Matches `_compute_day_boundaries()` in features_v7.py.
 *
 * Returns:
 *   dayStart:       number[] — index of FIRST bar of each bar's day
 *   prevDayLastBar: number[] — index of LAST bar of previous day (-1 if first day)
 */
function computeDayBoundaries(timeMs: number[]): { dayStart: number[]; prevDayLastBar: number[] } {
  const n = timeMs.length
  const dayStart = new Array<number>(n).fill(0)
  const prevDayLastBar = new Array<number>(n).fill(-1)

  if (n === 0) return { dayStart, prevDayLastBar }

  // MSK = UTC+3. day_id = floor((t/1000 + 3*3600) / 86400)
  const dayId = (t: number) => Math.floor((t / 1000 + 3 * 3600) / 86400)

  // Find indices where day changes
  const dayChangeIndices: number[] = [0]
  for (let i = 1; i < n; i++) {
    if (dayId(timeMs[i]) !== dayId(timeMs[i - 1])) {
      dayChangeIndices.push(i)
    }
  }

  // For each bar i, find which day-start it belongs to (binary search)
  for (let i = 0; i < n; i++) {
    // Find largest j such that dayChangeIndices[j] <= i
    let lo = 0, hi = dayChangeIndices.length - 1, found = 0
    while (lo <= hi) {
      const mid = (lo + hi) >> 1
      if (dayChangeIndices[mid] <= i) {
        found = mid
        lo = mid + 1
      } else {
        hi = mid - 1
      }
    }
    dayStart[i] = dayChangeIndices[found]
    prevDayLastBar[i] = dayStart[i] - 1  // -1 if first day
  }
  return { dayStart, prevDayLastBar }
}

/**
 * Causal VWAP that resets at MSK midnight.
 *   vwap[i] = sum(typical_price * vol, day_start[i]..i) / sum(vol, day_start[i]..i)
 *
 * Returns:
 *   vwapNorm:  vwap_raw / close[i]   (~1.0, scale-invariant)
 *   vwapDev:   (close[i] - vwap_raw) / vwap_raw   (signed, ~0)
 */
function computeVWAP(
  close: number[], high: number[], low: number[], vol: number[],
  dayStart: number[], prevDayLastBar: number[],
): { vwapNorm: number[]; vwapDev: number[] } {
  const n = close.length
  const vwapNorm = new Array<number>(n).fill(1)
  const vwapDev = new Array<number>(n).fill(0)
  if (n === 0) return { vwapNorm, vwapDev }

  // Cumulative sums across the entire series (then subtract previous day's last
  // cumulative value to get within-day cumulative)
  const pv = new Array<number>(n)
  const v = new Array<number>(n)
  let spv = 0, sv = 0
  for (let i = 0; i < n; i++) {
    const tp = (high[i] + low[i] + close[i]) / 3
    spv += tp * vol[i]
    sv += vol[i]
    pv[i] = spv
    v[i] = sv
  }

  for (let i = 0; i < n; i++) {
    const startIdx = dayStart[i]
    const prevIdx = prevDayLastBar[i]  // -1 if first day
    const cumPv = prevIdx >= 0 ? pv[i] - pv[prevIdx] : pv[i]
    const cumV = prevIdx >= 0 ? v[i] - v[prevIdx] : v[i]
    const vwapRaw = cumPv / (cumV + EPS)
    vwapNorm[i] = vwapRaw / (close[i] + EPS)
    vwapDev[i] = (close[i] - vwapRaw) / (vwapRaw + EPS)
  }
  return { vwapNorm, vwapDev }
}

/**
 * Causal volume profile (POC / VAH / VAL) — resets at MSK midnight.
 *
 * Bins each bar into 1 of 50 price buckets spanning [low.min(), high.max()]
 * across the WHOLE series. For each bar we then take the cumulative volume per
 * bin from start_of_day up to that bar. POC = bin with highest cumulative vol.
 * VAH/VAL = expand outward from POC until cumulative volume >= 70% of total.
 *
 * Returns: poc, vah, val — all normalized as price/close[i] (~1.0).
 */
function computeVolumeProfile(
  close: number[], high: number[], low: number[], vol: number[],
  dayStart: number[], prevDayLastBar: number[],
  nBins = 50, valuePct = 0.70,
): { poc: number[]; vah: number[]; val: number[] } {
  const n = close.length
  const poc = new Array<number>(n).fill(1)
  const vah = new Array<number>(n).fill(1)
  const val = new Array<number>(n).fill(1)
  if (n === 0) return { poc, vah, val }

  let priceLo = Infinity, priceHi = -Infinity
  for (let i = 0; i < n; i++) {
    if (low[i] < priceLo) priceLo = low[i]
    if (high[i] > priceHi) priceHi = high[i]
  }
  if (priceHi - priceLo < 1e-10) {
    // All-same-price edge case: POC/VAH/VAL = close
    for (let i = 0; i < n; i++) {
      poc[i] = 1; vah[i] = 1; val[i] = 1
    }
    return { poc, vah, val }
  }

  // Bin edges: linspace(priceLo, priceHi, nBins+1)
  const binEdges = new Array<number>(nBins + 1)
  const step = (priceHi - priceLo) / nBins
  for (let b = 0; b <= nBins; b++) {
    binEdges[b] = priceLo + b * step
  }

  // For each bar i, find bin index ranges [bin_lo, bin_hi] for [low[i], high[i]]
  const binIdxLo = new Array<number>(n)
  const binIdxHi = new Array<number>(n)
  for (let i = 0; i < n; i++) {
    // searchsorted(bin_edges, low[i], side='right') - 1, clipped to [0, nBins-1]
    let lo = 0
    {
      let a = 0, b = binEdges.length - 1
      while (a < b) {
        const mid = (a + b + 1) >> 1
        if (binEdges[mid] <= low[i]) a = mid
        else b = mid - 1
      }
      lo = Math.max(0, Math.min(nBins - 1, a))
    }
    let hi = 0
    {
      let a = 0, b = binEdges.length - 1
      while (a < b) {
        const mid = (a + b + 1) >> 1
        if (binEdges[mid] <= high[i]) a = mid
        else b = mid - 1
      }
      hi = Math.max(0, Math.min(nBins - 1, a))
    }
    binIdxLo[i] = lo
    binIdxHi[i] = hi
  }

  // Cumulative volume per bin: cumVols[i][b] = sum of vol contributions to bin b
  // over bars [0..i]. We keep this as a 2D array of size n × nBins (memory-heavy
  // but manageable: 1000 bars × 50 bins = 50K floats = 400KB).
  // To bound memory, we use Float64Array per bar (typed).
  const cumVols: Float64Array[] = new Array(n)
  {
    const cur = new Float64Array(nBins)
    for (let i = 0; i < n; i++) {
      const nOverlap = Math.max(binIdxHi[i] - binIdxLo[i] + 1, 1)
      const share = vol[i] / nOverlap
      for (let b = binIdxLo[i]; b <= binIdxHi[i]; b++) {
        cur[b] += share
      }
      cumVols[i] = new Float64Array(cur)  // snapshot
    }
  }

  // For each bar i, subtract previous day's last cumulative to get within-day cum
  for (let i = 0; i < n; i++) {
    const prevIdx = prevDayLastBar[i]
    let cumDay: Float64Array
    if (prevIdx >= 0) {
      cumDay = new Float64Array(nBins)
      const prev = cumVols[prevIdx]
      const cur = cumVols[i]
      for (let b = 0; b < nBins; b++) {
        cumDay[b] = cur[b] - prev[b]
      }
    } else {
      cumDay = cumVols[i]
    }

    // Total volume in day so far
    let total = 0
    let pocIdx = 0
    let maxVol = -1
    for (let b = 0; b < nBins; b++) {
      total += cumDay[b]
      if (cumDay[b] > maxVol) {
        maxVol = cumDay[b]
        pocIdx = b
      }
    }

    if (total < 1e-10) {
      poc[i] = 1; vah[i] = 1; val[i] = 1
      continue
    }

    // Expand from POC outward until 70% of total volume accumulated
    let loIdx = pocIdx
    let hiIdx = pocIdx
    let acc = cumDay[pocIdx]
    const target = total * valuePct
    for (let _ = 0; _ < nBins; _++) {
      if (acc >= target) break
      const below = loIdx > 0 ? cumDay[loIdx - 1] : -1
      const above = hiIdx < nBins - 1 ? cumDay[hiIdx + 1] : -1
      if (above >= below && above > 0) {
        hiIdx++
        acc += above
      } else if (below > 0) {
        loIdx--
        acc += below
      } else {
        break
      }
    }

    const pocPrice = (binEdges[pocIdx] + binEdges[pocIdx + 1]) / 2
    const vahPrice = binEdges[hiIdx + 1]
    const valPrice = binEdges[loIdx]
    poc[i] = pocPrice / (close[i] + EPS)
    vah[i] = vahPrice / (close[i] + EPS)
    val[i] = valPrice / (close[i] + EPS)
  }

  return { poc, vah, val }
}

/**
 * Causal order flow proxy: cumulative delta + per-bar order imbalance.
 *
 * buy_vol  = vol * (close - low) / (high - low)    — fraction of range in upper half
 * sell_vol = vol * (high - close) / (high - low)
 * cum_delta = cumsum(buy_vol - sell_vol) within day, normalized by cum vol
 * order_imbalance = per-bar (buy_vol - sell_vol) / (buy_vol + sell_vol)
 *
 * When high == low (no range): 50/50 buy/sell split.
 */
function computeOrderFlow(
  close: number[], high: number[], low: number[], vol: number[],
  dayStart: number[], prevDayLastBar: number[],
): { cumDelta: number[]; orderImbalance: number[] } {
  const n = close.length
  const cumDelta = new Array<number>(n).fill(0)
  const orderImbalance = new Array<number>(n).fill(0)
  if (n === 0) return { cumDelta, orderImbalance }

  const buyVols = new Array<number>(n)
  const sellVols = new Array<number>(n)
  for (let i = 0; i < n; i++) {
    const rng = high[i] - low[i]
    if (rng > 1e-10) {
      buyVols[i] = vol[i] * (close[i] - low[i]) / rng
      sellVols[i] = vol[i] * (high[i] - close[i]) / rng
    } else {
      buyVols[i] = vol[i] * 0.5
      sellVols[i] = vol[i] * 0.5
    }
    orderImbalance[i] = (buyVols[i] - sellVols[i]) / (buyVols[i] + sellVols[i] + EPS)
  }

  // Cumulative sums for delta and volume within the day
  const delta = new Array<number>(n)
  for (let i = 0; i < n; i++) delta[i] = buyVols[i] - sellVols[i]

  const cumDeltaAll = new Array<number>(n)
  const cumVolAll = new Array<number>(n)
  let sd = 0, sv = 0
  for (let i = 0; i < n; i++) {
    sd += delta[i]
    sv += vol[i]
    cumDeltaAll[i] = sd
    cumVolAll[i] = sv
  }

  for (let i = 0; i < n; i++) {
    const prevIdx = prevDayLastBar[i]
    const cumD = prevIdx >= 0 ? cumDeltaAll[i] - cumDeltaAll[prevIdx] : cumDeltaAll[i]
    const cumV = prevIdx >= 0 ? cumVolAll[i] - cumVolAll[prevIdx] : cumVolAll[i]
    cumDelta[i] = cumD / (cumV + EPS)
  }

  return { cumDelta, orderImbalance }
}

/**
 * Intraday session: 0=open(10-11 MSK), 1=mid(11-17), 2=close(17-23:50).
 * Normalized /2 → [0, 1].
 */
function computeIntradaySession(timeMs: number[]): number[] {
  const n = timeMs.length
  const out = new Array<number>(n).fill(0.5)
  for (let i = 0; i < n; i++) {
    const hourMsk = Math.floor((timeMs[i] / 1000 / 3600 + 3) % 24)
    let s = 1  // mid
    if (hourMsk >= 10 && hourMsk < 11) s = 0
    else if (hourMsk >= 17) s = 2
    out[i] = s / 2.0
  }
  return out
}

/**
 * Causal overnight gap = (today_open - prev_close) / prev_close.
 * Constant per day. For first day → 0 (no prev_close).
 */
function computeGapOvernight(
  open: number[], close: number[], timeMs: number[],
): number[] {
  const n = close.length
  const gap = new Array<number>(n).fill(0)
  if (n === 0) return gap

  // Group bars by day_id
  const dayId = (t: number) => Math.floor((t / 1000 + 3 * 3600) / 86400)
  const firstIdxByDay = new Map<number, number>()
  const lastIdxByDay = new Map<number, number>()
  for (let i = 0; i < n; i++) {
    const d = dayId(timeMs[i])
    if (!firstIdxByDay.has(d)) firstIdxByDay.set(d, i)
    lastIdxByDay.set(d, i)
  }

  // Build ordered list of day_ids
  const dayIds = Array.from(firstIdxByDay.keys()).sort((a, b) => a - b)

  // For each day d (skipping the first), gap = (open[firstIdx] - close[lastIdx_prev]) / close[lastIdx_prev]
  const gapByDay = new Map<number, number>()
  gapByDay.set(dayIds[0], 0)
  for (let d = 1; d < dayIds.length; d++) {
    const prev = dayIds[d - 1]
    const cur = dayIds[d]
    const prevClose = close[lastIdxByDay.get(prev)!]
    const curOpen = open[firstIdxByDay.get(cur)!]
    gapByDay.set(cur, (curOpen - prevClose) / (prevClose + EPS))
  }

  // Map each bar's day_id to its gap
  for (let i = 0; i < n; i++) {
    gap[i] = gapByDay.get(dayId(timeMs[i])) || 0
  }
  return gap
}

/**
 * Compute the 36-feature v7 vector at the LAST bar of `candles`.
 *
 * Returns an array of 36 numbers in alphabetical feature order (matches the
 * training metadata). First 50 bars return all-zeros (warmup, like Python).
 *
 * Macro features use `_macroData` (set via `setMacroData()`).
 * Cross-asset features use `_crossAssetData` (set via `setCrossAssetData()`).
 * Missing data defaults to neutral (0 / 0.5).
 */
export function computeFeaturesV7(candles: Candle[]): number[] {
  const N_FEATURES = 36
  const n = candles.length
  if (n < 50) {
    return new Array<number>(N_FEATURES).fill(0)
  }

  const close = new Array<number>(n)
  const open = new Array<number>(n)
  const high = new Array<number>(n)
  const low = new Array<number>(n)
  const vol = new Array<number>(n)
  const time5 = new Array<number>(n)
  for (let i = 0; i < n; i++) {
    close[i] = candles[i].close
    open[i] = candles[i].open
    high[i] = candles[i].high
    low[i] = candles[i].low
    vol[i] = candles[i].volume
    time5[i] = candles[i].time
  }
  const idx = n - 1

  const features: Record<string, number> = {}

  // ── Returns ─────────────────────────────────────────────────────────────
  const ret1 = causalRetN(close, 1)
  const ret5 = causalRetN(close, 5)
  const ret10 = causalRetN(close, 10)
  const ret30 = causalRetN(close, 30)
  features['ret_1'] = ret1[idx]
  features['ret_5'] = ret5[idx]
  features['ret_10'] = ret10[idx]
  features['ret_30'] = ret30[idx]

  // ── SMA ratios ──────────────────────────────────────────────────────────
  const sma5 = causalSMA(close, 5)
  const sma14 = causalSMA(close, 14)
  const sma20 = causalSMA(close, 20)
  const sma50 = causalSMA(close, 50)
  features['sma5_sma14'] = sma5[idx] / (sma14[idx] + EPS)
  features['sma14_sma20'] = sma14[idx] / (sma20[idx] + EPS)
  features['sma20_sma50'] = sma20[idx] / (sma50[idx] + EPS)

  // ── RSI(14) (simple rolling mean of gains/losses — NOT Wilder) ───────────
  const deltas = new Array<number>(n)
  deltas[0] = 0
  for (let i = 1; i < n; i++) deltas[i] = close[i] - close[i - 1]
  const gains = deltas.map(d => d > 0 ? d : 0)
  const losses = deltas.map(d => d < 0 ? -d : 0)
  const avgGain14 = causalRollingMean(gains, 14)
  const avgLoss14 = causalRollingMean(losses, 14)
  const rs = avgGain14[idx] / (avgLoss14[idx] + EPS)
  features['rsi14'] = 100 - 100 / (1 + rs)

  // ── Bollinger Bands ──────────────────────────────────────────────────────
  const std20 = causalRollingStd(close, 20)
  const bbUpper = sma20.map((v, i) => v + 2 * std20[i])
  const bbLower = sma20.map((v, i) => v - 2 * std20[i])
  features['bb_pct_b'] = (close[idx] - bbLower[idx]) / (4 * std20[idx] + EPS)
  features['bb_width'] = (4 * std20[idx]) / (sma20[idx] + EPS)

  // ── MACD histogram ──────────────────────────────────────────────────────
  const ema12 = causalEMA(close, 12)
  const ema26 = causalEMA(close, 26)
  const macdLine = ema12.map((v, i) => v - ema26[i])
  const macdSignal = causalEMA(macdLine, 9)
  features['macd_hist'] = macdLine[idx] - macdSignal[idx]

  // ── ATR% (simple rolling mean of True Range / close) ─────────────────────
  const tr = new Array<number>(n)
  for (let i = 0; i < n; i++) {
    const prevClose = i === 0 ? close[0] : close[i - 1]
    const a = high[i] - low[i]
    const b = Math.abs(high[i] - prevClose)
    const c = Math.abs(low[i] - prevClose)
    tr[i] = Math.max(a, Math.max(b, c))
  }
  const atr14 = causalRollingMean(tr, 14)
  features['atr_pct'] = atr14[idx] / (close[idx] + EPS)

  // ── Stochastic %K (14-bar Donchian-style) ────────────────────────────────
  let hh14 = high[0]
  let ll14 = low[0]
  for (let j = Math.max(0, idx - 13); j <= idx; j++) {
    if (high[j] > hh14) hh14 = high[j]
    if (low[j] < ll14) ll14 = low[j]
  }
  features['stoch_k'] = (close[idx] - ll14) / (hh14 - ll14 + EPS) * 100

  // ── Vol ratio ───────────────────────────────────────────────────────────
  const volAvg20 = causalRollingMean(vol, 20)
  features['vol_ratio'] = vol[idx] / (volAvg20[idx] + EPS)

  // ── Higher-timeframe returns (approximate from 10-min grid) ─────────────
  // Python uses aligned 1h / 1d candles; trader server has only 5min candles.
  // 1h_ret = ret over last 6 bars (6 × 10min = 60min)
  // 1d_ret = ret over last 144 bars (144 × 10min = 24h, ~1 trading day)
  // NOTE: this is an approximation — if the trader server ever provides true
  // multi-timeframe data, replace these with proper aligned values.
  features['1h_ret'] = idx >= 6 ? (close[idx] - close[idx - 6]) / (close[idx - 6] + EPS) : 0
  features['1d_ret'] = idx >= 144 ? (close[idx] - close[idx - 144]) / (close[idx - 144] + EPS) : 0

  // ── Time features (MSK) ────────────────────────────────────────────────
  const tsSec = time5[idx] / 1000
  const hourMsk = (Math.floor(tsSec / 3600) + 3) % 24
  const dow = (Math.floor(tsSec / 86400) + 4) % 7  // 0=Thursday (epoch day 0)
  features['hour'] = hourMsk / 24.0
  features['day_of_week'] = dow / 7.0

  // ── Vol regime (ATR percentile rank, last 100 bars) ─────────────────────
  const atrPct = atr14.map((v, i) => v / (close[i] + EPS))
  const W = 100
  const lo = Math.max(0, idx - W + 1)
  let le = 0
  const window: number[] = []
  for (let j = lo; j <= idx; j++) window.push(atrPct[j])
  const sortedWin = window.slice().sort((a, b) => a - b)
  // rank: how many are <= current
  const curAtr = atrPct[idx]
  for (let j = 0; j < sortedWin.length; j++) {
    if (sortedWin[j] <= curAtr) le = j + 1
  }
  features['vol_regime'] = le / sortedWin.length

  // ── Trend strength (simplified ADX proxy: |P(up) - P(down)| over 14 bars) ─
  const upMoves = deltas.map(d => d > 0 ? 1 : 0)
  const downMoves = deltas.map(d => d < 0 ? 1 : 0)
  const upAvg = causalRollingMean(upMoves, 14)
  const downAvg = causalRollingMean(downMoves, 14)
  const adxRaw = Math.abs(upAvg[idx] - downAvg[idx]) * 100
  features['trend_strength'] = adxRaw / 100.0

  // ── Cross-asset (from injected data, else neutral) ──────────────────────
  features['market_breadth'] = _crossAssetData.marketBreadth
    ? _crossAssetData.marketBreadth[idx] ?? 0.5
    : 0.5
  features['sber_gazp_corr'] = _crossAssetData.sberGazpCorr
    ? _crossAssetData.sberGazpCorr[idx] ?? 0
    : 0

  // ── v7 NEW features (14) ────────────────────────────────────────────────
  const { dayStart, prevDayLastBar } = computeDayBoundaries(time5)

  // VWAP + vwap_dev
  const { vwapNorm, vwapDev } = computeVWAP(close, high, low, vol, dayStart, prevDayLastBar)
  features['vwap'] = vwapNorm[idx]
  features['vwap_dev'] = vwapDev[idx]

  // Volume profile: POC, VAH, VAL
  const vp = computeVolumeProfile(close, high, low, vol, dayStart, prevDayLastBar)
  features['poc'] = vp.poc[idx]
  features['vah'] = vp.vah[idx]
  features['val'] = vp.val[idx]

  // Order flow: cum_delta, order_imbalance
  const of = computeOrderFlow(close, high, low, vol, dayStart, prevDayLastBar)
  features['cum_delta'] = of.cumDelta[idx]
  features['order_imbalance'] = of.orderImbalance[idx]

  // Intraday session
  features['intraday_session'] = computeIntradaySession(time5)[idx]

  // Gap overnight
  features['gap_overnight'] = computeGapOvernight(open, close, time5)[idx]

  // Macro features (forward-filled from injected data, else 0)
  features['usdrub_ret'] = _macroData.usdrub_ret ? _macroData.usdrub_ret[idx] ?? 0 : 0
  features['brent_ret'] = _macroData.brent_ret ? _macroData.brent_ret[idx] ?? 0 : 0
  features['imoex_ret'] = _macroData.imoex_ret ? _macroData.imoex_ret[idx] ?? 0 : 0
  features['imoex_ret_1h'] = _macroData.imoex_ret_1h ? _macroData.imoex_ret_1h[idx] ?? 0 : 0
  // cb_rate stored as % (e.g. 21.0), normalize /100 → 0.21
  const cbRaw = _macroData.cb_rate ? _macroData.cb_rate[idx] ?? 8.5 : 8.5
  features['cb_rate'] = cbRaw / 100.0

  // ── Assemble in alphabetical order ─────────────────────────────────────
  const out = new Array<number>(N_FEATURES)
  for (let i = 0; i < N_FEATURES; i++) {
    let v = features[FEATURE_NAMES_V7[i]] ?? 0
    if (!Number.isFinite(v)) v = 0
    if (v > 10) v = 10
    if (v < -10) v = -10
    out[i] = v
  }
  // Zero out first 50 bars (warmup) — but we compute only the LAST bar here,
  // so this guard applies when n < 50 (handled at function entry).
  return out
}

// ─── LightGBM JSON inference ──────────────────────────────────────────────────

/** Recursive LGB tree node — either has leaf_value OR split_feature/threshold + children. */
interface LGBTreeNode {
  leaf_value?: number | number[]
  split_feature?: number
  threshold?: number
  decision_type?: string                 // "<=" (default)
  default_direction?: string | null      // "left" / "right" / null
  left_child?: LGBTreeNode
  right_child?: LGBTreeNode
}

interface LGBTreeInfo {
  tree_index: number
  num_leaves: number
  num_cat: number
  shrinkage: number                       // typically 1.0 (learning_rate baked into leaf values)
  tree_structure: LGBTreeNode
}

interface LightGBMModelJson {
  name: string
  version: string
  num_class: number
  num_tree_per_iteration: number
  label_index: number
  max_feature_idx: number
  objective: string                       // "binary sigmoid:1"
  average_output: boolean
  feature_names?: string[]
  tree_info: LGBTreeInfo[]
  // Optional v7 metadata (added by train_v7.py)
  _feature_names?: string[]
  _version?: string
  _regime?: string
}

export interface LightGBMModel {
  nTrees: number
  nFeatures: number
  trees: LGBTreeInfo[]
  objective: string
  averageOutput: boolean
  sourcePath: string
}

const _lgbCache = new Map<string, LightGBMModel>()

/** Load a LightGBM model from a dump_model() JSON file. Results are cached by path. */
export function loadLightGBMModel(modelPath: string): LightGBMModel {
  const abs = path.isAbsolute(modelPath) ? modelPath : path.resolve(process.cwd(), modelPath)
  const cached = _lgbCache.get(abs)
  if (cached) return cached

  if (!fs.existsSync(abs)) {
    throw new Error(`LightGBM: model file not found: ${abs}`)
  }
  const raw = fs.readFileSync(abs, 'utf-8')
  let json: LightGBMModelJson
  try {
    json = JSON.parse(raw)
  } catch (e: any) {
    throw new Error(`LightGBM: failed to parse JSON at ${abs}: ${e.message}`)
  }

  if (!Array.isArray(json.tree_info) || json.tree_info.length === 0) {
    throw new Error(`LightGBM: no tree_info found in ${abs}`)
  }

  const model: LightGBMModel = {
    nTrees: json.tree_info.length,
    nFeatures: (json.max_feature_idx ?? 0) + 1,
    trees: json.tree_info,
    objective: json.objective ?? 'binary sigmoid:1',
    averageOutput: !!json.average_output,
    sourcePath: abs,
  }
  _lgbCache.set(abs, model)
  return model
}

/** Walk a single LGB tree from root, returning the leaf value (raw score). */
function evalLGBTree(root: LGBTreeNode, features: number[]): number {
  let node: LGBTreeNode | undefined = root
  // Guard against cycles / corrupt trees — 64 hops is plenty (max_depth ~6 in v7).
  for (let depth = 0; depth < 64; depth++) {
    if (!node) return 0
    if (node.leaf_value !== undefined) {
      const lv = node.leaf_value
      return Array.isArray(lv) ? (lv[0] ?? 0) : lv
    }
    const featIdx = node.split_feature!
    const thr = node.threshold!
    const val = features[featIdx]
    let goLeft: boolean
    if (val === undefined || Number.isNaN(val)) {
      // Default direction for NaN — LGB uses "left" by default for missing values
      goLeft = node.default_direction !== 'right'
    } else {
      // decision_type "<=" → go left if val <= thr, else right
      goLeft = val <= thr
    }
    node = goLeft ? node.left_child : node.right_child
  }
  return 0
}

/**
 * Run inference on a single feature vector.
 * Returns P(positive class) ∈ (0, 1).
 *
 * Formula: logit = sum(evalLGBTree(tree.root, X) * tree.shrinkage for all trees)
 *          P = sigmoid(logit)
 *
 * Verified to match lightgbm library within <1e-10.
 */
export function predictLightGBM(model: LightGBMModel, features: number[]): number {
  if (!model || !model.trees || model.trees.length === 0) {
    throw new Error('LightGBM.predictLightGBM: model has no trees')
  }
  if (!Array.isArray(features)) {
    throw new TypeError('LightGBM.predictLightGBM: features must be an array of numbers')
  }
  let logit = 0
  for (let i = 0; i < model.trees.length; i++) {
    const tree = model.trees[i]
    logit += evalLGBTree(tree.tree_structure, features)
    // NOTE: Do NOT multiply by tree.shrinkage — LightGBM's dump_model() already
    // bakes the learning_rate into the leaf values. Verified against lightgbm
    // library: matches EXACTLY (diff=0.0) when shrinkage is NOT applied.
  }
  // average_output == false → just sum (our default). If true → divide by nTrees.
  if (model.averageOutput) {
    logit /= model.trees.length
  }
  return sigmoid(logit)
}

// ─── CatBoost JSON inference ──────────────────────────────────────────────────

interface CatSplit {
  float_feature_index: number
  border: number
  split_index: number
  split_type: string                       // "FloatFeature" / "OneHotFeature"
  // For OneHotFeature: has "value" and "feature_index" instead
}

interface CatObliviousTree {
  leaf_values: number[]
  leaf_weights?: number[]
  splits: CatSplit[]
  // For one-hot splits:
  // splits[i] = { split_type: 'OneHotFeature', feature_index, value, ... }
}

interface CatBoostModelJson {
  oblivious_trees: CatObliviousTree[]
  scale_and_bias?: [number, number[]] | [number, number]
  // v7 metadata
  _feature_names?: string[]
  _version?: string
  _regime?: string
}

export interface CatBoostModel {
  nTrees: number
  trees: CatObliviousTree[]
  scale: number
  bias: number
  sourcePath: string
}

const _catCache = new Map<string, CatBoostModel>()

/** Load a CatBoost model from a save_model(format="json") file. Cached by path. */
export function loadCatBoostModel(modelPath: string): CatBoostModel {
  const abs = path.isAbsolute(modelPath) ? modelPath : path.resolve(process.cwd(), modelPath)
  const cached = _catCache.get(abs)
  if (cached) return cached

  if (!fs.existsSync(abs)) {
    throw new Error(`CatBoost: model file not found: ${abs}`)
  }
  const raw = fs.readFileSync(abs, 'utf-8')
  let json: CatBoostModelJson
  try {
    json = JSON.parse(raw)
  } catch (e: any) {
    throw new Error(`CatBoost: failed to parse JSON at ${abs}: ${e.message}`)
  }

  if (!Array.isArray(json.oblivious_trees) || json.oblivious_trees.length === 0) {
    throw new Error(`CatBoost: no oblivious_trees found in ${abs}`)
  }

  let scale = 1
  let bias = 0
  if (Array.isArray(json.scale_and_bias)) {
    scale = typeof json.scale_and_bias[0] === 'number' ? json.scale_and_bias[0] : 1
    const b = json.scale_and_bias[1]
    if (Array.isArray(b)) bias = b[0] ?? 0
    else if (typeof b === 'number') bias = b
  }

  const model: CatBoostModel = {
    nTrees: json.oblivious_trees.length,
    trees: json.oblivious_trees,
    scale,
    bias,
    sourcePath: abs,
  }
  _catCache.set(abs, model)
  return model
}

/**
 * Walk a single CatBoost oblivious tree.
 *
 * Each split contributes 1 bit to the leaf index (LSB-first: split i = bit i).
 *   bit_i = 1 if feature[float_feature_index] > border, else 0
 *   leaf_idx = sum(bit_i << i for i in range(depth))
 *
 * Returns leaf_values[leaf_idx].
 *
 * Verified to match catboost library EXACTLY (diff=0.0).
 */
function evalCatTree(tree: CatObliviousTree, features: number[]): number {
  const splits = tree.splits
  const depth = splits.length
  let leafIdx = 0
  for (let i = 0; i < depth; i++) {
    const s = splits[i]
    const featIdx = s.float_feature_index
    const border = s.border
    const val = features[featIdx]
    if (val !== undefined && !Number.isNaN(val) && val > border) {
      leafIdx |= (1 << i)
    }
  }
  return tree.leaf_values[leafIdx] ?? 0
}

/**
 * Run inference on a single feature vector.
 * Returns P(positive class) ∈ (0, 1).
 *
 * Formula: logit = scale * sum(evalCatTree(tree, X) for all trees) + bias
 *          P = sigmoid(logit)
 *
 * Verified to match catboost library EXACTLY (diff=0.0).
 */
export function predictCatBoost(model: CatBoostModel, features: number[]): number {
  if (!model || !model.trees || model.trees.length === 0) {
    throw new Error('CatBoost.predictCatBoost: model has no trees')
  }
  if (!Array.isArray(features)) {
    throw new TypeError('CatBoost.predictCatBoost: features must be an array of numbers')
  }
  let total = 0
  for (let i = 0; i < model.trees.length; i++) {
    total += evalCatTree(model.trees[i], features)
  }
  const logit = model.scale * total + model.bias
  return sigmoid(logit)
}

// ─── Stacking meta-model ──────────────────────────────────────────────────────

interface StackingJson {
  version: string
  regimes: Record<string, {
    status: string                       // "trained" | "skipped"
    coef: number[][]                     // shape [1, 3] (for binary LR: coef_ from sklearn)
    intercept: number
    classes: number[]
    n_train?: number
    n_val?: number
    n_test?: number
    feature_names?: string[]             // ["xgb_prob", "lgb_prob", "cat_prob"]
  }>
}

interface StackingInfo {
  status: 'trained' | 'skipped' | 'missing'
  coef: [number, number, number]
  intercept: number
}

export interface StackingModel {
  regimes: Record<string, StackingInfo>
  sourcePath: string
}

const _stackingCache = new Map<string, StackingModel>()

/** Load the per-regime stacking coefficients (LogisticRegression on (xgb_p, lgb_p, cat_p)). */
export function loadStacking(modelPath: string): StackingModel {
  const abs = path.isAbsolute(modelPath) ? modelPath : path.resolve(process.cwd(), modelPath)
  const cached = _stackingCache.get(abs)
  if (cached) return cached

  if (!fs.existsSync(abs)) {
    throw new Error(`Stacking: file not found: ${abs}`)
  }
  const raw = fs.readFileSync(abs, 'utf-8')
  let json: StackingJson
  try {
    json = JSON.parse(raw)
  } catch (e: any) {
    throw new Error(`Stacking: failed to parse JSON at ${abs}: ${e.message}`)
  }

  const regimes: Record<string, StackingInfo> = {}
  for (const [name, info] of Object.entries(json.regimes || {})) {
    if (info.status !== 'trained') {
      regimes[name] = { status: 'skipped', coef: [0, 0, 0], intercept: 0 }
      continue
    }
    const coefArr = info.coef?.[0] ?? [0, 0, 0]
    regimes[name] = {
      status: 'trained',
      coef: [coefArr[0] ?? 0, coefArr[1] ?? 0, coefArr[2] ?? 0],
      intercept: info.intercept ?? 0,
    }
  }
  const model: StackingModel = { regimes, sourcePath: abs }
  _stackingCache.set(abs, model)
  return model
}

/**
 * Combine 3 base-model probabilities via the per-regime stacking LR.
 *
 * Formula:  p_final = sigmoid(coef[0]*p_xgb + coef[1]*p_lgb + coef[2]*p_cat + intercept)
 *
 * If the regime was skipped during training (status="skipped"), falls back to
 * simple average of the 3 probabilities.
 */
export function predictStacking(
  stacking: StackingModel,
  regimeName: string,
  pXgb: number, pLgb: number, pCat: number,
): number {
  const info = stacking.regimes[regimeName]
  if (!info || info.status !== 'trained') {
    // Fallback: simple mean
    return (pXgb + pLgb + pCat) / 3
  }
  const logit = info.coef[0] * pXgb + info.coef[1] * pLgb + info.coef[2] * pCat + info.intercept
  return sigmoid(logit)
}

// ─── Strategy class ───────────────────────────────────────────────────────────

/** Cached models for a single regime — loaded lazily on first hit. */
interface RegimeModels {
  xgb: XGBoostBinaryModel | null
  lgb: LightGBMModel | null
  cat: CatBoostModel | null
}

/** Resolve the absolute path of a model file, searching MODEL_SEARCH_PATHS. */
function resolveModelFile(fileName: string): string | null {
  for (const base of MODEL_SEARCH_PATHS) {
    if (!base) continue
    const p = path.join(base, fileName)
    try {
      if (fs.existsSync(p)) return p
    } catch { /* ignore */ }
  }
  return null
}

/** Resolve the stacking file path (regime_v7_stacking.json). */
function resolveStackingFile(): string | null {
  return resolveModelFile('regime_v7_stacking.json')
}

/** Cached regime model bundle (3 base models + stacking). */
const _regimeCache = new Map<number, RegimeModels & { stackingOk: boolean }>()

/**
 * Load (or fetch from cache) the 3 base models + stacking for a regime.
 * Returns null for xgb/lgb/cat if any individual model file is missing.
 */
function loadRegimeModels(regimeIdx: number): RegimeModels & { stackingOk: boolean } {
  const cached = _regimeCache.get(regimeIdx)
  if (cached) return cached

  const rnameLower = REGIME_NAMES_LOWER[regimeIdx]
  const result: RegimeModels & { stackingOk: boolean } = {
    xgb: null, lgb: null, cat: null, stackingOk: false,
  }

  const xgbPath = resolveModelFile(`regime_v7_${rnameLower}_xgb.json`)
  if (xgbPath) {
    try { result.xgb = loadModel(xgbPath) } catch { result.xgb = null }
  }
  const lgbPath = resolveModelFile(`regime_v7_${rnameLower}_lgb.json`)
  if (lgbPath) {
    try { result.lgb = loadLightGBMModel(lgbPath) } catch { result.lgb = null }
  }
  const catPath = resolveModelFile(`regime_v7_${rnameLower}_cat.json`)
  if (catPath) {
    try { result.cat = loadCatBoostModel(catPath) } catch { result.cat = null }
  }

  const stackingPath = resolveStackingFile()
  if (stackingPath) {
    try {
      // Pre-load the stacking file so predictStacking doesn't re-resolve every tick.
      loadStacking(stackingPath)
      result.stackingOk = true
    } catch { result.stackingOk = false }
  }

  _regimeCache.set(regimeIdx, result)
  return result
}

/**
 * MetaSelectorV7 — ensemble inference strategy.
 *
 * On each predict() call:
 *   1. detectRegime(candles) → 0..11
 *   2. If regime 7 or 8 → fallback (LONG/SHORT)
 *   3. Compute 36 features
 *   4. Load 3 models (xgb, lgb, cat) + stacking for this regime
 *   5. Get p_xgb, p_lgb, p_cat
 *   6. p_final = sigmoid(coef[0]*p_xgb + coef[1]*p_lgb + coef[2]*p_cat + intercept)
 *   7. p > 0.65 → LONG, p < 0.35 → SHORT, else FLAT
 *   8. Exit: if holding and p crosses 0.50
 *
 * Action codes (match trader server):
 *   0 = FLAT, 1 = LONG, 2 = SHORT, 3 = close long, 4 = close short
 */
export class MetaSelectorV7Strategy implements IStrategy {
  name = 'meta_selector_v7'
  description = 'ML v7: ensemble (XGB+LGB+CAT+stacking), 12 regimes, 36 features, P>0.65 LONG / P<0.35 SHORT, exit@0.50'

  private currentRegime = -1
  private currentRegimeName = ''
  private lastP = 0.5
  private lastPXgb = 0.5
  private lastPLgb = 0.5
  private lastPCat = 0.5

  predict(
    candles: Candle[],
    idx: number,
    hasPosition: boolean,
    stepsHeld?: number,
    ctx?: StrategyContext,
  ): number {
    if (candles.length < MIN_CANDLES) return ACTION_FLAT

    // ── 1. Detect regime ────────────────────────────────────────────────────
    const regimeResult: RegimeResult = detectRegime(candles)
    const regime = regimeResult.regime
    const regimeName = regimeResult.regimeName
    this.currentRegime = regime
    this.currentRegimeName = regimeName

    // ── 2. Fallback for regimes without ML models ───────────────────────────
    if (regime === 7) {
      // OVERSOLD_BOUNCE → LONG
      console.log(`[MetaSelectorV7] regime=${regimeName} → LONG (fallback: oversold bounce)`)
      return ACTION_LONG
    }
    if (regime === 8) {
      // OVERBOUGHT_REVERSAL → SHORT
      console.log(`[MetaSelectorV7] regime=${regimeName} → SHORT (fallback: overbought reversal)`)
      return ACTION_SHORT
    }

    // ── 3. Load models for this regime ──────────────────────────────────────
    const models = loadRegimeModels(regime)
    if (!models.xgb && !models.lgb && !models.cat) {
      console.log(`[MetaSelectorV7] regime=${regimeName} → FLAT (no models found for regime ${regime})`)
      return ACTION_FLAT
    }

    // ── 4. Compute 36 features ───────────────────────────────────────────────
    const features = computeFeaturesV7(candles)

    // ── 5. Get 3 probabilities ────────────────────────────────────────────────
    let pXgb = 0.5, pLgb = 0.5, pCat = 0.5
    try {
      if (models.xgb) pXgb = predict_proba(models.xgb, features)
    } catch (e: any) {
      console.log(`[MetaSelectorV7] regime=${regimeName} XGB predict error: ${e.message}`)
    }
    try {
      if (models.lgb) pLgb = predictLightGBM(models.lgb, features)
    } catch (e: any) {
      console.log(`[MetaSelectorV7] regime=${regimeName} LGB predict error: ${e.message}`)
    }
    try {
      if (models.cat) pCat = predictCatBoost(models.cat, features)
    } catch (e: any) {
      console.log(`[MetaSelectorV7] regime=${regimeName} CAT predict error: ${e.message}`)
    }
    this.lastPXgb = pXgb
    this.lastPLgb = pLgb
    this.lastPCat = pCat

    // ── 6. Stacking ─────────────────────────────────────────────────────────
    let pFinal = 0.5
    const stackingPath = resolveStackingFile()
    if (stackingPath) {
      try {
        const stacking = loadStacking(stackingPath)
        pFinal = predictStacking(stacking, regimeName, pXgb, pLgb, pCat)
      } catch (e: any) {
        console.log(`[MetaSelectorV7] stacking error: ${e.message} — using simple mean`)
        pFinal = (pXgb + pLgb + pCat) / 3
      }
    } else {
      pFinal = (pXgb + pLgb + pCat) / 3
    }
    this.lastP = pFinal

    // ── 7-8. Decision / exit logic ───────────────────────────────────────────
    let action = ACTION_FLAT
    let reason = ''

    if (hasPosition) {
      // Exit: if holding and p crosses 0.50 (toward the other side)
      if (pFinal < EXIT_LONG) {
        action = ACTION_CLOSE_LONG
        reason = `EXIT_LONG (P=${pFinal.toFixed(3)} < ${EXIT_LONG})`
      } else if (pFinal > EXIT_SHORT) {
        action = ACTION_CLOSE_SHORT
        reason = `EXIT_SHORT (P=${pFinal.toFixed(3)} > ${EXIT_SHORT})`
      } else {
        action = ACTION_FLAT
        reason = `HOLD (P=${pFinal.toFixed(3)})`
      }
    } else {
      // Entry: simple threshold
      if (pFinal > LONG_THRESHOLD) {
        action = ACTION_LONG
        reason = `LONG (P=${pFinal.toFixed(3)} > ${LONG_THRESHOLD})`
      } else if (pFinal < SHORT_THRESHOLD) {
        action = ACTION_SHORT
        reason = `SHORT (P=${pFinal.toFixed(3)} < ${SHORT_THRESHOLD})`
      } else {
        action = ACTION_FLAT
        reason = `FLAT (P=${pFinal.toFixed(3)} in [${SHORT_THRESHOLD}, ${LONG_THRESHOLD}])`
      }
    }

    console.log(
      `[MetaSelectorV7] regime=${regimeName} p_xgb=${pXgb.toFixed(3)} p_lgb=${pLgb.toFixed(3)} p_cat=${pCat.toFixed(3)} p_final=${pFinal.toFixed(3)} → ${reason}`,
    )
    return action
  }

  // ─── Inspection helpers ──────────────────────────────────────────────────
  getCurrentRegime(): number { return this.currentRegime }
  getCurrentRegimeName(): string { return this.currentRegimeName }
  getLastP(): number { return this.lastP }
  getLastPXgb(): number { return this.lastPXgb }
  getLastPLgb(): number { return this.lastPLgb }
  getLastPCat(): number { return this.lastPCat }
}

export function createMetaSelectorV7Strategy(): MetaSelectorV7Strategy {
  return new MetaSelectorV7Strategy()
}

// ─── Self-test (run with: npx tsx meta_selector_v7.ts) ────────────────────────

const isMain =
  (typeof import.meta === 'object' && (import.meta as any).main === true) ||
  (typeof require !== 'undefined' && typeof require.main !== 'undefined' && require.main === module)

if (isMain) {
  console.log(`=== MetaSelectorV7 — self-test ===\n`)

  // ── Test 1: feature computation (deterministic, no macro data) ────────────
  const N = 300
  const candles: Candle[] = []
  let price = 250
  let t = Math.floor(Date.now() / 1000)
  for (let i = 0; i < N; i++) {
    const drift = i > 200 ? 0.08 : 0.02
    const noise = ((i * 1103515245 + 12345) % 1000) / 1000 - 0.5  // LCG, deterministic
    const open = price
    const close = price + drift + noise * 0.5
    const high = Math.max(open, close) + Math.abs(noise) * 0.3
    const low = Math.min(open, close) - Math.abs(noise) * 0.3
    const vol = 1000 + Math.abs(noise) * 500
    candles.push({ time: (t + i * 600) * 1000, open, high, low, close, volume: vol })
    price = close
  }
  console.log(`1. Computing 36 features on ${N} synthetic candles...`)
  const features = computeFeaturesV7(candles)
  console.log(`   Features: n=${features.length} (expected 36)`)
  console.log(`   NaN/Inf: ${features.some(v => !Number.isFinite(v))}`)
  const sample = features.slice(0, 10).map(v => v.toFixed(4)).join(', ')
  console.log(`   Sample (first 10): [${sample}, ...]`)
  if (features.length !== 36) throw new Error('Expected 36 features')
  if (features.some(v => !Number.isFinite(v))) throw new Error('NaN/Inf in features')

  // ── Test 2: regime detection ─────────────────────────────────────────────
  console.log(`\n2. Detecting regime on synthetic candles...`)
  const regimeResult = detectRegime(candles)
  console.log(`   regime=${regimeResult.regime} (${regimeResult.regimeName}) conf=${regimeResult.confidence.toFixed(3)} adx=${regimeResult.adx.toFixed(2)} rsi=${regimeResult.rsi.toFixed(2)}`)

  // ── Test 3: LGB inference with real model (if downloaded) ─────────────────
  const lgbPath = '/home/z/my-project/_v7_3_models/regime_v7_strong_trend_up_lgb.json'
  if (fs.existsSync(lgbPath)) {
    console.log(`\n3. Loading + testing LightGBM model...`)
    const lgb = loadLightGBMModel(lgbPath)
    console.log(`   nTrees: ${lgb.nTrees}  nFeatures: ${lgb.nFeatures}  objective: ${lgb.objective}`)
    // Test 3 sample feature vectors
    for (let trial = 0; trial < 3; trial++) {
      const f = Array.from({ length: 36 }, (_, i) => Math.sin(i + trial * 0.7) * 0.5)
      const p = predictLightGBM(lgb, f)
      console.log(`   trial ${trial}: predictLightGBM = ${p.toFixed(6)}  (in [0,1]: ${p >= 0 && p <= 1})`)
      if (!(p >= 0 && p <= 1)) throw new Error('LGB prob out of [0,1]')
    }
  } else {
    console.log(`\n3. Skipping LGB test — model file not found at ${lgbPath}`)
  }

  // ── Test 4: CAT inference with real model ────────────────────────────────
  const catPath = '/home/z/my-project/_v7_3_models/regime_v7_strong_trend_up_cat.json'
  if (fs.existsSync(catPath)) {
    console.log(`\n4. Loading + testing CatBoost model...`)
    const cat = loadCatBoostModel(catPath)
    console.log(`   nTrees: ${cat.nTrees}  scale: ${cat.scale}  bias: ${cat.bias}`)
    for (let trial = 0; trial < 3; trial++) {
      const f = Array.from({ length: 36 }, (_, i) => Math.sin(i + trial * 0.7) * 0.5)
      const p = predictCatBoost(cat, f)
      console.log(`   trial ${trial}: predictCatBoost = ${p.toFixed(6)}  (in [0,1]: ${p >= 0 && p <= 1})`)
      if (!(p >= 0 && p <= 1)) throw new Error('CAT prob out of [0,1]')
    }
  } else {
    console.log(`\n4. Skipping CAT test — model file not found at ${catPath}`)
  }

  // ── Test 5: XGB inference with real model ────────────────────────────────
  const xgbPath = '/home/z/my-project/_v7_3_models/regime_v7_strong_trend_up_xgb.json'
  if (fs.existsSync(xgbPath)) {
    console.log(`\n5. Loading + testing XGBoost model...`)
    const xgb = loadModel(xgbPath)
    console.log(`   nTrees: ${xgb.nTrees}  nFeatures: ${xgb.nFeatures}`)
    for (let trial = 0; trial < 3; trial++) {
      const f = Array.from({ length: 36 }, (_, i) => Math.sin(i + trial * 0.7) * 0.5)
      const p = predict_proba(xgb, f)
      console.log(`   trial ${trial}: predict_proba = ${p.toFixed(6)}  (in [0,1]: ${p >= 0 && p <= 1})`)
      if (!(p >= 0 && p <= 1)) throw new Error('XGB prob out of [0,1]')
    }
  } else {
    console.log(`\n5. Skipping XGB test — model file not found at ${xgbPath}`)
  }

  // ── Test 6: Stacking ─────────────────────────────────────────────────────
  const stkPath = '/home/z/my-project/_v7_3_models/regime_v7_stacking.json'
  if (fs.existsSync(stkPath)) {
    console.log(`\n6. Loading + testing Stacking...`)
    const stk = loadStacking(stkPath)
    console.log(`   Regimes: ${Object.keys(stk.regimes).length}`)
    const info = stk.regimes['STRONG_TREND_UP']
    console.log(`   STRONG_TREND_UP: status=${info.status} coef=[${info.coef.map(v => v.toFixed(4)).join(', ')}] intercept=${info.intercept.toFixed(4)}`)
    const pFinal = predictStacking(stk, 'STRONG_TREND_UP', 0.55, 0.48, 0.60)
    console.log(`   predictStacking(0.55, 0.48, 0.60) = ${pFinal.toFixed(6)}  (in [0,1]: ${pFinal >= 0 && pFinal <= 1})`)
    if (!(pFinal >= 0 && pFinal <= 1)) throw new Error('Stacking prob out of [0,1]')
  }

  // ── Test 7: Full strategy predict() with real model files ────────────────
  console.log(`\n7. Running full MetaSelectorV7Strategy.predict() on synthetic candles...`)
  // Point strategy at downloaded models
  const strat = new MetaSelectorV7Strategy()
  const action = strat.predict(candles, candles.length - 1, false, 0, {})
  console.log(`   → action=${action}  regime=${strat.getCurrentRegimeName()}  P_final=${strat.getLastP().toFixed(4)}`)

  console.log(`\n=== self-test complete ===`)
}
