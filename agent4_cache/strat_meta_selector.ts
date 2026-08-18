$ cat /opt/ai-trader/src/strategies/meta_selector.ts
--- rc=0 ---
/**
 * Meta-Selector Strategy — ML model that picks which strategy to run.
 *
 * Architecture:
 *   1. Compute 33 market features from candles (RSI, SMA, MACD, BB, ATR, regime, etc.)
 *   2. Run pure-TS XGBoost inference (loaded from JSON) → softprob over 20 strategies
 *   3. Pick top-3 predicted strategies, rotate between them every tick
 *
 * Trained on 4466 backtest samples × 11 tickers × 180 days.
 * Strategy pool: 20 of 22 (heikin_ashi & orb were missing from training data).
 *
 * Top-3 test accuracy: 22% (vs 15% random for 20 classes)
 * Per-regime direction captures the actual best strategy type (reversion/trend).
 *
 * Inference is pure TS — no Python at runtime. Model JSON is bundled.
 */
import { Candle } from '../core/types'
import { IStrategy, StrategyContext, createStrategy } from './base'
import { mlPredict as mlPredictV1 } from './ml_predict'
import { mlPredict as mlPredictV2 } from './ml_predict_v2'
import { indicators } from './indicators'
import * as fs from 'fs'
import * as path from 'path'

// ─── Types ────────────────────────────────────────────────────────────────────

interface XGBoostTree {
  id: number
  left_children: number[]    // -1 means leaf
  right_children: number[]
  split_indices: number[]    // feature index per node
  split_conditions: number[] // threshold per node
  base_weights: number[]     // leaf value when node is leaf
  tree_param: { num_nodes: number; num_deleted: number }
}

interface XGBoostModel {
  learner: {
    gradient_booster: {
      model: {
        trees: XGBoostTree[]
        tree_info: number[]
      }
    }
  }
}

interface Metadata {
  n_features: number
  feature_names: string[]
  n_classes_effective: number
  strategy_names: string[]
  class_map_encoded_to_original: { [key: string]: number }
  class_map_original_to_encoded: { [key: string]: number }
  missing_strategies: string[]
}

// ─── Feature computation (TS port of ml_features.py) ──────────────────────────

function causalSMA(arr: number[], window: number): number[] {
  const n = arr.length
  const out = new Array(n).fill(0)
  let sum = 0
  for (let i = 0; i < n; i++) {
    sum += arr[i]
    if (i >= window) sum -= arr[i - window]
    out[i] = i < window ? sum / (i + 1) : sum / window
  }
  return out
}

function rsi(closes: number[], period: number = 14): number[] {
  const n = closes.length
  const out = new Array(n).fill(50)
  let gains = 0, losses = 0
  for (let i = 1; i < n; i++) {
    const change = closes[i] - closes[i - 1]
    const gain = change > 0 ? change : 0
    const loss = change < 0 ? -change : 0
    if (i <= period) {
      gains += gain
      losses += loss
      if (i === period) {
        const avgG = gains / period
        const avgL = losses / period
        out[i] = avgL === 0 ? 100 : 100 - 100 / (1 + avgG / avgL)
      }
    } else {
      const prevRSI = out[i - 1]
      const prevAvgG = (prevRSI >= 50) ? (100 - prevRSI) * (gains / period) / (prevRSI === 100 ? 1 : prevRSI) : gains / period
      // Wilder smoothing
      const alpha = 1 / period
      gains = (1 - alpha) * gains + alpha * gain
      losses = (1 - alpha) * losses + alpha * loss
      out[i] = losses === 0 ? 100 : 100 - 100 / (1 + gains / losses)
    }
  }
  return out
}

function stddev(arr: number[], window: number): number[] {
  const n = arr.length
  const out = new Array(n).fill(0)
  for (let i = window - 1; i < n; i++) {
    let sum = 0, sumSq = 0
    for (let j = i - window + 1; j <= i; j++) {
      sum += arr[j]
      sumSq += arr[j] * arr[j]
    }
    const mean = sum / window
    const variance = sumSq / window - mean * mean
    out[i] = Math.sqrt(Math.max(0, variance))
  }
  return out
}

/**
 * Compute 33 market features at the latest bar.
 * Mirrors ml_features.py compute_features() + regime + trend_slope.
 */
function computeMetaFeatures(candles: Candle[]): number[] {
  const n = candles.length
  const close = candles.map(c => c.close)
  const high = candles.map(c => c.high)
  const low = candles.map(c => c.low)
  const volume = candles.map(c => c.volume)
  const open = candles.map(c => c.open)

  if (n < 50) {
    // Not enough history — return zeros (33 features)
    return new Array(33).fill(0)
  }

  // Returns
  const ret1 = new Array(n).fill(0)
  const ret5 = new Array(n).fill(0)
  const ret10 = new Array(n).fill(0)
  const ret30 = new Array(n).fill(0)
  for (let i = 1; i < n; i++) ret1[i] = (close[i] - close[i - 1]) / (close[i - 1] + 1e-10)
  for (let i = 5; i < n; i++) ret5[i] = (close[i] - close[i - 5]) / (close[i - 5] + 1e-10)
  for (let i = 10; i < n; i++) ret10[i] = (close[i] - close[i - 10]) / (close[i - 10] + 1e-10)
  for (let i = 30; i < n; i++) ret30[i] = (close[i] - close[i - 30]) / (close[i - 30] + 1e-10)
  const ret5log = ret5.map(r => Math.log(Math.abs(r) + 1) * Math.sign(r))

  // SMAs
  const sma5 = causalSMA(close, 5)
  const sma14 = causalSMA(close, 14)
  const sma20 = causalSMA(close, 20)
  const sma50 = causalSMA(close, 50)

  // RSI
  const rsi14 = rsi(close, 14)
  const rsi2 = rsi(close, 2)

  // SMA ratios
  const sma5_sma14 = sma5.map((v, i) => v / (sma14[i] + 1e-10))
  const sma14_sma20 = sma14.map((v, i) => v / (sma20[i] + 1e-10))
  const sma20_sma50 = sma20.map((v, i) => v / (sma50[i] + 1e-10))

  // Bollinger Bands
  const bbStd = stddev(close, 20)
  const bbLower = new Array(n).fill(0)
  const bbUpper = new Array(n).fill(0)
  const bbWidth = new Array(n).fill(0)
  const bbPctB = new Array(n).fill(0)
  for (let i = 0; i < n; i++) {
    bbLower[i] = sma20[i] - 2 * bbStd[i]
    bbUpper[i] = sma20[i] + 2 * bbStd[i]
    bbWidth[i] = (bbUpper[i] - bbLower[i]) / (sma20[i] + 1e-10)
    bbPctB[i] = (bbUpper[i] - bbLower[i]) > 0 ? (close[i] - bbLower[i]) / (bbUpper[i] - bbLower[i]) : 0
  }

  // MACD
  const ema12 = causalEMA(close, 12)
  const ema26 = causalEMA(close, 26)
  const macdLine = ema12.map((v, i) => v - ema26[i])
  const macdSignal = causalEMA(macdLine, 9)
  const macdHist = macdLine.map((v, i) => v - macdSignal[i])

  // ATR%
  const atr = new Array(n).fill(0)
  for (let i = 1; i < n; i++) {
    const tr = Math.max(
      high[i] - low[i],
      Math.abs(high[i] - close[i - 1]),
      Math.abs(low[i] - close[i - 1])
    )
    atr[i] = i === 1 ? tr : (atr[i - 1] * 13 + tr) / 14
  }
  const atrPct = atr.map((v, i) => v / (close[i] + 1e-10))

  // Stochastic %K
  const stochK = new Array(n).fill(50)
  for (let i = 13; i < n; i++) {
    const highest = Math.max(...high.slice(i - 13, i + 1))
    const lowest = Math.min(...low.slice(i - 13, i + 1))
    stochK[i] = highest === lowest ? 50 : ((close[i] - lowest) / (highest - lowest)) * 100
  }

  // ADX (simplified)
  const adxArr = new Array(n).fill(0)
  let plusDM = 0, minusDM = 0, tr14 = 0
  for (let i = 1; i < n; i++) {
    const up = high[i] - high[i - 1]
    const down = low[i - 1] - low[i]
    const pDM = up > down && up > 0 ? up : 0
    const mDM = down > up && down > 0 ? down : 0
    const tr = Math.max(high[i] - low[i], Math.abs(high[i] - close[i - 1]), Math.abs(low[i] - close[i - 1]))
    if (i <= 14) {
      plusDM += pDM
      minusDM += mDM
      tr14 += tr
      if (i === 14) {
        const pdi = 100 * (plusDM / 14) / (tr14 / 14 + 1e-10)
        const mdi = 100 * (minusDM / 14) / (tr14 / 14 + 1e-10)
        const dx = Math.abs(pdi - mdi) / (pdi + mdi + 1e-10) * 100
        adxArr[i] = dx
      }
    } else {
      plusDM = (plusDM * 13 + pDM) / 14
      minusDM = (minusDM * 13 + mDM) / 14
      tr14 = (tr14 * 13 + tr) / 14
      const pdi = 100 * plusDM / (tr14 + 1e-10)
      const mdi = 100 * minusDM / (tr14 + 1e-10)
      const dx = Math.abs(pdi - mdi) / (pdi + mdi + 1e-10) * 100
      adxArr[i] = (adxArr[i - 1] * 13 + dx) / 14
    }
  }

  // OBV slope
  const obv = new Array(n).fill(0)
  for (let i = 1; i < n; i++) {
    obv[i] = obv[i - 1] + (close[i] > close[i - 1] ? volume[i] : -volume[i])
  }
  const obvSlope = new Array(n).fill(0)
  for (let i = 5; i < n; i++) obvSlope[i] = (obv[i] - obv[i - 5]) / (obv[i - 5] + 1e-10)

  // Volume ratio (current vs avg 20)
  const vol20 = causalSMA(volume, 20)
  const volRatio = volume.map((v, i) => v / (vol20[i] + 1e-10))

  // Higher TF context (simplified — using 5min aggregates)
  // 1h return ~ 12 bars ago
  const idx = n - 1
  const ret1h = idx >= 12 ? (close[idx] - close[idx - 12]) / (close[idx - 12] + 1e-10) : 0
  const rsi1h = rsi14[idx]  // approximate
  const ret1d = idx >= 144 ? (close[idx] - close[idx - 144]) / (close[idx - 144] + 1e-10) : 0
  const trend1d = ret1d > 0 ? 1 : ret1d < 0 ? -1 : 0
  const trend1h = ret1h > 0 ? 1 : ret1h < 0 ? -1 : 0
  const ret5_ = ret5[idx]

  // Time features
  const ts = candles[idx].time
  const date = new Date(ts)
  const hour = date.getUTCHours() / 24.0
  const dayOfWeek = date.getUTCDay() / 7.0

  // Regime (TREND_UP=1, TREND_DOWN=2, RANGE=0)
  let regime = 0
  const adx = adxArr[idx]
  if (sma50[idx] > sma20[idx] * 0.999 && sma20[idx] > sma14[idx] * 0.999 && adx > 20) {
    regime = 1
  } else if (sma50[idx] < sma20[idx] * 1.001 && sma20[idx] < sma14[idx] * 1.001 && adx > 20) {
    regime = 2
  }

  // Trend slope (sma50 - sma14) / sma14
  const trendSlope = (sma50[idx] - sma14[idx]) / (sma14[idx] + 1e-10)

  // Build features in the SAME ORDER as training metadata:
  // ['1d_ret', '1d_trend', '1h_ret', '1h_rsi', '1h_trend', 'adx', 'atr_pct',
  //  'bb_pct_b', 'bb_width', 'day_of_week', 'hour', 'macd_hist', 'macd_line',
  //  'macd_signal', 'obv_slope', 'price_bb_lower', 'price_bb_upper', 'price_sma20',
  //  'price_sma50', 'ret_1', 'ret_10', 'ret_30', 'ret_5', 'ret_5_log', 'rsi14',
  //  'rsi2', 'sma14_sma20', 'sma20_sma50', 'sma5_sma14', 'stoch_k', 'vol_ratio',
  //  'regime', 'trend_slope']
  return [
    ret1d,
    trend1d,
    ret1h,
    rsi1h,
    trend1h,
    adx,
    atrPct[idx],
    bbPctB[idx],
    bbWidth[idx],
    dayOfWeek,
    hour,
    macdHist[idx],
    macdLine[idx],
    macdSignal[idx],
    obvSlope[idx],
    bbLower[idx] / (close[idx] + 1e-10),  // price_bb_lower (normalized)
    bbUpper[idx] / (close[idx] + 1e-10),  // price_bb_upper
    sma20[idx] / (close[idx] + 1e-10),    // price_sma20
    sma50[idx] / (close[idx] + 1e-10),    // price_sma50
    ret1[idx],
    ret10[idx],
    ret30[idx],
    ret5[idx],
    ret5log[idx],
    rsi14[idx],
    rsi2[idx],
    sma14_sma20[idx],
    sma20_sma50[idx],
    sma5_sma14[idx],
    stochK[idx],
    volRatio[idx],
    regime,
    trendSlope,
  ]
}

function causalEMA(arr: number[], period: number): number[] {
  const n = arr.length
  const out = new Array(n).fill(0)
  const alpha = 2 / (period + 1)
  out[0] = arr[0]
  for (let i = 1; i < n; i++) {
    out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
  }
  return out
}

// ─── XGBoost pure-TS inference ────────────────────────────────────────────────

let _modelCache: XGBoostModel | null = null
let _metadataCache: Metadata | null = null

function loadModel(): XGBoostModel {
  if (_modelCache) return _modelCache
  // Try multiple paths where the model file might live
  const paths = [
    '/opt/ai-trader/src/strategies/meta_classifier.json',
    '/opt/ai-trader/data/meta_classifier.json',
    path.join(__dirname, 'meta_classifier.json'),
  ]
  for (const p of paths) {
    try {
      if (fs.existsSync(p)) {
        _modelCache = JSON.parse(fs.readFileSync(p, 'utf-8'))
        return _modelCache!
      }
    } catch {}
  }
  throw new Error(`meta_classifier.json not found in any of: ${paths.join(', ')}`)
}

function loadMetadata(): Metadata {
  if (_metadataCache) return _metadataCache
  const paths = [
    '/opt/ai-trader/src/strategies/meta_metadata.json',
    '/opt/ai-trader/data/meta_metadata.json',
    path.join(__dirname, 'meta_metadata.json'),
  ]
  for (const p of paths) {
    try {
      if (fs.existsSync(p)) {
        _metadataCache = JSON.parse(fs.readFileSync(p, 'utf-8'))
        return _metadataCache!
      }
    } catch {}
  }
  throw new Error(`meta_metadata.json not found in any of: ${paths.join(', ')}`)
}

function evalTree(tree: XGBoostTree, features: number[]): number {
  // Walk the tree using array-based node representation.
  // Node 0 is root. left_children[i] === -1 means node i is a leaf.
  let node = 0
  let depth = 0
  while (tree.left_children[node] !== -1 && depth < 100) {
    const featureIdx = tree.split_indices[node]
    const threshold = tree.split_conditions[node]
    const val = features[featureIdx] ?? 0
    // XGBoost: if feature value < threshold → go left, else go right
    // default_left is usually 0 (go left when missing)
    if (Number.isNaN(val)) {
      node = tree.left_children[node]
    } else if (val < threshold) {
      node = tree.left_children[node]
    } else {
      node = tree.right_children[node]
    }
    depth++
  }
  // Leaf value is base_weights[node]
  return tree.base_weights[node]
}

function predictProba(features: number[]): number[] {
  const model = loadModel()
  const meta = loadMetadata()
  const trees = model.learner.gradient_booster.model.trees
  const nClasses = meta.n_classes_effective

  // Sum leaf values per class (trees are ordered: tree[0]→class0, tree[1]→class1, ...)
  const sums = new Array(nClasses).fill(0)
  for (let t = 0; t < trees.length; t++) {
    const cls = t % nClasses
    const leaf = evalTree(trees[t], features)
    sums[cls] += leaf
  }

  // Softmax
  const maxLogit = Math.max(...sums)
  const exps = sums.map(s => Math.exp(s - maxLogit))
  const sumExp = exps.reduce((a, b) => a + b, 0)
  return exps.map(e => e / sumExp)
}

// ─── Strategy switching logic ────────────────────────────────────────────────

const SUPPORTED_STRATEGIES = new Set([
  'v2_short', 'multi_timeframe', 'v2_inverted', 'mean_reversion', 'trend_follow',
  'random_hold_short', 'bb_reversion', 'macd_trend', 'donchian_breakout',
  'stoch_oscillator', 'vwap_reversion', 'momentum_volume', 'connors_rsi2',
  'zscore_reversion', 'supertrend', 'bollinger_squeeze', 'atr_bands',
  'dual_thrust', 'awesome_oscillator', 'golden_cross',
])

export class MetaSelectorStrategy implements IStrategy {
  name = 'meta_selector'
  description = 'ML meta-classifier: picks best strategy from 20-strategy pool based on market regime'

  private lastSwitchTs = 0
  private currentStrategy: IStrategy | null = null
  private currentStrategyName = ''
  private switchIntervalMs = 3 * 60 * 1000  // switch at most every 3 min
  private lastPredictions: Array<{ name: string; prob: number }> = []

  predict(candles: Candle[], idx: number, hasPosition: boolean, stepsHeld?: number, ctx?: StrategyContext): number {
    if (candles.length < 50) return 0  // not enough history

    const now = candles[idx].time

    // Switch strategy at most every switchIntervalMs
    const shouldSwitch = !this.currentStrategy || (now - this.lastSwitchTs > this.switchIntervalMs && !hasPosition)

    if (shouldSwitch) {
      try {
        const features = computeMetaFeatures(candles)
        const probs = predictProba(features)
        const meta = loadMetadata()

        // Decode: encoded_idx → original_idx → strategy_name
        const ranked = probs.map((p, encIdx) => ({
          encIdx,
          origIdx: meta.class_map_encoded_to_original[String(encIdx)],
          prob: p,
        })).sort((a, b) => b.prob - a.prob)

        this.lastPredictions = ranked.slice(0, 5).map(r => ({
          name: meta.strategy_names[r.origIdx],
          prob: r.prob,
        }))

        // Pick top-1 among SUPPORTED strategies (some like heikin_ashi/orb are excluded)
        const top3Str = ranked.slice(0, 3).map(x =>
          `${meta.strategy_names[x.origIdx]}:${(x.prob * 100).toFixed(0)}%`
        ).join(' ')
        let pickedName = ''
        for (const r of ranked) {
          const name = meta.strategy_names[r.origIdx]
          if (SUPPORTED_STRATEGIES.has(name) && r.prob > 0.05) {
            try {
              this.currentStrategy = createStrategy(name, { params: {} })
              this.currentStrategyName = name
              this.lastSwitchTs = now
              pickedName = name
              break
            } catch (e) {
              // Strategy failed to instantiate — try next
              continue
            }
          }
        }
        console.log(`[MetaSelector] PREDICT top3=${top3Str} → picked=${pickedName || '(fallback)'}`)
      } catch (e: any) {
        console.log(`[MetaSelector] model error: ${e.message}`)
        // Fallback: use v2_short if model not loaded
        if (!this.currentStrategy) {
          try {
            this.currentStrategy = createStrategy('v2_short', { params: {} })
            this.currentStrategyName = 'v2_short (fallback)'
            console.log(`[MetaSelector] fallback → v2_short`)
          } catch (e2) {
            return 0
          }
        }
      }
    }

    if (!this.currentStrategy) return 0
    return this.currentStrategy.predict(candles, idx, hasPosition, stepsHeld, ctx)
  }

  /** Returns last ML predictions for logging/monitoring */
  getLastPredictions(): Array<{ name: string; prob: number }> {
    return this.lastPredictions
  }

  getCurrentStrategyName(): string {
    return this.currentStrategyName
  }
}

export function createMetaSelectorStrategy(): MetaSelectorStrategy {
  return new MetaSelectorStrategy()
}


