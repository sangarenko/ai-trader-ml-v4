/**
 * ML Strategy — XGBoost inference in pure TypeScript.
 * 
 * No Python needed. Loads model from JSON (exported by export_xgboost_json.py).
 * Computes 31 features from candle data, runs XGBoost trees, returns action.
 * 
 * Trading rules:
 *   P(long) > 0.65 → action=1 (buy)
 *   P(short) > 0.80 → action=2 (sell short)
 *   otherwise → action=0 (hold)
 * 
 * Performance: ~0.1ms per predict (300 trees × if-then-else)
 */

import { IStrategy, StrategyContext } from './base'
import { Candle } from '../core/types'
import { indicators } from './indicators'
import * as fs from 'fs'
import * as path from 'path'

// ─── Types ───────────────────────────────────────────────────────────────────

interface TreeNode {
  nodeid: number
  depth?: number
  split?: number      // feature index
  split_condition?: number
  yes?: number
  no?: number
  missing?: number
  children?: TreeNode[]
  leaf?: number
}

interface XGBoostModel {
  n_trees: number
  n_features: number
  feature_names: string[]
  base_score: number
  trees: TreeNode[]
}

// ─── Feature computation ────────────────────────────────────────────────────

function causalSMA(arr: number[], w: number): number[] {
  const n = arr.length
  const result = new Array(n).fill(0)
  let sum = 0
  for (let i = 0; i < n; i++) {
    sum += arr[i]
    if (i >= w) sum -= arr[i - w]
    result[i] = sum / Math.min(i + 1, w)
  }
  return result
}

function rollingMean(arr: number[], w: number): number[] {
  return causalSMA(arr, w)
}

function ema(arr: number[], period: number): number[] {
  const n = arr.length
  const k = 2 / (period + 1)
  const result = new Array(n).fill(0)
  result[0] = arr[0]
  for (let i = 1; i < n; i++) {
    result[i] = arr[i] * k + result[i - 1] * (1 - k)
  }
  return result
}

function computeMLFeatures(candles: Candle[]): number[] {
  const n = candles.length
  if (n < 50) return new Array(31).fill(0)

  const closes = candles.map(c => c.close)
  const highs = candles.map(c => c.high)
  const lows = candles.map(c => c.low)
  const volumes = candles.map(c => c.volume)
  const opens = candles.map(c => c.open)

  const features: Record<string, number> = {}

  // === Returns ===
  const prevClose = new Array(n).fill(0)
  prevClose[0] = closes[0]
  for (let i = 1; i < n; i++) prevClose[i] = closes[i - 1]

  features['ret_1'] = (closes[n-1] - prevClose[n-1]) / (prevClose[n-1] + 1e-10)
  
  const ret5 = n > 5 ? (closes[n-1] - closes[n-6]) / (closes[n-6] + 1e-10) : 0
  const ret10 = n > 10 ? (closes[n-1] - closes[n-11]) / (closes[n-11] + 1e-10) : 0
  const ret30 = n > 30 ? (closes[n-1] - closes[n-31]) / (closes[n-31] + 1e-10) : 0
  features['ret_5'] = ret5
  features['ret_10'] = ret10
  features['ret_30'] = ret30
  features['ret_5_log'] = n > 5 ? Math.log(closes[n-1] / (closes[n-6] + 1e-10)) : 0

  // === SMAs ===
  const sma5 = causalSMA(closes, 5)
  const sma14 = causalSMA(closes, 14)
  const sma20 = causalSMA(closes, 20)
  const sma50 = causalSMA(closes, 50)

  features['sma5_sma14'] = sma5[n-1] / (sma14[n-1] + 1e-10)
  features['sma14_sma20'] = sma14[n-1] / (sma20[n-1] + 1e-10)
  features['sma20_sma50'] = sma20[n-1] / (sma50[n-1] + 1e-10)
  features['price_sma20'] = closes[n-1] / (sma20[n-1] + 1e-10)
  features['price_sma50'] = closes[n-1] / (sma50[n-1] + 1e-10)

  // === RSI ===
  const deltas = new Array(n).fill(0)
  for (let i = 1; i < n; i++) deltas[i] = closes[i] - closes[i-1]
  
  const gains = deltas.map(d => d > 0 ? d : 0)
  const losses = deltas.map(d => d < 0 ? -d : 0)
  
  const avgGain = rollingMean(gains, 14)
  const avgLoss = rollingMean(losses, 14)
  const rs = avgGain[n-1] / (avgLoss[n-1] + 1e-10)
  features['rsi14'] = 100 - 100 / (1 + rs)
  
  const avgGain2 = rollingMean(gains, 2)
  const avgLoss2 = rollingMean(losses, 2)
  const rs2 = avgGain2[n-1] / (avgLoss2[n-1] + 1e-10)
  features['rsi2'] = 100 - 100 / (1 + rs2)

  // === Bollinger Bands ===
  const std20slice = closes.slice(n - 20)
  const mean20 = std20slice.reduce((a, b) => a + b, 0) / 20
  const variance = std20slice.reduce((s, c) => s + (c - mean20) ** 2, 0) / 19
  const std20 = Math.sqrt(variance)
  const bbUpper = sma20[n-1] + 2 * std20
  const bbLower = sma20[n-1] - 2 * std20
  const bbWidth = (4 * std20) / (sma20[n-1] + 1e-10)
  
  features['bb_pct_b'] = (closes[n-1] - bbLower) / (4 * std20 + 1e-10)
  features['bb_width'] = bbWidth
  features['price_bb_upper'] = closes[n-1] / (bbUpper + 1e-10)
  features['price_bb_lower'] = closes[n-1] / (bbLower + 1e-10)

  // === MACD ===
  const ema12 = ema(closes, 12)
  const ema26 = ema(closes, 26)
  const macdLine = ema12[n-1] - ema26[n-1]
  const macdSignalArr = ema(closes.map((_, i) => ema12[i] - ema26[i]), 9)
  const macdSignal = macdSignalArr[n-1]
  const macdHist = macdLine - macdSignal
  
  features['macd_hist'] = macdHist / (closes[n-1] + 1e-10)
  features['macd_line'] = macdLine / (closes[n-1] + 1e-10)
  features['macd_signal'] = macdSignal / (closes[n-1] + 1e-10)

  // === ATR ===
  const trs: number[] = []
  for (let i = 1; i < n; i++) {
    const tr = Math.max(
      highs[i] - lows[i],
      Math.abs(highs[i] - closes[i-1]),
      Math.abs(lows[i] - closes[i-1])
    )
    trs.push(tr)
  }
  const atr = trs.slice(-14).reduce((a, b) => a + b, 0) / 14
  features['atr_pct'] = atr / (closes[n-1] + 1e-10)

  // === Volume ===
  const volAvg = volumes.slice(-20).reduce((a, b) => a + b, 0) / 20
  features['vol_ratio'] = volumes[n-1] / (volAvg + 1e-10)
  
  // OBV slope (simplified)
  let obv = 0
  for (let i = 1; i < n; i++) {
    if (closes[i] > closes[i-1]) obv += volumes[i]
    else if (closes[i] < closes[i-1]) obv -= volumes[i]
  }
  let obv10ago = 0
  for (let i = 1; i < n - 10; i++) {
    if (closes[i] > closes[i-1]) obv10ago += volumes[i]
    else if (closes[i] < closes[i-1]) obv10ago -= volumes[i]
  }
  features['obv_slope'] = (obv - obv10ago) / (Math.abs(obv10ago) + 1e-10)

  // === Stochastic ===
  const hh14 = Math.max(...highs.slice(-14))
  const ll14 = Math.min(...lows.slice(-14))
  features['stoch_k'] = hh14 > ll14 ? (closes[n-1] - ll14) / (hh14 - ll14) * 100 : 50

  // === ADX (simplified) ===
  let upMoves = 0, downMoves = 0
  for (let i = n - 14; i < n; i++) {
    if (i > 0) {
      if (closes[i] > closes[i-1]) upMoves++
      else if (closes[i] < closes[i-1]) downMoves++
    }
  }
  features['adx'] = Math.abs(upMoves - downMoves) / 14 * 100

  // === Time features (MSK = UTC+3) ===
  const ts = candles[n-1].time
  const date = new Date(ts)
  const utcHours = date.getUTCHours()
  const mskHour = (utcHours + 3) % 24
  const utcDay = Math.floor(date.getTime() / 86400000)
  const mskDay = (utcDay + 4) % 7  // Thursday=0 (epoch day 0 = Thursday)
  features['hour'] = mskHour / 24.0
  features['day_of_week'] = mskDay / 7.0

  // === Higher TF (simplified — using 5min data as proxy) ===
  // In production, these would come from 1h and 1d candles
  // For now, approximate from 5min closes
  if (n >= 60) {
    const hourClose = closes[n-1]
    const hourPrev = closes[n - 60] // 60 × 5min = 5 hours ≈ 1 bar
    features['1h_ret'] = (hourClose - hourPrev) / (hourPrev + 1e-10)
    const sma1h = causalSMA(closes.slice(-60), 10)
    features['1h_trend'] = hourClose / (sma1h[59] + 1e-10)
    // 1h RSI (simplified)
    const hGains = gains.slice(-60)
    const hLosses = losses.slice(-60)
    const hAvgGain = hGains.reduce((a, b) => a + b, 0) / 60
    const hAvgLoss = hLosses.reduce((a, b) => a + b, 0) / 60
    features['1h_rsi'] = 100 - 100 / (1 + hAvgGain / (hAvgLoss + 1e-10))
  } else {
    features['1h_ret'] = 0
    features['1h_trend'] = 1
    features['1h_rsi'] = 50
  }

  if (n >= 200) {
    const dayClose = closes[n-1]
    const dayPrev = closes[n - 200] // ~1 day on 5min
    features['1d_ret'] = (dayClose - dayPrev) / (dayPrev + 1e-10)
    const sma1d = causalSMA(closes.slice(-200), 5)
    features['1d_trend'] = dayClose / (sma1d[199] + 1e-10)
  } else {
    features['1d_ret'] = 0
    features['1d_trend'] = 1
  }

  // Build feature array in ALPHABETICAL order (matches training)
  const featureNames = [
    '1d_ret', '1d_trend', '1h_ret', '1h_rsi', '1h_trend', 'adx', 'atr_pct',
    'bb_pct_b', 'bb_width', 'day_of_week', 'hour', 'macd_hist', 'macd_line',
    'macd_signal', 'obv_slope', 'price_bb_lower', 'price_bb_upper',
    'price_sma20', 'price_sma50', 'ret_1', 'ret_10', 'ret_30', 'ret_5',
    'ret_5_log', 'rsi14', 'rsi2', 'sma14_sma20', 'sma20_sma50', 'sma5_sma14',
    'stoch_k', 'vol_ratio'
  ]

  return featureNames.map(name => {
    const val = features[name] || 0
    return Math.max(-10, Math.min(10, val)) // clip
  })
}

// ─── XGBoost inference (pure TS) ────────────────────────────────────────────

function walkTree(node: TreeNode, features: number[]): number {
  if (node.leaf !== undefined) {
    return node.leaf
  }

  const featureIdx = node.split!
  const threshold = node.split_condition!

  if (features[featureIdx] < threshold) {
    // yes child
    const yesChild = node.children?.find(c => c.nodeid === node.yes)
    return yesChild ? walkTree(yesChild, features) : 0
  } else {
    // no child
    const noChild = node.children?.find(c => c.nodeid === node.no)
    return noChild ? walkTree(noChild, features) : 0
  }
}

function predictXGBoost(model: XGBoostModel, features: number[]): number {
  // Sum all tree predictions
  let rawScore = model.base_score
  for (const tree of model.trees) {
    rawScore += walkTree(tree, features)
  }
  // Sigmoid
  return 1.0 / (1.0 + Math.exp(-rawScore))
}

// ─── ML Strategy ────────────────────────────────────────────────────────────

export class MLPredictStrategy implements IStrategy {
  name = 'MLPredict'
  description = 'ML XGBoost: 31 features → P(long)>0.65 buy, P(short)>0.80 sell'

  private modelLong: XGBoostModel | null = null
  private modelShort: XGBoostModel | null = null
  private loaded = false

  private loadModels(): void {
    if (this.loaded) return

    const modelDir = '/root/ai-trader-evolution/ml/models'
    
    try {
      const longPath = path.join(modelDir, 'ml_model_180d_long.json')
      const shortPath = path.join(modelDir, 'ml_model_180d_short.json')
      
      if (fs.existsSync(longPath)) {
        this.modelLong = JSON.parse(fs.readFileSync(longPath, 'utf-8'))
        console.log(`[MLPredict] Loaded LONG model: ${this.modelLong.n_trees} trees`)
      }
      if (fs.existsSync(shortPath)) {
        this.modelShort = JSON.parse(fs.readFileSync(shortPath, 'utf-8'))
        console.log(`[MLPredict] Loaded SHORT model: ${this.modelShort.n_trees} trees`)
      }
      
      this.loaded = true
    } catch (e) {
      console.error(`[MLPredict] Failed to load models: ${e}`)
    }
  }

  predict(candles: Candle[], idx: number, hasPosition: boolean, _stepsHeld?: number, _ctx?: StrategyContext): number {
    this.loadModels()
    
    if (!this.modelLong || !this.modelShort) {
      console.error('[MLPredict] Models not loaded')
      return 0
    }

    if (candles.length < 50) return 0

    // Get last 1000 candles (or all available)
    const startIdx = Math.max(0, idx - 999)
    const window = candles.slice(startIdx, idx + 1)
    
    // Compute 31 features
    const features = computeMLFeatures(window)
    
    // Predict
    const pLong = predictXGBoost(this.modelLong, features)
    const pShort = predictXGBoost(this.modelShort, features)
    
    // Log every 100 ticks
    if (idx % 100 === 0) {
      console.log(`[MLPredict] idx=${idx} P(long)=${pLong.toFixed(3)} P(short)=${pShort.toFixed(3)}`)
    }

    if (hasPosition) {
      // Exit logic: exit if probability drops below 0.5
      const holding = _ctx?.holding || 0
      if (holding > 0 && pLong < 0.50) return 3   // exit long
      if (holding < 0 && pShort < 0.50) return 3   // exit short
      return 0
    }

    // Entry logic
    if (pLong > 0.65) return 1   // buy signal
    if (pShort > 0.80) return 2  // short signal
    
    return 0  // hold
  }
}
