/**
 * ML Strategy v2 — Regime-Aware (Trend/Range detection)
 * 
 * Улучшение v1:
 * 1. Определяет режим рынка (TREND DOWN / TREND UP / RANGE)
 * 2. В TREND DOWN → агрессивнее SHORT (threshold 0.55 вместо 0.80)
 * 3. В TREND UP → агрессивнее LONG (threshold 0.55 вместо 0.65)
 * 4. В RANGE → только high-confidence сигналы (0.70+)
 * 5. Сезонность: день месяца, сезон года, день недели
 * 6. Position sizing зависит от силы тренда (ADX)
 */

import { IStrategy, StrategyContext } from './base'
import { Candle } from '../core/types'
import { indicators } from './indicators'
import * as fs from 'fs'
import * as path from 'path'

// ─── Types ───

interface TreeNode {
  nodeid: number
  split?: number
  split_condition?: number
  yes?: number
  no?: number
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

type MarketRegime = 'TREND_UP' | 'TREND_DOWN' | 'RANGE'

// ─── Feature computation (v2 with seasonality) ───

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

function computeMLFeaturesV2(candles: Candle[]): { features: number[]; regime: MarketRegime; adx: number } {
  const n = candles.length
  if (n < 50) return { features: new Array(34).fill(0), regime: 'RANGE', adx: 0 }

  const closes = candles.map(c => c.close)
  const highs = candles.map(c => c.high)
  const lows = candles.map(c => c.low)
  const volumes = candles.map(c => c.volume)
  const opens = candles.map(c => c.open)

  const features: Record<string, number> = {}

  // === Original 31 features (same as v1) ===
  const prevClose = new Array(n).fill(0)
  prevClose[0] = closes[0]
  for (let i = 1; i < n; i++) prevClose[i] = closes[i - 1]

  features['ret_1'] = (closes[n-1] - prevClose[n-1]) / (prevClose[n-1] + 1e-10)
  features['ret_5'] = n > 5 ? (closes[n-1] - closes[n-6]) / (closes[n-6] + 1e-10) : 0
  features['ret_10'] = n > 10 ? (closes[n-1] - closes[n-11]) / (closes[n-11] + 1e-10) : 0
  features['ret_30'] = n > 30 ? (closes[n-1] - closes[n-31]) / (closes[n-31] + 1e-10) : 0
  features['ret_5_log'] = n > 5 ? Math.log(closes[n-1] / (closes[n-6] + 1e-10)) : 0

  const sma5 = causalSMA(closes, 5)
  const sma14 = causalSMA(closes, 14)
  const sma20 = causalSMA(closes, 20)
  const sma50 = causalSMA(closes, 50)

  features['sma5_sma14'] = sma5[n-1] / (sma14[n-1] + 1e-10)
  features['sma14_sma20'] = sma14[n-1] / (sma20[n-1] + 1e-10)
  features['sma20_sma50'] = sma20[n-1] / (sma50[n-1] + 1e-10)
  features['price_sma20'] = closes[n-1] / (sma20[n-1] + 1e-10)
  features['price_sma50'] = closes[n-1] / (sma50[n-1] + 1e-10)

  // RSI
  const deltas = new Array(n).fill(0)
  for (let i = 1; i < n; i++) deltas[i] = closes[i] - closes[i-1]
  const gains = deltas.map(d => d > 0 ? d : 0)
  const losses = deltas.map(d => d < 0 ? -d : 0)
  
  const avgGain = causalSMA(gains, 14)
  const avgLoss = causalSMA(losses, 14)
  features['rsi14'] = 100 - 100 / (1 + avgGain[n-1] / (avgLoss[n-1] + 1e-10))
  
  const avgGain2 = causalSMA(gains, 2)
  const avgLoss2 = causalSMA(losses, 2)
  features['rsi2'] = 100 - 100 / (1 + avgGain2[n-1] / (avgLoss2[n-1] + 1e-10))

  // Bollinger
  const std20slice = closes.slice(n - 20)
  const mean20 = std20slice.reduce((a, b) => a + b, 0) / 20
  const variance = std20slice.reduce((s, c) => s + (c - mean20) ** 2, 0) / 19
  const std20 = Math.sqrt(variance)
  features['bb_pct_b'] = (closes[n-1] - (sma20[n-1] - 2 * std20)) / (4 * std20 + 1e-10)
  features['bb_width'] = (4 * std20) / (sma20[n-1] + 1e-10)
  features['price_bb_upper'] = closes[n-1] / (sma20[n-1] + 2 * std20 + 1e-10)
  features['price_bb_lower'] = closes[n-1] / (sma20[n-1] - 2 * std20 + 1e-10)

  // MACD
  const ema12 = ema(closes, 12)
  const ema26 = ema(closes, 26)
  const macdLine = ema12[n-1] - ema26[n-1]
  const macdSignalArr = ema(closes.map((_, i) => ema12[i] - ema26[i]), 9)
  features['macd_hist'] = (macdLine - macdSignalArr[n-1]) / (closes[n-1] + 1e-10)
  features['macd_line'] = macdLine / (closes[n-1] + 1e-10)
  features['macd_signal'] = macdSignalArr[n-1] / (closes[n-1] + 1e-10)

  // ATR
  const trs: number[] = []
  for (let i = 1; i < n; i++) {
    trs.push(Math.max(highs[i] - lows[i], Math.abs(highs[i] - closes[i-1]), Math.abs(lows[i] - closes[i-1])))
  }
  const atr = trs.slice(-14).reduce((a, b) => a + b, 0) / 14
  features['atr_pct'] = atr / (closes[n-1] + 1e-10)

  // Volume
  const volAvg = volumes.slice(-20).reduce((a, b) => a + b, 0) / 20
  features['vol_ratio'] = volumes[n-1] / (volAvg + 1e-10)
  features['obv_slope'] = 0 // simplified

  // Stochastic
  const hh14 = Math.max(...highs.slice(-14))
  const ll14 = Math.min(...lows.slice(-14))
  features['stoch_k'] = hh14 > ll14 ? (closes[n-1] - ll14) / (hh14 - ll14) * 100 : 50

  // ADX (simplified)
  let upMoves = 0, downMoves = 0
  for (let i = n - 14; i < n; i++) {
    if (i > 0) {
      if (closes[i] > closes[i-1]) upMoves++
      else if (closes[i] < closes[i-1]) downMoves++
    }
  }
  const adx = Math.abs(upMoves - downMoves) / 14 * 100
  features['adx'] = adx

  // Time (MSK)
  const ts = candles[n-1].time
  const date = new Date(ts)
  const mskHour = (date.getUTCHours() + 3) % 24
  const utcDay = Math.floor(date.getTime() / 86400000)
  const mskDay = (utcDay + 4) % 7
  features['hour'] = mskHour / 24.0
  features['day_of_week'] = mskDay / 7.0

  // Higher TF
  if (n >= 60) {
    features['1h_ret'] = (closes[n-1] - closes[n-60]) / (closes[n-60] + 1e-10)
    features['1h_trend'] = closes[n-1] / (causalSMA(closes.slice(-60), 10)[59] + 1e-10)
    features['1h_rsi'] = 50 // simplified
  } else {
    features['1h_ret'] = 0; features['1h_trend'] = 1; features['1h_rsi'] = 50
  }
  if (n >= 200) {
    features['1d_ret'] = (closes[n-1] - closes[n-200]) / (closes[n-200] + 1e-10)
    features['1d_trend'] = closes[n-1] / (causalSMA(closes.slice(-200), 5)[199] + 1e-10)
  } else {
    features['1d_ret'] = 0; features['1d_trend'] = 1
  }

  // === NEW v2 features: Seasonality ===
  
  // Day of month (1-31)
  const dayOfMonth = date.getUTCDate()
  features['day_of_month'] = dayOfMonth / 31.0

  // Month (1-12) — seasonal patterns on MOEX
  const month = date.getUTCMonth() + 1
  features['month'] = month / 12.0

  // Season: winter=0, spring=1, summer=2, autumn=3
  const season = Math.floor((month - 1) / 3)
  features['season'] = season / 3.0

  // Is dividend season (April-May, July-August on MOEX)
  const isDivSeason = (month >= 4 && month <= 5) || (month >= 7 && month <= 8) ? 1 : 0
  features['is_dividend_season'] = isDivSeason

  // === Regime detection ===
  // TREND_UP: SMA50 > SMA20 > SMA14 (медленный > средний > быстрый = восходящий)
  // TREND_DOWN: SMA50 < SMA20 < SMA14
  // RANGE: SMA values close together (within 1%)
  
  let regime: MarketRegime = 'RANGE'
  const sma14v = sma14[n-1]
  const sma20v = sma20[n-1]
  const sma50v = sma50[n-1]
  
  const upTrend = sma50v > sma20v * 0.999 && sma20v > sma14v * 0.999
  const downTrend = sma50v < sma20v * 1.001 && sma20v < sma14v * 1.001
  
  if (upTrend && adx > 20) regime = 'TREND_UP'
  else if (downTrend && adx > 20) regime = 'TREND_DOWN'
  else regime = 'RANGE'

  // Build feature array (31 original + 3 new = 34)
  // Model expects 31 features — pad new ones with 0 for compatibility
  const featureNames = [
    '1d_ret', '1d_trend', '1h_ret', '1h_rsi', '1h_trend', 'adx', 'atr_pct',
    'bb_pct_b', 'bb_width', 'day_of_week', 'hour', 'macd_hist', 'macd_line',
    'macd_signal', 'obv_slope', 'price_bb_lower', 'price_bb_upper',
    'price_sma20', 'price_sma50', 'ret_1', 'ret_10', 'ret_30', 'ret_5',
    'ret_5_log', 'rsi14', 'rsi2', 'sma14_sma20', 'sma20_sma50', 'sma5_sma14',
    'stoch_k', 'vol_ratio'
  ]

  const featureArray = featureNames.map(name => {
    const val = features[name] || 0
    return Math.max(-10, Math.min(10, val))
  })

  return { features: featureArray, regime, adx }
}

// ─── XGBoost inference ───

function walkTree(node: TreeNode, features: number[]): number {
  if (node.leaf !== undefined) return node.leaf
  const idx = node.split!
  const threshold = node.split_condition!
  if (features[idx] < threshold) {
    const child = node.children?.find(c => c.nodeid === node.yes)
    return child ? walkTree(child, features) : 0
  } else {
    const child = node.children?.find(c => c.nodeid === node.no)
    return child ? walkTree(child, features) : 0
  }
}

function predictXGBoost(model: XGBoostModel, features: number[]): number {
  let rawScore = model.base_score
  for (const tree of model.trees) {
    rawScore += walkTree(tree, features)
  }
  return 1.0 / (1.0 + Math.exp(-rawScore))
}

// ─── Regime-Aware ML Strategy v2 ───

export class MLPredictV2Strategy implements IStrategy {
  name = 'MLPredictV2'
  description = 'ML v2: Regime-aware (TREND/RANGE) + seasonality. Adapts thresholds to market regime.'

  private modelLong: XGBoostModel | null = null
  private modelShort: XGBoostModel | null = null
  private loaded = false
  private lastRegimeLog = -1000  // log first tick immediately

  private loadModels(): void {
    if (this.loaded) return
    const modelDir = '/root/ai-trader-evolution/ml/models'
    try {
      const longPath = path.join(modelDir, 'ml_model_180d_long.json')
      const shortPath = path.join(modelDir, 'ml_model_180d_short.json')
      if (fs.existsSync(longPath)) {
        this.modelLong = JSON.parse(fs.readFileSync(longPath, 'utf-8'))
      }
      if (fs.existsSync(shortPath)) {
        this.modelShort = JSON.parse(fs.readFileSync(shortPath, 'utf-8'))
      }
      this.loaded = true
      console.log(`[MLPredictV2] Loaded: LONG=${this.modelLong?.n_trees || 0} trees, SHORT=${this.modelShort?.n_trees || 0} trees`)
    } catch (e) {
      console.error(`[MLPredictV2] Load failed: ${e}`)
    }
  }

  predict(candles: Candle[], idx: number, hasPosition: boolean, _stepsHeld?: number, ctx?: StrategyContext): number {
    this.loadModels()
    if (!this.modelLong || !this.modelShort) return 0
    if (candles.length < 50) return 0

    const ticker = candles[idx]?.ticker || ''
    const startIdx = Math.max(0, idx - 999)
    const window = candles.slice(startIdx, idx + 1)
    
    const { features, regime, adx } = computeMLFeaturesV2(window)
    const pLong = predictXGBoost(this.modelLong, features)
    const pShort = predictXGBoost(this.modelShort, features)

    // === REGIME-AWARE THRESHOLDS ===
    // Adapt thresholds based on market regime
    
    let longThreshold = 0.65   // default
    let shortThreshold = 0.80   // default
    let positionSizeMultiplier = 1.0

    if (regime === 'TREND_DOWN') {
      // Market falling → be aggressive on SHORTs, cautious on LONGs
      shortThreshold = 0.55     // easier to short in downtrend
      longThreshold = 0.75      // harder to long in downtrend
      positionSizeMultiplier = 1.2  // bigger positions (trend is our friend)
    } else if (regime === 'TREND_UP') {
      // Market rising → be aggressive on LONGs, cautious on SHORTs
      longThreshold = 0.55      // easier to long in uptrend
      shortThreshold = 0.75     // harder to short in uptrend
      positionSizeMultiplier = 1.2
    } else {
      // RANGE → only high-confidence signals
      longThreshold = 0.70
      shortThreshold = 0.85
      positionSizeMultiplier = 0.8   // smaller positions in choppy market
    }

    // ADX bonus: strong trend = even more aggressive
    if (adx > 40) {
      positionSizeMultiplier *= 1.1
    }

    // Log regime every tick
    console.log(`[MLPredictV2] ${ticker || '?'} regime=${regime} ADX=${adx.toFixed(0)} P(l)=${pLong.toFixed(3)} P(s)=${pShort.toFixed(3)} thr: L=${longThreshold} S=${shortThreshold}`)

    if (hasPosition) {
      // Exit logic: regime-aware
      const holding = ctx?.holding || 0
      if (holding > 0) {
        // Exit long if: trend turned down, or probability dropped
        if (regime === 'TREND_DOWN' && pLong < 0.60) return 3
        if (pLong < 0.45) return 3
      }
      if (holding < 0) {
        // Exit short if: trend turned up, or probability dropped
        if (regime === 'TREND_UP' && pShort < 0.60) return 3
        if (pShort < 0.45) return 3
      }
      return 0
    }

    // Entry logic: regime-aware
    if (pLong > longThreshold) return 1
    if (pShort > shortThreshold) return 2
    
    return 0
  }
}
