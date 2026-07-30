/**
 * BollingerBounceStrategy — Bollinger Bands mean-reversion (John Bollinger, 1980s)
 *
 * Logic:
 *   LONG: price < lower BB + RSI < 30 (oversold at lower band)
 *   SHORT: price > upper BB + RSI > 70 (overbought at upper band)
 *   Exit: price returns to SMA20 (mean)
 */
import { IStrategy, StrategyContext } from './base'
import { Candle } from '../core/types'
import { indicators } from './indicators'

export class BollingerBounceStrategy implements IStrategy {
  name = 'BollingerBounce'
  description = 'BB bounce: LONG at lower BB + RSI<30, SHORT at upper BB + RSI>70'

  predict(candles: Candle[], idx: number, hasPosition: boolean, _stepsHeld?: number, ctx?: StrategyContext): number {
    const ind = indicators(candles, idx)
    if (!ind || idx < 20) return 0

    // Compute Bollinger Bands (20, 2)
    const closes20 = candles.slice(idx - 19, idx + 1).map(c => c.close)
    const sma20 = closes20.reduce((a: number, b: number) => a + b, 0) / 20
    const variance = closes20.reduce((s, c) => s + Math.pow(c - sma20, 2), 0) / 20
    const stdDev = Math.sqrt(variance)
    const bbUpper = sma20 + 2 * stdDev
    const bbLower = sma20 - 2 * stdDev
    const price = ind.cur

    if (hasPosition) {
      // Exit at mean (SMA20)
      const holding = ctx?.holding || 0
      if (holding > 0 && price > sma20) return 3
      if (holding < 0 && price < sma20) return 3
      return 0
    }

    if (price < bbLower && ind.rsi < 30) return 1   // long at lower band
    if (price > bbUpper && ind.rsi > 70) return 2   // short at upper band
    return 0
  }
}
