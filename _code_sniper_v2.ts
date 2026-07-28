/** SniperTrendV2 — rule-based, focus on shorts, long hold, RSI 30-55. */
import { IStrategy } from './base'
import { Candle } from '../core/types'
import { indicators } from './indicators'

export class SniperTrendV2Strategy implements IStrategy {
  name = 'SniperTrendV2'
  description = 'SniperTrend V2: фокус на шорты, дольше держит, RSI 30-55'

  predict(candles: Candle[], idx: number, hasPosition: boolean, _stepsHeld?: number): number {
    const ind = indicators(candles, idx)
    if (!ind) return 0
    if (hasPosition) {
      if (ind.sma5 > ind.sma14 * 1.003 && ind.rsi > 65) return 3
      if (ind.sma5 < ind.sma14 * 0.997 && ind.rsi < 35) return 3
      return 0
    }
    if (ind.sma5 < ind.sma14 * 0.999 && ind.rsi > 30 && ind.rsi < 55 && ind.allDown) return 2
    if (ind.sma5 > ind.sma14 * 1.002 && ind.rsi > 25 && ind.rsi < 40 && ind.allUp) return 1
    return 0
  }
}
