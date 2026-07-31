/**
 * RsiExtremesStrategy — Classic RSI mean-reversion (Welles Wilder, 1978)
 *
 * Logic:
 *   LONG: RSI < 25 (oversold)
 *   SHORT: RSI > 75 (overbought)
 *   Exit: RSI returns to 50 (neutral)
 */
import { IStrategy, StrategyContext } from './base'
import { Candle } from '../core/types'
import { indicators } from './indicators'

export class RsiExtremesStrategy implements IStrategy {
  name = 'RsiExtremes'
  description = 'RSI extremes: LONG RSI<25, SHORT RSI>75, exit at RSI 50'

  predict(candles: Candle[], idx: number, hasPosition: boolean, _stepsHeld?: number, ctx?: StrategyContext): number {
    const ind = indicators(candles, idx)
    if (!ind) return 0

    if (hasPosition) {
      // Exit at neutral RSI
      if (ind.rsi > 45 && ind.rsi < 55) return 3
      return 0
    }

    if (ind.rsi < 25) return 1   // oversold → long
    if (ind.rsi > 75) return 2   // overbought → short
    return 0
  }
}
