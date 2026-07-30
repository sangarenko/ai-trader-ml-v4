/**
 * StochOscillatorStrategy — Stochastic oscillator (George Lane, 1950s)
 *
 * Logic:
 *   LONG: %K < 20 + RSI < 40 (oversold)
 *   SHORT: %K > 80 + RSI > 60 (overbought)
 *   Exit: %K returns to 50 (neutral)
 */
import { IStrategy, StrategyContext } from './base'
import { Candle } from '../core/types'
import { indicators } from './indicators'

export class StochOscillatorStrategy implements IStrategy {
  name = 'StochOscillator'
  description = 'Stochastic: LONG %K<20+RSI<40, SHORT %K>80+RSI>60, exit %K=50'

  predict(candles: Candle[], idx: number, hasPosition: boolean, _stepsHeld?: number, _ctx?: StrategyContext): number {
    if (idx < 14) return 0

    const ind = indicators(candles, idx)
    if (!ind) return 0

    // Stochastic %K (14 period)
    const highs = candles.slice(idx - 13, idx + 1).map(c => c.high)
    const lows = candles.slice(idx - 13, idx + 1).map(c => c.low)
    const hh = Math.max(...highs)
    const ll = Math.min(...lows)
    const k = hh > ll ? (candles[idx].close - ll) / (hh - ll) * 100 : 50

    if (hasPosition) {
      if (k > 45 && k < 55) return 3  // exit at neutral
      return 0
    }

    if (k < 20 && ind.rsi < 40) return 1   // oversold
    if (k > 80 && ind.rsi > 60) return 2   // overbought
    return 0
  }
}
