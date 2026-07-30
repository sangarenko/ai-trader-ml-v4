/**
 * VwapReversionStrategy — VWAP intraday mean-reversion
 *
 * Logic:
 *   LONG: price < VWAP * 0.995 + RSI < 40 (below VWAP, oversold)
 *   SHORT: price > VWAP * 1.005 + RSI > 60 (above VWAP, overbought)
 *   Exit: price returns to VWAP
 *
 * VWAP computed over last 20 candles.
 */
import { IStrategy, StrategyContext } from './base'
import { Candle } from '../core/types'
import { indicators } from './indicators'

export class VwapReversionStrategy implements IStrategy {
  name = 'VwapReversion'
  description = 'VWAP reversion: LONG below VWAP+RSI<40, SHORT above VWAP+RSI>60'

  predict(candles: Candle[], idx: number, hasPosition: boolean, _stepsHeld?: number, ctx?: StrategyContext): number {
    const ind = indicators(candles, idx)
    if (!ind || idx < 20) return 0

    // VWAP over last 20 candles
    let pvSum = 0, volSum = 0
    for (let i = idx - 19; i <= idx; i++) {
      const typical = (candles[i].high + candles[i].low + candles[i].close) / 3
      pvSum += typical * candles[i].volume
      volSum += candles[i].volume
    }
    const vwap = volSum > 0 ? pvSum / volSum : ind.cur
    const price = ind.cur

    if (hasPosition) {
      // Exit at VWAP
      const holding = ctx?.holding || 0
      if (holding > 0 && price > vwap) return 3
      if (holding < 0 && price < vwap) return 3
      return 0
    }

    if (price < vwap * 0.995 && ind.rsi < 40) return 1
    if (price > vwap * 1.005 && ind.rsi > 60) return 2
    return 0
  }
}
