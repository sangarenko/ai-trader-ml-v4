/** DualThrustStrategy — Dual Thrust breakout (Michael Chalek)
 * Source: github.com/je-suis-tm/quant-trading
 *
 * Range = max(HH - LC, HC - LL) over N=5 days
 * K1 = K2 = 0.5
 * Entry LONG: price > Open + K1 * Range
 * Entry SHORT: price < Open - K2 * Range
 * Exit: end of day or opposite threshold
 */
import { IStrategy, StrategyContext } from './base'
import { Candle } from '../core/types'

export class DualThrustStrategy implements IStrategy {
  name = 'DualThrust'
  description = 'Dual Thrust breakout: classic intraday system'

  predict(candles: Candle[], idx: number, hasPosition: boolean, _stepsHeld?: number, _ctx?: StrategyContext): number {
    if (idx < 5) return 0

    // Compute range over last 5 candles
    const lookback = 5
    let hh = -Infinity, lc = Infinity, hc = -Infinity, ll = Infinity
    for (let i = idx - lookback + 1; i <= idx; i++) {
      if (candles[i].high > hh) hh = candles[i].high
      if (candles[i].close < lc) lc = candles[i].close
      if (candles[i].close > hc) hc = candles[i].close
      if (candles[i].low < ll) ll = candles[i].low
    }
    const range = Math.max(hh - lc, hc - ll)
    const k = 0.5
    const openPrice = candles[idx].open
    const price = candles[idx].close

    if (hasPosition) {
      // Exit on opposite threshold
      if (price < openPrice - k * range) return 3  // was long, price broke down
      if (price > openPrice + k * range) return 3  // was short, price broke up
      return 0
    }

    if (price > openPrice + k * range) return 1    // breakout up
    if (price < openPrice - k * range) return 2    // breakout down
    return 0
  }
}
