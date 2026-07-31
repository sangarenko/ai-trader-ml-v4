/** BollingerSqueezeStrategy — BB squeeze breakout
 * Profitable in 58% of backtests, R:R 2:1 to 4:1
 * Source: quantifiedstrategies.com
 *
 * Squeeze: BB bandwidth at 6-month low (volatility compression)
 * Entry LONG: squeeze + price closes above upper BB
 * Entry SHORT: squeeze + price closes below lower BB
 * Exit: price crosses middle BB (SMA20)
 */
import { IStrategy, StrategyContext } from './base'
import { Candle } from '../core/types'
import { indicators } from './indicators'

export class BollingerSqueezeStrategy implements IStrategy {
  name = 'BollingerSqueeze'
  description = 'BB squeeze breakout: low volatility then breakout, R:R 2:1+'

  predict(candles: Candle[], idx: number, hasPosition: boolean, _stepsHeld?: number, _ctx?: StrategyContext): number {
    if (idx < 125) return 0  // need 6 months (~125 candles on daily, but we use 5min)

    const ind = indicators(candles, idx)
    if (!ind) return 0

    // Compute BB(20, 2)
    const closes20 = candles.slice(idx - 19, idx + 1).map(c => c.close)
    const sma20 = closes20.reduce((a, b) => a + b, 0) / 20
    const variance = closes20.reduce((s, c) => s + Math.pow(c - sma20, 2), 0) / 20
    const std = Math.sqrt(variance)
    const bbUpper = sma20 + 2 * std
    const bbLower = sma20 - 2 * std
    const bandwidth = std > 0 ? (bbUpper - bbLower) / sma20 : 0
    const price = candles[idx].close

    // Check if bandwidth is at 125-candle low (squeeze)
    let minBandwidth = Infinity
    for (let i = idx - 124; i <= idx; i++) {
      const c20 = candles.slice(i - 19, i + 1).map(c => c.close)
      const s20 = c20.reduce((a, b) => a + b, 0) / 20
      const v = c20.reduce((s, c) => s + Math.pow(c - s20, 2), 0) / 20
      const sd = Math.sqrt(v)
      const bw = sd > 0 && s20 > 0 ? (s20 + 2 * sd - (s20 - 2 * sd)) / s20 : 0
      if (bw < minBandwidth) minBandwidth = bw
    }
    const isSqueeze = bandwidth <= minBandwidth * 1.1  // within 10% of min

    if (hasPosition) {
      // Exit at middle BB
      const holding = _ctx?.holding || 0
      if (holding > 0 && price < sma20) return 3
      if (holding < 0 && price > sma20) return 3
      return 0
    }

    if (isSqueeze && price > bbUpper) return 1   // squeeze breakout up
    if (isSqueeze && price < bbLower) return 2   // squeeze breakout down
    return 0
  }
}
