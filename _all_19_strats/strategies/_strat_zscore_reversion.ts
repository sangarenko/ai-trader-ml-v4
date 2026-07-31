/** ZScoreReversionStrategy — Z-Score mean reversion
 * 131% return (AlgoCraft), Sharpe 2.11 (Reddit)
 * Source: medium.com/algocraft
 *
 * Entry LONG: Z-score < -2.0 (price 2 SD below SMA20)
 * Entry SHORT: Z-score > +2.0
 * Exit: Z-score crosses ±0.5
 */
import { IStrategy, StrategyContext } from './base'
import { Candle } from '../core/types'
import { indicators } from './indicators'

export class ZScoreReversionStrategy implements IStrategy {
  name = 'ZScoreReversion'
  description = 'Z-Score reversion: LONG z<-2, SHORT z>+2, exit z=0.5'

  predict(candles: Candle[], idx: number, hasPosition: boolean, _stepsHeld?: number, _ctx?: StrategyContext): number {
    if (idx < 20) return 0

    const closes = candles.slice(idx - 19, idx + 1).map(c => c.close)
    const sma20 = closes.reduce((a, b) => a + b, 0) / 20
    const variance = closes.reduce((s, c) => s + Math.pow(c - sma20, 2), 0) / 20
    const std = Math.sqrt(variance)
    const price = candles[idx].close
    const zscore = std > 0 ? (price - sma20) / std : 0

    if (hasPosition) {
      if (zscore > -0.5 && zscore < 0.5) return 3  // reverted to mean
      return 0
    }

    if (zscore < -2.0) return 1   // 2 SD below mean → long
    if (zscore > 2.0) return 2    // 2 SD above → short
    return 0
  }
}
