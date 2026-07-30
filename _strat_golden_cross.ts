/** GoldenCrossStrategy — 50/200 SMA crossover
 * QuantifiedStrategies: $100k → $7.2M over 66 years
 *
 * Very slow signals (1-2 trades/year), best as regime filter
 * Entry LONG: SMA(50) crosses ABOVE SMA(200)
 * Exit LONG: SMA(50) crosses BELOW SMA(200) (death cross)
 */
import { IStrategy, StrategyContext } from './base'
import { Candle } from '../core/types'

export class GoldenCrossStrategy implements IStrategy {
  name = 'GoldenCross'
  description = 'Golden Cross 50/200 SMA: $100k→$7.2M over 66 years'

  predict(candles: Candle[], idx: number, hasPosition: boolean, _stepsHeld?: number, _ctx?: StrategyContext): number {
    if (idx < 200) return 0

    const closes50 = candles.slice(idx - 49, idx + 1).map(c => c.close)
    const closes200 = candles.slice(idx - 199, idx + 1).map(c => c.close)
    const sma50 = closes50.reduce((a, b) => a + b, 0) / 50
    const sma200 = closes200.reduce((a, b) => a + b, 0) / 200

    const prevCloses50 = candles.slice(idx - 50, idx).map(c => c.close)
    const prevCloses200 = candles.slice(idx - 200, idx).map(c => c.close)
    const prevSma50 = prevCloses50.reduce((a, b) => a + b, 0) / 50
    const prevSma200 = prevCloses200.reduce((a, b) => a + b, 0) / 200

    const goldenCross = prevSma50 <= prevSma200 && sma50 > sma200
    const deathCross = prevSma50 >= prevSma200 && sma50 < sma200

    if (hasPosition) {
      if (deathCross) return 3  // exit on death cross
      return 0
    }

    if (goldenCross) return 1  // golden cross → long
    return 0
  }
}
