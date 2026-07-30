/**
 * MomentumVolumeStrategy — Momentum + Volume confirmation
 *
 * Logic:
 *   LONG: ROC > 2% + volume ratio > 1.5 + RSI > 50 (strong up momentum)
 *   SHORT: ROC < -2% + volume ratio > 1.5 + RSI < 50 (strong down momentum)
 *   Exit: ROC reverses sign (momentum fades)
 */
import { IStrategy, StrategyContext } from './base'
import { Candle } from '../core/types'
import { indicators } from './indicators'

export class MomentumVolumeStrategy implements IStrategy {
  name = 'MomentumVolume'
  description = 'Momentum+Volume: LONG ROC>2%+vol surge, SHORT ROC<-2%+vol surge'

  predict(candles: Candle[], idx: number, hasPosition: boolean, _stepsHeld?: number, ctx?: StrategyContext): number {
    if (idx < 11) return 0

    const ind = indicators(candles, idx)
    if (!ind) return 0

    // Rate of Change (10 periods)
    const roc = (candles[idx].close - candles[idx - 10].close) / candles[idx - 10].close * 100

    // Volume ratio
    const avgVol = candles.slice(idx - 19, idx + 1).reduce((s, c) => s + c.volume, 0) / 20
    const volRatio = avgVol > 0 ? candles[idx].volume / avgVol : 1

    if (hasPosition) {
      const holding = ctx?.holding || 0
      if (holding > 0 && roc < 0) return 3   // exit long when momentum reverses
      if (holding < 0 && roc > 0) return 3   // exit short when momentum reverses
      return 0
    }

    if (roc > 2.0 && volRatio > 1.5 && ind.rsi > 50) return 1
    if (roc < -2.0 && volRatio > 1.5 && ind.rsi < 50) return 2
    return 0
  }
}
