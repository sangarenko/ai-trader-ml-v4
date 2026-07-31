/**
 * TurtleDonchianStrategy — Turtle Trading (Richard Dennis, 1980s)
 *
 * Source: Classic Donchian channel breakout
 *
 * Logic:
 *   LONG: price breaks above 20-period high + volume confirmation
 *   SHORT: price breaks below 20-period low + volume confirmation
 *   Exit LONG: price breaks below 20-period low (opposite)
 *   Exit SHORT: price breaks above 20-period high
 *
 * Trend-following. Catches big moves, many small losses.
 */
import { IStrategy, StrategyContext } from './base'
import { Candle } from '../core/types'
import { indicators } from './indicators'

export class TurtleDonchianStrategy implements IStrategy {
  name = 'TurtleDonchian'
  description = 'Turtle Trading: breakout 20-period high/low, exit on opposite breakout'

  predict(candles: Candle[], idx: number, hasPosition: boolean, _stepsHeld?: number, ctx?: StrategyContext): number {
    if (idx < 20) return 0

    // Donchian channel (20 periods)
    const highs = candles.slice(idx - 19, idx + 1).map(c => c.high)
    const lows = candles.slice(idx - 19, idx + 1).map(c => c.low)
    const donchianUpper = Math.max(...highs)
    const donchianLower = Math.min(...lows)

    const price = candles[idx].close
    const prevClose = candles[idx - 1].close
    const avgVol = candles.slice(idx - 19, idx + 1).reduce((s, c) => s + c.volume, 0) / 20
    const volRatio = avgVol > 0 ? candles[idx].volume / avgVol : 1

    if (hasPosition) {
      const holding = ctx?.holding || 0
      if (holding > 0 && price < donchianLower) return 3  // exit long on new low
      if (holding < 0 && price > donchianUpper) return 3  // exit short on new high
      return 0
    }

    // LONG: break above upper channel + volume
    if (price > donchianUpper * 0.999 && volRatio > 1.0) return 1
    // SHORT: break below lower channel + volume
    if (price < donchianLower * 1.001 && volRatio > 1.0) return 2

    return 0
  }
}
