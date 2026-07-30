/** OpeningRangeBreakoutStrategy — ORB (5-minute)
 * ConcretumGroup: Sharpe 2.81 on SPY
 *
 * Entry LONG: price > High of first 5-min candle of the day
 * Entry SHORT: price < Low of first 5-min candle
 * Exit: stop = opposite end of range, TP = 1.5x range, or EOD
 *
 * NOTE: For MOEX, opening range = 10:00-10:05 MSK
 */
import { IStrategy, StrategyContext } from './base'
import { Candle } from '../core/types'

export class OpeningRangeBreakoutStrategy implements IStrategy {
  name = 'OpeningRangeBreakout'
  description = 'ORB 5-min: Sharpe 2.81, classic intraday breakout'

  predict(candles: Candle[], idx: number, hasPosition: boolean, _stepsHeld?: number, ctx?: StrategyContext): number {
    if (idx < 2) return 0

    // Find today's first candle (simplified: look for gap in timestamps)
    // Each candle is 5min. Trading day starts at 10:00 MSK = 07:00 UTC
    // Find the first candle of the current day
    let dayStart = idx
    const curDate = new Date(candles[idx].time).toISOString().slice(0, 10)
    while (dayStart > 0) {
      const prevDate = new Date(candles[dayStart - 1].time).toISOString().slice(0, 10)
      if (prevDate !== curDate) break
      dayStart--
    }

    // Need at least 2 candles today (opening range = first candle)
    if (idx - dayStart < 1) return 0

    // Opening range = first candle of the day
    const orHigh = candles[dayStart].high
    const orLow = candles[dayStart].low
    const price = candles[idx].close

    if (hasPosition) {
      // Exit: price returns to opening range
      const holding = ctx?.holding || 0
      if (holding > 0 && price < orLow) return 3   // stop loss
      if (holding < 0 && price > orHigh) return 3  // stop loss
      // Take profit at 1.5x range
      const range = orHigh - orLow
      if (holding > 0 && price > orHigh + 1.5 * range) return 3
      if (holding < 0 && price < orLow - 1.5 * range) return 3
      return 0
    }

    // Entry: breakout above/below opening range (after first candle)
    if (idx > dayStart) {
      if (price > orHigh) return 1   // breakout up
      if (price < orLow) return 2   // breakout down
    }
    return 0
  }
}
