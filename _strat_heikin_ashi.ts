/** HeikinAshiTrendStrategy — Heikin-Ashi + SMA(50) trend follow
 * QuantifiedStrategies: max DD 29% vs 52% B&H, risk-adjusted 7.3%
 *
 * Heikin-Ashi smooths price noise. Green HA candle = bullish, red = bearish.
 * Entry LONG: HA green (HA close > HA open) AND HA close > SMA(50)
 * Entry SHORT: HA red AND HA close < SMA(50)
 * Exit: HA opposite color
 */
import { IStrategy, StrategyContext } from './base'
import { Candle } from '../core/types'

export class HeikinAshiTrendStrategy implements IStrategy {
  name = 'HeikinAshiTrend'
  description = 'Heikin-Ashi + SMA50: noise-filtered trend, DD 29% vs 52%'

  predict(candles: Candle[], idx: number, hasPosition: boolean, _stepsHeld?: number, _ctx?: StrategyContext): number {
    if (idx < 51) return 0

    // Compute Heikin-Ashi for current and previous candle
    function ha(candles: Candle[], i: number): { open: number; close: number; high: number; low: number } {
      if (i === 0) return { open: (candles[0].open + candles[0].close) / 2, close: (candles[0].open + candles[0].high + candles[0].low + candles[0].close) / 4, high: candles[0].high, low: candles[0].low }
      const prev = ha(candles, i - 1)
      const haClose = (candles[i].open + candles[i].high + candles[i].low + candles[i].close) / 4
      const haOpen = (prev.open + prev.close) / 2
      const haHigh = Math.max(candles[i].high, haOpen, haClose)
      const haLow = Math.min(candles[i].low, haOpen, haClose)
      return { open: haOpen, close: haClose, high: haHigh, low: haLow }
    }

    const curHA = ha(candles, idx)
    const prevHA = ha(candles, idx - 1)

    // SMA(50) on HA closes
    let haCloseSum = 0
    for (let i = idx - 49; i <= idx; i++) {
      haCloseSum += ha(candles, i).close
    }
    const sma50 = haCloseSum / 50

    const isGreen = curHA.close > curHA.open
    const wasRed = prevHA.close < prevHA.open
    const isRed = curHA.close < curHA.open
    const wasGreen = prevHA.close > prevHA.open

    if (hasPosition) {
      // Exit on HA color flip
      const holding = _ctx?.holding || 0
      if (holding > 0 && isRed) return 3
      if (holding < 0 && isGreen) return 3
      return 0
    }

    // Entry: color change + SMA filter
    if (wasRed && isGreen && curHA.close > sma50) return 1   // red→green above SMA50
    if (wasGreen && isRed && curHA.close < sma50) return 2   // green→red below SMA50
    return 0
  }
}
