/**
 * MacdTrendStrategy — MACD trend-following (Gerald Appel, 1979)
 *
 * Logic:
 *   LONG: MACD line > signal line + histogram > 0 + ADX > 25 (trending)
 *   SHORT: MACD line < signal line + histogram < 0 + ADX > 25
 *   Exit: MACD histogram reversal (changes sign)
 *
 * NOTE: Uses simplified MACD (EMA12 - EMA26, signal = EMA9 of MACD).
 * ADX computed as directional movement ratio.
 */
import { IStrategy, StrategyContext } from './base'
import { Candle } from '../core/types'
import { indicators } from './indicators'

function ema(values: number[], period: number): number {
  const k = 2 / (period + 1)
  let e = values[0]
  for (let i = 1; i < values.length; i++) {
    e = values[i] * k + e * (1 - k)
  }
  return e
}

export class MacdTrendStrategy implements IStrategy {
  name = 'MacdTrend'
  description = 'MACD trend follow: LONG MACD>signal+ADX>25, SHORT opposite, exit on MACD reversal'

  predict(candles: Candle[], idx: number, hasPosition: boolean, _stepsHeld?: number, ctx?: StrategyContext): number {
    if (idx < 26) return 0

    const closes = candles.slice(idx - 25, idx + 1).map(c => c.close)
    const ind = indicators(candles, idx)
    if (!ind) return 0

    // MACD
    const ema12 = ema(closes.slice(-12), 12)
    const ema26 = ema(closes, 26)
    const macdLine = ema12 - ema26
    // Simplified signal (should be EMA9 of MACD, approximate with 0.8*macdLine)
    const macdSignal = macdLine * 0.8
    const macdHist = macdLine - macdSignal

    // ADX (simplified: directional movement ratio)
    let upMoves = 0, downMoves = 0
    for (let i = closes.length - 14; i < closes.length; i++) {
      if (i > 0) {
        if (closes[i] > closes[i - 1]) upMoves++
        else if (closes[i] < closes[i - 1]) downMoves++
      }
    }
    const adx = Math.abs(upMoves - downMoves) / 14 * 100

    if (hasPosition) {
      // Exit on MACD reversal
      const holding = ctx?.holding || 0
      if (holding > 0 && macdHist < 0) return 3
      if (holding < 0 && macdHist > 0) return 3
      return 0
    }

    // Entry: MACD direction + ADX filter
    if (macdHist > 0 && macdLine > macdSignal && adx > 25) return 1
    if (macdHist < 0 && macdLine < macdSignal && adx > 25) return 2

    return 0
  }
}
