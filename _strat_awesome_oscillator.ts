/** AwesomeOscillatorStrategy — AO + MACD confirmation
 * Source: github.com/je-suis-tm/quant-trading
 *
 * AO = SMA(median, 5) - SMA(median, 34)
 * Entry LONG: AO crosses above 0 AND MACD line > signal
 * Entry SHORT: AO crosses below 0 AND MACD line < signal
 * Exit: AO crosses opposite
 */
import { IStrategy, StrategyContext } from './base'
import { Candle } from '../core/types'
import { indicators } from './indicators'

function ema(values: number[], period: number): number {
  const k = 2 / (period + 1)
  let e = values[0]
  for (let i = 1; i < values.length; i++) e = values[i] * k + e * (1 - k)
  return e
}

export class AwesomeOscillatorStrategy implements IStrategy {
  name = 'AwesomeOscillator'
  description = 'AO+MACD: momentum confirmation, fewer whipsaws'

  predict(candles: Candle[], idx: number, hasPosition: boolean, _stepsHeld?: number, ctx?: StrategyContext): number {
    if (idx < 34) return 0

    // Median prices
    const medians = candles.slice(idx - 33, idx + 1).map(c => (c.high + c.low) / 2)

    // AO = SMA5(median) - SMA34(median)
    const sma5 = medians.slice(-5).reduce((a, b) => a + b, 0) / 5
    const sma34 = medians.reduce((a, b) => a + b, 0) / 34
    const ao = sma5 - sma34

    // Previous AO
    const prevMedians = candles.slice(idx - 34, idx).map(c => (c.high + c.low) / 2)
    const prevSma5 = prevMedians.slice(-5).reduce((a, b) => a + b, 0) / 5
    const prevSma34 = prevMedians.reduce((a, b) => a + b, 0) / 34
    const prevAo = prevSma5 - prevSma34

    // MACD (simplified)
    const closes = candles.slice(idx - 25, idx + 1).map(c => c.close)
    const macdLine = ema(closes.slice(-12), 12) - ema(closes, 26)
    const macdSignal = macdLine * 0.8
    const macdHist = macdLine - macdSignal

    if (hasPosition) {
      // Exit on AO cross
      const holding = ctx?.holding || 0
      if (holding > 0 && ao < 0) return 3
      if (holding < 0 && ao > 0) return 3
      return 0
    }

    // Entry: AO cross + MACD confirmation
    if (prevAo < 0 && ao > 0 && macdHist > 0) return 1   // AO cross up + MACD bullish
    if (prevAo > 0 && ao < 0 && macdHist < 0) return 2   // AO cross down + MACD bearish
    return 0
  }
}
