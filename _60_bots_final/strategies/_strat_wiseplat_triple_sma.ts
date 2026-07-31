/**
 * WiseplatTripleSmaStrategy — WISEPLAT Strategy 04 (177% backtest)
 *
 * Source: https://github.com/WISEPLAT/backtrader_moexalgo/blob/master/StrategyExamplesMoexAlgo/04%20-%20Offline%20Backtest%20Indicators.py
 *
 * Logic:
 *   LONG: SMA9 crosses above SMA14 + SMA20 < SMA14 (trend reversal up from downtrend)
 *   Exit: RSI < 30 (oversold — exit on fear)
 *
 * Long-only strategy. No shorts.
 */
import { IStrategy, StrategyContext } from './base'
import { Candle } from '../core/types'
import { indicators } from './indicators'

export class WiseplatTripleSmaStrategy implements IStrategy {
  name = 'WiseplatTripleSma'
  description = 'WISEPLAT Strategy 04: LONG on SMA9/14 crossover + SMA20<SMA14, exit RSI<30'

  predict(candles: Candle[], idx: number, hasPosition: boolean, _stepsHeld?: number, _ctx?: StrategyContext): number {
    const ind = indicators(candles, idx)
    if (!ind || idx < 20) return 0

    // Compute SMA20
    const closes20 = candles.slice(idx - 19, idx + 1).map(c => c.close)
    const sma20 = closes20.reduce((a: number, b: number) => a + b, 0) / 20

    if (hasPosition) {
      // Exit: RSI < 30 (oversold)
      if (ind.rsi < 30) return 3
      return 0
    }

    // LONG entry: SMA5 > SMA14 (crossover up) + SMA20 < SMA14 (downtrend reversal) + RSI 30-55
    if (ind.sma5 > ind.sma14 * 1.001
        && sma20 < ind.sma14
        && ind.rsi > 30 && ind.rsi < 55) {
      return 1
    }

    return 0
  }
}
