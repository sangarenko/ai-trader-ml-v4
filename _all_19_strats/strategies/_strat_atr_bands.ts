/** AtrBandsStrategy — ATR Volatility Bands mean reversion
 * 33-year backtest profitable (QuantifiedStrategies)
 *
 * Entry LONG: price < SMA(20) - 2*ATR(20) (below lower band)
 * Entry SHORT: price > SMA(20) + 2*ATR(20) (above upper band)
 * Exit: price returns to SMA(20)
 */
import { IStrategy, StrategyContext } from './base'
import { Candle } from '../core/types'

export class AtrBandsStrategy implements IStrategy {
  name = 'AtrBands'
  description = 'ATR bands mean reversion: 33yr backtest profitable'

  predict(candles: Candle[], idx: number, hasPosition: boolean, _stepsHeld?: number, ctx?: StrategyContext): number {
    if (idx < 21) return 0

    // SMA(20)
    const closes20 = candles.slice(idx - 19, idx + 1).map(c => c.close)
    const sma20 = closes20.reduce((a, b) => a + b, 0) / 20

    // ATR(20)
    let trSum = 0
    for (let i = idx - 19; i <= idx; i++) {
      const tr = Math.max(
        candles[i].high - candles[i].low,
        Math.abs(candles[i].high - candles[i - 1].close),
        Math.abs(candles[i].low - candles[i - 1].close)
      )
      trSum += tr
    }
    const atr = trSum / 20

    const upper = sma20 + 2 * atr
    const lower = sma20 - 2 * atr
    const price = candles[idx].close

    if (hasPosition) {
      const holding = ctx?.holding || 0
      if (holding > 0 && price > sma20) return 3
      if (holding < 0 && price < sma20) return 3
      return 0
    }

    if (price < lower) return 1    // below lower ATR band → long
    if (price > upper) return 2    // above upper → short
    return 0
  }
}
