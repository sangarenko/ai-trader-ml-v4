/** ConnorsRSI2Strategy — Connors RSI(2) mean reversion
 * 77% win rate, ~30% annual return (since 1999)
 * Source: quantifiedstrategies.com/rsi-2-strategy
 *
 * Entry LONG: RSI(2) < 10 AND price > SMA(50) (uptrend filter)
 * Entry SHORT: RSI(2) > 90 AND price < SMA(50)
 * Exit: RSI(2) > 65 OR price > SMA(5)
 */
import { IStrategy, StrategyContext } from './base'
import { Candle } from '../core/types'
import { indicators } from './indicators'

export class ConnorsRSI2Strategy implements IStrategy {
  name = 'ConnorsRSI2'
  description = 'Connors RSI(2): LONG RSI2<10+SMA50 filter, 77% win rate'

  predict(candles: Candle[], idx: number, hasPosition: boolean, _stepsHeld?: number, _ctx?: StrategyContext): number {
    if (idx < 50) return 0

    // RSI(2) — very short period
    const closes2 = candles.slice(idx - 1, idx + 1).map(c => c.close)
    let gains = 0, losses = 0
    for (let i = 1; i < closes2.length; i++) {
      const ch = closes2[i] - closes2[i - 1]
      if (ch > 0) gains += ch; else losses -= ch
    }
    const rsi2 = losses === 0 ? 100 : (gains === 0 ? 0 : 100 - 100 / (1 + gains / losses))

    // SMA(50) — trend filter
    const closes50 = candles.slice(idx - 49, idx + 1).map(c => c.close)
    const sma50 = closes50.reduce((a, b) => a + b, 0) / 50
    const price = candles[idx].close

    // SMA(5) — exit target
    const closes5 = candles.slice(idx - 4, idx + 1).map(c => c.close)
    const sma5 = closes5.reduce((a, b) => a + b, 0) / 5

    if (hasPosition) {
      if (rsi2 > 65 || price > sma5) return 3
      return 0
    }

    if (rsi2 < 10 && price > sma50) return 1   // oversold in uptrend → long
    if (rsi2 > 90 && price < sma50) return 2   // overbought in downtrend → short
    return 0
  }
}
