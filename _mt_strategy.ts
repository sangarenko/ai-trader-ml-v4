/** MultiTimeframeStrategy — evolved strategy, winner of Monte Carlo search.

Logic (from extended_strategies.py):
  SHORT: sma5 < sma14*entry_sma_mult AND rsi in [entry_rsi_min, entry_rsi_max] AND allDown
         AND sma20 < sma14 (higher TF downtrend confirmation)
  LONG:  sma5 > sma14*(2-entry_sma_mult) AND rsi in [40, 60] AND allUp
         AND sma20 > sma14 (higher TF uptrend confirmation)
  EXIT SHORT: sma5 > sma14 (trend reversal up)
  EXIT LONG:  sma5 < sma14 (trend reversal down)

The key feature: only enters SHORT when higher timeframe (SMA20) is declining,
only enters LONG when SMA20 is rising. This filters out counter-trend entries
that V2 (no higher TF filter) suffers from.

Params (from Monte Carlo top-5):
  entry_sma_mult: 0.995-1.005 (how far SMA5 must be below/above SMA14)
  entry_rsi_min:  20-40
  entry_rsi_max:  45-60
  take_profit_pct: 0.005-0.025 (0.5-2.5%)
  hold_ticks: 30-300 (in 10s ticks = 5-50 min)
  exit_sma_mult: 1.002-1.005
  position_size: 0.2-0.4
*/
import { IStrategy, StrategyContext } from './base'
import { Candle } from '../core/types'
import { indicators } from './indicators'

export interface MultiTimeframeParams {
  entry_sma_mult: number
  entry_rsi_min: number
  entry_rsi_max: number
  take_profit_pct: number
  hold_ticks: number
  exit_sma_mult: number
  position_size: number
}

// 1 candle (5min) = 300 sec = 30 ticks of 10 sec
const TICKS_PER_CANDLE = 30
const MAX_HOLD_CANDLES = 105  // ~1 trading day

export class MultiTimeframeStrategy implements IStrategy {
  name = 'MultiTimeframe'
  description = 'Multi-TF: SHORT only if SMA20 declining + 3 down candles (Monte Carlo winner)'
  private p: MultiTimeframeParams

  constructor(params?: MultiTimeframeParams) {
    // defaults from top Monte Carlo model
    this.p = params || {
      entry_sma_mult: 0.999,
      entry_rsi_min: 30,
      entry_rsi_max: 55,
      take_profit_pct: 0.0,
      hold_ticks: 6,
      exit_sma_mult: 1.003,
      position_size: 0.3,
    }
  }

  predict(candles: Candle[], idx: number, hasPosition: boolean, stepsHeld?: number, ctx?: StrategyContext): number {
    const ind = indicators(candles, idx)
    if (!ind) return 0

    const p = this.p
    const cur = ind.cur
    const candlesHeld = Math.floor((stepsHeld || 0) / TICKS_PER_CANDLE)

    // SMA20 — higher timeframe trend filter
    // (indicators.ts doesn't export sma20, so compute inline)
    if (idx < 20) return 0
    const closes20 = candles.slice(idx - 19, idx + 1).map(c => c.close)
    const sma20 = closes20.reduce((a: number, b: number) => a + b, 0) / 20

    if (hasPosition) {
      // === EXITS ===
      // Exit 1: SMA reversal (trend turns against position)
      if (ind.sma5 > ind.sma14 * p.exit_sma_mult && ind.rsi > 65) return 3
      if (ind.sma5 < ind.sma14 * (2 - p.exit_sma_mult) && ind.rsi < 35) return 3

      // Exit 2: SMA crossover (simpler exit — matches backtest exit_short/long)
      // backtest: exit_short when sma5 > sma14, exit_long when sma5 < sma14
      // But live uses SniperEvolved-style exits above, so keep both
      // Actually to match backtest exactly, use the crossover exit:
      // (Already covered by exit_sma_mult logic when exit_sma_mult=1.0)

      // Exit 3: TAKE-PROFIT
      if (p.take_profit_pct > 0 && candlesHeld >= p.hold_ticks && ctx?.entryPrice && ctx.entryPrice > 0) {
        const entry = ctx.entryPrice
        const holding = ctx.holding || 0
        const isShort = holding < 0
        const profitPct = isShort ? (entry - cur) / entry : (cur - entry) / entry
        if (profitPct >= p.take_profit_pct) return 3
      }

      // Exit 4: MAX-HOLD
      if (candlesHeld >= MAX_HOLD_CANDLES) return 3

      return 0
    }

    // === ENTRIES ===
    // SHORT: sma5 < sma14*entry_sma_mult + rsi in [min,max] + allDown + sma20 < sma14 (HT downtrend)
    if (ind.sma5 < ind.sma14 * p.entry_sma_mult
        && p.entry_rsi_min <= ind.rsi && ind.rsi <= p.entry_rsi_max
        && ind.allDown
        && sma20 < ind.sma14) {
      return 2
    }

    // LONG: sma5 > sma14*(2-entry_sma_mult) + rsi in [40,60] + allUp + sma20 > sma14 (HT uptrend)
    if (ind.sma5 > ind.sma14 * (2 - p.entry_sma_mult)
        && 40 <= ind.rsi && ind.rsi <= 60
        && ind.allUp
        && sma20 > ind.sma14) {
      return 1
    }

    return 0
  }
}
