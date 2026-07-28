/**
 * RiskManager — minimal safety layer.
 * Only protects: stop-loss 3%, max-hold 24h, hold guard (min hold time).
 * No commission filter, no cooldown — algorithms decide.
 */

import { BotConfig, Position, TradeRecord } from './types'

const TICK_MS = 10_000
export const COMMISSION_RATE = 0.0005

export class RiskManager {
  static filter(
    action: number,
    config: BotConfig,
    holding: number,
    price: number,
    openPos: Position | undefined,
    _history: TradeRecord[],
    _expectedMove: number,
  ): { action: number; reason?: string } {
    const f = config.filters

    // MAX-HOLD — force close positions held too long
    if (openPos && holding !== 0) {
      const maxHoldHours = f.maxHoldHours || 24
      const maxHoldMs = maxHoldHours * 3600 * 1000
      if (Date.now() - openPos.ts > maxHoldMs) {
        return { action: 3, reason: `max-hold ${maxHoldHours}h` }
      }
    }

    // HOLD GUARD — don't close before holdTicks (prevents flip-flopping)
    // This is essential — without it bots open/close every tick on commission
    if (openPos && (action === 2 || action === 3) && holding !== 0) {
      const ticksHeld = Math.floor((Date.now() - openPos.ts) / TICK_MS)
      if (ticksHeld < f.holdTicks) {
        return { action: 0, reason: `hold ${ticksHeld}/${f.holdTicks}` }
      }
    }

    // STOP-LOSS — close if loss exceeds 3%
    if (openPos && holding !== 0 && action !== 0) {
      let entryPerShare = openPos.entryPrice
      if (openPos.qty > 0 && entryPerShare > price * 3) entryPerShare = entryPerShare / openPos.qty
      const priceRatio = entryPerShare > 0 ? price / entryPerShare : 1
      if (priceRatio > 0.3 && priceRatio < 3) {
        const grossPnl = (price - entryPerShare) * Math.abs(holding)
        const positionValue = entryPerShare * Math.abs(holding)
        const lossPct = grossPnl < 0 ? Math.abs(grossPnl) / positionValue : 0
        if (lossPct >= 0.03) {
          return { action: 3, reason: `stop-loss ${lossPct*100}%` }
        }
      }
    }

    return { action }
  }
}
