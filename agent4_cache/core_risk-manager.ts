$ cat /opt/ai-trader/src/core/risk-manager.ts
--- rc=0 ---
/**
 * RiskManager — filters trades before execution.
 * Prevents overtrading, commission death, and manages position hold time.
 */

import { BotConfig, Position, TradeRecord } from './types'

const TICK_MS = 10_000

export class RiskManager {
  /**
   * Check if a trade should be allowed based on risk filters.
   * Returns modified action (0 = block, original = allow).
   */
  static filter(
    action: number,
    config: BotConfig,
    holding: number,
    price: number,
    openPos: Position | undefined,
    history: TradeRecord[],
    expectedMove: number,
  ): { action: number; reason?: string } {
    const f = config.filters

    // Hold guard — don't close position before holdTicks
    if (openPos && (action === 2 || action === 3) && holding !== 0) {
      const ticksHeld = Math.floor((Date.now() - openPos.ts) / TICK_MS)
      if (ticksHeld < f.holdTicks) {
        return { action: 0, reason: `hold ${ticksHeld}/${f.holdTicks}` }
      }
    }

    // Commission filter — don't open if expected gross < commFilterMult × commission
    // FIX (B12): was hardcoded 10000 — now uses actual holding value for accurate calc
    if ((action === 1 || action === 2) && holding === 0) {
      const size = Math.abs(holding) * price || 10000 * config.positionSize // actual position value
      const roundTripComm = size * 0.0005 * 2
      if (expectedMove * size < roundTripComm * f.commFilterMult) {
        return { action: 0, reason: `skip-open: expGross < comm×${f.commFilterMult}` }
      }
    }

    // Commission filter — don't close if gross < commission (with stop-loss at 3%)
    if (openPos && (action === 2 || action === 3) && holding !== 0) {
      // FIX (B11): removed dead heuristic `if (entryPerShare > price * 3) entryPerShare /= qty`
      // — entryPrice is always per-share (set from exec_price at entry), never total.
      const entryPerShare = openPos.entryPrice
      const priceRatio = entryPerShare > 0 ? price / entryPerShare : 1
      if (priceRatio > 0.3 && priceRatio < 3) {
        const grossPnl = (price - entryPerShare) * Math.abs(holding)
        const positionValue = entryPerShare * Math.abs(holding)
        const roundTripComm = positionValue * 0.0005 * 2
        const lossPct = grossPnl < 0 ? Math.abs(grossPnl) / positionValue : 0
        if (grossPnl < roundTripComm * f.commFilterMult && lossPct < 0.03) {
          return { action: 0, reason: `skip-close: gross < comm` }
        }
      }
    }

    // Rate limit — max trades per hour
    const now = Date.now()
    const recentTrades = history.filter(h => now - h.ts < 3600000).length
    if (recentTrades >= f.maxTradesPerHour && (action === 1 || action === 2)) {
      return { action: 0, reason: `rate-limit: ${recentTrades}/${f.maxTradesPerHour}/hour` }
    }

    // Cooldown — wait after closing before opening new
    if ((action === 1 || action === 2) && holding === 0 && history.length > 0) {
      const lastTrade = history[0]
      const ticksSinceLast = Math.floor((now - lastTrade.ts) / TICK_MS)
      const wasClose = lastTrade.side === 'SELL' || lastTrade.side === 'CLOSE_SHORT' ||
                       (lastTrade.side === 'BUY' && lastTrade.pnl !== 0)
      if (wasClose && ticksSinceLast < f.cooldownTicks) {
        return { action: 0, reason: `cooldown ${ticksSinceLast}/${f.cooldownTicks}` }
      }
    }

    return { action }
  }
}


