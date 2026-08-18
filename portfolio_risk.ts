/**
 * portfolio_risk.ts — Cross-bot portfolio risk manager.
 *
 * Tracks open positions across multiple bots, enforces per-ticker / per-day /
 * drawdown limits, and computes portfolio Value-at-Risk using a correlation
 * matrix.
 *
 * Integration points:
 *   - Each bot calls `canOpenPosition()` BEFORE placing an order. If allowed,
 *     calls `registerPosition()` immediately after the order fills.
 *   - When a bot closes a position, calls `closePosition()`.
 *   - On each tick / bar: `getDailyPnL()` and `shouldStopForDay()` to gate
 *     new entries.
 *   - At startup or daily-rollover (00:00 MSK): `resetDailyCounters()`.
 *
 * Limits enforced:
 *   - Max 3 concurrent bots in the same ticker (configurable)
 *   - Max 2% daily loss → stop for the day (no new positions)
 *   - Max 5% drawdown → reduce position size by 50% (size multiplier exposed
 *     via `getPositionSizeMultiplier()`)
 *
 * VaR: 95% 1-day VaR = 1.645 * portfolio_sigma * portfolio_value
 *   where portfolio_sigma = sqrt(w' Σ w)
 *   and Σ is the correlation matrix × per-ticker daily volatilities.
 *
 * Author: Agent V7-3 (ts-inference-portfolio)
 */
import * as fs from 'fs'
import * as path from 'path'

// ─── Types ────────────────────────────────────────────────────────────────────

export interface Position {
  botName: string
  ticker: string
  side: 'long' | 'short'
  qty: number
  entryPrice: number
  /** Open timestamp (ms). Used for daily-PnL reset. */
  openedAt: number
  /** Current mark-to-market P&L (in account currency, e.g. RUB). */
  currentPnL?: number
}

export interface CanOpenResult {
  allowed: boolean
  reason: string
  /** Recommended position-size multiplier (0.5 if in drawdown-reduction, else 1.0). */
  sizeMultiplier: number
}

export interface CorrelationMatrix {
  /** Map ticker → index into the matrix. */
  index: Record<string, number>
  /** Symmetric correlation matrix [N][N], values in [-1, 1]. */
  matrix: number[][]
  /** Per-ticker daily volatility (e.g. 0.015 for 1.5% daily stdev). */
  volatilities: Record<string, number>
}

export interface PortfolioRiskConfig {
  maxConcurrentSameTicker: number
  maxDailyLossPct: number
  maxDrawdownPct: number
  /** Account value in account currency (for VaR + drawdown calc). */
  accountValue: number
  /** Daily VaR confidence level (0.95 → 1.645 sigma; 0.99 → 2.326 sigma). */
  varConfidence: number
  /** Path to optional correlation matrix JSON file (loaded on init). */
  correlationMatrixPath?: string
}

// ─── Default config ───────────────────────────────────────────────────────────

const DEFAULT_CONFIG: PortfolioRiskConfig = {
  maxConcurrentSameTicker: 3,
  maxDailyLossPct: 0.02,        // -2% → stop for day
  maxDrawdownPct: 0.05,         // -5% → reduce position size
  accountValue: 1_000_000,      // default 1M RUB
  varConfidence: 0.95,
}

// z-scores for common confidence levels
const Z_SCORES: Record<number, number> = {
  0.90: 1.282,
  0.95: 1.645,
  0.99: 2.326,
  0.999: 3.090,
}

// ─── PortfolioRiskManager ──────────────────────────────────────────────────────

export class PortfolioRiskManager {
  /** Open positions keyed by botName (one position per bot). */
  private positions: Map<string, Position> = new Map()

  /** Cross-ticker correlation matrix (optional). */
  private correlationMatrix: CorrelationMatrix = { index: {}, matrix: [], volatilities: {} }

  /** Configuration. */
  private config: PortfolioRiskConfig

  /** Daily P&L accumulator (resets at 00:00 MSK). */
  private dailyPnL: number = 0
  private dailyPnLDate: string = ''   // YYYY-MM-DD MSK

  /** Peak account value (for drawdown calc). */
  private peakAccountValue: number

  /** Whether trading is halted for the day. */
  private haltedForDay: boolean = false
  private haltReason: string = ''

  /** Position-size multiplier when in drawdown-reduction mode. */
  private sizeMultiplier: number = 1.0

  /** Lifetime realized P&L (running total, never resets). */
  private cumulativePnL: number = 0

  constructor(config?: Partial<PortfolioRiskConfig>) {
    this.config = { ...DEFAULT_CONFIG, ...config }
    this.peakAccountValue = this.config.accountValue

    // Auto-load correlation matrix if a path was provided
    if (this.config.correlationMatrixPath) {
      this.loadCorrelationMatrix(this.config.correlationMatrixPath)
    }
  }

  // ─── Position gating ───────────────────────────────────────────────────────

  /**
   * Check whether a bot can open a new position.
   *
   * Returns { allowed, reason, sizeMultiplier }:
   *   - allowed = false if any limit is breached (with reason explaining why)
   *   - sizeMultiplier = 0.5 in drawdown-reduction mode, else 1.0
   *
   * Limits enforced (in order):
   *   1. Halt-for-day (daily loss exceeded)
   *   2. Max concurrent same-ticker positions
   *   3. Drawdown size-reduction multiplier
   */
  canOpenPosition(
    botName: string,
    ticker: string,
    side: 'long' | 'short',
  ): CanOpenResult {
    // 1. Halt-for-day check
    if (this.haltedForDay) {
      return {
        allowed: false,
        reason: `halted for day: ${this.haltReason}`,
        sizeMultiplier: 0,
      }
    }

    // Already has an open position from this bot — don't allow stacking
    if (this.positions.has(botName)) {
      const existing = this.positions.get(botName)!
      if (existing.ticker === ticker && existing.side === side) {
        return {
          allowed: false,
          reason: `bot ${botName} already has a ${side} ${ticker} position`,
          sizeMultiplier: 0,
        }
      }
      // Allow flipping — bot will close existing first
    }

    // 2. Max concurrent same-ticker positions
    const sameTickerCount = this.countSameTicker(ticker, botName)
    if (sameTickerCount >= this.config.maxConcurrentSameTicker) {
      return {
        allowed: false,
        reason: `max ${this.config.maxConcurrentSameTicker} bots already in ${ticker}`,
        sizeMultiplier: 0,
      }
    }

    // 3. Drawdown reduction multiplier (don't block, just halve size)
    return {
      allowed: true,
      reason: 'OK',
      sizeMultiplier: this.sizeMultiplier,
    }
  }

  /** Register a new position after the order fills. Overwrites any existing one for this bot. */
  registerPosition(
    botName: string,
    ticker: string,
    side: 'long' | 'short',
    qty: number,
    entryPrice: number,
    openedAt: number = Date.now(),
  ): void {
    const pos: Position = {
      botName, ticker, side, qty, entryPrice, openedAt,
    }
    this.positions.set(botName, pos)
  }

  /** Close a position and realize P&L. Optional closePrice for P&L calculation. */
  closePosition(botName: string, closePrice?: number, closeQty?: number): number {
    const pos = this.positions.get(botName)
    if (!pos) return 0

    let pnl = 0
    if (closePrice !== undefined) {
      const sign = pos.side === 'long' ? 1 : -1
      const qty = closeQty !== undefined ? Math.min(closeQty, pos.qty) : pos.qty
      pnl = sign * (closePrice - pos.entryPrice) * qty
    }

    this.dailyPnL += pnl
    this.cumulativePnL += pnl

    // Update peak account value (for drawdown calc)
    const accountValue = this.config.accountValue + this.cumulativePnL
    if (accountValue > this.peakAccountValue) {
      this.peakAccountValue = accountValue
    }

    // Update drawdown multiplier
    this.updateDrawdownState(accountValue)

    // Remove position from map (if fully closed)
    if (closeQty === undefined || closeQty >= pos.qty) {
      this.positions.delete(botName)
    } else {
      // Partial close — reduce qty
      this.positions.set(botName, { ...pos, qty: pos.qty - closeQty })
    }

    // Check if we've hit the daily loss limit
    this.checkDailyLossLimit()

    return pnl
  }

  /** Mark an open position's current P&L (does not close it). */
  markToMarket(botName: string, currentPrice: number): void {
    const pos = this.positions.get(botName)
    if (!pos) return
    const sign = pos.side === 'long' ? 1 : -1
    pos.currentPnL = sign * (currentPrice - pos.entryPrice) * pos.qty
  }

  // ─── Portfolio analytics ───────────────────────────────────────────────────

  /** Count how many OTHER bots are currently in `ticker` (excluding `excludeBot`). */
  countSameTicker(ticker: string, excludeBot?: string): number {
    let count = 0
    for (const [name, pos] of this.positions) {
      if (name === excludeBot) continue
      if (pos.ticker === ticker) count++
    }
    return count
  }

  /** Get all open positions as an array. */
  getOpenPositions(): Position[] {
    return Array.from(this.positions.values())
  }

  /** Number of currently open positions (across all bots). */
  getOpenPositionsCount(): number {
    return this.positions.size
  }

  /** Get the position-size multiplier (1.0 normal, 0.5 in drawdown-reduction). */
  getPositionSizeMultiplier(): number {
    return this.sizeMultiplier
  }

  /** Get the realized daily P&L (resets at 00:00 MSK). */
  getDailyPnL(): number {
    this.maybeResetDailyCounters()
    return this.dailyPnL
  }

  /** Get the cumulative lifetime realized P&L (never resets). */
  getCumulativePnL(): number {
    return this.cumulativePnL
  }

  /** Get total unrealized P&L across all open positions. */
  getUnrealizedPnL(): number {
    let total = 0
    for (const pos of this.positions.values()) {
      if (pos.currentPnL !== undefined) total += pos.currentPnL
    }
    return total
  }

  /** Whether trading is halted for the day (after hitting daily-loss limit). */
  shouldStopForDay(): boolean {
    this.maybeResetDailyCounters()
    this.checkDailyLossLimit()
    return this.haltedForDay
  }

  /** Current drawdown from peak, as a fraction (0 = no drawdown, 0.05 = -5%). */
  getDrawdownPct(): number {
    const accountValue = this.config.accountValue + this.cumulativePnL
    if (this.peakAccountValue <= 0) return 0
    const dd = (accountValue - this.peakAccountValue) / this.peakAccountValue
    return Math.min(0, dd)
  }

  /**
   * Compute 95% 1-day Value-at-Risk (VaR) for the open portfolio.
   *
   * Uses the formula:
   *   VaR = z * sqrt(w' Σ w)
   *
   * where w is the per-ticker position notional vector, Σ = D * Corr * D
   * (D = diag of per-ticker daily volatilities).
   *
   * Returns VaR as a fraction of account value (e.g. 0.025 = 2.5% of account
   * at risk over 1 day at the configured confidence level).
   *
   * If no correlation matrix is loaded, falls back to a simple sum-of-notional
   * × avg-volatility estimate (ignoring diversification).
   */
  getPortfolioVaR(): number {
    if (this.positions.size === 0) return 0
    const z = Z_SCORES[this.config.varConfidence] ?? 1.645

    // Aggregate net notional per ticker (longs +, shorts -)
    const notionalByTicker = new Map<string, number>()
    let totalNotional = 0
    for (const pos of this.positions.values()) {
      const sign = pos.side === 'long' ? 1 : -1
      const notional = sign * pos.qty * pos.entryPrice
      notionalByTicker.set(
        pos.ticker,
        (notionalByTicker.get(pos.ticker) ?? 0) + notional,
      )
      totalNotional += Math.abs(pos.qty * pos.entryPrice)
    }

    if (totalNotional === 0) return 0

    // ── Case 1: have correlation matrix → use full w' Σ w formula ────────
    if (this.correlationMatrix.matrix.length > 0) {
      const tickers = Array.from(notionalByTicker.keys())
      const idx = this.correlationMatrix.index
      // Build vector of net notionals aligned with correlation matrix indices
      // (any ticker not in the matrix → treat as zero correlation, use its own vol)
      const w: number[] = []
      const vols: number[] = []
      const corrIndices: number[] = []
      for (const ticker of tickers) {
        const notional = notionalByTicker.get(ticker)!
        const vol = this.correlationMatrix.volatilities[ticker] ?? 0.015  // default 1.5%
        w.push(notional)
        vols.push(vol)
        corrIndices.push(idx[ticker] ?? -1)
      }

      // Σ_ij = vol_i * vol_j * corr_ij
      // portfolio_var = w' Σ w = sum_ij w_i * w_j * vol_i * vol_j * corr_ij
      let portfolioVar = 0
      for (let i = 0; i < w.length; i++) {
        for (let j = 0; j < w.length; j++) {
          let corr: number
          if (corrIndices[i] < 0 || corrIndices[j] < 0) {
            // Ticker not in correlation matrix — assume zero correlation with others
            corr = i === j ? 1 : 0
          } else {
            corr = this.correlationMatrix.matrix[corrIndices[i]][corrIndices[j]]
          }
          portfolioVar += w[i] * w[j] * vols[i] * vols[j] * corr
        }
      }
      const sigma = Math.sqrt(Math.max(portfolioVar, 0))
      const varValue = z * sigma
      return varValue / this.config.accountValue
    }

    // ── Case 2: no correlation matrix → simple sum-of-volatilities estimate ──
    // Conservatively assume zero diversification (worst case).
    let totalRisk = 0
    for (const [ticker, notional] of notionalByTicker) {
      const vol = this.correlationMatrix.volatilities[ticker] ?? 0.015
      totalRisk += Math.abs(notional) * vol
    }
    return z * totalRisk / this.config.accountValue
  }

  // ─── Correlation matrix loading ────────────────────────────────────────────

  /**
   * Load a correlation matrix from a JSON file.
   * Expected format:
   *   {
   *     "tickers": ["SBER", "GAZP", ...],
   *     "matrix":  [[1.0, 0.65, ...], [0.65, 1.0, ...], ...],
   *     "volatilities": { "SBER": 0.015, "GAZP": 0.018, ... }
   *   }
   */
  loadCorrelationMatrix(filePath: string): boolean {
    try {
      if (!fs.existsSync(filePath)) {
        console.warn(`[PortfolioRisk] correlation matrix file not found: ${filePath}`)
        return false
      }
      const raw = fs.readFileSync(filePath, 'utf-8')
      const data = JSON.parse(raw)
      const tickers: string[] = data.tickers || []
      const matrix: number[][] = data.matrix || []
      const vols: Record<string, number> = data.volatilities || {}

      const index: Record<string, number> = {}
      tickers.forEach((t, i) => { index[t] = i })

      this.correlationMatrix = { index, matrix, volatilities: vols }
      console.log(`[PortfolioRisk] loaded correlation matrix: ${tickers.length} tickers from ${filePath}`)
      return true
    } catch (e: any) {
      console.warn(`[PortfolioRisk] failed to load correlation matrix: ${e.message}`)
      return false
    }
  }

  /** Replace the in-memory correlation matrix directly. */
  setCorrelationMatrix(matrix: CorrelationMatrix): void {
    this.correlationMatrix = matrix
  }

  // ─── Daily counters / drawdown state ───────────────────────────────────────

  /** Reset daily P&L counter (called automatically at 00:00 MSK). */
  resetDailyCounters(): void {
    this.dailyPnL = 0
    this.haltedForDay = false
    this.haltReason = ''
    // NOTE: do NOT reset sizeMultiplier here — it depends on drawdown, which is
    // cumulative (not daily). It recovers as account value climbs back to peak.
  }

  /** Get the current MSK date as YYYY-MM-DD. */
  private getCurrentMSKDate(): string {
    const now = new Date()
    // MSK = UTC+3
    const mskTime = new Date(now.getTime() + 3 * 3600 * 1000)
    return mskTime.toISOString().slice(0, 10)
  }

  /** Auto-reset daily counters if we've crossed midnight MSK. */
  private maybeResetDailyCounters(): void {
    const today = this.getCurrentMSKDate()
    if (today !== this.dailyPnLDate) {
      this.dailyPnLDate = today
      this.resetDailyCounters()
    }
  }

  /** Check if daily loss limit has been exceeded → halt for day. */
  private checkDailyLossLimit(): void {
    if (this.haltedForDay) return
    const dailyPnLPct = this.dailyPnL / this.config.accountValue
    if (dailyPnLPct <= -this.config.maxDailyLossPct) {
      this.haltedForDay = true
      this.haltReason = `daily loss ${(dailyPnLPct * 100).toFixed(2)}% ≤ -${(this.config.maxDailyLossPct * 100).toFixed(2)}%`
      console.warn(`[PortfolioRisk] HALTED FOR DAY: ${this.haltReason}`)
    }
  }

  /** Update drawdown-reduction state based on current account value. */
  private updateDrawdownState(currentAccountValue: number): void {
    if (currentAccountValue > this.peakAccountValue) {
      this.peakAccountValue = currentAccountValue
    }
    const drawdownPct = (currentAccountValue - this.peakAccountValue) / this.peakAccountValue
    if (drawdownPct <= -this.config.maxDrawdownPct) {
      if (this.sizeMultiplier > 0.5) {
        this.sizeMultiplier = 0.5
        console.warn(`[PortfolioRisk] drawdown ${(drawdownPct * 100).toFixed(2)}% ≤ -${(this.config.maxDrawdownPct * 100).toFixed(2)}% → size multiplier = 0.5`)
      }
    } else if (drawdownPct > -this.config.maxDrawdownPct * 0.5) {
      // Recover to half of drawdown threshold → restore full size
      if (this.sizeMultiplier < 1.0) {
        this.sizeMultiplier = 1.0
        console.log(`[PortfolioRisk] drawdown recovered to ${(drawdownPct * 100).toFixed(2)}% → size multiplier = 1.0`)
      }
    }
  }

  // ─── Config / inspection ────────────────────────────────────────────────────

  getConfig(): PortfolioRiskConfig {
    return { ...this.config }
  }

  updateConfig(updates: Partial<PortfolioRiskConfig>): void {
    this.config = { ...this.config, ...updates }
    // Reset peak if account value changed
    if (updates.accountValue !== undefined) {
      const newPeak = Math.max(
        updates.accountValue,
        this.config.accountValue + this.cumulativePnL,
      )
      this.peakAccountValue = newPeak
    }
  }

  isHalted(): boolean {
    return this.haltedForDay
  }

  getHaltReason(): string {
    return this.haltReason
  }

  /** Snapshot for logging / dashboard. */
  getStatus(): {
    openPositions: number
    dailyPnL: number
    cumulativePnL: number
    unrealizedPnL: number
    drawdownPct: number
    sizeMultiplier: number
    haltedForDay: boolean
    haltReason: string
    portfolioVaR: number
    peakAccountValue: number
    currentAccountValue: number
  } {
    const accountValue = this.config.accountValue + this.cumulativePnL
    return {
      openPositions: this.positions.size,
      dailyPnL: this.dailyPnL,
      cumulativePnL: this.cumulativePnL,
      unrealizedPnL: this.getUnrealizedPnL(),
      drawdownPct: this.getDrawdownPct(),
      sizeMultiplier: this.sizeMultiplier,
      haltedForDay: this.haltedForDay,
      haltReason: this.haltReason,
      portfolioVaR: this.getPortfolioVaR(),
      peakAccountValue: this.peakAccountValue,
      currentAccountValue: accountValue,
    }
  }
}

// ─── Module-level singleton (one portfolio manager per process) ───────────────

let _instance: PortfolioRiskManager | null = null

/** Get the global PortfolioRiskManager singleton. Creates one on first call. */
export function getPortfolioRisk(config?: Partial<PortfolioRiskConfig>): PortfolioRiskManager {
  if (!_instance) {
    _instance = new PortfolioRiskManager(config)
  }
  return _instance
}

/** Replace the global singleton (useful for tests / config reload). */
export function setPortfolioRisk(mgr: PortfolioRiskManager): void {
  _instance = mgr
}

// ─── Self-test ────────────────────────────────────────────────────────────────

const isMain =
  (typeof import.meta === 'object' && (import.meta as any).main === true) ||
  (typeof require !== 'undefined' && typeof require.main !== 'undefined' && require.main === module)

if (isMain) {
  console.log(`=== PortfolioRiskManager — self-test ===\n`)

  // ── Test 1: basic gating ─────────────────────────────────────────────────
  console.log(`1. Basic gating tests...`)
  const mgr = new PortfolioRiskManager({
    accountValue: 1_000_000,
    maxConcurrentSameTicker: 3,
    maxDailyLossPct: 0.02,
    maxDrawdownPct: 0.05,
  })

  // First bot in SBER → allowed
  let r = mgr.canOpenPosition('bot-1', 'SBER', 'long')
  console.log(`   bot-1 SBER long: allowed=${r.allowed} mult=${r.sizeMultiplier}  reason="${r.reason}"`)
  if (!r.allowed) throw new Error('Expected bot-1 allowed')

  mgr.registerPosition('bot-1', 'SBER', 'long', 100, 250)

  // Second bot in SBER → allowed (max=3)
  r = mgr.canOpenPosition('bot-2', 'SBER', 'long')
  console.log(`   bot-2 SBER long: allowed=${r.allowed}  reason="${r.reason}"`)
  mgr.registerPosition('bot-2', 'SBER', 'long', 100, 250)

  // Third bot → allowed
  r = mgr.canOpenPosition('bot-3', 'SBER', 'long')
  console.log(`   bot-3 SBER long: allowed=${r.allowed}  reason="${r.reason}"`)
  mgr.registerPosition('bot-3', 'SBER', 'long', 100, 250)

  // Fourth bot → REJECTED
  r = mgr.canOpenPosition('bot-4', 'SBER', 'long')
  console.log(`   bot-4 SBER long: allowed=${r.allowed}  reason="${r.reason}"`)
  if (r.allowed) throw new Error('Expected bot-4 rejected (max 3 in SBER)')

  // Different ticker → allowed
  r = mgr.canOpenPosition('bot-5', 'GAZP', 'long')
  console.log(`   bot-5 GAZP long: allowed=${r.allowed}  reason="${r.reason}"`)
  if (!r.allowed) throw new Error('Expected bot-5 allowed')

  // ── Test 2: P&L tracking ──────────────────────────────────────────────────
  console.log(`\n2. P&L tracking...`)
  mgr.registerPosition('bot-5', 'GAZP', 'long', 100, 200)

  // bot-1 closes with +500 RUB profit
  const pnl1 = mgr.closePosition('bot-1', 255)  // +5 RUB/share × 100 = +500
  console.log(`   bot-1 close P&L: ${pnl1}  dailyPnL=${mgr.getDailyPnL()}  cumPnL=${mgr.getCumulativePnL()}`)
  if (Math.abs(pnl1 - 500) > 1) throw new Error(`Expected P&L +500, got ${pnl1}`)

  // ── Test 3: VaR (no correlation matrix) ───────────────────────────────────
  console.log(`\n3. Portfolio VaR (no correlation matrix)...`)
  // bot-2/3 are still in SBER long @ 250 × 100 shares each = 25000 each, total 50000
  // bot-5 is in GAZP long @ 200 × 100 = 20000
  // Total notional = 70000. Default vol 1.5%. z=1.645.
  // Expected (no diversification): 1.645 * 70000 * 0.015 / 1000000 = 0.001725 (0.17% of account)
  const var1 = mgr.getPortfolioVaR()
  console.log(`   VaR (no corr matrix): ${var1.toFixed(6)}  (${(var1 * 100).toFixed(4)}% of account)`)
  if (var1 <= 0) throw new Error('Expected positive VaR')

  // ── Test 4: VaR with correlation matrix ───────────────────────────────────
  console.log(`\n4. Portfolio VaR with correlation matrix...`)
  mgr.setCorrelationMatrix({
    index: { 'SBER': 0, 'GAZP': 1, 'LKOH': 2 },
    matrix: [
      [1.0, 0.65, 0.45],
      [0.65, 1.0, 0.30],
      [0.45, 0.30, 1.0],
    ],
    volatilities: { 'SBER': 0.015, 'GAZP': 0.018, 'LKOH': 0.020 },
  })
  const var2 = mgr.getPortfolioVaR()
  console.log(`   VaR (with corr matrix): ${var2.toFixed(6)}  (${(var2 * 100).toFixed(4)}% of account)`)
  // With diversification, VaR should be lower than the no-corr-matrix sum-of-notional
  // ... but our matrix has + corr, so could be similar. Just check it's >0 and finite.
  if (!Number.isFinite(var2) || var2 <= 0) throw new Error('Invalid VaR with corr matrix')

  // ── Test 5: daily loss limit ───────────────────────────────────────────────
  console.log(`\n5. Daily loss limit...`)
  // Trigger a big loss on bot-2: enter at 250, close at 200 → -50 × 100 = -5000 RUB
  // Account = 1M → -5000/1M = -0.5% (not yet at -2%)
  const pnl2 = mgr.closePosition('bot-2', 200)
  console.log(`   bot-2 close P&L: ${pnl2}  dailyPnL=${mgr.getDailyPnL()}  halted=${mgr.shouldStopForDay()}`)
  if (mgr.shouldStopForDay()) throw new Error('Should NOT halt yet')

  // bot-3 close: -100 × 100 = -10000 → cumulative daily = -14500 = -1.45% (still not at -2%)
  const pnl3 = mgr.closePosition('bot-3', 150)  // entry 250 → -100 × 100 = -10000
  console.log(`   bot-3 close P&L: ${pnl3}  dailyPnL=${mgr.getDailyPnL()}  halted=${mgr.shouldStopForDay()}`)
  if (mgr.shouldStopForDay()) throw new Error('Should NOT halt yet (-1.45%)')

  // bot-5 close: -100 × 100 = -10000 → cumulative daily = -24500 = -2.45% (OVER -2%!)
  const pnl5 = mgr.closePosition('bot-5', 100)  // entry 200 → -100 × 100 = -10000
  console.log(`   bot-5 close P&L: ${pnl5}  dailyPnL=${mgr.getDailyPnL()}  halted=${mgr.shouldStopForDay()}`)
  if (!mgr.shouldStopForDay()) throw new Error('Should halt after -2.45% daily loss')

  // Try to open new position → should be blocked
  r = mgr.canOpenPosition('bot-6', 'SBER', 'long')
  console.log(`   bot-6 SBER long after halt: allowed=${r.allowed}  reason="${r.reason}"`)
  if (r.allowed) throw new Error('Should block new position after halt')

  // ── Test 6: drawdown-reduction ───────────────────────────────────────────
  console.log(`\n6. Drawdown reduction (size multiplier)...`)
  // Fresh manager for clean test — set daily loss limit HIGHER than drawdown
  // limit so a single -6% loss triggers drawdown but NOT daily halt.
  const mgr2 = new PortfolioRiskManager({
    accountValue: 1_000_000,
    maxDrawdownPct: 0.05,
    maxDailyLossPct: 0.10,   // -10% → only triggers after multiple losses
  })
  // Open a position, then close at a big loss (-6% of account)
  mgr2.registerPosition('b1', 'SBER', 'long', 1000, 250)
  // Close at 190 → -60 × 1000 = -60000 → -6% drawdown (above -5% threshold)
  const p = mgr2.closePosition('b1', 190)
  console.log(`   b1 close P&L: ${p}  drawdown=${(mgr2.getDrawdownPct() * 100).toFixed(2)}%  sizeMult=${mgr2.getPositionSizeMultiplier()}  halted=${mgr2.isHalted()}`)
  if (mgr2.getPositionSizeMultiplier() !== 0.5) throw new Error('Expected sizeMultiplier=0.5')
  if (mgr2.isHalted()) throw new Error('Should NOT halt (-6% < -10% daily loss limit)')

  // New position should still be allowed but with halved size
  r = mgr2.canOpenPosition('b2', 'SBER', 'long')
  console.log(`   b2 SBER long: allowed=${r.allowed}  sizeMult=${r.sizeMultiplier}  reason="${r.reason}"`)
  if (!r.allowed || r.sizeMultiplier !== 0.5) throw new Error('Expected allowed with mult=0.5')

  // ── Test 7: status snapshot ─────────────────────────────────────────────────
  console.log(`\n7. Status snapshot...`)
  mgr2.registerPosition('b2', 'GAZP', 'long', 500, 200)
  mgr2.markToMarket('b2', 210)  // +10 × 500 = +5000 unrealized
  const status = mgr2.getStatus()
  console.log(`   Status: ${JSON.stringify(status, null, 2)}`)

  console.log(`\n=== self-test complete ===`)
}
