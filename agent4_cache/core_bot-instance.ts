$ cat /opt/ai-trader/src/core/bot-instance.ts
--- rc=0 ---
/**
 * BotInstance — one independent trading bot.
 * Always updates BotState in DB (balance, positions, last signal).
 * Records trades to Trade table on execution.
 */

import { BotConfig, BotLiveStats, Position, Candle } from './types'
import { IStrategy, createStrategy } from '../strategies/base'
import { OrderManager, COMMISSION_RATE } from './order-manager'
import { RiskManager } from './risk-manager'
import { PriceProvider } from './price-provider'

export class BotInstance {
  config: BotConfig
  strategy: IStrategy
  orders: OrderManager
  prices: PriceProvider
  stats: BotLiveStats
  private currentTickerIdx: number = 0
  private lastTickerRotate: number = Date.now()
  // OPTIMIZATION: getStatus only every 5 min (was every 10s = 108 gRPC/min → now 3.6/min)
  private lastBrokerSync: number = 0
  private readonly BROKER_SYNC_INTERVAL = 5 * 60 * 1000  // 5 minutes
  // Cooldown after failed orders: ticker → timestamp until which to skip.
  private failedOrderCooldown: Map<string, number> = new Map()
  private readonly COOLDOWN_MS = 5 * 60 * 1000
  // FIX (B6): throttle timestamps for error logging
  private lastExitErrTs: Map<string, number> = new Map()
  private lastBotStateErrTs: number = 0
  private lastCooldownCleanup: number = 0
  // Recently closed tickers: prevents SYNC from recording duplicate CLOSE trades.
  private recentlyClosed: Map<string, number> = new Map()
  private readonly RECENT_CLOSE_TTL = 60 * 1000

  constructor(config: BotConfig, orders: OrderManager, prices: PriceProvider) {
    this.config = config
    this.orders = orders
    this.prices = prices
    this.strategy = createStrategy(config.strategy, config)
    this.stats = {
      name: config.name,
      agentType: config.strategy,
      color: config.color,
      liveBuys: 0, liveSells: 0, liveTrades: 0, realizedPnl: 0,
      realBalance: 0, realSharesValue: 0, realTotalValue: 0,
      openPositions: [], history: [],
      lastSignal: { action: 0, ticker: '', ts: 0 },
    }
    this.currentTickerIdx = (config.startTickerIdx || 0) % (config.tickers.length || 1)
    console.log(`[${config.name}] Strategy: ${config.strategy}, tickers: ${config.tickers.join(',')}, interval: ${config.candleInterval || '5min'}`)
    console.log(`[${config.name}] Starting at ticker idx ${this.currentTickerIdx}: ${config.tickers[this.currentTickerIdx]}`)
  }

  getCurrentTicker(): string {
    if (this.config.tickerMode === 'fixed' || this.config.tickers.length <= 1) {
      return this.config.tickers[0]
    }
    const now = Date.now()
    const interval = (this.config.rotateIntervalSec || 300) * 1000
    if (now - this.lastTickerRotate > interval) {
      this.currentTickerIdx = (this.currentTickerIdx + 1) % this.config.tickers.length
      this.lastTickerRotate = now
      const newTicker = this.config.tickers[this.currentTickerIdx]
      console.log(`[${this.config.name}] Rotated to ${newTicker}`)
      return newTicker
    }
    return this.config.tickers[this.currentTickerIdx]
  }

  async tick(): Promise<void> {
    const interval = this.config.candleInterval || '5min'

    // === SHARED ACCOUNT: skip broker sync, use virtual balance ===
    // Shared bots (V2-Test1-10) run on one T-Bank account with 100k.
    // Each bot has its own virtualBalance (10k) tracked locally in BotState.
    // Don't call getStatus (would show 100k total, not individual 10k).
    const isShared = (this.config as any).sharedAccount === true
    if (!isShared) {
      await this.maybeSyncFromBroker()
    }

    // === SCAN-ALL: check ALL tickers for entry signals (not just rotated one) ===
    // Was: rotation 2-5 min per ticker → missed signals on 10 other tickers.
    // Now: every tick (10s) checks all 11 tickers via cached candles (60s TTL).
    let bestSignal: { ticker: string; action: number; price: number; holding: number } | null = null

    // FIX (rate-limit): shuffle tickers per-bot per-tick. Was: all 9 bots iterate
    // tickers in the SAME order (SBER, GAZP, ...) → when a signal exists on e.g.
    // TATN, all bots spot it at the same tick and fire 9 parallel SHORT TATN
    // orders → T-Bank limit (2/sec) rejects 7 of them → RESOURCE_EXHAUSTED +
    // 5min cooldown → bots sit idle.
    // Now: each bot scans tickers in a different random order, so when multiple
    // bots find signals they're spread across different tickers → fewer
    // simultaneous orders on the same ticker.
    const shuffled = [...this.config.tickers]
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1))
      ;[shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]]
    }

    for (const ticker of shuffled) {
      try {
        const candles = await this.prices.getCandlesForTicker(ticker, interval)
        if (candles.length < 15) continue

        const idx = candles.length - 1
        const price = candles[idx].close
        if (price === 0 || price < 1) continue

        const openPos = this.stats.openPositions.find(p => p.ticker === ticker)
        const holding = openPos ? (openPos.side === 'short' ? -openPos.qty : openPos.qty) : 0
        const stepsHeld = openPos ? Math.floor((Date.now() - openPos.ts) / 10000) : 0

        // Check for EXIT first (if has position on this ticker)
        if (openPos) {
          const action = this.strategy.predict(candles, idx, holding !== 0, stepsHeld, { ticker, entryPrice: openPos.entryPrice, holding })
          const expMove = idx > 0 ? Math.abs((candles[idx].close - candles[idx - 1].close) / candles[idx - 1].close) : 0
          const filtered = RiskManager.filter(action, this.config, holding, price, openPos, this.stats.history, expMove)
          if (filtered.action !== 0) {
            console.log(`[${this.config.name}] EXIT ${ticker} act=${filtered.action}`)
            const rubBalance = this.stats.realBalance || 10000
            await this.execute(filtered.action, ticker, holding, price, rubBalance)
            continue  // position closed, don't also open new on same ticker
          }
        }

        // Check for ENTRY (only if no position on this ticker)
        if (!openPos) {
          const action = this.strategy.predict(candles, idx, false, 0, { ticker, entryPrice: 0, holding: 0 })
          // UNLEASHED DEBUG: log every ticker with RSI/SMA to see why no signals
          if ((this.config as any).skipRiskManager === true) {
            const sma5 = candles.slice(Math.max(0, idx-4), idx+1).reduce((s,c)=>s+c.close,0) / Math.min(5, idx+1)
            const sma14 = candles.slice(Math.max(0, idx-13), idx+1).reduce((s,c)=>s+c.close,0) / Math.min(14, idx+1)
            const last3 = [candles[idx-2]?.close, candles[idx-1]?.close, candles[idx].close]
            const allUp = last3[0] < last3[1] && last3[1] < last3[2]
            const allDn = last3[0] > last3[1] && last3[1] > last3[2]
            const recentCloses = candles.slice(Math.max(0, idx-14), idx+1).map(c=>c.close)
            let g=0,l=0
            for (let i=1;i<recentCloses.length;i++){const ch=recentCloses[i]-recentCloses[i-1];if(ch>0)g+=ch;else l-=ch}
            const rsi = l===0 ? 100 : 100 - 100/(1+g/l)
            const shortSmaOk = sma5 < sma14*0.999
            const shortRsiOk = rsi > 30 && rsi < 55
            const longSmaOk = sma5 > sma14*1.002
            const longRsiOk = rsi > 25 && rsi < 40
            console.log(`[V2U] ${ticker} close=${price} sma5=${sma5.toFixed(2)} sma14=${sma14.toFixed(2)} rsi=${rsi.toFixed(1)} allUp=${allUp} allDn=${allDn} act=${action} | SHORT: sma=${shortSmaOk} rsi=${shortRsiOk} allDn=${allDn} | LONG: sma=${longSmaOk} rsi=${longRsiOk} allUp=${allUp}`)
          }
          // UNLEASHED mode: skip RiskManager entirely if config.skipRiskManager is set
          if (action !== 0) {
            const expMove = idx > 0 ? Math.abs((candles[idx].close - candles[idx - 1].close) / candles[idx - 1].close) : 0
            const skipRisk = (this.config as any).skipRiskManager === true
            const filtered = skipRisk ? { action } : RiskManager.filter(action, this.config, 0, price, undefined, this.stats.history, expMove)
            console.log(`[${this.config.name}] SIGNAL ${ticker} act=${action} expMove=${(expMove*100).toFixed(3)}% risk=${skipRisk ? 'SKIP' : 'CHECK'} filtered=${filtered.action}${filtered.reason ? ' reason=' + filtered.reason : ''}`)
            if (filtered.action !== 0) {
              // Found a signal — remember it (will execute the first one found)
              if (!bestSignal) {
                bestSignal = { ticker, action: filtered.action, price, holding: 0 }
              }
            }
          }
        }
      } catch (e: any) {
        const last = this.lastExitErrTs.get(ticker) || 0
        if (Date.now() - last > 60000) {
          console.error(`[${this.config.name}] scan ${ticker} failed: ${e.message}`)
          this.lastExitErrTs.set(ticker, Date.now())
        }
      }
    }

    // === EXECUTE best signal (if any) ===
    if (bestSignal) {
      const rubBalance = this.stats.realBalance || 10000
      console.log(`[${this.config.name}] ENTRY ${bestSignal.ticker} act=${bestSignal.action}`)
      await this.execute(bestSignal.action, bestSignal.ticker, bestSignal.holding, bestSignal.price, rubBalance)
    }

    // Update BotState in DB
    await this.updateBotState()

    // Log current state (first ticker as representative)
    const logTicker = this.config.tickers[0]
    const rubBalance = this.stats.realBalance || 10000
    console.log(`[${this.config.name}] scan done bal=${rubBalance.toFixed(0)} pos=${this.stats.openPositions.length} signal=${bestSignal ? bestSignal.ticker : 'none'}`)
  }

  /** Sync positions from broker every 5 min (was every tick = 108 gRPC/min). */
  private async maybeSyncFromBroker(): Promise<void> {
    if (Date.now() - this.lastBrokerSync < this.BROKER_SYNC_INTERVAL) return
    this.lastBrokerSync = Date.now()

    let status = await this.orders.getStatus(this.config.name)
    // FIX: auto-recover from 'Account not found' (50004). Daemon's do_POST auto-recover
    // already handles this for most calls, but if the FIRST getStatus after account
    // invalidation fails (race condition), worker sees the error. Detect 50004 and
    // call daemon's reset endpoint to recreate the account, then retry getStatus once.
    if (status.error && (String(status.error).includes('50004') || String(status.error).toLowerCase().includes('not found'))) {
      console.warn(`[${this.config.name}] account gone (50004), calling daemon reset to recreate...`)
      try {
        const resetResp = await fetch('http://127.0.0.1:3008/', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ cmd: 'reset', bot_name: this.config.name }),
          signal: AbortSignal.timeout(60000),
        })
        const resetData = await resetResp.json()
        if (resetData.ok) {
          console.log(`[${this.config.name}] ✓ account recreated (old=${resetData.old_id?.slice(0,8)}... new=${resetData.new_id?.slice(0,8)}...), retrying getStatus`)
          // Retry getStatus with new account
          status = await this.orders.getStatus(this.config.name)
          if (status.error) {
            console.error(`[${this.config.name}] status sync still failed after reset: ${status.error}`)
            return
          }
        } else {
          console.error(`[${this.config.name}] daemon reset failed: ${resetData.error}`)
          return
        }
      } catch (e: any) {
        console.error(`[${this.config.name}] reset call failed: ${e.message}`)
        return
      }
    } else if (status.error) {
      console.error(`[${this.config.name}] status sync failed: ${status.error}`)
      return
    }

    // Update balance/total from broker
    this.stats.realBalance = status.rub_balance || this.stats.realBalance
    this.stats.realSharesValue = status.shares_value || 0
    this.stats.realTotalValue = status.total_value || this.stats.realBalance

    // SYNC positions from broker (detect vanished, update openPositions)
    if (status.holdings) {
      const brokerTickers = new Set(status.holdings.filter((h: any) => h.balance !== 0).map((h: any) => h.ticker))
      const now = Date.now()
      for (const [k, v] of this.recentlyClosed) {
        if (v < now) this.recentlyClosed.delete(k)
      }
      const vanished = this.stats.openPositions.filter(
        p => !brokerTickers.has(p.ticker) && !this.recentlyClosed.has(p.ticker)
      )
      for (const v of vanished) {
        const lastPrice = status.holdings.find((h: any) => h.ticker === v.ticker)?.last_price || v.entryPrice
        const isShort = v.side === 'short'
        const pnl = isShort ? (v.entryPrice - lastPrice) * v.qty : (lastPrice - v.entryPrice) * v.qty
        this.stats.realizedPnl += pnl
        console.log(`[${this.config.name}] ⚠️ SYNC: ${v.ticker} vanished, recording CLOSE pnl=${pnl.toFixed(1)}`)
        await this.recordTrade(isShort ? 'CLOSE_SHORT' : 'SELL', v.ticker, v.qty, lastPrice, pnl, status.rub_balance || 0)
      }
      if (vanished.length) {
        const gone = new Set(vanished.map(v => v.ticker))
        this.stats.openPositions = this.stats.openPositions.filter(p => !gone.has(p.ticker))
      }
      const realPos = status.holdings
        .filter((h: any) => h.balance !== 0)
        .map((h: any) => {
          const existing = this.stats.openPositions.find(p => p.ticker === h.ticker)
          return {
            ticker: h.ticker,
            qty: Math.abs(h.balance),
            entryPrice: existing?.entryPrice ?? (h.avg_price || h.last_price || 0),
            ts: existing?.ts || Date.now(),
            ...(h.balance < 0 ? { side: 'short' as const } : {}),
          }
        })
      this.stats.openPositions = realPos
    }
  }

  private async execute(action: number, ticker: string, holding: number, price: number, rubBalance: number): Promise<void> {
    if (action === 0) return
    const isShared = (this.config as any).sharedAccount === true
    // Cooldown: skip if this ticker recently failed (e.g. instrument unavailable)
    const cooldownUntil = this.failedOrderCooldown.get(ticker)
    if (cooldownUntil && Date.now() < cooldownUntil) {
      return  // silent skip — logged once when failure happened
    }
    // FIX (B5): was `Math.floor(...) % 1 === 0` which is ALWAYS true (any int % 1 = 0).
    // Now: cleanup once per 60 seconds using explicit timestamp.
    if (Date.now() - this.lastCooldownCleanup > 60_000) {
      this.lastCooldownCleanup = Date.now()
      for (const [k, v] of this.failedOrderCooldown) {
        if (v < Date.now()) this.failedOrderCooldown.delete(k)
      }
    }
    // Lot sizes per ticker (MOEX standard)
    const LOT_SIZES = require('./lot-sizes').LOT_SIZES
    const lotSize = LOT_SIZES[ticker] || 1
    const maxCost = this.config.maxPositionCost || 3000  // max 3000 RUB per position

    // SHARED ACCOUNT: use virtual balance for margin check (not T-Bank's 100k total)
    const effectiveBalance = isShared ? this.stats.realBalance : rubBalance

    // CLOSING a position (holding != 0) needs no margin check — just exit.
    // Margin check applies only to OPENING new positions (holding === 0).
    let posSize = 0
    if (holding === 0) {
      // === FIX: use NET total value (cash + shares value) as margin base.
      // Was: rubBalance (which includes short-sale proceeds = inflated cash) - openShortsValue
      // (double-counting). realTotalValue already accounts for all open positions because
      // T-Bank computes total_value = cash + shares_value (shares_value is negative for shorts).
      // Available margin = total_value - value_already_locked_in_positions.
      const totalValue = isShared ? this.stats.realBalance : (this.stats.realTotalValue || effectiveBalance)
      let openPositionsValue = 0
      for (const pos of this.stats.openPositions) {
        const posPrice = pos.ticker === ticker ? price : pos.entryPrice
        // FIX: SHORT positions gave us cash (proceeds) at entry — they do NOT
        // lock capital in the same way LONG positions do. Subtracting their
        // value from availableMargin double-counts (the cash from short sale
        // is already in realBalance). Only subtract LONG positions' value.
        if (pos.side === 'short') {
          continue
        }
        openPositionsValue += pos.qty * posPrice
      }
      const availableMargin = Math.max(0, totalValue - openPositionsValue)
      posSize = Math.min(availableMargin * this.config.positionSize, maxCost)
      // Safety: skip if position too small (can't afford even 1 lot)
      const minLotCost = price * lotSize
      const skipMinLot = (this.config as any).skipRiskManager === true
      if (!skipMinLot && posSize < minLotCost) {
        console.log(`[${this.config.name}] SKIP action=${action}: availableMargin=${availableMargin.toFixed(0)} < minLotCost=${minLotCost.toFixed(0)} (${ticker})`)
        return
      }
    }

    if (action === 1 && holding === 0) {
      const lots = Math.max(1, Math.floor(posSize / (price * lotSize)))
      const qty = lots  // API expects quantity in LOTS
      const r = await this.orders.buy(ticker, qty, this.config.name, this.config.useLimitOrders)
      if (!r.error) {
        this.stats.liveBuys++; this.stats.liveTrades++
        const execPrice = r.exec_price || price
        const shares = qty * lotSize
        this.stats.openPositions.push({ ticker, qty: shares, entryPrice: execPrice, ts: Date.now() })
        const bal = isShared ? this.updateVirtualBalance('BUY', execPrice, shares) : (r.rub_balance || 0)
        await this.recordTrade('BUY', ticker, shares, execPrice, 0, bal)
      } else {
        console.log(`[${this.config.name}] BUY ${ticker} FAILED: ${r.error}`)
        this.failedOrderCooldown.set(ticker, Date.now() + this.COOLDOWN_MS)
      }
    } else if ((action === 2 || action === 3) && holding > 0) {
      const sellLots = Math.max(1, Math.floor(holding / lotSize))
      const r = await this.orders.sell(ticker, sellLots, this.config.name)
      if (!r.error) {
        this.stats.liveSells++; this.stats.liveTrades++
        const execPrice = r.exec_price || price
        const executedShares = (r.lots_executed || sellLots) * lotSize
        const pnl = this.calcPnl(execPrice, executedShares, true, ticker)
        if (executedShares >= holding) {
          this.stats.openPositions = this.stats.openPositions.filter(p => p.ticker !== ticker)
          this.recentlyClosed.set(ticker, Date.now() + this.RECENT_CLOSE_TTL)
        } else {
          const pos = this.stats.openPositions.find(p => p.ticker === ticker)
          if (pos) pos.qty = holding - executedShares
        }
        const bal = isShared ? this.updateVirtualBalance('SELL', execPrice, executedShares) : (r.rub_balance || 0)
        await this.recordTrade('SELL', ticker, executedShares, execPrice, pnl, bal)
      } else {
        console.log(`[${this.config.name}] SELL ${ticker} FAILED: ${r.error}`)
        this.failedOrderCooldown.set(ticker, Date.now() + this.COOLDOWN_MS)
      }
    } else if ((action === 1 || action === 2 || action === 3) && holding < 0) {
      const shortQty = Math.abs(holding)
      const buyLots = Math.max(1, Math.floor(shortQty / lotSize))
      const r = await this.orders.buy(ticker, buyLots, this.config.name)
      if (!r.error) {
        this.stats.liveBuys++; this.stats.liveTrades++
        const execPrice = r.exec_price || price
        const executedShares = (r.lots_executed || buyLots) * lotSize
        const pnl = this.calcPnl(execPrice, executedShares, false, ticker)
        if (executedShares >= shortQty) {
          this.stats.openPositions = this.stats.openPositions.filter(p => !(p.ticker === ticker && p.side === 'short'))
          this.recentlyClosed.set(ticker, Date.now() + this.RECENT_CLOSE_TTL)
        } else {
          const pos = this.stats.openPositions.find(p => p.ticker === ticker && p.side === 'short')
          if (pos) pos.qty = shortQty - executedShares
        }
        const bal = isShared ? this.updateVirtualBalance('CLOSE_SHORT', execPrice, executedShares) : (r.rub_balance || 0)
        await this.recordTrade('CLOSE_SHORT', ticker, executedShares, execPrice, pnl, bal)
      } else {
        console.log(`[${this.config.name}] CLOSE_SHORT ${ticker} FAILED: ${r.error}`)
        this.failedOrderCooldown.set(ticker, Date.now() + this.COOLDOWN_MS)
      }
    } else if ((action === 2 || action === 3) && holding === 0) {
      const lots = Math.max(1, Math.floor(posSize / (price * lotSize)))
      const qty = lots  // API expects LOTS
      const r = await this.orders.sell(ticker, qty, this.config.name)
      if (!r.error) {
        this.stats.liveSells++; this.stats.liveTrades++
        const execPrice = r.exec_price || price
        const shares = qty * lotSize
        this.stats.openPositions.push({ ticker, qty: shares, entryPrice: execPrice, ts: Date.now(), side: 'short' })
        const bal = isShared ? this.updateVirtualBalance('SHORT', execPrice, shares) : (r.rub_balance || 0)
        await this.recordTrade('SHORT', ticker, shares, execPrice, 0, bal)
      } else {
        console.log(`[${this.config.name}] SHORT ${ticker} FAILED: ${r.error}`)
        this.failedOrderCooldown.set(ticker, Date.now() + this.COOLDOWN_MS)
      }
    }
  }

  /**
   * SHARED ACCOUNT: update virtual balance after a trade.
   * Called only when config.sharedAccount === true.
   * Tracks bot's own 10k slice of the shared 100k account.
   * - BUY: cash -= (price * qty) + commission
   * - SELL: cash += (price * qty) - commission
   * - SHORT: cash += (price * qty) - commission (proceeds from short sale)
   * - CLOSE_SHORT: cash -= (price * qty) + commission (buy back shares)
   * Returns the new virtual cash balance for recordTrade.
   */
  private updateVirtualBalance(side: string, execPrice: number, qty: number): number {
    const tradeValue = execPrice * qty
    const commission = tradeValue * 0.0005  // 0.05% per side
    if (side === 'BUY' || side === 'CLOSE_SHORT') {
      // Spending cash: buy shares or buy back short
      this.stats.realBalance -= (tradeValue + commission)
    } else {
      // SELL or SHORT: receiving cash
      this.stats.realBalance += (tradeValue - commission)
    }
    // Update realTotalValue = cash + unrealized PnL from open positions
    let unrealized = 0
    for (const pos of this.stats.openPositions) {
      if (pos.side === 'short') {
        unrealized += (pos.entryPrice - execPrice) * pos.qty  // approx: use last execPrice as current
      } else {
        unrealized += (execPrice - pos.entryPrice) * pos.qty
      }
    }
    this.stats.realTotalValue = this.stats.realBalance + unrealized
    return this.stats.realBalance
  }

  /** Always update BotState — balance, positions, last signal */
  private async updateBotState(): Promise<void> {
    try {
      const { db } = require('../lib/db')
      await db.botState.upsert({
        where: { botName: this.config.name },
        create: {
          botName: this.config.name,
          realBalance: this.stats.realBalance,
          realTotalValue: this.stats.realTotalValue,
          realizedPnl: this.stats.realizedPnl,
          liveBuys: this.stats.liveBuys,
          liveSells: this.stats.liveSells,
          liveTrades: this.stats.liveTrades,
          openPositionsJson: JSON.stringify(this.stats.openPositions),
          lastSignalAction: this.stats.lastSignal.action,
          lastSignalTicker: this.stats.lastSignal.ticker,
          lastSignalTs: this.stats.lastSignal.ts,
          updatedAt: Date.now(),
        },
        update: {
          realBalance: this.stats.realBalance,
          realTotalValue: this.stats.realTotalValue,
          openPositionsJson: JSON.stringify(this.stats.openPositions),
          lastSignalAction: this.stats.lastSignal.action,
          lastSignalTicker: this.stats.lastSignal.ticker,
          lastSignalTs: this.stats.lastSignal.ts,
          updatedAt: Date.now(),
        }
      })
    } catch (e: any) {
      // FIX (B6): was (this as any)._botStateErrTs — now typed field
      if (Date.now() - this.lastBotStateErrTs > 60000) {
        console.error(`[${this.config.name}] updateBotState failed: ${e.message}`)
        this.lastBotStateErrTs = Date.now()
      }
    }
  }

  /** Record trade to DB */
  private async recordTrade(side: string, ticker: string, qty: number, price: number, pnl: number, bal: number): Promise<void> {
    try {
      const { db } = require('../lib/db')
      const ts = Date.now()
      await db.trade.create({
        data: {
          ts,
          botName: this.config.name,
          side, ticker, qty, price, pnl,
          balanceAfter: bal,
          interval: this.config.candleInterval || '5min',
        }
      })
      // FIX: push to in-memory history so RiskManager.rate-limit works.
      // Was: history only loaded once at startup (last 20 trades), never updated →
      // rate-limit check `recentTrades = history.filter(... <3600000).length` always
      // returned the same 20 → bots had no effective rate limit and overtraded (SMA-Cross
      // did 827 trades, 20₽ commissions, -17₽ PnL). Now: prepend new trade, trim to 50.
      this.stats.history.unshift({ ts, botName: this.config.name, side, ticker, qty, price, pnl, balanceAfter: bal })
      if (this.stats.history.length > 50) {
        this.stats.history = this.stats.history.slice(0, 50)
      }
      // Update BotState with trade stats (totalValue = bal, positions already closed)
      await db.botState.update({
        where: { botName: this.config.name },
        data: {
          realBalance: bal,
          realTotalValue: this.stats.realTotalValue || bal,
          realizedPnl: this.stats.realizedPnl,
          liveBuys: this.stats.liveBuys,
          liveSells: this.stats.liveSells,
          liveTrades: this.stats.liveTrades,
          openPositionsJson: JSON.stringify(this.stats.openPositions),
          updatedAt: Date.now(),
        }
      })
      console.log(`[${this.config.name}] ✅ ${side} ${qty} ${ticker} @ ${price.toFixed(2)}₽ P&L=${pnl >= 0 ? '+' : ''}${pnl.toFixed(0)}₽ [DB]`)
    } catch (e: any) {
      console.error(`[${this.config.name}] DB save failed: ${e.message}`)
    }
  }

  private calcPnl(execPrice: number, qty: number, isLong: boolean, ticker: string): number {
    const openPos = this.stats.openPositions.find(p => p.ticker === ticker)
    if (!openPos) return 0
    // entryPrice is stored per-share (set from exec_price at entry). Use directly.
    const entryPerShare = openPos.entryPrice
    const grossPnl = isLong ? (execPrice - entryPerShare) * qty : (entryPerShare - execPrice) * qty
    // Round-trip commission: 0.05% × (entry + exit) notional — both legs.
    // Balance already debits commission correctly at order time; this only
    // fixes the `pnl` column recorded into Trade table (was gross).
    const commission = 0.0005 * (entryPerShare + execPrice) * qty
    const pnl = grossPnl - commission
    this.stats.realizedPnl += pnl
    return pnl
  }
}


