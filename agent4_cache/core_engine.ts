$ cat /opt/ai-trader/src/core/engine.ts
--- rc=0 ---
/**
 * TradingEngine — main loop. Supports regular + pairs bots.
 */

import * as fs from 'fs'
import { BotInstance } from './bot-instance'
import { ScanAllBotInstance } from './scan-all-bot-instance'
import { PairsBotInstance } from './pairs-bot-instance'
import { OrderManager } from './order-manager'
import { PriceProvider } from './price-provider'

const CONFIG_DIR = '/opt/ai-trader/config/bots'
const TICK_MS = 10_000

export class TradingEngine {
  bots: (BotInstance | PairsBotInstance | ScanAllBotInstance)[] = []
  orders: OrderManager
  prices: PriceProvider
  private busy = false

  constructor() {
    this.orders = new OrderManager()
    this.prices = new PriceProvider()
  }

  loadConfigs(): number {
    const files = fs.readdirSync(CONFIG_DIR).filter(f => f.endsWith('.json'))
    this.bots = []
    for (const file of files) {
      try {
        const config = JSON.parse(fs.readFileSync(`${CONFIG_DIR}/${file}`, 'utf-8'))
        if (config.realTrading) continue
        // Use ScanAllBotInstance for scan-all mode (V7), PairsBotInstance for pairs,
        // regular BotInstance otherwise
        const bot = config.tickerMode === 'scan-all'
          ? new ScanAllBotInstance(config, this.orders, this.prices)
          : config.strategy === 'pairs-trading'
            ? new PairsBotInstance(config, this.orders, this.prices)
            : new BotInstance(config, this.orders, this.prices)
        this.bots.push(bot)
      } catch (e: any) {
        console.error(`[Engine] Failed to load ${file}: ${e.message}`)
      }
    }
    console.log(`[Engine] Loaded ${this.bots.length} bots`)
    return this.bots.length
  }

  async initialize(): Promise<void> {
    this.loadConfigs()
    await this.prices.initialize('SBER')
    await this.orders.ensureAccount()
    await this.initBotStates()
    await this.restoreStats()
  }

  private async initBotStates(): Promise<void> {
    try {
      const { db } = require('../lib/db')
      for (const bot of this.bots) {
        // SHARED ACCOUNT: use virtualBalance from config (10k), don't call getStatus
        // (would return 100k total shared balance, not individual bot's 10k slice)
        const isShared = (bot.config as any).sharedAccount === true
        if (isShared) {
          const vbal = (bot.config as any).virtualBalance || 10000
          await db.botState.upsert({
            where: { botName: bot.config.name },
            create: { botName: bot.config.name, realBalance: vbal, realTotalValue: vbal, startBalance: vbal },
            update: {}  // don't overwrite — restoreStats handles this
          })
          continue
        }
        const status = await this.orders.getStatus(bot.config.name)
        // FIX: after reset, sandbox account was just recreated — first getStatus
        // may return rub_balance=0 or total_value=0. Don't overwrite the 10000
        // that reset wrote. Only update if status returned non-zero values.
        const bal = status?.rub_balance && status.rub_balance > 0 ? status.rub_balance : 10000
        const total = status?.total_value && status.total_value > 0 ? status.total_value : bal
        await db.botState.upsert({
          where: { botName: bot.config.name },
          create: { botName: bot.config.name, realBalance: bal, realTotalValue: total, startBalance: 10000 },
          // FIX (B15): was `update: {}` (no-op) — now sync balance on restart
          update: { realBalance: bal, realTotalValue: total }
        })
      }
      console.log(`[Engine] Initialized ${this.bots.length} BotState rows`)
    } catch (e: any) {
      console.error(`[Engine] initBotStates: ${e.message}`)
    }
  }

  private async restoreStats(): Promise<void> {
    try {
      const { db } = require('../lib/db')
      for (const bot of this.bots) {
        const state = await db.botState.findUnique({ where: { botName: bot.config.name } })
        if (state) {
          bot.stats.realizedPnl = state.realizedPnl
          bot.stats.liveBuys = state.liveBuys
          bot.stats.liveSells = state.liveSells
          bot.stats.liveTrades = state.liveTrades
          // FIX: restore realBalance + realTotalValue from DB. Was: left at 0 (default),
          // then bot-instance.ts updateBotState() overwrote DB with 0 — losing the
          // 10000 that reset wrote. Now: stats matches DB → no spurious overwrite.
          bot.stats.realBalance = state.realBalance || 10000
          bot.stats.realTotalValue = state.realTotalValue || state.realBalance || 10000
          // Restore openPositions from DB (preserves entryPrice on restart)
          try {
            bot.stats.openPositions = JSON.parse(state.openPositionsJson || '[]')
          } catch { bot.stats.openPositions = [] }
          const recentTrades = await db.trade.findMany({
            where: { botName: bot.config.name },
            orderBy: { ts: 'desc' },
            take: 20,
          })
          bot.stats.history = recentTrades.map((t: any) => ({
            ts: t.ts, botName: t.botName, side: t.side, ticker: t.ticker,
            qty: t.qty, price: t.price, pnl: t.pnl, balanceAfter: t.balanceAfter
          }))
        }
      }
      console.log('[Engine] Restored stats from DB')
    } catch (e: any) {
      console.error('[Engine] restoreStats: ' + e.message)
    }
  }

  private skippedTicks = 0  // FIX (L2): track skipped ticks

  async tick(): Promise<void> {
    if (this.busy) {
      // FIX (L2): log when ticks are being skipped due to overload
      this.skippedTicks++
      if (this.skippedTicks >= 3) {
        this.log('error', `⏰ ${this.skippedTicks} ticks skipped (busy > ${TICK_MS * this.skippedTicks / 1000}s)`)
      }
      return
    }
    this.skippedTicks = 0
    this.busy = true
    try {
      const marketStatus = await this.orders.getTradingStatus('SBER')
      if (marketStatus?.can_trade === false) {
        // FIX: trading_status may be undefined if daemon returned incomplete data
        const status = marketStatus.trading_status ?? 'unknown'
        await this.log('live', `⚠️ MOEX status=${status} — trading anyway (sandbox 24/7)`)
      }
      await this.log('live', `📊 ${this.bots.length} bots trading`)

      // FIX (rate-limit): PREFETCH all unique tickers BEFORE bots run.
      // Was: 9 bots × 11 tickers each doing getCandlesForTicker → even with 60s
      // cache, when cache expired all 9 bots sequentially refetched the same 11
      // tickers (99 gRPC calls/min burst). Now: prefetchTickers() fetches each
      // ticker ONCE in parallel (inflight-dedup inside PriceProvider), bots get
      // instant cache hits → ~11 gRPC/min instead of 99.
      const allTickers = new Set<string>()
      for (const bot of this.bots) {
        for (const t of bot.config.tickers) allTickers.add(t)
      }
      // All bots use 5min interval — prefetch once for 5min. (If a bot uses a
      // different interval, it will miss the cache and fetch on its own.)
      await this.prices.prefetchTickers(Array.from(allTickers), '5min')

      for (const bot of this.bots) {
        try {
          await bot.tick()
        } catch (e: any) {
          await this.log('error', `${bot.config.name} tick failed: ${e.message}`)
        }
      }
    } finally {
      this.busy = false
    }
  }

  start(): void {
    setInterval(() => this.tick().catch(e => this.log('error', `tick: ${e.message}`)), TICK_MS)
    // FIX (B8): was running log cleanup on EVERY log() call (COUNT + DELETE = 2 queries).
    // Now: cleanup once per 60 seconds via separate interval.
    setInterval(() => this.cleanupLogs().catch(() => {}), 60_000)
    this.log('info', `🚀 Engine started. ${this.bots.length} bots. PID=${process.pid}`)
  }

  private async cleanupLogs(): Promise<void> {
    try {
      const { db } = require('../lib/db')
      const logCount = await db.log.count()
      if (logCount > 100) {
        await db.$executeRaw`DELETE FROM Log WHERE id NOT IN (SELECT id FROM Log ORDER BY id DESC LIMIT 100)`
      }
    } catch {}
  }

  private async log(type: string, msg: string): Promise<void> {
    console.log(`[${new Date().toISOString()}] [${type}] ${msg}`)
    try {
      const { db } = require('../lib/db')
      await db.log.create({ data: { ts: Date.now(), type, msg } })
    } catch {}
  }
}


