'use client'

import { useEffect, useState, useCallback, useMemo } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import {
  RefreshCw, Server, Zap, TrendingUp, TrendingDown, Activity, Users,
  ArrowLeft, Search, X, Bot as BotIcon,
} from 'lucide-react'

const TBANK_YELLOW = '#FFDD2D'
const TBANK_BLACK = '#0A0A0A'
const TBANK_GREEN = '#0DBC4C'
const TBANK_RED = '#E53935'
const TBANK_DARK = '#1A1A1A'

// ---------- types ----------
interface Position {
  ticker: string; qty: number; entryPrice: number; side?: string
  currentPrice?: number; unrealizedPnl?: number
}
interface Trade {
  ts: number; side: string; ticker: string; qty: number
  price: number; pnl: number; balanceAfter?: number
}
interface BotStats {
  name: string; agentType: string; color: string; description: string
  liveBuys: number; liveSells: number; liveTrades: number
  realizedPnl: number; grossRealizedPnl: number; commission: number
  unrealizedPnl: number; totalPnl: number
  realBalance: number; realTotalValue: number
  openPositions: Position[]
  history: Trade[]
  lastSignal: { action: number; ticker: string; ts: number }
}
interface BotConfig {
  name: string; color: string; strategy: string; description: string
  tickers: string[]; tickerMode?: string; rotateIntervalSec?: number
  positionSize?: number; candleInterval?: string; maxPositionCost?: number
  accountId?: string | null; sharedAccount?: boolean; virtualBalance?: number
  filters?: Record<string, number>
}
interface Account {
  id: string; index: number; label: string
  type: 'shared' | 'standalone'; accountId?: string
  botCount: number; bots: BotConfig[]
  totalBalance: number; totalPnl: number; totalTrades: number
}

// ---------- helpers ----------
function fmtRub(v: number, sign = false): string {
  const s = (v || 0).toLocaleString('ru-RU', { minimumFractionDigits: 0, maximumFractionDigits: 0 })
  return `${sign && v > 0 ? '+' : ''}${s} ₽`
}
function fmtInt(v: number): string { return Math.round(v || 0).toLocaleString('ru-RU') }
function fmtTime(ts: number): string {
  try { return new Date(ts).toLocaleTimeString('ru-RU') } catch { return '—' }
}
function pnlColor(v: number): string { return v > 0 ? TBANK_GREEN : v < 0 ? TBANK_RED : '#737373' }

// Per-bot balance. Standalone bots use realTotalValue (their own broker balance,
// started at 10000). Shared-account bots use virtualBalance + totalPnl because
// realTotalValue on a shared account is the broker's total, not the bot's own.
function botBalance(bot: BotConfig, stats?: BotStats): number {
  if (bot.virtualBalance != null) {
    return bot.virtualBalance + (stats?.totalPnl ?? 0)
  }
  return stats?.realTotalValue ?? bot.virtualBalance ?? 10000
}

function deriveAccounts(bots: BotConfig[], stats: Record<string, BotStats>): Account[] {
  const standalone: BotConfig[] = []
  const byAccount = new Map<string, BotConfig[]>()
  for (const b of bots) {
    if (b.accountId) {
      const arr = byAccount.get(b.accountId) ?? []
      arr.push(b); byAccount.set(b.accountId, arr)
    } else standalone.push(b)
  }
  const out: Account[] = []
  let idx = 1
  for (const b of standalone) {
    const s = stats[b.name]
    out.push({
      id: `standalone-${idx}`, index: idx, type: 'standalone',
      label: b.name, botCount: 1, bots: [b],
      totalBalance: botBalance(b, s),
      totalPnl: s?.totalPnl ?? 0,
      totalTrades: s?.liveTrades ?? 0,
    })
    idx++
  }
  const groups = [...byAccount.values()].sort((a, b) => b.length - a.length)
  for (const g of groups) {
    const aid = g[0].accountId!
    let tb = 0, tp = 0, tt = 0
    for (const b of g) {
      const s = stats[b.name]
      tb += botBalance(b, s)
      tp += s?.totalPnl ?? 0
      tt += s?.liveTrades ?? 0
    }
    out.push({
      id: `shared-${aid.slice(0, 8)}`, index: idx, type: 'shared',
      accountId: aid, label: `Аккаунт ${idx} · shared`,
      botCount: g.length, bots: g,
      totalBalance: tb, totalPnl: tp, totalTrades: tt,
    })
    idx++
  }
  return out
}

type SortKey = 'pnl' | 'trades' | 'balance' | 'name'
function sortBots(bots: BotConfig[], stats: Record<string, BotStats>, by: SortKey): BotConfig[] {
  const arr = [...bots]
  const val = (b: BotConfig) => {
    const s = stats[b.name]
    if (!s) return 0
    if (by === 'pnl') return s.totalPnl ?? 0
    if (by === 'trades') return s.liveTrades ?? 0
    if (by === 'balance') return s.realTotalValue ?? 0
    return 0
  }
  arr.sort((a, b) => by === 'name' ? a.name.localeCompare(b.name) : val(b) - val(a))
  return arr
}

// ---------- main ----------
export default function Home() {
  const [state, setState] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [online, setOnline] = useState(false)
  const [selectedAccountId, setSelectedAccountId] = useState<string | null>(null)
  const [selectedBotName, setSelectedBotName] = useState<string | null>(null)
  const [query, setQuery] = useState('')
  const [sortBy, setSortBy] = useState<SortKey>('pnl')
  const [showLogs, setShowLogs] = useState(false)
  const [mobileAccountsOpen, setMobileAccountsOpen] = useState(false)

  // admin token + reset
  const [adminToken, setAdminToken] = useState<string>('')
  const [showTokenInput, setShowTokenInput] = useState(false)
  const [tokenInput, setTokenInput] = useState('')
  const [resetting, setResetting] = useState(false)
  const [showResetConfirm, setShowResetConfirm] = useState(false)
  const [resetResult, setResetResult] = useState<string>('')

  useEffect(() => {
    const t = typeof window !== 'undefined' ? localStorage.getItem('ai-trader-admin-token') || '' : ''
    setAdminToken(t)
  }, [])

  const fetchState = useCallback(async () => {
    try {
      const resp = await fetch('/api/state', { signal: AbortSignal.timeout(15000) })
      if (!resp.ok) throw new Error('HTTP ' + resp.status)
      const data = await resp.json()
      if (data.error) { setOnline(false); return }
      setState(data); setOnline(true)
    } catch { setOnline(false) }
    finally { setLoading(false) }
  }, [])

  const resetBalances = useCallback(async () => {
    setShowResetConfirm(false)
    if (!adminToken) { setShowTokenInput(true); setResetting(false); return }
    setResetting(true); setResetResult('')
    try {
      const resp = await fetch('/api/state', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Admin-Token': adminToken },
        body: JSON.stringify({ action: 'reset' }),
      })
      const data = await resp.json()
      setResetResult(data.ok ? '✅ ' + (data.reset || 'Reset done') : '❌ ' + (data.error || 'Reset failed'))
      setTimeout(() => { fetchState(); setResetResult('') }, 5000)
    } catch (e: any) {
      setResetResult('❌ Reset failed: ' + e.message)
      setTimeout(() => setResetResult(''), 5000)
    } finally { setResetting(false) }
  }, [fetchState, adminToken])

  const saveAdminToken = useCallback(() => {
    localStorage.setItem('ai-trader-admin-token', tokenInput)
    setAdminToken(tokenInput); setShowTokenInput(false); setTokenInput('')
    setTimeout(() => setShowResetConfirm(true), 100)
  }, [tokenInput])

  useEffect(() => {
    fetchState()
    const id = setInterval(fetchState, 5000)
    return () => clearInterval(id)
  }, [fetchState])

  const bots: BotConfig[] = state?.bots || []
  const stats: Record<string, BotStats> = state?.botLiveStats || {}
  const uptime = state ? Math.floor(state.uptime / 60) : 0
  const totalLiveTrades = state?.liveTradeCount || 0
  const totalPnL = Object.values(stats).reduce((s: number, b: any) => s + (b.totalPnl || 0), 0)

  const accounts = useMemo(() => deriveAccounts(bots, stats), [bots, stats])

  // auto-select the shared account (biggest, "10th account") on first load
  useEffect(() => {
    if (accounts.length > 0 && !selectedAccountId) {
      const shared = accounts.find(a => a.type === 'shared') || accounts[accounts.length - 1]
      setSelectedAccountId(shared.id)
    }
  }, [accounts, selectedAccountId])

  const selectedAccount = useMemo(
    () => accounts.find(a => a.id === selectedAccountId) ?? null,
    [accounts, selectedAccountId],
  )

  const visibleBots = useMemo(
    () => selectedAccount ? sortBots(selectedAccount.bots, stats, sortBy) : [],
    [selectedAccount, stats, sortBy],
  )

  const selectedBot = useMemo(
    () => selectedBotName ? bots.find(b => b.name === selectedBotName) ?? null : null,
    [bots, selectedBotName],
  )
  const selectedBotStats = selectedBotName ? stats[selectedBotName] : null

  const filteredAccounts = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!q) return accounts
    return accounts.filter(a =>
      a.label.toLowerCase().includes(q) ||
      a.bots.some(b => b.name.toLowerCase().includes(q)),
    )
  }, [accounts, query])

  return (
    <div className="flex min-h-screen flex-col bg-white text-neutral-900">
      {/* ===== header ===== */}
      <header className="sticky top-0 z-40 shadow-md" style={{ background: TBANK_YELLOW, borderBottom: '3px solid #0A0A0A' }}>
        <div className="max-w-[1800px] mx-auto px-4 py-3 flex items-center justify-between flex-wrap gap-3">
          <div className="flex items-center gap-3">
            <div className="w-11 h-11 rounded-xl flex items-center justify-center shadow-lg" style={{ background: TBANK_BLACK }}>
              <Server className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-black tracking-tight" style={{ color: TBANK_BLACK }}>AI Trader</h1>
              <p className="text-xs font-medium" style={{ color: TBANK_DARK }}>
                {online ? 'Online' : 'Offline'} | {uptime}мин | {totalLiveTrades} сделок | T-Bank Sandbox
                {state?.techMode && <span style={{ color: TBANK_RED, fontWeight: 800 }}> | 🔧 ТЕХ-РЕЖИМ {state.techMode.remaining}с</span>}
              </p>
            </div>
          </div>
          <div style={{ display: 'flex', gap: '8px' }}>
            <Button onClick={fetchState} size="sm" style={{ background: TBANK_BLACK, color: TBANK_YELLOW, fontWeight: 700, border: 'none' }}>
              <RefreshCw className="w-4 h-4 mr-1" /> Обновить
            </Button>
            <Button
              onClick={() => setShowResetConfirm(true)}
              disabled={resetting}
              size="sm"
              style={{ background: resetting ? '#666' : TBANK_RED, color: 'white', fontWeight: 700, border: 'none', opacity: resetting ? 0.6 : 1 }}
            >
              <RefreshCw className={'w-4 h-4 mr-1 ' + (resetting ? 'animate-spin' : '')} />
              {resetting ? 'Сброс...' : 'Сброс балансов'}
            </Button>
          </div>
        </div>

        {/* reset confirm modal */}
        {showResetConfirm && (
          <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.5)', zIndex: 1000, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <div style={{ background: 'white', padding: 24, borderRadius: 8, maxWidth: 400, textAlign: 'center' }}>
              <p style={{ fontSize: 16, marginBottom: 16, color: '#333' }}>⚠️ Точно хотите сбросить всем балансы до 10000₽?</p>
              <p style={{ fontSize: 12, color: '#999', marginBottom: 20 }}>Все позиции будут закрыты, сделки удалены.</p>
              <div style={{ display: 'flex', gap: 12, justifyContent: 'center' }}>
                <button onClick={() => setShowResetConfirm(false)} style={{ padding: '8px 20px', border: '1px solid #ccc', borderRadius: 4, background: 'white', cursor: 'pointer' }}>Отмена</button>
                <button onClick={resetBalances} style={{ padding: '8px 20px', border: 'none', borderRadius: 4, background: TBANK_RED, color: 'white', fontWeight: 'bold', cursor: 'pointer' }}>Да, сбросить</button>
              </div>
            </div>
          </div>
        )}

        {/* token input modal */}
        {showTokenInput && (
          <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.5)', zIndex: 1002, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <div style={{ background: 'white', padding: 24, borderRadius: 8, maxWidth: 420, width: '90%' }}>
              <h3 style={{ fontSize: 16, fontWeight: 800, marginBottom: 8, color: '#0A0A0A' }}>🔑 Введите админ-токен</h3>
              <p style={{ fontSize: 12, color: '#666', marginBottom: 16 }}>Токен нужен для сброса балансов. Сохранится в localStorage.</p>
              <input
                type="password" value={tokenInput}
                onChange={(e) => setTokenInput(e.target.value)}
                placeholder="ADMIN_TOKEN" autoFocus
                onKeyDown={(e) => { if (e.key === 'Enter' && tokenInput) saveAdminToken() }}
                style={{ width: '100%', padding: 10, border: '1px solid #ccc', borderRadius: 4, fontSize: 14, fontFamily: 'monospace', marginBottom: 16, boxSizing: 'border-box' }}
              />
              <div style={{ display: 'flex', gap: 12, justifyContent: 'flex-end' }}>
                <button onClick={() => { setShowTokenInput(false); setTokenInput('') }} style={{ padding: '8px 20px', border: '1px solid #ccc', borderRadius: 4, background: 'white', cursor: 'pointer' }}>Отмена</button>
                <button onClick={saveAdminToken} disabled={!tokenInput} style={{ padding: '8px 20px', border: 'none', borderRadius: 4, background: tokenInput ? '#0A0A0A' : '#999', color: TBANK_YELLOW, fontWeight: 'bold', cursor: tokenInput ? 'pointer' : 'not-allowed' }}>Сохранить</button>
              </div>
              {adminToken && <p style={{ fontSize: 11, color: TBANK_GREEN, marginTop: 12 }}>✓ Токен уже сохранён.</p>}
            </div>
          </div>
        )}

        {resetResult && (
          <div style={{ position: 'fixed', top: 20, right: 20, padding: '12px 20px', borderRadius: 6, background: resetResult.startsWith('✅') ? TBANK_GREEN : TBANK_RED, color: 'white', fontWeight: 'bold', zIndex: 1001, boxShadow: '0 4px 12px rgba(0,0,0,0.2)' }}>
            {resetResult}
          </div>
        )}
      </header>

      {/* tech-mode banner */}
      {state?.techMode && (
        <div style={{ background: TBANK_RED, color: 'white', padding: '12px 16px', textAlign: 'center', fontWeight: 800, fontSize: 14, borderBottom: '2px solid ' + TBANK_YELLOW }}>
          🔧 ТЕХ-РЕЖИМ: идёт полный сброс. Осталось ~{state.techMode.remaining}с. Дашборд может показывать устаревшие данные.
        </div>
      )}

      {/* mode switcher */}
      <div style={{ background: TBANK_BLACK, borderBottom: '2px solid ' + TBANK_YELLOW, padding: '8px 16px' }}>
        <div style={{ maxWidth: 1800, margin: '0 auto', display: 'flex', gap: 8, alignItems: 'center', justifyContent: 'center', flexWrap: 'wrap' }}>
          <a href="/" style={{ padding: '8px 18px', borderRadius: 6, textDecoration: 'none', fontWeight: 700, fontSize: 13, background: TBANK_GREEN, color: TBANK_BLACK, border: '2px solid ' + TBANK_GREEN }}>
            🟢 Песочница
          </a>
          <a href="/real" title="Real trading отключён в sandbox" style={{ padding: '8px 18px', borderRadius: 6, textDecoration: 'none', fontWeight: 700, fontSize: 13, background: '#222', color: '#666', border: '2px solid #333', opacity: 0.5, cursor: 'not-allowed' }} onClick={(e) => e.preventDefault()}>
            🔴 Реальные деньги (off)
          </a>
        </div>
      </div>

      {/* KPI strip */}
      <div className="border-b-4" style={{ background: TBANK_BLACK, borderColor: TBANK_YELLOW }}>
        <div className="max-w-[1800px] mx-auto px-4 py-3 grid grid-cols-2 md:grid-cols-5 gap-3">
          <KpiBox icon={<Zap className="w-4 h-4" />} label="Ботов" value={String(bots.length)} sub="активных" color={TBANK_YELLOW} />
          <KpiBox icon={<Users className="w-4 h-4" />} label="Аккаунтов" value={String(accounts.length)} sub={`${accounts.filter(a => a.type === 'shared').length} shared`} color="#06b6d4" />
          <KpiBox icon={<TrendingUp className="w-4 h-4" />} label="P&L" value={`${totalPnL >= 0 ? '+' : ''}${totalPnL.toFixed(0)}`} sub={`${(totalPnL / (bots.length * 10000) * 100).toFixed(1)}%`} color={totalPnL >= 0 ? TBANK_GREEN : TBANK_RED} />
          <KpiBox icon={<Activity className="w-4 h-4" />} label="Uptime" value={`${uptime}мин`} sub="работает" color="#06b6d4" />
          <KpiBox icon={<TrendingDown className="w-4 h-4" />} label="Статус" value={online ? 'LIVE' : 'OFF'} sub="sandbox" color={online ? TBANK_GREEN : TBANK_RED} />
        </div>
      </div>

      {loading && <div className="flex-1 flex items-center justify-center"><RefreshCw className="w-6 h-6 animate-spin mr-2" /> Загрузка...</div>}

      {!loading && !online && (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center p-8" style={{ background: '#F5F5F5', borderRadius: 12 }}>
            <Server className="w-12 h-12 mx-auto mb-3" style={{ color: TBANK_RED }} />
            <h2 className="text-lg font-bold mb-2">Worker оффлайн</h2>
            <Button onClick={fetchState} size="sm">Повторить</Button>
          </div>
        </div>
      )}

      {/* ===== main: master-detail ===== */}
      {!loading && online && state && (
        <main className="flex-1 mx-auto max-w-[1800px] w-full flex items-start">
          {/* mobile account picker trigger */}
          <div className="lg:hidden p-2 border-b border-neutral-200 bg-neutral-50 flex items-center gap-2 w-full">
            <Button variant="outline" size="sm" className="gap-2" onClick={() => setMobileAccountsOpen(true)}>
              <Users className="w-4 h-4" /> Аккаунты
              {selectedAccount && <Badge variant="secondary" className="ml-1 bg-neutral-200">#{selectedAccount.index}</Badge>}
            </Button>
            <span className="text-sm truncate flex-1">{selectedAccount?.label}</span>
            <Button variant="outline" size="sm" onClick={() => setShowLogs(s => !s)}>
              Логи
            </Button>
          </div>

          {/* mobile accounts sheet */}
          {mobileAccountsOpen && (
            <div className="lg:hidden fixed inset-0 z-50 flex">
              <div className="absolute inset-0 bg-black/50" onClick={() => setMobileAccountsOpen(false)} />
              <div className="relative z-10 w-80 max-w-[85vw] bg-white h-full flex flex-col shadow-xl">
                <div className="flex items-center justify-between p-3 border-b">
                  <span className="font-bold">Аккаунты</span>
                  <button onClick={() => setMobileAccountsOpen(false)}><X className="w-4 h-4" /></button>
                </div>
                <AccountList
                  accounts={filteredAccounts} stats={stats}
                  selectedId={selectedAccountId}
                  onSelect={(id) => { setSelectedAccountId(id); setSelectedBotName(null); setMobileAccountsOpen(false) }}
                  query={query} onQuery={setQuery}
                />
              </div>
            </div>
          )}

          {/* desktop accounts sidebar — sticky under header */}
          <aside className="hidden lg:flex flex-col w-80 shrink-0 border-r border-neutral-200 bg-neutral-50 sticky top-[246px] max-h-[calc(100vh-246px)] overflow-hidden">
            <AccountList
              accounts={filteredAccounts} stats={stats}
              selectedId={selectedAccountId}
              onSelect={(id) => { setSelectedAccountId(id); setSelectedBotName(null) }}
              query={query} onQuery={setQuery}
            />
          </aside>

          {/* bots panel — scrolls with the page */}
          <section className="flex-1 min-w-0 flex flex-col">
            {selectedAccount ? (
              selectedBot ? (
                <BotDetail
                  bot={selectedBot} stats={selectedBotStats}
                  onBack={() => setSelectedBotName(null)}
                />
              ) : (
                <>
                  <AccountHeader
                    account={selectedAccount}
                    sortBy={sortBy} onSort={setSortBy}
                    onToggleLogs={() => setShowLogs(s => !s)}
                  />
                  <div className="p-4">
                    {visibleBots.length === 0 ? (
                      <div className="text-center text-neutral-400 py-12 text-sm">В этом аккаунте нет ботов</div>
                    ) : (
                      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4 gap-3">
                        {visibleBots.map(b => (
                          <BotCard key={b.name} bot={b} stats={stats[b.name]} onOpen={() => setSelectedBotName(b.name)} />
                        ))}
                      </div>
                    )}
                  </div>
                </>
              )
            ) : (
              <div className="flex-1 flex items-center justify-center text-neutral-400 text-sm">Выберите аккаунт слева</div>
            )}
          </section>

          {/* logs panel (desktop xl+) */}
          {showLogs && (
            <aside className="hidden xl:flex flex-col w-80 shrink-0 border-l border-neutral-200 bg-neutral-50 sticky top-[246px] max-h-[calc(100vh-246px)] overflow-hidden">
              <LogsPanel logs={state.logs || []} onClose={() => setShowLogs(false)} />
            </aside>
          )}
        </main>
      )}

      {/* mobile logs sheet */}
      {!loading && online && showLogs && (
        <div className="xl:hidden fixed inset-0 z-50 flex items-end">
          <div className="absolute inset-0 bg-black/50" onClick={() => setShowLogs(false)} />
          <div className="relative z-10 w-full h-2/3 bg-white flex flex-col shadow-xl rounded-t-xl">
            <LogsPanel logs={state?.logs || []} onClose={() => setShowLogs(false)} />
          </div>
        </div>
      )}

      {/* ===== footer ===== */}
      <footer className="mt-auto border-t-2 py-2 px-4" style={{ background: TBANK_BLACK, borderColor: TBANK_YELLOW }}>
        <div className="max-w-[1800px] mx-auto text-[11px] flex justify-between flex-wrap gap-2" style={{ color: TBANK_YELLOW }}>
          <span className="font-bold">AI Trader | {bots.length} ботов | {accounts.length} аккаунтов | Worker 24/7</span>
          <span>{totalLiveTrades} live-сделок | обновление каждые 5с</span>
        </div>
      </footer>
    </div>
  )
}

// ---------- account list ----------
function AccountList({
  accounts, stats, selectedId, onSelect, query, onQuery,
}: {
  accounts: Account[]
  stats: Record<string, BotStats>
  selectedId: string | null
  onSelect: (id: string) => void
  query: string
  onQuery: (q: string) => void
}) {
  return (
    <>
      <div className="p-3 border-b border-neutral-200 bg-white">
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-sm font-bold uppercase tracking-wide text-neutral-700">Аккаунты</h2>
          <Badge variant="secondary" className="bg-neutral-200 text-neutral-700">{accounts.length}</Badge>
        </div>
        <div className="relative">
          <Search className="pointer-events-none absolute left-2.5 top-1/2 w-4 h-4 -translate-y-1/2 text-neutral-400" />
          <input
            value={query}
            onChange={(e) => onQuery(e.target.value)}
            placeholder="Поиск аккаунта или бота…"
            className="w-full h-9 pl-8 pr-3 text-sm bg-neutral-50 border border-neutral-200 rounded-md focus:outline-none focus:ring-2 focus:ring-neutral-900"
          />
        </div>
      </div>
      <ScrollArea className="flex-1 max-h-[calc(100vh-340px)]">
        <ul className="space-y-1 p-2">
          {accounts.map(a => {
            const active = a.id === selectedId
            return (
              <li key={a.id}>
                <button
                  type="button"
                  onClick={() => onSelect(a.id)}
                  className="w-full flex items-stretch gap-3 rounded-lg border p-2.5 text-left transition-all"
                  style={{
                    borderColor: active ? '#0A0A0A' : 'transparent',
                    background: active ? '#fff' : 'transparent',
                    boxShadow: active ? '0 0 0 1px #0A0A0A' : 'none',
                  }}
                >
                  <span
                    className="mt-0.5 flex size-9 shrink-0 items-center justify-center rounded-md text-white"
                    style={{ background: a.type === 'shared' ? TBANK_BLACK : (a.bots[0]?.color || '#737373') }}
                  >
                    {a.type === 'shared' ? <Server className="w-5 h-5" /> : <BotIcon className="w-5 h-5" />}
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="flex items-center justify-between gap-2">
                      <span className="truncate text-sm font-semibold text-neutral-900">{a.label}</span>
                      <span className="shrink-0 text-[11px] font-medium text-neutral-400">#{a.index}</span>
                    </span>
                    <span className="mt-0.5 flex items-center gap-1.5">
                      <Badge
                        variant="outline"
                        className="h-4 px-1.5 text-[10px] font-semibold"
                        style={{
                          borderColor: a.type === 'shared' ? '#0A0A0A' : '#d4d4d4',
                          background: a.type === 'shared' ? '#0A0A0A' : '#fff',
                          color: a.type === 'shared' ? '#fff' : '#525252',
                        }}
                      >
                        {a.botCount} бот{a.botCount === 1 ? '' : a.botCount < 5 ? 'а' : 'ов'}
                      </Badge>
                      {a.type === 'shared' && (
                        <span className="size-2 rounded-full" style={{ background: TBANK_YELLOW }} title="shared broker account" />
                      )}
                    </span>
                    <span className="mt-1.5 flex items-center justify-between text-[11px]">
                      <span className="text-neutral-500">{fmtRub(a.totalBalance)}</span>
                      <span className="font-semibold tabular-nums" style={{ color: pnlColor(a.totalPnl) }}>
                        {fmtRub(a.totalPnl, true)}
                      </span>
                    </span>
                    <span className="mt-0.5 flex items-center justify-between text-[10px] text-neutral-400">
                      <span>{fmtInt(a.totalTrades)} сделок</span>
                    </span>
                  </span>
                </button>
              </li>
            )
          })}
          {accounts.length === 0 && (
            <li className="px-2 py-8 text-center text-sm text-neutral-400">Ничего не найдено</li>
          )}
        </ul>
      </ScrollArea>
    </>
  )
}

// ---------- account header ----------
function AccountHeader({
  account, sortBy, onSort, onToggleLogs,
}: {
  account: Account
  sortBy: SortKey
  onSort: (v: SortKey) => void
  onToggleLogs: () => void
}) {
  const positive = account.totalPnl >= 0
  return (
    <div className="sticky top-0 z-30 border-b border-neutral-200 bg-white/95 backdrop-blur">
      <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-3">
        <div className="flex min-w-0 items-center gap-3">
          <span
            className="flex size-10 items-center justify-center rounded-lg text-white"
            style={{ background: account.type === 'shared' ? TBANK_BLACK : (account.bots[0]?.color || '#737373') }}
          >
            {account.type === 'shared' ? <Server className="w-5 h-5" /> : <BotIcon className="w-5 h-5" />}
          </span>
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <h2 className="truncate text-lg font-bold text-neutral-900">{account.label}</h2>
              <Badge
                variant="outline"
                style={{
                  borderColor: account.type === 'shared' ? '#0A0A0A' : '#d4d4d4',
                  background: account.type === 'shared' ? '#0A0A0A' : '#fff',
                  color: account.type === 'shared' ? '#fff' : '#525252',
                }}
              >
                {account.type === 'shared' ? 'shared broker' : 'standalone'}
              </Badge>
            </div>
            <div className="mt-0.5 flex items-center gap-3 text-xs text-neutral-500">
              <span>{account.botCount} ботов</span>
              <span>·</span>
              <span>{fmtRub(account.totalBalance)}</span>
              <span>·</span>
              <span>{fmtInt(account.totalTrades)} сделок</span>
              {account.accountId && (
                <>
                  <span className="hidden sm:inline">·</span>
                  <code className="hidden sm:inline font-mono text-[10px] text-neutral-400 truncate">{account.accountId.slice(0, 8)}…</code>
                </>
              )}
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <div
            className="flex items-center gap-2 rounded-md border px-3 py-1.5"
            style={{
              borderColor: positive ? '#bbf7d0' : '#fecaca',
              background: positive ? '#f0fdf4' : '#fef2f2',
            }}
          >
            <span className="text-[10px] uppercase tracking-wide text-neutral-500">P&L аккаунта</span>
            <span className="font-mono text-sm font-bold tabular-nums" style={{ color: positive ? TBANK_GREEN : TBANK_RED }}>
              {fmtRub(account.totalPnl, true)}
            </span>
          </div>

          <label className="flex items-center gap-1.5 text-xs text-neutral-500">
            <span className="hidden sm:inline">Сорт.</span>
            <select
              value={sortBy}
              onChange={(e) => onSort(e.target.value as SortKey)}
              className="h-9 rounded-md border border-neutral-200 bg-white px-2 text-sm focus:outline-none focus:ring-2 focus:ring-neutral-900"
            >
              <option value="pnl">По P&L</option>
              <option value="trades">По сделкам</option>
              <option value="balance">По балансу</option>
              <option value="name">По имени</option>
            </select>
          </label>

          <Button variant="outline" size="sm" onClick={onToggleLogs}>Логи</Button>
        </div>
      </div>
    </div>
  )
}

// ---------- bot card ----------
function BotCard({ bot, stats, onOpen }: { bot: BotConfig; stats?: BotStats; onOpen: () => void }) {
  const pnl = stats?.totalPnl ?? 0
  const trades = stats?.liveTrades ?? 0
  const balance = botBalance(bot, stats)
  const baseline = bot.virtualBalance ?? 10000
  const diff = balance - baseline
  const positions = stats?.openPositions?.length ?? 0
  const lastSig = stats?.lastSignal
  const sigText = lastSig?.action === 1 ? 'BUY' : lastSig?.action === 2 ? 'SELL' : lastSig?.action === 3 ? 'CLOSE' : 'HOLD'
  const sigColor = lastSig?.action === 1 ? TBANK_GREEN : lastSig?.action === 2 ? TBANK_RED : '#999'
  const balanceColor = diff > 0 ? TBANK_GREEN : diff < 0 ? TBANK_RED : '#525252'

  return (
    <button
      type="button"
      onClick={onOpen}
      className="group relative flex w-full flex-col overflow-hidden rounded-xl border border-neutral-200 bg-white text-left shadow-sm transition-all hover:-translate-y-0.5 hover:border-neutral-400 hover:shadow-md focus:outline-none focus:ring-2 focus:ring-neutral-900"
    >
      <span className="h-1 w-full" style={{ background: bot.color }} />
      <div className="flex flex-col gap-3 p-4">
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0">
            <h3 className="truncate text-base font-bold text-neutral-900">{bot.name}</h3>
            <p className="mt-0.5 line-clamp-1 text-xs text-neutral-500">{bot.description}</p>
          </div>
          {lastSig && (
            <Badge variant="outline" className="shrink-0 text-[10px]" style={{ borderColor: sigColor, color: sigColor }}>
              {sigText} {lastSig.ticker || ''}
            </Badge>
          )}
        </div>

        <div className="flex items-center gap-1.5">
          <Badge variant="secondary" className="bg-neutral-100 font-mono text-[10px] text-neutral-700">
            {bot.strategy}
          </Badge>
          {bot.candleInterval && (
            <Badge variant="outline" className="font-mono text-[10px] text-neutral-600">{bot.candleInterval}</Badge>
          )}
          {bot.tickers.length > 0 && (
            <span className="truncate text-[10px] text-neutral-400">
              {bot.tickers.slice(0, 3).join(' · ')}{bot.tickers.length > 3 && ` +${bot.tickers.length - 3}`}
            </span>
          )}
        </div>

        <div className="grid grid-cols-2 gap-2">
          <MiniMetric label="Баланс" value={fmtRub(balance)} valueColor={balanceColor} />
          <MiniMetric label="P&L всего" value={fmtRub(pnl, true)} valueColor={pnlColor(pnl)} />
          <MiniMetric label="Сделок" value={fmtInt(trades)} icon={<Activity className="w-3 h-3" />} />
          <MiniMetric label="Позиций" value={String(positions)} icon={positions > 0 ? <TrendingUp className="w-3 h-3 text-emerald-600" /> : <TrendingDown className="w-3 h-3 text-neutral-400" />} />
        </div>

        <div
          className="flex items-center justify-between rounded-md border px-2.5 py-1.5 text-xs font-semibold"
          style={{
            borderColor: diff > 0 ? '#bbf7d0' : diff < 0 ? '#fecaca' : '#e5e5e5',
            background: diff > 0 ? '#f0fdf4' : diff < 0 ? '#fef2f2' : '#fafafa',
            color: diff > 0 ? '#047857' : diff < 0 ? '#b91c1c' : '#525252',
          }}
        >
          <span>{diff > 0 ? '▲ Заработал' : diff < 0 ? '▼ Слил' : '— Старт'}</span>
          <span className="tabular-nums">{fmtRub(diff, true)} ({((diff / baseline) * 100).toFixed(2)}%)</span>
        </div>
      </div>
    </button>
  )
}

function MiniMetric({ label, value, valueColor, icon }: { label: string; value: string; valueColor?: string; icon?: React.ReactNode }) {
  return (
    <div className="rounded-md bg-neutral-50 px-2.5 py-1.5">
      <div className="flex items-center gap-1 text-[10px] uppercase tracking-wide text-neutral-400">{icon}{label}</div>
      <div className="mt-0.5 truncate font-mono text-sm font-semibold tabular-nums text-neutral-900" style={valueColor ? { color: valueColor } : undefined}>{value}</div>
    </div>
  )
}

// ---------- bot detail ----------
function BotDetail({ bot, stats, onBack }: { bot: BotConfig; stats?: BotStats; onBack: () => void }) {
  const pnl = stats?.totalPnl ?? 0
  const realized = stats?.grossRealizedPnl ?? stats?.realizedPnl ?? 0
  const unrealized = stats?.unrealizedPnl ?? 0
  return (
    <>
      <div className="sticky top-0 z-30 border-b border-neutral-200 bg-white">
        <div className="flex items-center gap-3 px-4 py-3">
          <Button variant="ghost" size="sm" onClick={onBack}><ArrowLeft className="w-4 h-4 mr-1" /> Назад</Button>
          <span className="size-3 rounded-full" style={{ background: bot.color }} />
          <div className="min-w-0 flex-1">
            <h2 className="truncate text-lg font-bold text-neutral-900">{bot.name}</h2>
            <p className="text-xs text-neutral-500 truncate">{bot.description}</p>
          </div>
          <div className="flex gap-2">
            <DetailStat label="Баланс" value={fmtRub(stats?.realTotalValue ?? bot.virtualBalance ?? 0)} />
            <DetailStat label="P&L" value={fmtRub(pnl, true)} color={pnlColor(pnl)} />
          </div>
        </div>
      </div>
      <ScrollArea className="flex-1">
        <div className="space-y-4 p-4">
          {/* config */}
          <Card>
            <CardHeader className="pb-2"><CardTitle className="text-xs uppercase tracking-wide">Конфигурация</CardTitle></CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2">
                <KV label="Стратегия" value={bot.strategy} mono />
                <KV label="Интервал" value={bot.candleInterval || '—'} mono />
                <KV label="Position size" value={bot.positionSize ? `${(bot.positionSize * 100).toFixed(0)}%` : '—'} />
                <KV label="Max cost" value={fmtRub(bot.maxPositionCost || 0)} />
                <KV label="Ticker mode" value={bot.tickerMode || '—'} mono />
                <KV label="Rotate" value={bot.rotateIntervalSec ? `${bot.rotateIntervalSec}s` : '—'} mono />
                <KV label="Сделок" value={fmtInt(stats?.liveTrades ?? 0)} />
                <KV label="Комиссия" value={fmtRub(stats?.commission ?? 0)} valueColor={TBANK_RED} />
              </div>
              {bot.tickers.length > 0 && (
                <div className="mt-3">
                  <div className="mb-1 text-[11px] uppercase tracking-wide text-neutral-400">Тикеры ({bot.tickers.length})</div>
                  <div className="flex flex-wrap gap-1">
                    {bot.tickers.map(t => (
                      <Badge key={t} variant="outline" className="font-mono text-[10px] text-neutral-600">{t}</Badge>
                    ))}
                  </div>
                </div>
              )}
              {bot.filters && (
                <div className="mt-3 grid grid-cols-2 sm:grid-cols-5 gap-2">
                  <KV label="Hold ticks" value={String(bot.filters.holdTicks ?? '—')} mono />
                  <KV label="Cooldown" value={String(bot.filters.cooldownTicks ?? '—')} mono />
                  <KV label="Comm mult" value={String(bot.filters.commFilterMult ?? '—')} mono />
                  <KV label="Trades/hr" value={String(bot.filters.maxTradesPerHour ?? '—')} mono />
                  <KV label="Max hold hr" value={String(bot.filters.maxHoldHours ?? '—')} mono />
                </div>
              )}
              <div className="mt-3 grid grid-cols-3 gap-2 text-center text-xs">
                <div className="rounded-md bg-neutral-50 p-2">
                  <div className="text-[10px] uppercase text-neutral-400">Realized</div>
                  <div className="font-mono font-bold" style={{ color: pnlColor(realized) }}>{fmtRub(realized, true)}</div>
                </div>
                <div className="rounded-md bg-neutral-50 p-2">
                  <div className="text-[10px] uppercase text-neutral-400">Unrealized</div>
                  <div className="font-mono font-bold" style={{ color: pnlColor(unrealized) }}>{fmtRub(unrealized, true)}</div>
                </div>
                <div className="rounded-md bg-neutral-50 p-2">
                  <div className="text-[10px] uppercase text-neutral-400">Cash</div>
                  <div className="font-mono font-bold">{fmtRub(stats?.realBalance ?? 0)}</div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* open positions */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-xs uppercase tracking-wide flex items-center gap-2">
                Открытые позиции
                <Badge variant="secondary" className="bg-neutral-100">{stats?.openPositions?.length ?? 0}</Badge>
              </CardTitle>
            </CardHeader>
            <CardContent>
              {stats?.openPositions && stats.openPositions.length > 0 ? (
                <div className="overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="border-b text-left text-neutral-400">
                        <th className="py-1 pr-2">Тикер</th><th className="py-1 pr-2">Сторона</th>
                        <th className="py-1 pr-2 text-right">Кол</th><th className="py-1 pr-2 text-right">Вход</th>
                        <th className="py-1 pr-2 text-right">Тек</th><th className="py-1 pr-2 text-right">P&L</th>
                      </tr>
                    </thead>
                    <tbody>
                      {stats.openPositions.map((p, i) => (
                        <tr key={i} className="border-b border-neutral-100">
                          <td className="py-1 pr-2 font-mono font-semibold">{p.ticker}</td>
                          <td className="py-1 pr-2">
                            <span style={{ color: p.side === 'short' ? TBANK_RED : TBANK_GREEN, fontWeight: 700 }}>
                              {(p.side || '').toUpperCase()}
                            </span>
                          </td>
                          <td className="py-1 pr-2 text-right font-mono">{p.qty}</td>
                          <td className="py-1 pr-2 text-right font-mono">{p.entryPrice}</td>
                          <td className="py-1 pr-2 text-right font-mono">{p.currentPrice ?? p.entryPrice}</td>
                          <td className="py-1 pr-2 text-right font-mono font-bold" style={{ color: pnlColor(p.unrealizedPnl || 0) }}>
                            {fmtRub(p.unrealizedPnl || 0, true)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="rounded-md border border-dashed border-neutral-200 py-6 text-center text-sm text-neutral-400">Нет открытых позиций</div>
              )}
            </CardContent>
          </Card>

          {/* recent trades */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-xs uppercase tracking-wide flex items-center gap-2">
                Последние сделки
                <Badge variant="secondary" className="bg-neutral-100">{stats?.history?.length ?? 0}</Badge>
              </CardTitle>
            </CardHeader>
            <CardContent>
              {stats?.history && stats.history.length > 0 ? (
                <div className="overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="border-b text-left text-neutral-400">
                        <th className="py-1 pr-2">Время</th><th className="py-1 pr-2">Сторона</th>
                        <th className="py-1 pr-2">Тикер</th><th className="py-1 pr-2 text-right">Кол</th>
                        <th className="py-1 pr-2 text-right">Цена</th><th className="py-1 pr-2 text-right">P&L</th>
                        <th className="py-1 pr-2 text-right">Баланс</th>
                      </tr>
                    </thead>
                    <tbody>
                      {stats.history.map((t, i) => (
                        <tr key={i} className="border-b border-neutral-100">
                          <td className="py-1 pr-2 text-neutral-400">{fmtTime(t.ts)}</td>
                          <td className="py-1 pr-2">
                            <span className="font-mono font-semibold" style={{ color: t.side.includes('SHORT') ? TBANK_RED : t.side.includes('CLOSE') ? '#737373' : TBANK_GREEN }}>
                              {t.side}
                            </span>
                          </td>
                          <td className="py-1 pr-2 font-mono font-semibold">{t.ticker}</td>
                          <td className="py-1 pr-2 text-right font-mono">{t.qty}</td>
                          <td className="py-1 pr-2 text-right font-mono">{t.price}</td>
                          <td className="py-1 pr-2 text-right font-mono font-bold" style={{ color: pnlColor(t.pnl) }}>{fmtRub(t.pnl, true)}</td>
                          <td className="py-1 pr-2 text-right font-mono text-neutral-500">{t.balanceAfter != null ? fmtRub(t.balanceAfter) : '—'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="rounded-md border border-dashed border-neutral-200 py-6 text-center text-sm text-neutral-400">Нет истории сделок</div>
              )}
            </CardContent>
          </Card>
        </div>
      </ScrollArea>
    </>
  )
}

function DetailStat({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="rounded-md border border-neutral-200 bg-neutral-50 px-2.5 py-1 text-right">
      <div className="text-[10px] uppercase tracking-wide text-neutral-400">{label}</div>
      <div className="font-mono text-sm font-bold tabular-nums" style={{ color: color || '#0A0A0A' }}>{value}</div>
    </div>
  )
}

function KV({ label, value, mono, valueColor }: { label: string; value: string; mono?: boolean; valueColor?: string }) {
  return (
    <div className="rounded-md border border-neutral-100 bg-white px-2.5 py-1.5">
      <div className="text-[10px] uppercase tracking-wide text-neutral-400">{label}</div>
      <div className={`mt-0.5 truncate text-sm font-semibold ${mono ? 'font-mono' : ''}`} style={valueColor ? { color: valueColor } : undefined}>{value}</div>
    </div>
  )
}

// ---------- logs panel ----------
function LogsPanel({ logs, onClose }: { logs: any[]; onClose: () => void }) {
  const filtered = (logs || []).filter((log: any) => {
    const m = log.msg || ''
    if (m.includes('bots trading')) return false
    if (m.startsWith('⚠️ MOEX status')) return false
    return true
  })
  return (
    <>
      <div className="flex items-center justify-between p-3 border-b border-neutral-200 bg-white">
        <h3 className="text-sm font-bold uppercase tracking-wide text-neutral-700">Лог событий</h3>
        <button onClick={onClose} className="text-neutral-400 hover:text-neutral-700"><X className="w-4 h-4" /></button>
      </div>
      <ScrollArea className="flex-1">
        <div className="space-y-1 p-2">
          {filtered.map((log: any, i: number) => (
            <div key={i} className="text-[10px] font-mono p-1 rounded" style={{
              background: log.type === 'error' ? 'rgba(229,57,53,0.08)' : log.type === 'live' ? 'rgba(13,188,76,0.05)' : 'transparent',
            }}>
              <span className="text-neutral-400">{fmtTime(log.ts)}</span>{' '}
              <span style={{ color: log.type === 'error' ? TBANK_RED : log.type === 'live' ? TBANK_GREEN : TBANK_DARK }}>
                {log.msg}
              </span>
            </div>
          ))}
          {filtered.length === 0 && <div className="text-neutral-400 text-center py-4 text-[10px]">Лог пуст или только routine-события</div>}
        </div>
      </ScrollArea>
    </>
  )
}

// ---------- KPI box (kept from original) ----------
function KpiBox({ icon, label, value, sub, color }: { icon: React.ReactNode; label: string; value: string; sub: string; color: string }) {
  return (
    <div className="rounded-lg p-3" style={{ background: 'rgba(255,221,45,0.05)', border: '1px solid rgba(255,221,45,0.2)' }}>
      <div className="flex items-center gap-2 mb-1">
        <span style={{ color }}>{icon}</span>
        <span className="text-[10px] uppercase font-bold text-neutral-400">{label}</span>
      </div>
      <div className="text-lg font-black font-mono" style={{ color }}>{value}</div>
      <div className="text-[10px] text-neutral-500">{sub}</div>
    </div>
  )
}
