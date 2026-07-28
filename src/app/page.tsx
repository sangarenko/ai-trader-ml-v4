'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import type { Account, Bot, TraderState } from '@/lib/types';
import { deriveAccounts, fmtInt, fmtRub, sortBots } from '@/lib/dashboard';
import { StatsHeader } from '@/components/dashboard/stats-header';
import { AccountSidebar } from '@/components/dashboard/account-sidebar';
import { BotCard } from '@/components/dashboard/bot-card';
import { BotDetailDialog } from '@/components/dashboard/bot-detail';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from '@/components/ui/sheet';
import {
  Server,
  Users,
  ArrowLeft,
  SlidersHorizontal,
  Github,
  AlertCircle,
} from 'lucide-react';

const REFRESH_MS = 5000;
const ACCOUNTS_STORAGE_KEY = 'ai-trader:data-url';

export default function Home() {
  const [state, setState] = useState<TraderState | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [query, setQuery] = useState('');
  const [sortBy, setSortBy] = useState<'pnl' | 'trades' | 'balance' | 'name'>(
    'pnl',
  );
  const [openBot, setOpenBot] = useState<Bot | null>(null);
  const [mobileAccountsOpen, setMobileAccountsOpen] = useState(false);

  // data source: relative /api/state (works both in preview sandbox and on the
  // trader server where /api/state already exists).
  const dataUrl = '/api/state';

  const fetchState = useCallback(async () => {
    try {
      const res = await fetch(dataUrl, { cache: 'no-store' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = (await res.json()) as TraderState;
      setState(data);
      setLastUpdated(new Date());
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'fetch failed');
    } finally {
      setLoading(false);
    }
  }, [dataUrl]);

  useEffect(() => {
    fetchState();
    const id = setInterval(fetchState, REFRESH_MS);
    return () => clearInterval(id);
  }, [fetchState]);

  const accounts = useMemo<Account[]>(() => {
    if (!state) return [];
    return deriveAccounts(state);
  }, [state]);

  // auto-select first account once loaded
  useEffect(() => {
    if (accounts.length > 0 && !selectedId) {
      setSelectedId(accounts[accounts.length - 1].id); // default to the big shared account
    }
  }, [accounts, selectedId]);

  const selectedAccount = useMemo<Account | null>(
    () => accounts.find((a) => a.id === selectedId) ?? null,
    [accounts, selectedId],
  );

  const visibleBots = useMemo(() => {
    if (!selectedAccount) return [];
    return sortBots(selectedAccount.bots, sortBy);
  }, [selectedAccount, sortBy]);

  return (
    <div className="flex h-dvh flex-col bg-white text-neutral-900">
      <StatsHeader
        botCount={state?.bots?.length ?? 0}
        liveTradeCount={state?.liveTradeCount ?? 0}
        uptime={state?.uptime ?? 0}
        totalRealizedPnl={state?.totalRealizedPnl ?? 0}
        totalUnrealizedPnl={state?.totalUnrealizedPnl ?? 0}
        totalPnl={state?.totalPnl ?? 0}
        techMode={state?.techMode ?? false}
        lastUpdated={lastUpdated}
        loading={loading}
        onRefresh={fetchState}
      />

      <main className="mx-auto flex w-full max-w-[1800px] flex-1 overflow-hidden">
        {/* desktop sidebar */}
        <div className="hidden min-h-0 lg:flex">
          <AccountSidebar
            accounts={accounts}
            selectedId={selectedId}
            onSelect={setSelectedId}
            query={query}
            onQuery={setQuery}
          />
        </div>

        {/* mobile account picker */}
        <div className="block lg:hidden">
          <Sheet open={mobileAccountsOpen} onOpenChange={setMobileAccountsOpen}>
            <SheetTrigger asChild>
              <Button
                variant="outline"
                className="m-3 gap-2"
                size="sm"
              >
                <Users className="size-4" />
                Аккаунты
                {selectedAccount && (
                  <Badge variant="secondary" className="ml-1 bg-neutral-100">
                    {selectedAccount.index}
                  </Badge>
                )}
              </Button>
            </SheetTrigger>
            <SheetContent side="left" className="w-80 p-0">
              <SheetHeader className="border-b border-neutral-200 p-3">
                <SheetTitle className="text-sm">Аккаунты</SheetTitle>
              </SheetHeader>
              <AccountSidebar
                accounts={accounts}
                selectedId={selectedId}
                onSelect={(id) => {
                  setSelectedId(id);
                  setMobileAccountsOpen(false);
                }}
                query={query}
                onQuery={setQuery}
              />
            </SheetContent>
          </Sheet>
        </div>

        {/* bots panel */}
        <section className="flex min-h-0 min-w-0 flex-1 flex-col">
          {error && (
            <div className="m-4 flex items-center gap-2 rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">
              <AlertCircle className="size-4 shrink-0" />
              <span>
                Не удалось загрузить состояние: <code className="font-mono">{error}</code>.
                Проверьте, что <code className="font-mono">/api/state</code> доступен.
              </span>
            </div>
          )}

          {selectedAccount ? (
            <>
              <AccountHeader
                account={selectedAccount}
                sortBy={sortBy}
                onSort={setSortBy}
                onBackToAccounts={() => setMobileAccountsOpen(true)}
              />
              <div className="flex-1 overflow-y-auto p-4">
                {loading && visibleBots.length === 0 ? (
                  <BotGridSkeleton />
                ) : visibleBots.length === 0 ? (
                  <div className="flex h-full items-center justify-center text-sm text-neutral-400">
                    В этом аккаунте нет ботов
                  </div>
                ) : (
                  <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
                    {visibleBots.map((b) => (
                      <BotCard key={b.config.name} bot={b} onOpen={setOpenBot} />
                    ))}
                  </div>
                )}
              </div>
            </>
          ) : (
            <div className="flex flex-1 items-center justify-center text-sm text-neutral-400">
              {loading ? 'Загрузка…' : 'Выберите аккаунт слева'}
            </div>
          )}
        </section>
      </main>

      <BotDetailDialog bot={openBot} onOpenChange={(o) => !o && setOpenBot(null)} />

      <footer className="shrink-0 border-t border-neutral-200 bg-neutral-50">
        <div className="mx-auto flex max-w-[1800px] flex-wrap items-center justify-between gap-2 px-4 py-2.5 text-xs text-neutral-500">
          <div className="flex items-center gap-3">
            <span className="flex items-center gap-1.5">
              <Server className="size-3.5" />
              Источник: <code className="font-mono text-neutral-700">{dataUrl}</code>
            </span>
            <span>·</span>
            <span>обновление каждые {REFRESH_MS / 1000}с</span>
          </div>
          <div className="flex items-center gap-3">
            <span>
              аккаунтов: <strong className="text-neutral-700">{accounts.length}</strong>
            </span>
            <span>·</span>
            <span>
              ботов: <strong className="text-neutral-700">{state?.bots?.length ?? 0}</strong>
            </span>
            <span>·</span>
            <a
              href="https://tinkoff.github.io/investAPI/"
              target="_blank"
              rel="noreferrer"
              className="flex items-center gap-1 text-neutral-500 hover:text-neutral-900"
            >
              <Github className="size-3.5" />
              T-Invest API
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}

function AccountHeader({
  account,
  sortBy,
  onSort,
  onBackToAccounts,
}: {
  account: Account;
  sortBy: 'pnl' | 'trades' | 'balance' | 'name';
  onSort: (v: 'pnl' | 'trades' | 'balance' | 'name') => void;
  onBackToAccounts: () => void;
}) {
  const positive = account.totalPnl >= 0;
  return (
    <div className="sticky top-0 z-30 border-b border-neutral-200 bg-white/95 backdrop-blur">
      <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-3">
        <div className="flex min-w-0 items-center gap-3">
          <Button
            variant="ghost"
            size="sm"
            className="lg:hidden"
            onClick={onBackToAccounts}
          >
            <ArrowLeft className="size-4" />
          </Button>
          <span
            className="flex size-10 items-center justify-center rounded-lg text-white"
            style={{
              background:
                account.type === 'shared' ? '#0A0A0A' : account.bots[0]?.config.color,
            }}
          >
            {account.type === 'shared' ? (
              <Server className="size-5" />
            ) : (
              <Users className="size-5" />
            )}
          </span>
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <h2 className="truncate text-lg font-bold text-neutral-900">
                {account.label}
              </h2>
              <Badge
                variant="outline"
                className={
                  account.type === 'shared'
                    ? 'border-neutral-900 bg-neutral-900 text-white'
                    : 'border-neutral-300 text-neutral-600'
                }
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
                  <code className="hidden truncate font-mono text-[10px] text-neutral-400 sm:inline">
                    {account.accountId.slice(0, 8)}…
                  </code>
                </>
              )}
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <div
            className={`flex items-center gap-2 rounded-md border px-3 py-1.5 ${
              positive
                ? 'border-emerald-200 bg-emerald-50'
                : 'border-red-200 bg-red-50'
            }`}
          >
            <span className="text-[10px] uppercase tracking-wide text-neutral-500">
              P&L аккаунта
            </span>
            <span
              className={`font-mono text-sm font-bold tabular-nums ${
                positive ? 'text-emerald-700' : 'text-red-700'
              }`}
            >
              {fmtRub(account.totalPnl, { sign: true })}
            </span>
          </div>

          <div className="flex items-center gap-1.5">
            <SlidersHorizontal className="size-4 text-neutral-400" />
            <Select value={sortBy} onValueChange={(v) => onSort(v as typeof sortBy)}>
              <SelectTrigger className="h-9 w-36">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="pnl">По P&L</SelectItem>
                <SelectItem value="trades">По сделкам</SelectItem>
                <SelectItem value="balance">По балансу</SelectItem>
                <SelectItem value="name">По имени</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
      </div>
    </div>
  );
}

function BotGridSkeleton() {
  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
      {Array.from({ length: 8 }).map((_, i) => (
        <div
          key={i}
          className="h-56 animate-pulse rounded-xl border border-neutral-200 bg-neutral-50"
        />
      ))}
    </div>
  );
}
