'use client';

import { cn } from '@/lib/utils';
import type { Account } from '@/lib/types';
import {
  accountColor,
  fmtInt,
  fmtRub,
  pnlClass,
  T_BANK_YELLOW,
} from '@/lib/dashboard';
import { Server, User, Search } from 'lucide-react';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Badge } from '@/components/ui/badge';

interface Props {
  accounts: Account[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  query: string;
  onQuery: (q: string) => void;
}

export function AccountSidebar({
  accounts,
  selectedId,
  onSelect,
  query,
  onQuery,
}: Props) {
  const q = query.trim().toLowerCase();
  const filtered = q
    ? accounts.filter(
        (a) =>
          a.label.toLowerCase().includes(q) ||
          a.bots.some((b) => b.config.name.toLowerCase().includes(q)),
      )
    : accounts;

  return (
    <aside className="flex w-80 shrink-0 flex-col border-r border-neutral-200 bg-neutral-50">
      <div className="border-b border-neutral-200 p-3">
        <div className="mb-2 flex items-center justify-between">
          <h2 className="text-sm font-bold uppercase tracking-wide text-neutral-700">
            Аккаунты
          </h2>
          <Badge
            variant="secondary"
            className="bg-neutral-200 text-neutral-700"
          >
            {accounts.length}
          </Badge>
        </div>
        <div className="relative">
          <Search className="pointer-events-none absolute left-2.5 top-1/2 size-4 -translate-y-1/2 text-neutral-400" />
          <Input
            value={query}
            onChange={(e) => onQuery(e.target.value)}
            placeholder="Поиск аккаунта или бота…"
            className="h-9 bg-white pl-8 text-sm"
          />
        </div>
      </div>
      <ScrollArea className="flex-1">
        <ul className="space-y-1 p-2">
          {filtered.map((a) => {
            const active = a.id === selectedId;
            const accent = accountColor(a);
            return (
              <li key={a.id}>
                <button
                  type="button"
                  onClick={() => onSelect(a.id)}
                  className={cn(
                    'group flex w-full items-stretch gap-3 rounded-lg border p-2.5 text-left transition-all',
                    active
                      ? 'border-neutral-900 bg-white shadow-sm ring-1 ring-neutral-900'
                      : 'border-transparent hover:border-neutral-300 hover:bg-white',
                  )}
                >
                  <span
                    className="mt-0.5 flex size-9 shrink-0 items-center justify-center rounded-md text-white"
                    style={{
                      background:
                        a.type === 'shared' ? '#0A0A0A' : accent,
                    }}
                  >
                    {a.type === 'shared' ? (
                      <Server className="size-5" />
                    ) : (
                      <User className="size-5" />
                    )}
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="flex items-center justify-between gap-2">
                      <span className="truncate text-sm font-semibold text-neutral-900">
                        {a.label}
                      </span>
                      <span className="shrink-0 text-[11px] font-medium text-neutral-400">
                        #{a.index}
                      </span>
                    </span>
                    <span className="mt-0.5 flex items-center gap-1.5">
                      <Badge
                        variant="outline"
                        className={cn(
                          'h-4 px-1.5 text-[10px] font-semibold',
                          a.type === 'shared'
                            ? 'border-neutral-900 bg-neutral-900 text-white'
                            : 'border-neutral-300 bg-white text-neutral-600',
                        )}
                      >
                        {a.botCount} бот{a.botCount === 1 ? '' : a.botCount < 5 ? 'а' : 'ов'}
                      </Badge>
                      {a.type === 'shared' && (
                        <span
                          className="size-2 rounded-full"
                          style={{ background: T_BANK_YELLOW }}
                          title="shared broker account"
                        />
                      )}
                    </span>
                    <span className="mt-1.5 flex items-center justify-between text-[11px]">
                      <span className="text-neutral-500">
                        {fmtRub(a.totalBalance)}
                      </span>
                      <span className={cn('font-semibold tabular-nums', pnlClass(a.totalPnl))}>
                        {fmtRub(a.totalPnl, { sign: true })}
                      </span>
                    </span>
                    <span className="mt-0.5 flex items-center justify-between text-[10px] text-neutral-400">
                      <span>{fmtInt(a.totalTrades)} сделок</span>
                      <span>{a.bots.length} records</span>
                    </span>
                  </span>
                </button>
              </li>
            );
          })}
          {filtered.length === 0 && (
            <li className="px-2 py-8 text-center text-sm text-neutral-400">
              Ничего не найдено
            </li>
          )}
        </ul>
      </ScrollArea>
    </aside>
  );
}
