'use client';

import { cn } from '@/lib/utils';
import type { Bot } from '@/lib/types';
import {
  botPnlPct,
  fmtInt,
  fmtPct,
  fmtRub,
  pnlBgClass,
  pnlClass,
} from '@/lib/dashboard';
import { Badge } from '@/components/ui/badge';
import { Activity, TrendingDown, TrendingUp } from 'lucide-react';

interface Props {
  bot: Bot;
  onOpen: (bot: Bot) => void;
}

export function BotCard({ bot, onOpen }: Props) {
  const { config, stats } = bot;
  const pnl = stats?.totalPnl ?? 0;
  const pnlPct = botPnlPct(bot);
  const open = stats?.openPositions?.length ?? 0;
  const trades = stats?.liveTrades ?? 0;
  const balance = stats?.realTotalValue ?? config.virtualBalance ?? 0;
  const positive = pnl >= 0;

  return (
    <button
      type="button"
      onClick={() => onOpen(bot)}
      className="group relative flex w-full flex-col overflow-hidden rounded-xl border border-neutral-200 bg-white text-left shadow-sm transition-all hover:-translate-y-0.5 hover:border-neutral-400 hover:shadow-md focus:outline-none focus:ring-2 focus:ring-neutral-900"
    >
      {/* color stripe */}
      <span
        className="h-1 w-full"
        style={{ background: config.color }}
      />
      <div className="flex flex-col gap-3 p-4">
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0">
            <h3 className="truncate text-base font-bold text-neutral-900">
              {config.name}
            </h3>
            <p className="mt-0.5 line-clamp-1 text-xs text-neutral-500">
              {config.description}
            </p>
          </div>
          <Badge
            variant="outline"
            className="shrink-0 border-neutral-300 bg-neutral-50 font-mono text-[10px] text-neutral-600"
          >
            {config.candleInterval}
          </Badge>
        </div>

        <div className="flex items-center gap-1.5">
          <Badge
            variant="secondary"
            className="bg-neutral-100 font-mono text-[10px] text-neutral-700"
          >
            {config.strategy}
          </Badge>
          {config.tickers.length > 0 && (
            <span className="truncate text-[10px] text-neutral-400">
              {config.tickers.slice(0, 3).join(' · ')}
              {config.tickers.length > 3 && ` +${config.tickers.length - 3}`}
            </span>
          )}
        </div>

        <div className="grid grid-cols-2 gap-2">
          <Metric label="Баланс" value={fmtRub(balance)} />
          <Metric
            label="P&L всего"
            value={fmtRub(pnl, { sign: true })}
            valueClass={pnlClass(pnl)}
          />
          <Metric
            label="Сделок"
            value={fmtInt(trades)}
            icon={<Activity className="size-3" />}
          />
          <Metric
            label="Позиций"
            value={String(open)}
            icon={
              open > 0 ? (
                <TrendingUp className="size-3 text-emerald-600" />
              ) : (
                <TrendingDown className="size-3 text-neutral-400" />
              )
            }
          />
        </div>

        <div
          className={cn(
            'flex items-center justify-between rounded-md border px-2.5 py-1.5 text-xs font-semibold',
            pnlBgClass(pnl),
          )}
        >
          <span>Доходность</span>
          <span className="tabular-nums">{fmtPct(pnlPct)}</span>
        </div>
      </div>
    </button>
  );
}

function Metric({
  label,
  value,
  valueClass,
  icon,
}: {
  label: string;
  value: string;
  valueClass?: string;
  icon?: React.ReactNode;
}) {
  return (
    <div className="rounded-md bg-neutral-50 px-2.5 py-1.5">
      <div className="flex items-center gap-1 text-[10px] uppercase tracking-wide text-neutral-400">
        {icon}
        {label}
      </div>
      <div
        className={cn(
          'mt-0.5 truncate font-mono text-sm font-semibold tabular-nums text-neutral-900',
          valueClass,
        )}
      >
        {value}
      </div>
    </div>
  );
}
