'use client';

import { fmtInt, fmtRub, fmtUptime, pnlClass } from '@/lib/dashboard';
import { Badge } from '@/components/ui/badge';
import { Bot, Clock, RefreshCw, TrendingUp, Zap } from 'lucide-react';

interface Props {
  botCount: number;
  liveTradeCount: number;
  uptime: number;
  totalRealizedPnl: number;
  totalUnrealizedPnl: number;
  totalPnl: number;
  techMode: boolean;
  lastUpdated: Date | null;
  loading: boolean;
  onRefresh: () => void;
}

export function StatsHeader({
  botCount,
  liveTradeCount,
  uptime,
  totalRealizedPnl,
  totalUnrealizedPnl,
  totalPnl,
  techMode,
  lastUpdated,
  loading,
  onRefresh,
}: Props) {
  return (
    <header
      className="sticky top-0 z-40 border-b-4 border-[#FFDD2D] shadow-md"
      style={{ background: '#0A0A0A' }}
    >
      <div className="mx-auto flex max-w-[1800px] flex-wrap items-center justify-between gap-3 px-4 py-3">
        <div className="flex items-center gap-3">
          <div
            className="flex size-11 items-center justify-center rounded-xl shadow-lg"
            style={{ background: '#FFDD2D' }}
          >
            <Bot className="size-6 text-neutral-900" />
          </div>
          <div>
            <h1 className="text-xl font-black tracking-tight text-white">
              AI Trader
              <span className="ml-2 rounded bg-white/10 px-1.5 py-0.5 align-middle text-[10px] font-semibold uppercase tracking-wide text-[#FFDD2D]">
                T-Bank Sandbox
              </span>
            </h1>
            <p className="text-xs font-medium text-neutral-400">
              {fmtUptime(uptime)} · {fmtInt(liveTradeCount)} сделок ·{' '}
              {botCount} ботов ·{' '}
              {techMode ? (
                <span className="text-amber-400">tech mode</span>
              ) : (
                <span className="text-emerald-400">live</span>
              )}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <StatPill
            icon={<TrendingUp className="size-3.5" />}
            label="Реализованный"
            value={fmtRub(totalRealizedPnl, { sign: true })}
            valueClass={pnlClass(totalRealizedPnl)}
          />
          <StatPill
            icon={<Zap className="size-3.5" />}
            label="Нереализованный"
            value={fmtRub(totalUnrealizedPnl, { sign: true })}
            valueClass={pnlClass(totalUnrealizedPnl)}
          />
          <StatPill
            icon={<Clock className="size-3.5" />}
            label="Итого P&L"
            value={fmtRub(totalPnl, { sign: true })}
            valueClass={pnlClass(totalPnl)}
            highlight
          />
          <button
            type="button"
            onClick={onRefresh}
            disabled={loading}
            className="flex items-center gap-1.5 rounded-md border border-white/20 bg-white/5 px-2.5 py-1.5 text-xs font-medium text-white transition-colors hover:bg-white/10 disabled:opacity-50"
            title="Обновить"
          >
            <RefreshCw className={`size-3.5 ${loading ? 'animate-spin' : ''}`} />
            <span className="hidden sm:inline">
              {lastUpdated
                ? lastUpdated.toLocaleTimeString('ru-RU')
                : '—'}
            </span>
          </button>
        </div>
      </div>
    </header>
  );
}

function StatPill({
  icon,
  label,
  value,
  valueClass,
  highlight,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  valueClass?: string;
  highlight?: boolean;
}) {
  return (
    <div
      className={`hidden items-center gap-2 rounded-md border px-3 py-1.5 md:flex ${
        highlight
          ? 'border-[#FFDD2D]/40 bg-[#FFDD2D]/10'
          : 'border-white/10 bg-white/5'
      }`}
    >
      <span className="text-neutral-400">{icon}</span>
      <div className="leading-tight">
        <div className="text-[9px] uppercase tracking-wide text-neutral-400">
          {label}
        </div>
        <div
          className={`font-mono text-sm font-bold tabular-nums text-white ${valueClass ?? ''}`}
        >
          {value}
        </div>
      </div>
    </div>
  );
}
