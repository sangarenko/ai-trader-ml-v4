'use client';

import type { Bot } from '@/lib/types';
import {
  botPnlPct,
  fmtInt,
  fmtPct,
  fmtRub,
  fmtTime,
  pnlClass,
} from '@/lib/dashboard';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';

interface Props {
  bot: Bot | null;
  onOpenChange: (open: boolean) => void;
}

export function BotDetailDialog({ bot, onOpenChange }: Props) {
  return (
    <Dialog open={!!bot} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[88vh] max-w-4xl gap-0 overflow-hidden p-0 sm:max-w-4xl">
        {bot && <DetailBody bot={bot} />}
      </DialogContent>
    </Dialog>
  );
}

function DetailBody({ bot }: { bot: Bot }) {
  const { config, stats } = bot;
  const pnl = stats?.totalPnl ?? 0;
  const pnlPct = botPnlPct(bot);

  return (
    <>
      <DialogHeader className="border-b border-neutral-200 p-5 pb-4">
        <div className="flex items-start gap-3">
          <span
            className="mt-1 size-3 rounded-full"
            style={{ background: config.color }}
          />
          <div className="min-w-0 flex-1">
            <DialogTitle className="text-xl font-bold text-neutral-900">
              {config.name}
            </DialogTitle>
            <DialogDescription className="mt-0.5 text-sm text-neutral-500">
              {config.description}
            </DialogDescription>
          </div>
          <div className="flex shrink-0 gap-2">
            <Stat label="Баланс" value={fmtRub(stats?.realTotalValue ?? config.virtualBalance ?? 0)} />
            <Stat
              label="P&L"
              value={fmtRub(pnl, { sign: true })}
              valueClass={pnlClass(pnl)}
            />
            <Stat
              label="Доходность"
              value={fmtPct(pnlPct)}
              valueClass={pnlClass(pnl)}
            />
          </div>
        </div>
      </DialogHeader>

      <ScrollArea className="max-h-[70vh]">
        <div className="space-y-5 p-5">
          {/* config */}
          <section>
            <SectionTitle>Конфигурация</SectionTitle>
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4">
              <KV label="Стратегия" value={config.strategy} mono />
              <KV label="Интервал" value={config.candleInterval} mono />
              <KV label="Position size" value={`${(config.positionSize * 100).toFixed(0)}%`} />
              <KV label="Max cost" value={fmtRub(config.maxPositionCost)} />
              <KV label="Ticker mode" value={config.tickerMode} mono />
              <KV
                label="Rotate"
                value={`${config.rotateIntervalSec}s`}
                mono
              />
              <KV label="Trades" value={fmtInt(stats?.liveTrades ?? 0)} />
              <KV
                label="Commission"
                value={fmtRub(stats?.commission ?? 0)}
                valueClass="text-red-600"
              />
            </div>
            {config.tickers.length > 0 && (
              <div className="mt-3">
                <div className="mb-1 text-[11px] uppercase tracking-wide text-neutral-400">
                  Тикеры ({config.tickers.length})
                </div>
                <div className="flex flex-wrap gap-1">
                  {config.tickers.map((t) => (
                    <Badge
                      key={t}
                      variant="outline"
                      className="font-mono text-[10px] text-neutral-600"
                    >
                      {t}
                    </Badge>
                  ))}
                </div>
              </div>
            )}
            {config.filters && (
              <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-5">
                <KV label="Hold ticks" value={String(config.filters.holdTicks)} mono />
                <KV label="Cooldown" value={String(config.filters.cooldownTicks)} mono />
                <KV label="Comm mult" value={String(config.filters.commFilterMult)} mono />
                <KV label="Trades/hr" value={String(config.filters.maxTradesPerHour)} mono />
                <KV label="Max hold hr" value={String(config.filters.maxHoldHours)} mono />
              </div>
            )}
          </section>

          <Separator />

          {/* open positions */}
          <section>
            <SectionTitle>
              Открытые позиции
              <Badge variant="secondary" className="ml-2 bg-neutral-100">
                {stats?.openPositions?.length ?? 0}
              </Badge>
            </SectionTitle>
            {stats?.openPositions && stats.openPositions.length > 0 ? (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Тикер</TableHead>
                    <TableHead>Сторона</TableHead>
                    <TableHead className="text-right">Кол-во</TableHead>
                    <TableHead className="text-right">Вход</TableHead>
                    <TableHead className="text-right">Тек.</TableHead>
                    <TableHead className="text-right">P&L</TableHead>
                    <TableHead className="text-right">Время</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {stats.openPositions.map((p, i) => (
                    <TableRow key={`${p.ticker}-${i}`}>
                      <TableCell className="font-mono font-semibold">
                        {p.ticker}
                      </TableCell>
                      <TableCell>
                        <Badge
                          variant="outline"
                          className={
                            p.side === 'short'
                              ? 'border-red-200 bg-red-50 text-red-700'
                              : 'border-emerald-200 bg-emerald-50 text-emerald-700'
                          }
                        >
                          {p.side.toUpperCase()}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-right font-mono">
                        {p.qty}
                      </TableCell>
                      <TableCell className="text-right font-mono">
                        {p.entryPrice}
                      </TableCell>
                      <TableCell className="text-right font-mono">
                        {p.currentPrice}
                      </TableCell>
                      <TableCell
                        className={`text-right font-mono font-semibold ${pnlClass(p.unrealizedPnl)}`}
                      >
                        {fmtRub(p.unrealizedPnl, { sign: true })}
                      </TableCell>
                      <TableCell className="text-right text-xs text-neutral-400">
                        {fmtTime(p.ts)}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            ) : (
              <EmptyState>Нет открытых позиций</EmptyState>
            )}
          </section>

          <Separator />

          {/* recent trades */}
          <section>
            <SectionTitle>
              Последние сделки
              <Badge variant="secondary" className="ml-2 bg-neutral-100">
                {stats?.history?.length ?? 0}
              </Badge>
            </SectionTitle>
            {stats?.history && stats.history.length > 0 ? (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Время</TableHead>
                    <TableHead>Сторона</TableHead>
                    <TableHead>Тикер</TableHead>
                    <TableHead className="text-right">Кол-во</TableHead>
                    <TableHead className="text-right">Цена</TableHead>
                    <TableHead className="text-right">P&L</TableHead>
                    <TableHead className="text-right">Баланс</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {stats.history.map((t, i) => (
                    <TableRow key={i}>
                      <TableCell className="text-xs text-neutral-400">
                        {fmtTime(t.ts)}
                      </TableCell>
                      <TableCell>
                        <span
                          className={`font-mono text-[11px] font-semibold ${
                            t.side.includes('SHORT')
                              ? 'text-red-600'
                              : t.side.includes('CLOSE')
                                ? 'text-neutral-500'
                                : 'text-emerald-600'
                          }`}
                        >
                          {t.side}
                        </span>
                      </TableCell>
                      <TableCell className="font-mono font-semibold">
                        {t.ticker}
                      </TableCell>
                      <TableCell className="text-right font-mono">
                        {t.qty}
                      </TableCell>
                      <TableCell className="text-right font-mono">
                        {t.price}
                      </TableCell>
                      <TableCell
                        className={`text-right font-mono font-semibold ${pnlClass(t.pnl)}`}
                      >
                        {fmtRub(t.pnl, { sign: true })}
                      </TableCell>
                      <TableCell className="text-right font-mono text-xs text-neutral-500">
                        {fmtRub(t.balanceAfter)}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            ) : (
              <EmptyState>Нет истории сделок</EmptyState>
            )}
          </section>
        </div>
      </ScrollArea>
    </>
  );
}

function SectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <h4 className="mb-2 flex items-center text-xs font-bold uppercase tracking-wide text-neutral-600">
      {children}
    </h4>
  );
}

function Stat({
  label,
  value,
  valueClass,
}: {
  label: string;
  value: string;
  valueClass?: string;
}) {
  return (
    <div className="rounded-md border border-neutral-200 bg-neutral-50 px-2.5 py-1 text-right">
      <div className="text-[10px] uppercase tracking-wide text-neutral-400">
        {label}
      </div>
      <div
        className={`font-mono text-sm font-bold tabular-nums text-neutral-900 ${valueClass ?? ''}`}
      >
        {value}
      </div>
    </div>
  );
}

function KV({
  label,
  value,
  mono,
  valueClass,
}: {
  label: string;
  value: string;
  mono?: boolean;
  valueClass?: string;
}) {
  return (
    <div className="rounded-md border border-neutral-100 bg-white px-2.5 py-1.5">
      <div className="text-[10px] uppercase tracking-wide text-neutral-400">
        {label}
      </div>
      <div
        className={`mt-0.5 truncate text-sm font-semibold text-neutral-900 ${mono ? 'font-mono' : ''} ${valueClass ?? ''}`}
      >
        {value}
      </div>
    </div>
  );
}

function EmptyState({ children }: { children: React.ReactNode }) {
  return (
    <div className="rounded-md border border-dashed border-neutral-200 py-6 text-center text-sm text-neutral-400">
      {children}
    </div>
  );
}
