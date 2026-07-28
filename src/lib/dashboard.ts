import type {
  Account,
  Bot,
  BotConfig,
  BotLiveStats,
  TraderState,
} from './types';

const T_BANK_YELLOW = '#FFDD2D';

/**
 * Group bots into "accounts".
 *
 * - Standalone bots (no `accountId`) each form their own account (1 bot = 1 account).
 * - Bots sharing the same `accountId` collapse into one shared account.
 *
 * Standalone accounts are numbered first (1..N), the shared account is the last
 * one — mirroring the "9 standalone + 10th shared with 50 bots" layout the user
 * described.
 */
export function deriveAccounts(state: TraderState): Account[] {
  const liveStats = state.botLiveStats ?? {};
  const merge = (cfg: BotConfig): Bot => ({
    config: cfg,
    stats: liveStats[cfg.name] ?? null,
  });

  const standalone: BotConfig[] = [];
  const byAccount = new Map<string, BotConfig[]>();

  for (const cfg of state.bots ?? []) {
    if (cfg.accountId) {
      const arr = byAccount.get(cfg.accountId) ?? [];
      arr.push(cfg);
      byAccount.set(cfg.accountId, arr);
    } else {
      standalone.push(cfg);
    }
  }

  const accounts: Account[] = [];
  let idx = 1;

  // standalone first (each bot = its own account)
  for (const cfg of standalone) {
    const bots = [merge(cfg)];
    accounts.push(buildAccount(String(idx), idx, 'standalone', undefined, bots));
    idx++;
  }

  // shared accounts after, ordered by bot count desc (big group = "10th account")
  const sharedGroups = [...byAccount.values()].sort(
    (a, b) => b.length - a.length,
  );
  for (const group of sharedGroups) {
    const accountId = group[0].accountId!;
    const bots = group.map(merge);
    accounts.push(
      buildAccount(String(idx), idx, 'shared', accountId, bots),
    );
    idx++;
  }

  return accounts;
}

function buildAccount(
  id: string,
  index: number,
  type: Account['type'],
  accountId: string | undefined,
  bots: Bot[],
): Account {
  let totalBalance = 0;
  let totalPnl = 0;
  let totalTrades = 0;
  let realizedPnl = 0;
  let unrealizedPnl = 0;

  for (const b of bots) {
    const s = b.stats;
    if (s) {
      totalBalance += s.realTotalValue ?? s.realBalance ?? 0;
      totalPnl += s.totalPnl ?? 0;
      totalTrades += s.liveTrades ?? 0;
      realizedPnl += s.realizedPnl ?? 0;
      unrealizedPnl += s.unrealizedPnl ?? 0;
    } else if (b.config.virtualBalance) {
      totalBalance += b.config.virtualBalance;
    }
  }

  const label =
    type === 'shared'
      ? `Аккаунт ${index} · shared`
      : bots[0]?.config.name ?? `Аккаунт ${index}`;

  return {
    id,
    index,
    label,
    type,
    accountId,
    botCount: bots.length,
    bots,
    totalBalance,
    totalPnl,
    totalTrades,
    realizedPnl,
    unrealizedPnl,
  };
}

export function fmtRub(v: number, opts: { sign?: boolean } = {}): string {
  const sign = opts.sign && v > 0 ? '+' : '';
  const s = v.toLocaleString('ru-RU', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
  return `${sign}${s} ₽`;
}

export function fmtPct(v: number): string {
  const sign = v > 0 ? '+' : '';
  return `${sign}${v.toFixed(2)}%`;
}

export function fmtInt(v: number): string {
  return Math.round(v).toLocaleString('ru-RU');
}

export function fmtUptime(seconds: number): string {
  if (!seconds || seconds < 0) return '0м';
  const d = Math.floor(seconds / 86400);
  const h = Math.floor((seconds % 86400) / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  if (d > 0) return `${d}д ${h}ч`;
  if (h > 0) return `${h}ч ${m}м`;
  return `${m}м`;
}

export function fmtTime(ts: number): string {
  try {
    return new Date(ts).toLocaleTimeString('ru-RU', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  } catch {
    return '—';
  }
}

export function pnlClass(v: number): string {
  if (v > 0) return 'text-emerald-600';
  if (v < 0) return 'text-red-600';
  return 'text-neutral-500';
}

export function pnlBgClass(v: number): string {
  if (v > 0) return 'bg-emerald-50 text-emerald-700 border-emerald-200';
  if (v < 0) return 'bg-red-50 text-red-700 border-red-200';
  return 'bg-neutral-50 text-neutral-600 border-neutral-200';
}

export { T_BANK_YELLOW };

export function botPnlPct(bot: Bot): number {
  const s = bot.stats;
  if (!s) return 0;
  const base =
    bot.config.virtualBalance ?? s.realBalance - s.realizedPnl ?? 10000;
  if (!base) return 0;
  return (s.totalPnl / base) * 100;
}

export function sortBots(
  bots: Bot[],
  by: 'pnl' | 'trades' | 'balance' | 'name',
): Bot[] {
  const arr = [...bots];
  switch (by) {
    case 'pnl':
      arr.sort(
        (a, b) => (b.stats?.totalPnl ?? 0) - (a.stats?.totalPnl ?? 0),
      );
      break;
    case 'trades':
      arr.sort(
        (a, b) => (b.stats?.liveTrades ?? 0) - (a.stats?.liveTrades ?? 0),
      );
      break;
    case 'balance':
      arr.sort(
        (a, b) =>
          (b.stats?.realTotalValue ?? 0) - (a.stats?.realTotalValue ?? 0),
      );
      break;
    case 'name':
      arr.sort((a, b) => a.config.name.localeCompare(b.config.name));
      break;
  }
  return arr;
}

export function accountColor(account: Account): string {
  if (account.type === 'shared') return T_BANK_YELLOW;
  return account.bots[0]?.config.color ?? '#737373';
}
