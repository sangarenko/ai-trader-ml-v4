// Types for the AI Trader — T-Bank Sandbox dashboard

export type TickerMode = 'rotate' | 'fixed';

export interface BotFilters {
  holdTicks: number;
  cooldownTicks: number;
  commFilterMult: number;
  maxTradesPerHour: number;
  maxHoldHours: number;
}

export interface BotConfig {
  name: string;
  color: string;
  strategy: string;
  description: string;
  tickers: string[];
  tickerMode: TickerMode;
  rotateIntervalSec: number;
  positionSize: number;
  filters: BotFilters;
  candleInterval: string;
  maxPositionCost: number;
  startTickerIdx: number;
  // account linkage (present when bot runs on a shared broker account)
  accountId?: string;
  sharedAccount?: boolean;
  virtualBalance?: number;
}

export interface Position {
  ticker: string;
  qty: number;
  entryPrice: number;
  ts: number;
  side: 'long' | 'short';
  currentPrice: number;
  unrealizedPnl: number;
}

export interface Trade {
  ts: number;
  botName: string;
  side: string;
  ticker: string;
  qty: number;
  price: number;
  pnl: number;
  balanceAfter: number;
}

export interface BotLiveStats {
  name: string;
  agentType: string;
  color: string;
  liveBuys: number;
  liveSells: number;
  liveTrades: number;
  realizedPnl: number;
  grossRealizedPnl: number;
  commission: number;
  unrealizedPnl: number;
  realBalance: number;
  realSharesValue: number;
  realTotalValue: number;
  totalPnl: number;
  openPositions: Position[];
  history: Trade[];
}

export interface Agent {
  name: string;
  color: string;
  agentType: string;
  description: string;
  balance: number;
  bestBalance: number;
  totalTrades: number;
  cumulativePnl: number;
  trades: Trade[];
}

export interface LogEntry {
  ts?: number;
  level?: string;
  msg?: string;
  message?: string;
  [k: string]: unknown;
}

export interface TraderState {
  bots: BotConfig[];
  botLiveStats: Record<string, BotLiveStats>;
  logs: LogEntry[];
  liveTradeCount: number;
  uptime: number;
  totalRealizedPnl: number;
  totalUnrealizedPnl: number;
  totalPnl: number;
  techMode: boolean;
  agents: Agent[];
}

// Derived view model -------------------------------------------------------

export interface Bot {
  config: BotConfig;
  stats: BotLiveStats | null;
}

export interface Account {
  id: string;
  index: number; // 1-based display order
  label: string; // "Аккаунт 1" | bot name
  type: 'shared' | 'standalone';
  accountId?: string;
  botCount: number;
  bots: Bot[];
  totalBalance: number;
  totalPnl: number;
  totalTrades: number;
  realizedPnl: number;
  unrealizedPnl: number;
}
