---
Task ID: 1
Agent: main (Z.ai Code)
Task: Redesign AI Trader — T-Bank Sandbox dashboard: group bots by account (master-detail), 9 standalone accounts + 1 shared account with 50 bots.

Work Log:
- Fetched existing dashboard at http://2.26.122.152:3002/ (Next.js, "AI Trader — T-Bank Sandbox", RL trading bots on T-Bank demo via T-Invest API).
- Discovered /api/state endpoint (281KB JSON, 59 bots).
- Parsed structure: bots have `accountId` + `sharedAccount` fields. 50 bots share accountId `ce2e4b55-…`, 9 are standalone (accountId=null).
- Saved trimmed snapshot to data/state-snapshot.json (115KB) for sandbox preview.
- Built types (src/lib/types.ts) + transform (src/lib/dashboard.ts): deriveAccounts() groups standalone bots as accounts #1-9, shared group as account #10.
- Created /api/state route (returns snapshot; replaceable with real state on deploy).
- Built components: StatsHeader (T-Bank yellow/black theme, global P&L), AccountSidebar (10 accounts, search, live stats), BotCard (color stripe, balance/PnL/trades/positions), BotDetailDialog (config, open positions table, recent trades table).
- Rewrote src/app/page.tsx: h-dvh layout, auto-refresh every 5s, account auto-select, sort by PnL/trades/balance/name, mobile Sheet sidebar, sticky footer.
- Fixed layout: h-dvh + min-h-0 + internal scroll → footer sticks to viewport bottom, bot list scrolls internally.
- Verified via Agent Browser: 10 accounts render, account switch works (#1→1 bot, #10→50 bots), bot detail dialog opens with positions/trades, no console errors, footer sticky (footerTop=863, bodyH=900=viewportH), lint clean.

Stage Summary:
- Dashboard fully functional in sandbox preview.
- Data source: relative /api/state (works in sandbox via snapshot AND on trader server where real /api/state exists — no code change needed on deploy).
- 59 bots correctly separated: 9 standalone accounts (1 bot each) + 1 shared account (50 bots).
- Master-detail UI: click account left → bots appear right.
- To deploy on user's server: copy src/app/page.tsx, src/components/dashboard/*, src/lib/{types,dashboard}.ts. The /api/state route can be dropped (their server already has it).
