#!/usr/bin/env python3
"""Task 4b — Re-run sqlite queries with correct column names.
Schema: Trade(id, ts, botName, side, ticker, qty, price, pnl, balanceAfter, interval)
"""
import paramiko, os

TRADER_HOST = "2.26.122.152"
TRADER_USER = "root"
TRADER_PASS = "uiF=!6FrBb&9U1Xh"

CACHE = "/home/z/my-project/agent4_cache"
os.makedirs(CACHE, exist_ok=True)

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(TRADER_HOST, username=TRADER_USER, password=TRADER_PASS,
               timeout=30, look_for_keys=False, allow_agent=False)


def run(cmd, timeout=120):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    rc = stdout.channel.recv_exit_status()
    return rc, out, err


def save(name, content):
    path = os.path.join(CACHE, name)
    with open(path, "w") as f:
        f.write(content)
    print(f"  saved {name} ({len(content)} bytes)")


def step(title, sql):
    cmd = f'sqlite3 -header -column /opt/ai-trader/db/trader.db "{sql}"'
    rc, out, err = run(cmd)
    save(title, f"$ {cmd}\n--- rc={rc} ---\nSTDOUT:\n{out}\nSTDERR:\n{err}\n")


# Per-bot summary
step("30_per_bot_summary", """
SELECT botName AS bot,
       COUNT(*) AS n,
       ROUND(SUM(pnl),2) AS total_pnl,
       ROUND(AVG(pnl),3) AS avg_pnl,
       MIN(pnl) AS worst,
       MAX(pnl) AS best,
       ROUND(SUM(CASE WHEN pnl>0 THEN 1.0 ELSE 0 END)/COUNT(*),3) AS win_rate
FROM Trade
GROUP BY botName
ORDER BY total_pnl ASC;
""")

# Side breakdown (BUY/SELL/SHORT/CLOSE_SHORT counts per bot)
step("31_per_bot_side", """
SELECT botName AS bot, side,
       COUNT(*) AS n,
       ROUND(SUM(pnl),2) AS total_pnl,
       ROUND(AVG(pnl),3) AS avg_pnl
FROM Trade
GROUP BY botName, side
ORDER BY botName, side;
""")

# Ticker breakdown
step("32_ticker_breakdown", """
SELECT ticker,
       COUNT(*) AS n,
       ROUND(SUM(pnl),2) AS total_pnl,
       ROUND(AVG(pnl),3) AS avg_pnl,
       ROUND(SUM(CASE WHEN pnl>0 THEN 1.0 ELSE 0 END)/COUNT(*),3) AS win_rate
FROM Trade
GROUP BY ticker
ORDER BY total_pnl ASC;
""")

# Overall aggregates (only SELL and CLOSE_SHORT rows are the close trades; pnl is recorded there)
step("33_overall", """
SELECT
  (SELECT COUNT(*) FROM Trade) AS total_rows,
  (SELECT COUNT(*) FROM Trade WHERE side IN ('SELL','CLOSE_SHORT')) AS n_closes,
  (SELECT ROUND(SUM(pnl),2) FROM Trade WHERE side IN ('SELL','CLOSE_SHORT')) AS total_realized_pnl,
  (SELECT ROUND(AVG(pnl),3) FROM Trade WHERE side IN ('SELL','CLOSE_SHORT')) AS avg_realized_pnl,
  (SELECT COUNT(*) FROM Trade WHERE side IN ('SELL','CLOSE_SHORT') AND pnl<0) AS n_losing_closes,
  (SELECT COUNT(*) FROM Trade WHERE side IN ('SELL','CLOSE_SHORT') AND pnl>0) AS n_winning_closes,
  (SELECT ROUND(SUM(pnl),2) FROM Trade WHERE side IN ('SELL','CLOSE_SHORT') AND pnl<0) AS sum_loss,
  (SELECT ROUND(SUM(pnl),2) FROM Trade WHERE side IN ('SELL','CLOSE_SHORT') AND pnl>0) AS sum_win;
""")

# Losing trades sample (only close-side trades)
step("34_losing_trades", """
SELECT botName AS bot, ticker, side, qty, ROUND(price,3) AS price,
       ROUND(pnl,3) AS pnl, ROUND(balanceAfter,2) AS bal_after,
       datetime(ts/1000,'unixepoch','localtime') AS closed
FROM Trade
WHERE side IN ('SELL','CLOSE_SHORT') AND pnl < 0
ORDER BY pnl ASC
LIMIT 50;
""")

# Winning trades sample
step("35_winning_trades", """
SELECT botName AS bot, ticker, side, qty, ROUND(price,3) AS price,
       ROUND(pnl,3) AS pnl, ROUND(balanceAfter,2) AS bal_after,
       datetime(ts/1000,'unixepoch','localtime') AS closed
FROM Trade
WHERE side IN ('SELL','CLOSE_SHORT') AND pnl > 0
ORDER BY pnl DESC
LIMIT 30;
""")

# Per-bot: avg position size (qty*price from BUY/SHORT), avg hold time
step("36_per_bot_position_size", """
SELECT botName AS bot,
       COUNT(*) AS n_trades,
       ROUND(AVG(qty*price),2) AS avg_trade_value,
       ROUND(AVG(qty),2) AS avg_qty
FROM Trade
GROUP BY botName;
""")

# Per-bot: realized PnL over time (last 30 closes by bot)
step("37_recent_closes_by_bot", """
SELECT botName AS bot, ticker, side, qty, ROUND(price,3) AS price,
       ROUND(pnl,3) AS pnl, datetime(ts/1000,'unixepoch','localtime') AS closed
FROM Trade
WHERE side IN ('SELL','CLOSE_SHORT')
ORDER BY ts DESC
LIMIT 50;
""")

# BotState current snapshot
step("38_botstate_snapshot", """
SELECT botName AS bot,
       ROUND(realBalance,2) AS bal,
       ROUND(realTotalValue,2) AS total,
       ROUND(realizedPnl,2) AS realized,
       liveBuys AS buys, liveSells AS sells, liveTrades AS trades,
       datetime(updatedAt/1000,'unixepoch','localtime') AS updated
FROM BotState
ORDER BY realizedPnl ASC;
""")

# PnL distribution by magnitude buckets (only losing trades)
step("39_loss_magnitude_distribution", """
SELECT
  CASE
    WHEN pnl BETWEEN -1 AND 0 THEN 'L: -1 to 0'
    WHEN pnl BETWEEN -2 AND -1 THEN 'L: -2 to -1'
    WHEN pnl BETWEEN -5 AND -2 THEN 'L: -5 to -2'
    WHEN pnl BETWEEN -10 AND -5 THEN 'L: -10 to -5'
    WHEN pnl < -10 THEN 'L: < -10'
  END AS bucket,
  COUNT(*) AS n,
  ROUND(SUM(pnl),2) AS total_pnl
FROM Trade
WHERE side IN ('SELL','CLOSE_SHORT') AND pnl<0
GROUP BY bucket
ORDER BY bucket;
""")

# How many losing trades have |pnl| < ~1 RUB (commission death pattern)
step("40_commission_death_check", """
SELECT
  botName AS bot,
  COUNT(*) AS n_closes,
  SUM(CASE WHEN pnl < 0 AND pnl > -2 THEN 1 ELSE 0 END) AS small_loss_under_2rub,
  SUM(CASE WHEN pnl > 0 AND pnl < 2 THEN 1 ELSE 0 END) AS small_win_under_2rub,
  ROUND(SUM(CASE WHEN pnl < 0 AND pnl > -2 THEN pnl ELSE 0 END),2) AS small_loss_sum,
  ROUND(SUM(CASE WHEN pnl > 0 AND pnl < 2 THEN pnl ELSE 0 END),2) AS small_win_sum,
  ROUND(SUM(pnl),2) AS total_pnl
FROM Trade
WHERE side IN ('SELL','CLOSE_SHORT')
GROUP BY botName
ORDER BY total_pnl ASC;
""")

# Time-of-day distribution (do they trade at bad times?)
step("41_hour_breakdown", """
SELECT
  CAST(strftime('%H', ts/1000,'unixepoch','localtime') AS INT) AS hour_msk,
  COUNT(*) AS n,
  ROUND(SUM(pnl),2) AS total_pnl,
  ROUND(AVG(pnl),3) AS avg_pnl
FROM Trade
WHERE side IN ('SELL','CLOSE_SHORT')
GROUP BY hour_msk
ORDER BY hour_msk;
""")

# Date range
step("42_date_range", """
SELECT
  datetime(MIN(ts)/1000,'unixepoch','localtime') AS first_trade,
  datetime(MAX(ts)/1000,'unixepoch','localtime') AS last_trade,
  COUNT(*) AS total
FROM Trade;
""")

client.close()
print("\nDone. All queries cached in", CACHE)
