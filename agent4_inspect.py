#!/usr/bin/env python3
"""Task 4 — Inference archaeologist.
Read-only study of trader server ML inference + why bots lose money.
Saves raw outputs to /home/z/my-project/agent4_cache/.
"""
import paramiko, os, json, time

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


def step(title, cmd, fname=None, timeout=120):
    print(f"\n=== {title} ===")
    rc, out, err = run(cmd, timeout=timeout)
    if fname is None:
        fname = title.replace(" ", "_").replace("/", "_").replace(":", "_") + ".txt"
    save(fname, f"$ {cmd}\n--- rc={rc} ---\nSTDOUT:\n{out}\nSTDERR:\n{err}\n")
    return out, err, rc


# ---------------------------------------------------------
# 1) Inventory
# ---------------------------------------------------------
step("01_strategies_dir", "ls -la /opt/ai-trader/src/strategies/")
step("02_core_dir",      "ls -la /opt/ai-trader/src/core/")
step("03_bots_dir",      "ls -la /opt/ai-trader/config/bots/")
step("04_db_dir",        "ls -la /opt/ai-trader/db/")
step("05_pkg_json",      "cat /opt/ai-trader/package.json 2>/dev/null | head -80")

# ---------------------------------------------------------
# 2) Strategy files
# ---------------------------------------------------------
STRATEGIES = [
    "ml_predict.ts",
    "ml_predict_v2.ts",
    "meta_selector.ts",
    "meta_selector_v4.ts",
    "regime_detector.ts",
    "xgboost_binary_ts.ts",
    "base.ts",
]
for s in STRATEGIES:
    path = f"/opt/ai-trader/src/strategies/{s}"
    rc, out, err = run(f"test -f {path} && cat {path} || echo MISSING")
    fname = f"strat_{s}"
    save(fname, f"$ cat {path}\n--- rc={rc} ---\n{out}\n{err}\n")

# ---------------------------------------------------------
# 3) Core engine files
# ---------------------------------------------------------
CORE = ["risk-manager.ts", "bot-instance.ts", "engine.ts"]
for c in CORE:
    path = f"/opt/ai-trader/src/core/{c}"
    rc, out, err = run(f"test -f {path} && cat {path} || echo MISSING")
    save(f"core_{c}", f"$ cat {path}\n--- rc={rc} ---\n{out}\n{err}\n")

# ---------------------------------------------------------
# 4) All bot configs
# ---------------------------------------------------------
rc, lst, _ = run("ls /opt/ai-trader/config/bots/ 2>/dev/null")
bot_files = [f.strip() for f in lst.splitlines() if f.strip()]
print(f"\nFound {len(bot_files)} bot config files: {bot_files}")
for bf in bot_files:
    rc, out, err = run(f"cat /opt/ai-trader/config/bots/{bf}")
    save(f"bot_{bf}", f"$ cat /opt/ai-trader/config/bots/{bf}\n--- rc={rc} ---\n{out}\n{err}\n")

# ---------------------------------------------------------
# 5) sqlite queries — Trade table
# ---------------------------------------------------------
# Schema first
step("06_db_tables",
     "sqlite3 /opt/ai-trader/db/trader.db '.tables'")
step("07_db_schema_trades",
     "sqlite3 /opt/ai-trader/db/trader.db '.schema Trade' 2>&1; echo '---bots---'; sqlite3 /opt/ai-trader/db/trader.db '.schema Bot' 2>&1")

# Per-bot summary
step("08_per_bot_summary",
     """sqlite3 -header -column /opt/ai-trader/db/trader.db "
        SELECT botId,
               COUNT(*) AS n_trades,
               ROUND(SUM(pnl),2) AS total_pnl,
               ROUND(AVG(pnl),2) AS avg_pnl,
               ROUND(SUM(CASE WHEN pnl>0 THEN 1.0 ELSE 0 END)/COUNT(*),3) AS win_rate,
               ROUND(SUM(grossPnl),2) AS sum_gross,
               ROUND(SUM(commission),2) AS sum_comm,
               ROUND(AVG(commission),2) AS avg_comm,
               ROUND(AVG(grossPnl),2) AS avg_gross
        FROM Trade
        GROUP BY botId
        ORDER BY total_pnl ASC;
     "
     """,
     timeout=60)

# All losing trades sample
step("09_losing_trades_sample",
     """sqlite3 -header -column /opt/ai-trader/db/trader.db "
        SELECT botId, ticker, side, ROUND(entryPrice,3) AS entry,
               ROUND(exitPrice,3) AS exit, ROUND(grossPnl,3) AS gross,
               ROUND(commission,3) AS comm, ROUND(pnl,3) AS pnl,
               datetime(openedAt/1000,'unixepoch','localtime') AS opened,
               datetime(closedAt/1000,'unixepoch','localtime') AS closed
        FROM Trade
        WHERE pnl < 0
        ORDER BY pnl ASC
        LIMIT 50;
     "
     """,
     timeout=60)

# Winning trades sample (for comparison)
step("10_winning_trades_sample",
     """sqlite3 -header -column /opt/ai-trader/db/trader.db "
        SELECT botId, ticker, side, ROUND(entryPrice,3) AS entry,
               ROUND(exitPrice,3) AS exit, ROUND(grossPnl,3) AS gross,
               ROUND(commission,3) AS comm, ROUND(pnl,3) AS pnl,
               datetime(openedAt/1000,'unixepoch','localtime') AS opened,
               datetime(closedAt/1000,'unixepoch','localtime') AS closed
        FROM Trade
        WHERE pnl > 0
        ORDER BY pnl DESC
        LIMIT 30;
     "
     """,
     timeout=60)

# Aggregate stats
step("11_db_aggregate",
     """sqlite3 -header -column /opt/ai-trader/db/trader.db "
        SELECT
            COUNT(*) AS n_trades,
            ROUND(SUM(pnl),2) AS total_pnl,
            ROUND(SUM(grossPnl),2) AS total_gross,
            ROUND(SUM(commission),2) AS total_comm,
            ROUND(AVG(commission),3) AS avg_comm_per_trade,
            ROUND(AVG(ABS(grossPnl)),3) AS avg_abs_gross,
            ROUND(SUM(CASE WHEN pnl>0 THEN 1.0 ELSE 0 END)/COUNT(*),3) AS win_rate,
            ROUND(SUM(CASE WHEN grossPnl>0 AND pnl<0 THEN 1 ELSE 0 END)*1.0/COUNT(*),3) AS frac_comm_killed_profit
        FROM Trade;
     "
     """,
     timeout=60)

# Side distribution (long/short skew?)
step("12_side_breakdown",
     """sqlite3 -header -column /opt/ai-trader/db/trader.db "
        SELECT side, COUNT(*) AS n,
               ROUND(SUM(pnl),2) AS total_pnl,
               ROUND(AVG(pnl),3) AS avg_pnl,
               ROUND(SUM(CASE WHEN pnl>0 THEN 1.0 ELSE 0 END)/COUNT(*),3) AS win_rate
        FROM Trade
        GROUP BY side;
     "
     """,
     timeout=60)

# Ticker breakdown
step("13_ticker_breakdown",
     """sqlite3 -header -column /opt/ai-trader/db/trader.db "
        SELECT ticker, COUNT(*) AS n,
               ROUND(SUM(pnl),2) AS total_pnl,
               ROUND(AVG(pnl),3) AS avg_pnl,
               ROUND(SUM(commission),2) AS total_comm
        FROM Trade
        GROUP BY ticker
        ORDER BY total_pnl ASC;
     "
     """,
     timeout=60)

# Average position size, avg hold time
step("14_position_size_hold",
     """sqlite3 -header -column /opt/ai-trader/db/trader.db "
        SELECT botId,
               ROUND(AVG(quantity),3) AS avg_qty,
               ROUND(AVG(entryPrice),2) AS avg_entry_price,
               ROUND(AVG((closedAt-openedAt)/60000.0),1) AS avg_hold_min,
               COUNT(*) AS n
        FROM Trade
        GROUP BY botId;
     "
     """,
     timeout=60)

# Was-there-prediction metadata? Check column names.
step("15_db_full_schema",
     "sqlite3 /opt/ai-trader/db/trader.db '.schema' | head -200")

# ---------------------------------------------------------
# 6) Logs
# ---------------------------------------------------------
step("16_worker_log_tail",
     "tail -200 /var/log/ai-trader-worker.log 2>/dev/null || echo 'no worker log'")
step("17_worker_log_skip_filter",
     "grep -iE 'skip|filter|commFilter|commission|insufficient|predicted|regime|long|short' /var/log/ai-trader-worker.log 2>/dev/null | tail -100 || echo 'no log'",
     timeout=60)
step("18_daemon_log_tail",
     "tail -200 /var/log/tbank-trade-daemon.log 2>/dev/null || echo 'no daemon log'")
step("19_daemon_errors",
     "grep -iE 'error|exception|fail|undefined|null|NaN' /var/log/tbank-trade-daemon.log 2>/dev/null | tail -100 || echo 'no log'",
     timeout=60)

# Worker status
step("20_systemd_worker",
     "systemctl status ai-trader-worker --no-pager 2>&1 | head -30")
step("21_systemd_daemon",
     "systemctl status tbank-trade-daemon --no-pager 2>&1 | head -30")

# Process list
step("22_ps",
     "ps aux | grep -iE 'ai-trader|tbank|node|bun' | grep -v grep")

# Recent files in db dir
step("23_db_files",
     "ls -la /opt/ai-trader/db/ 2>&1; echo '---'; du -h /opt/ai-trader/db/*.db 2>&1")

client.close()
print("\nDone. All outputs cached in", CACHE)
