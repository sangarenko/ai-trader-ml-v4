#!/usr/bin/env python3
"""Task 4c — Verify model file existence + collect regime/metadata details."""
import paramiko, os

TRADER_HOST = "2.26.122.152"
TRADER_USER = "root"
TRADER_PASS = "uiF=!6FrBb&9U1Xh"
CACHE = "/home/z/my-project/agent4_cache"

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


# V1/V2 model files on trader
rc, out, err = run("ls -la /root/ai-trader-evolution/ml/models/ 2>&1 | head -30")
save("50_evolution_models_dir.txt", f"$ ls /root/ai-trader-evolution/ml/models/\n--- rc={rc} ---\n{out}\n{err}\n")

# V4 model files on trader
rc, out, err = run("ls -la /opt/ai-trader/data/regime_*.json 2>&1 | head -20")
save("51_v4_data_models.txt", f"$ ls /opt/ai-trader/data/regime_*.json\n--- rc={rc} ---\n{out}\n{err}\n")

rc, out, err = run("ls -la /opt/ai-trader/src/strategies/regime_*.json 2>&1 | head -20")
save("52_v4_strategies_models.txt", f"$ ls /opt/ai-trader/src/strategies/regime_*.json\n--- rc={rc} ---\n{out}\n{err}\n")

# Meta-selector files
rc, out, err = run("ls -la /opt/ai-trader/src/strategies/meta_classifier.json /opt/ai-trader/src/strategies/meta_metadata.json 2>&1")
save("53_meta_files.txt", f"$ ls meta_*.json\n--- rc={rc} ---\n{out}\n{err}\n")

# meta_metadata.json content (small file)
rc, out, err = run("cat /opt/ai-trader/src/strategies/meta_metadata.json 2>&1")
save("54_meta_metadata.json", f"$ cat meta_metadata.json\n--- rc={rc} ---\n{out}\n{err}\n")

# Open positions (currently held by each bot — to see unrealized PnL exposure)
rc, out, err = run("""sqlite3 -header -column /opt/ai-trader/db/trader.db "
   SELECT botName AS bot, openPositionsJson
   FROM BotState
   WHERE openPositionsJson IS NOT NULL AND openPositionsJson != '[]';
" """)
save("55_open_positions.txt", f"$ sqlite3 query\n--- rc={rc} ---\nSTDOUT:\n{out}\nSTDERR:\n{err}\n")

# Look at ML-Trader's open positions (the one with bal=14687, total=-6766)
rc, out, err = run("""sqlite3 /opt/ai-trader/db/trader.db "
   SELECT botName, openPositionsJson FROM BotState WHERE botName='ML-Trader';
" """)
save("56_ml_trader_open_pos.txt", f"$ sqlite3 query\n--- rc={rc} ---\nSTDOUT:\n{out}\nSTDERR:\n{err}\n")

# Lot sizes (to confirm position math)
rc, out, err = run("cat /opt/ai-trader/src/core/lot-sizes.ts 2>&1 || cat /opt/ai-trader/src/core/lot-sizes.js 2>&1")
save("57_lot_sizes.txt", f"$ cat lot-sizes\n--- rc={rc} ---\n{out}\n{err}\n")

# Average realized PnL per bot per hour of day (when do they lose most?)
rc, out, err = run("""sqlite3 -header -column /opt/ai-trader/db/trader.db "
   SELECT botName AS bot,
          CAST(strftime('%H', ts/1000,'unixepoch','localtime') AS INT) AS hr,
          COUNT(*) AS n,
          ROUND(SUM(pnl),2) AS pnl
   FROM Trade
   WHERE side IN ('SELL','CLOSE_SHORT')
   GROUP BY bot, hr
   HAVING pnl < -10
   ORDER BY pnl ASC
   LIMIT 30;
" """)
save("58_bot_hour_big_losses.txt", f"$ sqlite3 query\n--- rc={rc} ---\nSTDOUT:\n{out}\nSTDERR:\n{err}\n")

# V4 regime probabilities histogram (which regimes fire most)
rc, out, err = run("""grep -E 'regime=' /var/log/ai-trader-worker.log 2>/dev/null | grep MetaSelectorV4 | awk -F'regime=' '{print $2}' | awk '{print $1}' | sort | uniq -c | sort -rn | head -20""")
save("59_v4_regime_freq.txt", f"$ grep regime MetaSelectorV4 freq\n--- rc={rc} ---\nSTDOUT:\n{out}\nSTDERR:\n{err}\n")

# How often each bot fires long vs short signal in logs
rc, out, err = run("""grep -E 'MetaSelectorV4.*P\(up\)=' /var/log/ai-trader-worker.log 2>/dev/null | awk -F'→ ' '{print $2}' | sort | uniq -c | sort -rn | head -10""")
save("60_v4_decision_dist.txt", f"$ v4 decision distribution\n--- rc={rc} ---\nSTDOUT:\n{out}\nSTDERR:\n{err}\n")

# MLPredictV2 regime distribution in logs
rc, out, err = run("""grep -E 'regime=' /var/log/ai-trader-worker.log 2>/dev/null | grep MLPredictV2 | awk -F'regime=' '{print $2}' | awk '{print $1}' | sort | uniq -c | sort -rn | head -10""")
save("61_v2_regime_freq.txt", f"$ MLPredictV2 regime freq\n--- rc={rc} ---\nSTDOUT:\n{out}\nSTDERR:\n{err}\n")

client.close()
print("Done")
