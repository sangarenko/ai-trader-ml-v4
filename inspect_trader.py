#!/usr/bin/env python3
"""Quick inspection of trader server files to validate deploy logic."""
import paramiko

TRADER_HOST = "2.26.122.152"
TRADER_USER = "root"
TRADER_PASS = "uiF=!6FrBb&9U1Xh"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(TRADER_HOST, username=TRADER_USER, password=TRADER_PASS, timeout=30, look_for_keys=False, allow_agent=False)

def run(cmd):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=60)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    rc = stdout.channel.recv_exit_status()
    return rc, out, err

# 1. List strategies dir
rc, out, err = run("ls -la /opt/ai-trader/src/strategies/ | head -50")
print("=== strategies/ ===")
print(out)

# 2. base.ts — show first 100 lines + find default + case patterns
rc, out, err = run("wc -l /opt/ai-trader/src/strategies/base.ts && grep -nE 'case .*:|default:|require\\(' /opt/ai-trader/src/strategies/base.ts | head -60")
print("=== base.ts structure ===")
print(out)

rc, out, err = run("cat /opt/ai-trader/src/strategies/base.ts")
print("=== base.ts FULL ===")
print(out)
with open("/tmp/trader_base.ts", "w") as f:
    f.write(out)

# 3. bot-meta-selector.json
rc, out, err = run("cat /opt/ai-trader/config/bots/bot-meta-selector.json")
print("=== bot-meta-selector.json ===")
print(out)
with open("/tmp/bot-meta-selector.json", "w") as f:
    f.write(out)

# 4. List all bot configs
rc, out, err = run("ls -la /opt/ai-trader/config/bots/ | head -40")
print("=== bots/ dir ===")
print(out)

# 5. sandbox-accounts.json
rc, out, err = run("cat /opt/ai-trader/scripts/sandbox-accounts.json")
print("=== sandbox-accounts.json ===")
print(out)
with open("/tmp/sandbox-accounts.json", "w") as f:
    f.write(out)

# 6. Check worker state
rc, out, err = run("systemctl is-active ai-trader-worker; systemctl status ai-trader-worker --no-pager | head -15")
print("=== worker status ===")
print(out)

# 7. existing log file?
rc, out, err = run("ls -la /var/log/ai-trader-worker.log 2>/dev/null; tail -5 /var/log/ai-trader-worker.log 2>/dev/null")
print("=== worker log ===")
print(out)

client.close()
