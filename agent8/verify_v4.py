#!/usr/bin/env python3
"""Task 8 Part 1: Verify v4 live on trader server + Part 3 sweep status on evo."""
import json
import paramiko
import sys

TRADER_HOST = "2.26.122.152"
TRADER_USER = "root"
TRADER_PASS = "uiF=!6FrBb&9U1Xh"

EVO_HOST = "2.26.123.205"
EVO_USER = "root"
EVO_PASS = "8oX6eTX8YQ_mrjoq"


def ssh_connect(host, user, pwd, timeout=30):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=user, password=pwd, timeout=timeout,
                  look_for_keys=False, allow_agent=False)
    return client


def run(client, cmd, timeout=120):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout, get_pty=False)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    rc = stdout.channel.recv_exit_status()
    return rc, out, err


def section(title):
    print("\n" + "=" * 78)
    print("=" * 78)
    print(" " + title)
    print("=" * 78)


def main():
    # ============= TRADER SERVER (Part 1) =============
    section("PART 1: Verify v4 LIVE on trader server (2.26.122.152)")

    tc = ssh_connect(TRADER_HOST, TRADER_USER, TRADER_PASS)

    # 1. Worker active?
    rc, out, err = run(tc, "systemctl is-active ai-trader-worker")
    worker_active = out.strip()
    print(f"\n[1] systemctl is-active ai-trader-worker: {worker_active!r}")

    # service info
    rc, out, err = run(tc, "systemctl status ai-trader-worker --no-pager -l | head -10")
    print("\n[1b] systemctl status (head 10 lines):")
    print(out)

    # 2. 14 bots loaded
    rc, out, err = run(tc, "grep 'Loaded.*bots' /var/log/ai-trader-worker.log | tail -3")
    print(f"\n[2] 'Loaded.*bots' lines (last 3):")
    print(out if out.strip() else "  (no match)")

    # 3. MetaSelectorV4 predictions - last 20 PREDICT lines
    rc, out, err = run(tc,
        "grep MetaSelectorV4 /var/log/ai-trader-worker.log | grep -E 'P\\(up\\)|regime=' | tail -20")
    print(f"\n[3] MetaSelectorV4 predictions (last 20 PREDICT lines):")
    print(out if out.strip() else "  (no PREDICT logs found)")

    # 3b. Count of MetaSelectorV4 lines overall and per-regime
    rc, out, err = run(tc,
        "grep -c MetaSelectorV4 /var/log/ai-trader-worker.log")
    print(f"\n[3b] Total MetaSelectorV4 log lines: {out.strip()}")

    rc, out, err = run(tc,
        "grep MetaSelectorV4 /var/log/ai-trader-worker.log | grep -oE 'regime=[A-Z_]+' | sort | uniq -c | sort -rn")
    print("\n[3c] Regime distribution in MetaSelectorV4 logs:")
    print(out if out.strip() else "  (none)")

    rc, out, err = run(tc,
        "grep MetaSelectorV4 /var/log/ai-trader-worker.log | grep -oE '→ (LONG|SHORT|FLAT)' | sort | uniq -c | sort -rn")
    print("\n[3d] Decision distribution:")
    print(out if out.strip() else "  (none)")

    # 4. Errors
    rc, out, err = run(tc,
        "grep -iE 'error|fail|crash' /var/log/ai-trader-worker.log | grep -vE '30079|Not enough|unknown bot|status=5|busy' | tail -10")
    print(f"\n[4] Real errors (filtered):")
    print(out if out.strip() else "  (no real errors)")

    # 4b. All errors (unfiltered, for context)
    rc, out, err = run(tc,
        "grep -iE 'error|fail|crash' /var/log/ai-trader-worker.log | tail -10")
    print("\n[4b] All errors (unfiltered, last 10):")
    print(out if out.strip() else "  (none)")

    # 5. Per-bot scan logs
    rc, out, err = run(tc,
        "grep 'scan done' /var/log/ai-trader-worker.log | awk '{print $2}' | sort -u | head -20")
    print(f"\n[5] 'scan done' per bot (timestamps? or bot names?):")
    print(out if out.strip() else "  (none — maybe log format differs)")
    rc, out, err = run(tc,
        "grep -E 'scan done|scanned' /var/log/ai-trader-worker.log | tail -5")
    print("  raw scan-done sample:")
    print(out if out.strip() else "  (none)")

    # 6. Daemon health on port 3008 — get sandbox accounts / portfolio balances
    rc, out, err = run(tc, "curl -sS --max-time 5 http://127.0.0.1:3008/api/accounts 2>&1 | head -c 5000")
    print(f"\n[6] GET /api/accounts (daemon port 3008):")
    print(out if out.strip() else "  (no response)")

    # Try alternative endpoints
    rc, out, err = run(tc, "curl -sS --max-time 5 http://127.0.0.1:3008/ 2>&1 | head -c 2000")
    print("\n[6b] GET / on daemon port 3008:")
    snippet = out.strip()[:1500] if out.strip() else "(no response)"
    print(snippet)

    # Try portfolio endpoints
    for ep in ["/api/portfolio", "/api/sandbox-accounts", "/api/bots", "/api/balances", "/healthz", "/api/status"]:
        rc, out, err = run(tc, f"curl -sS --max-time 5 http://127.0.0.1:3008{ep} 2>&1 | head -c 1500")
        snippet = out.strip()[:800] if out.strip() else "(empty)"
        print(f"\n[6c] GET {ep}: {snippet}")

    # sandbox-accounts.json — verify MetaSelectorV4 entry
    rc, out, err = run(tc, "cat /opt/ai-trader/sandbox-accounts.json")
    print(f"\n[7] /opt/ai-trader/sandbox-accounts.json:")
    try:
        j = json.loads(out)
        print(json.dumps(j, indent=2, ensure_ascii=False)[:2000])
    except Exception:
        print(out[:1500])

    # Bot configs available?
    rc, out, err = run(tc, "ls -la /opt/ai-trader/config/bots/ | grep -i meta")
    print(f"\n[8] meta* bot configs:")
    print(out)

    # How many bots configured
    rc, out, err = run(tc, "ls /opt/ai-trader/config/bots/*.json | wc -l")
    print(f"\n[8b] Bot config count: {out.strip()}")

    # Worker log size and last lines
    rc, out, err = run(tc, "wc -l /var/log/ai-trader-worker.log; echo '---'; tail -30 /var/log/ai-trader-worker.log")
    print(f"\n[9] Worker log size + last 30 lines:")
    print(out)

    tc.close()

    # ============= EVOLUTION SERVER (Part 3: sweep status) =============
    section("PART 3: Sweep status on evolution server (2.26.123.205)")

    ec = ssh_connect(EVO_HOST, EVO_USER, EVO_PASS)

    # Sweep running?
    rc, out, err = run(ec, "pgrep -af meta_sweep || echo 'SWEEP_NOT_RUNNING'")
    print(f"\n[A] pgrep meta_sweep:")
    print(out)

    # Sweep log
    rc, out, err = run(ec, "ls -la /tmp/meta_sweep.log 2>&1; echo '---'; tail -50 /tmp/meta_sweep.log 2>&1")
    print(f"\n[B] /tmp/meta_sweep.log (size + tail):")
    print(out)

    # Sweep results file?
    rc, out, err = run(ec,
        "ls -la /root/ai-trader-evolution/ml/sweep_results/ 2>&1; echo '---'; "
        "find /root/ai-trader-evolution/ml -name 'sweep_*.json' -o -name '*sweep*.json' 2>&1 | head -20")
    print(f"\n[C] sweep_results/ listing:")
    print(out)

    # Top 10 results if done
    rc, out, err = run(ec,
        "if [ -f /root/ai-trader-evolution/ml/sweep_results/sweep_results.json ]; then "
        "  echo '--- top 10 sweep results ---'; "
        "  python3 -c \""
        "import json; "
        "d = json.load(open('/root/ai-trader-evolution/ml/sweep_results/sweep_results.json')); "
        "print('Total experiments:', len(d) if isinstance(d, list) else 'N/A'); "
        "results = d if isinstance(d, list) else d.get('results', d.get('experiments', [])); "
        "results.sort(key=lambda r: r.get('test_pnl', r.get('val_pnl', 0)), reverse=True); "
        "[print(f\\\"#{i+1:3d} exp={r.get('exp_id', '?'):5s} val_pnl={r.get('val_pnl', r.get('validation_pnl', 0)):+8.2f} test_pnl={r.get('test_pnl', r.get('test_pnl', 0)):+8.2f} strategy={r.get('strategy', r.get('strat', '?'))}\\\") for i, r in enumerate(results[:10])]\" 2>&1; "
        "else echo 'NO RESULTS FILE YET'; fi")
    print(f"\n[D] Top 10 sweep results:")
    print(out)

    # CPU usage on evolution (to see if sweep is running)
    rc, out, err = run(ec, "top -bn1 | head -15")
    print(f"\n[E] Top processes (CPU):")
    print(out)

    ec.close()
    print("\n=== VERIFICATION COMPLETE ===\n")


if __name__ == "__main__":
    main()
