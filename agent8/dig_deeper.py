#!/usr/bin/env python3
"""Task 8 Part 1b: deeper dig on the gaps from verify_v4.py."""
import json
import paramiko

TRADER_HOST = "2.26.122.152"
TRADER_USER = "root"
TRADER_PASS = "uiF=!6FrBb&9U1Xh"

EVO_HOST = "2.26.123.205"
EVO_USER = "root"
EVO_PASS = "8oX6eTX8YQ_mrjoq"


def ssh_connect(host, user, pwd, timeout=30):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(host, username=user, password=pwd, timeout=timeout, look_for_keys=False, allow_agent=False)
    return c


def run(c, cmd, timeout=120):
    stdin, stdout, stderr = c.exec_command(cmd, timeout=timeout, get_pty=False)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return stdout.channel.recv_exit_status(), out, err


def main():
    print("=" * 78)
    print("TRADER: investigate sandbox-accounts.json + daemon POST API")
    print("=" * 78)

    tc = ssh_connect(TRADER_HOST, TRADER_USER, TRADER_PASS)

    # Read sandbox-accounts.json via cat with full path verification
    rc, out, err = run(tc, "ls -la /opt/ai-trader/sandbox-accounts.json; echo '---'; cat /opt/ai-trader/sandbox-accounts.json")
    print(f"\n[sandbox-accounts.json]:")
    print(out)

    # Try POST / endpoint with action=accounts (the daemon uses POST per the GET / response)
    for payload in [
        '{"action":"accounts"}',
        '{"action":"list"}',
        '{"action":"portfolio"}',
        '{"action":"get_accounts"}',
        '{"action":"sandbox_accounts"}',
        '{"action":"status"}',
        '{"action":"bots"}',
        '{}',
    ]:
        rc, out, err = run(tc, f"curl -sS --max-time 5 -X POST -H 'Content-Type: application/json' -d '{payload}' http://127.0.0.1:3008/ 2>&1 | head -c 1500")
        print(f"\nPOST / payload={payload!r}:")
        print(f"  rc={rc}  out={out.strip()[:1000]}")

    # Look at the daemon's source code
    rc, out, err = run(tc, "find /opt/ai-trader -maxdepth 3 -name 'daemon*.py' -o -name '*daemon*.py' -o -name 'tbank_trade_daemon*.py' 2>&1 | head -5")
    print(f"\n[daemon file]: {out.strip()}")
    rc, out, err = run(tc, "systemctl cat tbank-trade-daemon 2>&1 | head -25")
    print(f"\n[daemon unit]:\n{out}")

    # Look for the daemon file
    rc, out, err = run(tc, "systemctl cat tbank-trade-daemon 2>&1 | grep -E 'ExecStart|WorkingDirectory' | head -5")
    print(f"\n[daemon exec]: {out.strip()}")
    
    # Get sandbox accounts via daemon — find its source code
    rc, out, err = run(tc, "systemctl show tbank-trade-daemon -p ExecStart --value 2>&1")
    print(f"\n[ExecStart]: {out.strip()}")

    tc.close()

    # Now check sweep results file structure on evolution
    print("\n" + "=" * 78)
    print("EVOLUTION: investigate sweep_results.json structure + best_experiment.json")
    print("=" * 78)

    ec = ssh_connect(EVO_HOST, EVO_USER, EVO_PASS)

    # Top-level structure of sweep_results.json
    rc, out, err = run(ec, "python3 -c \"import json; d=json.load(open('/root/ai-trader-evolution/ml/sweep_results/sweep_results.json')); print('type:', type(d).__name__); print('keys/len:', list(d.keys()) if isinstance(d, dict) else len(d)); print('sample entry keys:', list(d[0].keys()) if isinstance(d, list) and d else 'N/A'); print('sample entry 0:', json.dumps(d[0], indent=2)[:2000] if isinstance(d, list) and d else json.dumps(d, indent=2)[:2000])\"")
    print(f"\n[sweep_results.json structure]:")
    print(out)

    # best_experiment.json
    rc, out, err = run(ec, "cat /root/ai-trader-evolution/ml/sweep_results/best_experiment.json")
    print(f"\n[best_experiment.json]:")
    print(out)

    # Top 10 properly - try a few key candidates
    rc, out, err = run(ec, """python3 << 'PYEOF'
import json
d = json.load(open('/root/ai-trader-evolution/ml/sweep_results/sweep_results.json'))
items = d if isinstance(d, list) else d.get('results', d.get('experiments', d.get('experiments', [])))
print('Total items:', len(items))
print('Sample keys:', list(items[0].keys()) if items else 'none')
# find best by various keys
def val_of(r, *keys):
    for k in keys:
        if k in r:
            return r[k]
    return None
# show keys in each entry
print()
print('=== sample entry 0 ===')
print(json.dumps(items[0], indent=2)[:2000])
PYEOF
""")
    print(f"\n[Python inspect]:")
    print(out)

    # meta_sweep.py — check what fields it outputs
    rc, out, err = run(ec, "grep -nE 'save|results\\.|append|json.dump' /root/ai-trader-evolution/ml/meta_sweep.py | head -20")
    print(f"\n[meta_sweep.py save/append lines]:")
    print(out)

    ec.close()


if __name__ == "__main__":
    main()
