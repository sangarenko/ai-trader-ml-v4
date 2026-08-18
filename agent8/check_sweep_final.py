#!/usr/bin/env python3
"""Final sweep status check + top 3 experiments."""
import json
import paramiko

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
    return stdout.channel.recv_exit_status(), stdout.read().decode("utf-8", errors="replace"), stderr.read().decode("utf-8", errors="replace")


def main():
    c = ssh_connect(EVO_HOST, EVO_USER, EVO_PASS)

    # Sweep status
    rc, out, err = run(c, "pgrep -af meta_sweep || echo 'SWEEP_DONE'")
    print(f"\n[A] pgrep meta_sweep:")
    print(out)

    # Sweep log tail
    rc, out, err = run(c, "tail -15 /tmp/meta_sweep.log")
    print(f"\n[B] /tmp/meta_sweep.log tail:")
    print(out)

    # Top 3 experiments by switch_144 total_pnl (the main sweep metric per meta_sweep.py line 509)
    rc, out, err = run(c, """python3 << 'PYEOF'
import json
d = json.load(open('/root/ai-trader-evolution/ml/sweep_results/sweep_results.json'))
print('Total experiments:', len(d))
# sort by backtest.switch_144.total_pnl descending
def get_pnl(r):
    try:
        return r['backtest']['switch_144']['total_pnl']
    except Exception:
        return -1e18
d_sorted = sorted(d, key=get_pnl, reverse=True)
print()
print('=== Top 3 experiments by switch_144 P&L ===')
for i, r in enumerate(d_sorted[:3]):
    cfg = r['exp_config']
    bt = r['backtest']['switch_144']
    print(f'#{i+1}: pool={cfg["strategy_pool"]}, feats={cfg["feature_subset"]}, '
          f'n_est={cfg["n_estimators"]}, max_depth={cfg["max_depth"]}, lr={cfg["learning_rate"]}, '
          f'min_cw={cfg["min_child_weight"]}, gamma={cfg["gamma"]}, reg_lambda={cfg["reg_lambda"]}')
    print(f'    n_samples={r["n_samples"]}, n_feat={r["n_features"]}, n_classes_eff={r["n_classes_effective"]}, pool_size={r["strategy_pool_size"]}')
    print(f'    metrics: val_top1={r["metrics"]["val_top1"]:.3f}, val_top3={r["metrics"]["val_top3"]:.3f}, test_top1={r["metrics"]["test_top1"]:.3f}, test_top3={r["metrics"]["test_top3"]:.3f}')
    print(f'    backtest switch_144: total_pnl={bt["total_pnl"]:+.2f}, total_trades={bt["total_trades"]}, return_pct={bt["return_pct"]:+.4f}%')
    print(f'    elapsed: {r["elapsed_s"]:.1f}s')
print()
print('=== Sweep overall stats ===')
pnls = [get_pnl(r) for r in d]
import statistics
print(f'  Best:  {max(pnls):+.2f} RUB')
print(f'  Worst: {min(pnls):+.2f} RUB')
print(f'  Mean:  {statistics.mean(pnls):+.2f} RUB')
print(f'  Median: {statistics.median(pnls):+.2f} RUB')
print(f'  # positive: {sum(1 for p in pnls if p > 0)}/{len(pnls)}')
PYEOF
""")
    print(f"\n[C] Top 3 sweep experiments:")
    print(out)

    # Also show best_experiment.json (the file the sweep saves)
    rc, out, err = run(c, "python3 -c \"import json; d=json.load(open('/root/ai-trader-evolution/ml/sweep_results/best_experiment.json')); cfg=d['exp_config']; bt=d['backtest']; print('best_experiment.json:'); print('  config:', cfg); print('  switch_144 P&L:', bt['switch_144']['total_pnl'], 'trades:', bt['switch_144']['total_trades']); print('  switch_288 P&L:', bt['switch_288']['total_pnl'])\"")
    print(f"\n[D] best_experiment.json summary:")
    print(out)

    c.close()


if __name__ == "__main__":
    main()
