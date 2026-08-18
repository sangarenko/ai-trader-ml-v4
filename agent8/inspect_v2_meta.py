#!/usr/bin/env python3
"""Inspect v2 meta-classifier (meta_models/meta_classifier.pkl) + meta_metadata.json + meta_labeler_v2."""
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

    # Look at meta_metadata.json (v1) and meta_metadata_v2.json
    rc, out, err = run(c, "cat /root/ai-trader-evolution/ml/meta_models/meta_metadata.json")
    print(f"\n[meta_metadata.json]:")
    print(out)

    rc, out, err = run(c, "cat /root/ai-trader-evolution/ml/meta_models_v2/meta_metadata_v2.json")
    print(f"\n[meta_metadata_v2.json]:")
    print(out)

    # Look at meta_backtest.py (it's the v2 backtest logic) — print top 80 lines
    rc, out, err = run(c, "head -80 /root/ai-trader-evolution/ml/meta_backtest.py")
    print(f"\n[meta_backtest.py head 80]:")
    print(out)

    # Look at the npz keys of meta_labels_v2.npz
    rc, out, err = run(c, "python3 -c \"import numpy as np; d=np.load('/root/ai-trader-evolution/ml/data_cache/meta_labels_v2.npz', allow_pickle=True); print('keys:', list(d.keys())); [print(f'  {k}: shape={d[k].shape} dtype={d[k].dtype}') for k in d.keys()]\"")
    print(f"\n[meta_labels_v2.npz structure]:")
    print(out)

    # Get the strategy_names list
    rc, out, err = run(c, "python3 -c \"import numpy as np; d=np.load('/root/ai-trader-evolution/ml/data_cache/meta_labels_v2.npz', allow_pickle=True); print('strategy_names:', list(d['strategy_names'])); print('regime_names:', list(d['regime_names']))\"")
    print(f"\n[strategy_names + regime_names]:")
    print(out)

    # Look at meta_labeler_v2.py to understand compute_regime_v2 signature
    rc, out, err = run(c, "grep -nE 'def compute_regime_v2|^REGIME_NAMES|def precompute|^def compute|TICKERS =|np.savez' /root/ai-trader-evolution/ml/meta_labeler_v2.py | head -30")
    print(f"\n[meta_labeler_v2.py defs]:")
    print(out)

    # Look at fast_backtest_v2.py precompute_indicators signature
    rc, out, err = run(c, "grep -nE 'def precompute_indicators|^def |^class ' /root/ai-trader-evolution/fast_mc/fast_backtest_v2.py | head -30")
    print(f"\n[fast_backtest_v2.py defs]:")
    print(out)

    # Look at random_hold_short strategy
    rc, out, err = run(c, "grep -nE -A 25 \"'random_hold_short':\" /root/ai-trader-evolution/fast_mc/all_22_strategies.py")
    print(f"\n[random_hold_short strategy definition]:")
    print(out)

    # Look at meta_backtest.py - find where it builds the v2 backtest (P&L per ticker)
    rc, out, err = run(c, "grep -nE 'random_hold_short|always_run_best|pnls_matrix|per_ticker' /root/ai-trader-evolution/ml/meta_backtest.py | head -30")
    print(f"\n[meta_backtest.py P&L aggregation]:")
    print(out)

    # Look at meta_backtest_result.json
    rc, out, err = run(c, "cat /root/ai-trader-evolution/ml/meta_models/meta_backtest_result.json")
    print(f"\n[meta_backtest_result.json]:")
    print(out)

    # Check the meta_classifier.pkl: load it and inspect what it expects
    rc, out, err = run(c, "python3 -c \"import pickle; m = pickle.load(open('/root/ai-trader-evolution/ml/meta_models/meta_classifier.pkl','rb')); print('type:', type(m).__name__); print('attrs:', [a for a in dir(m) if not a.startswith('_')][:20]); "
                         "print('n_features_in_:', getattr(m, 'n_features_in_', 'N/A')); "
                         "print('classes_:', getattr(m, 'classes_', 'N/A')); "
                         "import json; md = json.load(open('/root/ai-trader-evolution/ml/meta_models/meta_metadata.json')); print('meta feature_names:', md.get('feature_names')); print('meta strategy_names:', md.get('strategy_names'))\" 2>&1 | head -40")
    print(f"\n[meta_classifier.pkl inspection]:")
    print(out)

    # Look at how meta_labeler_v2.py builds the v2 dataset (the npz)
    rc, out, err = run(c, "head -60 /root/ai-trader-evolution/ml/meta_labeler_v2.py")
    print(f"\n[meta_labeler_v2.py head 60]:")
    print(out)

    # What is meta_labeler_v2.py compute_regime_v2 logic?
    rc, out, err = run(c, "grep -nE -A 30 'def compute_regime_v2' /root/ai-trader-evolution/ml/meta_labeler_v2.py | head -60")
    print(f"\n[compute_regime_v2 body]:")
    print(out)

    c.close()


if __name__ == "__main__":
    main()
