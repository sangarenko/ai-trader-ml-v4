#!/usr/bin/env python3
"""Inspect features_v4.py + train_regime_models_v4.py + metadata to understand v4 training pipeline."""
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

    # 1. Download train_regime_models_v4.py locally to /tmp/agent8
    sftp = c.open_sftp()
    for f in ["features_v4.py", "train_regime_models_v4.py", "regime_strategy_mapping.py"]:
        sftp.get(f"/root/ai-trader-evolution/ml/{f}", f"/home/z/my-project/agent8/{f}")
        print(f"downloaded {f}")
    sftp.get("/root/ai-trader-evolution/ml/meta_models_v2/regime_models_v4_metadata.json",
             "/home/z/my-project/agent8/regime_models_v4_metadata.json")
    print("downloaded regime_models_v4_metadata.json")
    sftp.get("/root/ai-trader-evolution/ml/meta_models_v2/regime_models_v4_train_summary.json",
             "/home/z/my-project/agent8/regime_models_v4_train_summary.json")
    print("downloaded regime_models_v4_train_summary.json")
    sftp.close()

    # 2. Show top of train_regime_models_v4.py
    rc, out, err = run(c, "head -80 /root/ai-trader-evolution/ml/train_regime_models_v4.py")
    print(f"\n[train_regime_models_v4.py head 80]:")
    print(out)

    # 3. Look for compute_regime_v2 / compute_regime function in train_regime_models_v4.py + features_v4.py
    rc, out, err = run(c, "grep -nE 'def compute_regime|REGIME_NAMES|regime_names|REGIME_THRESHOLDS|compute_features' /root/ai-trader-evolution/ml/train_regime_models_v4.py /root/ai-trader-evolution/ml/features_v4.py | head -40")
    print(f"\n[regime-related lines]:")
    print(out)

    # 4. Show metadata file fully (regime_names + feature_names)
    rc, out, err = run(c, "python3 -c \"import json; d=json.load(open('/root/ai-trader-evolution/ml/meta_models_v2/regime_models_v4_metadata.json')); m=d['_meta']; print('regime_names:', m['regime_names']); print('feature_names:', m['feature_names']); print('horizon_bars:', m['horizon_bars']); print('horizon_minutes:', m['horizon_minutes']); print('threshold:', m['threshold']); print('long_threshold:', m['long_threshold']); print('short_threshold:', m['short_threshold']); print('decision_rule:', m['decision_rule']); print('fallback:', m.get('fallback')); print('regimes (top-level keys):', [k for k in d.keys() if k != '_meta'])\"")
    print(f"\n[metadata meta]:")
    print(out)

    # 5. Show metadata per-regime summary (e.g., RANGE_TIGHT)
    rc, out, err = run(c, "python3 -c \"import json; d=json.load(open('/root/ai-trader-evolution/ml/meta_models_v2/regime_models_v4_metadata.json')); rt=d.get('RANGE_TIGHT', {}); print('RANGE_TIGHT keys:', list(rt.keys())); print(json.dumps(rt, indent=2)[:1500])\"")
    print(f"\n[RANGE_TIGHT metadata]:")
    print(out)

    # 6. Check what's in the data_cache mtf_SBER_180d.npz
    rc, out, err = run(c, "python3 -c \"import numpy as np; d=np.load('/root/ai-trader-evolution/ml/data_cache/mtf_SBER_180d.npz', allow_pickle=True); print('keys:', list(d.keys())); [print(f'  {k}: shape={d[k].shape} dtype={d[k].dtype}') for k in d.keys()]\"")
    print(f"\n[mtf_SBER_180d.npz structure]:")
    print(out)

    # 7. Check what load_all_tickers + align returns (used by training)
    rc, out, err = run(c, "grep -nE 'def align|aligned\\[|5min_close|load_all_tickers|load_candles' /root/ai-trader-evolution/ml/ml_data_pipeline.py /root/ai-trader-evolution/ml/data_loader.py /root/ai-trader-evolution/ml/ml_features.py | head -40")
    print(f"\n[load+align helpers]:")
    print(out)

    # 8. Look at how train_regime_models_v4 builds the dataset
    rc, out, err = run(c, "grep -nE 'aligned|load_candles|mtf_|np.load|fetch_' /root/ai-trader-evolution/ml/train_regime_models_v4.py | head -30")
    print(f"\n[train_v4 dataset loading]:")
    print(out)

    # 9. Look at best MC strategy to compare with
    rc, out, err = run(c, "ls /root/ai-trader-evolution/fast_mc/results/ | head -20; echo '---'; python3 -c \"import json; d=json.load(open('/root/ai-trader-evolution/fast_mc/results/all_models_1m_core0.json')); print('type:', type(d).__name__); print('len:', len(d) if isinstance(d, list) else list(d.keys())[:5]); print('first entry keys:', list(d[0].keys()) if isinstance(d, list) and d else 'N/A'); print('first 2 entries:', json.dumps(d[:2] if isinstance(d, list) else list(d.items())[:2], indent=2)[:2000])\" 2>&1 | head -80")
    print(f"\n[fast_mc all_models_1m_core0.json]:")
    print(out)

    # 10. Find the random_hold_short strategy
    rc, out, err = run(c, "grep -lE 'random_hold_short' /root/ai-trader-evolution/fast_mc/*.py /root/ai-trader-evolution/ml/*.py 2>&1 | head -5")
    print(f"\n[files referencing random_hold_short]:")
    print(out)

    rc, out, err = run(c, "grep -nE 'random_hold_short|random.*hold|strat.*random' /root/ai-trader-evolution/fast_mc/*.py 2>&1 | head -20")
    print(f"\n[random_hold_short references]:")
    print(out)

    c.close()


if __name__ == "__main__":
    main()
