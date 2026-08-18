#!/usr/bin/env python3
"""Inspect evolution server ML infra: data_loader, ml_features, regime_detector, meta_models_v2 metadata."""
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

    # 1. List /root/ai-trader-evolution/ml
    rc, out, err = run(c, "ls -la /root/ai-trader-evolution/ml/")
    print(f"\n[ml/ dir listing]:")
    print(out)

    # 2. List /root/ai-trader-evolution/ml/meta_models_v2
    rc, out, err = run(c, "ls -la /root/ai-trader-evolution/ml/meta_models_v2/")
    print(f"\n[meta_models_v2/]:")
    print(out)

    # 3. List /root/ai-trader-evolution/ml/meta_models
    rc, out, err = run(c, "ls -la /root/ai-trader-evolution/ml/meta_models/ 2>&1")
    print(f"\n[meta_models/ (v2):]")
    print(out)

    # 4. Show ml_features.py compute_features signature
    rc, out, err = run(c, "grep -nE '^def |^class ' /root/ai-trader-evolution/ml/ml_features.py | head -30")
    print(f"\n[ml_features.py defs]:")
    print(out)

    # 5. Show data_loader.py defs
    rc, out, err = run(c, "grep -nE '^def |^class ' /root/ai-trader-evolution/ml/data_loader.py | head -30")
    print(f"\n[data_loader.py defs]:")
    print(out)

    # 6. Show regime_detector.py defs
    rc, out, err = run(c, "grep -nE '^def |^class ' /root/ai-trader-evolution/ml/regime_detector.py 2>&1 | head -30; "
                         "echo '---'; ls /root/ai-trader-evolution/ml/regime*.py 2>&1")
    print(f"\n[regime file + defs]:")
    print(out)

    # 7. Show regime_models_v4_metadata.json structure
    rc, out, err = run(c, "python3 -c \"import json; d=json.load(open('/root/ai-trader-evolution/ml/meta_models_v2/regime_models_v4_metadata.json')); print('top keys:', list(d.keys())); "
                         "print('_meta keys:', list(d.get('_meta', {}).keys())); "
                         "print('regimes count:', len(d.get('regimes', []))); "
                         "print('regime[0]:', json.dumps(d.get('regimes', [{}])[0], indent=2)[:1500])\"")
    print(f"\n[regime_models_v4_metadata.json]:")
    print(out)

    # 8. Find training script that uses compute_features + compute_regime
    rc, out, err = run(c, "ls /root/ai-trader-evolution/ml/train_v4*.py /root/ai-trader-evolution/ml/*train*v4*.py 2>&1")
    print(f"\n[train v4 files]:")
    print(out)
    rc, out, err = run(c, "grep -lE 'meta_models_v2|regime_models_v4' /root/ai-trader-evolution/ml/*.py 2>&1 | head -10")
    print(f"\n[files referencing v4]:")
    print(out)

    # 9. Show data_cache contents
    rc, out, err = run(c, "ls -la /root/ai-trader-evolution/ml/data_cache/ 2>&1 | head -30")
    print(f"\n[data_cache/]:")
    print(out)

    # 10. Find Monte Carlo best strategies file
    rc, out, err = run(c, "find /root/ai-trader-evolution -maxdepth 3 -name '*.json' 2>/dev/null | grep -iE 'best|mc|monte|profitable' | head -10")
    print(f"\n[MC best files]:")
    print(out)

    # 11. Show ml_features.py compute_features body
    rc, out, err = run(c, "head -120 /root/ai-trader-evolution/ml/ml_features.py")
    print(f"\n[ml_features.py head 120]:")
    print(out)

    c.close()


if __name__ == "__main__":
    main()
