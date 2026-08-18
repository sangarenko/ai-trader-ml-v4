#!/usr/bin/env python3
"""Find trend_slope definition in meta_labeler_v2.py + meta_trainer_v2.py."""
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

    # Search for trend_slope definition
    rc, out, err = run(c, "grep -nE 'trend_slope|trend.*slope|slope.*trend' /root/ai-trader-evolution/ml/*.py /root/ai-trader-evolution/fast_mc/*.py 2>&1 | head -30")
    print(f"\n[trend_slope references]:")
    print(out)

    # Look at meta_trainer_v2.py - it's the v2 trainer that built the 33-feature dataset
    rc, out, err = run(c, "head -200 /root/ai-trader-evolution/ml/meta_trainer_v2.py")
    print(f"\n[meta_trainer_v2.py head 200]:")
    print(out)

    c.close()


if __name__ == "__main__":
    main()
