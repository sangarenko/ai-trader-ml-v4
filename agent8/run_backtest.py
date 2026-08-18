#!/usr/bin/env python3
"""Upload backtest_v4.py to evolution server + run it + capture output."""
import paramiko
import sys
import time
import os

EVO_HOST = "2.26.123.205"
EVO_USER = "root"
EVO_PASS = "8oX6eTX8YQ_mrjoq"

LOCAL_FILE = "/home/z/my-project/agent8/backtest_v4.py"
REMOTE_FILE = "/root/ai-trader-evolution/ml/backtest_v4.py"
OUTPUT_LOG = "/home/z/my-project/agent8/backtest_v4_run.log"


def ssh_connect(host, user, pwd, timeout=30):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(host, username=user, password=pwd, timeout=timeout, look_for_keys=False, allow_agent=False)
    return c


def main():
    print(f"Uploading {LOCAL_FILE} -> {REMOTE_FILE} on {EVO_HOST}...")
    c = ssh_connect(EVO_HOST, EVO_USER, EVO_PASS)
    sftp = c.open_sftp()
    sftp.put(LOCAL_FILE, REMOTE_FILE)
    sftp.close()
    print("Upload done.")

    # Get file size for verification
    stdin, stdout, stderr = c.exec_command(f"ls -la {REMOTE_FILE}")
    print(stdout.read().decode())

    # Quick syntax check before running
    print("\n[Syntax check]")
    cmd = "python3 -c 'import py_compile; py_compile.compile(\"" + REMOTE_FILE + "\", doraise=True); print(\"OK\")'"
    stdin, stdout, stderr = c.exec_command(cmd)
    print("stdout:", stdout.read().decode())
    print("stderr:", stderr.read().decode())

    # Run the backtest. Use a long timeout since it loops over 11 tickers × ~15k bars each.
    print(f"\nRunning backtest_v4.py (timeout 600s)...")
    print(f"Output saved to {OUTPUT_LOG}")
    start = time.time()
    stdin, stdout, stderr = c.exec_command(
        f"cd /root/ai-trader-evolution/ml && python3 backtest_v4.py 2>&1",
        timeout=900, get_pty=True
    )
    # Stream output as it comes (with a soft timeout)
    out_chunks = []
    while True:
        chunk = stdout.channel.recv(4096).decode("utf-8", errors="replace")
        if not chunk:
            break
        out_chunks.append(chunk)
        print(chunk, end="")
        # Also write to log file
        with open(OUTPUT_LOG, "a") as f:
            f.write(chunk)
        if time.time() - start > 900:
            print("\n[TIMEOUT] hit 900s budget, aborting output stream")
            break

    rc = stdout.channel.recv_exit_status()
    err = stderr.read().decode("utf-8", errors="replace")
    print(f"\n[exit_code={rc}] elapsed={time.time()-start:.1f}s")
    if err:
        print(f"STDERR:\n{err}")

    # Fetch the result JSON
    print("\n[Fetching v4_backtest_result.json]")
    try:
        sftp = c.open_sftp()
        sftp.get("/root/ai-trader-evolution/ml/meta_models_v2/v4_backtest_result.json",
                 "/home/z/my-project/agent8/v4_backtest_result.json")
        print("downloaded /home/z/my-project/agent8/v4_backtest_result.json")
        # Show stats
        sftp.close()
    except Exception as e:
        print(f"could not fetch result json: {e}")

    c.close()


if __name__ == "__main__":
    main()
