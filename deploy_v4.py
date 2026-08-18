#!/usr/bin/env python3
"""Task 7: Deploy v4 to trader server.

Steps:
1. Read base.ts + bot-meta-selector.json + sandbox-accounts.json from trader (for inspection / patching).
2. Upload 3 TS files to /opt/ai-trader/src/strategies/.
3. Download 10 model .json files from evolution server.
4. Upload them to trader /opt/ai-trader/src/strategies/.
5. Backup + patch base.ts to register meta_selector_v4.
6. Create bot-meta-selector-v4.json.
7. Patch sandbox-accounts.json.
8. Clear BotState via prisma.
9. Restart ai-trader-worker.
10. Wait 15s, verify active.
11. Tail logs for "Loaded 14 bots" + no TS errors.
12. Wait 30s, grep PREDICT logs.
"""
import os
import sys
import time
import json
import stat
import io
import paramiko

# ----- server credentials -----
TRADER_HOST = "2.26.122.152"
TRADER_USER = "root"
TRADER_PASS = "uiF=!6FrBb&9U1Xh"

EVO_HOST = "2.26.123.205"
EVO_USER = "root"
EVO_PASS = "8oX6eTX8YQ_mrjoq"

LOCAL_DIR = "/home/z/my-project"
STG_DIR = "/opt/ai-trader/src/strategies"
BOT_DIR = "/opt/ai-trader/config/bots"
SCRIPTS_DIR = "/opt/ai-trader/scripts"

TS_FILES = [
    "meta_selector_v4.ts",
    "regime_detector.ts",
    "xgboost_binary_ts.ts",
]

MODELS = [
    "regime_strong_trend_up.json",
    "regime_mild_trend_up.json",
    "regime_range_tight.json",
    "regime_range_wide.json",
    "regime_mild_trend_down.json",
    "regime_strong_trend_down.json",
    "regime_crash.json",
    "regime_breakout_up.json",
    "regime_breakdown.json",
    "regime_high_vol_regime.json",
]

EVO_MODELS_PATH = "/root/ai-trader-evolution/ml/meta_models_v2"


def ssh_connect(host, user, pwd, timeout=30):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=user, password=pwd, timeout=timeout, look_for_keys=False, allow_agent=False)
    return client


def run(client, cmd, timeout=120, check=False):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout, get_pty=False)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    rc = stdout.channel.recv_exit_status()
    if check and rc != 0:
        raise RuntimeError(f"command failed rc={rc}: {cmd}\nSTDOUT:\n{out}\nSTDERR:\n{err}")
    return rc, out, err


def sftp_put_file(sftp, local, remote):
    sftp.put(local, remote)
    st = sftp.stat(remote)
    return st.st_size


def sftp_put_text(sftp, text, remote):
    with sftp.file(remote, "w") as f:
        f.write(text)
    st = sftp.stat(remote)
    return st.st_size


def sftp_get_file(sftp, remote, local):
    sftp.get(remote, local)
    return os.path.getsize(local)


def main():
    log = print
    log("=== Task 7: Deploy v4 to trader server ===")

    # --- connect to trader ---
    log("[1] Connecting to trader server...")
    trader = ssh_connect(TRADER_HOST, TRADER_USER, TRADER_PASS)
    tsftp = trader.open_sftp()

    # --- read current base.ts, bot-meta-selector.json, sandbox-accounts.json ---
    log("[2] Reading trader-side existing files...")
    rc, base_ts, err = run(trader, f"cat {STG_DIR}/base.ts")
    log(f"    base.ts: rc={rc}, bytes={len(base_ts)}")
    if rc != 0:
        log(err)
        sys.exit(1)
    with open("/tmp/trader_base.ts", "w") as f:
        f.write(base_ts)

    rc, bot_v2, err = run(trader, f"cat {BOT_DIR}/bot-meta-selector.json")
    log(f"    bot-meta-selector.json: rc={rc}, bytes={len(bot_v2)}")
    if rc != 0:
        log(err)
        sys.exit(1)
    with open("/tmp/bot-meta-selector.json", "w") as f:
        f.write(bot_v2)

    rc, sandbox_accs, err = run(trader, f"cat {SCRIPTS_DIR}/sandbox-accounts.json")
    log(f"    sandbox-accounts.json: rc={rc}, bytes={len(sandbox_accs)}")
    if rc != 0:
        log(err)
        sys.exit(1)
    with open("/tmp/sandbox-accounts.json", "w") as f:
        f.write(sandbox_accs)

    # --- upload TS files ---
    log("[3] Uploading 3 TS files to trader /opt/ai-trader/src/strategies/ ...")
    for fname in TS_FILES:
        local = os.path.join(LOCAL_DIR, fname)
        remote = f"{STG_DIR}/{fname}"
        sz = sftp_put_file(tsftp, local, remote)
        log(f"    {fname}: uploaded {sz} bytes")

    # --- connect to evolution ---
    log("[4] Connecting to evolution server...")
    evo = ssh_connect(EVO_HOST, EVO_USER, EVO_PASS)
    esftp = evo.open_sftp()

    # --- download 10 models from evolution ---
    log("[5] Downloading 10 model .json files from evolution...")
    os.makedirs("/tmp/v4_models", exist_ok=True)
    for fname in MODELS:
        remote = f"{EVO_MODELS_PATH}/{fname}"
        local = f"/tmp/v4_models/{fname}"
        sz = sftp_get_file(esftp, remote, local)
        log(f"    {fname}: downloaded {sz} bytes")

    esftp.close()
    evo.close()

    # --- upload 10 models to trader ---
    log("[6] Uploading 10 model .json files to trader /opt/ai-trader/src/strategies/ ...")
    for fname in MODELS:
        local = f"/tmp/v4_models/{fname}"
        remote = f"{STG_DIR}/{fname}"
        sz = sftp_put_file(tsftp, local, remote)
        log(f"    {fname}: uploaded {sz} bytes")

    # --- patch base.ts ---
    log("[7] Patching base.ts (backup first) ...")
    rc, _, err = run(trader, f"cp {STG_DIR}/base.ts {STG_DIR}/base.ts.bak.before_v4 && echo OK")
    log(f"    backup: {err.strip()}")

    # Insert the case before the default: in base.ts
    new_case = """    case 'meta_selector_v4': {
      const { MetaSelectorV4Strategy } = require('./meta_selector_v4')
      return new MetaSelectorV4Strategy()
    }
    default:"""

    if "meta_selector_v4" in base_ts:
        log("    base.ts already contains meta_selector_v4 case — skipping insertion")
    else:
        # find first occurrence of "default:" in the file
        idx = base_ts.find("default:")
        if idx == -1:
            log("    ERROR: 'default:' not found in base.ts")
            sys.exit(1)
        # check preceding whitespace (indentation)
        line_start = base_ts.rfind("\n", 0, idx) + 1
        indent = base_ts[line_start:idx]
        # re-create the case with same indent (without "    default:" leading 4 spaces — we will keep the same)
        new_block = (
            "case 'meta_selector_v4': {\n"
            + "      const { MetaSelectorV4Strategy } = require('./meta_selector_v4')\n"
            + "      return new MetaSelectorV4Strategy()\n"
            + "    }\n"
            + indent + "default:"
        )
        new_base = base_ts[:idx] + new_block + base_ts[idx + len("default:"):]
        # write back via sftp
        sz = sftp_put_text(tsftp, new_base, f"{STG_DIR}/base.ts")
        log(f"    base.ts patched: {sz} bytes (added meta_selector_v4 case)")

    # --- read v2 bot config for pattern + extract account_id ---
    log("[8] Building bot-meta-selector-v4.json ...")
    bot_v2_obj = json.loads(bot_v2)
    log(f"    bot_v2 keys: {list(bot_v2_obj.keys())}")

    # Extract account_id from v2 bot config OR sandbox-accounts.json
    account_id = None
    if "account_id" in bot_v2_obj:
        account_id = bot_v2_obj["account_id"]
    elif "accountId" in bot_v2_obj:
        account_id = bot_v2_obj["accountId"]

    if not account_id:
        # try sandbox-accounts.json — find MetaSelector entry
        sa_obj = json.loads(sandbox_accs)
        # could be list or dict
        log(f"    sandbox-accounts structure: {type(sa_obj).__name__}")
        # heuristic: find any account_id field
        if isinstance(sa_obj, list):
            for entry in sa_obj:
                if "name" in entry and "MetaSelector" in str(entry.get("name", "")):
                    account_id = entry.get("account_id") or entry.get("accountId") or entry.get("id")
                    break
            if not account_id and sa_obj:
                account_id = sa_obj[0].get("account_id") or sa_obj[0].get("accountId")
        elif isinstance(sa_obj, dict):
            # maybe "bots" or similar
            for k, v in sa_obj.items():
                if isinstance(v, list) and v:
                    account_id = v[0].get("account_id") or v[0].get("accountId")
                    break
                if isinstance(v, dict) and ("account_id" in v or "accountId" in v):
                    account_id = v.get("account_id") or v.get("accountId")
                    break
    log(f"    shared account_id = {account_id}")

    # Build v4 bot config using v2 pattern
    bot_v4 = json.loads(bot_v2)  # deep copy
    bot_v4["name"] = "MetaSelectorV4"
    if "strategy" in bot_v4:
        bot_v4["strategy"] = "meta_selector_v4"
    if "strategyName" in bot_v4:
        bot_v4["strategyName"] = "meta_selector_v4"
    if "positionSize" in bot_v4:
        bot_v4["positionSize"] = 0.10
    if "maxPositionCost" in bot_v4:
        bot_v4["maxPositionCost"] = 1500
    # make sure to not duplicate positionSize if structure uses other keys
    # force-write positionSize & maxPositionCost if not present (some configs use different keys)
    bot_v4.setdefault("positionSize", 0.10)
    bot_v4.setdefault("maxPositionCost", 1500)
    # write
    v4_config_path = f"{BOT_DIR}/bot-meta-selector-v4.json"
    v4_text = json.dumps(bot_v4, indent=2, ensure_ascii=False)
    sz = sftp_put_text(tsftp, v4_text, v4_config_path)
    log(f"    bot-meta-selector-v4.json: {sz} bytes written")

    # --- patch sandbox-accounts.json: add MetaSelectorV4 entry ---
    log("[9] Patching sandbox-accounts.json ...")
    sa_obj = json.loads(sandbox_accs)

    def make_v4_entry(template):
        entry = json.loads(json.dumps(template))
        entry["name"] = "MetaSelectorV4"
        entry["strategy"] = "meta_selector_v4"
        return entry

    if isinstance(sa_obj, list):
        # find MetaSelector entry as template
        template = None
        for e in sa_obj:
            if "name" in e and "MetaSelector" in str(e["name"]) and "V4" not in str(e["name"]):
                template = e
                break
        if template is None:
            template = sa_obj[0] if sa_obj else {}
        new_entry = make_v4_entry(template)
        # check if already present
        already = any(e.get("name") == "MetaSelectorV4" for e in sa_obj)
        if not already:
            sa_obj.append(new_entry)
            log(f"    appended MetaSelectorV4 entry (list form)")
        else:
            log(f"    MetaSelectorV4 already in list — skip")
    elif isinstance(sa_obj, dict):
        handled = False
        for list_key in ("bots", "accounts"):
            if list_key in sa_obj and isinstance(sa_obj[list_key], list):
                template = None
                for e in sa_obj[list_key]:
                    if "MetaSelector" in str(e.get("bot_name", e.get("name", ""))) and "V4" not in str(e):
                        template = e
                        break
                if template is None and sa_obj[list_key]:
                    template = sa_obj[list_key][0]
                new_entry = json.loads(json.dumps(template))
                # normalize key names: use bot_name (existing schema) or name
                if "bot_name" in new_entry:
                    new_entry["bot_name"] = "MetaSelectorV4"
                else:
                    new_entry["name"] = "MetaSelectorV4"
                # ensure account_id (shared) preserved from template
                already = any(
                    (e.get("bot_name") == "MetaSelectorV4") or (e.get("name") == "MetaSelectorV4")
                    for e in sa_obj[list_key]
                )
                if not already:
                    sa_obj[list_key].append(new_entry)
                    log(f"    appended MetaSelectorV4 entry under .{list_key}: {json.dumps(new_entry, ensure_ascii=False)}")
                else:
                    log(f"    MetaSelectorV4 already in .{list_key} — skip")
                handled = True
                break
        if not handled:
            log(f"    unexpected sandbox-accounts shape (dict keys: {list(sa_obj.keys())}); adding top-level key")
            sa_obj["MetaSelectorV4"] = {
                "name": "MetaSelectorV4",
                "strategy": "meta_selector_v4",
                "account_id": account_id,
            }

    sa_new_text = json.dumps(sa_obj, indent=2, ensure_ascii=False)
    # backup
    rc, _, err = run(trader, f"cp {SCRIPTS_DIR}/sandbox-accounts.json {SCRIPTS_DIR}/sandbox-accounts.json.bak.before_v4 && echo OK")
    log(f"    backup sandbox: {err.strip()}")
    sz = sftp_put_text(tsftp, sa_new_text, f"{SCRIPTS_DIR}/sandbox-accounts.json")
    log(f"    sandbox-accounts.json: {sz} bytes written")

    # --- clear BotState ---
    log("[10] Clearing BotState table via prisma ...")
    rc, out, err = run(trader, "cd /opt/ai-trader && ./node_modules/.bin/prisma db execute --stdin <<< \"DELETE FROM BotState;\"")
    log(f"    prisma rc={rc}")
    if out.strip():
        log(f"    STDOUT: {out.strip()}")
    if err.strip():
        log(f"    STDERR: {err.strip()}")

    # --- restart worker ---
    log("[11] Restarting ai-trader-worker ...")
    rc, out, err = run(trader, "systemctl restart ai-trader-worker && sleep 2 && systemctl is-active ai-trader-worker")
    log(f"    restart rc={rc}, active state: {out.strip()}")

    # wait 15s
    log("[12] Waiting 15s for worker to stabilise ...")
    time.sleep(15)
    rc, out, err = run(trader, "systemctl is-active ai-trader-worker")
    log(f"    is-active: {out.strip()}")

    # check logs
    log("[13] Tailing worker log for 'Loaded N bots' + TS errors ...")
    rc, out, err = run(trader, "tail -60 /var/log/ai-trader-worker.log")
    log("---- worker.log tail ----")
    log(out)
    log("---- end tail ----")

    # filter interesting lines
    rc, out2, err2 = run(trader, "tail -200 /var/log/ai-trader-worker.log | grep -E 'MetaSelectorV4|Engine|error|Error|TS[0-9]+:|tsc|Loaded [0-9]+ bot' | head -80")
    log("---- filtered lines ----")
    log(out2 if out2.strip() else "(no matching lines)")

    # wait 30s
    log("[14] Waiting 30s for PREDICT logs ...")
    time.sleep(30)
    rc, out3, err3 = run(trader, "grep MetaSelectorV4 /var/log/ai-trader-worker.log | tail -20")
    log("---- MetaSelectorV4 PREDICT lines ----")
    log(out3 if out3.strip() else "(no MetaSelectorV4 lines yet)")

    tsftp.close()
    trader.close()

    log("=== Deploy complete ===")


if __name__ == "__main__":
    main()
