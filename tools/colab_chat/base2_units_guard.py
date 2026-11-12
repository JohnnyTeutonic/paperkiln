#!/usr/bin/env python3
"""Base 2's units guard: the Colab compute-unit balance read every --every-min minutes, and a stop asked of every
listed runner when the units spent since the starting balance reach --stop-at.

    python3 tools/colab_chat/base2_units_guard.py --campaign base2 --stop-at 620 --start-balance UNITS \\
        --state-dir ~/base2_data_runner --state-dir ~/base2_arm_N_runner --state-dir ~/base2_arm_S_runner \\
        --state-dir ~/base2_arm_C_runner [--every-min 30] [--log ~/base2_units_guard/guard.jsonl] [--once]

The balance is read as AI_ML/chat_screens_claude/cloud_eval/ccu_read.py reads it (a read-only GET of Colab's
/tun/m/ccu-info through the CLI's own login), run with the Colab CLI's Python (--cli-python, default
~/.local/share/uv/tools/google-colab-cli/bin/python). A reading counts only when that script exits 0 and prints a
finite number as "currentBalance"; anything else is a failed reading, logged as a code, never as text.

The log (JSONL) belongs to one campaign (--campaign): every record carries the id, and a log that holds a record
of another campaign, or of none, is refused at start. It holds numbers, the campaign id, and the state folders a
stop was delivered to. The starting balance is --start-balance, else the one the log already records, else the
first reading, which is then recorded. Spent is the starting balance less the current one: units bought or
granted meanwhile lower it, so the bound is then reached later; the log shows every reading.

At the bound the guard records that a stop is requested (in the log, FIRST), then writes STOP_REQUEST (its text
the reason) into each listed state folder, recording each delivery as it is made. At every later reading it
delivers to any listed folder not yet delivered to (a folder that holds a STOP_REQUEST or STOP_REQUEST.taken
already counts as delivered). runner_multi.py takes the request at its next tick: the confirmed stop, HALTED with
kind LEAD_STOP.

ALARM, a file beside the log (and a loud line on stdout): any listed state folder that does not exist once the
stop is requested (that arm cannot be asked), or ALARM_AFTER failed readings in a row (two hours at the default
30 minutes). The next good reading with nothing missing removes it. Nothing here starts, stops or touches a
session itself. Claude Opus 5.5, 12 Nov 2026.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CLI_PY = "~/.local/share/uv/tools/google-colab-cli/bin/python"
CCU_READ = "/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/AI_ML/chat_screens_claude/cloud_eval/ccu_read.py"
ALARM_AFTER = 4   # failed readings in a row before the ALARM file


def finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def numbers(d):
    return {k: v for k, v in (d or {}).items() if finite(v)} if isinstance(d, dict) else {}


def read_ccu(cli_python=CLI_PY, script=CCU_READ, timeout=120):
    """The ccu-info figures as numbers, or {"error": ...} (a number, or a short word: nothing from the reply's text).
    Only a zero exit with a finite "currentBalance" is a reading."""
    try:
        p = subprocess.run([os.path.expanduser(cli_python), script], capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"error": type(e).__name__}
    if p.returncode != 0:
        return {"error": f"exit {p.returncode}"}
    try:
        d = json.loads(p.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"error": "not json"}
    if not isinstance(d, dict):
        return {"error": "not an object"}
    if "error" in d:
        return {"error": d["error"] if finite(d["error"]) else "reader"}
    out = numbers(d)
    return out if finite(out.get("currentBalance")) else {"error": "no finite balance"}


def records(path, campaign):
    """The log's records; SystemExit when one belongs to another campaign (or to none)."""
    out = []
    if os.path.exists(path):
        for ln in open(path, encoding="utf-8"):
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            if r.get("campaign") != campaign:
                raise SystemExit(f"{path} holds a record of campaign {r.get('campaign')!r}, not {campaign!r}: a log "
                                 "belongs to one campaign; give this one its own --log")
            out.append(r)
    return out


def append(path, rec):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


def alarm(a, lines):
    """The ALARM file beside the log, and the same lines on stdout; with no lines, the file is removed."""
    p = os.path.join(os.path.dirname(os.path.abspath(a.log)), "ALARM")
    if not lines:
        if os.path.exists(p):
            os.remove(p)
        return
    with open(p + ".tmp", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(p + ".tmp", p)
    for ln in lines:
        print(f"ALARM (base2_units_guard): {ln}", flush=True)


def deliver(a, past, text, now):
    """STOP_REQUEST to every listed folder not yet delivered to, each delivery logged as it is made. The listed
    folders that do not exist."""
    done = {r["delivered"] for r in past if isinstance(r.get("delivered"), str)}
    missing = []
    for d in a.state_dir:
        d = os.path.abspath(os.path.expanduser(d))
        if d in done:
            continue
        if not os.path.isdir(d):
            missing.append(d)
            continue
        req = os.path.join(d, "STOP_REQUEST")
        if not (os.path.exists(req) or os.path.exists(req + ".taken")):
            with open(req + ".tmp", "w", encoding="utf-8") as f:
                f.write(text + "\n")
            os.replace(req + ".tmp", req)
        append(a.log, {"at": now, "campaign": a.campaign, "delivered": d})
    return missing


def once(a, read):
    """One reading: logged; at or past the bound, the stop requested and delivered. Returns the record written."""
    past = records(a.log, a.campaign)
    d = read()
    now = time.time()
    requested = any(r.get("stop_requested") for r in past)
    if "error" in d or not finite(d.get("currentBalance")):
        rec = {"at": now, "campaign": a.campaign, "error": d.get("error", "no finite balance")}
        append(a.log, rec)
        failed = 1
        for r in reversed(past):
            if "error" not in r and "balance" not in r:
                continue   # a delivery record
            if "error" not in r:
                break
            failed += 1
        if failed >= ALARM_AFTER:
            alarm(a, [f"{failed} readings of the balance have failed in a row: the units are not watched"])
        return rec
    bal = float(d["currentBalance"])
    start = a.start_balance
    if start is None:
        start = next((r["start"] for r in past if finite(r.get("start"))), bal)
    spent = round(start - bal, 4)
    rec = {"at": now, "campaign": a.campaign, "balance": bal, "start": start, "spent": spent, "stop_at": a.stop_at}
    if finite(d.get("consumptionRateHourly")):
        rec["rate_hourly"] = d["consumptionRateHourly"]
    new = spent >= a.stop_at and not requested
    if new:
        rec["stop_requested"] = True
    append(a.log, rec)   # the request is in the log before any file is written
    missing = []
    if requested or new:
        reason = next((r for r in past if r.get("stop_requested")), rec)
        text = (f"units: {reason['spent']:.2f} spent since a balance of {reason['start']:.2f}, at or over the bound of "
                f"{reason['stop_at']:.2f} (base2_units_guard.py, campaign {a.campaign})")
        missing = deliver(a, records(a.log, a.campaign), text, now)
    alarm(a, [f"the stop is requested and the state folder {m} does not exist: that arm is not asked" for m in missing])
    return rec


def main(argv=None, read=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--campaign", required=True, help="the campaign this guard and its log belong to")
    ap.add_argument("--state-dir", action="append", required=True, help="a runner's state folder (repeat for each)")
    ap.add_argument("--stop-at", type=float, required=True, help="units spent since the starting balance that stop every arm")
    ap.add_argument("--start-balance", type=float, default=None)
    ap.add_argument("--every-min", type=float, default=30)
    ap.add_argument("--log", default=os.path.expanduser("~/base2_units_guard/guard.jsonl"))
    ap.add_argument("--cli-python", default=CLI_PY)
    ap.add_argument("--ccu-read", default=CCU_READ)
    ap.add_argument("--once", action="store_true", help="one reading, then exit")
    a = ap.parse_args(argv)
    if not (finite(a.stop_at) and a.stop_at > 0 and finite(a.every_min) and a.every_min > 0):
        ap.error("--stop-at and --every-min must be finite and above 0")
    if a.start_balance is not None and not finite(a.start_balance):
        ap.error("--start-balance must be a finite number")
    records(a.log, a.campaign)   # a log of another campaign is refused before any reading
    read = read or (lambda: read_ccu(a.cli_python, a.ccu_read))
    while True:
        rec = once(a, read)
        print(json.dumps(rec), flush=True)
        if a.once:
            return 0
        time.sleep(a.every_min * 60)


if __name__ == "__main__":
    sys.exit(main())
