#!/usr/bin/env python3
"""Base 2's units guard: the Colab compute-unit balance read every --every-min minutes, and a stop asked of every
listed runner, once, when the units spent since the starting balance reach --stop-at.

    python3 tools/colab_chat/base2_units_guard.py --stop-at 640 \\
        --state-dir ~/base2_arm_N_runner --state-dir ~/base2_arm_S_runner --state-dir ~/base2_arm_C_runner \\
        [--start-balance UNITS] [--every-min 30] [--log ~/base2_units_guard.jsonl] [--once]

The balance is read as AI_ML/chat_screens_claude/cloud_eval/ccu_read.py reads it (a read-only GET of Colab's
/tun/m/ccu-info through the CLI's own login), run with the Colab CLI's Python (--cli-python, default
~/.local/share/uv/tools/google-colab-cli/bin/python); that script prints numbers only. Each reading is one line of
the JSONL log: numbers only (time, balance, hourly rate, the starting balance, spent, the bound), or the reader's
error code. The starting balance is --start-balance, else the one the log already records, else the first
reading, which is then recorded. Spent is the starting balance less the current one: units bought meanwhile
lower it, so the bound is then reached later; the log shows every reading.

At the bound each state folder gets a STOP_REQUEST file whose text is the reason (runner_multi.py takes it at its
next tick: the confirmed stop, HALTED with kind LEAD_STOP). It is written once: the log records it, and a later
reading writes nothing more. Nothing here starts, stops or touches a session itself. Claude Opus 5.5, 12 Nov 2026.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CLI_PY = "~/.local/share/uv/tools/google-colab-cli/bin/python"
CCU_READ = "/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/AI_ML/chat_screens_claude/cloud_eval/ccu_read.py"


def numbers(d):
    return {k: v for k, v in (d or {}).items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


def read_ccu(cli_python=CLI_PY, script=CCU_READ, timeout=120):
    """The ccu-info figures as numbers, or {"error": ...} (a number, or a short word: nothing from the reply's text)."""
    try:
        p = subprocess.run([os.path.expanduser(cli_python), script], capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"error": type(e).__name__}
    try:
        d = json.loads(p.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"error": f"exit {p.returncode}"}
    if "error" in d:
        return {"error": d["error"] if isinstance(d["error"], (int, float)) else "reader"}
    return numbers(d)


def records(path):
    out = []
    if os.path.exists(path):
        for ln in open(path, encoding="utf-8"):
            try:
                out.append(json.loads(ln))
            except ValueError:
                continue
    return out


def append(path, rec):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


def request_stop(state_dirs, text):
    """STOP_REQUEST in each state folder (written whole, then renamed into place). The folders it was written to."""
    done = []
    for d in state_dirs:
        d = os.path.expanduser(d)
        if not os.path.isdir(d):
            continue
        tmp = os.path.join(d, "STOP_REQUEST.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        os.replace(tmp, os.path.join(d, "STOP_REQUEST"))
        done.append(d)
    return done


def once(a, read):
    """One reading: logged; at the bound, the stop asked for (once). Returns the record written."""
    past = records(a.log)
    d = read()
    now = time.time()
    if "error" in d or not isinstance(d.get("currentBalance"), (int, float)):
        rec = {"at": now, "error": d.get("error", "no balance")}
        append(a.log, rec)
        return rec
    bal = float(d["currentBalance"])
    start = a.start_balance
    if start is None:
        start = next((r["start"] for r in past if isinstance(r.get("start"), (int, float))), bal)
    spent = round(start - bal, 4)
    rec = {"at": now, "balance": bal, "start": start, "spent": spent, "stop_at": a.stop_at}
    if isinstance(d.get("consumptionRateHourly"), (int, float)):
        rec["rate_hourly"] = d["consumptionRateHourly"]
    if spent >= a.stop_at and not any(r.get("stop_requested") for r in past):
        text = (f"units: {spent:.2f} spent since a balance of {start:.2f}, at or over the bound of {a.stop_at:.2f} "
                f"(base2_units_guard.py)")
        rec["stop_requested"] = True
        rec["state_dirs_written"] = len(request_stop(a.state_dir, text))
    append(a.log, rec)
    return rec


def main(argv=None, read=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--state-dir", action="append", required=True, help="a runner's state folder (repeat for each arm)")
    ap.add_argument("--stop-at", type=float, required=True, help="units spent since the starting balance that stop every arm")
    ap.add_argument("--start-balance", type=float, default=None)
    ap.add_argument("--every-min", type=float, default=30)
    ap.add_argument("--log", default=os.path.expanduser("~/base2_units_guard.jsonl"))
    ap.add_argument("--cli-python", default=CLI_PY)
    ap.add_argument("--ccu-read", default=CCU_READ)
    ap.add_argument("--once", action="store_true", help="one reading, then exit")
    a = ap.parse_args(argv)
    if a.stop_at <= 0 or a.every_min <= 0:
        ap.error("--stop-at and --every-min must be above 0")
    read = read or (lambda: read_ccu(a.cli_python, a.ccu_read))
    while True:
        rec = once(a, read)
        print(json.dumps({k: v for k, v in rec.items() if k != "state_dirs_written"}), flush=True)
        if a.once:
            return 0
        time.sleep(a.every_min * 60)


if __name__ == "__main__":
    sys.exit(main())
