#!/usr/bin/env python3
"""Home side of the exploratory successive-halving arm (experiments/tiny_chat_2026-10/
PREREGISTRATION.md amendment 1; jobs tools/colab_chat/tiny_chat_asha.json; plan
experiments/tiny_chat_2026-10/asha_plan.json).

    python3 tools/colab_chat/tiny_chat_asha.py sync  --art ART --state-dir SD
    python3 tools/colab_chat/tiny_chat_asha.py drive --art ART --state-dir SD -- python3 -u tools/colab_chat/runner_multi.py ...
    python3 tools/colab_chat/tiny_chat_asha.py report --art ART

Promotion happens on the VM (tiny_chat_run.py asha_promote) from the rung.json files there. A
replaced VM has lost them, so `sync` gathers every result and decision already home (ART/<job>/,
the runner's state SD/state.json) into ART/asha_state.json, which the runner uploads on every
launch (--upload asha_state.json=ART/asha_state.json); the promotion step and jobs_vm.py's
promoted_by read it there. `drive` runs the given runner command and refreshes the state file
every minute while it runs (stateless: everything is recomputed from ART), so the watchdog can
keep it alive like any runner. `report` prints the rounds (descriptive; the arm is exploratory).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PLAN = os.path.join(ROOT, "experiments", "tiny_chat_2026-10", "asha_plan.json")


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def gather(art, state_dir, plan):
    """{"rungs": {round: {config: rung.json}}, "failed": {round: [config]}, "decisions": {round: {config: bool}}}."""
    runner = os.path.join(os.path.expanduser(state_dir), "state.json")
    jobs = jload(runner)["jobs"] if os.path.exists(runner) else {}
    out = {"rungs": {}, "failed": {}, "decisions": {}, "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
    rounds = [r["id"] for r in plan["rounds"]]
    for rid in rounds[:-1]:  # the last round's candidates are full runs with post jobs, not promoted further
        for c in plan["configs"]:
            name = f"asha_{rid}_{c['id']}"
            p = os.path.join(art, name, "rung.json")
            js = jobs.get(name, {})
            arrived = "rung.json" in js.get("fetched", []) if jobs else True  # never a file mid-download
            if os.path.exists(p) and arrived:
                out["rungs"].setdefault(rid, {})[c["id"]] = jload(p)
            elif js.get("home") and js.get("status") == "FAILED":
                out["failed"].setdefault(rid, []).append(c["id"])
    for rid in rounds[1:]:
        p = os.path.join(art, f"asha_promote_{rid}", "decisions.json")
        arrived = "decisions.json" in jobs.get(f"asha_promote_{rid}", {}).get("fetched", []) if jobs else True
        if os.path.exists(p) and arrived:
            out["decisions"].update(jload(p).get("decisions", {}))
    return out


def write_state(art, state_dir, plan):
    path = os.path.join(art, "asha_state.json")
    st = gather(art, state_dir, plan)
    os.makedirs(art, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(st, f, indent=1)
    os.replace(tmp, path)
    return path, st


def cmd_sync(a):
    path, st = write_state(a.art, a.state_dir, jload(a.plan))
    print(json.dumps({"state": path, "rungs": {r: len(v) for r, v in st["rungs"].items()},
                      "failed": st["failed"], "decided": sorted(st["decisions"])}))
    return 0


def cmd_drive(a):
    plan = jload(a.plan)
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        raise SystemExit("drive: give the runner command after --")
    write_state(a.art, a.state_dir, plan)  # the runner checks --upload sources exist at start
    p = subprocess.Popen(cmd)
    try:
        while p.poll() is None:
            time.sleep(60)
            try:
                write_state(a.art, a.state_dir, plan)
            except (OSError, ValueError) as e:  # a file mid-download: next minute
                print(f"asha sync: {e}", file=sys.stderr, flush=True)
    except KeyboardInterrupt:
        p.terminate()
    write_state(a.art, a.state_dir, plan)
    return p.wait()


def cmd_report(a):
    plan = jload(a.plan)
    for k, r in enumerate(plan["rounds"]):
        rid = r["id"]
        rows = []
        for c in plan["configs"]:
            p = os.path.join(a.art, f"asha_{rid}_{c['id']}", "rung.json")
            q = os.path.join(a.art, f"asha_post_{c['id']}", "summary.json")
            if os.path.exists(p):
                s = jload(p)
                rows.append((c["id"], c["shape"], c["lr"], c["warmup_frac"], c["share"], s.get("acc_gold"),
                             s.get("Y_base"), s.get("final_val"), s.get("diverged")))
            elif k == len(plan["rounds"]) - 1 and os.path.exists(q):
                s = jload(q)
                rows.append((c["id"], c["shape"], c["lr"], c["warmup_frac"], c["share"],
                             (s.get("final_rung") or {}).get("acc_gold"), s.get("primary"), None, None))
        print(f"\n{rid} (budget {r['budget']} steps; EXPLORATORY): config shape lr warmup share acc_gold "
              f"{'Y(SFT)' if k == len(plan['rounds']) - 1 else 'Y_base'} final_val diverged")
        for row in sorted(rows, key=lambda x: -(x[5] if x[5] is not None else -1)):
            print("  " + " ".join("-" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v)) for v in row))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    for m in ("sync", "drive", "report"):
        p = sub.add_parser(m)
        p.add_argument("--art", required=True)
        p.add_argument("--plan", default=PLAN)
        if m != "report":
            p.add_argument("--state-dir", required=True, help="the runner's --state-dir")
        if m == "drive":
            p.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    return {"sync": cmd_sync, "drive": cmd_drive, "report": cmd_report}[a.mode](a)


if __name__ == "__main__":
    raise SystemExit(main())
