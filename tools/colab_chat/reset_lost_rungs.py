#!/usr/bin/env python3
"""Rerun rung jobs that failed "rung lost" once their snapshot is home (runner_multi.py, 25 Oct 2026).

    python3 tools/colab_chat/reset_lost_rungs.py --jobs tools/colab_chat/tiny_chat_e1.json \\
        --art /mnt/c/ml_artifacts/tiny_chat/e1 --state-dir ~/pk_tc_e1_runner            # list only
    python3 tools/colab_chat/reset_lost_rungs.py ... --apply [RUNG_JOB ...]               # request

Lists every rung job that is home, FAILED, with "rung lost" in its log, and whether its snapshot is
at home (ART/<training job>/rung_snapshots/rung_<step>/). --apply writes STATE_DIR/rung_reset.json
naming those with a home snapshot (all of them, or the named ones); this tool never edits
state.json, which a live runner owns. The runner applies the request when it next starts (the
watchdog restarts it after its --max-hours, or after a reclaim kills it): each named job gets a
fresh WAITING entry, its old log is kept as <job>_train.lost.log, and the snapshot goes back to the
VM in prepare(). A rung job whose snapshot never came home cannot be rerun and is reported as lost.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import runner_multi as rm  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--jobs", required=True, help="the campaign's jobs JSON (path, or relative to the repo root)")
    ap.add_argument("--art", required=True, help="the runner's --art")
    ap.add_argument("--state-dir", required=True, help="the runner's --state-dir")
    ap.add_argument("--apply", action="store_true", help="write the reset request for the runner's next start")
    ap.add_argument("rungs", nargs="*", help="only these rung jobs (default: every one that qualifies)")
    a = ap.parse_args(argv)
    jp = a.jobs if os.path.exists(a.jobs) else os.path.join(os.path.dirname(os.path.dirname(HERE)), a.jobs)
    jobs = json.load(open(jp))["jobs"]
    rm.ART = a.art
    home = os.path.expanduser(a.state_dir)
    st = json.load(open(os.path.join(home, "state.json")))
    lost = rm.lost_rungs(st, jobs)
    names = {j["name"] for j in jobs if j.get("rung_of")}
    bad = [n for n in a.rungs if n not in names]
    if bad:
        sys.exit(f"not rung jobs of this campaign: {bad}")
    for n, pt, s, at_home in lost:
        print(f"{n}: {pt} step {s}: " + ("snapshot at home, can be rerun" if at_home else
                                          "LOST (no home copy of the snapshot; cannot be rerun)"))
    if not lost:
        print("no rung job failed 'rung lost'")
    ok = [n for n, _, _, at_home in lost if at_home and (not a.rungs or n in a.rungs)]
    refused = [n for n in a.rungs if n not in ok]
    if refused:
        print(f"not resettable (not a home FAILED 'rung lost' job with its snapshot at home): {refused}")
    if not a.apply:
        return 0
    if not ok:
        print("nothing to request")
        return 1 if refused else 0
    p = os.path.join(home, rm.RESET_REQUEST)
    try:  # a request the runner has not applied yet is kept
        ok = list(dict.fromkeys(json.load(open(p)).get("rungs", []) + ok))
    except (OSError, ValueError):
        pass
    with open(p + ".tmp", "w") as f:
        json.dump({"rungs": ok}, f, indent=1)
    os.replace(p + ".tmp", p)
    print(f"requested {ok} in {p}; the runner applies it when it next starts")
    return 1 if refused else 0


if __name__ == "__main__":
    sys.exit(main())
