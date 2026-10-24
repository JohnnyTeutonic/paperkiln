#!/usr/bin/env python3
"""Colab-path equivalence of runner_multi.py: the committed runner (OLD) and the refactored one (NEW)
each drive the same fake colab CLI (fakecolab.py) through the same scenario: a failed `new`, three
silent ticks, a launch with a 46 MB --upload (two parts), a data job, a training job with relays of a
46 MB checkpoint (parts cut on the VM), events.jsonl downloads that fail once (exec fallback), a reclaim
mid-training (relaunch: data job rerun, resume checkpoint uploaded in parts, READY marker), a second
reclaim after the training job is home (its final weights uploaded for the dependent post job), a
failing data job, and the end (all home, session stopped). Every colab call (argv and exec code
hash), the runner log (timestamps stripped), state.json and the artifact tree must be identical.

    git show HEAD:tools/colab_chat/runner_multi.py > ~/eq_old/runner_multi.py   # e.g. the committed one
    RECLAIM=6,14 python3 tests/runner_multi_equiv/run_eq.py ~/eq_old/runner_multi.py \\
        tools/colab_chat/runner_multi.py ~/eq_work     # WORKDIR on ext4, not OneDrive

RECLAIM lists the vm_status calls after which the fake VM is reclaimed (24 Oct 2026: EQUIVALENT for
6,14; 4,9,17; 3,5,8,11 and none, 188+ colab calls each, between 14032d9b's runner and the Transport one).
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
JOBS = {"jobs": [
    {"name": "data", "cmd": ["x"], "outputs": ["out.txt", "sub/report.json"]},
    {"name": "pt", "model_name": "m", "after": "data", "spec": {}},
    {"name": "post", "after": "pt", "init_job": "pt", "cmd": ["y"], "outputs": ["summary.json"]},
    {"name": "bad", "after": "post", "cmd": ["z"], "outputs": ["x.json"], "fake_fail": True},
]}
WRAP = """import importlib.util, sys
spec = importlib.util.spec_from_file_location("runner_multi", sys.argv[1])
rm = importlib.util.module_from_spec(spec)
sys.path.insert(0, __import__("os").path.dirname(sys.argv[1]))
spec.loader.exec_module(rm)
rm.COL, rm.COLPY, rm.ADOPT, rm.TICK = sys.argv[2], sys.argv[3], "ADOPT", 0
sys.argv = ["runner_multi.py"] + sys.argv[4:]
rm.main()
"""


def tree(d):
    out = {}
    for root, _, files in os.walk(d):
        for f in files:
            p = os.path.join(root, f)
            out[os.path.relpath(p, d)] = hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
    return out


def run(runner, work, scenario, jobs=JOBS, fresh=True):
    """One runner run against the fake. fresh=False (run_rungs.py) runs again on the same work dir: the
    state dir, artifacts and the fake's session state carry over, and the scenario keys are updated."""
    layout = os.path.join(work, "repo", "tools", "colab_chat")
    fake = os.path.join(work, "fake")
    if fresh:
        shutil.rmtree(work, ignore_errors=True)
        os.makedirs(layout)
        os.makedirs(fake)
        json.dump({"alive": False, "launches": 0, "name": None, **scenario}, open(os.path.join(fake, "fake.json"), "w"))
    else:
        d = json.load(open(os.path.join(fake, "fake.json")))
        json.dump({**d, **scenario}, open(os.path.join(fake, "fake.json"), "w"))
    for f in os.listdir(os.path.dirname(runner)):
        if f.endswith(".py") or f.endswith(".sh"):
            shutil.copyfile(os.path.join(os.path.dirname(runner), f), os.path.join(layout, f))
    json.dump(jobs, open(os.path.join(layout, "eqjobs.json"), "w"))
    col = os.path.join(work, "colab")
    open(col, "w").write(f"#!/bin/sh\nexec {sys.executable} {HERE}/fakecolab.py \"$@\"\n")
    colpy = os.path.join(work, "colpy")
    open(colpy, "w").write(f"#!/bin/sh\nshift\nexec {sys.executable} {HERE}/fakecolab.py adopt \"$@\"\n")
    os.chmod(col, 0o755)
    os.chmod(colpy, 0o755)
    up = os.path.join(work, "vocab.gguf")
    open(up, "wb").write(os.urandom(0) + hashlib.sha256(b"v").digest() * (47 * 1024 * 1024 // 32))
    wrap = os.path.join(work, "wrap.py")
    open(wrap, "w").write(WRAP)
    env = dict(os.environ, FAKE_ROOT=fake, FAKE_JOBS=os.path.join(layout, "eqjobs.json"))
    p = subprocess.run([sys.executable, wrap, os.path.join(layout, "runner_multi.py"), col, colpy,
                        "--commit", "abc123", "--jobs", "tools/colab_chat/eqjobs.json", "--name", "pk-eq",
                        "--art", os.path.join(work, "art"), "--state-dir", os.path.join(work, "state"),
                        "--max-hours", "0.2", "--upload", f"vocab.gguf={up}"],
                       capture_output=True, text=True, env=env, timeout=1800)
    calls = open(os.path.join(fake, "calls.log")).read().replace(work, "WORK")
    rlog = "\n".join(ln[15:] for ln in open(os.path.join(work, "state", "runner.log")).read().splitlines())
    st = json.load(open(os.path.join(work, "state", "state.json")))
    return {"rc": p.returncode, "stderr": p.stderr[-3000:], "calls": calls, "log": rlog.replace(work, "WORK"),
            "state": st, "art": tree(os.path.join(work, "art"))}


def main():
    old, new, work = sys.argv[1:4]
    scenario = {"fail_new": [1], "reclaim_at": [int(x) for x in os.environ.get("RECLAIM", "6,14").split(",")]}
    a = run(old, os.path.join(work, "w"), scenario)
    shutil.move(os.path.join(work, "w"), os.path.join(work, "w_old"))
    b = run(new, os.path.join(work, "w"), scenario)
    shutil.rmtree(os.path.join(work, "w_new"), ignore_errors=True)
    shutil.move(os.path.join(work, "w"), os.path.join(work, "w_new"))
    sb = dict(b["state"])
    added = {k: sb.pop(k) for k in ("campaign_home",) if k in sb and k not in a["state"]}  # OLD before a90ec0e2
    res = {"rc": [a["rc"], b["rc"]], "calls_equal": a["calls"] == b["calls"], "log_equal": a["log"] == b["log"],
           "state_equal": a["state"] == sb, "art_equal": a["art"] == b["art"], "new_state_fields": added,
           "n_calls": a["calls"].count("\n"), "n_log": a["log"].count("\n")}
    print(json.dumps(res, indent=1))
    open(os.path.join(work, "old.log"), "w").write(a["log"])
    open(os.path.join(work, "new.log"), "w").write(b["log"])
    open(os.path.join(work, "old.calls"), "w").write(a["calls"])
    open(os.path.join(work, "new.calls"), "w").write(b["calls"])
    if a["stderr"] or b["stderr"]:
        print("OLD stderr:", a["stderr"][-1500:], "\nNEW stderr:", b["stderr"][-1500:])
    ok = all(res[k] for k in ("calls_equal", "log_equal", "state_equal", "art_equal")) and res["rc"] == [0, 0]
    print("EQUIVALENT" if ok else "DIFFERENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
