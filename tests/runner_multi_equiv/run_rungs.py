#!/usr/bin/env python3
"""Rung snapshots through a replacement VM, end to end on the fake colab CLI (fakecolab.py).

A training job snapshots its weights at steps 100 and 200 (jobs_vm.py's snapshot_steps) and two light
rung jobs score them, held back (a busy light slot) until after the VM is reclaimed; the training job
is home by then. Three runs:

  OLD    the runner before the fix: both rung jobs fail "rung lost" on the replacement VM (the bug).
  NEW    each snapshot comes home once (never in a tick that relayed), goes back to the exact VM path
         before the replacement VM boots, both rung jobs score the original weights (sha256), and the
         home copies are deleted once the rung jobs are home.
  RESET  on OLD's end state, one lost snapshot placed at home (a fixture standing for a fetched copy):
         reset_lost_rungs.py lists it as resettable and the other as lost, --apply writes the request,
         and the new runner's next start reruns that rung job only; no other job's entry changes.

    python3 tests/runner_multi_equiv/run_rungs.py OLD_RUNNER NEW_RUNNER WORKDIR   # WORKDIR on ext4
"""
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("FAKE_ROOT", HERE)  # fakecolab reads it at import; only blob() and BIG are used here
import fakecolab  # noqa: E402
import run_eq  # noqa: E402

JOBS = {"jobs": [
    {"name": "data", "cmd": ["x"], "outputs": ["out.txt"]},
    {"name": "pt", "model_name": "m", "after": "data", "relay_every_steps": 200, "snapshot_steps": [100, 200],
     "spec": {}},
    {"name": "rung_s100", "after": "data", "light": True, "rung_of": {"job": "pt", "step": 100}, "cmd": ["r"],
     "outputs": ["rung.json"]},
    {"name": "rung_s200", "after": "data", "light": True, "rung_of": {"job": "pt", "step": 200}, "cmd": ["r"],
     "outputs": ["rung.json"]},
]}
# vm_status calls: 1-2 data, 3 pt starts, 4 step 100 (snapshot), 5 step 200 (snapshot, relay), 6 step 300 (pt
# FINISHED). The VM is reclaimed at the 7th; the rung jobs may start from the 8th (the replacement VM).
SCENARIO = {"reclaim_at": [7], "rung_start_at": 8}


def sha(step):
    return hashlib.sha256(fakecolab.blob(("pt", step), fakecolab.BIG)).hexdigest()


def rung_sha(work, n):
    p = os.path.join(work, "art", n, "rung.json")
    return json.load(open(p))["weights_sha256"] if os.path.exists(p) else None


def main():
    old, new, work = sys.argv[1:4]
    os.makedirs(work, exist_ok=True)
    checks = {}

    a = run_eq.run(old, os.path.join(work, "w_old"), SCENARIO, jobs=JOBS)
    w_old = os.path.join(work, "w_old")
    so = a["state"]["jobs"]
    checks["OLD: runs to the end"] = a["rc"] == 0
    checks["OLD: both rung jobs lost (the bug reproduced)"] = all(
        so[n]["status"] == "FAILED" and "rung lost" in open(os.path.join(w_old, "art", n, f"{n}_train.log")).read()
        for n in ("rung_s100", "rung_s200"))

    b = run_eq.run(new, os.path.join(work, "w_new"), SCENARIO, jobs=JOBS)
    w_new = os.path.join(work, "w_new")
    sn, log, calls = b["state"], b["log"], b["calls"].splitlines()
    checks["NEW: runs to the end"] = b["rc"] == 0
    checks["NEW: both rung jobs FINISHED and home"] = all(
        sn["jobs"][n]["status"] == "FINISHED" and sn["jobs"][n]["home"] for n in ("rung_s100", "rung_s200"))
    checks["NEW: each rung scored the original snapshot (sha256)"] = (
        rung_sha(w_new, "rung_s100") == sha(100) and rung_sha(w_new, "rung_s200") == sha(200))
    rec = sn.get("rung_snapshots", {})
    checks["NEW: both snapshots recorded with the right sha256, then deleted"] = (
        sorted(rec) == ["pt/rung_100", "pt/rung_200"] and rec["pt/rung_100"]["sha256"] == sha(100)
        and rec["pt/rung_200"]["sha256"] == sha(200) and all(e.get("deleted") for e in rec.values()))
    checks["NEW: no home copy left"] = not any(
        files for _, _, files in os.walk(os.path.join(w_new, "art", "pt", "rung_snapshots")))
    checks["NEW: downloaded once each, the step-200 one not in the relay's tick"] = (
        log.count("pt: rung snapshot at step 100 downloaded") == 1
        and log.count("pt: rung snapshot at step 200 downloaded") == 1
        and log.index("pt: relayed checkpoint at step 200") < log.index("pt: rung snapshot at step 200 downloaded")
        and log.index("pt: FINISHED") < log.index("pt: rung snapshot at step 200 downloaded"))
    up = log.find("prepare: 2 rung snapshots uploaded from home")
    checks["NEW: replacement VM got both snapshots before its boot (counted in the log)"] = (
        up > log.find("(launch 2)") > 0 and log.find("boot launched", up) > up)
    news = [i for i, c in enumerate(calls) if c.startswith('["new"')]
    ups = [i for i, c in enumerate(calls) if c.startswith('["upload"') and "__rung_" in c]
    checks["NEW: snapshot uploads (2 x 2 model parts + 2 state.txt) only on the replacement VM"] = (
        len(ups) == 6 and ups[0] > news[-1] and len(news) == 2)  # launch 1, launch 2
    vm = os.path.join(w_new, "fake", "vm", "chat", "pt")
    checks["NEW: the rung jobs dropped the uploaded snapshots at the exact path"] = (
        os.path.isdir(os.path.join(vm, "rung_100")) and not os.path.exists(os.path.join(vm, "rung_100", "model.safetensors"))
        and os.path.exists(os.path.join(vm, "rung_100", "state.txt")))

    # RESET: OLD's end state; the step-100 snapshot placed at home as a fetched copy would be.
    snap = os.path.join(w_old, "art", "pt", "rung_snapshots", "rung_100")
    os.makedirs(snap)
    open(os.path.join(snap, "model.safetensors"), "wb").write(fakecolab.blob(("pt", 100), fakecolab.BIG))
    open(os.path.join(snap, "state.txt"), "w").write("100\n{}\n")
    tool = os.path.join(os.path.dirname(new), "reset_lost_rungs.py")
    jobsf = os.path.join(w_old, "repo", "tools", "colab_chat", "eqjobs.json")
    args = [sys.executable, tool, "--jobs", jobsf, "--art", os.path.join(w_old, "art"),
            "--state-dir", os.path.join(w_old, "state")]
    p = subprocess.run(args, capture_output=True, text=True)
    checks["RESET: the tool lists s100 resettable, s200 lost"] = (
        p.returncode == 0 and "rung_s100: pt step 100: snapshot at home, can be rerun" in p.stdout
        and "rung_s200: pt step 200: LOST" in p.stdout)
    before = json.load(open(os.path.join(w_old, "state", "state.json")))
    checks["RESET: listing writes nothing"] = not os.path.exists(os.path.join(w_old, "state", "rung_reset.json"))
    p = subprocess.run(args + ["--apply"], capture_output=True, text=True)
    checks["RESET: --apply writes the request, state.json untouched"] = (
        p.returncode == 0 and json.load(open(os.path.join(w_old, "state", "rung_reset.json"))) == {"rungs": ["rung_s100"]}
        and json.load(open(os.path.join(w_old, "state", "state.json"))) == before)
    c = run_eq.run(new, w_old, {"reclaim_at": [], "rung_start_at": 0}, jobs=JOBS, fresh=False)
    after = c["state"]["jobs"]
    checks["RESET: runs to the end"] = c["rc"] == 0
    checks["RESET: rung_s100 rerun, FINISHED on the original weights"] = (
        after["rung_s100"]["status"] == "FINISHED" and after["rung_s100"]["home"]
        and rung_sha(w_old, "rung_s100") == sha(100))
    checks["RESET: the lost log kept, the request consumed"] = (
        "rung lost" in open(os.path.join(w_old, "art", "rung_s100", "rung_s100_train.lost.log")).read()
        and not os.path.exists(os.path.join(w_old, "state", "rung_reset.json")))
    def same(x, y):  # failed_seen counts every tick a job shows FAILED, home or not (unchanged runner behaviour)
        return {k: v for k, v in x.items() if k != "failed_seen"} == {k: v for k, v in y.items() if k != "failed_seen"}
    checks["RESET: no other job's entry changed"] = all(
        same(after[n], before["jobs"][n]) for n in ("data", "pt", "rung_s200"))
    checks["RESET: the snapshot went up once and was not downloaded back"] = (
        "prepare: 1 rung snapshots uploaded from home" in c["log"] and "rung snapshot at step 100 downloaded"
        not in c["log"][len(a["log"]):])
    checks["RESET: the home copy deleted once the rung job is home"] = not os.path.exists(
        os.path.join(snap, "model.safetensors"))

    for k, v in checks.items():
        print(("PASS " if v else "FAIL ") + k)
    for name, r in (("OLD", a), ("NEW", b), ("RESET", c)):
        open(os.path.join(work, f"{name}.log"), "w").write(r["log"])
        open(os.path.join(work, f"{name}.calls"), "w").write(r["calls"])
        if r["stderr"]:
            print(f"{name} stderr:", r["stderr"][-1500:])
    ok = all(checks.values())
    print("RUNGS OK" if ok else "RUNGS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
