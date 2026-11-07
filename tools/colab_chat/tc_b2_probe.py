#!/usr/bin/env python3
"""The race's 12x768 pod, before anything else trains on it (stronger_base_spec.md, section 11). Claude Opus 5.5, 7 Nov 2026.

    python3 tools/colab_chat/tc_b2_probe.py --out OUT --spec SPEC.json

Sixty steps of the real shape at 8 x 2, T 1,024, on the pinned small corpus, in a scratch directory. No 12-layer figure exists
at T 1,024: this is the first. GPU memory is read twice, from the device pool's own high-water mark (MICROTORCH_STEP_TIMING) and
from nvidia-smi every 2 s, and the larger counts. It must leave room for the rung job's 2,048 MB pool beside training, and the rate
must be one the registered cost can carry. Writes OUT/probe.json. Exit 1 when either bar fails: the job is halt_on_failure, so the
pod ends before any spend on the 12-layer arm, and 8x768 goes on alone.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time

BIN = os.environ.get("JOBS_MTSTUDIO", "/content/pk/build_cuda/mtstudio")
DEVICE_MIB = 23034
ROOM_MIB = 2048 + 256        # the rung job's pool, and its margin
MAX_S_PER_STEP = 2.6         # steady, over steps 21 to 60
TAIL = 20                    # skipped: warm-up of the pool and the caches


def sampler(stop, out):
    while not stop.is_set():
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True,
                               text=True, timeout=10)
            out.append(int(r.stdout.split()[0]))
        except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
            pass
        stop.wait(2)


def read(ev_path):
    steps, peaks = [], []
    for line in open(ev_path, errors="replace"):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("event") == "step":
            steps.append(e)
            for key in ("peak_mb",):
                if isinstance(e.get(key), (int, float)):
                    peaks.append(e[key])
    return steps, peaks


def verdict(rep):
    """Pure, for the tests: the two bars on the measured numbers."""
    bad = []
    peak = max(rep["peak_pool_mib"] or 0, rep["peak_smi_mib"] or 0)
    if not peak:
        bad.append("no memory reading")
    elif peak + ROOM_MIB > DEVICE_MIB:
        bad.append(f"peak {peak} MiB leaves no room for the rung job")
    s = rep["s_per_step"]
    if s is None:
        bad.append("no rate")
    elif s > MAX_S_PER_STEP:
        bad.append(f"{s:.3f} s a step is over {MAX_S_PER_STEP}")
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--spec", required=True)
    a = ap.parse_args(argv)
    d = os.path.join(a.out, "run")
    os.makedirs(a.out, exist_ok=True)
    s = json.load(open(a.spec))
    s["out_dir"] = d
    path = os.path.join(a.out, "probe_spec.json")
    with open(path, "w") as f:
        json.dump(s, f, indent=1)
    if os.path.isdir(d):  # a replacement pod starts clean
        subprocess.run(["rm", "-rf", d], check=True)
    stop, smi = threading.Event(), []
    t = threading.Thread(target=sampler, args=(stop, smi), daemon=True)
    t.start()
    t0 = time.time()
    env = {**os.environ, "MICROTORCH_STEP_TIMING": "1"}
    with open(os.path.join(a.out, "probe.log"), "w") as log:
        rc = subprocess.run([BIN, "run", path], stdout=log, stderr=subprocess.STDOUT, env=env).returncode
    wall = time.time() - t0
    stop.set()
    t.join(5)
    steps, peaks = read(os.path.join(d, "events.jsonl")) if os.path.exists(os.path.join(d, "events.jsonl")) else ([], [])
    times = [sum(e["t_ms"].values()) / 1000 for e in steps[TAIL:] if isinstance(e.get("t_ms"), dict)]
    rep = {"producer": "Claude Opus 5.5, 7 Nov 2026", "rc": rc, "wall_seconds": round(wall, 1), "steps": len(steps),
           "params": next((json.loads(x).get("params") for x in open(os.path.join(d, "events.jsonl"), errors="replace")
                           if '"event":"model"' in x), None) if steps else None,
           "peak_pool_mib": round(max(peaks)) if peaks else None, "peak_smi_mib": max(smi) if smi else None,
           "s_per_step": round(statistics.median(times), 4) if times else None,
           "bars": {"device_mib": DEVICE_MIB, "room_mib": ROOM_MIB, "max_s_per_step": MAX_S_PER_STEP}}
    rep["failed"] = (["the run failed"] if rc != 0 else []) + verdict(rep)
    rep["ok"] = not rep["failed"]
    with open(os.path.join(a.out, "probe.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
