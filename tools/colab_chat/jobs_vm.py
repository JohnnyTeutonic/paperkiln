#!/usr/bin/env python3
"""VM side of a multi-job Colab session (tools/colab_chat/runner_multi.py).

    python3 jobs_vm.py JOBS.json

Each job in JOBS.json is {"name", "spec", "after"?, "init_job"?, "init_file"?,
"init_key"?}. A job trains with `mtstudio run` in /content/chat/<name>/ (its
out_dir), writes /content/chat/STATUS_<name> (WAITING, RUNNING, FINISHED,
FAILED) and logs to /content/chat/<name>_train.log. A job with "after" waits
until that job is FINISHED; "init_job" points train.init_from (or
train.upcycle_from with "init_key": "upcycle_from") at that job's final
weights, and "init_file" at a file the runner uploaded. Checkpoints the runner
relayed home come back in /content/chat/upload/<name>/ on a relaunch and are
restored before the job starts, so mtstudio resumes from them. At most
"max_parallel" jobs train at once. Idempotent: a relaunch runs it again.
"""
import json
import os
import shutil
import subprocess
import sys
import time

# Overridable for a local dry run (JOBS_CHAT_DIR, JOBS_MTSTUDIO).
C = os.environ.get("JOBS_CHAT_DIR", "/content/chat")
BIN = os.environ.get("JOBS_MTSTUDIO", "/content/pk/build_cuda/mtstudio")
ENV = {"MICROTORCH_DEVICE": "cuda", "MICROTORCH_DEVICE_OPS": "1", "MICROTORCH_ATTN_BLOCKED": "1",
       "MICROTORCH_STEP_RESIDENCY": "1", "MICROTORCH_DEFER_DOWNLOADS": "1", "MICROTORCH_CUBLAS": "1",
       "OMP_NUM_THREADS": "4"}


def status(name):
    p = f"{C}/STATUS_{name}"
    return open(p).read().strip() if os.path.exists(p) else "NONE"


def set_status(name, s):
    with open(f"{C}/STATUS_{name}.tmp", "w") as f:
        f.write(s + "\n")
    os.replace(f"{C}/STATUS_{name}.tmp", f"{C}/STATUS_{name}")


def restore(name):
    """Reassemble an uploaded checkpoint (45 MB parts) into the job's dir."""
    up, out = f"{C}/upload/{name}", f"{C}/{name}"
    if not os.path.isdir(up):
        return
    os.makedirs(out, exist_ok=True)
    for f in ("model.safetensors", "optim.safetensors"):
        parts = sorted(p for p in os.listdir(up) if p.startswith(f + ".part"))
        if parts:
            with open(f"{out}/{f}", "wb") as dst:
                for p in parts:
                    with open(f"{up}/{p}", "rb") as src:
                        shutil.copyfileobj(src, dst)
            for p in parts:
                os.remove(f"{up}/{p}")
    for f in ("state.txt", "events.jsonl"):
        if os.path.exists(f"{up}/{f}"):
            os.replace(f"{up}/{f}", f"{out}/{f}")
    print(f"{name}: restored checkpoint", flush=True)


def spec_for(job, jobs):
    spec = json.loads(json.dumps(job["spec"]))
    spec["name"] = job.get("model_name", job["name"])
    spec["out_dir"] = f"{C}/{job['name']}"
    tr = spec.setdefault("train", {})
    key = job.get("init_key", "init_from")
    if job.get("init_job"):
        parent = jobs[job["init_job"]]
        tr[key] = f"{C}/{parent['name']}/{parent.get('model_name', parent['name'])}.safetensors"
    elif job.get("init_file"):
        tr[key] = f"{C}/upload/{job['init_file']}"
    path = f"{C}/{job['name']}_spec.json"
    with open(path, "w") as f:
        json.dump(spec, f, indent=1)
    return path


def main():
    cfg = json.load(open(sys.argv[1]))
    jobs = {j["name"]: j for j in cfg["jobs"]}
    max_parallel = cfg.get("max_parallel", len(jobs))
    running = {}
    for name in jobs:
        if status(name) not in ("FINISHED", "FAILED"):
            restore(name)
            set_status(name, "WAITING")
    while True:
        for name, p in list(running.items()):
            rc = p.poll()
            if rc is not None:
                set_status(name, "FINISHED" if rc == 0 else "FAILED")
                print(f"{name}: exit {rc} {time.strftime('%H:%M:%S')}", flush=True)
                del running[name]
        pending = [n for n in jobs if status(n) == "WAITING"]
        if not pending and not running:
            break
        for name in pending:
            if len(running) >= max_parallel:
                break
            after = jobs[name].get("after")
            if after and status(after) == "FAILED":
                set_status(name, "FAILED")
                print(f"{name}: skipped, {after} failed", flush=True)
                continue
            if after and status(after) != "FINISHED":
                continue
            spec = spec_for(jobs[name], jobs)
            log = open(f"{C}/{name}_train.log", "a")
            # A job's "env" overrides the fast defaults (e.g. gpt2-family runs
            # need MICROTORCH_DEFER_DOWNLOADS=0: deferral corrupts them).
            env = {**os.environ, **ENV, **{k: str(v) for k, v in jobs[name].get("env", {}).items()}}
            running[name] = subprocess.Popen([BIN, "run", spec], stdout=log, stderr=subprocess.STDOUT, env=env)
            set_status(name, "RUNNING")
            print(f"{name}: started {time.strftime('%H:%M:%S')}", flush=True)
        time.sleep(20)
    print("all jobs done", flush=True)


if __name__ == "__main__":
    main()
