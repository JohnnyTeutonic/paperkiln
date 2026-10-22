#!/usr/bin/env python3
"""VM side of a multi-job Colab session (tools/colab_chat/runner_multi.py).

    python3 jobs_vm.py JOBS.json

Each job in JOBS.json is {"name", "spec", "after"?, "init_job"?, "init_file"?,
"init_key"?}, or a data job {"name", "cmd": [argv...], "outputs": [files]}
whose argv runs from the repo root with "{out}" = /content/chat/<name> and
"{upload}" = /content/chat/upload (finished when it exits 0; no checkpoints).
A training job runs `mtstudio run` in /content/chat/<name>/ (its
out_dir), writes /content/chat/STATUS_<name> (WAITING, RUNNING, FINISHED,
FAILED) and logs to /content/chat/<name>_train.log. A job with "after" (a
name or a list) waits until those jobs are FINISHED; "init_job" points train.init_from (or
train.upcycle_from with "init_key": "upcycle_from") at that job's final
weights, and "init_file" at a file the runner uploaded. Checkpoints the runner
relayed home come back in /content/chat/upload/<name>/ on a relaunch and are
restored before the job starts, so mtstudio resumes from them. At most
"max_parallel" jobs train at once. Idempotent: a relaunch runs it again.

Rungs and successive halving (experiments/tiny_chat_2026-10/PREREGISTRATION.md, amendment 1):
- "snapshot_steps": [500, ...] on a training job: when its checkpoint (state.txt) reaches one of
  those steps, model.safetensors and state.txt are copied to <out>/rung_<step>/ (checked against
  a concurrent save, as the runner's relay is), so the weights at that step outlive the next save.
- "rung_of": {"job": J, "step": S} on a data job: it waits for J's rung_S snapshot; when J's
  checkpoint has passed S (or J has ended) and no snapshot exists (the VM that held it was
  replaced), the job is marked FAILED with "rung lost" in its log, so the session still ends.
- "light": true: the job runs in its own pool of "max_parallel_light" slots (default 1), not in
  max_parallel's, so short scoring jobs never hold back a training job.
- "after_terminal": true: wait until every "after" job has ended, FINISHED or FAILED, and run
  either way (a promotion step that must see failed candidates too).
- "promoted_by": {"files": [F, ...], "round": R, "key": K}: hold the job until one of the JSON
  files holds decisions[R][K]; true runs it, false marks it FAILED with "not promoted" in its log
  (FAILED is the runner's only terminal state besides FINISHED, and its dependants follow it).
"""
import json
import os
import shutil
import subprocess
import sys
import time

# Overridable for a local dry run (JOBS_CHAT_DIR, JOBS_MTSTUDIO, JOBS_REPO).
C = os.environ.get("JOBS_CHAT_DIR", "/content/chat")
REPO = os.environ.get("JOBS_REPO", "/content/pk")
BIN = os.environ.get("JOBS_MTSTUDIO", "/content/pk/build_cuda/mtstudio")
TICK = float(os.environ.get("JOBS_TICK", "20"))  # seconds between manager passes (tests: shorter)
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
    # Weights without optimiser state (the runner uploads only those): mtstudio resumes the
    # optimiser cold, and its AdamW step count must restart too. Zero moments with a step count of
    # 44,000 make bias correction a no-op and the first updates several times lr.
    if not os.path.exists(f"{out}/optim.safetensors") and os.path.exists(f"{out}/state.txt"):
        lines = open(f"{out}/state.txt").read().split("\n")
        if len(lines) > 1 and lines[1].strip():
            rec = json.loads(lines[1])
            rec.pop("adamw_t", None)
            lines[1] = json.dumps(rec)
            open(f"{out}/state.txt", "w").write("\n".join(lines))
        print(f"{name}: optimiser state not uploaded; cold resume, AdamW step count from 0", flush=True)
    print(f"{name}: restored checkpoint", flush=True)


def awaiting_upload(name):
    """The runner is still uploading this job's resume checkpoint (PENDING without READY)."""
    up = f"{C}/upload/{name}"
    return os.path.exists(f"{up}/PENDING") and not os.path.exists(f"{up}/READY")


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


class Adopted:
    """A job process this manager did not start (a previous manager's): poll()
    like Popen, with success read from the run's result.json at exit."""

    def __init__(self, pid, name):
        self.pid, self.name = pid, name

    def poll(self):
        if os.path.exists(f"/proc/{self.pid}"):
            return None
        rc = f"{C}/{self.name}.rc"
        if os.path.exists(rc):
            return int(open(rc).read().strip() or 1)
        return 0 if os.path.exists(f"{C}/{self.name}/result.json") else 1


def live_job_pid(name):
    """The pid of a live `mtstudio run <this job's spec>`, or of a data job's
    wrapper, if any."""
    pf = f"{C}/{name}.pid"
    if os.path.exists(pf):
        try:
            pid = int(open(pf).read())
            if os.path.exists(f"/proc/{pid}") and not os.path.exists(f"{C}/{name}.rc"):
                return pid
        except ValueError:
            pass
    want = f"{C}/{name}_spec.json"
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            args = open(f"/proc/{d}/cmdline", "rb").read().split(b"\0")
        except OSError:
            continue
        if b"run" in args and want.encode() in args:
            return int(d)
    return None


def ckpt_step(d):
    """The step of the checkpoint in directory d (state.txt line 1), or None."""
    try:
        return int(open(f"{d}/state.txt").readline().strip() or 0)
    except (OSError, ValueError):
        return None


def take_snapshot(name, job):
    """Copy a training job's checkpoint to <out>/rung_<step>/ when its step is a snapshot step."""
    out = f"{C}/{name}"
    step = ckpt_step(out)
    if step is None or step not in job.get("snapshot_steps", []):
        return
    dst = f"{out}/rung_{step}"
    if os.path.isdir(dst):  # taken (its rung job may since have dropped the weights)
        return
    tmp = dst + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    try:
        for f in ("model.safetensors", "state.txt"):
            shutil.copyfile(f"{out}/{f}", f"{tmp}/{f}")
    except OSError:
        shutil.rmtree(tmp, ignore_errors=True)
        return
    if ckpt_step(out) == step and ckpt_step(tmp) == step:
        os.replace(tmp, dst)
        print(f"{name}: rung snapshot at step {step}", flush=True)
    else:  # overtaken by a save: the next loop tries again if the step still matches
        shutil.rmtree(tmp, ignore_errors=True)


def note(name, msg):
    with open(f"{C}/{name}_train.log", "a") as f:
        f.write(msg + "\n")


def rung_state(job):
    """'ready' (the snapshot exists), 'lost' (the training job is past the step or ended without
    one) or 'wait'."""
    r = job["rung_of"]
    snap = f"{C}/{r['job']}/rung_{r['step']}/model.safetensors"
    if os.path.exists(snap):
        return "ready"
    step = ckpt_step(f"{C}/{r['job']}")
    if status(r["job"]) in ("FINISHED", "FAILED") or (step is not None and step > r["step"]):
        return "lost"
    return "wait"


def promotion(job):
    """True, False, or None while no decision file holds this job's entry."""
    p = job["promoted_by"]
    for f in p["files"]:
        try:
            d = json.load(open(f))["decisions"][p["round"]]
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if p["key"] in d:
            return bool(d[p["key"]])
    return None


def take_lock():
    """One manager per VM: two managers reading the same status files start
    each other's jobs (20 Oct 2026)."""
    lock = f"{C}/jobs_vm.pid"
    if os.path.exists(lock):
        try:
            other = int(open(lock).read())
            if other != os.getpid() and os.path.exists(f"/proc/{other}"):
                sys.exit(f"another jobs_vm.py (pid {other}) is running; refusing")
        except ValueError:
            pass
    open(lock, "w").write(str(os.getpid()))


def main():
    cfg = json.load(open(sys.argv[1]))
    jobs = {j["name"]: j for j in cfg["jobs"]}
    max_parallel = cfg.get("max_parallel", len(jobs))
    max_light = cfg.get("max_parallel_light", 1)
    take_lock()
    t0 = time.time()
    running = {}
    deferred = set()  # restored once the runner's upload of their checkpoint is READY
    for name in jobs:
        st = status(name)
        if st == "RUNNING":
            pid = live_job_pid(name)
            if pid:
                running[name] = Adopted(pid, name)
                print(f"{name}: adopted running pid {pid}", flush=True)
                continue
        if st not in ("FINISHED", "FAILED"):
            if awaiting_upload(name):
                deferred.add(name)
            else:
                restore(name)
            set_status(name, "WAITING")
    while True:
        for name, p in list(running.items()):
            rc = p.poll()
            if rc is not None:
                set_status(name, "FINISHED" if rc == 0 else "FAILED")
                print(f"{name}: exit {rc} {time.strftime('%H:%M:%S')}", flush=True)
                del running[name]
        for name, j in jobs.items():  # rung snapshots: before any rung job looks for them
            if j.get("snapshot_steps") and "cmd" not in j and status(name) not in ("FINISHED", "FAILED"):
                take_snapshot(name, j)
        pending = [n for n in jobs if status(n) == "WAITING"]
        if not pending and not running:
            break
        for name in pending:
            light = bool(jobs[name].get("light"))
            in_pool = sum(1 for n in running if bool(jobs[n].get("light")) == light)
            if in_pool >= (max_light if light else max_parallel):
                continue
            # "delay_s": hold a job until that long after this manager started,
            # so big corpora are not tokenised at once (host RAM is shared).
            if time.time() - t0 < jobs[name].get("delay_s", 0):
                continue
            after = jobs[name].get("after") or []
            after = [after] if isinstance(after, str) else after  # one job or a list
            if jobs[name].get("after_terminal"):
                if any(status(x) not in ("FINISHED", "FAILED") for x in after):
                    continue
            else:
                if any(status(x) == "FAILED" for x in after):
                    set_status(name, "FAILED")
                    print(f"{name}: skipped, {after} failed", flush=True)
                    continue
                if any(status(x) != "FINISHED" for x in after):
                    continue
            if jobs[name].get("rung_of"):
                rs = rung_state(jobs[name])
                if rs == "wait":
                    continue
                if rs == "lost":
                    note(name, f"rung lost: no snapshot of {jobs[name]['rung_of']} on this VM")
                    set_status(name, "FAILED")
                    print(f"{name}: rung lost", flush=True)
                    continue
            if jobs[name].get("promoted_by"):
                pr = promotion(jobs[name])
                if pr is None:
                    continue
                if pr is False:
                    note(name, f"not promoted ({jobs[name]['promoted_by']['round']}, successive halving)")
                    set_status(name, "FAILED")
                    print(f"{name}: not promoted", flush=True)
                    continue
            if name in deferred:  # its checkpoint is still going up: hold it, then restore
                if awaiting_upload(name):
                    continue
                restore(name)
                deferred.discard(name)
            log = open(f"{C}/{name}_train.log", "a")
            # A job's "env" overrides the fast defaults (e.g. gpt2-family runs
            # need MICROTORCH_DEFER_DOWNLOADS=0: deferral corrupts them).
            env = {**os.environ, **ENV, **{k: str(v) for k, v in jobs[name].get("env", {}).items()}}
            if "cmd" in jobs[name]:
                # A data job (corpus build, decontamination, tokenising) on the
                # VM's spare CPUs: argv run from the repo root, "{out}" replaced
                # by the job's own directory /content/chat/<name>; the runner
                # brings the files listed in "outputs" home.
                out = f"{C}/{name}"
                os.makedirs(out, exist_ok=True)
                argv = [a.replace("{out}", out).replace("{upload}", f"{C}/upload") for a in jobs[name]["cmd"]]
                # The wrapper records pid and exit code, so a later manager can
                # adopt a data job that outlived this one.
                for f in (f"{C}/{name}.pid", f"{C}/{name}.rc"):
                    if os.path.exists(f):
                        os.remove(f)
                wrap = ['sh', '-c', f'echo $$ > {C}/{name}.pid; "$@"; r=$?; echo $r > {C}/{name}.rc; exit $r', 'sh',
                        *argv]
                running[name] = subprocess.Popen(wrap, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                                 env=env)
            else:
                spec = spec_for(jobs[name], jobs)
                running[name] = subprocess.Popen([BIN, "run", spec], stdout=log, stderr=subprocess.STDOUT, env=env)
            set_status(name, "RUNNING")
            print(f"{name}: started {time.strftime('%H:%M:%S')}", flush=True)
        time.sleep(TICK)
    print("all jobs done", flush=True)


if __name__ == "__main__":
    main()
