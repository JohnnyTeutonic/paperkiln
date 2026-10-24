#!/usr/bin/env python3
"""A stand-in for the colab CLI (and colab_adopt.py) that simulates one Colab VM on local disk, so
runner_multi.py's Colab path can be run end to end and every call it makes recorded.

    FAKE_ROOT=DIR fakecolab.py sessions|new|stop|exec|upload|download ...
    FAKE_ROOT=DIR fakecolab.py adopt --refresh | --adopt EP=NAME

DIR/fake.json holds the session state and the scenario; DIR/vm is the VM's /content; DIR/calls.log
gets one line per call (argv, plus the sha256 of an exec'd file's code). The VM's jobs are simulated
from the jobs JSON (env FAKE_JOBS) one step per vm_status exec: a data job runs one step and writes
its outputs; a training job advances 100 steps a step, writing a CHUNK-crossing model file, and
finishes at FAKE_STEPS. Reclaims happen after the vm_status calls listed in the scenario.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = os.environ["FAKE_ROOT"]
VM = os.path.join(ROOT, "vm")
CHAT = os.path.join(VM, "chat")
FJ = os.path.join(ROOT, "fake.json")
BIG = 46 * 1024 * 1024 + 123
STEPS = int(os.environ.get("FAKE_STEPS", "300"))


def load():
    return json.load(open(FJ))


def save(d):
    json.dump(d, open(FJ, "w"), indent=1)


def call_log(argv, extra=""):
    with open(os.path.join(ROOT, "calls.log"), "a") as f:
        f.write(json.dumps(argv) + (" " + extra if extra else "") + "\n")


def status(n):
    p = os.path.join(CHAT, f"STATUS_{n}")
    return open(p).read().strip() if os.path.exists(p) else "NONE"


def set_status(n, s):
    open(os.path.join(CHAT, f"STATUS_{n}"), "w").write(s + "\n")


def blob(seed, size):
    h = hashlib.sha256(str(seed).encode()).digest()
    return (h * (size // len(h) + 1))[:size]


def restore(n):
    up, out = os.path.join(CHAT, "upload", n), os.path.join(CHAT, n)
    if not os.path.isdir(up):
        return
    os.makedirs(out, exist_ok=True)
    parts = sorted(p for p in os.listdir(up) if p.startswith("model.safetensors.part"))
    if parts:
        with open(os.path.join(out, "model.safetensors"), "wb") as dst:
            for p in parts:
                dst.write(open(os.path.join(up, p), "rb").read())
        for p in parts:
            os.remove(os.path.join(up, p))
    for f in ("state.txt", "events.jsonl"):
        if os.path.exists(os.path.join(up, f)):
            os.replace(os.path.join(up, f), os.path.join(out, f))


def ckpt_step(dr):
    try:
        return int(open(os.path.join(dr, "state.txt")).readline().strip() or 0)
    except (OSError, ValueError):
        return None


def take_snapshots(jobs):
    """jobs_vm.take_snapshot: a training job's checkpoint at one of its snapshot_steps is copied to
    rung_<step>/ (model.safetensors, state.txt) unless that directory exists."""
    for j in jobs:
        if not j.get("snapshot_steps") or "cmd" in j or status(j["name"]) in ("FINISHED", "FAILED"):
            continue
        out = os.path.join(CHAT, j["name"])
        s = ckpt_step(out)
        dst = os.path.join(out, f"rung_{s}")
        if s in j["snapshot_steps"] and not os.path.isdir(dst):
            os.makedirs(dst)
            for f in ("model.safetensors", "state.txt"):
                shutil.copyfile(os.path.join(out, f), os.path.join(dst, f))


def sim_step(d):
    jobs = json.load(open(os.environ["FAKE_JOBS"]))["jobs"]
    take_snapshots(jobs)
    for j in jobs:
        n, s = j["name"], status(j["name"])
        if s in ("FINISHED", "FAILED"):
            continue
        up = os.path.join(CHAT, "upload", n)
        if os.path.exists(os.path.join(up, "PENDING")) and not os.path.exists(os.path.join(up, "READY")):
            continue
        if os.path.exists(os.path.join(up, "READY")):
            restore(n)
            os.remove(os.path.join(up, "READY"))
            os.remove(os.path.join(up, "PENDING"))
        after = j.get("after") or []
        after = [after] if isinstance(after, str) else after
        deps = after + ([j["init_job"]] if j.get("init_job") else [])
        if any(status(x) == "FAILED" for x in deps):
            set_status(n, "FAILED")
            continue
        if any(status(x) != "FINISHED" for x in deps):
            continue
        out = os.path.join(CHAT, n)
        os.makedirs(out, exist_ok=True)
        log = os.path.join(CHAT, f"{n}_train.log")
        ro = j.get("rung_of")
        snap = os.path.join(CHAT, ro["job"], f"rung_{ro['step']}", "model.safetensors") if ro else None
        if ro and s != "RUNNING":  # jobs_vm.rung_state; "rung_start_at" stands for a busy light slot
            if d.get("status_calls", 0) < d.get("rung_start_at", 0):
                continue
            if not os.path.exists(snap):
                ps = ckpt_step(os.path.join(CHAT, ro["job"]))
                if status(ro["job"]) in ("FINISHED", "FAILED") or (ps is not None and ps > ro["step"]):
                    open(log, "a").write(f"rung lost: no snapshot of {ro} on this VM\n")
                    set_status(n, "FAILED")
                    return
                continue
        if ro and s == "RUNNING":  # tiny_chat_run.py rung: score the snapshot, record its sha256, drop it
            open(os.path.join(out, "rung.json"), "w").write(json.dumps(
                {"step": ro["step"], "weights_sha256": hashlib.sha256(open(snap, "rb").read()).hexdigest()}))
            os.remove(snap)
            open(log, "a").write(f"{n} done\n")
            set_status(n, "FINISHED")
            return
        if s != "RUNNING":
            set_status(n, "RUNNING")
            open(log, "a").write(f"{n} started\n")
            return
        if j.get("fake_fail"):
            open(log, "a").write(f"{n} failed" + chr(10))
            set_status(n, "FAILED")
            return
        if "cmd" in j:
            for f in j.get("outputs", []):
                p = os.path.join(out, f)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                open(p, "w").write(f"{n} output {f} launch {d['launches']}\n")
            open(log, "a").write(f"{n} done\n")
            set_status(n, "FINISHED")
            return
        try:
            step = int(open(os.path.join(out, "state.txt")).readline().strip() or 0)
        except OSError:
            step = 0
        step += 100
        open(os.path.join(out, "model.safetensors"), "wb").write(blob((n, step), BIG))
        open(os.path.join(out, "optim.safetensors"), "wb").write(blob((n, "o", step), 1000))
        open(os.path.join(out, "events.jsonl"), "a").write(json.dumps({"event": "step", "step": step}) + "\n")
        open(os.path.join(out, "state.txt"), "w").write(f"{step}\n{{}}\n")
        open(log, "a").write(f'{{"event":"step","step":{step}}}\n')
        take_snapshots(jobs)
        if step >= STEPS:
            m = j.get("model_name", n)
            shutil.copyfile(os.path.join(out, "model.safetensors"), os.path.join(out, f"{m}.safetensors"))
            open(os.path.join(out, f"{m}.gguf"), "wb").write(blob((n, "g"), 5000))
            open(os.path.join(out, "result.json"), "w").write(json.dumps({"final_step": step}))
            open(os.path.join(out, f"{m}.ir.json"), "w").write("{}")
            open(os.path.join(CHAT, f"{n}_spec.json"), "w").write(json.dumps({"name": m}))
            set_status(n, "FINISHED")
        return


def main():
    argv = sys.argv[1:]
    d = load()
    cmd = argv[0]
    if cmd == "adopt":
        call_log(["ADOPT", *argv[1:]])
        if argv[1] == "--adopt":
            print("not adopted")
            return 1
        print("refreshed")
        return 0
    if cmd == "exec":
        code = open(argv[argv.index("-f") + 1]).read()
        call_log(argv, hashlib.sha256(code.encode()).hexdigest()[:16])
    else:
        call_log(argv)
    if cmd == "sessions":
        if d["alive"]:
            print(f"[{d['name']}] fake-endpoint-{d['launches']}  L4  running")
        print("[other] somebody-else")
        return 0
    if cmd == "new":
        d["new_calls"] = d.get("new_calls", 0) + 1
        if d["new_calls"] in d.get("fail_new", []):
            save(d)
            print("Precondition Failed")
            return 1
        d["alive"], d["name"] = True, argv[argv.index("-s") + 1]
        d["launches"] += 1
        shutil.rmtree(VM, ignore_errors=True)
        os.makedirs(CHAT)
        save(d)
        print("created")
        return 0
    if cmd == "stop":
        d["alive"] = False
        save(d)
        return 0
    if not d["alive"]:
        print("Error: session not found")
        return 1
    if cmd == "exec":
        if "boot_jobs.sh" in code:
            open(os.path.join(CHAT, "STATUS"), "w").write("RUNNING\n")
            print("BOOTED")
            return 0
        if "VMJSON" in code:
            d["status_calls"] = d.get("status_calls", 0) + 1
            if d["status_calls"] in d.get("reclaim_at", []):
                d["alive"] = False
                save(d)
                print("Error: session not found")
                return 1
            save(d)
            sim_step(d)
        code = code.replace("/content/", VM + "/")
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=VM)
        sys.stdout.write(p.stdout + p.stderr)
        return p.returncode
    if cmd == "upload":
        src, dst = argv[-2], argv[-1]
        dst = dst.replace("/content/", VM + "/")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        return 0
    if cmd == "download":
        src, dst = argv[-2], argv[-1]
        k = "dl:" + src
        if src.endswith("events.jsonl") and k not in d.get("failed_dl", []):  # once per path: exec_fetch path
            d.setdefault("failed_dl", []).append(k)
            save(d)
            print("HTTP 500")
            return 1
        src = src.replace("/content/", VM + "/")
        if not os.path.exists(src):
            return 1
        shutil.copyfile(src, dst)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
