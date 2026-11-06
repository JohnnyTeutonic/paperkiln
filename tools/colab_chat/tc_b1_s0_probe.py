#!/usr/bin/env python3
"""Stage 0's crash probe on the pod (stronger_base_spec.md, section 6). Claude Opus 5.5, 7 Nov 2026.

    python3 tools/colab_chat/tc_b1_s0_probe.py --out OUT --spec tools/colab_chat/tc_b1_probe_spec.json

Three short runs of the real trainer on the volume, with the job's own device settings:
1. control, 150 steps, a save every 50.
2. the same spec, killed with SIGKILL while the step-100 generation is being written, then started again: it must resume from
   step 50 and end with the control's weights, optimiser and state, byte for byte.
3. the control's directory with its live generation damaged: a restart must fall back to the previous generation; with every
   generation damaged it must refuse.
Writes OUT/probe.json. Exit 0 only when all of that held.
"""
import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time

BIN = os.environ.get("JOBS_MTSTUDIO", "/content/pk/build_cuda/mtstudio")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def write_spec(tmpl, d, name):
    s = json.load(open(tmpl))
    s["name"], s["out_dir"] = name, d
    os.makedirs(d, exist_ok=True)
    p = d + ".spec.json"
    json.dump(s, open(p, "w"), indent=1)
    return p


def env():
    return {**os.environ, "MICROTORCH_CHECKPOINT_TXN": "1"}


def run(spec, log):
    t0 = time.time()
    with open(log, "a") as f:
        rc = subprocess.run([BIN, "run", spec], stdout=f, stderr=subprocess.STDOUT, env=env()).returncode
    return rc, round(time.time() - t0, 1)


def kill_during_write(spec, d, log, step):
    """Starts the run and SIGKILLs it once the optimiser file of generation g<step>.tmp has bytes in it."""
    target = f"{d}/ckpt_gen/g{step}.tmp/optim.safetensors"
    with open(log, "a") as f:
        p = subprocess.Popen([BIN, "run", spec], stdout=f, stderr=subprocess.STDOUT, env=env())
        while p.poll() is None:
            if os.path.exists(target) and os.path.getsize(target) > 0:
                p.send_signal(signal.SIGKILL)
                p.wait()
                return True, os.path.getsize(target)
            time.sleep(0.02)
    return False, 0


def events(d, kind):
    out = []
    for line in open(f"{d}/events.jsonl", errors="replace"):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("event") == kind:
            out.append(e)
    return out


def files(d, name):
    return {f: sha(f"{d}/{f}") for f in ("model.safetensors", "optim.safetensors", "state.txt", f"{name}.safetensors")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--spec")
    ap.add_argument("--torn", help="write a torn checkpoint for the job in this directory (the halt drill), then exit")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.torn:  # a generation that can never validate, published as live: what a torn save would have left behind
        g = f"{a.torn}/ckpt_gen/g1"
        os.makedirs(g, exist_ok=True)
        for f in ("model.safetensors", "optim.safetensors", "state.txt", "MANIFEST.json"):
            with open(f"{g}/{f}", "wb") as h:
                h.write(b"torn")
        if not os.path.lexists(f"{a.torn}/ckpt_gen/live"):
            os.symlink("g1", f"{a.torn}/ckpt_gen/live")
        with open(f"{a.out}/torn.json", "w") as f:
            json.dump({"torn": a.torn}, f)
        return 0
    rep = {"producer": "Claude Opus 5.5, 7 Nov 2026"}
    ctl, cr = f"{a.out}/control", f"{a.out}/crash"
    sc, sk = write_spec(a.spec, ctl, "probe"), write_spec(a.spec, cr, "probe")
    rc, rep["control_seconds"] = run(sc, f"{a.out}/control.log")
    rep["control_rc"] = rc
    killed, partial = kill_during_write(sk, cr, f"{a.out}/crash.log", 100)
    rep["killed_during_write"], rep["partial_bytes_at_kill"] = killed, partial
    rep["crash_live_after_kill"] = os.readlink(f"{cr}/ckpt_gen/live") if os.path.lexists(f"{cr}/ckpt_gen/live") else None
    rc, rep["resume_seconds"] = run(sk, f"{a.out}/crash.log")
    rep["resume_rc"] = rc
    rep["recover_events"] = events(cr, "checkpoint_recover")
    rep["resume_events"] = events(cr, "resume")
    fc, fk = files(ctl, "probe"), files(cr, "probe")
    rep["control_files"], rep["resumed_files"] = fc, fk
    rep["bit_identical"] = fc == fk
    rep["save_seconds"] = [e["seconds"] for e in events(ctl, "checkpoint")]
    # A damaged live generation falls back; every generation damaged is a refusal.
    fb = f"{a.out}/fallback"
    shutil.copytree(ctl, fb, symlinks=True)
    s2 = write_spec(a.spec, fb, "probe")
    live = os.readlink(f"{fb}/ckpt_gen/live")
    with open(f"{fb}/ckpt_gen/{live}/optim.safetensors", "r+b") as f:
        f.seek(4096)
        f.write(b"\xff" * 16)
    rc, _ = run(s2, f"{a.out}/fallback.log")
    rec = events(fb, "checkpoint_recover")[-1]
    rep["fallback"] = {"rc": rc, "live": rec.get("live"), "fallback": rec.get("fallback"), "rejected": rec.get("rejected")}
    for g in os.listdir(f"{fb}/ckpt_gen"):
        if g.startswith("g") and g[1:].isdigit():
            with open(f"{fb}/ckpt_gen/{g}/model.safetensors", "r+b") as f:
                f.seek(4096)
                f.write(b"\xff" * 16)
    rc, _ = run(s2, f"{a.out}/refuse.log")
    rep["refused_rc"] = rc
    rep["ok"] = (rep["control_rc"] == 0 and killed and rep["resume_rc"] == 0 and rep["bit_identical"]
                 and rep["recover_events"][-1].get("recovered_step") == 50 and rep["fallback"]["rc"] == 0
                 and rep["fallback"]["fallback"] is True and rep["fallback"]["live"] != live and rep["refused_rc"] != 0)
    with open(f"{a.out}/probe.json", "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps({k: rep[k] for k in ("ok", "bit_identical", "resume_seconds", "save_seconds")}))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
