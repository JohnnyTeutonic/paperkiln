#!/usr/bin/env python3
"""Reclaim-resilient Colab (or RunPod) driver for a session of several training jobs.

    python3 tools/colab_chat/runner_multi.py --commit SHA --jobs tools/colab_chat/JOBS.json \\
        --name pk-cap --art /mnt/c/ml_artifacts/capacity --state-dir ~/pk_cap_runner \\
        --upload fw_base.safetensors=/mnt/c/ml_artifacts/fw_base/fw_base.safetensors ...

The multi-job sibling of runner.py, with the same rules (hard timeouts on
every colab call, the session key refreshed each tick, a dropped session
re-adopted by its own recorded endpoint only, the session stopped when all
jobs are done or at the deadline). The VM side is boot_jobs.sh + jobs_vm.py:
each job trains in /content/chat/<job>/ and reports STATUS_<job>. Per job,
this runner relays the checkpoint home whenever its step advances (45 MB
chunks back up on a relaunch, so a reclaim costs at most one checkpoint
interval per job) and downloads the final outputs when it finishes.
--upload files go to /content/chat/upload/ on every launch (job inputs such
as an init checkpoint or SFT examples). Artifacts: ART/<job>/.

Backends (24 Oct 2026). Every VM operation goes through a Transport: create, listed/alive, exec,
put, get, stop. --backend colab (the default) is the colab CLI exactly as before. --backend runpod
rents a pod (runpod_api.py): REST create/terminate, exec over ssh, put/get by rsync, a network
volume per campaign mounted at /content (checkpoints, data and the build survive a pod's death, so
a relaunch in the same data centre re-uploads nothing the volume holds), the account-wide pod cap
and balance floor checked before every create, a spend ledger in state.json, and the pod
terminated the moment the queue is empty or the driver exits (finally, atexit, SIGTERM/SIGINT/
SIGHUP). runpod_sweep.py (run by the Windows watchdog each tick) terminates pods whose driver is
dead and deletes the volumes of campaigns that are home. A state dir moves from Colab to RunPod
like a replacement VM: stop the Colab driver, start the same command with --backend runpod.

Rung snapshots (25 Oct 2026). jobs_vm.py copies a training job's weights at each of its
"snapshot_steps" to /content/chat/<job>/rung_<step>/ (model.safetensors, state.txt: weights only),
and a "rung_of" job scores them later, often hours later. They lived only on the VM, so a
replacement VM failed the rung jobs not yet run ("rung lost"). Now each snapshot comes home once,
one per tick and never in a tick that relayed a checkpoint (ART/<job>/rung_snapshots/rung_<step>/,
recorded in state.json "rung_snapshots"); prepare() puts it back at the same VM path before the
boot when an unfinished rung job would otherwise lose it (its training job is home or resumes past
the step; a copy already on a RunPod volume is used as it is); the home copy is deleted once every
rung job of that snapshot is home and FINISHED (its rung.json records the weights' sha256). A rung
job that failed "rung lost" while its snapshot is home is rerun after reset_lost_rungs.py, applied
when the runner next starts. A campaign without rung jobs makes exactly the calls it made before.

Picks (25 Oct 2026). A tiny_chat_run.py pick reads every post job's summary.json and only the best run's
SFT model, and a post job's files live only on the VM that ran it, so a replacement VM reran every home
post an unfinished pick waits for (E5: six, ~1.4 L4 hours each, ahead of its pretraining). prepare() now
reruns, per pick, only the home post that pick's own rule (tiny_chat_run.pick_best over the home
summaries) can still choose; every other home post gets its home summary.json at the path pick reads and
is marked FINISHED without running (spared_posts). Posts not yet home run anyway. A campaign without pick
jobs makes exactly the calls it made before (tests/runner_multi_equiv: run_eq.py, run_picks.py).
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

COL = os.path.expanduser("~/.local/bin/colab")
COLPY = os.path.expanduser("~/.local/share/uv/tools/google-colab-cli/bin/python")
HERE = os.path.dirname(os.path.abspath(__file__))
ADOPT = os.path.join(os.path.dirname(HERE), "colab_adopt.py")
CHUNK = 45 * 1024 * 1024
TICK = 240
NAME = ART = HOME = None
BOOT_SCRIPT = "tools/colab_chat/boot_jobs.sh"
T = None  # the Transport (ColabTransport unless --backend runpod)


def log(msg):
    line = f"{time.strftime('%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(os.path.join(HOME, "runner.log"), "a") as f:
        f.write(line + "\n")


def sh(args, timeout):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL,
                           start_new_session=True)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"


def colab(*args, timeout=240):
    return sh([COL, *args], timeout)


# ---------------------------------------------------------------- transports
class ColabTransport:
    """The colab CLI, call for call what this runner did before the Transport existed."""
    backend = "colab"
    persistent = False   # a fresh VM's disk starts empty
    whole_files = False  # uploads go in CHUNK parts, downloads above CHUNK in parts cut on the VM

    def __init__(self, gpu="L4"):
        self.gpu = gpu

    def create(self, st):
        return colab("new", "-s", NAME, "--gpu", self.gpu, timeout=600)

    def listed(self):
        rc, out = colab("sessions", timeout=120)
        m = re.search(r"^\[" + re.escape(NAME) + r"\]\s+(\S+)", out, re.M)
        return m.group(1) if m else None

    def refresh(self, st):
        sh([COLPY, ADOPT, "--refresh"], 180)

    def adopt(self, ep):
        rc, out = sh([COLPY, ADOPT, "--adopt", f"{ep}={NAME}"], 180) if ep else (1, "")
        return rc == 0 and "adopted" in out

    def gone(self, st):
        return False  # the Colab listing is never proof: three silent ticks decide

    def exec(self, code, timeout=200):
        path = os.path.join(HOME, "exec.py")
        with open(path, "w") as f:
            f.write(code)
        return colab("exec", "-s", NAME, "-f", path, "--timeout", "150", timeout=timeout)

    def put(self, src, dst, timeout):
        return colab("upload", "-s", NAME, src, dst, timeout=timeout)

    def get(self, src, dst, timeout):
        return colab("download", "-s", NAME, src, dst, timeout=timeout)

    def stop(self, st):
        colab("stop", "-s", NAME, timeout=120)

    def release(self, st):
        pass  # nothing outlives a Colab session

    def over_budget(self, st):
        return None


def exec_py(code, timeout=200):
    return T.exec(code, timeout)


def listed():
    return T.listed()


class State:
    def __init__(self):
        p = os.path.join(HOME, "state.json")
        self.d = json.load(open(p)) if os.path.exists(p) else {"endpoint": None, "launches": 0, "jobs": {}}

    def job(self, name):
        return self.d["jobs"].setdefault(name, {"relayed_step": 0, "status": None, "home": False})

    def save(self):
        json.dump(self.d, open(os.path.join(HOME, "state.json"), "w"), indent=1)


def remote_sizes(d):
    """{file name: size} in a VM directory, or {} when it cannot be read."""
    rc, out = exec_py("import os, json\n"
                      f"d = {d!r}\n"
                      "print('SIZES', json.dumps({f: os.path.getsize(os.path.join(d, f)) for f in os.listdir(d)}"
                      " if os.path.isdir(d) else {}))\n")
    m = re.search(r"^SIZES (.*)$", out, re.M)
    return json.loads(m.group(1)) if m else {}


def upload_file(src, dst):
    """Upload in 45 MB parts named dst.partNN (jobs_vm.py reassembles job
    checkpoints; whole input files are reassembled by the exec below). Parts
    already on the VM at full size are skipped, so a failed upload resumes
    instead of starting over (at ~0.6 MB/s a restart cost 20 minutes), and a
    failed part is retried before the upload gives up. A whole-file transport
    (RunPod) sends the file as one part, dst.part00, after the same checks."""
    if T.whole_files:
        return T.upload_whole(src, dst)
    have = remote_sizes(os.path.dirname(dst))
    with open(src, "rb") as fh:
        i = 0
        while True:
            buf = fh.read(CHUNK)
            if not buf:
                break
            name = f"{os.path.basename(dst)}.part{i:02d}"
            if have.get(name) == len(buf):
                i += 1
                continue
            part = os.path.join(HOME, f"up.part{i:02d}")
            open(part, "wb").write(buf)
            for _ in range(3):
                rc, _ = T.put(part, f"{dst}.part{i:02d}", 1500)
                if rc == 0:
                    break
                log(f"upload of {src} part {i} failed; retrying")
            os.remove(part)
            if rc != 0:
                raise RuntimeError(f"upload of {src} part {i} failed")
            i += 1


SLICE = 4 * 1024 * 1024


def exec_fetch(src, size, dst):
    """Fallback for a VM whose Jupyter contents API fails on every file (HTTP 500, pk_base_v1's VM
    from 22 Oct 2026, 18:00 UTC) while exec still works: the file comes through exec's stdout as
    base64 in SLICE-sized pieces, each checked against its own sha256 (about 1 MB/s)."""
    def piece(off):
        for _ in range(4):  # an exec occasionally comes back empty under load; retry before giving up
            rc, out = exec_py(
                "import base64, hashlib\n"
                f"with open({src!r}, 'rb') as f:\n"
                f"    f.seek({off})\n"
                f"    b = f.read({SLICE})\n"
                "print('SLICESHA', hashlib.sha256(b).hexdigest())\n"
                "print('SLICEB64', base64.b64encode(b).decode())\n")
            sha = re.search(r"^SLICESHA ([0-9a-f]{64})$", out, re.M)
            b64 = re.search(r"^SLICEB64 ([A-Za-z0-9+/=]+)$", out, re.M)
            if sha and b64:
                buf = base64.b64decode(b64.group(1))
                if hashlib.sha256(buf).hexdigest() == sha.group(1):
                    return buf
        return None

    with open(dst, "wb") as out_f:
        for off in range(0, size, SLICE):
            buf = piece(off)
            if buf is None:
                log(f"exec fetch of {src} failed at offset {off}")
                return False
            out_f.write(buf)
    return os.path.getsize(dst) == size


def download_file(src, dst, timeout=1800):
    """Download a VM file; one above CHUNK comes in CHUNK-sized parts cut on the VM, each part
    resumable across ticks, reassembled here and checked against the VM's sha256 (23 Oct 2026:
    the Jupyter contents API returned HTTP 500 for every 390 MB relay of pk_base_v1). Returns
    0 on success, as colab() does. A whole-file transport (RunPod) fetches it in one resumable
    rsync, checked against the VM's sha256 the same way."""
    if T.whole_files:
        return T.download_whole(src, dst, timeout)
    key = re.sub(r"[^A-Za-z0-9_.-]", "_", src.strip("/"))
    rc, out = exec_py(
        "import hashlib, json, os\n"
        f"src, key, chunk = {src!r}, {key!r}, {CHUNK}\n"
        "if not os.path.exists(src):\n"
        "    print('DLMETA', json.dumps(None))\n"
        "else:\n"
        "    size = os.path.getsize(src)\n"
        "    h = hashlib.sha256()\n"
        "    with open(src, 'rb') as f:\n"
        "        for b in iter(lambda: f.read(1 << 22), b''):\n"
        "            h.update(b)\n"
        "    sha = h.hexdigest()\n"
        "    key = key + '.' + sha[:16]  # parts of an older file at the same path are never reused\n"
        "    parts = []\n"
        "    if size > chunk:\n"
        "        d = '/content/chat/dl'\n"
        "        os.makedirs(d, exist_ok=True)\n"
        "        with open(src, 'rb') as f:\n"
        "            i = 0\n"
        "            while True:\n"
        "                buf = f.read(chunk)\n"
        "                if not buf:\n"
        "                    break\n"
        "                p = f'{d}/{key}.part{i:02d}'\n"
        "                if not (os.path.exists(p) and os.path.getsize(p) == len(buf)):\n"
        "                    open(p + '.tmp', 'wb').write(buf)\n"
        "                    os.replace(p + '.tmp', p)\n"
        "                parts.append([p, len(buf)])\n"
        "                i += 1\n"
        "    print('DLMETA', json.dumps({'size': size, 'sha256': sha, 'parts': parts}))\n",
        timeout=600)
    m = re.search(r"^DLMETA (.*)$", out, re.M)
    meta = json.loads(m.group(1)) if m else None
    if not meta:
        return 1
    if not meta["parts"]:
        rc, _ = T.get(src, dst, timeout)
        if rc == 0:
            return 0
        if not exec_fetch(src, meta["size"], dst + ".dltmp"):
            return 1
        with open(dst + ".dltmp", "rb") as f:
            if hashlib.sha256(f.read()).hexdigest() != meta["sha256"]:
                return 1
        os.replace(dst + ".dltmp", dst)
        return 0
    local = os.path.join(HOME, "dl")
    os.makedirs(local, exist_ok=True)
    for p, n in meta["parts"]:
        lp = os.path.join(local, os.path.basename(p))
        if os.path.exists(lp) and os.path.getsize(lp) == n:
            continue
        rc, _ = T.get(p, lp + ".tmp", 900)
        if rc != 0 or not os.path.exists(lp + ".tmp") or os.path.getsize(lp + ".tmp") != n:
            if not exec_fetch(p, n, lp + ".tmp"):
                return 1
        os.replace(lp + ".tmp", lp)
    h = hashlib.sha256()
    with open(dst + ".dltmp", "wb") as out_f:
        for p, _ in meta["parts"]:
            with open(os.path.join(local, os.path.basename(p)), "rb") as f:
                buf = f.read()
            h.update(buf)
            out_f.write(buf)
    if h.hexdigest() != meta["sha256"]:
        os.remove(dst + ".dltmp")
        for p, _ in meta["parts"]:
            os.remove(os.path.join(local, os.path.basename(p)))
        log(f"download of {src}: sha256 mismatch after reassembly; parts discarded")
        return 1
    os.replace(dst + ".dltmp", dst)
    for p, _ in meta["parts"]:
        os.remove(os.path.join(local, os.path.basename(p)))
    exec_py(f"import os\nfor p in {[p for p, _ in meta['parts']]!r}:\n    os.path.exists(p) and os.remove(p)\n")
    return 0


def relay(st, name, every=0):
    """If the job's step advanced (by at least `every`), snapshot its
    checkpoint set on the VM and download the snapshot. Downloading the live
    files raced the saves: at ~1.6 MB/s a 0.85-1.1 GB set takes longer than a
    save interval, so every relay came home torn and was discarded. True when a
    download was attempted (that tick fetches no rung snapshot)."""
    art = os.path.join(ART, name)
    os.makedirs(art, exist_ok=True)
    js = st.job(name)
    src, snap = f"/content/chat/{name}", f"/content/chat/relay_{name}"
    rc, out = exec_py(
        "import os, shutil\n"
        f"src, snap = {src!r}, {snap!r}\n"
        "def step():\n"
        "    p = os.path.join(src, 'state.txt')\n"
        "    return int(open(p).readline().strip() or 0) if os.path.exists(p) else 0\n"
        "s0 = step()\n"
        # A snapshot not yet home is kept until it is (a slow download must not restart each tick).
        "sp = os.path.join(snap, 'state.txt')\n"
        "s1 = int(open(sp).readline().strip() or 0) if os.path.exists(sp) else 0\n"
        f"if s1 > {js['relayed_step']}:\n"
        "    print('SNAP', s1)\n"
        f"elif s0 > {js['relayed_step']} and s0 - {js['relayed_step']} >= {every}:\n"
        "    os.makedirs(snap + '.tmp', exist_ok=True)\n"
        "    for f in ('model.safetensors', 'optim.safetensors', 'events.jsonl', 'state.txt'):\n"
        "        shutil.copyfile(os.path.join(src, f), os.path.join(snap + '.tmp', f))\n"
        "    if step() == s0 and int(open(os.path.join(snap + '.tmp', 'state.txt')).readline()) == s0:\n"
        "        shutil.rmtree(snap, ignore_errors=True)\n"
        "        os.replace(snap + '.tmp', snap)\n"
        "        print('SNAP', s0)\n"
        "    else:\n"
        "        print('SNAPTORN', s0)\n"
        "else:\n"
        "    print('SNAPSKIP', s0)\n")
    m = re.search(r"SNAP (\d+)", out)
    if not m:
        if "SNAPTORN" in out:
            log(f"{name}: snapshot overtaken by a save; will retry")
        return False
    step = int(m.group(1))
    ok = True
    # Weights, events and step only (23 Oct 2026): a resume is weights-only (cold AdamW) anyway, and
    # the 0.78 GB optimiser download failed on every relay from step 28,000 of pk_base_v1.
    for f in ("model.safetensors", "events.jsonl", "state.txt"):
        dst = os.path.join(art, "state.txt.new" if f == "state.txt" else f + ".new")
        ok = ok and download_file(f"{snap}/{f}", dst, timeout=2400) == 0
    tmp = os.path.join(art, "state.txt.new")
    ok = ok and os.path.exists(tmp) and int(open(tmp).readline().strip() or 0) == step
    if not ok:
        log(f"{name}: relay download of the step-{step} snapshot failed; will retry")
        return True
    for f in ("model.safetensors", "events.jsonl"):
        os.replace(os.path.join(art, f + ".new"), os.path.join(art, f))
    stale = os.path.join(art, "optim.safetensors")  # older than these weights: never resume from it
    if os.path.exists(stale):
        os.remove(stale)
    os.replace(tmp, os.path.join(art, "state.txt"))  # last: state.txt names a complete set
    js["relayed_step"] = step
    st.save()
    log(f"{name}: relayed checkpoint at step {step}")
    return True


def fetch_final(st, name, model_name):
    art = os.path.join(ART, name)
    os.makedirs(art, exist_ok=True)
    ok = True
    for f in (f"{name}/{model_name}.safetensors", f"{name}/{model_name}.gguf", f"{name}/events.jsonl",
              f"{name}/result.json", f"{name}_spec.json", f"{name}_train.log"):
        ok = ok and download_file(f"/content/chat/{f}", os.path.join(art, os.path.basename(f))) == 0
    # Optional extra exports (e.g. q8_0_all) are fetched best-effort.
    download_file(f"/content/chat/{name}/{model_name}_q8_0_all.gguf", os.path.join(art, f"{model_name}_q8_0_all.gguf"))
    T.get(f"/content/chat/{name}/{model_name}.ir.json", os.path.join(art, f"{model_name}.ir.json"), 300)
    if ok:
        st.job(name)["home"] = True
        st.save()
        log(f"{name}: final outputs downloaded")


def output_files(name, outputs):
    """What fetch_outputs brings home for a data job: its declared outputs and its log."""
    return [*outputs, f"../{name}_train.log"]


def outputs_landed(js, name, outputs):
    """True when every file of output_files() is recorded as downloaded in the job's state."""
    return set(output_files(name, outputs)) <= set(js.get("fetched") or [])


def fetch_outputs(st, name, outputs, best_effort=False):
    """A data job's declared outputs, paths relative to /content/chat/<name>/,
    plus its log: ONE file per call (per tick), so a gigabyte of outputs never
    holds up the checkpoint relays (21 Oct 2026: a 26-minute download blocked
    them and a reclaim lost an hour of training). The job counts as home, and
    FINISHED, in the call whose download completes the set (28 Oct 2026: it used to
    wait for the next tick's call, and a VM lost in between left a job whose files
    were all home unmarked, so the replacement VM ran it again: post_e1_rec_MC7_r2,
    1.5 L4-hours on each of two relaunches). best_effort (a failed job): all at once,
    misses ignored; its status stays as the VM reported it."""
    art = os.path.join(ART, name)
    os.makedirs(art, exist_ok=True)
    js = st.job(name)
    got = set(js.get("fetched", []))
    for f in output_files(name, outputs):
        if f in got:
            continue
        # Outputs keep their relative paths (several sources write report.json).
        dst = os.path.join(art, os.path.basename(f) if f.startswith("../") else f)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        rc = download_file(f"/content/chat/{name}/{f}", dst)
        if rc == 0:
            js.setdefault("fetched", []).append(f)
            got.add(f)
            st.save()
            log(f"{name}: {f} downloaded")
        if not best_effort and not T.whole_files:  # RunPod's rsync is fast: all of them now, no idle pod ticks
            break
    if best_effort or outputs_landed(js, name, outputs):
        js["home"] = True
        if not best_effort:
            js["status"] = "FINISHED"
        st.save()
        log(f"{name}: outputs downloaded")


def adopt_home(st, jobs):
    """On adopting a state dir (runner start, every launch's prepare): a data job whose outputs are all
    recorded as downloaded is home and FINISHED, whatever status the last VM reported, so prepare() marks
    it FINISHED on the VM instead of running it again (state written before the fix above could hold
    such a job unmarked). A FAILED job keeps its status (its dependants must still see the failure); the
    files are checked on disk too, so a state entry alone never marks a job home. Returns the names."""
    done = []
    for j in jobs:
        n = j["name"]
        js = st.d["jobs"].get(n)
        if "cmd" not in j or not js or js.get("home") or js.get("status") == "FAILED":
            continue
        outs = j.get("outputs", [])
        if not outputs_landed(js, n, outs):
            continue
        art = os.path.join(ART, n)
        if not all(os.path.exists(os.path.join(art, os.path.basename(f) if f.startswith("../") else f))
                   for f in output_files(n, outs)):
            continue
        js["home"], js["status"] = True, "FINISHED"
        done.append(n)
    if done:
        st.save()
        log(f"outputs already home, marked FINISHED (not run again): {done}")
    return done


# ---------------------------------------------------------------- rung snapshots
RUNG_FILES = ("model.safetensors", "state.txt")  # what jobs_vm.take_snapshot copies; the rung job reads both


def rung_owners(jobs):
    """{(training job, step): [rung jobs that score its snapshot at that step]}, in jobs order."""
    own = {}
    for j in jobs:
        ro = j.get("rung_of")
        if ro:
            own.setdefault((ro["job"], int(ro["step"])), []).append(j["name"])
    return own


def snap_home(pt, step):
    return os.path.join(ART, pt, "rung_snapshots", f"rung_{step}")


def snap_vm(pt, step):
    return f"/content/chat/{pt}/rung_{step}"


def file_step(p):
    try:
        return int(open(p).readline().strip() or 0)
    except (OSError, ValueError):
        return None


def snap_at_home(pt, step):
    d = snap_home(pt, step)
    return os.path.exists(os.path.join(d, "model.safetensors")) and file_step(os.path.join(d, "state.txt")) == step


def fetch_rung_snapshot(st, jobs, r, busy):
    """Each tick, after the relays: home copies whose rung jobs are all home and FINISHED are deleted
    (rung.json is the receipt); then, unless this tick relayed a checkpoint (busy), ONE snapshot on
    the VM that an unscored rung job still needs comes home (one exec finds them; none while no
    training job with rung jobs is active on the VM)."""
    own = rung_owners(jobs)
    rec = st.d.get("rung_snapshots") or {}
    for key, e in rec.items():
        pt, step = key.rsplit("/rung_", 1)
        names = own.get((pt, int(step)), [])
        if not e.get("deleted") and names and all(
                st.job(n)["home"] and st.job(n)["status"] == "FINISHED" for n in names):
            shutil.rmtree(snap_home(pt, int(step)), ignore_errors=True)
            e["deleted"] = time.time()
            st.save()
            log(f"{key}: rung snapshot's home copy deleted ({', '.join(names)} home)")
    if busy:
        return
    cand = [(pt, s) for (pt, s), names in own.items() if f"{pt}/rung_{s}" not in rec
            and r.get(pt) in ("RUNNING", "FINISHED", "FAILED")
            and not all(st.job(n)["home"] or r.get(n) == "FINISHED" for n in names)]
    if not cand:
        return
    rc, out = exec_py(
        "import json, os\n"
        f"cand = {[[pt, s, snap_vm(pt, s)] for pt, s in cand]!r}\n"
        "def step(p):\n"
        "    try:\n"
        "        return int(open(p).readline().strip() or 0)\n"
        "    except (OSError, ValueError):\n"
        "        return None\n"
        "have = [[pt, s] for pt, s, d in cand if os.path.exists(d + '/model.safetensors') and step(d + '/state.txt') == s]\n"
        "print('RUNGSNAP', json.dumps(have))\n")
    m = re.search(r"^RUNGSNAP (.*)$", out, re.M)
    have = json.loads(m.group(1)) if m else []
    if not have:
        return
    pt, s = have[0]
    d = snap_home(pt, s)
    os.makedirs(d, exist_ok=True)
    ok = True
    for f in RUNG_FILES:
        ok = ok and download_file(f"{snap_vm(pt, s)}/{f}", os.path.join(d, f + ".new"), timeout=2400) == 0
    ok = ok and file_step(os.path.join(d, "state.txt.new")) == s
    if not ok:
        log(f"{pt}: download of the rung_{s} snapshot failed; will retry")
        return
    for f in RUNG_FILES:
        os.replace(os.path.join(d, f + ".new"), os.path.join(d, f))
    h = hashlib.sha256()
    with open(os.path.join(d, "model.safetensors"), "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    size = os.path.getsize(os.path.join(d, "model.safetensors"))
    st.d.setdefault("rung_snapshots", {})[f"{pt}/rung_{s}"] = {"bytes": size, "sha256": h.hexdigest(),
                                                               "fetched": time.time()}
    st.save()
    log(f"{pt}: rung snapshot at step {s} downloaded ({size / 1e6:.0f} MB)")


def rung_inputs(st, jobs):
    """prepare(): the home rung snapshots this VM must hold before the boot, as (key, src, VM path)
    whole-file inputs. A snapshot is put back when an unfinished rung job needs it and the VM would
    not make it again: its training job is home, or resumes (from the relayed or the volume's
    checkpoint) past the step. Already on the VM (a RunPod volume): left as it is. The training
    job's events.jsonl goes too when the VM has none (the rung job reads the step's val_loss there).
    Rung jobs reset by reset_lost_rungs.py lose the FAILED status a volume still holds. [] when
    there is nothing to do (no VM call); None when the VM did not answer."""
    need, resets = [], []
    for (pt, s), names in rung_owners(jobs).items():
        open_ = [n for n in names if not st.job(n)["home"]]
        resets += [n for n in open_ if st.job(n).get("rung_reset")]
        if open_ and snap_at_home(pt, s):
            need.append([pt, s])
    if not need and not resets:
        return []
    rc, out = exec_py(
        "import json, os\n"
        f"need, resets = {need!r}, {resets!r}\n"
        "def step(p):\n"
        "    try:\n"
        "        return int(open(p).readline().strip() or 0)\n"
        "    except (OSError, ValueError):\n"
        "        return None\n"
        "r = {}\n"
        "for pt, s in need:\n"
        "    d = '/content/chat/' + pt\n"
        "    snap = d + '/rung_' + str(s)\n"
        "    r[pt + '/' + str(s)] = [os.path.exists(snap + '/model.safetensors') and step(snap + '/state.txt') == s,\n"
        "                          step(d + '/state.txt'), os.path.exists(d + '/events.jsonl')]\n"
        "for n in resets:\n"
        "    p = '/content/chat/STATUS_' + n\n"
        "    if os.path.exists(p):\n"
        "        os.remove(p)\n"
        "print('RUNGHAS', json.dumps(r))\n")
    m = re.search(r"^RUNGHAS (.*)$", out, re.M)
    if not m:
        return None
    vm = json.loads(m.group(1))
    for n in resets:
        st.job(n).pop("rung_reset", None)
    if resets:
        st.save()
        log(f"prepare: reset rung jobs {resets} run again on this VM")
    up, events = [], set()
    for pt, s in need:
        on_vm, vm_step, has_ev = vm[f"{pt}/{s}"]
        if on_vm:
            log(f"{pt}: rung_{s} snapshot already on this VM (volume); nothing uploaded")
            continue
        resume = max(local_step(pt), vm_step or 0)
        if not st.job(pt)["home"] and resume <= s:
            continue  # the job resumes at or before the step and snapshots it again on this VM
        for f in RUNG_FILES:
            up.append((f"{pt}__rung_{s}__{f}", os.path.join(snap_home(pt, s), f), f"{snap_vm(pt, s)}/{f}"))
        rec = st.d.setdefault("rung_snapshots", {})
        if f"{pt}/rung_{s}" not in rec:  # a home copy not fetched by this runner: never downloaded back
            rec[f"{pt}/rung_{s}"] = {"bytes": os.path.getsize(os.path.join(snap_home(pt, s), "model.safetensors")),
                                     "fetched": None}
            st.save()
        ev = os.path.join(ART, pt, "events.jsonl")
        if not has_ev and pt not in events and os.path.exists(ev):
            events.add(pt)
            up.append((f"{pt}__events.jsonl", ev, f"/content/chat/{pt}/events.jsonl"))
    return up


def lost_rungs(d, jobs):
    """Rung jobs in state dict d that are home, FAILED, and whose home log says "rung lost":
    [(name, training job, step, snapshot at home)]."""
    out = []
    for j in jobs:
        ro, n = j.get("rung_of"), j["name"]
        js = d["jobs"].get(n) or {}
        if not ro or not js.get("home") or js.get("status") != "FAILED":
            continue
        lp = os.path.join(ART, n, f"{n}_train.log")
        try:
            lost = "rung lost" in open(lp, errors="replace").read()
        except OSError:
            lost = False
        if lost:
            out.append((n, ro["job"], int(ro["step"]), snap_at_home(ro["job"], int(ro["step"]))))
    return out


RESET_REQUEST = "rung_reset.json"


def apply_rung_resets(st, jobs):
    """At startup (no other runner is live): the rung jobs named in HOME/rung_reset.json
    (reset_lost_rungs.py) that failed "rung lost" and whose snapshot is now home go back to
    WAITING (a fresh state entry, marked rung_reset). No other job is touched. The request file is
    removed once read; a named job that does not qualify is logged and left as it is."""
    p = os.path.join(HOME, RESET_REQUEST)
    if not os.path.exists(p):
        return
    try:
        want = list(json.load(open(p)).get("rungs", []))
    except (OSError, ValueError) as e:
        log(f"rung reset request unreadable ({e}); ignored")
        os.remove(p)
        return
    ok = {n for n, _, _, home in lost_rungs(st.d, jobs) if home}
    done = []
    for n in want:
        if n not in ok:
            log(f"{n}: rung reset refused (not a home FAILED 'rung lost' job with its snapshot at home)")
            continue
        lp = os.path.join(ART, n, f"{n}_train.log")
        if os.path.exists(lp):  # the next run's log comes home under this name
            os.replace(lp, os.path.join(ART, n, f"{n}_train.lost.log"))
        st.d["jobs"][n] = {"relayed_step": 0, "status": None, "home": False, "rung_reset": time.time()}
        done.append(n)
    if done:
        st.d.pop("campaign_home", None)  # a RunPod release would otherwise delete the volume at the end
        st.save()
        log(f"rung jobs reset to WAITING (their snapshots go up at the next prepare): {done}")
    os.remove(p)


def launch(st, commit, jobs_rel, jobs, uploads):
    T.stop(st)
    rc, out = T.create(st)
    if rc != 0:
        log(f"no {T.gpu} ({out.strip()[-120:]}); retry later")
        return False
    st.d["endpoint"] = listed()
    st.d["launches"] += 1
    st.d["booted"] = False
    st.save()
    log(f"session up, endpoint {st.d['endpoint']} (launch {st.d['launches']})")
    return prepare(st, commit, jobs_rel, jobs, uploads)


def upload_checkpoint(st, name):
    """One job's resume checkpoint (model.safetensors in parts, state.txt, events.jsonl), then its
    READY marker; raises RuntimeError on a failed upload (resumed by part on the next try)."""
    art = os.path.join(ART, name)
    upload_file(os.path.join(art, "model.safetensors"), f"/content/chat/upload/{name}/model.safetensors")
    for f in ("state.txt", "events.jsonl"):
        rc, _ = T.put(os.path.join(art, f), f"/content/chat/upload/{name}/{f}", 600)
        if rc != 0:
            raise RuntimeError(f"upload of {name}/{f} failed")
    rc, out = exec_py(f"open('/content/chat/upload/{name}/READY', 'w').close()\nprint('READY_OK')\n")
    if "READY_OK" not in out:
        raise RuntimeError(f"{name}: READY marker not written")
    st.d["pending_uploads"] = [n for n in st.d.get("pending_uploads", []) if n != name]
    st.save()
    log(f"{name}: checkpoint uploaded for resume (weights only; the optimiser resumes cold)")


def upload_pending(st):
    """After the boot: the next pending resume checkpoint, one job per tick."""
    pend = st.d.get("pending_uploads") or []
    if not pend:
        return
    try:
        upload_checkpoint(st, pend[0])
    except RuntimeError as e:
        log(str(e))


def local_step(name):
    p = os.path.join(ART, name, "state.txt")
    try:
        return int(open(p).readline().strip() or 0)
    except (OSError, ValueError):
        return 0


def volume_has(pend, finals):
    """A persistent VM disk (RunPod's network volume): the pending jobs whose own checkpoint on the
    volume is at or past the relayed one (they resume from it, warm, and need no upload) and the
    home parents' final weights already in place (same sha256). None when the VM cannot be read."""
    shas = {}
    for _, src, dst in finals:
        h = hashlib.sha256()
        with open(src, "rb") as f:
            for b in iter(lambda: f.read(1 << 22), b""):
                h.update(b)
        shas[dst] = h.hexdigest()
    rc, out = exec_py(
        "import hashlib, json, os\n"
        f"pend, finals = {pend!r}, {sorted(shas)!r}\n"
        "steps, have = {}, {}\n"
        "for n in pend:\n"
        "    d = '/content/chat/' + n\n"
        "    try:\n"
        "        s = int(open(d + '/state.txt').readline().strip() or 0)\n"
        "    except (OSError, ValueError):\n"
        "        s = None\n"
        "    steps[n] = s if os.path.exists(d + '/model.safetensors') else None\n"
        "for p in finals:\n"
        "    if os.path.exists(p):\n"
        "        h = hashlib.sha256()\n"
        "        with open(p, 'rb') as f:\n"
        "            for b in iter(lambda: f.read(1 << 22), b''):\n"
        "                h.update(b)\n"
        "        have[p] = h.hexdigest()\n"
        "print('VOLHAS', json.dumps({'steps': steps, 'finals': have}))\n", timeout=600)
    m = re.search(r"^VOLHAS (.*)$", out, re.M)
    if not m:
        return None
    d = json.loads(m.group(1))
    on_volume = {n for n in pend if d["steps"].get(n) is not None and d["steps"][n] >= local_step(n)}
    for n in sorted(on_volume):
        log(f"{n}: the volume holds its checkpoint at step {d['steps'][n]} (relayed {local_step(n)}); "
            "it resumes from the volume, nothing uploaded")
    placed = {dst for dst, sha in shas.items() if d["finals"].get(dst) == sha}
    return on_volume, placed


def job_needs(j):
    """The jobs j waits for or starts from: its "after" (a name or a list) and its "init_job"."""
    after = j.get("after") or []
    after = [after] if isinstance(after, str) else list(after)
    return after + ([j["init_job"]] if j.get("init_job") else [])


def pick_inputs(j, by):
    """The post jobs of a `tiny_chat_run.py pick` job (its --run /content/chat/<post> dirs, each a data job it
    waits for), or None for any other job."""
    cmd = j.get("cmd") or []
    if not any(a.endswith("tiny_chat_run.py") and cmd[i + 1:i + 2] == ["pick"] for i, a in enumerate(cmd)):
        return None
    runs = [cmd[i + 1] for i, a in enumerate(cmd[:-1]) if a == "--run"]
    posts = [r[len("/content/chat/"):] for r in runs if r.startswith("/content/chat/")]
    if (not runs or len(posts) != len(runs) or len(set(posts)) != len(posts)
            or any(p not in by or "cmd" not in by[p] for p in posts) or not set(posts) <= set(job_needs(j))):
        return None
    return posts


def spared_posts(st, jobs, by, needed, failed):
    """prepare() on a replacement VM (25 Oct 2026: E5's replacement reran six home post jobs, ~1.4 L4 hours
    each, before its pretraining). A post job's files live only on the VM that ran it; its pick reads, of
    every run, summary.json's seed and primary, and only of the best run its SFT model. So a home post job
    that its unfinished pick can never choose is not rerun: tiny_chat_run.pick_best (pick's own rule) over
    the home summaries of the pick's home posts names the one post that can still be chosen; it reruns, and
    each other home post gets its home summary.json at the VM path pick reads and is marked FINISHED. A
    post not yet home runs anyway, so a pick never reruns more than one home post, and whatever pick
    chooses on the VM has its SFT model there. (Deferring the rerun until every post is home would need a
    hold state jobs_vm.py at the campaigns' pinned commits does not have: pick starts within a manager tick
    of its last post.) A post is spared only when every job that waits for it is an unfinished pick that
    cannot choose it, or a home job nothing unfinished needs; a pick with a FAILED input fails on the VM
    whatever happens, so none of its home inputs reruns. Returns {post: home summary.json}."""
    picks = {j["name"]: pick_inputs(j, by) for j in jobs if "cmd" in j and not st.job(j["name"])["home"]}
    picks = {k: v for k, v in picks.items() if v}
    if not picks:
        return {}
    import tiny_chat_run  # the rule pick runs on the VM (unchanged since 24151f80, so the same at every pinned commit)
    verdict = {}  # post -> [can never be chosen, per pick]
    for pk, posts in picks.items():
        rows = {p: tiny_chat_run.pick_row(os.path.join(ART, p)) for p in posts
                if st.job(p)["home"] and p not in failed}
        rows = {p: r for p, r in rows.items() if r["summary"]}
        if any(p in failed for p in posts) and not by[pk].get("after_terminal"):
            for p in rows:  # the pick fails on the VM: nothing of it is chosen
                verdict.setdefault(p, []).append(True)
            continue
        best = tiny_chat_run.pick_best(list(rows.values()))
        for p, r in rows.items():
            verdict.setdefault(p, []).append(r is not best)
    spare = {}
    for p, v in verdict.items():
        deps = [j["name"] for j in jobs if p in job_needs(j)]
        if p in needed and all(v) and all(
                (d in picks and p in picks[d]) or (st.job(d)["home"] and d not in needed) for d in deps):
            spare[p] = os.path.join(ART, p, "summary.json")
    return spare


def prepare(st, commit, jobs_rel, jobs, uploads):
    """Inputs, relayed checkpoints and the boot onto a session that is up but
    not yet booted. Retried every tick until it succeeds (st.d["booted"]), so
    a failed upload never leaves a provisioned VM idle."""
    adopt_home(st, jobs)  # files home but state unmarked: FINISHED below, not run again
    dirs = ["/content/chat/upload"] + [f"/content/chat/upload/{j['name']}" for j in jobs]
    rc, out = exec_py(f"import os\nfor d in {dirs!r}: os.makedirs(d, exist_ok=True)\nprint('DIRS_OK')\n")
    if "DIRS_OK" not in out:
        log(f"prepare: upload dirs not created: {out.strip()[-120:]}")
        return False
    # Jobs already home keep their outcome on this VM: jobs_vm.py runs every job not FINISHED or
    # FAILED, so a fresh VM would otherwise redo them (22 Oct 2026: lever1 rebuilt after it was home).
    # A home job that FAILED stays FAILED, so its dependants still follow it (23 Oct 2026: marking
    # it FINISHED let a failed gate's dependants run). A home data job ("cmd") that an unfinished
    # job still needs, directly or through others, runs again: its files live on the VM that made
    # them, and a replacement VM has none (an unfinished pretraining job would find no corpus).
    # (On a persistent volume the data job's STATUS and files are still there, so it is not rerun.)
    by = {j["name"]: j for j in jobs}

    def closure(stop=()):
        needed, todo = set(), [n for j in jobs if not st.job(j["name"])["home"] for n in job_needs(j)]
        while todo:
            n = todo.pop()
            if n in by and n not in needed:
                needed.add(n)
                # Followed through home training jobs too: a job that runs after one often reads that
                # job's own inputs without declaring them (24 Oct 2026: the pilot's post job read
                # data/panel_fineweb.ids on a replacement VM, after only its pretraining job), so every
                # data job upstream of unfinished work runs again. Training jobs never rerun. A spared
                # post job (below) never runs here, so nothing is needed for it.
                if n not in stop:
                    todo.extend(job_needs(by[n]))
        return needed

    needed = closure()
    failed = [j["name"] for j in jobs if st.job(j["name"])["home"] and st.job(j["name"]).get("status") == "FAILED"]
    # Home post jobs their unfinished pick can never choose: their summary.json goes up instead of a rerun.
    try:
        spare = spared_posts(st, jobs, by, needed, failed)
    except (OSError, ValueError, KeyError, TypeError, ImportError) as e:  # never let it stall a launch
        log(f"prepare: post jobs not checked against their picks ({e!r}); home posts rerun")
        spare = {}
    if spare:
        needed = closure(spare)
    rerun = [j["name"] for j in jobs if st.job(j["name"])["home"] and "cmd" in j and j["name"] in needed
             and j["name"] not in failed and j["name"] not in spare]
    finished = [j["name"] for j in jobs if st.job(j["name"])["home"] and j["name"] not in failed + rerun]
    rc, out = exec_py(f"for n in {finished!r}: open('/content/chat/STATUS_' + n, 'w').write('FINISHED\\n')\n"
                      f"for n in {failed!r}: open('/content/chat/STATUS_' + n, 'w').write('FAILED\\n')\n"
                      "print('HOME_OK')\n")
    if "HOME_OK" not in out:
        log(f"prepare: home jobs not marked: {out.strip()[-120:]}")
        return False
    if rerun:
        log(f"prepare: data jobs rerun on this VM for unfinished dependants: {rerun}")
    if spare:
        log(f"prepare: post jobs their pick can never choose are not rerun (FINISHED, summary.json from home): "
            f"{sorted(spare)}")
    spare_up = [(f"{p}__summary.json", src, f"/content/chat/{p}/summary.json") for p, src in sorted(spare.items())]
    # An unfinished job that starts from, or runs after, a home training job needs that job's final
    # weights on the fresh VM: uploaded as a whole-file input and moved into its out_dir below.
    finals = []
    for n in sorted(needed):
        parent = by.get(n)
        if parent and "cmd" not in parent and st.job(n)["home"] and n not in failed:
            model = parent.get("model_name", n)
            src = os.path.join(ART, n, model + ".safetensors")
            if os.path.exists(src):
                finals.append((f"{n}__final.safetensors", src, f"/content/chat/{n}/{model}.safetensors"))
    # Resume checkpoints of unfinished jobs, smallest first (see below).
    pend = sorted((j["name"] for j in jobs if not st.job(j["name"])["home"]
                   and os.path.exists(os.path.join(ART, j["name"], "state.txt"))),
                  key=lambda n: os.path.getsize(os.path.join(ART, n, "model.safetensors")))
    if T.persistent:  # what the volume already holds is not uploaded again
        has = volume_has(pend, finals)
        if has is None:
            log("prepare: the volume could not be read; will retry")
            return False
        pend = [n for n in pend if n not in has[0]]
        finals = [f for f in finals if f[2] not in has[1]]
    # Rung snapshots that unfinished rung jobs need, from their home copies (before the boot: jobs_vm.py at the
    # campaigns' pinned commits fails a rung job whose snapshot is missing as soon as its "after" jobs finish).
    try:
        rung_up = rung_inputs(st, jobs)
    except (OSError, ValueError, KeyError, TypeError) as e:  # never let the rung backstop stall a launch
        log(f"prepare: rung snapshots skipped ({e!r})")
        rung_up = []
    if rung_up is None:
        log("prepare: rung snapshot check got no reply; will retry")
        return False
    try:
        for key, src in uploads:
            upload_file(src, f"/content/chat/upload/{key}")
        for key, src, _ in finals:
            upload_file(src, f"/content/chat/upload/{key}")
            log(f"{key}: uploaded for a dependent job")
        if rung_up:
            t0 = time.time()
            for key, src, _ in rung_up:
                upload_file(src, f"/content/chat/upload/{key}")
            snaps = sorted({dst.rsplit("/", 1)[0][len("/content/chat/"):] for _, _, dst in rung_up
                            if "/rung_" in dst})
            mb = sum(os.path.getsize(src) for _, src, _ in rung_up) / 1e6
            log(f"prepare: {len(snaps)} rung snapshots uploaded from home ({mb:.0f} MB, "
                f"{(time.time() - t0) / 60:.1f} min): {snaps}")
        for key, src, _ in spare_up:
            upload_file(src, f"/content/chat/upload/{key}")
        # The first resume checkpoint goes up before the boot;
        # the rest one per tick while the VM trains (upload_pending), and jobs_vm.py holds each job
        # until its READY marker. Weights, step and events only: no optimiser state (22 Oct 2026,
        # VMs living 40-60 min against a 50-min 1.9 GB upload); jobs_vm.py resumes those cold.
        rc, out = exec_py(f"for n in {pend!r}: open('/content/chat/upload/' + n + '/PENDING', 'w').close()\n"
                          "print('PENDING_OK')\n")
        if "PENDING_OK" not in out:
            log(f"prepare: pending markers not written: {out.strip()[-120:]}")
            return False
        st.d["pending_uploads"] = pend
        st.save()
        if pend:
            upload_checkpoint(st, pend[0])
    except RuntimeError as e:
        log(str(e))
        return False
    # Reassemble the whole-file inputs; a home parent's final weights then move into its out_dir.
    names = [k for k, _ in uploads] + [k for k, _, _ in finals] + [k for k, _, _ in rung_up + spare_up]
    moves = [(k, dst) for k, _, dst in finals] + [(k, dst) for k, _, dst in rung_up + spare_up]
    rc, out = exec_py(
        "import os, shutil\n"
        f"for k in {names!r}:\n"
        "    d = '/content/chat/upload'\n"
        "    parts = sorted(p for p in os.listdir(d) if p.startswith(k + '.part'))\n"
        "    if not parts: continue  # reassembled on an earlier try\n"
        "    with open(os.path.join(d, k), 'wb') as dst:\n"
        "        for p in parts:\n"
        "            shutil.copyfileobj(open(os.path.join(d, p), 'rb'), dst)\n"
        "    for p in parts: os.remove(os.path.join(d, p))\n"
        "    print(k, os.path.getsize(os.path.join(d, k)))\n"
        f"for k, dst in {moves!r}:\n"
        "    src = os.path.join('/content/chat/upload', k)\n"
        "    if os.path.exists(src):  # moved on an earlier try otherwise\n"
        "        os.makedirs(os.path.dirname(dst), exist_ok=True)\n"
        "        os.replace(src, dst)\n"
        "print('INPUTS_OK')\n")
    if "INPUTS_OK" not in out:
        log(f"input reassembly failed: {out.strip()[-160:]}")
        return False
    log("inputs: " + " ".join(ln for ln in out.splitlines() if ln and not ln.startswith("[")))
    rc, out = exec_py(boot_launch_code(commit, jobs_rel, BOOT_SCRIPT))
    if "BOOTED" not in out:
        log(f"boot exec failed: {out.strip()[-160:]}")
        return False
    st.d["booted"] = True
    st.save()
    log("boot launched")
    return True


def boot_launch_code(commit, jobs_rel, boot_script="tools/colab_chat/boot_jobs.sh"):
    """Pinned boot selection; the legacy training boot remains the default."""
    if boot_script not in ("tools/colab_chat/boot_jobs.sh", "tools/colab_chat/boot_commands.sh"):
        raise ValueError("unsupported boot script")
    url = f"https://raw.githubusercontent.com/JohnnyTeutonic/paperkiln/{commit}/{boot_script}"
    return (
        "import subprocess, urllib.request\n"
        f"open('/content/boot_jobs.sh','wb').write(urllib.request.urlopen('{url}').read())\n"
        f"subprocess.Popen('nohup bash /content/boot_jobs.sh {commit} {jobs_rel} > /dev/null 2>&1 &', shell=True)\n"
        "print('BOOTED')\n")


def vm_status(jobs):
    names = [j["name"] for j in jobs]
    rc, out = exec_py(
        "import os, json\n"
        "def rd(p, n=0):\n"
        "    return (open(p).read()[-n:] if n else open(p).read()).strip() if os.path.exists(p) else None\n"
        "r = {'vm': rd('/content/chat/STATUS') or 'BOOTING'}\n"
        f"for n in {names!r}:\n"
        "    r[n] = rd(f'/content/chat/STATUS_{n}') or 'NONE'\n"
        "    t = rd(f'/content/chat/{n}_train.log', 300) or ''\n"
        "    r[n + '_tail'] = t.replace(chr(10), ' | ')[-160:]\n"
        "print('VMJSON', json.dumps(r))\n")
    m = re.search(r"VMJSON (\{.*\})", out)
    return json.loads(m.group(1)) if m else None


def make_transport(a):
    if a.backend == "colab":
        return ColabTransport(a.gpu)
    from runpod_transport import RunPodTransport  # tools/colab_chat/runpod_transport.py
    return RunPodTransport(gpu=a.gpu, cloud=a.cloud, volume_gb=a.volume_gb, datacenter=a.datacenter,
                           image=a.image, max_usd=a.max_usd, commit=a.commit, name=NAME, home=HOME, log=log)


def main():
    global NAME, ART, HOME, T, BOOT_SCRIPT
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--commit", required=True)
    ap.add_argument("--jobs", required=True, help="jobs JSON, path relative to the paperkiln repo root")
    ap.add_argument("--boot-script", default="tools/colab_chat/boot_jobs.sh",
                    choices=["tools/colab_chat/boot_jobs.sh", "tools/colab_chat/boot_commands.sh"],
                    help="pinned VM boot; command-only boot skips trainer and corpus setup")
    ap.add_argument("--max-hours", type=float, default=23.5)
    ap.add_argument("--name", required=True, help="Colab session name (this run's own)")
    ap.add_argument("--art", required=True, help="local artifact directory")
    ap.add_argument("--state-dir", required=True, help="runner state, pid file and log")
    ap.add_argument("--upload", action="append", default=[], help="KEY=LOCAL_PATH, sent to /content/chat/upload/KEY")
    ap.add_argument("--backend", choices=["colab", "runpod"], default="colab")
    ap.add_argument("--gpu", default="L4", help="GPU type (both backends; L4 is the study GPU)")
    ap.add_argument("--cloud", choices=["SECURE", "COMMUNITY"], default="SECURE", help="RunPod cloud")
    ap.add_argument("--volume-gb", type=int, default=50, help="RunPod network volume size")
    ap.add_argument("--datacenter", default=None, help="RunPod data centre (default: one with the GPU in stock)")
    ap.add_argument("--image", default=None, help="RunPod image (default: runpod_transport.IMAGE)")
    ap.add_argument("--max-usd", type=float, default=None,
                    help="RunPod: stop and create nothing once this runner's ledger passes this many dollars")
    a = ap.parse_args()
    BOOT_SCRIPT = a.boot_script
    NAME, ART, HOME = a.name, a.art, os.path.expanduser(a.state_dir)
    os.makedirs(HOME, exist_ok=True)
    os.makedirs(ART, exist_ok=True)
    repo = os.path.dirname(os.path.dirname(HERE))
    jobs = json.load(open(os.path.join(repo, a.jobs)))["jobs"]
    uploads = [tuple(u.split("=", 1)) for u in a.upload]
    for k, src in uploads:
        if not os.path.exists(src):
            sys.exit(f"--upload {k}: {src} does not exist")
    pid = os.path.join(HOME, "runner.pid")
    if os.path.exists(pid):
        try:
            os.kill(int(open(pid).read()), 0)
            sys.exit("another runner is live (pid file); refusing")
        except (OSError, ValueError):
            pass
    st = State()
    adopt_home(st, jobs)
    has_rungs = any(j.get("rung_of") for j in jobs)
    if has_rungs:
        try:
            apply_rung_resets(st, jobs)  # reset_lost_rungs.py's request; safe here: no other runner is live
        except (OSError, ValueError, KeyError, TypeError) as e:
            log(f"rung reset request not applied ({e!r})")
    # One GPU type per campaign (a claim-carrying arm never mixes GPU types; AGENTS.md): recorded on
    # the first run under the new flag, refused when a later run asks for another.
    if st.d.get("gpu") not in (None, a.gpu):
        sys.exit(f"this state dir ran on {st.d['gpu']}; refusing --gpu {a.gpu} (one GPU type per campaign)")
    # A paid pod per watchdog restart must not loop on a boot that cannot succeed: a RunPod runner whose boot
    # failed at this commit stays down until the commit changes (or boot_failed_commit is removed from state.json).
    if a.backend != "colab" and st.d.get("boot_failed_commit") == a.commit:
        sys.exit(f"the boot failed at {a.commit} (ART/boot.log); refusing to rent another pod for it")
    # Nor for a campaign that is home (a restart after the end would otherwise launch a pod and find nothing to do).
    if a.backend != "colab" and all(st.d["jobs"].get(j["name"], {}).get("home") for j in jobs):
        sys.exit("every job is home; nothing to run (no pod rented)")
    T = make_transport(a)
    open(pid, "w").write(str(os.getpid()))
    if a.backend != "colab":
        st.d["gpu"] = a.gpu
        st.d["backend"] = a.backend
        st.save()
        T.attach(st)  # registers the runner for runpod_sweep.py; atexit and signal handlers end the pod

    deadline = time.time() + a.max_hours * 3600
    T.deadline = deadline  # transport must enforce it while a transfer blocks the main loop
    if listed() is None and not st.d.get("endpoint"):
        if not launch(st, a.commit, a.jobs, jobs, uploads):
            if time.time() < deadline and not T.over_budget(st):
                time.sleep(min(TICK, max(0, deadline - time.time())))
    try:
        while time.time() < deadline:
            why = T.over_budget(st)
            if why:
                log(why)
                break
            T.refresh(st)
            if listed() is None:
                ep = st.d.get("endpoint")
                if not T.adopt(ep):
                    # The listing comes back empty now and then while the VM is
                    # fine, and a relaunch begins with `colab stop`: on 20 Oct
                    # one empty listing destroyed a live VM's unrelayed work.
                    # Relaunch only after three straight ticks with no listing,
                    # no adoption and no answer from the VM itself. (RunPod: the API saying the pod
                    # no longer exists is proof, and the relaunch comes at once.)
                    probe = vm_status(jobs)
                    st.d["gone_ticks"] = 0 if probe is not None else st.d.get("gone_ticks", 0) + 1
                    if probe is None and T.gone(st):
                        st.d["gone_ticks"] = max(st.d["gone_ticks"], 3)
                    st.save()
                    if probe is None and st.d["gone_ticks"] >= 3:
                        log("session gone on 3 consecutive ticks; relaunching from the relayed checkpoints")
                        st.d["gone_ticks"] = 0
                        if not launch(st, a.commit, a.jobs, jobs, uploads):
                            time.sleep(TICK)
                            continue
                    elif probe is None:
                        log(f"session not listed and VM silent ({st.d['gone_ticks']}/3); waiting")
                        time.sleep(TICK)
                        continue
            else:
                st.d["gone_ticks"] = 0
            if not st.d.get("booted", True):
                log("session up but not booted; preparing it")
                if not prepare(st, a.commit, a.jobs, jobs, uploads):
                    time.sleep(TICK)
                    continue
            upload_pending(st)
            r = vm_status(jobs)
            if r is None:
                log("vm status: no reply")
                time.sleep(TICK)
                continue
            if r["vm"] == "BOOT_FAILED":
                log("boot failed on the VM; downloading boot.log")
                T.get("/content/chat/boot.log", os.path.join(ART, "boot.log"), 120)
                st.d["boot_failed_commit"] = a.commit
                st.save()
                break
            done = True
            relayed = False
            for j in jobs:
                n = j["name"]
                js = st.job(n)
                if r[n] != js["status"]:
                    log(f"{n}: {r[n]}  {r.get(n + '_tail', '')[-120:]}")
                    js["status"] = r[n]
                    st.save()
                if r[n] == "RUNNING" and not js["home"] and "cmd" not in j:
                    relayed = relay(st, n, j.get("relay_every_steps", 0)) or relayed
                if r[n] == "FINISHED" and not js["home"]:
                    if "cmd" in j:
                        fetch_outputs(st, n, j.get("outputs", []))
                    else:
                        fetch_final(st, n, j.get("model_name", n))
                # FAILED counts only when seen on two ticks running: a status
                # read mid-reset (a job being restarted by hand) is not final.
                js["failed_seen"] = js.get("failed_seen", 0) + 1 if r[n] == "FAILED" else 0
                if r[n] == "FAILED" and js["failed_seen"] >= 2 and not js["home"]:
                    if "cmd" in j:  # a failed data job's outputs are its diagnosis: whatever exists
                        fetch_outputs(st, n, j.get("outputs", []), best_effort=True)
                    T.get(f"/content/chat/{n}_train.log", os.path.join(ART, n, f"{n}_train.log"), 300)
                    js["home"] = True
                    st.save()
                done = done and js["home"]
            if has_rungs:
                try:
                    fetch_rung_snapshot(st, jobs, r, relayed)
                except (OSError, ValueError, KeyError, TypeError) as e:  # a backstop must never end a driver
                    log(f"rung snapshots: {e!r}")
            if done:
                log("all jobs finished and home")
                st.d["campaign_home"] = True
                st.save()
                break
            time.sleep(TICK)
    finally:
        T.stop(st)
        if st.d.get("campaign_home"):
            T.release(st)
        log("session stopped; runner exiting")
        os.remove(pid)


if __name__ == "__main__":
    main()
