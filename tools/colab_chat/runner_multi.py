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

State that must survive a failure (7 Nov 2026, the stronger base). A job with "halt_on_failure" is never counted
home when it fails: the runner writes HOME/HALTED (the watchdog's stop_file), ends the pod, and leaves the volume and every
checkpoint on it, so a failed save or resume cannot become HOME and release() the volume. "halt_drill" and
"pod_loss_drill_step" exercise that path and the replacement path on a real pod (tools/colab_chat/tc_b1_s0.json).

Picks (25 Oct 2026). A tiny_chat_run.py pick reads every post job's summary.json and only the best run's
SFT model, and a post job's files live only on the VM that ran it, so a replacement VM reran every home
post an unfinished pick waits for (E5: six, ~1.4 L4 hours each, ahead of its pretraining). prepare() now
reruns, per pick, only the home post that pick's own rule (tiny_chat_run.pick_best over the home
summaries) can still choose; every other home post gets its home summary.json at the path pick reads and
is marked FINISHED without running (spared_posts). Posts not yet home run anyway. A campaign without pick
jobs makes exactly the calls it made before (tests/runner_multi_equiv: run_eq.py, run_picks.py).

Checkpoint store (11 Nov 2026, base 2). --ckpt-store hf with --store-repo, --store-token-file and --boot-script
tools/colab_chat/boot_base2.sh: a training job's generations (weights, optimiser, state) go from the VM to a
private Hugging Face repo and come back on the next VM (hf_store.py, ckpt_courier.py), so no checkpoint crosses
the home link and a resume keeps its optimiser. See "checkpoint store" below. A job's branch is its
"store_branch", else ckpt-<name>; the jobs file's "store_inputs" name what the boot pulls. The very first start of
a campaign needs --store-first-start; after that a branch that lists less than home has seen pushed halts the
runner. No session is created while the store cannot be read from home, or after two failed boots at one commit.
A finished training job is home only when its final generation and marks are in the store and the marks at home.
With the store, --commit is the full SHA. Before every stop the runner makes, the session's own record comes home
to STATE/sessions/launchNN/ (courier.log, boot.log, courier_status.json, store_restore/*.json). A "pod_loss_drill_step"
writes STATE/drill_record.json: the trainer's training step and the courier's pushed step read from the VM just
before the stop, whether the stop fell between two pushes, and on which side of the tail switch. A courier that
reaches --store-push-cap-gb halts the campaign (STORE_PUSH_CAP). The hub limits commits per repository (128 an hour):
a save is one commit, each VM's commits pass a budget (--store-commit-budget-per-hour), and storage is released from
home only, on a timer and at the campaign's end (--store-release-storage, --store-release-every-min), never from a
VM. A state folder that has never launched refuses
branches that already hold generations (an earlier attempt's); --store-campaign gives a new attempt its own names.
A campaign without the option makes exactly the calls it made before (tests/test_runner_store.py).

Stops (12 Nov 2026). Every Colab stop's return code is read and a failed one tried again; a stop that is not
confirmed writes STATE/STOP_UNRESOLVED, naming the session, creates no session beside it, and the runner exits 4.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time

try:
    import fcntl  # the state folder's lock (POSIX; the runner runs in WSL)
except ImportError:  # pragma: no cover
    fcntl = None

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
        return colab("stop", "-s", NAME, timeout=120)[0]  # the return code: stop_session() reads it

    def absent(self):
        """True only when the listing itself answered and does not hold this session."""
        rc, out = colab("sessions", timeout=120)
        return rc == 0 and not re.search(r"^\[" + re.escape(NAME) + r"\]\s+(\S+)", out, re.M)

    def release(self, st):
        pass  # nothing outlives a Colab session

    def lose_pod(self, st):
        return self.stop(st)  # the drill: the session ends at the provider (the rehearsal's forced loss)

    def volume_present(self, st):
        return False

    def over_budget(self, st):
        return None


def exec_py(code, timeout=200):
    return T.exec(code, timeout)


STOP_TRIES = 3


def stop_session(st, why, lose=False):
    """Stop the session and know that it stopped. The Colab stop's return code is read; a failed stop is tried
    again, STOP_TRIES times in all; a stop that keeps failing counts as done only when the listing itself answers
    and no longer holds the session (there was nothing to stop). Otherwise the session may still be billing: a
    loud STOP_UNRESOLVED record naming it is written to the state folder and this returns False; the callers then
    make no new session and the runner exits non-zero. A stop that succeeds makes exactly the one call it made
    before. Other transports (RunPod) confirm their own stops and are called as before."""
    call = T.lose_pod if lose else T.stop
    if getattr(T, "backend", None) != "colab":
        call(st)
        return True
    rc = None
    for i in range(STOP_TRIES):
        rc = call(st)
        if rc is None or rc == 0:  # None: a stand-in transport that reports nothing, as before this check existed
            resolved = os.path.join(HOME, "STOP_UNRESOLVED")
            if os.path.exists(resolved):  # an earlier stop of this state folder was left open; this one closed it
                os.replace(resolved, os.path.join(HOME, "STOP_RESOLVED"))
            return True
        if T.absent():
            log(f"stop ({why}) returned {rc}; the listing answers and does not hold {NAME}: nothing to stop")
            return True
        log(f"stop ({why}) returned {rc} and {NAME} is not known to be gone (try {i + 1} of {STOP_TRIES})")
        time.sleep(min(10, TICK / 24))
    rec = {"kind": "STOP_UNRESOLVED", "session": NAME, "endpoint": st.d.get("endpoint"), "launch": st.d.get("launches"),
           "why": why, "return_code": rc, "tries": STOP_TRIES, "at": time.time(),
           "do": f"the session may still be running and billing: stop it by hand (colab stop -s {NAME}) and check the listing"}
    with open(os.path.join(HOME, "STOP_UNRESOLVED"), "w") as handle:
        json.dump(rec, handle, indent=1)
        handle.flush()
        os.fsync(handle.fileno())
    log(f"STOP_UNRESOLVED: the stop of session {NAME} ({why}) failed {STOP_TRIES} times (return code {rc}) and the session "
        f"is not known to be gone. IT MAY STILL BE BILLING. Stop it by hand: colab stop -s {NAME}")
    return False


def listed():
    return T.listed()


class State:
    def __init__(self):
        p = os.path.join(HOME, "state.json")
        self.d = json.load(open(p)) if os.path.exists(p) else {"endpoint": None, "launches": 0, "jobs": {}}

    def job(self, name):
        return self.d["jobs"].setdefault(name, {"relayed_step": 0, "status": None, "home": False})

    def save(self):
        tmp = os.path.join(HOME, "state.json.tmp")
        with open(tmp, "w") as f:
            json.dump(self.d, f, indent=1)
        os.replace(tmp, os.path.join(HOME, "state.json"))


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
        # A transactional checkpoint: one generation, named once. It outlives the next commit, so a save during the copy
        # cannot tear it; a plain checkpoint has the flat files and the step check.
        "live = os.path.join(src, 'ckpt_gen', 'live')\n"
        "gen = os.path.join(src, 'ckpt_gen', os.readlink(live)) if os.path.islink(live) else src\n"
        "def step():\n"
        "    p = os.path.join(gen, 'state.txt')\n"
        "    return int(open(p).readline().strip() or 0) if os.path.exists(p) else 0\n"
        "s0 = step()\n"
        # A snapshot not yet home is kept until it is (a slow download must not restart each tick).
        "sp = os.path.join(snap, 'state.txt')\n"
        "s1 = int(open(sp).readline().strip() or 0) if os.path.exists(sp) else 0\n"
        f"if s1 > {js['relayed_step']}:\n"
        "    print('SNAP', s1)\n"
        f"elif s0 > {js['relayed_step']} and s0 - {js['relayed_step']} >= {every}:\n"
        "    os.makedirs(snap + '.tmp', exist_ok=True)\n"
        "    for f in ('model.safetensors', 'optim.safetensors', 'state.txt'):\n"
        "        shutil.copyfile(os.path.join(gen, f), os.path.join(snap + '.tmp', f))\n"
        "    shutil.copyfile(os.path.join(src, 'events.jsonl'), os.path.join(snap + '.tmp', 'events.jsonl'))\n"
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


def export_formats(j):
    """The formats a training job's spec says it exports ("export": {"formats": [...]}), or None when the spec
    does not say (the trainer's default, and every campaign before base 2)."""
    f = ((j.get("spec") or {}).get("export") or {}).get("formats")
    return list(f) if isinstance(f, list) else None


def fetch_final(st, name, model_name, formats=None):
    """A finished training job's outputs. formats None (the spec names none): the files and the order of every
    campaign so far. With the spec's own list: the weights in the formats it declares and no other (a job that
    exports no gguf is never asked for one, 12 Nov 2026: such a job could not come home and its session billed to
    the deadline), and events.jsonl, result.json, the spec and the log each fetched whatever happened to the file
    before it."""
    art = os.path.join(ART, name)
    os.makedirs(art, exist_ok=True)
    ok = True
    if formats is None:
        for f in (f"{name}/{model_name}.safetensors", f"{name}/{model_name}.gguf", f"{name}/events.jsonl",
                  f"{name}/result.json", f"{name}_spec.json", f"{name}_train.log"):
            ok = ok and download_file(f"/content/chat/{f}", os.path.join(art, os.path.basename(f))) == 0
    else:
        need = [f"{name}/{model_name}.{ext}" for ext in ("safetensors", "gguf") if ext in formats]
        for f in (*need, f"{name}/events.jsonl", f"{name}/result.json", f"{name}_spec.json", f"{name}_train.log"):
            ok = (download_file(f"/content/chat/{f}", os.path.join(art, os.path.basename(f))) == 0) and ok
    if formats is None or "gguf" in formats:
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


def fetch_diagnosis(name, outputs):
    """Whatever a halting job left, before its pod ends: a failed job's outputs are its diagnosis. Not fetch_outputs: this job stays unhome, and
    nothing here may stop the halt."""
    for f in outputs:
        dst = os.path.join(ART, name, f)
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if download_file(f"/content/chat/{name}/{f}", dst, 300) == 0:
                log(f"{name}: {f} downloaded for the halt")
        except Exception as e:  # noqa: BLE001
            log(f"{name}: {f} not fetched ({e!r})")


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
    keep = {j["name"] for j in jobs if j.get("keep_home_snapshot")}  # read again at home after its rung job
    for key, e in rec.items():
        pt, step = key.rsplit("/rung_", 1)
        names = own.get((pt, int(step)), [])
        if not e.get("deleted") and names and not keep.intersection(names) and all(
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
        # The campaign is open again. A RunPod release would otherwise delete the volume at the end; and the release
        # at the campaign's end is owed again when it ends again, with a new allowance of rounds.
        for k in ("campaign_home", "store_release_final_done", "store_release_final_owed", "store_release_final_rounds",
                  "store_release_final_malformed"):
            st.d.pop(k, None)
        st.save()
        log(f"rung jobs reset to WAITING (their snapshots go up at the next prepare): {done}")
    os.remove(p)


def launch(st, commit, jobs_rel, jobs, uploads):
    if getattr(T, "deadline", None) is not None and time.time() >= T.deadline:
        log("deadline expired before launch; no session created")
        return False
    if STORE:
        if not store_preflight(st, jobs):
            return False
        store_forget_token("before an old session is stopped")
    if not stop_session(st, "before a new session is created"):
        raise SystemExit(4)  # an old session that may still be up: a second one is never created beside it
    if getattr(T, "deadline", None) is not None and time.time() >= T.deadline:
        log("deadline expired while stopping prior session; no replacement created")
        return False
    rc, out = T.create(st)
    if rc != 0:
        log(f"no {T.gpu} ({out.strip()[-120:]}); retry later")
        return False
    st.d["endpoint"] = listed()
    st.d["launches"] += 1
    st.d["booted"] = False
    if STORE:
        store_stall_reset(st, jobs)  # a new session: what the old one's loss and this one's boot take is no stall
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


def validate_checkpoint_families(jobs, boot_script):
    """Each named boot admits one opt-in contract, never mixed campaigns."""
    sfull = any("full_checkpoint_v2" in j for j in jobs)
    dpo = any("dpo_full_checkpoint_v1" in j for j in jobs)
    if sfull and dpo:
        raise ValueError("mixed S-full/DPO checkpoint campaigns are unsupported")
    expected = ("tools/colab_chat/boot_sfull_v2_20261101.sh" if sfull else
                "tools/colab_chat/boot_dpo_checkpoint_v1.sh" if dpo else None)
    if expected is not None and boot_script != expected:
        raise ValueError("checkpoint family requires its separately admitted named boot")


def halt_campaign(job, why):
    """A failed job whose state must survive: the marker the watchdog's stop_file reads. The job stays unhome, so
    campaign_home is never set and release() never runs; the caller ends the pod and exits. A person reads it first."""
    marker = os.path.join(HOME, "HALTED")
    if not os.path.exists(marker):
        with open(marker, "x") as handle:
            json.dump({"kind": "JOB_FAILED_STATE_KEPT", "job": job, "why": why, "at": time.time()}, handle)
            handle.flush()
            os.fsync(handle.fileno())
    log(f"{job}: {why}; HALTED written, pod ends, the volume and its checkpoints are kept")


def halt_drill(st, job):
    """Stage 0's last job: a real mtstudio refused a real torn checkpoint on the real volume and the VM reports FAILED.
    With the job unhome, release() must refuse and the volume must still exist. The record goes to HOME/halt_drill.json;
    the job then counts as home, so the drill's own end is the ordinary campaign end. Any other outcome stops the campaign.
    It proves FAILED, not home and a refused release, with the refusal read from the job's log; the HALTED branch itself runs
    only under the fake CLI (tests/test_tc_b1.py), because a real HALTED would leave the volume for a person to read."""
    T.stop(st)
    try:
        log_text = open(os.path.join(ART, job, f"{job}_train.log"), errors="replace").read()
    except OSError:
        log_text = ""
    refused, error = False, None
    try:
        T.release(st)
    except Exception as e:  # noqa: BLE001 runpod_api.Refused, or a provider failure, recorded either way
        refused, error = type(e).__name__ == "Refused", repr(e)
    rec = {"job": job, "status_seen": st.job(job).get("status"), "home_before": st.job(job)["home"],
           "refusal_in_log": "no whole generation" in log_text, "release_refused": refused, "error": error,
           "volume_present": T.volume_present(st), "at": time.time()}
    with open(os.path.join(HOME, "halt_drill.json"), "w") as handle:
        json.dump(rec, handle, indent=1)
    shutil.copyfile(os.path.join(HOME, "halt_drill.json"), os.path.join(ART, "halt_drill.json"))
    log(f"halt drill: {json.dumps(rec)}")
    if not (refused and rec["volume_present"] and rec["home_before"] is False and rec["refusal_in_log"]
            and rec["status_seen"] == "FAILED"):
        halt_campaign(job, "halt drill failed")
        raise SystemExit(3)


def pause_at_rung(j, st):
    """A job's "pause_rungs": once a named rung job is home, its read is at home or lost, and this arm stops there unless HOME/RELEASED_<rung>
    exists. The marker is HALTED (the watchdog's stop_file), the pod ends, and the volume and every checkpoint stay. tc_b2_stop.py
    release-rung lets it go on. Returns the rung, or None."""
    for rung in j.get("pause_rungs", []):
        if st.job(rung)["home"] and not os.path.exists(os.path.join(HOME, f"RELEASED_{rung}")):
            marker = os.path.join(HOME, "HALTED")
            if not os.path.exists(marker):
                with open(marker, "x") as handle:
                    json.dump({"kind": "CUTOFF_REACHED", "rung": rung, "job": j["name"], "at": time.time()}, handle)
                    handle.flush()
                    os.fsync(handle.fileno())
            log(f"{j['name']}: paused at {rung}: pod ends, volume and checkpoints stay")
            return rung
    return None


def gate_upload(j, st):
    """A job's "gate_upload": {"home", "vm"}: one small file sent to the VM once it exists at home, for a job holding on `promoted_by`."""
    g, js = j.get("gate_upload"), st.job(j["name"])
    if g and not js.get("gate_sent") and os.path.exists(g["home"]):
        try:
            json.load(open(g["home"]))
        except ValueError:
            return  # being written: the next tick
        if T.put(g["home"], g["vm"] + ".tmp", 300)[0] == 0:
            rc, out = exec_py(f"import os\nos.replace({g['vm'] + '.tmp'!r}, {g['vm']!r})\nprint('GATE_OK')\n")
            if "GATE_OK" in out:
                js["gate_sent"] = True
                st.save()
                log(f"{j['name']}: gate file sent")


# ---------------------------------------------------------------- checkpoint store (--ckpt-store hf)
# Base 2 (11 Nov 2026; AI_ML/chat_screens_claude/base2/BASE2_PLAN.md section 1.5). With a store, a training job's
# checkpoints go from the VM to a private Hugging Face repo (ckpt_courier.py) and come back on the next VM
# (boot_base2.sh), optimiser included. This runner then:
# - creates no session while the store cannot be read from home (store_preflight, with a growing wait), halts when a
#   branch lists less than home has seen pushed, and refuses to start again after two failed boots at one commit;
# - sends the store's config (repo, branches, campaign, the launch count as fencing epoch, each job's minimum step,
#   the jobs admitted to a first start) and the token file, and moves the token to a tmpfs with its next call;
# - uploads and relays no checkpoint of a training job; reads each job's pushed step from the store and says so,
#   loudly, when it has not advanced for --store-stall-min minutes (a halt with --store-stall-halt);
# - brings mark snapshots home from the store; counts a finished training job as home only when its final
#   generation and every expected mark are in the store and the marks are home ("durability pending" until then,
#   a halt after --store-settle-min minutes);
# - is the store's one releaser (store_release, with --store-release-storage): on a timer and at the campaign's end,
#   under the store's lock, it deletes the objects no branch names and measures what the store holds, which is the
#   storage guard's count from then on. A VM never releases: the hub limits commits per repository (128 an hour),
#   a save is one commit, and the lock's two commits are spent here, a few an hour, within home's own budget;
# - ends the courier and removes the token from the VM before every stop it makes.
# Every store call at home runs in a child with a wall-time limit. STORE None (no --ckpt-store): none of this runs
# and every call is as before (tests/test_runner_store.py).
STORE = None
STORE_BOOT = "tools/colab_chat/boot_base2.sh"
STORE_TOKEN_VM = "/content/chat/upload/store_token"  # on the VM's disk for the seconds until the next call moves it
STORE_CONFIG_VM = "/content/chat/store.json"
STORE_BOOT_FAILURES = 2
# The courier's final delivery window (--store-courier-wait-s; the same number is boot_base2.sh's default, and the
# boot reads the runner's from store.json). After its jobs end a VM's courier has this long to push the last
# generation and marks (a budget wait is at most 15 minutes and a 429's at most 30) before the boot takes the token
# away. --store-settle-min is never shorter than this and a tick.
STORE_COURIER_WAIT_S = 5400
# A finished run's last generation that reads "in the store and not whole" is read again before the campaign is
# halted on it: this many completed checks that differ, one a tick, each kept in state.json. One listing that is
# transiently wrong must not halt a finished four-day run; a generation that really is damaged reads the same every time.
STORE_FINAL_CHECKS = 3
# The release at the campaign's end. Its lock is an essential commit of home's budget: it may use the whole budget
# and is owed the budget's essential wait (15 minutes), after which it goes whatever the count is. When it cannot
# run at once it is tried again AFTER the session has been stopped (the wait bills nothing), every
# STORE_FINAL_RELEASE_NAP_S seconds for at most STORE_FINAL_RELEASE_WAIT_S; if it still has not run, one line says
# so and names the command that does it by hand.
STORE_FINAL_RELEASE_WAIT_S = 1200
STORE_FINAL_RELEASE_NAP_S = 30
STORE_FINAL_RELEASE_FAILED_NAP_S = 300  # after a round that failed (it may have sent its commits): not 30 s
STORE_HOME_ESSENTIAL_WAIT_S = 900  # hf_store.ESSENTIAL_WAIT_S, for home's budget
# A round of the end's release that ran and met a ledger it could not read: this many in all, counted in state.json
# across every start of this state folder ("store_release_final_rounds").
STORE_FINAL_LEDGER_ROUNDS = 2
# A TIMED release that did not run is not left for a full interval (the storage guard's count is only reset by a
# release that ran, so two missed turns in a row can carry a healthy arm past its push cap). One that was held (no
# commit was sent: another runner's lock, or no room in the budget) is tried again at the next tick; one that failed
# or met a ledger it could not read is tried again once, this many seconds later, and then at its regular turn.
STORE_RELEASE_RETRY_S = 600
_STORE = {}


def store_config(a, ap):
    import hf_store
    if not a.store_repo:
        ap.error("--ckpt-store hf needs --store-repo (there is no default store)")
    if hf_store.refused_repo(a.store_repo):
        ap.error(f"{a.store_repo} is the pinned inputs repo; the store must be its own repo")
    if not re.fullmatch(r"[0-9a-f]{40}", a.commit):
        ap.error("--ckpt-store hf takes --commit as the full 40-character SHA (a short one can come to name another "
                 "commit, and the boot and its failed-boot count are keyed by it)")
    local = a.store_repo.startswith("dir:")
    if not local and not a.store_token_file:
        ap.error("--ckpt-store hf needs --store-token-file (there is no default token path)")
    if a.store_token_file:
        try:
            hf_store.read_token(os.path.expanduser(a.store_token_file))
        except hf_store.StoreError as e:
            ap.error(str(e))
    if a.boot_script != STORE_BOOT:
        ap.error(f"--ckpt-store hf runs with --boot-script {STORE_BOOT}")
    doc = json.load(open(os.path.join(os.path.dirname(os.path.dirname(HERE)), a.jobs)))
    # An explicit --store-campaign is a namespace: it is appended to every job's branch and put before every job's
    # name in its marks' paths, so a new attempt under a new campaign name never meets an earlier attempt's
    # generations or marks. Without the flag the names are the jobs file's own.
    slug = re.sub(r"[^A-Za-z0-9]+", "-", a.store_campaign).strip("-") if a.store_campaign else ""
    if a.store_campaign and not slug:
        ap.error("--store-campaign needs a letter or a digit")
    cap_gb = a.store_push_cap_gb if a.store_push_cap_gb is not None else float(doc.get("store_push_cap_gb") or 0)
    every = a.store_release_every_min if a.store_release_every_min is not None \
        else float(doc.get("store_release_every_min") or 120)
    if every <= 0 or a.store_commit_budget_per_hour < 0 or a.store_home_commit_budget_per_hour < 0:
        ap.error("--store-release-every-min must be above 0, and the commit budgets 0 (no limit) or above")
    # The registration's switches may be the jobs file's keys (make_base2.arm() writes them): a flag wins.
    grace = a.store_release_grace_s if a.store_release_grace_s is not None else float(doc.get("store_release_grace_s") or 7200)
    stall_min = a.store_stall_min if a.store_stall_min is not None else float(doc.get("store_stall_min", 60))
    if grace < hf_store.MIN_RELEASE_GRACE_S:
        ap.error(f"--store-release-grace-s is at least {hf_store.MIN_RELEASE_GRACE_S}: a restore may be reading a "
                 "generation that was just removed")
    if a.store_courier_wait_s < 0 or a.store_settle_min * 60 < a.store_courier_wait_s + TICK:
        ap.error(f"--store-settle-min is at least {(a.store_courier_wait_s + TICK) / 60:.0f}: the VM's courier has "
                 f"{a.store_courier_wait_s:.0f} s after its jobs end to deliver the last generation "
                 "(--store-courier-wait-s), and the runner must not halt before that window and one tick are over")
    return {"repo": a.store_repo, "repo_type": a.store_repo_type, "reclaim": a.store_release_storage,
            "release_grace_s": grace, "release_every_min": every,
            "release_offset_min": a.store_release_offset_min if a.store_release_offset_min is not None
            else float(doc.get("store_release_offset_min") or 0),
            "commit_budget": a.store_commit_budget_per_hour, "home_commit_budget": a.store_home_commit_budget_per_hour,
            "courier_wait_s": a.store_courier_wait_s,
            "branch_suffix": "-" + slug if slug else "", "mark_ns": slug + "." if slug else "",
            "push_cap_bytes": int(cap_gb * 1e9),
            "min_push_interval_s": a.store_min_push_interval, "campaign": a.store_campaign or a.name,
            "first_start": a.store_first_start, "inputs": doc.get("store_inputs", []),
            "call_limit": a.store_call_limit_s, "stall_min": stall_min,
            "stall_halt": a.store_stall_halt or bool(doc.get("store_stall_halt")),
            "unverified_halt_min": a.store_unverified_halt_min if a.store_unverified_halt_min is not None
            else float(doc.get("store_unverified_halt_min") or 0),
            "settle_min": a.store_settle_min, "boot_limit_min": a.store_boot_limit_min,
            "token_file": os.path.expanduser(a.store_token_file) if a.store_token_file else None}


def store_branches(jobs):
    """{training job: its branch of the store}: the job's "store_branch", else ckpt-<name>, then the campaign's
    suffix when --store-campaign was given."""
    suffix = (STORE or {}).get("branch_suffix", "")
    return {j["name"]: j.get("store_branch", "ckpt-" + j["name"]) + suffix for j in jobs if "cmd" not in j}


def store_open(jobs):
    """The store, opened at home with the same token file. Reads (steps, manifests, mark snapshots) and, with
    --store-release-storage, the release: its lock and unlock are the only commits home makes, and they pass a
    commit budget of their own, counted in HOME/store_commits.json.

    What home commits, from the code: nothing but a release's two (hf_store.Store._lock and _unlock; the watcher,
    the marks' fetch, the preflight and the durability checks are reads, and main's branch is never created). A
    timed release's lock is counted with its removal (two) and refused within the budget's reserve (a quarter);
    an unlock that fails is sent once more, counted (hf_store.UNLOCK_TRIES). So a round is three commits at the
    worst, and one runner's worst rolling hour is a timed round and its one early retry, and the two rounds that
    are all the end's release has: 12 commits (make_base2.home_commits_rolling). The default budget of 12 holds
    that, and its nine outside the reserve hold four timed releases an hour (a release every 15 minutes or slower)."""
    if "store" not in _STORE:
        import hf_store
        cfg = {k: STORE.get(k) for k in ("repo", "repo_type", "token_file", "mark_ns")}
        cfg["store_commit_budget_per_hour"] = STORE.get("home_commit_budget") or 0
        cfg["commit_ledger"] = os.path.join(HOME, "store_commits.json")
        cfg["commit_essential_wait_s"] = STORE_HOME_ESSENTIAL_WAIT_S
        _STORE["store"] = hf_store.open_store(cfg, log=log, tries=2, delay=5)
    return _STORE["store"]


def store_err(e):
    """An error for the log or the state file: the token's text removed, and the query string of every address
    (after the hub's redirect an address is a signed one, and a transport error carries it)."""
    import hf_store
    tok = _STORE["store"].b.token if "store" in _STORE else None
    return hf_store.no_query(hf_store.redact(f"{type(e).__name__}: {e}", tok))[:200]


def store_auth(st, name, e):
    """The HUB refused home's token (HTTP 401 or 403 on a reply known to be the hub's: hf_store.auth_failure; a
    listing, a manifest, a commit, the preflight, a mark's fetch): said as that, by name, once until a store read
    from home succeeds again. A 401 or 403 from an object store or a CDN (a signed address that has expired, on a
    download after the hub's redirect) is said as that and nothing about the token: the next try gets a new
    address. The runner
    reads the token file once, when the store is opened, and each courier holds the token its session was given,
    so a new token takes: the file replaced at home (mode 600), this runner ended (its end stops the session with
    confirmation) and started again; the new session gets the new token and resumes from the store. With
    --store-stall-halt the arm is halted when nothing has reached the store for --store-stall-min minutes."""
    import hf_store
    if hf_store.http_status(e) in (401, 403) and hf_store.reply_side(e) == "object store":
        log(f"{name}: the refusal (HTTP {hf_store.http_status(e)}) is the object store's, not the hub's: the token is not "
            "the cause, and the next try is given a new address")
        return
    if not hf_store.auth_failure(e) or st.d.get("store_auth_failure"):
        return
    st.d["store_auth_failure"] = {"since": time.time(), "http_status": hf_store.http_status(e), "job": name}
    st.save()
    log(f"{name}: STORE AUTHENTICATION FAILURE at home: the hub refused the token (HTTP {hf_store.http_status(e)}). An "
        f"expired or revoked token reads so on every call, here and on the VM. To mend it: replace the token file "
        f"(mode 600), end this runner and start it again; the new session resumes from the store")


def store_clean_log(path):
    """A log that came home from a VM by itself (boot.log after a failed boot): the token's text and the query string
    of every address (a signed one, in the library's warning for a download it resumes) are removed in place. Never
    raises."""
    try:
        import hf_store
        tok = _STORE["store"].b.token if "store" in _STORE else None
        with open(path, "rb") as f:
            data = f.read()
        if tok:
            data = data.replace(tok.encode(), b"[token]")
        with open(path + ".tmp", "wb") as f:
            f.write(hf_store.no_query(data.decode("utf-8", "replace")).encode("utf-8"))
        os.replace(path + ".tmp", path)
    except OSError:
        pass


def store_stall_reset(st, jobs):
    """The stall clock of every store job starts again: called at every launch and when a session is prepared. The
    clock ("store_seen_at") is then set at the first tick the job is seen RUNNING on the new session, so the time
    a lost session took to be noticed, the new session's creation and its boot are never counted as a stall."""
    for n in store_branches(jobs):
        st.job(n).pop("store_seen_at", None)


def store_left_session(st):
    """At a start on a state folder that is home: what is up under this folder's name. ("ours", endpoint) for a
    session the listing shows under the name, unless this folder KNOWS its own was elsewhere: ("other", endpoint)
    only when the folder recorded an endpoint, the listed one differs from it, and no stop of this folder is
    unresolved. A folder that recorded no endpoint (its launch's listing came back empty and the runner died
    before the loop recorded one), or one with STOP_UNRESOLVED (it left a session up, by its own word), takes the
    listed session for its own: leaving one's own session up and billing is the worse mistake, and the names are
    one an arm by the registration. ("none", None) when the listing itself answers and holds none; ("unknown",
    None) when the listing did not answer, three times."""
    rec, unresolved = st.d.get("endpoint"), os.path.exists(os.path.join(HOME, "STOP_UNRESOLVED"))
    for i in range(3):
        ep = listed()
        if ep is not None:
            return ("ours" if rec is None or unresolved or ep == rec else "other"), ep
        if T.absent():
            return "none", None
        time.sleep(min(5, TICK / 48))
    return "unknown", None


def store_jobs_fingerprint(jobs):
    """The sha256 of the jobs as the runner uses them: every job's WHOLE entry, in the file's order (its name, spec,
    cmd, after, rung_of, snapshot_steps, store_branch, outputs and whatever else it carries), and the file's
    "store_inputs"."""
    return hashlib.sha256(json.dumps({"jobs": jobs, "store_inputs": (STORE or {}).get("inputs")}, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def store_owner(st, name, jobs):
    """A state folder belongs to one session name and one set of jobs, both recorded in state.json at its first
    start with the store ("store_name", "store_jobs_sha256"; the session's endpoint is recorded at each launch and
    whenever the loop's listing shows another).
    A start under another --name is refused: the name is what a stop, a token's removal and the release lock's
    takeover act on, and a finished folder started under a live arm's name would act on that arm.
    The jobs: EXACTLY WHEN a change is refused: once ANY job of the folder is home, a start whose jobs differ in
    anything the fingerprint covers (store_jobs_fingerprint: any key of any job, their order, the store's inputs)
    is refused, because the folder's `home` flags are of the jobs it ran and would skip work the new jobs need.
    Before the first job is home a changed jobs file is taken and recorded. The first job (the pins) is home
    within minutes, so AN ARM'S JOBS FILE IS FROZEN FOR ITS LIFE: an added job, a changed command or dependency,
    a changed spec all need a new --state-dir. A rung reset does not touch the jobs file and passes.
    Before any session, call or commit."""
    was = st.d.get("store_name")
    if was not in (None, name):
        sys.exit(f"this state folder belongs to the session name {was!r}; refusing --name {name!r} (a state folder keeps "
                 "one name: start the arm with its own --state-dir, or this folder with its own --name)")
    fp, had = store_jobs_fingerprint(jobs), st.d.get("store_jobs_sha256")
    if had not in (None, fp) and any(js.get("home") for js in st.d.get("jobs", {}).values()):
        sys.exit("this state folder ran other jobs (their names or specs differ from this jobs file's) and has jobs "
                 "marked home; refusing: its home flags would skip work these jobs need. Use a new --state-dir")
    if (was, had) != (name, fp):
        st.d["store_name"], st.d["store_jobs_sha256"] = name, fp
        st.save()


def store_call(fn, limit=None):
    """A store call from home, in a child that is ended at the limit: a hung transfer never holds the loop, the
    deadline check or the session's stop."""
    import hf_store
    tok = _STORE["store"].b.token if "store" in _STORE else None
    return hf_store.bounded(fn, limit or STORE["call_limit"], tok)


def store_halt(kind, **rec):
    """The watchdog's stop file: nothing runs again until a person has read it. The caller raises SystemExit."""
    marker = os.path.join(HOME, "HALTED")
    if not os.path.exists(marker):
        with open(marker, "x") as handle:
            json.dump({"kind": kind, **rec, "at": time.time()}, handle)
            handle.flush()
            os.fsync(handle.fileno())
    log(f"store: {kind}: {rec}; HALTED written, the session ends")


def store_preflight(st, jobs):
    """Before a session is created: every branch and every input manifest can be read from home, and no branch
    lists less than home has seen pushed. False (after a growing wait is set) while the store cannot be read: a
    session that could only fail its boot is never paid for."""
    import hf_store
    now = time.time()
    if now < st.d.get("store_preflight_next", 0):
        return False
    try:
        store = store_open(jobs)
        heads = store_call(lambda: hf_store.preflight(store, store_branches(jobs).values(), STORE["inputs"]))
    except Exception as e:  # noqa: BLE001
        n = st.d["store_preflight_failures"] = st.d.get("store_preflight_failures", 0) + 1
        wait = min(1800, 60 * 2 ** min(n, 5))
        st.d["store_preflight_next"] = now + wait
        st.save()
        log(f"store: not readable from home ({store_err(e)}); no session is created; next try in {wait} s (failure {n})")
        store_auth(st, "preflight", e)
        return False
    st.d["store_preflight_failures"], st.d["store_preflight_next"] = 0, 0
    st.save()
    # An attempt belongs to its state folder. A state folder that has never created a session and finds generations
    # on its jobs' branches is looking at an earlier attempt's: with the same campaign name and jobs file their
    # identity matches, and the VM would resume from them without a word. Refused here, before any session. Once
    # this state folder has passed the check it carries "store_attempt", and its own restarts go on.
    if not st.d.get("store_attempt"):
        earlier = {b: heads[b] for n, b in store_branches(jobs).items()
                   if heads.get(b) is not None and not st.job(n)["relayed_step"] and not st.job(n)["home"]}
        if earlier and not st.d["launches"]:
            msg = (f"store: {', '.join(f'{b} holds generations up to g{s}' for b, s in sorted(earlier.items()))}, and this "
                   f"state folder has never launched: they are an earlier attempt's, and this run would resume from them. "
                   f"Nothing is created. To continue that attempt, start with its own --state-dir. To start again, give "
                   f"this attempt its own names in the store with --store-campaign NEW_NAME (it is appended to every "
                   f"branch and put before every mark), or delete those branches and their marks first.")
            log(msg)
            raise SystemExit(msg)
        st.d["store_attempt"] = {"campaign": STORE["campaign"], "at": now}
        st.save()
    for n, branch in store_branches(jobs).items():
        js, head = st.job(n), heads.get(branch)
        if js["home"]:
            continue
        if js["relayed_step"] and (head is None or head < js["relayed_step"]):
            store_halt("STORE_HISTORY_MISSING", job=n, branch=branch, home_saw=js["relayed_step"], store_lists=head)
            raise SystemExit(3)
        if head is None and not STORE["first_start"]:
            st.d["store_preflight_next"] = now + 1800
            st.save()
            log(f"store: {branch} holds no generation and --store-first-start was not given; {n} would start from "
                "step 0, so no session is created")
            return False
    return True


def store_send(st, jobs):
    """prepare(), before the boot: the store's config (no secret in it) by exec, then the token as a file, moved to
    a tmpfs by the very next call. The token is never in exec code, a command line or a log; only its path is."""
    names = store_branches(jobs)
    cfg = {"repo": STORE["repo"], "repo_type": STORE["repo_type"], "mark_ns": STORE.get("mark_ns", ""),
           # A VM never releases storage (store_release does, from home), so its config carries no release
           # setting. Its commits pass a budget: the hub limits commit creation per repository.
           "store_commit_budget_per_hour": STORE.get("commit_budget", 0),
           # The courier's final delivery window, which the boot reads: after the VM's jobs end the courier has
           # this long to push the last of it before the boot takes the token away.
           "courier_wait_s": STORE.get("courier_wait_s", STORE_COURIER_WAIT_S),
           # The storage guard: the cap, and what is counted for this campaign so far: the courier's last word, or
           # the last measurement a release from home took (the new VM's courier counts on from there).
           "push_cap_bytes": STORE.get("push_cap_bytes", 0), "pushed_bytes_start": st.d.get("store_pushed_bytes", 0),
           "min_push_interval_s": STORE["min_push_interval_s"], "branches": names, "campaign": STORE["campaign"],
           "epoch": st.d["launches"], "min_step": {n: st.job(n)["relayed_step"] for n in names},
           "first_start": [n for n in names if STORE["first_start"] and not st.job(n)["relayed_step"]]}
    rc, out = exec_py("import json, os\n"
                      f"cfg, p = {cfg!r}, {STORE_CONFIG_VM!r}\n"
                      "try:\n"
                      "    old = json.load(open(p))\n"
                      "except (OSError, ValueError):\n"
                      "    old = {}\n"
                      "if old.get('token_file'):\n"
                      "    cfg['token_file'] = old['token_file']  # adopted by an earlier call on this VM\n"
                      "json.dump(cfg, open(p + '.tmp', 'w'), indent=1)\n"
                      "os.replace(p + '.tmp', p)\n"
                      "print('STORE_OK')\n")
    if "STORE_OK" not in out:
        log(f"prepare: store config not written: {out.strip()[-120:]}")
        return False
    if STORE["token_file"]:
        rc, _ = T.put(STORE["token_file"], STORE_TOKEN_VM, 300)
        out = ""
        if rc == 0:
            rc, out = exec_py(
                "import json, os\n"
                f"src, p = {STORE_TOKEN_VM!r}, {STORE_CONFIG_VM!r}\n"
                "d = os.environ.get('PK_STORE_TOKEN_DIR', '/dev/shm/pk_store')\n"
                "os.makedirs(d, mode=0o700, exist_ok=True)\n"
                "dst = os.path.join(d, 'token')\n"
                "tmp = dst + '.tmp%d' % os.getpid()\n"
                "try:\n"
                "    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)\n"
                "    with os.fdopen(fd, 'w') as f:\n"
                "        f.write(open(src).read())\n"
                "    os.replace(tmp, dst)\n"
                "finally:\n"
                "    for q in (tmp, src):\n"
                "        if os.path.exists(q):\n"
                "            os.remove(q)\n"
                "cfg = json.load(open(p))\n"
                "cfg['token_file'] = dst\n"
                "json.dump(cfg, open(p + '.tmp', 'w'), indent=1)\n"
                "os.replace(p + '.tmp', p)\n"
                "print('TOKEN_ADOPTED')\n")
        if rc != 0 or "TOKEN_ADOPTED" not in out:
            store_forget_token("after a failed hand-over")
            log("prepare: store token file not handed over; will retry")
            return False
    store_stall_reset(st, jobs)  # the stall clock starts again on this VM
    st.d["store_boot_at"] = time.time()
    st.save()
    log(f"store: config and token file sent ({STORE['repo']}, branches {sorted(names.values())}, epoch {cfg['epoch']})")
    return True


def store_progress(st, jobs, name):
    """A running store job: its pushed step, read at home from the newest generation's manifest in the store. No VM
    call and no checkpoint download; relayed_step then means "pushed to the store". When it has not advanced for
    --store-stall-min minutes the log says so on every tick, and with --store-stall-halt the arm halts.

    The stall time is counted WITHIN A SESSION: from the first tick the job is seen RUNNING on it, or from the last
    new generation. Every launch starts the clock again (store_stall_reset), so the minutes it took to notice a
    lost session, to create the next one and to boot it are not in it. What can be in it on a healthy arm: the
    first save after the resume (one save interval) and a rung that holds training (jobs_vm.py's "exclusive": up
    to its timeout of an hour, and five minutes more): about 85 minutes, so 120 is the least a stall halt takes."""
    js, now = st.job(name), time.time()
    js.setdefault("store_seen_at", now)
    try:
        store, branch = store_open(jobs), store_branches(jobs)[name]
        step = store_call(lambda: store.head_step(branch))
    except Exception as e:  # noqa: BLE001 a store that does not answer never ends a driver by itself
        log(f"{name}: store not read ({store_err(e)}); will retry")
        store_auth(st, name, e)
        step = None
    else:
        if st.d.pop("store_auth_failure", None):
            log(f"{name}: the store answers home's token again")
    if step is not None and step > js["relayed_step"]:
        js["relayed_step"], js["store_seen_at"] = step, now
        st.save()
        log(f"{name}: the store holds its generation at step {step}")
        return
    st.save()
    idle = (now - js["store_seen_at"]) / 60
    if STORE["stall_min"] and idle > STORE["stall_min"]:
        log(f"{name}: STORE STALLED: no new generation in the store for {idle:.0f} min (step {js['relayed_step']}) while "
            "the job runs; a session lost now loses that much training")
        if STORE["stall_halt"]:
            store_halt("STORE_STALLED", job=name, step=js["relayed_step"], minutes=round(idle))
            raise SystemExit(3)


def store_fetch_mark(st, jobs):
    """Each tick: ONE mark snapshot that the store's main branch holds and home does not, straight from the store
    to ART/<job>/rung_snapshots/rung_<step>/, each file checked by sha256. Never from the VM, and never deleted."""
    rec = st.d.setdefault("store_marks", {})
    wanted = [j["name"] for j in jobs if "cmd" not in j and j.get("snapshot_steps")]
    store = store_open(jobs)

    def one():
        tree = store.b.tree("main") or {}
        for name in wanted:
            for step in store.mark_steps(name, tree):
                if f"{name}/rung_{step}" not in rec:
                    man = store.pull_mark(name, step, snap_home(name, step), expect={"campaign": STORE["campaign"], "job": name})
                    w = man["files"]["model.safetensors"]
                    return [name, step, w["bytes"], w["sha256"]]
        return None

    try:
        got = store_call(one, max(1800, STORE["call_limit"]))
    except Exception as e:  # noqa: BLE001
        log(f"store marks: {store_err(e)}; will retry")
        store_auth(st, "marks", e)
        return
    if got:
        name, step, size, sha = got
        rec[f"{name}/rung_{step}"] = {"bytes": size, "sha256": sha, "fetched": time.time()}
        st.save()
        log(f"{name}: mark snapshot at step {step} fetched from the store ({size / 1e6:.0f} MB)")


def store_durable(st, jobs, j):
    """A training job that has FINISHED on the VM is home only when what it leaves is safe: its final generation is
    in the store, whole by the courier's own check made from here (the listing against the manifest the VM wrote
    from its hashed bytes, and a byte read of each large file); every mark at or below the final step is in the
    store; and every such mark is home, verified by sha256. Until then the state is "finished, durability
    pending": the session stays (a lost one is recreated and the job resumes to push again), and after
    --store-settle-min minutes the campaign halts for a person instead of being certified. A check that could not
    be read is pending, not a verdict; five running are said apart, with the HTTP status (STORE CHECK UNREAD), and
    the halt's record carries the count. A final generation that is in the store and NOT whole halts the campaign
    without the settle window (STORE_FINAL_GENERATION_NOT_WHOLE): it is written once and waiting cannot mend it.
    Before that halt the check is made again: STORE_FINAL_CHECKS completed checks that differ, ONE A TICK (so the
    three span two ticks, eight minutes, and no single bounded store call has to hold them), each kept in
    state.json with what it saw as soon as it is made. A check that cannot be read in between leaves them as they
    are; one that reads whole clears them. The halt's record carries every one of them."""
    import hf_store
    n, now = j["name"], time.time()
    js, branch = st.job(n), store_branches(jobs)[n]
    final = ((j.get("spec") or {}).get("train") or {}).get("steps") or js.get("final_step")
    if not final:  # a spec without its step count (the fake's): the VM's own final checkpoint names it
        rc, out = exec_py(f"print('FINALSTEP', open('/content/chat/{n}/state.txt').readline().strip())\n")
        m = re.search(r"^FINALSTEP (\d+)$", out, re.M)
        final = js["final_step"] = int(m.group(1)) if m else None
    missing, unread = [], None
    pend = st.d.setdefault("durability_pending", {})
    try:
        if not final:
            raise ValueError("the final step is not known yet")
        store = store_open(jobs)

        def look():
            # The courier's own check, made from home: the listing, the manifest read again, and a byte of each
            # large file. An error (anything unread) leaves the job pending; False is a completed comparison,
            # returned with what it saw.
            listed = final in store.steps(branch, store.b.tree(branch) or {})
            whole, saw = store.why_not_whole(branch, final) if listed else (False, None)
            if whole and store.read_manifest(branch, f"gens/g{final}").get("job") != n:
                whole, saw = False, "the manifest names another job"
            return [bool(whole), store.mark_steps(n), listed, saw]

        gen_ok, marks, listed, saw = store_call(look)
        differ = (pend.get(n) or {}).get("final_checks") or []
        if gen_ok and differ:
            pend[n].pop("final_checks")
            log(f"{n}: the final generation g{final} reads whole after {len(differ)} reading(s) that differed; they "
                "were not the store's state")
        if listed and not gen_ok:
            # In the store and not whole. Kept at once, with what it saw; believed at the STORE_FINAL_CHECKS-th:
            # a generation is written once, so waiting cannot mend it. The halt is by name, with every check; the
            # two verified generations before it are in the store.
            p = pend.setdefault(n, {"since": now})
            differ = p["final_checks"] = differ + [{"at": now, "saw": saw}]
            st.save()
            if len(differ) >= STORE_FINAL_CHECKS:
                store_halt("STORE_FINAL_GENERATION_NOT_WHOLE", job=n, branch=branch, step=final, checks=len(differ), seen=differ,
                           why="the run's last generation is in the store and is not whole (a file is short, missing, of "
                               "another hash or job, or its bytes cannot be had); it will not be written again")
                raise SystemExit(3)
            log(f"{n}: finished; the final generation g{final} read as in the store and not whole (check {len(differ)} of "
                f"{STORE_FINAL_CHECKS}: {saw}); read again at the next tick before anything is halted")
            return False
        want = [s for s in j.get("snapshot_steps", []) if s <= final]
        if not gen_ok:
            missing.append(f"final generation g{final} not whole in the store")
        missing += [f"mark {s} not in the store" for s in want if s not in marks]
        missing += [f"mark {s} not home" for s in want if s in marks and f"{n}/rung_{s}" not in st.d.get("store_marks", {})]
    except Exception as e:  # noqa: BLE001
        missing.append(f"store not read ({store_err(e)})")
        if final:
            unread = hf_store.http_status(e) or "no HTTP status"
            store_auth(st, n, e)
    if not missing:
        if pend.pop(n, None):
            log(f"{n}: durable: final generation g{final} and marks in the store, marks home")
        st.save()
        return True
    p = pend.setdefault(n, {"since": now})
    if p.get("why") != missing:
        log(f"{n}: finished, durability pending: {'; '.join(missing)}")
    p["why"] = missing
    # Checks that could not be read, one after another: said apart at the fifth, with the HTTP status, so that a
    # store that answers the same refusal every time is not left to the settle window without a word.
    p["unread"], p["http_status"] = (p.get("unread", 0) + 1, unread) if unread else (0, None)
    if p["unread"] == 5:
        log(f"{n}: STORE CHECK UNREAD: the check of the final generation g{final} could not be read from home "
            f"{p['unread']} times running (HTTP status: {unread}). No verdict is drawn from it; the campaign halts "
            f"with STORE_DURABILITY_PENDING when the settle window is over unless a check is read before")
    st.save()
    if now - p["since"] > STORE["settle_min"] * 60:
        store_halt("STORE_DURABILITY_PENDING", job=n, missing=missing, minutes=STORE["settle_min"],
                   unread_checks_running=p["unread"], http_status=p["http_status"],
                   final_checks_that_differed=p.get("final_checks") or [])
        raise SystemExit(3)
    return False


def store_boot_failed(st, commit):
    """One more failed boot at this commit; main() refuses to start once there are STORE_BOOT_FAILURES."""
    bf = st.d.get("store_boot_failures") or {}
    st.d["store_boot_failures"] = {"commit": commit, "n": (bf.get("n", 0) if bf.get("commit") == commit else 0) + 1}
    st.save()
    log(f"store: boot failure {st.d['store_boot_failures']['n']} of {STORE_BOOT_FAILURES} at {commit}")


def store_forget_token(when="before the session stops"):
    """End the courier and its children (they hold the token in memory), wait for them, then remove every copy of
    the token from the VM."""
    out = ""
    try:
        rc, out = exec_py(
            "import glob, json, os, signal, time\n"
            "c = '/content/chat'\n"
            "try:\n"
            "    pid = int(open(c + '/courier.pid').read())\n"
            "except (OSError, ValueError):\n"
            "    pid = None\n"
            "for sig in (signal.SIGTERM, signal.SIGKILL):\n"
            "    if not pid or not os.path.exists('/proc/%d' % pid):\n"
            "        break\n"
            "    try:\n"
            "        os.killpg(pid, sig) if os.getpgid(pid) == pid else os.kill(pid, sig)\n"
            "    except OSError:\n"
            "        break\n"
            "    for _ in range(100):\n"
            "        if not os.path.exists('/proc/%d' % pid):\n"
            "            break\n"
            "        time.sleep(0.1)\n"
            f"paths = [{STORE_TOKEN_VM!r}]\n"
            "try:\n"
            "    tf = json.load(open(c + '/store.json')).get('token_file')\n"
            "except (OSError, ValueError):\n"
            "    tf = None\n"
            "d = os.environ.get('PK_STORE_TOKEN_DIR', '/dev/shm/pk_store')\n"
            # The tmpfs folder is searched whatever the config says: a hand-over that died after the move and
            # before the config named the file leaves the token there and nowhere on record.
            "paths += glob.glob(d + '/token*') + ([tf] + glob.glob(tf + '.tmp*') if tf else [])\n"
            "for p in paths:\n"
            "    if os.path.exists(p):\n"
            "        os.remove(p)\n"
            "print('TOKEN_LEFT' if any(os.path.exists(p) for p in paths) else 'TOKEN_GONE')\n", timeout=120)
    except Exception:  # noqa: BLE001 the stop that follows must run whatever happens here
        pass
    log(f"store: courier ended and token removed from the VM ({when})" if "TOKEN_GONE" in out else
        f"store: token removal not confirmed {when} (the VM did not answer, or a copy is left)")


STORE_VM_READ = (
    "import json, os, time\n"
    "c = '/content/chat'\n"
    "names = {names!r}\n"
    "def last_step(p, tail=1 << 18):\n"
    "    try:\n"
    "        size = os.path.getsize(p)\n"
    "        with open(p, 'rb') as f:\n"
    "            f.seek(max(0, size - tail))\n"
    "            lines = f.read().split(b'\\n')\n"
    "        at = os.path.getmtime(p)\n"
    "    except OSError:\n"
    "        return [None, None]\n"
    "    for ln in reversed(lines):\n"
    "        if b'\"step\"' not in ln:\n"
    "            continue\n"
    "        try:\n"
    "            e = json.loads(ln)\n"
    "        except ValueError:\n"
    "            continue\n"
    "        if isinstance(e, dict) and e.get('event') == 'step' and isinstance(e.get('step'), int):\n"
    "            return [e['step'], at]\n"
    "    return [None, None]\n"
    "try:\n"
    "    cs = json.load(open(c + '/courier_status.json'))\n"
    "except (OSError, ValueError):\n"
    "    cs = None\n"
    "keep = ('updated', 'boot_id', 'epoch', 'bytes_since_release', 'pushed_bytes', 'measure_seq', 'push_cap_bytes', 'capped',\n"
    "        'http_429', 'commits_last_hour', 'commit_budget_per_hour')\n"
    "jk = ('pushed_step', 'pushed_at', 'live_step', 'pushes', 'failures', 'last_error', 'fenced', 'check_unread',\n"
    "      'unverified_since', 'auth_failure', 'to_verify', 'not_whole')\n"
    "r = {{'at': time.time(), 'courier': None, 'training': {{n: last_step(c + '/' + n + '/events.jsonl') for n in names}}}}\n"
    "if isinstance(cs, dict):\n"
    "    r['courier'] = {{k: cs.get(k) for k in keep}}\n"
    "    r['courier']['jobs'] = {{n: {{k: (cs.get('jobs') or {{}}).get(n, {{}}).get(k) for k in jk}} for n in names}}\n"
    "print('STOREVM', json.dumps(r))\n")


def store_vm_read(jobs):
    """One bounded call to the VM: the courier's own status (pushed steps, the bytes counted against the push cap
    and the bytes this boot has pushed, a reached push cap, HTTP 429 and commit counts) and, per training job, the trainer's current
    training step, which is the step of the last `step` event in its own events.jsonl with that file's modification
    time: where the trainer is, not what it has saved. None when the VM does not answer."""
    try:
        rc, out = exec_py(STORE_VM_READ.format(names=sorted(store_branches(jobs))), timeout=120)
    except Exception:  # noqa: BLE001 a read that fails is "no answer"
        return None
    m = re.search(r"^STOREVM (.*)$", out or "", re.M)
    try:
        return json.loads(m.group(1)) if m else None
    except ValueError:
        return None


def store_guard(st, jobs, vm):
    """The storage guard, each tick. The courier counts what the store held for this campaign at the last release
    from home (store_release hands it the measurement) plus the bytes it has pushed since, and stops pushing at
    --store-push-cap-gb; its count is kept in state.json so the next VM's courier counts on from it. A courier
    that has reached the cap halts the campaign here, by name: STORE_PUSH_CAP.

    Also here: a pushed generation whose check the courier could not read ten times running or more (its status,
    "check_unread") is said in this log once for each generation, with the HTTP status the courier met, so that a
    store that refuses the byte read every time is seen at home while the run goes on. A refusal of the VM's token
    by the hub (the courier's "auth_failure") is said as an authentication failure, once for each run of them.
    Neither line halts anything by itself.

    What does halt, when the registration switches it on (--store-unverified-halt-min, or the jobs file's
    "store_unverified_halt_min"; 0, the default, is off): the courier's "unverified_since", the time of the oldest
    push since which no check of a push has read whole, older than that many minutes by the VM's own clock. That
    is the state in which every check is unread or every generation is not whole: nothing is pruned, the store
    grows by a generation a save and training bills with nothing that counts. The campaign halts by name,
    STORE_UNVERIFIED, through the confirmed stop. Pushes that do not arrive at all are --store-stall-halt's."""
    c = (vm or {}).get("courier")
    if not c:
        return
    said = st.d.get("store_check_unread_said") or []
    for n, cj in sorted((c.get("jobs") or {}).items()):
        cj = cj or {}
        for step, u in sorted((cj.get("check_unread") or {}).items()):
            if isinstance(u, dict) and (u.get("count") or 0) >= 10 and f"{n}/g{step}" not in said:
                said.append(f"{n}/g{step}")
                st.d["store_check_unread_said"] = said
                st.save()
                log(f"{n}: STORE CHECK UNREAD on the VM: the courier could not read the check of g{step} {u['count']} "
                    f"times running ({u.get('error')}). The generation counts for nothing until it is read and nothing "
                    "is removed on its account; a refusal that stays wants a person to read the store")
        au = cj.get("auth_failure")
        if isinstance(au, dict) and au.get("since") != (st.d.get("store_auth_vm_said") or {}).get(n):
            st.d.setdefault("store_auth_vm_said", {})[n] = au.get("since")
            st.save()
            log(f"{n}: STORE AUTHENTICATION FAILURE on the VM: the hub refused the courier's token (HTTP "
                f"{au.get('http_status')}, during {au.get('during')}). Nothing is stored while it lasts. To mend it: "
                "replace the token file at home (mode 600), end this runner and start it again")
        lim, since, at = STORE.get("unverified_halt_min") or 0, cj.get("unverified_since"), (vm or {}).get("at")
        if lim and isinstance(since, (int, float)) and isinstance(at, (int, float)) and at - since > lim * 60:
            store_halt("STORE_UNVERIFIED", job=n, minutes=round((at - since) / 60), limit_min=lim,
                       why="pushes have reached the store and no check of one has read whole for that long: every check "
                           "unread, or every generation not whole; nothing is pruned and nothing counts",
                       to_verify=cj.get("to_verify"), not_whole=cj.get("not_whole"), check_unread=cj.get("check_unread"),
                       auth_failure=au, pushed_step=cj.get("pushed_step"))
            raise SystemExit(3)
    if isinstance(c.get("bytes_since_release"), int) and c["bytes_since_release"] != st.d.get("store_pushed_bytes"):
        st.d["store_pushed_bytes"] = c["bytes_since_release"]
        st.save()
    if c.get("capped"):
        running = [n for n in store_branches(jobs) if not st.job(n)["home"]]
        store_halt("STORE_PUSH_CAP", job=running[0] if running else None, why=c["capped"].get("why"),
                   bytes_since_release=c["capped"].get("bytes_since_release"), cap_bytes=c["capped"].get("cap"),
                   releases=st.d.get("store_releases"))
        raise SystemExit(3)


# A measurement's write on the VM is conditional and atomic. It is for one boot (the courier whose pushed-bytes
# figure it carries) and has a sequence number: it is written only on that boot, and only over a file of a lower
# sequence, so a call that was delayed and arrives after a newer measurement cannot put the older one back (the
# courier would lose the newer reset and count too much). The file is written whole and renamed into place.
STORE_MEASURE_VM = (
    "import json, os\n"
    "c = '/content/chat'\n"
    "m, p = {m!r}, c + '/store_measure.json'\n"
    "try:\n"
    "    boot = open(c + '/boot_id').read().strip()\n"
    "except OSError:\n"
    "    boot = None\n"
    "try:\n"
    "    old = json.load(open(p))\n"
    "except (OSError, ValueError):\n"
    "    old = None\n"
    "if boot != m['boot_id']:\n"
    "    print('MEASURE_OTHER_BOOT')\n"
    "elif isinstance(old, dict) and old.get('boot_id') == m['boot_id'] and isinstance(old.get('seq'), int) and old['seq'] >= m['seq']:\n"
    "    print('MEASURE_OLDER')\n"
    "else:\n"
    "    tmp = p + '.tmp%d' % os.getpid()\n"
    "    with open(tmp, 'w') as f:\n"
    "        json.dump(m, f)\n"
    "    os.replace(tmp, p)\n"
    "    print('MEASURE_OK')\n")


def store_release(st, jobs, final=False, limit=None):
    """Storage release, from home (--store-release-storage): the one releaser of the whole repo. Every
    --store-release-every-min minutes, and once more at the campaign's end (final), under the store's release lock,
    by difference over every branch, sparing objects younger than --store-release-grace-s. No VM releases: a
    release costs two commits (the lock and its removal), and the hub limits commits per repository. Several
    runners on one repo (three arms) each keep this timer; the lock arbitrates, and a timed lock that is not taken
    (held by another, or no room in home's commit budget) is left until the next turn of the timer. A release
    deletes only what a commit removed and recorded in a ledger (hf_store.release_report): never an object on
    main, never another campaign's live object, never an upload whose commit has not come.

    The release at the campaign's end (final) is an essential commit of home's budget: it may use the whole budget,
    without the reserve a timed lock leaves. Its FIRST attempt is made in the loop, before the session is stopped,
    and can take up to the call's limit while the session still bills. It is a debt from before that attempt
    (state.json "store_release_final_owed", written first) until it is recorded as made
    ("store_release_final_done"); store_release_owed() pays it after the session has been stopped, or at the next
    start of a runner that died. limit: seconds this call may take at the most (the owed wait's remainder).
    Returns, for the caller of a final release: "done"; "held" (nothing was sent: another's lock, or no room in the
    budget); "refused" (the lock's commit went out and the hub refused it); "failed"; or "ledgers" (the round ran
    and a ledger did not read); None for a timed release that was not due.

    A release that ran also measured the store. What is counted for this campaign from then on, each LFS object
    once: what its own branches name under gens/, its own marks, and what its own branches' ledgers name that is
    still stored. Nothing a branch inherited from main, nothing of another campaign, no object that belongs to
    nobody. That number replaces the storage guard's count: in state.json, and on the VM
    (CHAT/store_measure.json, with the courier's own pushed-bytes figure as read just before the release, so
    what it has pushed since is counted on top). It is this campaign's share, not the account's usage. Never
    raises: a release that fails is tried again at the next turn."""
    if not STORE["reclaim"]:
        return
    now, every = time.time(), STORE["release_every_min"] * 60
    if st.d.get("store_release_next") is None:
        # The first regular turn: a full interval after the campaign's start, and this arm's offset after that
        # (--store-release-offset-min: the registration gives the three arms 0, a third and two thirds of the
        # interval, so that their rounds do not fall due together).
        st.d["store_release_next"] = st.d["store_release_regular"] = now + every + STORE.get("release_offset_min", 0) * 60
        st.save()
    if not final:
        store_measure_send(st)  # a measurement that did not reach the VM is sent again, each tick, until it does
    if not final and now < st.d["store_release_next"]:
        return
    if not final:
        # The regular turns are ANCHORED: each is `every` after the one before it, not after the moment a round
        # happened to run, so a round that was held and ran a tick later does not shift the arm's turns (and the
        # arms' offsets last). A regular turn gives the one early retry back.
        regular = st.d.get("store_release_regular") or now
        if now >= regular:
            while regular <= now:
                regular += every
            st.d["store_release_regular"], st.d["store_release_spent"] = regular, 0
        st.d["store_release_next"] = regular
    rec = st.d.setdefault("store_releases", {"count": 0, "held": 0, "failed": 0, "objects": 0, "bytes": 0})
    st.save()
    c = {} if final else ((store_vm_read(jobs) or {}).get("courier") or {})
    t0 = time.time()

    def again(sent):
        """When a release that did not run is tried again, as words; for a timed one the time is set here. sent:
        commits may have gone out (a failure, a round that ran), so the early retry is one, STORE_RELEASE_RETRY_S
        later; with nothing sent it is the next tick."""
        if final:
            return "it stays owed"
        if not sent:
            st.d["store_release_next"] = now
            return "nothing was sent; tried again at the next tick"
        regular = st.d.get("store_release_regular") or now + every
        if st.d.get("store_release_spent", 0) < 1 and now + STORE_RELEASE_RETRY_S < regular:
            st.d["store_release_spent"] = 1
            st.d["store_release_next"] = now + STORE_RELEASE_RETRY_S
            return f"tried again in {STORE_RELEASE_RETRY_S / 60:.0f} min, once"
        st.d["store_release_next"] = regular
        return "next at its regular turn"

    def say(msg, why):
        # A release that does not run is tried again and again (at each tick, or through the owed wait): one line
        # for each change of state, not one a try.
        slot, key = ("owed_said" if final else "timed_said"), re.sub(r"\d+", "N", why)
        if _STORE.get(slot) != key:
            log(msg)
        _STORE[slot] = key

    if final:
        # The debt is in state.json BEFORE the release is attempted: a runner that dies in the call, or after it
        # and before its success is recorded, finds it at its next start (main() services it without a session).
        st.d["store_release_final_owed"] = st.d.get("store_release_final_owed") or "not tried yet"
        # And the round is COUNTED before it is made: a runner that dies after the round's commits have gone out
        # and before its outcome is saved has still spent one of the folder's STORE_FINAL_LEDGER_ROUNDS. Only a
        # round that was merely held (nothing was sent) is given back, below; one that is made clears the count.
        st.d["store_release_final_rounds"] = st.d.get("store_release_final_rounds", 0) + 1
        st.save()
    lim = max(900, STORE["call_limit"]) if limit is None else max(1, min(max(900, STORE["call_limit"]), limit))
    try:
        store = store_open(jobs)
        # takeover: a lock that carries this runner's own name was left by a process of it that died (or by an
        # unlock that failed); one runner makes one release at a time, so it is taken again, not waited out.
        # (takeover, with this call's limit: hf_store.release_report says when a lock of this runner's own name
        # is taken again.)
        r = store_call(lambda: store.release_report(f"home:{NAME}", grace=STORE["release_grace_s"], essential=final,
                                                    takeover=lim, now=now), lim)
    except Exception as e:  # noqa: BLE001 a release that fails never ends a driver
        rec["failed"] += 1
        if final:
            st.d["store_release_final_owed"] = f"the release failed ({store_err(e)})"
        when = again(True)
        st.save()
        say(f"store: release failed ({store_err(e)}); {when}", "failed " + type(e).__name__)
        return "failed"
    if r["held"]:
        rec["held"] += 1
        sent = "commit was refused" in str(r.get("why"))  # the lock's commit went out and was refused: not "merely held"
        if final:
            st.d["store_release_final_owed"] = f"the lock was not taken ({r.get('why')})"
            if not sent:  # nothing was sent: the round reserved above was not one
                st.d["store_release_final_rounds"] = max(0, st.d.get("store_release_final_rounds", 1) - 1)
        when = again(sent)
        st.save()
        say(f"store: release: the lock was not taken ({r.get('why') or 'another releaser holds it, or main or the budget gave no room'}); {when}",
            "held " + str(r.get("why")))
        return "refused" if sent else "held"
    if r.get("ledgers_unread"):
        # A ledger that could not be read: nothing was deleted, and what it holds is in nobody's count. The
        # guard's count stays as it is (the courier's own, which only grows) until a release reads them all.
        rec["unread"] = rec.get("unread", 0) + 1
        named = ", ".join(r.get("unread") or []) or f"{r['ledgers_unread']} of them"
        if final:
            # (The round was counted before it was made.) A ledger that is not well formed is kept by its own
            # identity, its path and the sha256 of its bytes: no new round while that same file is still there.
            st.d["store_release_final_owed"] = f"the round ran and {r['ledgers_unread']} ledger(s) could not be read: {named}"
            if r.get("malformed"):
                st.d["store_release_final_malformed"] = {**(st.d.get("store_release_final_malformed") or {}), **r["malformed"]}
        when = again(True)
        st.save()
        say(f"store: release: {r['ledgers_unread']} ledger(s) could not be read ({named}); nothing was deleted and the "
            f"storage guard's count is left as it is; {when}", "ledgers " + named)
        return "ledgers"
    _STORE["timed_said" if not final else "owed_said"] = None
    if final:  # made: recorded at once, before the measurement is used
        for k in ("store_release_final_owed", "store_release_final_rounds", "store_release_final_malformed"):
            st.d.pop(k, None)
        st.d["store_release_final_done"] = True
        st.save()
    else:
        st.d["store_release_spent"] = 0
    names, ns = store_branches(jobs), STORE.get("mark_ns", "")
    import hf_store
    mine = hf_store.campaign_bytes(r, names.values(), [ns + n for n in names])  # each object once
    seq = st.d["store_measure_seq"] = st.d.get("store_measure_seq", 0) + 1
    rec.update(count=rec["count"] + 1, objects=rec["objects"] + r["objects"], bytes=rec["bytes"] + r["bytes"],
               last={"at": now, "objects": r["objects"], "bytes": r["bytes"], "stored_bytes": r["stored_bytes"],
                     "orphan_bytes": r["orphan_bytes"], "orphan_objects": r["orphan_objects"],
                     "pending_bytes": r["pending_bytes"], "counted_bytes": mine, "final": final})
    st.d["store_pushed_bytes"] = mine
    st.save()
    log(f"store: release: {r['objects']} objects deleted, {r['bytes']} bytes freed; the store holds {r['stored_bytes']} "
        f"bytes; {r['orphan_objects']} objects ({r['orphan_bytes']} bytes) are in no branch and no ledger and are left "
        f"alone; {mine} counted for this campaign" + (f"; {r['unknown_sizes']} objects of unknown size" if r["unknown_sizes"] else ""))
    log("store: stat " + json.dumps({"what": "release", "objects": r["objects"], "bytes": r["bytes"],
                                     "stored_bytes": r["stored_bytes"], "orphan_bytes": r["orphan_bytes"],
                                     "orphan_objects": r["orphan_objects"], "pending_bytes": r["pending_bytes"],
                                     "ledgers": r["ledgers"], "ledgers_done": r["ledgers_done"],
                                     "unknown_sizes": r["unknown_sizes"],
                                     "counted_bytes": mine, "grace_s": STORE["release_grace_s"], "final": final,
                                     "seconds": round(time.time() - t0, 2)}, sort_keys=True))
    if isinstance(c.get("pushed_bytes"), int) and c.get("boot_id"):
        st.d["store_measure_pending"] = {"seq": seq, "boot_id": c["boot_id"], "bytes": mine, "at_pushed": c["pushed_bytes"], "at": now}
        st.save()
        store_measure_send(st)
    return "done"


def store_measure_send(st):
    """A release's measurement, to the VM (CHAT/store_measure.json). One that did not arrive is kept in state.json
    and sent again at each tick until it does: until then the courier counts on from its own, higher, number, and
    a measurement lost for good would count as a release that never ran. The courier takes it only on the boot it
    was measured for, so one that arrives after the session was replaced does nothing."""
    m = st.d.get("store_measure_pending")
    if not m:
        return
    try:
        rc, out = exec_py(STORE_MEASURE_VM.format(m=m), timeout=120)
    except Exception:  # noqa: BLE001
        out = ""
    if "MEASURE_OK" in (out or ""):
        st.d.pop("store_measure_pending")
        st.save()
        if _STORE.pop("measure_said", None):
            log("store: the measurement has reached the VM")
    elif "MEASURE_OLDER" in (out or "") or "MEASURE_OTHER_BOOT" in (out or ""):
        # The VM answered and did not take it: a newer one is there already, or the session is another boot's
        # now. It is settled; sending it again would change nothing.
        st.d.pop("store_measure_pending")
        st.save()
        _STORE.pop("measure_said", None)
        log("store: the measurement was not taken on the VM ("
            + ("a newer one is there" if "MEASURE_OLDER" in out else "the session is of another boot now") + ")")
    elif not _STORE.get("measure_said"):
        _STORE["measure_said"] = True
        log("store: the measurement did not reach the VM; its courier counts on from its own, higher, number; it is "
            "sent again at each tick")


def store_release_debt(st):
    """Why the release at the campaign's end is still owed, or None: with --store-release-storage, until it is
    recorded as made. The text an attempt left, else the plain fact for a campaign that is home."""
    if not STORE or not STORE["reclaim"] or st.d.get("store_release_final_done"):
        return None
    return st.d.get("store_release_final_owed") or \
        ("the campaign is home and its last release is not recorded as made" if st.d.get("campaign_home") else None)


def store_release_owed(st, jobs, stopped=True):
    """The release at the campaign's end, when it is still owed at the runner's end: because its first attempt, made
    in the loop BEFORE the session was stopped, did not run; or because an earlier runner died before or during it
    or before recording it (main() comes here without creating a session). NOT COVERED: a RunPod runner started on a
    state folder that is home exits earlier, at main()'s "every job is home; nothing to run", and pays no debt; the
    start-up path is Colab's.

    stopped=True (the session's stop is confirmed, so the wait bills nothing): it is tried now and then every
    STORE_FINAL_RELEASE_NAP_S seconds, for STORE_FINAL_RELEASE_WAIT_S in all and no longer: each call is given what
    is left of that time as its limit, and is ended at it. Its lock is essential to home's budget, which lets it
    pass at the latest STORE_HOME_ESSENTIAL_WAIT_S after it first asked. A round that RAN and met a ledger that
    could not be read is tried once more and no further (each round is a lock and an unlock: two commits, up to
    three), the loop's own round and those of earlier starts counted, in state.json, each BEFORE it is made (a
    round that failed after its lock counts, a runner killed in one has spent it; only a round that was merely
    held is given back); a round that failed is followed by five minutes, not thirty seconds; with a ledger that is not
    well formed it is not tried again at all until its branch moves (store_release_blocked). A start on a folder
    in that state makes no commit: it writes the line and ends. The wait is measured on the monotonic clock. The
    log has one line for each change of state, not one a try.

    stopped=False (STOP_UNRESOLVED): nothing is tried; the debt is said.

    On every way out with the debt unpaid, ONE line says that the release did not run, why (a ledger by branch and
    path), and the command that makes it by hand, and the command's config is written beside the state
    (store_by_hand). The debt stays in state.json for the next start. Never raises."""
    if not store_release_debt(st):
        return
    note = "" if stopped else "the session's stop is not confirmed (STOP_UNRESOLVED), so the release was not tried; "
    try:
        blocked = store_release_blocked(st, jobs) if stopped else None
        if blocked:
            note = blocked + "; "
        elif stopped:
            end = time.monotonic() + STORE_FINAL_RELEASE_WAIT_S  # monotonic: a clock set back cannot stretch the wait
            log(f"store: the release at the campaign's end is owed ({store_release_debt(st)}); no session bills, and it is "
                f"tried for at most {STORE_FINAL_RELEASE_WAIT_S:.0f} s")
            while end - time.monotonic() >= 1:
                out = store_release(st, jobs, final=True, limit=end - time.monotonic())
                if out == "done":
                    return
                # Every round that was not merely held is counted (before it was made): with the allowance used,
                # or a ledger that is not well formed found, no more commits are spent on it.
                blocked = store_release_blocked(st, jobs, read=False) if out != "held" else None
                if blocked:
                    note = blocked + "; "
                    break
                # A held round sent nothing and is tried again soon; one that failed after taking the lock sent
                # commits, and is given five minutes before the one round that is left.
                nap = min(STORE_FINAL_RELEASE_FAILED_NAP_S if out == "failed" else STORE_FINAL_RELEASE_NAP_S,
                          end - time.monotonic() - 1)
                if nap <= 0:
                    break
                time.sleep(nap)
    except Exception as e:  # noqa: BLE001 nothing here may keep the runner from ending
        log(f"store: the release at the campaign's end stopped on an error ({type(e).__name__})")
    finally:  # on EVERY way out, a KeyboardInterrupt too: the line, when the debt is still there
        store_by_hand(st, note)


def store_release_blocked(st, jobs, read=True):
    """Why no further round of the end's release is made, or None. Both reasons are in state.json, so they hold
    across every start of this state folder (a watchdog that starts a finished runner again spends no commit):
    - a ledger that was not well formed reads the same in every round. Each is kept by its OWN identity
      ("store_release_final_malformed": {"branch:pruned/g<step>.json": the sha256 of its bytes}), which a save on
      its branch does not change (the branch of an arm that is still training moves every few minutes). No new
      round while ANY of them is still there as it was: each is read here (a read, not a commit; a read that fails
      changes nothing), and one that is gone or has other bytes is dropped from the record, alone. When none is
      left the release is tried again, with its allowance whole. Or a person removes the key;
    - STORE_FINAL_LEDGER_ROUNDS rounds of the end's release have been begun and none was recorded as made
      ("store_release_final_rounds"; a round is counted before it is made and given back only when it was merely
      held): a ledger that could not be read, a failure after the lock, or a runner that died in a round. No more
      until a person removes the key."""
    bad = st.d.get("store_release_final_malformed")
    if bad:
        if read:
            try:
                store = store_open(jobs)
                now_ = store_call(lambda: {k: store.file_sha256(*k.split(":", 1)) for k in sorted(bad)})
            except Exception:  # noqa: BLE001
                now_ = bad
            still = {k: v for k, v in bad.items() if now_.get(k) == v}
            if still != bad:
                log(f"store: a ledger that was not well formed is gone or has changed ({sorted(set(bad) - set(still))})"
                    + (f"; {sorted(still)} is still there as it was" if still else "; the release is tried again"))
                if still:
                    st.d["store_release_final_malformed"] = still
                else:
                    st.d.pop("store_release_final_malformed")
                    st.d.pop("store_release_final_rounds", None)
                st.save()
            bad = still
        if bad:
            return (f"a ledger that is not well formed ({', '.join(sorted(bad))}) is still in the store as it was, and no "
                    "new round is made while it is (or until store_release_final_malformed is removed from state.json)")
    if st.d.get("store_release_final_rounds", 0) >= STORE_FINAL_LEDGER_ROUNDS:
        return (f"{STORE_FINAL_LEDGER_ROUNDS} rounds of the release were begun and none was recorded as made (a ledger that "
                "could not be read, a failure after the lock, or a runner that died in one), and no more are made until "
                "store_release_final_rounds is removed from state.json")
    return None


def store_by_hand(st, note=""):
    """The one line for a release at the campaign's end that did not run, and the config its command reads:
    HOME/store_release_by_hand.json, mode 600, written whole and renamed into place. It holds no secret (the repo
    and the token file's PATH) and no commit budget, so the command is not held back by home's count. Never raises."""
    try:
        import shlex
        why = store_release_debt(st)
        if not why:
            return
        cfg = os.path.join(HOME, "store_release_by_hand.json")
        tmp = f"{cfg}.tmp{os.getpid()}"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump({**{k: STORE.get(k) for k in ("repo", "repo_type", "token_file", "mark_ns")},
                       "store_commit_budget_per_hour": 0}, f, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, cfg)
        cmd = (f"python3 {shlex.quote(os.path.join(HERE, 'hf_store.py'))} release --config {shlex.quote(cfg)} "
               f"--grace {STORE['release_grace_s']:.0f} --owner {shlex.quote('hand:' + str(NAME))}")
        bad = " A ledger that is not well formed stops every release, this command's too: read the file named above and " \
              "repair or remove it first." if "not well formed" in why else ""
        log(f"store: THE RELEASE AT THE CAMPAIGN'S END DID NOT RUN: {note}{why}. Nothing is lost: the removed generations' "
            f"ledgers and objects stay in the store until a release reads them, and the debt stays in state.json for "
            f"this runner's next start.{bad} By hand: {cmd} (it prints \"released\": the objects deleted; exit 0 with "
            f"\"released\": null means the lock was held and NOTHING was released: run it again)")
    except Exception as e:  # noqa: BLE001
        log(f"store: THE RELEASE AT THE CAMPAIGN'S END DID NOT RUN, and its by-hand config could not be written "
            f"({type(e).__name__}); by hand: hf_store.py release, with a config that names the repo and the token file")


def store_commit_stat(jobs):
    """Once an hour: the commits home sent to the store in it (the release's locks), as one line of the log."""
    if not STORE["reclaim"]:
        return  # home commits nothing without the release
    try:
        b = store_open(jobs).budget
        rep = b.report() if b else None
    except Exception:  # noqa: BLE001
        rep = None
    if rep:
        log("store: stat " + json.dumps({"what": "commits", "where": "home", **rep}, sort_keys=True))


BRING_HOME_TAIL = 300 * 1024  # bytes of each file's end that come home
BRING_HOME = (
    "import base64, glob, json, os\n"
    "c = '/content/chat'\n"
    "names = ['courier.log', 'boot.log', 'courier_status.json', 'store_inputs.json', 'COURIER_FENCED', 'COURIER_CAPPED',\n"
    "         'COURIER_NOT_WHOLE', 'COURIER_CHECK_UNREAD', 'jobs.log', 'watch/watch.jsonl']\n"
    "names += sorted('store_restore/' + os.path.basename(p) for p in glob.glob(c + '/store_restore/*.json'))[:24]\n"
    "out = {{}}\n"
    "for n in names:\n"
    "    p = os.path.join(c, n)\n"
    "    if os.path.isfile(p):\n"
    "        size = os.path.getsize(p)\n"
    "        with open(p, 'rb') as f:\n"
    "            f.seek(max(0, size - {tail}))\n"
    "            out[n] = [size, base64.b64encode(f.read()).decode()]\n"
    "print('BRINGHOME', json.dumps(out))\n")


def store_bring_home(st, when):
    """Before every stop this runner makes (the drill, the end, a halt): the session's own record comes home, into
    HOME/sessions/launchNN/, because a stopped VM takes it along. courier.log (each push's bytes and seconds, each
    restore pull's, HTTP 429 counts, each release's objects and bytes freed), boot.log (the build, inputs and
    restore times, the VM's own TOKEN_GONE), courier_status.json, store_restore/*.json, and store_inputs.json,
    jobs.log, a rehearsal's watch/watch.jsonl (the watcher's samples of this session, which a forced stop would
    otherwise take along) and the courier's FENCED and CAPPED markers when they exist. Bounded: one exec with a time limit, the
    last 300 KB of each file. None of these files holds the token; its text is removed here all the same. Never
    raises: nothing here may keep a stop from happening."""
    try:
        if not st.d.get("launches"):
            return False  # this state folder has never had a session: there is nothing to ask
        sess = f"launch{int(st.d['launches']):02d}"
        rc, out = exec_py(BRING_HOME.format(tail=BRING_HOME_TAIL), timeout=200)
        m = re.search(r"^BRINGHOME (.*)$", out or "", re.M)
        if not m:
            log(f"store: session files NOT brought home ({when}): the VM did not answer")
            return False
        got = json.loads(m.group(1))
        dst = os.path.join(HOME, "sessions", sess)
        os.makedirs(dst, exist_ok=True)
        tok = _STORE["store"].b.token if "store" in _STORE else None
        index = {"session": NAME, "launch": st.d.get("launches"), "endpoint": st.d.get("endpoint"), "when": when,
                 "at": time.time(), "files": {}}
        for name, (size, b64) in sorted(got.items()):
            data = base64.b64decode(b64)
            whole = len(data) >= size
            if tok:
                data = data.replace(tok.encode(), b"[token]")
            if name.endswith(".log"):
                # A log on the VM can hold a signed address (the library's own warning for a download it resumes,
                # in boot.log): the query string of every address is removed from what is kept at home.
                import hf_store
                data = hf_store.no_query(data.decode("utf-8", "replace")).encode("utf-8")
            with open(os.path.join(dst, name.replace("/", "__")), "wb") as f:
                f.write(data)
            index["files"][name] = {"bytes_on_vm": size, "bytes_home": len(data), "whole": whole}
        with open(os.path.join(dst, "index.json"), "w") as f:
            json.dump(index, f, indent=1)
        log(f"store: session files home in sessions/{sess} ({when}): {', '.join(sorted(got)) or 'none on the VM'}")
        return True
    except Exception as e:  # noqa: BLE001
        log(f"store: session files NOT brought home ({when}): {type(e).__name__}")
        return False


def tail_switch(j):
    """The last step a job trains on its main corpus (the tail starts at the next one), or None for a job with no
    tail: int(from_fraction * steps), as the trainer computes it."""
    spec = j.get("spec") or {}
    frac = ((spec.get("data") or {}).get("anneal") or {}).get("from_fraction")
    steps = (spec.get("train") or {}).get("steps")
    return int(frac * steps) if isinstance(frac, (int, float)) and steps else None


def store_drill_evidence(st, jobs, j):
    """The drill's evidence, read from the VM immediately before the stop (the courier has just been ended, so its
    pushed step can no longer move): the trainer's training step from its own event stream and the courier's pushed
    step, each with its time. `between_pushes` is true only when the training step is above the pushed step. The
    side of the tail switch the trainer is on is recorded beside the side the drill step is on, and `side_missed`
    says so when they differ (a drill meant for the main part that landed in the tail, or the other way round)."""
    n = j["name"]
    rec = st.d["pod_loss_drill"]
    vm = store_vm_read(jobs)
    tr = ((vm or {}).get("training") or {}).get(n) or [None, None]
    cj = (((vm or {}).get("courier") or {}).get("jobs") or {}).get(n) or {}
    step, pushed = tr[0], cj.get("pushed_step")
    last_main = tail_switch(j)
    side = None if step is None or last_main is None else ("tail" if step > last_main else "main")
    meant = None if last_main is None else ("tail" if j["pod_loss_drill_step"] > last_main else "main")
    rec.update({
        "evidence": {"read_at_home": time.time(), "vm_clock": (vm or {}).get("at"), "vm_answered": vm is not None,
                     "training_step": step, "training_step_at": tr[1],
                     "pushed_step": pushed, "pushed_at": cj.get("pushed_at"), "saved_step": cj.get("live_step"),
                     "courier_status_at": ((vm or {}).get("courier") or {}).get("updated")},
        "between_pushes": step is not None and pushed is not None and step > pushed,
        "steps_lost": step - pushed if step is not None and pushed is not None else None,
        "last_main_step": last_main, "drill_side": meant, "side": side,
        "side_missed": side is not None and meant is not None and side != meant})
    st.save()
    log(f"{n}: drill evidence: training step {step}, pushed step {pushed}, between pushes {rec['between_pushes']}, "
        f"side {side} (the drill step is on the {meant} side)" + ("; SIDE MISSED" if rec["side_missed"] else "")
        + ("" if vm is not None else "; THE VM DID NOT ANSWER, nothing was read"))
    if vm is not None:
        store_guard_count(st, vm)


def store_guard_count(st, vm):
    c = (vm or {}).get("courier") or {}
    if isinstance(c.get("bytes_since_release"), int):
        st.d["store_pushed_bytes"] = c["bytes_since_release"]
        st.save()


def store_drill_stopped(st, jobs, j, confirmed):
    """After the drill's stop: when it returned, whether it was confirmed, and what the store then lists for the
    job (read from home; None when the store did not answer). The whole record goes to HOME/drill_record.json."""
    rec = st.d["pod_loss_drill"]
    rec["stop_returned_at"], rec["stop_confirmed"] = time.time(), bool(confirmed)
    try:
        store, branch = store_open(jobs), store_branches(jobs)[j["name"]]
        rec["store_step_after_stop"] = store_call(lambda: store.head_step(branch))
    except Exception as e:  # noqa: BLE001
        rec["store_step_after_stop"], rec["store_read_error"] = None, store_err(e)
    st.save()
    with open(os.path.join(HOME, "drill_record.json"), "w") as f:
        json.dump(rec, f, indent=1)
    log(f"{j['name']}: drill record written (drill_record.json): {json.dumps(rec)}")


def halt_dpo_checkpoint(error):
    marker = os.path.join(HOME, "HALTED")
    if not os.path.exists(marker):
        with open(marker, "x") as handle:
            json.dump({"kind": "DPO_FULL_STATE_CHECKPOINT_FAILURE", "error": str(error)}, handle)
            handle.flush()
            os.fsync(handle.fileno())


def prepare(st, commit, jobs_rel, jobs, uploads):
    """Inputs, relayed checkpoints and the boot onto a session that is up but
    not yet booted. Retried every tick until it succeeds (st.d["booted"]), so
    a failed upload never leaves a provisioned VM idle."""
    validate_checkpoint_families(jobs, BOOT_SCRIPT)
    dpo_leases = []
    if any(j.get("dpo_full_checkpoint_v1") for j in jobs):
        from dpo_checkpoint_transport_v1 import home_uploads
        try:
            extra, dpo_leases = home_uploads(jobs, ART)
            uploads = [*uploads, *extra]
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
            halt_dpo_checkpoint(error)
            raise
    adopt_home(st, jobs)  # files home but state unmarked: FINISHED below, not run again
    dirs = ["/content/chat/upload"] + [f"/content/chat/upload/{j['name']}" for j in jobs]
    rc, out = exec_py(f"import os\nfor d in {dirs!r}: os.makedirs(d, exist_ok=True)\nprint('DIRS_OK')\n")
    if "DIRS_OK" not in out:
        log(f"prepare: upload dirs not created: {out.strip()[-120:]}")
        return False
    # tc_b2_stop.py reset-failed: a person has read the failure. A volume holds FAILED until this clears it, and jobs_vm.py runs a job anew.
    resets = st.d.get("reset_jobs") or []
    if resets:
        rc, out = exec_py(f"import os\nfor n in {resets!r}:\n    for f in ('STATUS_' + n, n + '.rc', n + '.pid'):\n"
                          "        p = '/content/chat/' + f\n        if os.path.exists(p):\n            os.remove(p)\nprint('RESET_OK')\n")
        if "RESET_OK" not in out:
            log(f"prepare: failed jobs not reset: {out.strip()[-120:]}")
            return False
        st.d["reset_jobs"] = []
        st.save()
        log(f"prepare: jobs reset on the volume: {resets}")
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
    if STORE:  # a training job's checkpoint never crosses the home link: boot_base2.sh pulls it on the VM
        pend = [n for n in pend if "cmd" in by[n]]
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
    if STORE:  # mark snapshots come from the store on the VM (ckpt_courier.py restore), not from home
        rung_up = [u for u in rung_up if "/rung_" not in u[2]]
    full_leases = []
    if any(j.get("full_checkpoint_v2") for j in jobs):
        from sfull_checkpoint_transport_v2 import home_uploads
        extra, full_leases = home_uploads(jobs, ART)
        uploads = [*uploads, *extra]
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
    if STORE and not store_send(st, jobs):
        return False
    rc, out = exec_py(boot_launch_code(commit, jobs_rel, BOOT_SCRIPT))
    if "BOOTED" not in out:
        log(f"boot exec failed: {out.strip()[-160:]}")
        return False
    st.d["booted"] = True
    st.save()
    log("boot launched")
    if full_leases:
        from sfull_checkpoint_transport_v2 import release_uploads
        release_uploads(full_leases)
    if dpo_leases:
        from dpo_checkpoint_transport_v1 import release_uploads
        release_uploads(dpo_leases)
    return True


def boot_launch_code(commit, jobs_rel, boot_script="tools/colab_chat/boot_jobs.sh"):
    """Pinned boot selection; the legacy training boot remains the default."""
    if boot_script not in ("tools/colab_chat/boot_jobs.sh", "tools/colab_chat/boot_commands.sh", "tools/colab_chat/boot_commands_sparse.sh", "tools/colab_chat/boot_sfull_data_only_20261031.sh", "tools/colab_chat/boot_sfull_v2_20261101.sh", "tools/colab_chat/boot_dpo_checkpoint_v1.sh",
                           "tools/colab_chat/boot_base2.sh"):
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


def parse_deadline_utc(value):
    """Timezone-aware ISO8601 only; return an absolute Unix timestamp."""
    from datetime import datetime
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})", value):
        raise argparse.ArgumentTypeError("deadline must be timezone-aware ISO8601, e.g. 2026-10-30T13:17:00Z")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.utcoffset() is None:
            raise ValueError("timezone required")
        return parsed.timestamp()
    except ValueError as exc:
        raise argparse.ArgumentTypeError("invalid ISO8601 deadline") from exc


def resolve_deadline(max_hours, absolute, now):
    """A watchdog restart cannot extend a supplied absolute deadline."""
    if not math.isfinite(max_hours) or max_hours <= 0:
        raise ValueError("--max-hours must be finite and positive")
    deadline = now + max_hours * 3600
    if not math.isfinite(deadline):
        raise ValueError("--max-hours must produce a finite deadline")
    if absolute is not None:
        deadline = min(deadline, absolute)
    if deadline <= now:
        raise ValueError("runner deadline has already expired; no session may be created")
    return deadline


def sleep_tick(deadline):
    """Wait at most one normal tick, and never intentionally sleep past the ceiling."""
    delay = min(TICK, max(0, deadline - time.time()))
    if delay > 0:
        time.sleep(delay)


def main():
    global NAME, ART, HOME, T, BOOT_SCRIPT, STORE
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--commit", required=True)
    ap.add_argument("--jobs", required=True, help="jobs JSON, path relative to the paperkiln repo root")
    ap.add_argument("--boot-script", default="tools/colab_chat/boot_jobs.sh",
                    choices=["tools/colab_chat/boot_jobs.sh", "tools/colab_chat/boot_commands.sh", "tools/colab_chat/boot_commands_sparse.sh", "tools/colab_chat/boot_sfull_data_only_20261031.sh", "tools/colab_chat/boot_sfull_v2_20261101.sh", "tools/colab_chat/boot_dpo_checkpoint_v1.sh",
                             "tools/colab_chat/boot_base2.sh"],
                    help="pinned VM boot; command-only boot skips trainer and corpus setup")
    ap.add_argument("--max-hours", type=float, default=23.5)
    ap.add_argument("--deadline-utc", type=parse_deadline_utc, default=None,
                    help="absolute timezone-aware ISO8601 ceiling, preserved across watchdog restarts")
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
    ap.add_argument("--ckpt-store", choices=["hf"], default=None,
                    help="checkpoints, mark snapshots and inputs go through a store repo from the VM (hf_store.py, "
                         "ckpt_courier.py, boot_base2.sh), not through the home link")
    ap.add_argument("--store-repo", default=None, help="the store repo, OWNER/NAME (no default)")
    ap.add_argument("--store-token-file", default=None, help="file holding the store's write token, mode 600 (no default)")
    ap.add_argument("--store-repo-type", choices=["dataset", "model"], default="dataset")
    ap.add_argument("--store-min-push-interval", type=float, default=0,
                    help="seconds between a job's generation pushes (0: every save), for a store that limits commits")
    ap.add_argument("--store-release-storage", action="store_true",
                    help="release storage from home: every --store-release-every-min minutes, and at the campaign's "
                         "end, this runner deletes for good the LFS objects of generations that a commit removed "
                         "and recorded in a ledger, under the store's release lock (off: safe to leave off; no VM "
                         "ever releases). Never an object a branch names, an object on main, or an upload that is "
                         "not committed yet; those in no branch and no ledger are reported and left (by hand: "
                         "hf_store.py release --orphans, with no VM writing). Several runners on one repo may each "
                         "carry the flag: the lock lets one release at a time")
    ap.add_argument("--store-release-every-min", type=float, default=None,
                    help="minutes between this runner's releases (each costs two commits of the repo's hourly limit). "
                         "Default: the jobs file's \"store_release_every_min\", else 120")
    ap.add_argument("--store-release-offset-min", type=float, default=None,
                    help="minutes added to this runner's first release turn; its turns are then --store-release-every-min "
                         "apart from that one, whenever a round happens to run. Three arms on one repo take 0, a third "
                         "and two thirds of the interval, so their rounds do not fall due together. Default: the jobs "
                         "file's \"store_release_offset_min\", else 0")
    ap.add_argument("--store-release-grace-s", type=float, default=None,
                    help="(default: the jobs file's \"store_release_grace_s\", else 7200) "
                         "release leaves alone what a ledger younger than this names: the time a restore that listed a "
                         "generation just before its removal has to finish downloading it. Refused under 600")
    ap.add_argument("--store-commit-budget-per-hour", type=int, default=24,
                    help="commits and branch creations an hour ONE VM may send to the store (its courier and data "
                         "job together). A ceiling for that writer, not for the repository: the hub allows a "
                         "repository 128 an hour, all writers together, and nothing ties the VMs' counts together. "
                         "The defaults sum under it for three arms (3 x 24 + 3 x 12 = 108). 0: no limit")
    ap.add_argument("--store-home-commit-budget-per-hour", type=int, default=12,
                    help="commits an hour THIS runner may send to the store (the release's lock and unlock, and an "
                         "unlock sent again). A quarter is kept back from the timed release, so four of those an hour "
                         "at the most; the release at the campaign's end may use it all and waits for room. Per "
                         "runner, not per repository. 0: no limit")
    ap.add_argument("--store-push-cap-gb", type=float, default=None,
                    help="storage guard: counted are the bytes the store held for this campaign when a release from "
                         "home last measured it (each object once: what its branches name under gens/, its marks, "
                         "and what its ledgers name that is still stored), plus what its courier has pushed since; "
                         "without a release, everything pushed. This campaign's share, not the account's usage. At "
                         "this many GB the courier stops pushing and the runner halts (STORE_PUSH_CAP). Default: the "
                         "jobs file's \"store_push_cap_gb\", else 0 (off)")
    ap.add_argument("--store-first-start", action="store_true",
                    help="admit a job whose branch holds no generation, and that home has never seen pushed, to start from step 0")
    ap.add_argument("--store-campaign", default=None,
                    help="campaign name written into every generation (default: --name). Given explicitly it is also a "
                         "namespace in the store: appended to every job's branch and put before every job's marks, so "
                         "a new attempt under a new name never resumes from an earlier attempt's generations")
    ap.add_argument("--store-call-limit-s", type=float, default=300, help="wall-time limit of one store call from home")
    ap.add_argument("--store-stall-min", type=float, default=None,
                    help="say so loudly when a running job's store step has not advanced for this many minutes (0: never). "
                         "Default: the jobs file's \"store_stall_min\", else 60. A rung that holds training (jobs_vm.py: "
                         "\"exclusive\", up to its \"timeout_s\" of an hour) leaves the job RUNNING with no new "
                         "generation for that long, so with --store-stall-halt this must be well over an hour")
    ap.add_argument("--store-stall-halt", action="store_true",
                    help="halt the arm on that alarm (STORE_STALLED, through the confirmed stop). It covers pushes that "
                         "do not arrive, whatever the cause (a refused token, 429s, a transport that fails): what is "
                         "watched is the newest generation the store lists, read from home, while the job runs. It does "
                         "not cover pushes that arrive and cannot be checked (--store-unverified-halt-min). Also the "
                         "jobs file's \"store_stall_halt\": true")
    ap.add_argument("--store-unverified-halt-min", type=float, default=None,
                    help="halt the arm (STORE_UNVERIFIED, through the confirmed stop) when the courier has gone this "
                         "many minutes with pushes in the store and none of them read back whole: every check unread "
                         "(a store that refuses the byte read) or every one not whole. Nothing is pruned in that "
                         "state, so the store grows by a generation a save. Default: the jobs file's "
                         "\"store_unverified_halt_min\", else 0 (off)")
    ap.add_argument("--store-settle-min", type=float, default=95,
                    help="minutes a finished job may stay 'durability pending' before the campaign halts. Never under "
                         "the courier's final delivery window (--store-courier-wait-s) and a tick")
    ap.add_argument("--store-courier-wait-s", type=float, default=STORE_COURIER_WAIT_S,
                    help="the courier's final delivery window: seconds a VM's courier has, after its jobs have ended, "
                         "to push the last generation and marks before the boot removes the token (sent to the VM; "
                         "the courier's own waits are at most 15 minutes for its budget and 30 for a 429)")
    ap.add_argument("--store-boot-limit-min", type=float, default=75,
                    help="minutes without a boot status after which the boot counts as failed and the session is stopped")
    a = ap.parse_args()
    try:
        resolve_deadline(a.max_hours, a.deadline_utc, time.time())
    except ValueError as exc:
        ap.error(str(exc))
    if a.ckpt_store:
        STORE = store_config(a, ap)
    elif a.store_repo or a.store_token_file or a.store_min_push_interval or a.boot_script == STORE_BOOT:
        ap.error("--store-repo, --store-token-file and boot_base2.sh need --ckpt-store hf")
    BOOT_SCRIPT = a.boot_script
    NAME, ART, HOME = a.name, a.art, os.path.expanduser(a.state_dir)
    os.makedirs(HOME, exist_ok=True)
    os.makedirs(ART, exist_ok=True)
    repo = os.path.dirname(os.path.dirname(HERE))
    jobs = json.load(open(os.path.join(repo, a.jobs)))["jobs"]
    validate_checkpoint_families(jobs, a.boot_script)
    uploads = [tuple(u.split("=", 1)) for u in a.upload]
    for k, src in uploads:
        if not os.path.exists(src):
            sys.exit(f"--upload {k}: {src} does not exist")
    if os.path.exists(os.path.join(HOME, "HALTED")):
        sys.exit("HALTED marker present: nothing runs until a person has read it (the marker stays)")
    pid = os.path.join(HOME, "runner.pid")
    # One runner a state folder, by the folder's lock: taken here without waiting and held, open, for this process's
    # life (a forked store call shares it and dies with this process). The pid file is written further down for
    # people to read and decides nothing: a pid from before a restart of WSL can be some other live process.
    if fcntl is not None:
        _STORE["runner_lock"] = open(os.path.join(HOME, "runner.lock"), "a")
        try:
            fcntl.flock(_STORE["runner_lock"], fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.exit("another runner holds this state folder (runner.lock); refusing")
        except OSError as e:  # not "held": the file system would not lock at all (a 9p mount can refuse)
            sys.exit(f"the state folder's lock could not be taken ({type(e).__name__}: {e}); refusing to run without it. "
                     "Keep state folders on a Linux file system (ext4), not under /mnt/c")
    elif os.path.exists(pid):
        try:
            os.kill(int(open(pid).read()), 0)
            sys.exit("another runner is live (pid file); refusing")
        except (OSError, ValueError):
            pass
    st = State()
    if STORE:
        store_owner(st, a.name, jobs)
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
    bf = st.d.get("store_boot_failures") or {}
    if STORE and bf.get("commit") == a.commit and bf.get("n", 0) >= STORE_BOOT_FAILURES:
        sys.exit(f"{bf['n']} boots failed at {a.commit} (ART/boot.log); refusing to create another session for it "
                 "(remove store_boot_failures from state.json once the cause is fixed)")
    if a.backend != "colab" and st.d.get("boot_failed_commit") == a.commit:
        sys.exit(f"the boot failed at {a.commit} (ART/boot.log); refusing to rent another pod for it")
    # Nor for a campaign that is home (a restart after the end would otherwise launch a pod and find nothing to do).
    if a.backend != "colab" and all(st.d["jobs"].get(j["name"], {}).get("home") for j in jobs):
        sys.exit("every job is home; nothing to run (no pod rented)")
    try:
        resolve_deadline(a.max_hours, a.deadline_utc, time.time())
    except ValueError as exc:
        ap.error(str(exc))
    T = make_transport(a)
    open(pid, "w").write(str(os.getpid()))
    if a.backend != "colab":
        st.d["gpu"] = a.gpu
        st.d["backend"] = a.backend
        st.save()
        T.attach(st)  # registers the runner for runpod_sweep.py; atexit and signal handlers end the pod

    deadline = resolve_deadline(a.max_hours, a.deadline_utc, time.time())
    T.deadline = deadline  # transport must enforce it while a transfer blocks the main loop
    # With the store, a state folder in which every job is home has nothing to run: a runner started on it (after a
    # crash or a kill at the campaign's end, or by a watchdog) creates NO session. It goes straight to the clean-up
    # below, which stops a session an earlier runner left up, and pays what the campaign's end still owes: the
    # release (store_release_owed).
    # Only a session whose listed endpoint is the one this folder recorded at its launch is this folder's: that one
    # has its token removed and is stopped. Any other session under the name is left alone. (RunPod is not covered
    # by this path: its runner exits above, at "every job is home", and pays no debt.)
    home_already = bool(STORE) and bool(jobs) and all(st.d["jobs"].get(j["name"], {}).get("home") for j in jobs)
    vm_cleanup, stop_wanted, stop_unknown = True, True, False
    try:
        # The first launch is inside the clean-up below too: a failure after the session exists (its preparation,
        # the token's hand-over) still takes the token off the VM and stops the session, with confirmation.
        if home_already:
            left, ep = store_left_session(st)
            vm_cleanup = stop_wanted = left == "ours"
            stop_unknown = left in ("unknown", "other")  # neither is a stop this runner can call confirmed
            log("every job is home in this state folder: no session is created"
                + {"ours": "; the session an earlier runner left is stopped",
                   "none": "; the listing answers and holds no session of this name: no token to remove, nothing to stop",
                   "other": f"; A SESSION IS LISTED UNDER THIS NAME at {ep}, which is NOT the endpoint this folder recorded "
                            f"({st.d.get('endpoint')}): it is taken for another runner's and is left alone (no token "
                            f"removed, no stop). If it is this folder's after all it is still billing: check `colab "
                            f"sessions`, and stop it by hand (colab stop -s {NAME})",
                   "unknown": f"; THE LISTING DID NOT ANSWER, so it is not known whether the session this folder recorded "
                              f"({st.d.get('endpoint')}) is still up: its token was NOT removed and it was NOT stopped. "
                              f"Check `colab sessions`, and stop it by hand if it is there (colab stop -s {NAME})"}[left]
                + ("; the release at the campaign's end is owed" if STORE["reclaim"] and not st.d.get("store_release_final_done") else ""))
            st.d["campaign_home"] = True
            st.save()
        elif listed() is None and not st.d.get("endpoint"):
            if not launch(st, a.commit, a.jobs, jobs, uploads):
                if time.time() < deadline and not T.over_budget(st):
                    sleep_tick(deadline)
        while not home_already and time.time() < deadline:
            why = T.over_budget(st)
            if why:
                log(why)
                break
            T.refresh(st)
            seen = listed()
            if STORE and seen is not None and seen != st.d.get("endpoint"):
                # The endpoint is recorded at the launch from ONE listing, which can come back empty while the VM is
                # fine. A session listed under this runner's own name while it holds the state folder's lock is its
                # own: what the listing shows is recorded, so a later start knows which session this folder's is.
                log(f"session listed at {seen} (recorded: {st.d.get('endpoint')}); recorded")
                st.d["endpoint"] = seen
                st.save()
            if seen is None:
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
                            sleep_tick(deadline)
                            continue
                    elif probe is None:
                        log(f"session not listed and VM silent ({st.d['gone_ticks']}/3); waiting")
                        sleep_tick(deadline)
                        continue
            else:
                st.d["gone_ticks"] = 0
            if not st.d.get("booted", True):
                log("session up but not booted; preparing it")
                if not prepare(st, a.commit, a.jobs, jobs, uploads):
                    sleep_tick(deadline)
                    continue
            upload_pending(st)
            r = vm_status(jobs)
            if r is None:
                log("vm status: no reply")
                sleep_tick(deadline)
                continue
            # boot_jobs.sh writes FINISHED when the manager exits. With jobs open that is a dead manager: nothing would ever end this pod.
            js_open = [j["name"] for j in jobs if j.get("halt_on_failure") and r[j["name"]] in ("RUNNING", "WAITING")]
            st.d["manager_gone"] = st.d.get("manager_gone", 0) + 1 if r["vm"] == "FINISHED" and js_open else 0
            if st.d["manager_gone"] >= 3:
                halt_campaign(js_open[0], "the VM's job manager has ended with jobs open")
                raise SystemExit(3)
            if STORE and r["vm"] == "RUNNING" and st.d.get("store_boot_failures"):
                st.d.pop("store_boot_failures")  # a boot that came up: earlier failures at this commit no longer count
            if STORE and r["vm"] == "BOOTING" and time.time() - st.d.get("store_boot_at", time.time()) > STORE["boot_limit_min"] * 60:
                log(f"boot over {STORE['boot_limit_min']} min without a status; counted as a failed boot")
                r["vm"] = "BOOT_FAILED"
            if STORE and r["vm"] == "BOOT_FAILED":
                store_boot_failed(st, a.commit)
            if STORE and r["vm"] in ("RUNNING", "FINISHED"):
                store_guard(st, jobs, store_vm_read(jobs))  # halts, by name, on a courier that reached its push cap
            if r["vm"] == "BOOT_FAILED":
                log("boot failed on the VM; downloading boot.log")
                T.get("/content/chat/boot.log", os.path.join(ART, "boot.log"), 120)
                if STORE:
                    store_clean_log(os.path.join(ART, "boot.log"))
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
                if r[n] in ("RUNNING", "FINISHED") and not js["home"] and j.get("full_checkpoint_v2"):
                    from sfull_checkpoint_transport_v2 import relay as full_relay
                    try:
                        relayed = full_relay(exec_py, download_file, st, j, ART) or relayed
                    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
                        with open(os.path.join(HOME, "HALTED"), "x") as marker:
                            json.dump({"kind": "S_FULL_CHECKPOINT_V2_FAILURE", "error": str(error)}, marker)
                        raise
                elif r[n] in ("RUNNING", "FINISHED") and not js["home"] and j.get("dpo_full_checkpoint_v1"):
                    from dpo_checkpoint_transport_v1 import relay as dpo_relay
                    try:
                        relayed = dpo_relay(exec_py, download_file, st, j, ART) or relayed
                    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
                        halt_dpo_checkpoint(error)
                        raise
                elif r[n] == "RUNNING" and not js["home"] and "cmd" not in j:
                    if STORE:
                        store_progress(st, jobs, n)
                    else:
                        relayed = relay(st, n, j.get("relay_every_steps", 0)) or relayed
                    drill = j.get("pod_loss_drill_step")
                    if drill and js["relayed_step"] >= drill and not st.d.get("pod_loss_drill"):
                        st.d["pod_loss_drill"] = {"job": n, "relayed_step": js["relayed_step"], "at": time.time()}
                        st.save()
                        log(f"{n}: pod loss drill: the pod ends at the provider now; relayed {js['relayed_step']}")
                        if STORE:  # every stop this runner makes takes the token off the VM first
                            store_forget_token("before the drill's stop")
                            store_bring_home(st, "before the drill's stop")
                            store_drill_evidence(st, jobs, j)  # last: the read is immediately before the stop
                        stopped = stop_session(st, "the pod loss drill", lose=True)
                        if STORE:
                            store_drill_stopped(st, jobs, j, stopped)
                        if not stopped:
                            raise SystemExit(4)
                        done = False
                        break
                if STORE and r[n] == "FINISHED" and not js["home"] and "cmd" not in j and not store_durable(st, jobs, j):
                    done = False
                    continue  # finished, durability pending: not home, the session stays
                if r[n] == "FINISHED" and not js["home"]:
                    if j.get("full_checkpoint_v2"):
                        from sfull_checkpoint_transport_v2 import finished_home_poll
                        try:
                            final_ready = finished_home_poll(j, ART, js)
                            st.save()
                        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
                            log(f"{n}: final full checkpoint refused: {error}")
                            with open(os.path.join(HOME, "HALTED"), "x") as marker:
                                json.dump({"kind": "S_FULL_FINAL_CHECKPOINT_FAILURE", "job": n, "error": str(error)}, marker)
                            raise
                        if not final_ready:
                            done = False
                            continue  # final model/AdamW/RNG/events must land, not merely weights
                    if j.get("dpo_full_checkpoint_v1"):
                        from dpo_checkpoint_transport_v1 import final_home_ready
                        if not final_home_ready(j, ART):
                            done = False
                            continue  # final optimizer/RNG/events bundle must land before home
                    if "cmd" in j:
                        fetch_outputs(st, n, j.get("outputs", []))
                    else:
                        fetch_final(st, n, j.get("model_name", n), export_formats(j))
                # FAILED counts only when seen on two ticks running: a status
                # read mid-reset (a job being restarted by hand) is not final.
                js["failed_seen"] = js.get("failed_seen", 0) + 1 if r[n] == "FAILED" else 0
                if r[n] == "FAILED" and js["failed_seen"] >= 2 and not js["home"]:
                    if j.get("halt_on_failure") or j.get("halt_drill"):
                        os.makedirs(os.path.join(ART, n), exist_ok=True)
                        if "cmd" in j:
                            fetch_diagnosis(n, j.get("outputs", []))
                        T.get(f"/content/chat/{n}_train.log", os.path.join(ART, n, f"{n}_train.log"), 300)
                        if j.get("halt_on_failure"):
                            halt_campaign(n, "job FAILED on the VM")
                            raise SystemExit(3)
                        halt_drill(st, n)
                        js["home"] = True
                        st.save()
                        continue
                    if "cmd" in j:  # a failed data job's outputs are its diagnosis: whatever exists
                        fetch_outputs(st, n, j.get("outputs", []), best_effort=True)
                    T.get(f"/content/chat/{n}_train.log", os.path.join(ART, n, f"{n}_train.log"), 300)
                    js["home"] = True
                    st.save()
                done = done and js["home"]
            for j in jobs:
                gate_upload(j, st)
            if STORE:
                store_fetch_mark(st, jobs)
                store_release(st, jobs, final=done)  # on its timer; and once more when the campaign is home
                store_commit_stat(jobs)
            if has_rungs and not STORE:
                try:
                    fetch_rung_snapshot(st, jobs, r, relayed)
                except (OSError, ValueError, KeyError, TypeError) as e:  # a backstop must never end a driver
                    log(f"rung snapshots: {e!r}")
            if any(pause_at_rung(j, st) for j in jobs):
                raise SystemExit(0)
            if done:
                log("all jobs finished and home")
                st.d["campaign_home"] = True
                st.save()
                break
            sleep_tick(deadline)
    finally:
        if STORE:
            if vm_cleanup:  # not for a start on a state folder that is home and has no session: there is no VM to ask
                store_forget_token()
                store_bring_home(st, "before the session stops")
            import hf_store
            hf_store.end_children()
        # A start on a folder that is home stops only its own session; with a listing that did not answer nothing
        # was stopped and that is not a confirmed stop.
        stopped = stop_session(st, "at the runner's end") if stop_wanted else not stop_unknown
        if st.d.get("campaign_home"):
            T.release(st)
        if STORE:
            # The campaign's last release, when it is still owed: tried now that no session bills; with a stop that
            # is not confirmed it is not tried, and the debt is said with its by-hand command all the same.
            store_release_owed(st, jobs, stopped=stopped)
        log("session stopped; runner exiting" if stopped else "SESSION STOP NOT CONFIRMED (STOP_UNRESOLVED); runner exiting")
        os.remove(pid)
        if not stopped:
            sys.exit(4)


if __name__ == "__main__":
    main()
