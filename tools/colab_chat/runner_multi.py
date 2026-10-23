#!/usr/bin/env python3
"""Reclaim-resilient Colab driver for a session of several training jobs.

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
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
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


def exec_py(code, timeout=200):
    path = os.path.join(HOME, "exec.py")
    with open(path, "w") as f:
        f.write(code)
    return colab("exec", "-s", NAME, "-f", path, "--timeout", "150", timeout=timeout)


def listed():
    rc, out = colab("sessions", timeout=120)
    m = re.search(r"^\[" + re.escape(NAME) + r"\]\s+(\S+)", out, re.M)
    return m.group(1) if m else None


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
    failed part is retried before the upload gives up."""
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
                rc, _ = colab("upload", "-s", NAME, part, f"{dst}.part{i:02d}", timeout=1500)
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
    0 on success, as colab() does."""
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
        rc, _ = colab("download", "-s", NAME, src, dst, timeout=timeout)
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
        rc, _ = colab("download", "-s", NAME, p, lp + ".tmp", timeout=900)
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
    save interval, so every relay came home torn and was discarded."""
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
        return
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
        return
    for f in ("model.safetensors", "events.jsonl"):
        os.replace(os.path.join(art, f + ".new"), os.path.join(art, f))
    stale = os.path.join(art, "optim.safetensors")  # older than these weights: never resume from it
    if os.path.exists(stale):
        os.remove(stale)
    os.replace(tmp, os.path.join(art, "state.txt"))  # last: state.txt names a complete set
    js["relayed_step"] = step
    st.save()
    log(f"{name}: relayed checkpoint at step {step}")


def fetch_final(st, name, model_name):
    art = os.path.join(ART, name)
    os.makedirs(art, exist_ok=True)
    ok = True
    for f in (f"{name}/{model_name}.safetensors", f"{name}/{model_name}.gguf", f"{name}/events.jsonl",
              f"{name}/result.json", f"{name}_spec.json", f"{name}_train.log"):
        ok = ok and download_file(f"/content/chat/{f}", os.path.join(art, os.path.basename(f))) == 0
    # Optional extra exports (e.g. q8_0_all) are fetched best-effort.
    download_file(f"/content/chat/{name}/{model_name}_q8_0_all.gguf", os.path.join(art, f"{model_name}_q8_0_all.gguf"))
    colab("download", "-s", NAME, f"/content/chat/{name}/{model_name}.ir.json",
          os.path.join(art, f"{model_name}.ir.json"), timeout=300)
    if ok:
        st.job(name)["home"] = True
        st.save()
        log(f"{name}: final outputs downloaded")


def fetch_outputs(st, name, outputs, best_effort=False):
    """A data job's declared outputs, paths relative to /content/chat/<name>/,
    plus its log: ONE file per call (per tick), so a gigabyte of outputs never
    holds up the checkpoint relays (21 Oct 2026: a 26-minute download blocked
    them and a reclaim lost an hour of training). The job counts as home when
    every file has arrived. best_effort (a failed job): all at once, misses ignored."""
    art = os.path.join(ART, name)
    os.makedirs(art, exist_ok=True)
    js = st.job(name)
    got = set(js.get("fetched", []))
    for f in [*outputs, f"../{name}_train.log"]:
        if f in got:
            continue
        # Outputs keep their relative paths (several sources write report.json).
        dst = os.path.join(art, os.path.basename(f) if f.startswith("../") else f)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        rc = download_file(f"/content/chat/{name}/{f}", dst)
        if rc == 0:
            js.setdefault("fetched", []).append(f)
            st.save()
            log(f"{name}: {f} downloaded")
        if not best_effort:
            return
    if best_effort or len(js.get("fetched", [])) == len(outputs) + 1:
        js["home"] = True
        st.save()
        log(f"{name}: outputs downloaded")


def launch(st, commit, jobs_rel, jobs, uploads):
    colab("stop", "-s", NAME, timeout=120)
    rc, out = colab("new", "-s", NAME, "--gpu", "L4", timeout=600)
    if rc != 0:
        log(f"no L4 ({out.strip()[-120:]}); retry later")
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
        rc, _ = colab("upload", "-s", NAME, os.path.join(art, f), f"/content/chat/upload/{name}/{f}", timeout=600)
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


def prepare(st, commit, jobs_rel, jobs, uploads):
    """Inputs, relayed checkpoints and the boot onto a session that is up but
    not yet booted. Retried every tick until it succeeds (st.d["booted"]), so
    a failed upload never leaves a provisioned VM idle."""
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
    by = {j["name"]: j for j in jobs}

    def needs(j):
        after = j.get("after") or []
        after = [after] if isinstance(after, str) else list(after)
        return after + ([j["init_job"]] if j.get("init_job") else [])

    needed, todo = set(), [n for j in jobs if not st.job(j["name"])["home"] for n in needs(j)]
    while todo:
        n = todo.pop()
        if n in by and n not in needed:
            needed.add(n)
            # Followed through home training jobs too: a job that runs after one often reads that
            # job's own inputs without declaring them (24 Oct 2026: the pilot's post job read
            # data/panel_fineweb.ids on a replacement VM, after only its pretraining job), so every
            # data job upstream of unfinished work runs again. Training jobs never rerun.
            todo.extend(needs(by[n]))
    failed = [j["name"] for j in jobs if st.job(j["name"])["home"] and st.job(j["name"]).get("status") == "FAILED"]
    rerun = [j["name"] for j in jobs if st.job(j["name"])["home"] and "cmd" in j and j["name"] in needed
             and j["name"] not in failed]
    finished = [j["name"] for j in jobs if st.job(j["name"])["home"] and j["name"] not in failed + rerun]
    rc, out = exec_py(f"for n in {finished!r}: open('/content/chat/STATUS_' + n, 'w').write('FINISHED\\n')\n"
                      f"for n in {failed!r}: open('/content/chat/STATUS_' + n, 'w').write('FAILED\\n')\n"
                      "print('HOME_OK')\n")
    if "HOME_OK" not in out:
        log(f"prepare: home jobs not marked: {out.strip()[-120:]}")
        return False
    if rerun:
        log(f"prepare: data jobs rerun on this VM for unfinished dependants: {rerun}")
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
    try:
        for key, src in uploads:
            upload_file(src, f"/content/chat/upload/{key}")
        for key, src, _ in finals:
            upload_file(src, f"/content/chat/upload/{key}")
            log(f"{key}: uploaded for a dependent job")
        # Resume checkpoints of unfinished jobs, smallest first. The first goes up before the boot;
        # the rest one per tick while the VM trains (upload_pending), and jobs_vm.py holds each job
        # until its READY marker. Weights, step and events only: no optimiser state (22 Oct 2026,
        # VMs living 40-60 min against a 50-min 1.9 GB upload); jobs_vm.py resumes those cold.
        pend = sorted((j["name"] for j in jobs if not st.job(j["name"])["home"]
                       and os.path.exists(os.path.join(ART, j["name"], "state.txt"))),
                      key=lambda n: os.path.getsize(os.path.join(ART, n, "model.safetensors")))
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
    names = [k for k, _ in uploads] + [k for k, _, _ in finals]
    moves = [(k, dst) for k, _, dst in finals]
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
    url = f"https://raw.githubusercontent.com/JohnnyTeutonic/paperkiln/{commit}/tools/colab_chat/boot_jobs.sh"
    rc, out = exec_py(
        "import subprocess, urllib.request\n"
        f"open('/content/boot_jobs.sh','wb').write(urllib.request.urlopen('{url}').read())\n"
        f"subprocess.Popen('nohup bash /content/boot_jobs.sh {commit} {jobs_rel} > /dev/null 2>&1 &', shell=True)\n"
        "print('BOOTED')\n")
    if "BOOTED" not in out:
        log(f"boot exec failed: {out.strip()[-160:]}")
        return False
    st.d["booted"] = True
    st.save()
    log("boot launched")
    return True


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


def main():
    global NAME, ART, HOME
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--commit", required=True)
    ap.add_argument("--jobs", required=True, help="jobs JSON, path relative to the paperkiln repo root")
    ap.add_argument("--max-hours", type=float, default=23.5)
    ap.add_argument("--name", required=True, help="Colab session name (this run's own)")
    ap.add_argument("--art", required=True, help="local artifact directory")
    ap.add_argument("--state-dir", required=True, help="runner state, pid file and log")
    ap.add_argument("--upload", action="append", default=[], help="KEY=LOCAL_PATH, sent to /content/chat/upload/KEY")
    a = ap.parse_args()
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
    open(pid, "w").write(str(os.getpid()))
    st = State()
    deadline = time.time() + a.max_hours * 3600
    if listed() is None and not st.d.get("endpoint"):
        if not launch(st, a.commit, a.jobs, jobs, uploads):
            time.sleep(TICK)
    try:
        while time.time() < deadline:
            sh([COLPY, ADOPT, "--refresh"], 180)
            if listed() is None:
                ep = st.d.get("endpoint")
                rc, out = sh([COLPY, ADOPT, "--adopt", f"{ep}={NAME}"], 180) if ep else (1, "")
                if not (rc == 0 and "adopted" in out):
                    # The listing comes back empty now and then while the VM is
                    # fine, and a relaunch begins with `colab stop`: on 20 Oct
                    # one empty listing destroyed a live VM's unrelayed work.
                    # Relaunch only after three straight ticks with no listing,
                    # no adoption and no answer from the VM itself.
                    probe = vm_status(jobs)
                    st.d["gone_ticks"] = 0 if probe is not None else st.d.get("gone_ticks", 0) + 1
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
                colab("download", "-s", NAME, "/content/chat/boot.log", os.path.join(ART, "boot.log"), timeout=120)
                break
            done = True
            for j in jobs:
                n = j["name"]
                js = st.job(n)
                if r[n] != js["status"]:
                    log(f"{n}: {r[n]}  {r.get(n + '_tail', '')[-120:]}")
                    js["status"] = r[n]
                    st.save()
                if r[n] == "RUNNING" and not js["home"] and "cmd" not in j:
                    relay(st, n, j.get("relay_every_steps", 0))
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
                    colab("download", "-s", NAME, f"/content/chat/{n}_train.log",
                          os.path.join(ART, n, f"{n}_train.log"), timeout=300)
                    js["home"] = True
                    st.save()
                done = done and js["home"]
            if done:
                log("all jobs finished and home")
                break
            time.sleep(TICK)
    finally:
        colab("stop", "-s", NAME, timeout=120)
        log("session stopped; runner exiting")
        os.remove(pid)


if __name__ == "__main__":
    main()
