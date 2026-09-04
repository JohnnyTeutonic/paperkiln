#!/usr/bin/env python3
"""Reclaim-resilient Colab driver for an mtsweep arm (transfer_s1 and
any later ladder rung).

    python3 tools/colab_transfer_runner.py \
        --sweep experiments/transfer_s1/sweep_bridge.json \
        --session tr-bridge --local-out /mnt/c/ml_artifacts/transfer/bridge \
        --expect 10 [--gpu T4] [--max-hours 6]

WHY THIS EXISTS. Colab reclaims VMs on its own schedule; the transfer
study is 154 runs. supervisor.py's admission rule is the law here — a
lane may only run if its checkpoint interval is well under the reclaim
interval AND the checkpoint is RELAYED OFF the VM and pushed back on
relaunch. mtsweep is resumable (a cell with result.json is skipped), so
the unit of progress is one run (~5 min) and the relay unit is the run
directory. A reclaim therefore costs at most the run in flight.

The expensive part of a relaunch is not the runs, it is the ~10 minute
CUDA build. So the binary is BUILT ONCE and cached locally; every later
launch uploads the cached binary instead of rebuilding. Reclaim cost
drops from ~10 min to ~1 min.

Everything is idempotent: re-running this command after any failure
resumes from whatever is already in --local-out.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import time

COL = os.path.expanduser("~/.local/bin/colab")
REPO = "/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete"
CORPUS = f"{REPO}/transformer_cpp/data/tinystories-txt/train-0-small.txt"
VOCAB = f"{REPO}/transformer_cpp/releases/chat7b.gguf"
PAPERKILN = "https://github.com/JohnnyTeutonic/paperkiln.git"
COALFIRE = "https://github.com/JohnnyTeutonic/coalfire.cpp.git"


def sh(args, timeout=300):
    """Run a CLI call under a timeout that CANNOT be defeated.

    `subprocess.run(capture_output=True, timeout=N)` hands the child a
    PIPE. On timeout it kills the DIRECT child, then blocks in
    communicate() draining pipes that any surviving GRANDCHILD still
    holds open. The colab CLI spawns exactly such grandchildren, and on
    1 Sep 2026 this hung the arm-S driver for 2h13m inside ONE tick --
    against a 180s timeout. While it hung, the vm was reclaimed, nobody
    noticed, and ~3 hours of L4 compute produced nothing bankable.

    Writing output to a temp FILE shares no pipe with anyone, so the
    wait really ends when we kill. We also start a new process group and
    kill the whole group, so grandchildren die with the parent.
    """
    import tempfile
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8",
                                errors="replace") as out:
        try:
            p = subprocess.Popen(args, stdout=out,
                                 stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL,
                                 start_new_session=True)
        except Exception as exc:                     # noqa: BLE001
            return 127, f"SPAWN_FAILED {exc}"
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except Exception:                        # noqa: BLE001
                p.kill()
            try:
                p.wait(timeout=30)
            except Exception:                        # noqa: BLE001
                pass
            out.seek(0)
            return 124, "TIMEOUT " + out.read()[-1500:]
        out.seek(0)
        return rc, out.read()


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def exec_py(session, code, timeout=300):
    """Run python source on the VM (written to a temp file — inline
    quoting through several shells is how these pipelines break)."""
    path = f"/tmp/_tr_{abs(hash(code)) % 10**8}.py"
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    return sh([COL, "exec", "-s", session, "-f", path,
               "--timeout", str(min(timeout, 900))], timeout=timeout + 60)


def alive(session):
    """Is the session still registered?

    This used to EXECUTE code on the vm and call a timeout death. That
    was actively destructive: `colab exec` was measured taking 5-15 min
    against a vm running 4 cells, so a merely BUSY vm failed the probe,
    the driver concluded it had died, and new_session() stopped it --
    killing four in-flight runs to 'recover' a healthy session. Observed
    1 Sep 2026: a session created 02:36 was destroyed at 02:42 this way,
    which read in the log as a 6-minute session lifetime.

    The control-plane listing needs nothing from the vm's kernel, so
    load cannot make it lie. Whether the WORK is alive is a separate
    question, and sweep_alive() answers that one with its own tolerance.
    """
    rc, out = sh([COL, "sessions"], timeout=180)
    if rc != 0:
        return True          # control plane unreachable: assume alive,
                             # never destroy a session on our own flakiness
    return f"[{session}]" in out


def new_session(session, gpu):
    sh([COL, "stop", "-s", session], timeout=90)
    rc, out = sh([COL, "new", "-s", session, "--gpu", gpu], timeout=600)
    ok = rc == 0 and "READY" in out
    log(f"session {session}: {'READY' if ok else 'FAILED ' + out[-200:]}")
    return ok


UPLOAD_CHUNK = 32 * 1024 * 1024


def upload(session, local, remote, timeout=1800):
    """Upload with retries; files above UPLOAD_CHUNK go up in pieces.

    A single 111 MB `colab upload` died with an SSL EOF on 4 Sep 2026
    (the partial-checkpoint push for the resume probe), while the 1.5 MB
    binary and a 127 MB DOWNLOAD both worked. Large uploads are therefore
    split into 32 MB parts, each retried, and reassembled on the vm with
    a byte-count check, so a flaky transport costs a retry, not a resume.
    """
    size = os.path.getsize(local)
    if size > UPLOAD_CHUNK:
        return upload_chunked(session, local, remote, size)
    for attempt in range(3):
        rc, out = sh([COL, "upload", "-s", session, local, remote],
                     timeout=timeout)
        if rc == 0:
            return True
        log(f"upload FAILED (attempt {attempt + 1}/3) {local} -> {remote}: "
            f"{out[-160:]}")
        time.sleep(10)
    return False


def upload_chunked(session, local, remote, size):
    n = (size + UPLOAD_CHUNK - 1) // UPLOAD_CHUNK
    parts = "/tmp/tr_chunks"
    shutil.rmtree(parts, ignore_errors=True)
    os.makedirs(parts)
    with open(local, "rb") as f:
        for i in range(n):
            with open(os.path.join(parts, f"part{i:04d}"), "wb") as p:
                p.write(f.read(UPLOAD_CHUNK))
    log(f"chunked upload: {size / 1e6:.0f} MB in {n} parts -> {remote}")
    for i in range(n):
        lp = os.path.join(parts, f"part{i:04d}")
        rp = f"{remote}.part{i:04d}"
        ok = False
        for attempt in range(4):
            rc, out = sh([COL, "upload", "-s", session, lp, rp], timeout=900)
            if rc == 0:
                ok = True
                break
            log(f"chunk {i + 1}/{n} FAILED (attempt {attempt + 1}/4): {out[-160:]}")
            time.sleep(10)
        if not ok:
            shutil.rmtree(parts, ignore_errors=True)
            return False
    shutil.rmtree(parts, ignore_errors=True)
    rc, out = exec_py(session, (
        "import os\n"
        f"remote = {remote!r}\n"
        f"n = {n}\n"
        "with open(remote, 'wb') as w:\n"
        "    for i in range(n):\n"
        "        p = f'{remote}.part{i:04d}'\n"
        "        with open(p, 'rb') as r:\n"
        "            w.write(r.read())\n"
        "        os.remove(p)\n"
        "got = os.path.getsize(remote)\n"
        f"print('ASSEMBLED_OK' if got == {size} else f'ASSEMBLED_BAD {{got}}')\n"),
        timeout=600)
    if "ASSEMBLED_OK" not in out:
        log(f"chunked upload reassembly failed: {out[-160:]}")
        return False
    return True


def done_runs(local_out):
    return {os.path.basename(d) for d in
            glob.glob(os.path.join(local_out, "runs", "*"))
            if os.path.exists(os.path.join(d, "result.json"))}


PARTIAL_FILES = ("state.txt", "model.safetensors", "optim.safetensors",
                 "events.jsonl")


def partial_runs(local_out):
    """name -> checkpoint step, for every partial checkpoint held locally.
    A partial is a run that died mid-flight (the 60-minute prune) whose
    last complete checkpoint we brought home; mtstudio resumes from it
    once it is back under <out_root>/runs/<name>/ on the next vm."""
    out = {}
    for d in glob.glob(os.path.join(local_out, "partial", "*")):
        st = os.path.join(d, "state.txt")
        if not os.path.exists(st):
            continue
        try:
            step = int(open(st, encoding="utf-8").readline().strip())
        except (OSError, ValueError):
            continue
        if step > 0:
            out[os.path.basename(d)] = step
    return out


def push_resume(session, local_out, out_root):
    """Ship completed run dirs back so mtsweep skips them, and partial
    checkpoints back so mtstudio RESUMES them. Returns (done, partial)."""
    runs = done_runs(local_out)
    partial = {n: s for n, s in partial_runs(local_out).items() if n not in runs}
    if not runs and not partial:
        return 0, 0
    stage = "/tmp/tr_resume_stage"
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(stage)
    for n in runs:
        shutil.copytree(os.path.join(local_out, "runs", n), os.path.join(stage, n))
    for n in partial:
        shutil.copytree(os.path.join(local_out, "partial", n), os.path.join(stage, n))
    z = "/tmp/tr_resume.zip"
    if os.path.exists(z):
        os.remove(z)
    shutil.make_archive("/tmp/tr_resume", "zip", stage)
    if not upload(session, z, "/content/tr_resume.zip"):
        return 0, 0
    exec_py(session, (
        "import os, zipfile\n"
        f"d = {out_root + '/runs'!r}\n"
        "os.makedirs(d, exist_ok=True)\n"
        "zipfile.ZipFile('/content/tr_resume.zip').extractall(d)\n"
        "print('RESUME_RESTORED', len(os.listdir(d)))\n"), timeout=900)
    if partial:
        log("partial checkpoints pushed for resume: " +
            ", ".join(f"{n}@{s}" for n, s in sorted(partial.items())))
    return len(runs), len(partial)


def relay_partial(session, local_out, out_root, seen):
    """Bring home the latest COMPLETE checkpoint of every unfinished run.

    Only runs whose checkpoint is newer than what we already hold are
    shipped (`seen` maps name -> step we last relayed). A checkpoint is
    taken only if state.txt is at least 30 s old and both safetensors
    files are older than it: mtstudio writes model, then optim, then
    state.txt last, so a state.txt that has been still for 30 s means the
    set is complete. Without this guard a zip could capture a truncated
    model file and the resume would load garbage.
    """
    code = (
        "import glob, os, time, zipfile\n"
        f"root = {out_root + '/runs'!r}\n"
        f"seen = {seen!r}\n"
        "now = time.time()\n"
        "picked = []\n"
        "for d in sorted(glob.glob(os.path.join(root, '*'))):\n"
        "    name = os.path.basename(d)\n"
        "    if os.path.exists(os.path.join(d, 'result.json')):\n"
        "        continue\n"
        "    st = os.path.join(d, 'state.txt')\n"
        "    mo = os.path.join(d, 'model.safetensors')\n"
        "    op = os.path.join(d, 'optim.safetensors')\n"
        "    if not (os.path.exists(st) and os.path.exists(mo) and os.path.exists(op)):\n"
        "        continue\n"
        "    ms = os.path.getmtime(st)\n"
        "    if now - ms < 30 or os.path.getmtime(mo) > ms or os.path.getmtime(op) > ms:\n"
        "        continue\n"
        "    try:\n"
        "        step = int(open(st).readline().strip())\n"
        "    except Exception:\n"
        "        continue\n"
        "    if step <= 0 or step <= seen.get(name, 0):\n"
        "        continue\n"
        "    picked.append((name, step))\n"
        "z = '/content/tr_partial.zip'\n"
        "with zipfile.ZipFile(z, 'w', zipfile.ZIP_STORED) as zf:\n"
        "    for name, step in picked:\n"
        f"        for fn in {PARTIAL_FILES!r}:\n"
        "            p = os.path.join(root, name, fn)\n"
        "            if os.path.exists(p):\n"
        "                zf.write(p, os.path.join(name, fn))\n"
        "print('PARTIAL_READY', len(picked), ' '.join(f'{n}@{s}' for n, s in picked))\n")
    # 900, not 600 (4 Sep 2026): colab exec on a vm running 4 cells was
    # measured at 5-15 min (see alive()). A relay exec killed at 10 min
    # costs the 10 min AND the relay; one that finishes at 12 keeps it.
    rc, out = exec_py(session, code, timeout=900)
    if rc != 0 or "PARTIAL_READY" not in out:
        return {}
    tail = out.split("PARTIAL_READY")[1].split()
    n = int(tail[0])
    if n == 0:
        return {}
    picked = {}
    for tok in tail[1:1 + n]:
        name, step = tok.rsplit("@", 1)
        picked[name] = int(step)
    tmp = "/tmp/tr_partial.zip"
    if os.path.exists(tmp):
        os.remove(tmp)
    rc, _ = sh([COL, "download", "-s", session, "/content/tr_partial.zip", tmp],
               timeout=1800)
    if rc != 0 or not os.path.exists(tmp):
        return {}
    import zipfile
    pdir = os.path.join(local_out, "partial")
    os.makedirs(pdir, exist_ok=True)
    try:
        # Extract to a temp dir first so a broken download cannot
        # half-overwrite a good partial.
        tmpdir = pdir + ".incoming"
        shutil.rmtree(tmpdir, ignore_errors=True)
        zipfile.ZipFile(tmp).extractall(tmpdir)
        for name in picked:
            src = os.path.join(tmpdir, name)
            if os.path.isdir(src):
                dst = os.path.join(pdir, name)
                shutil.rmtree(dst, ignore_errors=True)
                shutil.move(src, dst)
        shutil.rmtree(tmpdir, ignore_errors=True)
    except zipfile.BadZipFile:
        return {}
    return picked


def relay(session, local_out, out_root):
    """Zip finished runs on the VM, bring them home."""
    code = (
        "import glob, os, zipfile\n"
        f"root = {out_root + '/runs'!r}\n"
        "done = [d for d in sorted(glob.glob(os.path.join(root, '*')))\n"
        "        if os.path.exists(os.path.join(d, 'result.json'))]\n"
        "z = '/content/tr_relay.zip'\n"
        "with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as zf:\n"
        "    for d in done:\n"
        "        for fn in ('result.json', 'events.jsonl'):\n"
        "            p = os.path.join(d, fn)\n"
        "            if os.path.exists(p):\n"
        "                zf.write(p, os.path.join(os.path.basename(d), fn))\n"
        "print('RELAY_READY', len(done))\n")
    # 900, not 600 (4 Sep 2026): colab exec on a vm running 4 cells was
    # measured at 5-15 min (see alive()). A relay exec killed at 10 min
    # costs the 10 min AND the relay; one that finishes at 12 keeps it.
    rc, out = exec_py(session, code, timeout=900)
    if rc != 0 or "RELAY_READY" not in out:
        return 0
    n = int(out.split("RELAY_READY")[1].split()[0])
    if n == 0:
        return 0
    tmp = "/tmp/tr_relay.zip"
    if os.path.exists(tmp):
        os.remove(tmp)
    rc, _ = sh([COL, "download", "-s", session, "/content/tr_relay.zip", tmp],
               timeout=900)
    if rc != 0 or not os.path.exists(tmp):
        return 0
    import zipfile
    os.makedirs(os.path.join(local_out, "runs"), exist_ok=True)
    try:
        zipfile.ZipFile(tmp).extractall(os.path.join(local_out, "runs"))
    except zipfile.BadZipFile:
        return 0
    # A run that has finished no longer needs its partial checkpoint.
    for name in done_runs(local_out):
        shutil.rmtree(os.path.join(local_out, "partial", name), ignore_errors=True)
    return len(done_runs(local_out))


BUILD_INPUTS = ["src", "include", "tools/mtstudio.cpp", "tools/parity_model.hpp",
                "CMakeLists.txt"]


def source_fingerprint():
    """12 hex chars identifying the build inputs at origin/master."""
    import hashlib
    ids = []
    for path in BUILD_INPUTS:
        rc, out = sh(["git", "-C", os.path.join(REPO, "microtorch"),
                      "rev-parse", f"origin/master:{path}"], timeout=30)
        ids.append(out.strip() if rc == 0 else f"missing:{path}")
    return hashlib.sha1("\n".join(ids).encode()).hexdigest()[:12]


def ensure_binary(session, cache, sweep_rel):
    """Upload the cached mtstudio if we have one; else build and cache."""
    if os.path.exists(cache):
        log("uploading cached mtstudio (skips the ~10 min build)")
        if upload(session, cache, "/content/mtstudio"):
            exec_py(session, ("import os\n"
                              "os.chmod('/content/mtstudio', 0o755)\n"
                              "print('BIN_OK')\n"), timeout=120)
            return True
    log("no cached binary — building on the VM (~10 min)")
    code = (
        "import subprocess\n"
        "cmds = [\n"
        "  'cd /content && rm -rf microtorch transformer_cpp',\n"
        f"  'cd /content && git clone --depth 1 -b master {PAPERKILN} microtorch',\n"
        f"  'cd /content && git clone --depth 1 {COALFIRE} transformer_cpp',\n"
        "  'mkdir -p /content/microtorch/build_cuda',\n"
        "  'cd /content/microtorch/build_cuda && cmake .. "
        "-DCMAKE_BUILD_TYPE=Release -DMICROTORCH_CUDA=ON > cmake.log 2>&1',\n"
        "  'cd /content/microtorch/build_cuda && make microtorch mtstudio "
        "-j$(nproc) > build.log 2>&1',\n"
        "  'cp /content/microtorch/build_cuda/mtstudio /content/mtstudio',\n"
        "]\n"
        "for c in cmds:\n"
        "    r = subprocess.run(c, shell=True)\n"
        "    if r.returncode:\n"
        "        subprocess.run('tail -30 /content/microtorch/build_cuda/"
        "build.log', shell=True)\n"
        "        raise SystemExit('BUILD_FAILED: ' + c)\n"
        "print('BUILD_OK')\n")
    rc, out = exec_py(session, code, timeout=2400)
    if "BUILD_OK" not in out:
        log(f"build failed: {out[-400:]}")
        return False
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    rc, _ = sh([COL, "download", "-s", session, "/content/mtstudio", cache],
               timeout=900)
    log(f"binary cached at {cache}" if rc == 0 else "binary cache FAILED")
    return True


def ensure_repo_and_data(session):
    """Repo (for mtsweep.py + the sweep spec) and the two data files."""
    code = (
        "import os, subprocess\n"
        "need_repo = not os.path.exists('/content/microtorch/tools/mtsweep.py')\n"
        "if need_repo:\n"
        "    subprocess.run('cd /content && rm -rf microtorch && git clone "
        f"--depth 1 -b master {PAPERKILN} microtorch', shell=True)\n"
        "os.makedirs('/content/data', exist_ok=True)\n"
        "print('HAVE_CORPUS', os.path.exists('/content/data/corpus.txt'))\n"
        "print('HAVE_VOCAB', os.path.exists('/content/data/chat7b.gguf'))\n"
        "print('REPO_OK', os.path.exists('/content/microtorch/tools/mtsweep.py'))\n")
    # 300s, not 900. A shallow clone takes ~80s on a healthy vm; the long
    # timeout only meant a HUNG vm cost 16 min to notice (and 3 strikes
    # = 48 min to discard). `colab exec` hangs indefinitely on some vms
    # even when they are otherwise idle — observed 1 Sep 02:54 and 02:31.
    rc, out = exec_py(session, code, timeout=300)
    if "HAVE_CORPUS True" not in out:
        upload(session, CORPUS, "/content/data/corpus.txt")
    if "HAVE_VOCAB True" not in out:
        upload(session, VOCAB, "/content/data/chat7b.gguf")
    return "REPO_OK True" in out


def launch_sweep(session, sweep_rel, jobs=1, omp=4):
    code = (
        "import subprocess\n"
        "env = ('MICROTORCH_DEVICE=cuda MICROTORCH_DEVICE_OPS=1 ')\n"
        # DEVICE OP SET, re-enabled 31 Aug 2026 at commit 697e281. The leak
        # that forced gemm-only is FIXED: In::owned was clobbered by member
        # initialization order, so ~In never freed and every operand of
        # every device op leaked (156 MiB/step, OOM at step 95). Re-measured
        # on a T4 at this study's exact shape after the fix:
        #     ops       400 steps, memory FLAT at 173 MiB, 0.84 s/step
        #     gemm-only 200 steps, memory FLAT at 143 MiB, 1.10 s/step
        # So the op set is now both stable and 1.31x faster. test_cuda_ops
        # leg 8 (200 tapes, device memory flat) is the regression guard.
        # DEFER_DOWNLOADS is deliberately OFF (31 Aug 2026). mtstudio's own
        # training loop crashes under deferral with glibc heap corruption at
        # step 1 — the dying-temporary class again, in a path no test covers:
        # B2.3's validation exercised test_cuda_ops and bench_b2, never
        # mtstudio. Measured on the VM at this study's real shape: res, ops
        # and gpu all run clean and converge to an IDENTICAL loss
        # (4.962438106536865 at step 30) while defer dies at step 1.
        # Residency without deferral is a validated configuration and keeps
        # most of the speedup, so the study runs on it; the defer bug is
        # tracked separately in docs/open/BACKLOG.md.
        # CONCURRENCY (1 Sep 2026). One cell uses ~1.2 cores and ~180 MiB
        # of GPU, so a run is limited by how many cells share a vm, not by
        # per-run speed. Measured on an L4 (12 vCPU), 200 steps per cell:
        #     1 cell  OMP=4  0.68 s/step  1.47 steps/s   41 min/run
        #     4 cells OMP=2  0.80 s/step  5.00 steps/s   48 min/run
        #     8 cells OMP=1  1.19 s/step  6.73 steps/s   71 min/run
        # 4 is the operating point: 3.4x the throughput while each run
        # still lands INSIDE a vm lifetime. That second property is the
        # binding one -- only completed runs are banked, so a 71-min run
        # against a ~52-min reclaim interval banks nothing at all.
        # Kill any sweep already on this vm first. Relaunching happens
        # on re-provision and on driver restart, and a second mtsweep
        # would race the first over the same output directories.
        "subprocess.run('pkill -f mtsweep.py; pkill -f mtstudio', "
        "shell=True)\n"
        "import time as _t; _t.sleep(2)\n"
        # DETACHED LAUNCH (4 Sep 2026). The old form, `subprocess.run(cmd +
        # ' &', shell=True)`, let mtsweep inherit the exec's stdin and the
        # kernel's pipe fds, so `colab exec` sometimes did not return until
        # its timeout even though the sweep was running. The driver then
        # logged 'sweep launched: False', and its NEXT attempt pkill-ed the
        # sweep it had actually started. Arm M, second L4 session: pushed
        # 19:47, launch 'failed' at 19:53 and 20:08, succeeded 20:10, pruned
        # 20:36 -- 26 minutes of training in a 61-minute session. Popen with
        # every fd on /dev/null and its own session cannot hold the pipe.
        "cmd = ('cd /content/microtorch && ' + env + 'exec python3 "
        "tools/mtsweep.py ' +\n"
        f"       {sweep_rel!r} + ' --mtstudio /content/mtstudio "
        f"--jobs {int(jobs)} --omp {int(omp)}"
        "' + ' >> /content/sweep.log 2>&1')\n"
        "subprocess.Popen(cmd, shell=True, stdin=subprocess.DEVNULL,\n"
        "                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,\n"
        "                 close_fds=True, start_new_session=True)\n"
        "_t.sleep(5)\n"
        "r = subprocess.run('pgrep -f mtsweep.py', shell=True, capture_output=True, text=True)\n"
        "print('SWEEP_LAUNCHED' if r.stdout.strip() else 'SWEEP_NOT_RUNNING')\n")
    rc, out = exec_py(session, code, timeout=300)
    return "SWEEP_LAUNCHED" in out


def sweep_alive(session):
    """Is the sweep ACTUALLY running on the VM right now?

    `launched` is only a belief formed once, and a backgrounded nohup
    reports success even when the command dies immediately. Worse, a
    session can answer `alive()` from a FRESH vm after a reclaim, with
    the repo, the binary and the sweep all gone — the driver then relays
    an empty directory forever while holding a GPU. Observed on the
    bridge arm, 31 Aug 2026: 'sweep launched: True' at 23:32, and at
    23:40 the vm had no mtsweep process, no /content/sweep.log and 0 MiB
    of GPU memory in use. Belief is not evidence; look at the process.
    """
    rc, out = exec_py(session, (
        "import os, subprocess\n"
        "r = subprocess.run(\"ps aux | grep -E 'mtsweep|mtstudio' | "
        "grep -v grep\", shell=True, capture_output=True, text=True)\n"
        "n = len([l for l in r.stdout.splitlines() if l.strip()])\n"
        "print('PROCS', n)\n"
        "print('REPO', os.path.exists('/content/microtorch/tools/mtsweep.py'))\n"
        "print('BIN', os.path.exists('/content/mtstudio'))\n"), timeout=180)
    if rc != 0 or "PROCS" not in out:
        return None                      # inconclusive: do not act on it
    return "PROCS 0" not in out and "REPO True" in out and "BIN True" in out


def full_setup(session, cache, sweep_rel, local_out, out_root):
    if not ensure_repo_and_data(session):
        return False
    if not ensure_binary(session, cache, sweep_rel):
        return False
    nd, npart = push_resume(session, local_out, out_root)
    log(f"resume state pushed: {nd} finished, {npart} partial")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True, help="repo-relative sweep json")
    ap.add_argument("--session", required=True)
    ap.add_argument("--local-out", required=True)
    ap.add_argument("--expect", type=int, required=True)
    ap.add_argument("--gpu", default="L4",
                    help="L4 is the study GPU (PREREGISTRATION clarification 5: "
                         "every claim-carrying arm on L4, no mixing). T4 is a "
                         "2-vCPU probe machine; pass it only for probes.")
    ap.add_argument("--jobs", type=int, default=1,
                    help="cells run concurrently on the vm (see "
                         "launch_sweep: 4 is the measured L4 optimum)")
    ap.add_argument("--omp", type=int, default=4,
                    help="OMP threads per cell; keep jobs*omp <= vCPUs")
    ap.add_argument("--max-hours", type=float, default=8.0)
    ap.add_argument("--tick", type=int, default=90)
    ap.add_argument("--partial-every", type=int, default=600,
                    help="seconds between partial-checkpoint relays "
                         "(each one moves ~100 MB per running cell; "
                         "match it to checkpoint_every in the sweep)")
    args = ap.parse_args()

    with open(os.path.join(REPO, "microtorch", args.sweep),
              encoding="utf-8") as f:
        out_root = json.load(f)["out_root"]
    # Cache is keyed BY GPU AND SOURCE: a binary built on a T4 (sm_75) is
    # not necessarily loadable on an L4 (sm_89), and a binary built from an
    # older tree silently lacks whatever was fixed since (the 4 Sep 2026
    # events-trim fix would never have reached Colab under a GPU-only key).
    # The fingerprint is taken from origin/master, which is what the vm
    # clones, so an unpushed local change cannot mislabel a cache entry.
    cache = os.path.join(os.path.dirname(args.local_out.rstrip("/")),
                         f"mtstudio_cuda_{args.gpu.lower()}_{source_fingerprint()}")
    os.makedirs(os.path.join(args.local_out, "runs"), exist_ok=True)

    # A HALTED file means this arm has been shown it CANNOT complete as
    # configured, and launching it would create a session, upload a binary,
    # bank zero runs, and burn units until something killed it. Arm M did
    # exactly that on 2 Sep at 06:35 -- a scheduled shift saw "no driver
    # alive, arm incomplete", relaunched per its instructions, and spent a
    # session for nothing. Documentation in NOW.md did not prevent it,
    # because the thing reading the instructions was not reading NOW.md.
    # So the refusal lives here, next to the launch it has to stop.
    halt = os.path.join(args.local_out, "HALTED")
    if os.path.exists(halt):
        with open(halt, encoding="utf-8", errors="replace") as f:
            why = f.read().strip()
        log(f"REFUSING TO LAUNCH: {args.local_out}/HALTED is present")
        for line in why.split("\n"):
            log(f"  {line}")
        log("delete that file if the arm has been reconfigured.")
        return 2

    deadline = time.time() + args.max_hours * 3600
    log(f"arm {args.sweep} -> {args.local_out} (expect {args.expect}); "
        f"already done: {len(done_runs(args.local_out))}")

    launched = False
    provisioned = False  # repo + binary + resume state are ON the vm
    prov_fail = 0        # consecutive provisioning failures on this vm
    strikes = 0          # consecutive ticks the sweep was observed dead
    # Partial checkpoints: what we hold, by step, so relays only move
    # NEW checkpoints; and when the last relay happened.
    seen_partial = partial_runs(args.local_out)
    if seen_partial:
        log("holding partial checkpoints: " +
            ", ".join(f"{n}@{s}" for n, s in sorted(seen_partial.items())))
    last_partial = 0.0
    while time.time() < deadline:
        n = len(done_runs(args.local_out))
        if n >= args.expect:
            log(f"ARM COMPLETE: {n}/{args.expect} runs banked locally")
            sh([COL, "stop", "-s", args.session], timeout=90)
            return 0
        if not alive(args.session):
            log("session not alive — (re)creating")
            if not new_session(args.session, args.gpu):
                time.sleep(120)
                continue
            launched = False
            provisioned = False
            strikes = 0
        # ADOPT an already-healthy sweep before touching anything. The
        # driver gets restarted (a patch, a crash) while a vm is happily
        # running cells, and it starts with provisioned=False. Without
        # this it would re-provision -- whose setup exec then times out
        # against the busy vm, which the escalation reads as a sick vm
        # and DISCARDS, throwing away a working wave. Observed 1 Sep
        # 02:52. If the repo, binary and processes are all there, the vm
        # is already doing the job: leave it alone.
        if not provisioned and not launched:
            if sweep_alive(args.session) is True:
                log("adopting the sweep already running on this vm")
                provisioned = launched = True
                prov_fail = 0
        # Provisioning is tracked SEPARATELY from session liveness. A
        # setup can fail on a live session (a repo clone timed out at
        # 16 min on a fresh vm, 1 Sep 2026); the old code then fell
        # straight through to launch_sweep and started a sweep on a vm
        # with no repo, and only the strike counter eventually undid it.
        # Never launch onto an unprovisioned vm.
        if not provisioned:
            provisioned = full_setup(args.session, cache, args.sweep,
                                     args.local_out, out_root)
            log(f"provisioned: {provisioned}")
            if not provisioned:
                # Don't grind on a sick vm. Two failures and we throw it
                # away and ask for another; retrying a vm whose git clone
                # hangs just burns the session for as long as it lives.
                # Three strikes, and only after a wait long enough that
                # transient exec latency on a BUSY vm cannot masquerade
                # as a sick one — that mistake discards working waves.
                prov_fail += 1
                if prov_fail >= 3:
                    log("provisioning failed 3x — discarding this vm")
                    sh([COL, "stop", "-s", args.session], timeout=90)
                    prov_fail = 0
                time.sleep(120)
                continue
            prov_fail = 0
        elif launched:
            # Session answers — but is the WORK alive? A reclaim can hand
            # back a fresh, empty vm that passes alive(). Two strikes, so
            # one flaky exec never triggers a rebuild.
            state = sweep_alive(args.session)
            if state is False:
                strikes += 1
                log(f"session up but sweep NOT running (strike {strikes}/2)")
                if strikes >= 2:
                    log("re-provisioning the vm and relaunching the sweep")
                    provisioned = False
                    launched = False
                    strikes = 0
                    continue
            elif state is True:
                strikes = 0
        if not launched:
            launched = launch_sweep(args.session, args.sweep,
                                    args.jobs, args.omp)
            log(f"sweep launched: {launched}")
        time.sleep(args.tick)
        t_tick = time.time()
        got = relay(args.session, args.local_out, out_root)
        # relay() returns 0 on FAILURE as well as on "nothing new", which
        # used to print a bogus "local runs 0/N" and hide relay trouble.
        # Report the truth: count what is actually on disk.
        on_disk = len(done_runs(args.local_out))
        slow = time.time() - t_tick
        note = f"  [relay {'ok' if got else 'FAILED/empty'}]"
        if slow > 300:
            note += f"  [SLOW TICK {slow/60:.1f} min]"
        log(f"local runs {on_disk}/{args.expect}{note}")
        # Partial-checkpoint relay: the insurance against the 60-minute
        # prune. Every --partial-every seconds bring home each running
        # cell's newest complete checkpoint; if the vm dies, push_resume
        # on the next vm restores it and mtstudio picks up mid-run.
        if launched and time.time() - last_partial >= args.partial_every:
            last_partial = time.time()
            picked = relay_partial(args.session, args.local_out, out_root,
                                   seen_partial)
            if picked:
                seen_partial.update(picked)
                log("partial checkpoints relayed: " +
                    ", ".join(f"{n}@{s}" for n, s in sorted(picked.items())))
    log("deadline reached")
    sh([COL, "stop", "-s", args.session], timeout=90)
    return 1


if __name__ == "__main__":
    sys.exit(main())
