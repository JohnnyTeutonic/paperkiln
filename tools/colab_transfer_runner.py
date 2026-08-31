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
    try:
        p = subprocess.run(args, capture_output=True, text=True,
                           timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"


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


def upload(session, local, remote, timeout=1800):
    rc, out = sh([COL, "upload", "-s", session, local, remote],
                 timeout=timeout)
    if rc != 0:
        log(f"upload FAILED {local} -> {remote}: {out[-200:]}")
    return rc == 0


def done_runs(local_out):
    return {os.path.basename(d) for d in
            glob.glob(os.path.join(local_out, "runs", "*"))
            if os.path.exists(os.path.join(d, "result.json"))}


def push_resume(session, local_out, out_root):
    """Ship completed run dirs back so mtsweep skips them."""
    runs = done_runs(local_out)
    if not runs:
        return 0
    z = "/tmp/tr_resume.zip"
    if os.path.exists(z):
        os.remove(z)
    shutil.make_archive("/tmp/tr_resume", "zip",
                        os.path.join(local_out, "runs"))
    if not upload(session, z, "/content/tr_resume.zip"):
        return 0
    exec_py(session, (
        "import os, zipfile\n"
        f"d = {out_root + '/runs'!r}\n"
        "os.makedirs(d, exist_ok=True)\n"
        "zipfile.ZipFile('/content/tr_resume.zip').extractall(d)\n"
        "print('RESUME_RESTORED', len(os.listdir(d)))\n"), timeout=600)
    return len(runs)


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
    rc, out = exec_py(session, code, timeout=600)
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
    return len(done_runs(local_out))


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
    rc, out = exec_py(session, code, timeout=900)
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
        "cmd = ('cd /content/microtorch && ' + env + 'nohup python3 "
        "tools/mtsweep.py ' +\n"
        f"       {sweep_rel!r} + ' --mtstudio /content/mtstudio "
        f"--jobs {int(jobs)} --omp {int(omp)}"
        "' + ' >> /content/sweep.log 2>&1 &')\n"
        "subprocess.run(cmd, shell=True)\n"
        "print('SWEEP_LAUNCHED')\n")
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
    push_resume(session, local_out, out_root)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True, help="repo-relative sweep json")
    ap.add_argument("--session", required=True)
    ap.add_argument("--local-out", required=True)
    ap.add_argument("--expect", type=int, required=True)
    ap.add_argument("--gpu", default="T4")
    ap.add_argument("--jobs", type=int, default=1,
                    help="cells run concurrently on the vm (see "
                         "launch_sweep: 4 is the measured L4 optimum)")
    ap.add_argument("--omp", type=int, default=4,
                    help="OMP threads per cell; keep jobs*omp <= vCPUs")
    ap.add_argument("--max-hours", type=float, default=8.0)
    ap.add_argument("--tick", type=int, default=90)
    args = ap.parse_args()

    with open(os.path.join(REPO, "microtorch", args.sweep),
              encoding="utf-8") as f:
        out_root = json.load(f)["out_root"]
    # Cache is keyed BY GPU: a binary built on a T4 (sm_75) is not
    # necessarily loadable on an L4 (sm_89), and silently uploading the
    # wrong one would fail on the vm where it is hard to see.
    cache = os.path.join(os.path.dirname(args.local_out.rstrip("/")),
                         f"mtstudio_cuda_{args.gpu.lower()}")
    os.makedirs(os.path.join(args.local_out, "runs"), exist_ok=True)
    deadline = time.time() + args.max_hours * 3600
    log(f"arm {args.sweep} -> {args.local_out} (expect {args.expect}); "
        f"already done: {len(done_runs(args.local_out))}")

    launched = False
    provisioned = False  # repo + binary + resume state are ON the vm
    prov_fail = 0        # consecutive provisioning failures on this vm
    strikes = 0          # consecutive ticks the sweep was observed dead
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
                prov_fail += 1
                if prov_fail >= 2:
                    log("provisioning failed twice — discarding this vm")
                    sh([COL, "stop", "-s", args.session], timeout=90)
                    prov_fail = 0
                time.sleep(30)
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
        got = relay(args.session, args.local_out, out_root)
        log(f"relayed; local runs {got}/{args.expect}")
    log("deadline reached")
    sh([COL, "stop", "-s", args.session], timeout=90)
    return 1


if __name__ == "__main__":
    sys.exit(main())
