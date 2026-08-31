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
    rc, out = exec_py(session, 'print("TR_ALIVE")\n', timeout=120)
    return rc == 0 and "TR_ALIVE" in out


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


def launch_sweep(session, sweep_rel):
    code = (
        "import subprocess\n"
        "env = ('MICROTORCH_DEVICE=cuda ')\n"
        # GEMM-ONLY (31 Aug 2026). Measured on the VM at this study's real
        # shape, 400 steps: gpu reaches 400 clean; ops OOMs at step 95; res
        # OOMs at step 96; defer corrupts the heap at step 1. The device op
        # set leaks device memory, and NOTHING in the suite runs long
        # enough to see it — bench_b2 does 9 steps, test_step_residency 50,
        # and the leak kills at ~95. All four configs converge to identical
        # losses where they survive, so gemm-only is correct, just slower
        # (~0.9 s/step here vs ~0.7 for res). Tracked in
        # docs/open/BACKLOG.md 4c.
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
        "cmd = ('cd /content/microtorch && ' + env + 'nohup python3 "
        "tools/mtsweep.py ' +\n"
        f"       {sweep_rel!r} + ' --mtstudio /content/mtstudio --jobs 1 "
        "--omp 4 >> /content/sweep.log 2>&1 &')\n"
        "subprocess.run(cmd, shell=True)\n"
        "print('SWEEP_LAUNCHED')\n")
    rc, out = exec_py(session, code, timeout=300)
    return "SWEEP_LAUNCHED" in out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True, help="repo-relative sweep json")
    ap.add_argument("--session", required=True)
    ap.add_argument("--local-out", required=True)
    ap.add_argument("--expect", type=int, required=True)
    ap.add_argument("--gpu", default="T4")
    ap.add_argument("--max-hours", type=float, default=8.0)
    ap.add_argument("--tick", type=int, default=90)
    args = ap.parse_args()

    with open(os.path.join(REPO, "microtorch", args.sweep),
              encoding="utf-8") as f:
        out_root = json.load(f)["out_root"]
    cache = os.path.join(os.path.dirname(args.local_out.rstrip("/")),
                         "mtstudio_cuda")
    os.makedirs(os.path.join(args.local_out, "runs"), exist_ok=True)
    deadline = time.time() + args.max_hours * 3600
    log(f"arm {args.sweep} -> {args.local_out} (expect {args.expect}); "
        f"already done: {len(done_runs(args.local_out))}")

    launched = False
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
            if not ensure_repo_and_data(args.session):
                log("repo/data setup failed; retrying")
                continue
            if not ensure_binary(args.session, cache, args.sweep):
                continue
            push_resume(args.session, args.local_out, out_root)
        if not launched:
            launched = launch_sweep(args.session, args.sweep)
            log(f"sweep launched: {launched}")
        time.sleep(args.tick)
        got = relay(args.session, args.local_out, out_root)
        log(f"relayed; local runs {got}/{args.expect}")
    log("deadline reached")
    sh([COL, "stop", "-s", args.session], timeout=90)
    return 1


if __name__ == "__main__":
    sys.exit(main())
