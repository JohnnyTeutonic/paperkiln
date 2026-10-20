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


def upload_file(src, dst):
    """Upload in 45 MB parts named dst.partNN (jobs_vm.py reassembles job
    checkpoints; whole input files are reassembled by the exec below)."""
    with open(src, "rb") as fh:
        i = 0
        while True:
            buf = fh.read(CHUNK)
            if not buf:
                break
            part = os.path.join(HOME, f"up.part{i:02d}")
            open(part, "wb").write(buf)
            rc, _ = colab("upload", "-s", NAME, part, f"{dst}.part{i:02d}", timeout=1500)
            os.remove(part)
            if rc != 0:
                raise RuntimeError(f"upload of {src} part {i} failed")
            i += 1


def relay(st, name):
    """Download the job's state.txt; if its step advanced, the checkpoint too."""
    art = os.path.join(ART, name)
    os.makedirs(art, exist_ok=True)
    js = st.job(name)
    tmp = os.path.join(art, "state.txt.new")
    rc, _ = colab("download", "-s", NAME, f"/content/chat/{name}/state.txt", tmp, timeout=200)
    if rc != 0 or not os.path.exists(tmp):
        return
    step = int(open(tmp).readline().strip() or 0)
    if step <= js["relayed_step"]:
        os.remove(tmp)
        return
    ok = True
    for f in ("model.safetensors", "optim.safetensors", "events.jsonl"):
        rc, _ = colab("download", "-s", NAME, f"/content/chat/{name}/{f}", os.path.join(art, f + ".new"), timeout=1800)
        ok = ok and rc == 0
    tmp2 = os.path.join(art, "state.txt.check")
    rc, _ = colab("download", "-s", NAME, f"/content/chat/{name}/state.txt", tmp2, timeout=200)
    if rc == 0 and os.path.exists(tmp2):
        ok = ok and int(open(tmp2).readline().strip() or 0) == step
        os.remove(tmp2)
    else:
        ok = False
    if not ok:
        log(f"{name}: relay at step {step} incomplete or overtaken by a save; will retry")
        return
    for f in ("model.safetensors", "optim.safetensors", "events.jsonl"):
        os.replace(os.path.join(art, f + ".new"), os.path.join(art, f))
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
        rc, _ = colab("download", "-s", NAME, f"/content/chat/{f}", os.path.join(art, os.path.basename(f)), timeout=1800)
        ok = ok and rc == 0
    # Optional extra exports (e.g. q8_0_all) are fetched best-effort.
    colab("download", "-s", NAME, f"/content/chat/{name}/{model_name}_q8_0_all.gguf",
          os.path.join(art, f"{model_name}_q8_0_all.gguf"), timeout=1800)
    colab("download", "-s", NAME, f"/content/chat/{name}/{model_name}.ir.json",
          os.path.join(art, f"{model_name}.ir.json"), timeout=300)
    if ok:
        st.job(name)["home"] = True
        st.save()
        log(f"{name}: final outputs downloaded")


def launch(st, commit, jobs_rel, jobs, uploads):
    colab("stop", "-s", NAME, timeout=120)
    rc, out = colab("new", "-s", NAME, "--gpu", "L4", timeout=600)
    if rc != 0:
        log(f"no L4 ({out.strip()[-120:]}); retry later")
        return False
    st.d["endpoint"] = listed()
    st.d["launches"] += 1
    st.save()
    log(f"session up, endpoint {st.d['endpoint']} (launch {st.d['launches']})")
    exec_py("import os\nos.makedirs('/content/chat/upload', exist_ok=True)\nprint('OK')\n")
    try:
        for key, src in uploads:
            upload_file(src, f"/content/chat/upload/{key}")
        # Job checkpoints relayed earlier (unfinished jobs only; jobs_vm.py restores them).
        for j in jobs:
            art = os.path.join(ART, j["name"])
            if st.job(j["name"])["home"] or not os.path.exists(os.path.join(art, "state.txt")):
                continue
            for f in ("model.safetensors", "optim.safetensors"):
                upload_file(os.path.join(art, f), f"/content/chat/upload/{j['name']}/{f}")
            for f in ("state.txt", "events.jsonl"):
                colab("upload", "-s", NAME, os.path.join(art, f), f"/content/chat/upload/{j['name']}/{f}", timeout=600)
            log(f"{j['name']}: checkpoint uploaded for resume")
    except RuntimeError as e:
        log(str(e))
        return False
    # Reassemble the whole-file inputs.
    names = [k for k, _ in uploads]
    rc, out = exec_py(
        "import os, shutil\n"
        f"for k in {names!r}:\n"
        "    d = '/content/chat/upload'\n"
        "    parts = sorted(p for p in os.listdir(d) if p.startswith(k + '.part'))\n"
        "    with open(os.path.join(d, k), 'wb') as dst:\n"
        "        for p in parts:\n"
        "            shutil.copyfileobj(open(os.path.join(d, p), 'rb'), dst)\n"
        "    for p in parts: os.remove(os.path.join(d, p))\n"
        "    print(k, os.path.getsize(os.path.join(d, k)))\n"
        "print('INPUTS_OK')\n")
    if "INPUTS_OK" not in out:
        log(f"input reassembly failed: {out.strip()[-160:]}")
        return False
    log("inputs: " + " ".join(l for l in out.splitlines() if l and not l.startswith("[")))
    url = f"https://raw.githubusercontent.com/JohnnyTeutonic/paperkiln/{commit}/tools/colab_chat/boot_jobs.sh"
    rc, out = exec_py(
        "import subprocess, urllib.request\n"
        f"open('/content/boot_jobs.sh','wb').write(urllib.request.urlopen('{url}').read())\n"
        f"subprocess.Popen('nohup bash /content/boot_jobs.sh {commit} {jobs_rel} > /dev/null 2>&1 &', shell=True)\n"
        "print('BOOTED')\n")
    if "BOOTED" not in out:
        log(f"boot exec failed: {out.strip()[-160:]}")
        return False
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
                    log("session gone (reclaimed?); relaunching from the relayed checkpoints")
                    if not launch(st, a.commit, a.jobs, jobs, uploads):
                        time.sleep(TICK)
                        continue
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
                if r[n] in ("RUNNING", "FINISHED") and not js["home"]:
                    relay(st, n)
                if r[n] == "FINISHED" and not js["home"]:
                    fetch_final(st, n, j.get("model_name", n))
                # FAILED counts only when seen on two ticks running: a status
                # read mid-reset (a job being restarted by hand) is not final.
                js["failed_seen"] = js.get("failed_seen", 0) + 1 if r[n] == "FAILED" else 0
                if r[n] == "FAILED" and js["failed_seen"] >= 2 and not js["home"]:
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
