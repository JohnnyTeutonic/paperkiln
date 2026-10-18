#!/usr/bin/env python3
"""Reclaim-resilient Colab driver for the chat base pretraining.

    python3 tools/colab_chat/runner.py --commit SHA [--max-hours 20]

Run in WSL as a foreground process (AGENTS.md section 4). It is separate from
tools/colab_supervisor/supervisor.py on purpose: its own session name
(pk-chat), state directory and pid file, so it can never collide with
another driver or reap a session that is not its own. It follows that
supervisor's rules:

- every colab call has a hard timeout;
- the session key is refreshed each tick (keys expire hourly);
- a dropped session is re-adopted by its own recorded endpoint only;
- the checkpoint is RELAYED OFF the VM whenever it advances, and pushed
  back on relaunch (45 MB chunks), so a reclaim costs at most one
  checkpoint interval;
- the session is stopped when training is done, or at the deadline.

Artifacts go to /mnt/c/ml_artifacts/chat_base/ (outside OneDrive's sync).
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
NAME = "pk-chat"
ART = "/mnt/c/ml_artifacts/chat_base"
HOME = os.path.expanduser("~/pk_chat_runner")
CHUNK = 45 * 1024 * 1024
TICK = 240


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
        self.d = json.load(open(p)) if os.path.exists(p) else {"endpoint": None, "relayed_step": 0, "launches": 0}

    def save(self):
        json.dump(self.d, open(os.path.join(HOME, "state.json"), "w"), indent=1)


def relay_checkpoint(st):
    """Download state.txt; if the step advanced, the model and optimiser too."""
    os.makedirs(ART, exist_ok=True)
    tmp = os.path.join(ART, "state.txt.new")
    rc, _ = colab("download", "-s", NAME, "/content/chat/base/state.txt", tmp, timeout=200)
    if rc != 0 or not os.path.exists(tmp):
        return
    step = int(open(tmp).readline().strip() or 0)
    if step <= st.d["relayed_step"]:
        os.remove(tmp)
        return
    ok = True
    for f in ("model.safetensors", "optim.safetensors", "events.jsonl"):
        rc, _ = colab("download", "-s", NAME, f"/content/chat/base/{f}", os.path.join(ART, f + ".new"), timeout=900)
        ok = ok and rc == 0
    # mtstudio may have saved again while we downloaded: keep the set only if
    # state.txt still names the same step (a torn set would fail to resume).
    tmp2 = os.path.join(ART, "state.txt.check")
    rc, _ = colab("download", "-s", NAME, "/content/chat/base/state.txt", tmp2, timeout=200)
    if rc == 0 and os.path.exists(tmp2):
        ok = ok and int(open(tmp2).readline().strip() or 0) == step
        os.remove(tmp2)
    else:
        ok = False
    if not ok:
        log(f"checkpoint relay at step {step} incomplete or overtaken by a save; will retry")
        return
    for f in ("model.safetensors", "optim.safetensors", "events.jsonl"):
        os.replace(os.path.join(ART, f + ".new"), os.path.join(ART, f))
    os.replace(tmp, os.path.join(ART, "state.txt"))  # last: state.txt names a complete set
    st.d["relayed_step"] = step
    st.save()
    log(f"relayed checkpoint at step {step}")


def upload_checkpoint():
    if not os.path.exists(os.path.join(ART, "state.txt")):
        return
    for f in ("model.safetensors", "optim.safetensors"):
        src = os.path.join(ART, f)
        with open(src, "rb") as fh:
            i = 0
            while True:
                buf = fh.read(CHUNK)
                if not buf:
                    break
                part = os.path.join(HOME, f"{f}.part{i:02d}")
                open(part, "wb").write(buf)
                rc, _ = colab("upload", "-s", NAME, part, f"/content/chat/upload/{f}.part{i:02d}", timeout=1500)
                os.remove(part)
                if rc != 0:
                    raise RuntimeError(f"upload of {f} part {i} failed")
                i += 1
    for f in ("state.txt", "events.jsonl"):
        colab("upload", "-s", NAME, os.path.join(ART, f), f"/content/chat/upload/{f}", timeout=600)
    log("checkpoint uploaded for resume")


def launch(st, commit):
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
        upload_checkpoint()
    except RuntimeError as e:
        log(str(e))
        return False
    url = f"https://raw.githubusercontent.com/JohnnyTeutonic/paperkiln/{commit}/tools/colab_chat/boot_base.sh"
    rc, out = exec_py(
        "import subprocess, urllib.request\n"
        f"open('/content/boot_base.sh','wb').write(urllib.request.urlopen('{url}').read())\n"
        f"subprocess.Popen('nohup bash /content/boot_base.sh {commit} > /dev/null 2>&1 &', shell=True)\n"
        "print('BOOTED')\n")
    if "BOOTED" not in out:
        log(f"boot exec failed: {out.strip()[-160:]}")
        return False
    log("boot launched")
    return True


def vm_status():
    rc, out = exec_py(
        "import os\n"
        "s = open('/content/chat/STATUS').read().strip() if os.path.exists('/content/chat/STATUS') else 'BOOTING'\n"
        "t = open('/content/chat/train.log').read()[-400:] if os.path.exists('/content/chat/train.log') else ''\n"
        "print('VMSTATUS', s)\nprint(t.replace('\\n', ' | ')[-300:])\n")
    m = re.search(r"VMSTATUS (\w+)", out)
    return (m.group(1) if m else None), out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--commit", required=True)
    ap.add_argument("--max-hours", type=float, default=20.0)
    a = ap.parse_args()
    os.makedirs(HOME, exist_ok=True)
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
    up = listed() is not None
    if not up and not launch(st, a.commit):
        time.sleep(TICK)
    try:
        while time.time() < deadline:
            sh([COLPY, ADOPT, "--refresh"], 180)
            if listed() is None:
                ep = st.d.get("endpoint")
                rc, out = sh([COLPY, ADOPT, "--adopt", f"{ep}={NAME}"], 180) if ep else (1, "")
                if not (rc == 0 and "adopted" in out):
                    log("session gone (reclaimed?); relaunching from the last relayed checkpoint")
                    if not launch(st, a.commit):
                        time.sleep(TICK)
                        continue
            status, out = vm_status()
            log(f"vm {status}: {out.strip().splitlines()[-1][-160:] if out.strip() else ''}")
            if status in ("RUNNING", "FINISHED"):
                relay_checkpoint(st)
            if status == "FINISHED":
                for f in ("chat_base.safetensors", "chat_base.gguf", "events.jsonl", "result.json"):
                    colab("download", "-s", NAME, f"/content/chat/base/{f}", os.path.join(ART, f), timeout=900)
                colab("download", "-s", NAME, "/content/chat/tinychat_hf_full_v1.sha256",
                      os.path.join(ART, "tinychat_hf_full_v1.sha256"), timeout=120)
                log("FINISHED: final artifacts downloaded")
                break
            if status == "BOOT_FAILED":
                log("boot failed on the VM; see /content/chat/boot.log (downloading it)")
                colab("download", "-s", NAME, "/content/chat/boot.log", os.path.join(ART, "boot.log"), timeout=120)
                break
            time.sleep(TICK)
    finally:
        colab("stop", "-s", NAME, timeout=120)
        log("session stopped; runner exiting")
        os.remove(pid)


if __name__ == "__main__":
    main()
