#!/usr/bin/env python3
"""A stand-in for ssh and rsync (chosen by the name it is called under) for the Jarvis tests: the VM is
FAKE_ROOT/vm, which plays /content, and exec goes through runner_multi_equiv/fakecolab.py's VM simulation.
FAKE_ROOT/calls_ssh.log gets every argv. No network.

A file FAKE_ROOT/ssh_down makes every call fail as ssh does (255); FAKE_ROOT/ssh_hang makes it hang."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.environ["FAKE_ROOT"]
VM = os.path.join(ROOT, "vm")
FAKECOLAB = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                         "runner_multi_equiv", "fakecolab.py")


def here(path):
    return path.replace("/content", VM, 1) if path.startswith("/content") else path


def rsync(argv):
    src, dst = argv[-2], argv[-1]
    src, dst = (here(p.split(":", 1)[1]) if "@" in p.split(":", 1)[0] else p for p in (src, dst))
    if not os.path.exists(src):
        return 23
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)
    return 0


def ssh(argv):
    assert any(a.startswith("ubuntu@") for a in argv), argv
    code = sys.stdin.read()
    if "pk_prelude.sh" in code:  # jarvis_prelude.sh has its own test; this VM only needs the result
        os.makedirs(os.path.join(VM, "chat", "upload"), exist_ok=True)
        print("PRELUDE_RC", os.environ.get("FAKE_PRELUDE_RC", "0"))
        if os.environ.get("FAKE_PRELUDE_RC", "0") == "0":
            print("PROVIDER_WRITTEN")
        return 0
    if "SSH_OK" in code:
        print("SSH_OK")
        return 0
    fj = os.path.join(ROOT, "fake.json")
    if not os.path.exists(fj):
        with open(fj, "w") as f:
            json.dump({"alive": True, "launches": 1, "name": "jarvis"}, f)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
    try:
        p = subprocess.run([sys.executable, FAKECOLAB, "exec", "-f", f.name], capture_output=True, text=True)
    finally:
        os.remove(f.name)
    sys.stdout.write((p.stdout + p.stderr).replace(VM + "/", "/content/"))  # fakecolab rewrote the paths
    return p.returncode


def main():
    with open(os.path.join(ROOT, "calls_ssh.log"), "a") as f:
        f.write(json.dumps(sys.argv) + "\n")
    if os.path.exists(os.path.join(ROOT, "ssh_hang")):
        time.sleep(3600)
    if os.path.exists(os.path.join(ROOT, "ssh_down")):
        print("ssh: connect to host port 22: Connection refused", file=sys.stderr)
        return 255
    return rsync(sys.argv) if os.path.basename(sys.argv[0]) == "rsync" else ssh(sys.argv)


if __name__ == "__main__":
    sys.exit(main())
