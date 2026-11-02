#!/usr/bin/env python3
"""Local driver for one L4 run (microtorch/AGENTS.md path C). Run in a foreground WSL process from the paperkiln
checkout's root:

    python3 experiments/mamba_parallel_scan_2026-11-03/drive.py FULL_COMMIT_SHA RUN_ID

Session name claude-scan-RUN_ID (unique per run). Outputs: /mnt/c/ml_artifacts/mamba_parallel_scan_20261103/RUN_ID/
(driver.log, scan_out.tgz, run_RUN_ID/out/); receipts copied to this experiment's receipts/RUN_ID/.

Deadlines from the driver's start: work (polling) ends at 60 minutes, retrieval at 75, shutdown at 90. Every
external call runs with a timeout capped by the deadline that governs it; a call whose deadline has passed is not
started. Timed-out children are killed. Exit status:
  0  run validated: status.json matches run, commit and HEAD, every required gate 0, VM exit 0; cleanup verified
  1  the experiment failed (receipts preserved)
  2  transport, receipt or validation could not be completed
  3  cleanup unresolved: the session may still be running (stop it by hand; the log says how)
"""
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path

COL = os.environ.get("DRIVE_TEST_COLAB") or os.path.expanduser("~/.local/bin/colab")
CLI_PY = os.environ.get("DRIVE_TEST_CLI_PY") or os.path.expanduser("~/.local/share/uv/tools/google-colab-cli/bin/python")
HERE = Path(__file__).resolve().parent
ARTIFACTS = Path(os.environ.get("DRIVE_TEST_ARTIFACTS") or "/mnt/c/ml_artifacts/mamba_parallel_scan_20261103")
REAP = os.environ["DRIVE_TEST_REAP"].split() if os.environ.get("DRIVE_TEST_REAP") else [sys.executable, "tools/colab_reap_orphans.py"]
ADOPT = os.environ["DRIVE_TEST_ADOPT"].split() if os.environ.get("DRIVE_TEST_ADOPT") else [CLI_PY, "tools/colab_adopt.py"]
MINUTES = float(os.environ.get("DRIVE_TEST_MINUTE_SECONDS") or 60)  # tests shrink the deadlines
REQUIRED_GATES = {"checkout", "configure", "build", "host_test", "numpy", "strict_unset", "replay_unset", "strict_2",
                  "replay_2", "strict_3", "replay_3", "switch_refuses_malformed", "scan_bench"}


class Stop(Exception):
    pass


class Driver:
    def __init__(self, commit, run):
        self.commit, self.run = commit, run
        self.name = f"claude-scan-{run}"
        self.out = ARTIFACTS / run
        self.receipts = Path(os.environ.get("DRIVE_TEST_RECEIPTS") or HERE / "receipts") / run
        self.start = time.monotonic()
        self.work_end, self.retrieve_end, self.shutdown_end = (self.start + m * MINUTES for m in (60, 75, 90))
        self.intent = False      # a creation was attempted: cleanup must reconcile
        self.endpoint = None
        self.log_file = None

    # ---- logging: the file is the record; the terminal is best effort
    def log(self, *parts):
        line = f"{time.strftime('%H:%M:%S')} " + " ".join(str(p) for p in parts)
        self.log_file.write(line + "\n")
        self.log_file.flush()
        try:
            print(line, flush=True)
        except (BrokenPipeError, OSError):
            pass

    def call(self, argv, want, deadline, label):
        """Run argv with timeout min(want, time to deadline). Returns (rc, stdout) or (None, '') if not started or
        timed out (the child is killed)."""
        left = deadline - time.monotonic()
        if left <= 1:
            self.log(f"{label}: not started, deadline passed")
            return None, ""
        try:
            r = subprocess.run(argv, capture_output=True, text=True, timeout=min(want, left))
        except subprocess.TimeoutExpired:
            self.log(f"{label}: timed out, killed")
            return None, ""
        self.log(f"{label}: exit {r.returncode}")
        return r.returncode, r.stdout + r.stderr

    def sessions(self, deadline):
        """Authoritative listing {name: endpoint}, or None if the listing failed."""
        rc, out = self.call([COL, "sessions"], 120, deadline, "sessions")
        if rc != 0:
            return None
        listing = {}
        for line in out.splitlines():
            m = re.match(r"^\[([^\]]+)\]\s+(\S+)", line)
            if m:
                listing[m.group(1)] = m.group(2)
        return listing

    # ---- phases
    def preflight(self):
        listing = self.sessions(self.work_end)
        if listing is None:
            raise Stop("preflight: session listing failed; not creating anything")
        self.log("preflight sessions:", json.dumps(listing))
        if self.name in listing:
            raise Stop(f"preflight: a session named {self.name} exists; not ours to touch")
        for argv, label in ((REAP, "orphans"),
                            (ADOPT + ["--list"], "adopt --list")):
            rc, out = self.call(argv, 300, self.work_end, label)
            self.log(f"{label}:\n" + out.strip())
            if rc != 0:
                raise Stop(f"preflight: {label} failed")

    def create(self):
        self.intent = True
        self.call([COL, "new", "-s", self.name, "--gpu", "L4"], 600, self.work_end, "new")
        for _ in range(3):  # reconcile whatever the creation call returned
            listing = self.sessions(self.work_end)
            if listing is not None and self.name in listing:
                self.endpoint = listing[self.name]
                self.log("owned session", self.name, "endpoint", self.endpoint)
                return
            time.sleep(MINUTES / 3)
        raise Stop("creation not confirmed in the session listing")

    def exec_py(self, code, want, deadline, label):
        p = Path(f"/tmp/claude_scan_{self.run}_{label}.py")
        p.write_text(code)
        return self.call([COL, "exec", "-s", self.name, "-f", str(p), "--timeout", str(max(30, want - 60))],
                         want, deadline, label)

    def launch(self):
        url = (f"https://raw.githubusercontent.com/JohnnyTeutonic/paperkiln/{self.commit}/"
               "experiments/mamba_parallel_scan_2026-11-03/vm_run.sh")
        code = (
            "import subprocess, urllib.request\n"
            f"src = urllib.request.urlopen({url!r}).read()\n"
            f"open('/content/vm_run_{self.run}.sh', 'wb').write(src.replace(b'\\r\\n', b'\\n'))\n"
            f"subprocess.Popen('nohup bash /content/vm_run_{self.run}.sh {self.commit} {self.run} "
            f"> /content/vm_run_{self.run}.log 2>&1 &', shell=True)\n"
            "print('LAUNCHED')\n")
        rc, out = self.exec_py(code, 200, self.work_end, "launch")
        return rc == 0 and "LAUNCHED" in out

    def poll(self):
        code = (f"import os\np = '/content/run_{self.run}/out/DONE'\n"
                f"print('DONE' if os.path.exists(p) and open(p).read().strip() == {self.run!r} else 'RUNNING')\n")
        while time.monotonic() < self.work_end:
            time.sleep(max(0, min(2 * MINUTES, self.work_end - time.monotonic())))
            self.call(ADOPT + ["--refresh"], 180, self.work_end, "refresh")
            rc, out = self.exec_py(code, 200, self.work_end, "poll")
            if rc == 0 and "DONE" in out.split():
                return True
        return False

    def retrieve(self):
        code = ("import subprocess\n"
                f"subprocess.run('cp /content/vm_run_{self.run}.log /content/run_{self.run}/out/ 2>/dev/null; "
                f"tar czf /content/scan_{self.run}.tgz -C /content run_{self.run}/out', shell=True, check=True)\n"
                "print('PACKED')\n")
        rc, out = self.exec_py(code, 300, self.retrieve_end, "pack")
        if rc != 0 or "PACKED" not in out:
            return False
        archive = self.out / "scan_out.tgz"
        rc, _ = self.call([COL, "download", "-s", self.name, f"/content/scan_{self.run}.tgz", str(archive)], 600,
                          self.retrieve_end, "download")
        if rc != 0 or not archive.is_file():
            return False
        try:
            with tarfile.open(archive) as t:
                for m in t.getmembers():
                    if m.issym() or m.islnk() or m.name.startswith("/") or ".." in Path(m.name).parts:
                        self.log("refusing archive member", m.name)
                        return False
                t.extractall(self.out)
        except (tarfile.TarError, OSError) as e:
            self.log("extract failed:", e)
            return False
        return True

    def copy_receipts(self):
        src = self.out / f"run_{self.run}" / "out"
        try:
            self.receipts.mkdir(parents=True, exist_ok=False)
            for f in src.iterdir():
                if f.is_file():
                    shutil.copy2(f, self.receipts / f.name)
            return True
        except OSError as e:
            self.log("receipt copy failed:", e)
            return False

    def validate(self):
        try:
            s = json.loads((self.out / f"run_{self.run}" / "out" / "status.json").read_text())
        except (OSError, ValueError) as e:
            self.log("status.json missing or unreadable:", e)
            return None
        problems = []
        if s.get("run_id") != self.run or s.get("commit") != self.commit or s.get("head") != self.commit:
            problems.append("run/commit/HEAD mismatch")
        gates = s.get("gates", {})
        if set(gates) != REQUIRED_GATES:
            problems.append(f"gate set differs: missing {sorted(REQUIRED_GATES - set(gates))}, "
                            f"extra {sorted(set(gates) - REQUIRED_GATES)}")
        failed = sorted(k for k, v in gates.items() if v != 0)
        if failed:
            problems.append(f"failed gates {failed}")
        if s.get("exit_status") != 0:
            problems.append(f"VM exit status {s.get('exit_status')}")
        if s.get("pass") is not True:
            problems.append("pass is not true")
        self.log("status.json:", json.dumps(s))
        self.log("validation:", "PASS" if not problems else "; ".join(problems))
        return not problems

    def cleanup(self):
        """Verified shutdown; True only on an authoritative listing without our name or endpoint."""
        if not self.intent:
            return True
        for attempt in range(1, 5):
            self.call([COL, "stop", "-s", self.name], 120, self.shutdown_end, f"stop {attempt}")
            time.sleep(min(MINUTES / 6, max(0, self.shutdown_end - time.monotonic())))
            listing = self.sessions(self.shutdown_end)
            if listing is not None:
                endpoints = set(listing.values())
                if self.name not in listing and (self.endpoint is None or self.endpoint not in endpoints):
                    self.log(f"CLEANUP OK: {self.name} ({self.endpoint}) not listed")
                    return True
                if self.name not in listing and self.endpoint in endpoints:  # the CLI lost the name
                    self.call(ADOPT + ["--adopt", f"{self.endpoint}={self.name}"], 180,
                              self.shutdown_end, "adopt")
            if time.monotonic() >= self.shutdown_end:
                break
        self.log(f"CLEANUP UNRESOLVED: {self.name} ({self.endpoint}) may still be running. "
                 f"Stop it by hand: {COL} stop -s {self.name}")
        return False

    def main(self):
        self.out.mkdir(parents=True, exist_ok=False)
        self.log_file = open(self.out / "driver.log", "a")
        self.log(f"run {self.run} commit {self.commit} session {self.name}")
        transport = receipts = False
        valid = None
        try:
            self.preflight()
            self.create()
            launched = self.launch()
            done = self.poll() if launched else False
            self.log("launched", launched, "done", done)
            transport = self.retrieve()  # also after a launch failure or deadline: diagnostics are evidence
            receipts = transport and self.copy_receipts()
            valid = self.validate() if transport else None
        except Stop as e:
            self.log("stopped:", e)
        except Exception as e:  # signals arrive here as Stop or KeyboardInterrupt-like exceptions
            self.log("error:", repr(e))
        finally:
            for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):  # nothing may interrupt the shutdown
                signal.signal(s, signal.SIG_IGN)
            clean = self.cleanup()
            rc, out = self.call([COL, "sessions"], 120, self.shutdown_end + 300, "final sessions")
            self.log("final sessions:\n" + out.strip())
            rc, out = self.call(REAP, 300, self.shutdown_end + 300,
                                "final orphans")
            self.log("final orphans:\n" + out.strip())
            code = 3 if not clean else 2 if not (transport and receipts and valid is not None) else 0 if valid else 1
            self.log(f"exit {code} (transport {transport}, receipts {receipts}, valid {valid}, cleanup {clean})")
            self.log_file.close()
            if receipts:
                try:
                    shutil.copy2(self.out / "driver.log", self.receipts / "driver.log")
                except OSError:
                    code = max(code, 2)
            return code


def main():
    if len(sys.argv) != 3 or not re.fullmatch(r"[0-9a-f]{40}", sys.argv[1]) \
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", sys.argv[2]):
        print(__doc__, file=sys.stderr)
        return 2
    d = Driver(sys.argv[1], sys.argv[2])

    def on_signal(signum, _frame):
        raise Stop(f"signal {signum}")

    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, on_signal)
    return d.main()


if __name__ == "__main__":
    sys.exit(main())
