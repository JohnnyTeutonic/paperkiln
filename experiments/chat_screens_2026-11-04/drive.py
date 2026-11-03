#!/usr/bin/env python3
"""Local driver for the first-wave screening session (microtorch/AGENTS.md path C; the root workspace's
AI_ML/chat_screens_claude/PROTOCOL.md v1.2 and SESSION.md). Run in a foreground WSL process from the paperkiln
checkout's root:

    python3 experiments/chat_screens_2026-11-04/drive.py FULL_COMMIT_SHA RUN_ID

FULL_COMMIT_SHA is this experiment's commit on GitHub (vm_run.sh is fetched at it); the trainer is built on the VM
from the numerical source pinned in vm_run.sh. Session name claude-screens-RUN_ID. Outputs:
/mnt/c/ml_artifacts/chat_screens_claude/w1/RUN_ID/ (driver.log, staging/, one folder per screen holding model.gguf,
manifest.json and train.log, and run_RUN_ID/out/ with the VM's logs and status.json); the top-level files of out/
and every screen's manifest are copied to this experiment's receipts/RUN_ID/, with archive.sha256 hashing every
retrieved file, after the VM is gone.

Before any session is created, every input is hashed against PINS (the protocol's pins and PACKETS.md); a mismatch
creates nothing. After creation the inputs are uploaded, then hashed on the VM against the manifest; a part that
differs is uploaded again (at most twice). While the VM trains, each screen it marks READY is collected at once: its
parts are downloaded, hashed, joined and the whole GGUF hashed against the VM's manifest. A screen counts as home only
then. No screen is ever retrained by the driver.

Deadlines from the driver's start (DEADLINES, minutes): polling ends at 100, workload-stop attempts at 104,
retrieval (the final logs and any screen still uncollected) at 112, shutdown attempts at 117, the final inventory at
120: the session's 2 L4-hour cap. Every external call has a timeout capped by the deadline that governs it; a call
past its deadline is not started; a timed-out child is killed.

Safeguards carried over from the Mamba scan driver (experiments/mamba_parallel_scan_2026-11-03/drive.py): the launch
publishes vm_run.sh atomically and a launch without acknowledgement is reconciled by a read-only probe, never blindly
repeated; read-only and idempotent calls are retried; a run that did not finish is stopped through the VM script's
published PID before retrieval; each retrieval extracts into its own staging folder; shutdown is certified only by an
authoritative session listing that shows neither our session name, nor the endpoint we owned, nor any endpoint that
was not there before our creation call.
Exit status:
  0  every gate passed, status.json matches run, commit, source and manifest, and all four screens are home verified
  1  the experiment failed (a gate failed; screens already home and receipts are preserved)
  2  transport, receipt or validation could not be completed, a screen is not home, or a workload stop was not
     confirmed
  3  cleanup unresolved: a VM may still be running (the log names it and how to stop it)
Test hooks (DRIVE_TEST_*) are honoured only with DRIVE_TEST_HARNESS=1, and refused otherwise.
"""
import hashlib
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

TEST = os.environ.get("DRIVE_TEST_HARNESS") == "1"


def hook(name, default):
    return os.environ.get(name) if TEST and os.environ.get(name) else default


COL = hook("DRIVE_TEST_COLAB", os.path.expanduser("~/.local/bin/colab"))
CLI_PY = hook("DRIVE_TEST_CLI_PY", os.path.expanduser("~/.local/share/uv/tools/google-colab-cli/bin/python"))
REAP = hook("DRIVE_TEST_REAP", f"{sys.executable} tools/colab_reap_orphans.py").split()
ADOPT = hook("DRIVE_TEST_ADOPT", f"{CLI_PY} tools/colab_adopt.py").split()
HERE = Path(__file__).resolve().parent
ARTIFACTS = Path(hook("DRIVE_TEST_ARTIFACTS", "/mnt/c/ml_artifacts/chat_screens_claude/w1"))
RECEIPTS = Path(hook("DRIVE_TEST_RECEIPTS", str(HERE / "receipts")))
MINUTE = float(hook("DRIVE_TEST_MINUTE_SECONDS", "60"))
LOG_FAIL_AFTER = int(hook("DRIVE_TEST_LOG_FAIL_AFTER", "0"))
RAISE_ON = hook("DRIVE_TEST_RAISE_ON", "")
PART_BYTES = int(hook("DRIVE_TEST_PART_BYTES", str(90 * 1024 * 1024)))
SOURCE = "1ccb693f55d7328d4a8b0b7238fe8a2cafb6edf1"
SCREENS = ("S0", "S1", "S2", "S3")
REQUIRED_GATES = {"provenance", "checkout", "inputs", "configure", "build", "specs"} | {
    f"{g}_{s}" for s in SCREENS for g in ("train", "package")}
DEADLINES = (100, 104, 112, 117, 120)
VM_IN = "/content/screens_in"
# The inputs' private Hugging Face dataset repo (Jonathan, 4 Nov 2026), seeded with hf_seed.py: the base parts and every
# other pinned file under their VM names. The token (scoped to that repo only) lives in HF_TOKEN_FILE, mode 600, and is
# never printed; it is copied to our own VM only for the download and deleted there afterwards. Without the token file
# the driver uploads every input instead.
HF_REPO = hook("DRIVE_TEST_HF_REPO", "JohnnyTeutonic/chat-screens-inputs")
HF_TOKEN_FILE = Path(hook("DRIVE_TEST_HF_TOKEN_FILE", os.path.expanduser("~/.config/claude_screens/hf_token")))
# The repo commit hf_seed.py verified, written there after verification; downloads pin it (an immutable revision).
HF_REVISION_FILE = Path(hook("DRIVE_TEST_HF_REVISION_FILE", os.path.expanduser("~/.config/claude_screens/hf_revision")))

# Every input, by its name on the VM: (local path, SHA-256). PROTOCOL.md section 2 and PACKETS.md.
ART = "/mnt/c/ml_artifacts"
PINS = {
    "sfull_sft1500.safetensors": (
        f"{ART}/tiny_chat/c12_Sfull_r1_cont_v3_20261101/post_c12_Sfull_r1/sft/.checkpoint_v2/"
        "generation-bc6c1bfa7feb44bc9f13e4df158440a3/model.safetensors",
        "886e7db5dc2dc457f01d5f21847e5d63dc50e02785562d167d253e46c0f834f1"),
    "vocab.gguf": (f"{ART}/capacity/vocab/vocab_llama3_bpe.gguf",
                   "0b880aa073a7e27c0770d5ecc5215b55d6d415080d20bd3dc7b169ecabadf198"),
    "S0.jsonl": (f"{ART}/sfull_uplift_20261103/token-packet-v3/train.jsonl",
                 "a319c322b0d7a5f134ecd22223b8c7f6b11277794a8ed485ee0aaf085ad59602"),
    "S1.jsonl": (f"{ART}/chat_screens_claude/packets/v1/S1/train.jsonl",
                 "6fd1304ead0dc3e5d6b60cef554b52b8d494f0b224264fce4a68481901c1afa4"),
    "S2.jsonl": (f"{ART}/chat_screens_claude/packets/v1/S2/train.jsonl",
                 "bb8d2fadbb0e3a09073742a1aaded22a67af653f3a4397210b8c27a966034a4b"),
    "S3.jsonl": (f"{ART}/chat_screens_claude/packets/v1/S3/train.jsonl",
                 "6ec1a8d855e981d50b8f63eff908d16828f21d47671293789fc7a0169b25568b"),
    "arm_a_lr3e5.json": (str(HERE / "arm_a_lr3e5.json"),
                         "66c718e0b501c751aae0a1cd22eddfdc666a7cfb46a4c074fa41b67aa220125e"),
}
if TEST and os.environ.get("DRIVE_TEST_PINS"):
    PINS = {k: tuple(v) for k, v in json.loads(os.environ["DRIVE_TEST_PINS"]).items()}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def launch_code(url, commit, run, manifest_sha, root="/content"):
    """The VM-side launch: fetch vm_run.sh and start it detached. Each call writes its own temporary file and renames
    it over the final path, so a delayed duplicate never rewrites a script a bash is reading; vm_run.sh's atomic
    run-directory claim then lets only one launch do any work. (`root` is for tests only.)"""
    return (
        "import os, subprocess, tempfile, urllib.request\n"
        f"src = urllib.request.urlopen({url!r}).read().replace(b'\\r\\n', b'\\n')\n"
        f"fd, tmp = tempfile.mkstemp(prefix='vm_run_{run}.', suffix='.tmp', dir={root!r})\n"
        "with os.fdopen(fd, 'wb') as f:\n"
        "    f.write(src)\n"
        f"os.replace(tmp, '{root}/vm_run_{run}.sh')\n"
        f"subprocess.Popen('nohup bash {root}/vm_run_{run}.sh {commit} {run} {manifest_sha} "
        f">> {root}/vm_run_{run}.log 2>&1 &', shell=True)\n"
        "print('LAUNCHED')\n")


class Stop(Exception):
    pass


class Driver:
    def __init__(self, commit, run):
        self.commit, self.run = commit, run
        self.name = f"claude-screens-{run}"
        self.out = ARTIFACTS / run
        self.stage = self.out / "staging"
        self.receipts = RECEIPTS / run
        t = time.monotonic()
        self.work_end, self.stop_end, self.retrieve_end, self.shutdown_end, self.inventory_end = (
            t + m * MINUTE for m in DEADLINES)
        self.intent = False
        self.endpoint = None
        self.before = None
        self.cancel = False
        self.log_file = None
        self.log_lines = 0
        self.raised = False
        self.log_failed = False
        self.done = False
        self.stop_confirmed = None
        self.launch_attempted = False
        self.manifest = None
        self.manifest_sha = None
        self.home = {}  # screen -> verified GGUF SHA-256

    # ---- logging never raises: the file is the record, the terminal and stderr best effort
    def log(self, *parts):
        line = f"{time.strftime('%H:%M:%S')} " + " ".join(str(p) for p in parts)
        try:
            self.log_lines += 1
            if LOG_FAIL_AFTER and self.log_lines > LOG_FAIL_AFTER:
                raise OSError("test: log write failed")
            self.log_file.write(line + "\n")
            self.log_file.flush()
        except (OSError, ValueError, AttributeError):
            self.log_failed = True
            try:
                sys.stderr.write("[log file unavailable] " + line + "\n")
            except (OSError, ValueError):
                pass
        try:
            print(line, flush=True)
        except (OSError, ValueError):
            pass

    def check(self):
        if self.cancel:
            raise Stop("cancelled by signal")

    def call(self, argv, want, deadline, label):
        """Run argv, timeout min(want, time to deadline). (rc, output), or (None, '') if not started, timed out
        (the child is killed) or failed to start. Never raises."""
        left = deadline - time.monotonic()
        if left <= 1:
            self.log(f"{label}: not started, deadline passed")
            return None, ""
        try:
            if RAISE_ON and RAISE_ON in argv and not self.raised:
                self.raised = True
                raise OSError("test: command failed to start")
            r = subprocess.run(argv, capture_output=True, text=True, timeout=min(want, left))
        except subprocess.TimeoutExpired:
            self.log(f"{label}: timed out, killed")
            return None, ""
        except OSError as e:
            self.log(f"{label}: could not run: {e}")
            return None, ""
        self.log(f"{label}: exit {r.returncode}")
        return r.returncode, r.stdout + r.stderr

    def sleep(self, seconds, deadline):
        end = min(time.monotonic() + seconds, deadline)
        while time.monotonic() < end and not self.cancel:
            time.sleep(min(1.0, max(0.0, end - time.monotonic())))

    def sessions(self, deadline):
        """Rows [(name, endpoint)] from an authoritative listing, or None if the listing failed or has a row
        that does not parse (fail closed)."""
        rc, out = self.call([COL, "sessions"], 120, deadline, "sessions")
        if rc != 0:
            return None
        rows = []
        for line in out.splitlines():
            if not line.startswith("["):
                continue
            m = re.match(r"^\[([^\]]+)\]\s+(\S+)", line)
            if not m:
                self.log("unparsed session row:", line)
                return None
            if m.group(1) == "colab":
                continue
            rows.append((m.group(1), m.group(2)))
        return rows

    # ---- phases
    def stage_inputs(self):
        """Local, before any session: every input hashes as pinned; the base weights are split into parts; the
        manifest pins every file and part. Raises Stop on any mismatch (nothing has been created)."""
        self.stage.mkdir(parents=True, exist_ok=False)
        files = {}
        for name, (path, want) in PINS.items():
            got = sha256(path)
            if got != want:
                raise Stop(f"input {name} hashes {got}, pinned {want}: nothing created")
            files[name] = want
        parts = []
        with open(PINS["sfull_sft1500.safetensors"][0], "rb") as f:
            for i in range(10 ** 6):
                chunk = f.read(PART_BYTES)
                if not chunk:
                    break
                p = self.stage / f"base.part{i:02d}"
                p.write_bytes(chunk)
                parts.append({"name": p.name, "sha256": hashlib.sha256(chunk).hexdigest()})
        self.manifest = {"files": files, "base_parts": parts, "arm_a_spec_sha256": files["arm_a_lr3e5.json"],
                         "source": SOURCE, "run_id": self.run}
        man = self.stage / "manifest.json"
        man.write_text(json.dumps(self.manifest, indent=2, sort_keys=True))
        self.manifest_sha = sha256(man)
        self.log("inputs pinned and staged:", len(files), "files,", len(parts), "base parts, manifest",
                 self.manifest_sha)

    def uploads(self):
        """[(local path, VM name)] in upload order: the manifest, then the small inputs, then the base parts."""
        out = [(self.stage / "manifest.json", "manifest.json")]
        out += [(Path(PINS[n][0]), n) for n in PINS if n != "sfull_sft1500.safetensors"]
        out += [(self.stage / p["name"], p["name"]) for p in self.manifest["base_parts"]]
        return out

    def preflight(self):
        rows = self.sessions(self.work_end)
        if rows is None:
            raise Stop("preflight: session listing failed; not creating anything")
        self.log("preflight sessions:", json.dumps(rows))
        if any(n == self.name for n, _ in rows):
            raise Stop(f"preflight: a session named {self.name} exists; not ours to touch")
        for argv, label in ((REAP, "orphans"), (ADOPT + ["--list"], "adopt --list")):
            rc, out = self.call(argv, 300, self.work_end, label)
            self.log(f"{label}:\n" + out.strip())
            if rc != 0:
                raise Stop(f"preflight: {label} failed")
        self.before = {ep for _, ep in rows}

    def create(self):
        self.intent = True
        self.call([COL, "new", "-s", self.name, "--gpu", "L4"], 600, self.work_end, "new")
        for _ in range(3):
            rows = self.sessions(self.work_end)
            if rows is not None:
                mine = [ep for n, ep in rows if n == self.name]
                if len(mine) == 1:
                    self.endpoint = mine[0]
                    self.log("owned session", self.name, "endpoint", self.endpoint)
                    return
                new = sorted({ep for _, ep in rows} - self.before)
                if new:
                    self.log("endpoints new since creation, not under our name:", new)
            self.sleep(MINUTE / 3, self.work_end)
            self.check()
        raise Stop("creation not confirmed under our name")

    def exec_py(self, code, want, deadline, label):
        p = Path(f"/tmp/claude_screens_{self.run}_{label}.py")
        p.write_text(code)
        return self.call([COL, "exec", "-s", self.name, "-f", str(p), "--timeout", str(max(30, want - 60))],
                         want, deadline, label)

    def vm_hashes(self, names):
        """{name: sha256 or None} for files in VM_IN, from one read-only exec (retried), or None if it failed."""
        code = ("import hashlib, json, os\n"
                f"names = {sorted(names)!r}\n"
                "def sha(p):\n"
                "    h = hashlib.sha256()\n"
                "    with open(p, 'rb') as f:\n"
                "        for b in iter(lambda: f.read(1 << 20), b''):\n"
                "            h.update(b)\n"
                "    return h.hexdigest()\n"
                f"print('HASHES ' + json.dumps({{n: (sha(os.path.join({VM_IN!r}, n)) if os.path.isfile("
                f"os.path.join({VM_IN!r}, n)) else None) for n in names}}))\n")
        for attempt in (1, 2):
            rc, out = self.exec_py(code, 300, self.work_end, f"vm_hashes{attempt}")
            for line in out.splitlines():
                if rc == 0 and line.startswith("HASHES "):
                    try:
                        return json.loads(line[len("HASHES "):])
                    except ValueError:
                        pass
        return None

    def hf_fetch(self, names):
        """Have the VM download `names` from HF_REPO into VM_IN with the scoped token, then delete the token file on
        the VM. Returns True if the download step reported success (hashes are checked separately). Never raises."""
        if not HF_TOKEN_FILE.is_file() or not HF_REVISION_FILE.is_file():
            self.log("no HF token or verified revision: inputs will be uploaded")
            return False
        revision = HF_REVISION_FILE.read_text().strip()
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            self.log("HF revision file malformed: inputs will be uploaded")
            return False
        self.log("HF fetch from", HF_REPO, "at revision", revision)
        rc, _ = self.call([COL, "upload", "-s", self.name, str(HF_TOKEN_FILE), f"{VM_IN}/.hf_token"], 300,
                          self.work_end, "upload token")
        if rc != 0:
            return False
        code = ("import os, time\n"
                "from huggingface_hub import hf_hub_download\n"
                f"tok_path = os.path.join({VM_IN!r}, '.hf_token')\n"
                "os.chmod(tok_path, 0o600)\n"
                "token = open(tok_path).read().strip()\n"
                "ok = True\n"
                "t = time.monotonic()\n"
                "try:\n"
                f"    for n in {sorted(names)!r}:\n"
                "        try:\n"
                f"            hf_hub_download({HF_REPO!r}, n, repo_type='dataset', revision={revision!r}, token=token, local_dir={VM_IN!r})\n"
                "        except Exception as e:\n"
                "            ok = False\n"
                "            print('FAILED', n, type(e).__name__)\n"
                "finally:\n"
                "    os.remove(tok_path)\n"
                "print('HF_DONE' if ok else 'HF_PARTIAL', round(time.monotonic() - t))\n")
        rc, out = self.exec_py(code, 1500, self.work_end, "hf_fetch")
        self.log("hf_fetch:", " ".join(out.split()[-6:]))
        return rc == 0 and "HF_DONE" in out.split()

    def upload(self):
        """Inputs onto the VM: the per-run manifest is uploaded; the data files come from the HF repo when the token
        file exists, else they are uploaded. Then everything is hashed on the VM; what differs is uploaded directly,
        at most twice more. True when every file on the VM matches the manifest."""
        rc, _ = self.exec_py(f"import os\nos.makedirs({VM_IN!r}, exist_ok=True)\nprint('READY')\n", 120,
                             self.work_end, "mkdir_in")
        want = {n: sha256(p) for p, n in self.uploads()}
        pending = self.uploads()
        data = [n for _, n in self.uploads() if n != "manifest.json"]
        if self.hf_fetch(data):
            got = self.vm_hashes(want) or {}
            pending = [(p, n) for p, n in self.uploads() if got.get(n) != want[n]]
            self.log(f"after HF fetch, {len(pending)} file(s) still to upload:", [n for _, n in pending])
            if not pending:
                return True
        for round_ in (1, 2, 3):
            for local, name in pending:
                self.check()
                for attempt in (1, 2):
                    rc, _ = self.call([COL, "upload", "-s", self.name, str(local), f"{VM_IN}/{name}"], 1800,
                                      self.work_end, f"upload {name} r{round_}a{attempt}")
                    if rc == 0:
                        break
            got = self.vm_hashes(want)
            if got is None:
                self.log("could not hash the uploads on the VM")
                return False
            bad = sorted(n for n in want if got.get(n) != want[n])
            self.log(f"upload round {round_}: {len(want) - len(bad)} of {len(want)} verified on the VM", bad)
            if not bad:
                return True
            pending = [(p, n) for p, n in self.uploads() if n in bad]
        return False

    def launch_state(self):
        code = ("import os, time\n"
                f"pid, run = '/content/vm_{self.run}.pid', '/content/run_{self.run}'\n"
                "for _ in range(90):\n"
                "    if os.path.exists(pid) or os.path.isdir(run):\n"
                "        print('STARTED'); raise SystemExit\n"
                "    time.sleep(1)\n"
                "print('ABSENT')\n")
        for attempt in (1, 2, 3):
            self.check()
            rc, out = self.exec_py(code, 200, self.work_end, f"launch_probe{attempt}")
            if rc == 0 and ("STARTED" in out.split() or "ABSENT" in out.split()):
                return "started" if "STARTED" in out.split() else "absent"
        return None

    def launch(self):
        url = (f"https://raw.githubusercontent.com/JohnnyTeutonic/paperkiln/{self.commit}/"
               "experiments/chat_screens_2026-11-04/vm_run.sh")
        code = launch_code(url, self.commit, self.run, self.manifest_sha)
        for label in ("launch", "launch_retry"):
            self.check()
            self.launch_attempted = True
            rc, out = self.exec_py(code, 200, self.work_end, label)
            if rc == 0 and "LAUNCHED" in out:
                return True
            state = self.launch_state()
            self.log(f"{label}: no acknowledgement; the VM script is {state or 'unknown'}")
            if state == "started":
                return True
            if state is None:
                return False
        return False

    def vm_state(self, deadline, label):
        """(done, [ready screens]) from one read-only exec, or (None, None) if it failed."""
        code = ("import json, os\n"
                f"o = '/content/run_{self.run}/out'\n"
                "def holds(p):\n"
                "    try:\n"
                f"        return open(p).read().strip() == {self.run!r}\n"
                "    except OSError:\n"
                "        return False\n"
                f"ready = [s for s in {list(SCREENS)!r} if holds(f'{{o}}/screens/{{s}}/READY')]\n"
                "print('STATE ' + json.dumps({'done': holds(o + '/DONE'), 'ready': ready}))\n")
        rc, out = self.exec_py(code, 200, deadline, label)
        for line in out.splitlines():
            if rc == 0 and line.startswith("STATE "):
                try:
                    s = json.loads(line[len("STATE "):])
                    return bool(s["done"]), [x for x in s["ready"] if x in SCREENS]
                except (ValueError, KeyError, TypeError):
                    pass
        return None, None

    def collect(self, screen, deadline):
        """Download one READY screen's manifest, parts and log; hash every part; join; hash the GGUF. Only a fully
        verified screen is promoted to OUT/screen. True on success. Never raises."""
        try:
            remote = f"/content/run_{self.run}/out/screens/{screen}"
            work = self.out / f"collect_{screen}"
            shutil.rmtree(work, ignore_errors=True)
            work.mkdir(parents=True)
            rc, _ = self.call([COL, "download", "-s", self.name, f"{remote}/manifest.json",
                               str(work / "manifest.json")], 300, deadline, f"{screen} manifest")
            if rc != 0:
                return False
            man = json.loads((work / "manifest.json").read_text())
            if man.get("screen") != screen or man.get("run_id") != self.run:
                self.log(f"{screen}: manifest identity mismatch", man.get("screen"), man.get("run_id"))
                return False
            for part in man["parts"]:
                name = part["name"]
                if not re.fullmatch(r"model\.gguf\.part\d{2}", name):
                    self.log(f"{screen}: refusing part name", name)
                    return False
                for attempt in (1, 2):
                    rc, _ = self.call([COL, "download", "-s", self.name, f"{remote}/{name}", str(work / name)],
                                      900, deadline, f"{screen} {name} a{attempt}")
                    if rc == 0 and (work / name).is_file() and sha256(work / name) == part["sha256"]:
                        break
                else:
                    self.log(f"{screen}: part {name} not verified")
                    return False
            with open(work / "model.gguf", "wb") as g:
                for part in man["parts"]:
                    with open(work / part["name"], "rb") as f:
                        shutil.copyfileobj(f, g, 1 << 20)
            if sha256(work / "model.gguf") != man["gguf_sha256"]:
                self.log(f"{screen}: joined GGUF hash differs from the VM's manifest")
                return False
            for part in man["parts"]:
                (work / part["name"]).unlink()
            self.call([COL, "download", "-s", self.name, f"{remote}/train.log", str(work / "train.log")], 300,
                      deadline, f"{screen} train.log")
            (work).rename(self.out / screen)
            self.home[screen] = man["gguf_sha256"]
            self.log(f"{screen}: HOME, GGUF {man['gguf_sha256']} ({man['gguf_bytes']} bytes)")
            return True
        except (OSError, ValueError, KeyError, TypeError) as e:
            self.log(f"{screen}: collection failed:", repr(e))
            return False

    def poll(self):
        while time.monotonic() < self.work_end and not self.cancel:
            self.sleep(2 * MINUTE, self.work_end)
            if self.cancel:
                break
            self.call(ADOPT + ["--refresh"], 180, self.work_end, "refresh")
            done, ready = self.vm_state(self.work_end, "poll")
            if done is None:
                continue
            for s in ready:
                if s not in self.home:
                    self.collect(s, self.work_end)
            if done:
                return True
        return False

    def stop_workload(self):
        """As the Mamba driver: SIGTERM the VM script's published PID (checked to be ours); its handler stops the
        running step's process group, writes stop.json and exits through the trap that writes status.json and DONE.
        True (interrupted or completed), False (not confirmed) or None (no launch was attempted)."""
        if not self.launch_attempted:
            self.log("workload stop: no launch was attempted")
            return None
        code = (
            "import json, os, signal, time\n"
            f"run, script = {self.run!r}, '/content/vm_run_{self.run}.sh'\n"
            f"pid_file, done, stop = '/content/vm_{self.run}.pid', '/content/run_{self.run}/out/DONE', "
            f"'/content/run_{self.run}/out/stop.json'\n"
            "def reply(result, **kw):\n"
            "    print('STOP_REPLY ' + json.dumps(dict(result=result, **kw))); raise SystemExit\n"
            "def done_ok():\n"
            "    try:\n"
            "        return open(done).read().strip() == run\n"
            "    except OSError:\n"
            "        return False\n"
            "for _ in range(60):\n"
            "    if os.path.exists(pid_file):\n"
            "        break\n"
            "    time.sleep(1)\n"
            "else:\n"
            "    reply('not_observed')\n"
            "pid = int(open(pid_file).read())\n"
            "try:\n"
            "    ours = script.encode() in open(f'/proc/{pid}/cmdline', 'rb').read()\n"
            "except OSError:\n"
            "    ours = False\n"
            "if ours and not done_ok():\n"
            "    try:\n"
            "        os.kill(pid, signal.SIGTERM)\n"
            "    except ProcessLookupError:\n"
            "        pass\n"
            "for _ in range(120):\n"
            "    if done_ok():\n"
            "        break\n"
            "    time.sleep(1)\n"
            "s = json.load(open(stop)) if os.path.exists(stop) else None\n"
            "if not done_ok():\n"
            "    reply('unconfirmed', stop=s, signalled=ours)\n"
            "if s is None:\n"
            "    reply('completed', signalled=ours)\n"
            "gone = s.get('step_group_gone') is True or s.get('step_started') is False\n"
            "reply('interrupted' if gone else 'unconfirmed', stop=s, signalled=ours)\n")
        result, out = None, ""
        for attempt in (1, 2):
            rc, out = self.exec_py(code, 260, self.stop_end, f"stop_workload{attempt}")
            reply = None
            for line in out.splitlines():
                if line.startswith("STOP_REPLY "):
                    try:
                        reply = json.loads(line[len("STOP_REPLY "):])
                    except ValueError:
                        reply = None
            result = reply.get("result") if rc == 0 and isinstance(reply, dict) else None
            if result is not None:
                break
        confirmed = result in ("interrupted", "completed")
        self.log("workload stop:", f"confirmed ({result})" if confirmed else
                 f"NOT CONFIRMED ({result}): outputs may still be changing", out.strip()[-300:])
        return confirmed

    def retrieve(self):
        """The VM's logs and status (without the GGUF parts), then any READY screen not yet home."""
        code = ("import subprocess\n"
                f"subprocess.run('cp /content/vm_run_{self.run}.log /content/run_{self.run}/out/ 2>/dev/null', "
                "shell=True)\n"
                f"r = subprocess.run('tar czf /content/screens_{self.run}.tgz --warning=no-file-changed "
                f"--exclude=\"model.gguf.part*\" -C /content run_{self.run}/out', shell=True)\n"
                "print('PACKED' if r.returncode in (0, 1) else 'PACK_FAILED %d' % r.returncode)\n")
        archive = self.out / "logs.tgz"
        prefix = f"run_{self.run}/out"
        got_logs = False
        for attempt in (1, 2):
            archive.unlink(missing_ok=True)
            rc, out = self.exec_py(code, 300, self.retrieve_end, f"pack{attempt}")
            if rc != 0 or "PACKED" not in out:
                continue
            rc, _ = self.call([COL, "download", "-s", self.name, f"/content/screens_{self.run}.tgz", str(archive)],
                              600, self.retrieve_end, f"download logs {attempt}")
            if rc != 0 or not archive.is_file():
                continue
            stage = self.out / f"extract_{attempt}"
            try:
                shutil.rmtree(stage, ignore_errors=True)
                with tarfile.open(archive) as t:
                    members = t.getmembers()
                    bad = [m.name for m in members if not (m.isfile() or m.isdir())
                           or not (m.name == prefix or m.name.startswith(prefix + "/")) or ".." in Path(m.name).parts]
                    if bad:
                        self.log("refusing archive members", bad[:5])
                        continue
                    t.extractall(stage, members=members)
                (stage / f"run_{self.run}").rename(self.out / f"run_{self.run}")
                shutil.rmtree(stage, ignore_errors=True)
                got_logs = True
                break
            except (tarfile.TarError, OSError) as e:
                self.log(f"archive attempt {attempt} unreadable:", e)
        if not got_logs:
            self.log("LOGS NOT RETRIEVED: the VM holds the only copy, and cleanup will stop it")
        _, ready = self.vm_state(self.retrieve_end, "final_state")
        for s in ready or []:
            if s not in self.home:
                self.collect(s, self.retrieve_end)
        return got_logs

    def copy_receipts(self):
        src = self.out / f"run_{self.run}" / "out"
        try:
            self.receipts.mkdir(parents=True, exist_ok=False)
            for f in src.iterdir():
                if f.is_file():
                    shutil.copy2(f, self.receipts / f.name)
            for s in SCREENS:
                if (self.out / s / "manifest.json").is_file():
                    shutil.copy2(self.out / s / "manifest.json", self.receipts / f"{s}_manifest.json")
            lines = []
            for f in sorted(p for p in self.out.rglob("*") if p.is_file() and "staging" not in p.parts
                            and p.name != "driver.log"):
                lines.append(f"{sha256(f)}  {f.relative_to(self.out).as_posix()}\n")
            tmp = self.receipts / "archive.sha256.tmp"
            tmp.write_text("".join(lines))
            tmp.rename(self.receipts / "archive.sha256")
            (self.receipts / "home.json").write_text(json.dumps(self.home, indent=2, sort_keys=True))
            return True
        except Exception as e:
            self.log("receipt copy failed:", repr(e))
            return False

    def validate(self):
        """True: validated pass. False: a well-formed status of a failed experiment. None: missing or malformed."""
        out = self.out / f"run_{self.run}" / "out"
        try:
            s = json.loads((out / "status.json").read_text())
        except (OSError, ValueError) as e:
            self.log("status.json missing or unreadable:", e)
            return None
        gates = s.get("gates") if isinstance(s, dict) else None
        if not isinstance(gates, dict) or not all(type(v) is int for v in gates.values()) \
                or type(s.get("exit_status")) is not int or not isinstance(s.get("pass"), bool):
            self.log("status.json malformed:", json.dumps(s)[:500])
            return None
        if s.get("run_id") != self.run or s.get("commit") != self.commit or s.get("source") != SOURCE \
                or s.get("head") != SOURCE or s.get("manifest_sha256") != self.manifest_sha:
            self.log("status.json identity mismatch:", s.get("run_id"), s.get("commit"), s.get("source"),
                     s.get("head"), s.get("manifest_sha256"))
            return None
        problems = []
        if set(gates) != REQUIRED_GATES:
            problems.append(f"gate set differs: missing {sorted(REQUIRED_GATES - set(gates))}, "
                            f"extra {sorted(set(gates) - REQUIRED_GATES)}")
        failed = sorted(k for k, v in gates.items() if v != 0)
        if failed:
            problems.append(f"failed gates {failed}")
        if s["exit_status"] != 0:
            problems.append(f"VM exit status {s['exit_status']}")
        if s["pass"] is not True:
            problems.append("pass is not true")
        try:
            stop = json.loads((out / "stop.json").read_text()) if (out / "stop.json").exists() else {}
        except (OSError, ValueError):
            stop = {"step_group_gone": None}
        if (out / "QUIESCENCE_UNCONFIRMED").exists() or stop.get("step_group_gone", True) is not True:
            self.log("quiescence not confirmed by the VM script")
            self.stop_confirmed = False
        self.log("status.json:", json.dumps(s))
        self.log("validation:", "PASS" if not problems else "; ".join(problems))
        return not problems

    def cleanup(self):
        if not self.intent:
            return True
        for attempt in range(1, 6):
            try:
                if time.monotonic() >= self.shutdown_end:
                    break
                self.call([COL, "stop", "-s", self.name], 120, self.shutdown_end, f"stop {attempt}")
                self.sleep(MINUTE / 6, self.shutdown_end)
                rows = self.sessions(self.shutdown_end)
                if rows is None:
                    continue
                names = {n for n, _ in rows}
                endpoints = {ep for _, ep in rows}
                new = endpoints - (self.before or set()) - ({self.endpoint} if self.endpoint else set())
                ours_listed = self.name in names or (self.endpoint is not None and self.endpoint in endpoints)
                if not ours_listed and not new:
                    self.log(f"CLEANUP OK: {self.name} ({self.endpoint}) and no new endpoint listed")
                    return True
                if self.endpoint is not None and self.endpoint in endpoints and self.name not in names:
                    self.call(ADOPT + ["--adopt", f"{self.endpoint}={self.name}"], 180, self.shutdown_end, "adopt")
                if new:
                    self.log("endpoints new since our creation call, ownership not established:", sorted(new))
            except Exception as e:
                self.log(f"cleanup attempt {attempt} error:", repr(e))
        self.log(f"CLEANUP UNRESOLVED: {self.name} (endpoint {self.endpoint}) may still be running, or a VM our "
                 f"creation call allocated is unnamed. Inspect `{COL} sessions`; stop ours with "
                 f"`{COL} stop -s {self.name}` or adopt the endpoint first (tools/colab_adopt.py --adopt EP=NAME).")
        return False

    def main(self):
        self.out.mkdir(parents=True, exist_ok=False)
        try:
            self.log_file = open(self.out / "driver.log", "a")
        except OSError:
            self.log_file = None
        self.log(f"run {self.run} commit {self.commit} source {SOURCE} session {self.name}"
                 + (" [TEST HARNESS]" if TEST else ""))
        logs = receipts = False
        valid = None
        try:
            self.stage_inputs()
            self.preflight()
            self.check()
            self.create()
            self.check()
            if not self.upload():
                raise Stop("inputs not verified on the VM: not launching")
            launched = self.launch()
            done = self.poll() if launched else False
            self.done = done
            self.log("launched", launched, "done", done, "cancelled", self.cancel, "home", sorted(self.home))
            if not done:
                self.stop_confirmed = self.stop_workload()
            logs = self.retrieve()
            valid = self.validate() if logs else None
        except Stop as e:
            self.log("stopped:", e)
            if self.launch_attempted and not self.done:
                self.stop_confirmed = self.stop_workload()
                logs = self.retrieve()
                valid = self.validate() if logs else None
        except Exception as e:
            self.log("error:", repr(e))
        finally:
            clean = False
            try:
                clean = self.cleanup()
            except Exception as e:
                self.log("cleanup error:", repr(e))
            for label, argv in (("final sessions", [COL, "sessions"]), ("final orphans", REAP)):
                rc, out = self.call(argv, 120, self.inventory_end, label)
                self.log(f"{label}:\n" + out.strip())
            if self.out.joinpath("staging").exists():
                shutil.rmtree(self.out / "staging", ignore_errors=True)  # copies of pinned inputs only
            receipts = logs and self.copy_receipts()
            all_home = set(self.home) == set(SCREENS)
            code = 3 if not clean else 2 if not (logs and receipts and valid is not None) else 0 if valid else 1
            if code == 0 and not (self.done and all_home):
                self.log(f"validated status but done {self.done}, home {sorted(self.home)}: exit 2")
                code = 2
            if self.log_failed:
                code = max(code, 2)
            if self.stop_confirmed is False:
                self.log("the workload stop was not confirmed: exit at least 2")
                code = max(code, 2)
            self.log(f"exit {code} (logs {logs}, receipts {receipts}, valid {valid}, cleanup {clean}, "
                     f"home {sorted(self.home)})")
            try:
                if self.log_file:
                    self.log_file.close()
                if receipts:
                    shutil.copy2(self.out / "driver.log", self.receipts / "driver.log")
            except OSError:
                code = max(code, 2)
            return code


def main():
    if not TEST and any(k.startswith("DRIVE_TEST_") for k in os.environ):
        print("DRIVE_TEST_* is set but DRIVE_TEST_HARNESS is not: refusing a real run with test hooks",
              file=sys.stderr)
        return 2
    if len(sys.argv) != 3 or not re.fullmatch(r"[0-9a-f]{40}", sys.argv[1]) \
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", sys.argv[2]):
        print(__doc__, file=sys.stderr)
        return 2
    d = Driver(sys.argv[1], sys.argv[2])

    def on_signal(signum, _frame):
        d.cancel = True

    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, on_signal)
    return d.main()


if __name__ == "__main__":
    sys.exit(main())
