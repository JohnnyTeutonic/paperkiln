"""Tests of the screening drive.py against a mock colab CLI (no network, no Colab). python3 -m unittest test_drive -v

The mock maps the VM's /content to a temporary folder. Uploads and downloads copy files; exec runs the driver's own
VM-side code with /content rewritten to that folder, except the steps that need a real VM (launch, stop, the HF
download), which the mock simulates. After launch, each poll makes the next screen READY (GGUF parts, manifest,
train.log), and after the last screen the VM writes status.json and DONE.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

HERE = Path(__file__).resolve().parent
COMMIT = "0123456789abcdef0123456789abcdef01234567"
SOURCE = "1ccb693f55d7328d4a8b0b7238fe8a2cafb6edf1"
SCREENS = ["S0", "S1", "S2", "S3"]
GATES = sorted(["provenance", "checkout", "inputs", "configure", "build", "specs"]
               + [f"{g}_{s}" for s in SCREENS for g in ("train", "package")])

MOCK = r'''#!/usr/bin/env python3
import hashlib, json, os, re, shutil, subprocess, sys, time
from pathlib import Path
state_p = Path(os.environ["MOCK_STATE"]); content = Path(os.environ["MOCK_CONTENT"]); mode = os.environ["MOCK_MODE"]
hub = Path(os.environ["MOCK_HUB"])
st = json.loads(state_p.read_text()) if state_p.exists() else {"sessions": {}, "counts": {}}
save = lambda: state_p.write_text(json.dumps(st))
def count(k):
    st["counts"][k] = st["counts"].get(k, 0) + 1; save(); return st["counts"][k]
def vm(path):
    return content / path[len("/content/"):] if path.startswith("/content/") else Path(path)
def sha(b):
    return hashlib.sha256(b).hexdigest()
args = sys.argv[1:]
cmd = args[0]
if cmd == "sessions":
    for n, ep in st["sessions"].items(): print(f"[{n}] {ep} L4")
    sys.exit(0)
name = args[args.index("-s") + 1]
if cmd == "new":
    st["sessions"][name] = "ep-" + name; st["created"] = True; count("new"); save(); sys.exit(0)
if cmd == "stop":
    if mode != "stop_ignored": st["sessions"].pop(name, None); save()
    sys.exit(0)
if cmd == "upload":
    src, dst = Path(args[-2]), vm(args[-1])
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = count("upload:" + dst.name)
    data = src.read_bytes()
    if mode == "upload_corrupt_once" and dst.name == "base.part01" and n == 1:
        data = b"x" + data[1:]
    dst.write_bytes(data)
    if not dst.name.startswith("."):
        count("uploads_data" if dst.name != "manifest.json" else "uploads_manifest")
    sys.exit(0)
if cmd == "download":
    src, dst = vm(args[-2]), Path(args[-1])
    if not src.exists(): sys.exit(1)
    n = count("download:" + str(src))
    data = src.read_bytes()
    if (mode == "bad_part_once" and n == 1 or mode == "bad_part_always") and src.name == "model.gguf.part01":
        data = b"y" + data[1:]
    dst.write_bytes(data); sys.exit(0)
def advance(run):
    """One step of VM progress: the next screen READY, or status.json and DONE after the last one."""
    out = content / f"run_{run}" / "out"
    k = st.get("progress", 0)
    fail_at = 2 if mode == "gate_fail" else None
    gates = {g: 0 for g in os.environ["MOCK_GATES"].split()}
    if k < 4 and k != fail_at and mode != "never_done":
        s = f"S{k}"
        d = out / "screens" / s; d.mkdir(parents=True, exist_ok=True)
        gguf = (s * 1300).encode()
        parts = []
        for i in range(0, len(gguf), 1000):
            p = d / f"model.gguf.part{i // 1000:02d}"; p.write_bytes(gguf[i:i + 1000])
            parts.append({"name": p.name, "sha256": sha(p.read_bytes()), "bytes": p.stat().st_size})
        (d / "manifest.json").write_text(json.dumps({"screen": s, "run_id": run, "gguf_sha256": sha(gguf),
                                                     "gguf_bytes": len(gguf), "parts": parts}))
        (d / "train.log").write_text('{"event": "step", "step": 256, "loss": 2.5}\n')
        (d / "READY").write_text(run)
        st["progress"] = k + 1; save()
        return
    if mode == "never_done" or "status" in st:
        return
    if fail_at is not None:
        for g in [g for g in gates if g.endswith(("_S2", "_S3"))]: del gates[g]
        gates["train_S2"] = 1
    head = SOURCE if mode != "wrong_source" else "e" * 40
    s = {"run_id": run, "commit": os.environ["MOCK_COMMIT"], "source": SOURCE, "head": head,
         "manifest_sha256": st["manifest_sha"], "exit_status": 0 if fail_at is None else 1,
         "pass": fail_at is None, "packaged": [f"S{i}" for i in range(st.get("progress", 0))], "gates": gates}
    (out / "status.json").write_text(json.dumps(s)); (out / "DONE").write_text(run)
    st["status"] = s; save()
SOURCE = "1ccb693f55d7328d4a8b0b7238fe8a2cafb6edf1"
if cmd == "exec":
    f = args[args.index("-f") + 1]
    code = Path(f).read_text()
    m = re.match(r".*claude_screens_(.+?)_(mkdir_in|vm_hashes\d|hf_fetch|launch_retry|launch_probe\d|launch|poll|"
                 r"final_state|pack\d|stop_workload\d)\.py$", f)
    run, label = m.group(1), m.group(2)
    out = content / f"run_{run}" / "out"
    if label in ("launch", "launch_retry"):
        count("launch")
        msha = re.search(r"vm_run_\S+\.sh \S+ \S+ ([0-9a-f]{64}) ", code).group(1)
        st["manifest_sha"] = msha; save()
        out.mkdir(parents=True, exist_ok=True)
        (content / f"vm_{run}.pid").write_text("4242")
        print("LAUNCHED"); sys.exit(0)
    if label.startswith("launch_probe"):
        print("STARTED" if (content / f"vm_{run}.pid").exists() else "ABSENT"); sys.exit(0)
    if label == "hf_fetch":
        count("hf_fetch")
        names = json.loads(re.search(r"for n in (\[.*?\]):", code).group(1).replace("'", '"'))
        ok = True
        for n in names:
            src = hub / n
            if src.exists(): shutil.copyfile(src, content / "screens_in" / n)
            else: ok = False; print("FAILED", n, "EntryNotFoundError")
        tok = content / "screens_in" / ".hf_token"
        st["token_seen"] = tok.exists(); tok.unlink(missing_ok=True); save()
        print("HF_DONE" if ok else "HF_PARTIAL", 0); sys.exit(0)
    if label.startswith("stop_workload"):
        count("stop_workload")
        s = st.get("status") or {"run_id": run, "commit": os.environ["MOCK_COMMIT"], "source": SOURCE,
                                 "head": SOURCE, "manifest_sha256": st.get("manifest_sha"), "exit_status": 1,
                                 "pass": False, "packaged": [], "gates": {"provenance": 0, "train_S0": 143}}
        out.mkdir(parents=True, exist_ok=True)
        (out / "status.json").write_text(json.dumps(s))
        (out / "stop.json").write_text(json.dumps({"step": "train_S0", "step_started": True, "step_group_gone": True}))
        (out / "DONE").write_text(run)
        print('STOP_REPLY {"result": "interrupted"}'); sys.exit(0)
    if label == "poll":
        advance(run)
    code = code.replace("/content", str(content))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    sys.stdout.write(r.stdout); sys.stderr.write(r.stderr); sys.exit(r.returncode)
sys.exit(2)
'''


def write_inputs(tmp, bad_pin=False):
    """Small stand-ins for every pinned input, and their pins (the base weights span several parts)."""
    d = tmp / "inputs"
    d.mkdir()
    pins = {}
    names = ["sfull_sft1500.safetensors", "vocab.gguf", "S0.jsonl", "S1.jsonl", "S2.jsonl", "S3.jsonl",
             "arm_a_lr3e5.json"]
    for i, n in enumerate(names):
        data = (n * (300 if n.startswith("sfull") else 3)).encode()
        (d / n).write_bytes(data)
        pins[n] = [str(d / n), hashlib.sha256(data).hexdigest()]
    if bad_pin:
        pins["S2.jsonl"][1] = "0" * 64
    return pins


def seed_hub(tmp, pins, missing=()):
    """The HF repo stand-in: every data file under its VM name, base weights as the driver's parts."""
    hub = tmp / "hub"
    hub.mkdir()
    for n, (path, _) in pins.items():
        if n == "sfull_sft1500.safetensors":
            data = Path(path).read_bytes()
            for i in range(0, len(data), 2000):
                (hub / f"base.part{i // 2000:02d}").write_bytes(data[i:i + 2000])
        else:
            (hub / n).write_bytes(Path(path).read_bytes())
    for n in missing:
        (hub / n).unlink()
    return hub


class DriveTests(unittest.TestCase):
    def run_drive(self, mode, run="t1", bad_pin=False, hf=False, hub_missing=(), minute="0.05"):
        tmp = Path(tempfile.mkdtemp())
        mock = tmp / "colab"
        mock.write_text(MOCK)
        mock.chmod(0o755)
        (tmp / "content").mkdir()
        pins = write_inputs(tmp, bad_pin)
        hub = seed_hub(tmp, pins, hub_missing)
        tok, rev = tmp / "hf_token", tmp / "hf_revision"
        if hf:
            tok.write_text("hf_fixture_token\n")
            rev.write_text("a" * 40 + "\n")
        env = dict(os.environ, DRIVE_TEST_HARNESS="1", DRIVE_TEST_COLAB=str(mock), DRIVE_TEST_CLI_PY=sys.executable,
                   DRIVE_TEST_REAP="true", DRIVE_TEST_ADOPT="true", DRIVE_TEST_ARTIFACTS=str(tmp / "art"),
                   DRIVE_TEST_RECEIPTS=str(tmp / "receipts"), DRIVE_TEST_MINUTE_SECONDS=minute,
                   DRIVE_TEST_PINS=json.dumps(pins), DRIVE_TEST_PART_BYTES="2000",
                   DRIVE_TEST_HF_TOKEN_FILE=str(tok), DRIVE_TEST_HF_REVISION_FILE=str(rev),
                   MOCK_STATE=str(tmp / "state.json"), MOCK_CONTENT=str(tmp / "content"), MOCK_MODE=mode,
                   MOCK_HUB=str(hub), MOCK_GATES=" ".join(GATES), MOCK_COMMIT=COMMIT)
        t0 = time.monotonic()
        r = subprocess.run([sys.executable, str(HERE / "drive.py"), COMMIT, run], env=env, capture_output=True,
                           text=True, timeout=180)
        state = json.loads((tmp / "state.json").read_text()) if (tmp / "state.json").exists() else {
            "sessions": {}, "counts": {}}
        log = (tmp / "art" / run / "driver.log").read_text() if (tmp / "art" / run / "driver.log").exists() else ""
        return r.returncode, state, log, tmp, time.monotonic() - t0

    def assert_home(self, tmp, screens, run="t1"):
        for s in SCREENS:
            gguf = tmp / "art" / run / s / "model.gguf"
            if s in screens:
                self.assertEqual(gguf.read_bytes(), (s * 1300).encode(), s)
                self.assertFalse(list((tmp / "art" / run / s).glob("model.gguf.part*")), s)
            else:
                self.assertFalse(gguf.exists(), s)

    def test_normal_run_collects_every_screen_and_stops(self):
        rc, state, log, tmp, _ = self.run_drive("normal")
        self.assertEqual(rc, 0, log)
        self.assertEqual(state["sessions"], {})
        self.assertIn("CLEANUP OK", log)
        self.assertIn("validation: PASS", log)
        self.assert_home(tmp, SCREENS)
        receipts = tmp / "receipts" / "t1"
        for f in ("status.json", "driver.log", "archive.sha256", "home.json", "S3_manifest.json"):
            self.assertTrue((receipts / f).is_file(), f)
        self.assertEqual(sorted(json.loads((receipts / "home.json").read_text())), SCREENS)
        self.assertNotIn("staging", (receipts / "archive.sha256").read_text())
        self.assertFalse((tmp / "art" / "t1" / "staging").exists())
        self.assertEqual(state["counts"].get("hf_fetch", 0), 0)

    def test_a_pin_mismatch_creates_nothing(self):
        rc, state, log, tmp, _ = self.run_drive("normal", bad_pin=True)
        self.assertEqual(rc, 2, log)
        self.assertIn("S2.jsonl hashes", log)
        self.assertEqual(state["counts"].get("new", 0), 0)

    def test_a_corrupted_upload_is_sent_again(self):
        rc, state, log, tmp, _ = self.run_drive("upload_corrupt_once")
        self.assertEqual(rc, 0, log)
        self.assertIn("upload round 1:", log)
        self.assertIn("['base.part01']", log)
        self.assertEqual(state["counts"]["upload:base.part01"], 2)

    def test_hf_route_uploads_only_the_manifest_and_removes_the_token(self):
        rc, state, log, tmp, _ = self.run_drive("normal", hf=True)
        self.assertEqual(rc, 0, log)
        self.assertEqual(state["counts"]["hf_fetch"], 1)
        self.assertEqual(state["counts"].get("uploads_data", 0), 0)
        self.assertEqual(state["counts"]["uploads_manifest"], 1)
        self.assertTrue(state["token_seen"])
        self.assertFalse((tmp / "content" / "screens_in" / ".hf_token").exists())
        self.assertNotIn("hf_fixture_token", log)
        self.assertIn("at revision " + "a" * 40, log)

    def test_hf_route_falls_back_to_upload_for_a_missing_file(self):
        rc, state, log, tmp, _ = self.run_drive("normal", hf=True, hub_missing=("S1.jsonl",))
        self.assertEqual(rc, 0, log)
        self.assertEqual(state["counts"]["uploads_data"], 1)
        self.assertEqual(state["counts"]["upload:S1.jsonl"], 1)

    def test_hf_route_needs_a_verified_revision(self):
        rc, state, log, tmp, _ = self.run_drive("normal", hf=False)
        self.assertEqual(rc, 0, log)
        self.assertIn("no HF token or verified revision", log)

    def test_a_corrupted_part_is_downloaded_again(self):
        rc, state, log, tmp, _ = self.run_drive("bad_part_once")
        self.assertEqual(rc, 0, log)
        self.assert_home(tmp, SCREENS)

    def test_a_part_that_never_verifies_leaves_its_screen_away(self):
        rc, state, log, tmp, _ = self.run_drive("bad_part_always")
        self.assertEqual(rc, 2, log)
        self.assert_home(tmp, [])
        self.assertIn("part model.gguf.part01 not verified", log)

    def test_a_failed_screen_keeps_the_screens_already_home(self):
        rc, state, log, tmp, _ = self.run_drive("gate_fail")
        self.assertEqual(rc, 1, log)
        self.assert_home(tmp, ["S0", "S1"])
        self.assertIn("failed gates ['train_S2']", log)
        self.assertEqual(state["sessions"], {})

    def test_a_source_mismatch_is_not_validated(self):
        rc, state, log, tmp, _ = self.run_drive("wrong_source")
        self.assertEqual(rc, 2, log)
        self.assertIn("identity mismatch", log)

    def test_a_vm_that_never_finishes_is_stopped_and_retrieved(self):
        rc, state, log, tmp, _ = self.run_drive("never_done", minute="0.3")
        self.assertIn(rc, (1, 2), log)
        self.assertGreaterEqual(state["counts"]["stop_workload"], 1)
        self.assertIn("workload stop: confirmed (interrupted)", log)
        self.assertEqual(state["sessions"], {})

    def test_a_session_that_will_not_stop_exits_3(self):
        rc, state, log, tmp, _ = self.run_drive("stop_ignored")
        self.assertEqual(rc, 3, log)
        self.assertIn("CLEANUP UNRESOLVED", log)

    def test_test_hooks_are_refused_without_the_harness(self):
        env = dict(os.environ, DRIVE_TEST_COLAB="/bin/true")
        env.pop("DRIVE_TEST_HARNESS", None)
        r = subprocess.run([sys.executable, str(HERE / "drive.py"), COMMIT, "t1"], env=env, capture_output=True,
                           text=True)
        self.assertEqual(r.returncode, 2)
        self.assertIn("refusing a real run with test hooks", r.stderr)


if __name__ == "__main__":
    unittest.main()
