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
    if mode == "sessions_fail" and st.get("created"):
        print("error: auth", file=sys.stderr); sys.exit(1)
    for n, ep in st["sessions"].items(): print(f"[{n}] {ep} L4")
    sys.exit(0)
name = args[args.index("-s") + 1]
if cmd == "new" and mode == "sibling":
    st["sessions"][name] = "ep-" + name; st["sessions"]["claude-screens-other"] = "ep-other"
    st["created"] = True; count("new"); save(); sys.exit(0)
if cmd == "new":
    if mode == "orphan_new":
        st["sessions"]["?"] = "ep-orphan"; st["created"] = True; count("new"); save(); sys.exit(1)
    st["sessions"][name] = "ep-" + name; st["created"] = True; count("new"); save()
    if mode == "new_timeout": time.sleep(60)
    sys.exit(0)
if cmd == "stop":
    if mode != "stop_ignored":
        st["sessions"].pop(name, None)
        save()
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
    screens = os.environ["MOCK_SCREENS"].split(",")
    fail_at = 2 if mode == "gate_fail" else None
    gates = {g: 0 for g in os.environ["MOCK_GATES"].split()}
    if k < len(screens) and k != fail_at and mode not in ("never_done", "stop_unconfirmed"):
        s = screens[k]
        d = out / "screens" / s; d.mkdir(parents=True, exist_ok=True)
        gguf = (s * 1300).encode()
        parts = []
        for i in range(0, len(gguf), 1000):
            p = d / f"model.gguf.part{i // 1000:02d}"; p.write_bytes(gguf[i:i + 1000])
            parts.append({"name": p.name, "sha256": sha(p.read_bytes()), "bytes": p.stat().st_size})
        (d / "train.log").write_text('{"event": "step", "step": 256, "loss": 2.5}\n')
        (d / "result.json").write_text('{"final_step": 256}')
        (d / "manifest.json").write_text(json.dumps({"screen": s, "run_id": run, "gguf_sha256": sha(gguf),
                                                     "gguf_bytes": len(gguf), "parts": parts,
                                                     "train_log_sha256": sha((d / "train.log").read_bytes()),
                                                     "result_json_sha256": sha((d / "result.json").read_bytes())}))
        (d / "READY").write_text(run)
        st["progress"] = k + 1; save()
        return
    if mode in ("never_done", "stop_unconfirmed") or "status" in st:
        return
    if fail_at is not None:
        for g in [g for g in gates if g.endswith(tuple("_" + x for x in screens[2:]))]: del gates[g]
        gates["train_" + screens[2]] = 1
    head = SOURCE if mode != "wrong_source" else "e" * 40
    s = {"run_id": run, "seed": int(os.environ["MOCK_SEED"]), "screens": screens,
         "commit": os.environ["MOCK_COMMIT"], "source": SOURCE, "head": head,
         "manifest_sha256": st["manifest_sha"], "exit_status": 0 if fail_at is None else 1,
         "pass": fail_at is None, "packaged": screens[:st.get("progress", 0)], "gates": gates}
    (out / "status.json").write_text(json.dumps(s)); (out / "DONE").write_text(run)
    st["status"] = s; save()
SOURCE = "1ccb693f55d7328d4a8b0b7238fe8a2cafb6edf1"
if cmd == "exec":
    f = args[args.index("-f") + 1]
    code = Path(f).read_text()
    m = re.match(r".*claude_screens_(.+?)_(mkdir_in|vm_hashes\d|hf_fetch|launch_retry|launch_probe\d|launch|poll|"
                 r"final_state|pack\d|stop_workload\d|token_remove\d)\.py$", f)
    run, label = m.group(1), m.group(2)
    out = content / f"run_{run}" / "out"
    if label in ("launch", "launch_retry"):
        n = count("launch")
        if mode == "launch_lost_absent" and n == 1:
            sys.exit(1)
        msha = re.search(r"vm_run_\S+\.sh \S+ \S+ ([0-9a-f]{64}) (\d) (\S+) ", code).group(1)
        st["manifest_sha"] = msha; save()
        out.mkdir(parents=True, exist_ok=True)
        (content / f"vm_{run}.pid").write_text("4242")
        if mode == "launch_lost_started" and n == 1:
            sys.exit(1)
        print("LAUNCHED"); sys.exit(0)
    if label == "mkdir_in" and (mode == "mkdir_never" or (mode == "mkdir_fail_once" and count("mkdir") == 1)):
        sys.exit(1)
    if label.startswith("launch_probe"):
        print("STARTED" if (content / f"vm_{run}.pid").exists() else "ABSENT"); sys.exit(0)
    if label == "hf_fetch" and mode == "hf_exec_fails":
        count("hf_fetch"); sys.exit(1)
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
        if mode == "stop_unconfirmed":
            print('STOP_REPLY {"result": "unconfirmed"}'); sys.exit(0)
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
    if label.startswith("pack") and mode == "logs_lost":
        sys.exit(1)
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
             "S4.jsonl", "S5.jsonl", "S6.jsonl", "S7.jsonl", "S8.jsonl", "arm_a_lr3e5.json"]
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
    def run_drive(self, mode, run="t1", bad_pin=False, hf=False, hub_missing=(), minute="0.1", extra=None,
                  pre=None, seed=None, screens=None):
        use = screens or SCREENS
        gates = sorted(["provenance", "checkout", "inputs", "configure", "build", "specs"]
                       + [f"{g}_{s}" for s in use for g in ("train", "package")])
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
                   MOCK_HUB=str(hub), MOCK_GATES=" ".join(gates), MOCK_COMMIT=COMMIT,
                   MOCK_SEED=str(seed or 1), MOCK_SCREENS=",".join(use))
        env.update(extra or {})
        if pre:
            pre(tmp)
        t0 = time.monotonic()
        argv = [sys.executable, str(HERE / "drive.py"), COMMIT, run] + (
            [str(seed), ",".join(screens)] if seed else []) + (
            [extra["DRIVE_TEST_CAP"]] if extra and "DRIVE_TEST_CAP" in extra else [])
        r = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=180)
        state = json.loads((tmp / "state.json").read_text()) if (tmp / "state.json").exists() else {
            "sessions": {}, "counts": {}}
        log = (tmp / "art" / run / "driver.log").read_text() if (tmp / "art" / run / "driver.log").exists() else ""
        return r.returncode, state, log, tmp, time.monotonic() - t0

    def assert_home(self, tmp, screens, run="t1", universe=None):
        for s in universe or SCREENS:
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

    def test_a_failed_github_precondition_creates_nothing(self):
        rc, state, log, tmp, _ = self.run_drive("normal", extra={"DRIVE_TEST_GITHUB": "fail"})
        self.assertEqual(rc, 2, log)
        self.assertIn("github preflight failed", log)
        self.assertEqual(state["counts"].get("new", 0), 0)

    def test_a_failed_hf_exec_removes_the_token_and_uploads(self):
        rc, state, log, tmp, _ = self.run_drive("hf_exec_fails", hf=True)
        self.assertEqual(rc, 0, log)
        self.assertFalse((tmp / "content" / "screens_in" / ".hf_token").exists())
        self.assertGreater(state["counts"]["uploads_data"], 0)
        self.assertNotIn("hf_fixture_token", log)

    def test_receipts_are_kept_when_the_logs_are_lost(self):
        rc, state, log, tmp, _ = self.run_drive("logs_lost")
        self.assertEqual(rc, 2, log)
        receipts = tmp / "receipts" / "t1"
        self.assertEqual(sorted(json.loads((receipts / "home.json").read_text())), SCREENS)
        for f in ("input_manifest.json", "S0_manifest.json", "driver.log", "archive.sha256"):
            self.assertTrue((receipts / f).is_file(), f)

    def test_screen_results_and_logs_come_home_hashed(self):
        rc, state, log, tmp, _ = self.run_drive("normal")
        self.assertEqual(rc, 0, log)
        for s in SCREENS:
            self.assertTrue((tmp / "art" / "t1" / s / "result.json").is_file(), s)
        self.assertTrue((tmp / "receipts" / "t1" / "input_manifest.json").is_file())

    def test_an_existing_run_folder_is_refused(self):
        rc, state, log, tmp, _ = self.run_drive("normal", pre=lambda t: (t / "art" / "t1").mkdir(parents=True))
        self.assertEqual(rc, 2)
        self.assertEqual(state["counts"].get("new", 0), 0)

    def test_a_lost_launch_acknowledgement_is_reconciled_not_repeated(self):
        rc, state, log, tmp, _ = self.run_drive("launch_lost_started")
        self.assertEqual(rc, 0, log)
        self.assertEqual(state["counts"]["launch"], 1)
        self.assertIn("the VM script is started", log)

    def test_an_absent_launch_is_tried_once_more(self):
        rc, state, log, tmp, _ = self.run_drive("launch_lost_absent")
        self.assertEqual(rc, 0, log)
        self.assertEqual(state["counts"]["launch"], 2)

    def test_an_unconfirmed_stop_exits_at_least_2(self):
        rc, state, log, tmp, _ = self.run_drive("stop_unconfirmed", minute="0.3")
        self.assertGreaterEqual(rc, 2, log)
        self.assertIn("NOT CONFIRMED", log)

    def test_a_failed_listing_is_never_certified_clean(self):
        rc, state, log, tmp, _ = self.run_drive("sessions_fail")
        self.assertEqual(rc, 3, log)
        self.assertNotIn("CLEANUP OK", log)

    def test_an_unnamed_allocation_is_never_certified(self):
        rc, state, log, tmp, _ = self.run_drive("orphan_new")
        self.assertNotIn("CLEANUP OK", log)
        self.assertIn("creation not confirmed", log)

    def test_a_hanging_creation_call_still_stops_the_session(self):
        rc, state, log, tmp, _ = self.run_drive("new_timeout", minute="0.3")
        self.assertEqual(state["sessions"], {}, log)

    def test_a_log_failure_exits_at_least_2(self):
        rc, state, log, tmp, _ = self.run_drive("normal", extra={"DRIVE_TEST_LOG_FAIL_AFTER": "5"})
        self.assertGreaterEqual(rc, 2)

    def test_the_launch_code_publishes_vm_run_atomically(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("drive_mod", HERE / "drive.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        code = mod.launch_code("https://example.invalid/vm_run.sh", COMMIT, "t1", "a" * 64)
        self.assertIn("tempfile.mkstemp", code)
        self.assertIn("os.replace(tmp", code)
        self.assertIn("a" * 64, code)

    def test_the_deadlines_fit_the_session_cap(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("drive_mod2", HERE / "drive.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertEqual(list(mod.DEADLINES), sorted(mod.DEADLINES))
        self.assertLessEqual(mod.DEADLINES[-1], 120)

    def test_a_wave_2_session_trains_its_screens_at_its_seed(self):
        rc, state, log, tmp, _ = self.run_drive("normal", seed=2, screens=["S0", "S2", "S4"])
        self.assertEqual(rc, 0, log)
        self.assert_home(tmp, ["S0", "S2", "S4"], universe=["S0", "S1", "S2", "S3", "S4"])
        man = json.loads((tmp / "receipts" / "t1" / "input_manifest.json").read_text())
        self.assertEqual((man["seed"], man["screens"]), (2, ["S0", "S2", "S4"]))
        self.assertEqual(sorted(man["files"]), sorted(["sfull_sft1500.safetensors", "vocab.gguf", "arm_a_lr3e5.json",
                                                       "S0.jsonl", "S2.jsonl", "S4.jsonl"]))

    def test_a_status_with_another_seed_is_not_validated(self):
        rc, state, log, tmp, _ = self.run_drive("normal", seed=2, screens=["S0", "S2", "S4"],
                                                extra={"MOCK_SEED": "3"})
        self.assertEqual(rc, 1, log)
        self.assertIn("seed or screens differ", log)

    def test_bad_seed_or_screens_are_refused(self):
        for extra in (["0", "S0"], ["2", "S0,S9"], ["2", "S0,S0"], ["x", "S0"]):
            r = subprocess.run([sys.executable, str(HERE / "drive.py"), COMMIT, "t1"] + extra,
                               env=dict(os.environ, DRIVE_TEST_HARNESS="1"), capture_output=True, text=True)
            self.assertEqual(r.returncode, 2, extra)

    def test_a_parallel_sibling_session_does_not_block_cleanup(self):
        rc, state, log, tmp, _ = self.run_drive("sibling")
        self.assertEqual(rc, 0, log)
        self.assertIn("CLEANUP OK", log)
        self.assertIn("claude-screens-other", state["sessions"])

    def test_three_screen_sessions_use_the_short_cap(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("drive_mod3", HERE / "drive.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertEqual(mod.DEADLINES_SHORT[-1], 100)
        self.assertEqual(list(mod.DEADLINES_SHORT), sorted(mod.DEADLINES_SHORT))

    def test_a_wave_3_session_trains_only_s5(self):
        rc, state, log, tmp, _ = self.run_drive("normal", seed=3, screens=["S5"])
        self.assertEqual(rc, 0, log)
        self.assert_home(tmp, ["S5"], universe=["S0", "S1", "S2", "S3", "S4", "S5"])
        man = json.loads((tmp / "receipts" / "t1" / "input_manifest.json").read_text())
        self.assertEqual((man["seed"], man["screens"]), (3, ["S5"]))
        self.assertEqual(sorted(man["files"]), sorted(["sfull_sft1500.safetensors", "vocab.gguf", "arm_a_lr3e5.json",
                                                       "S5.jsonl"]))

    def test_a_capped_diversity_session(self):
        rc, state, log, tmp, _ = self.run_drive("normal", seed=2, screens=["S7", "S8", "S6"],
                                                extra={"DRIVE_TEST_CAP": "60"}, minute="0.3")
        self.assertEqual(rc, 0, log)
        self.assert_home(tmp, ["S6", "S7", "S8"], universe=["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"])

    def test_the_cap_sets_the_deadlines_and_is_bounded(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("drive_mod4", HERE / "drive.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertEqual(mod.capped(60), (40, 44, 52, 57, 60))
        for bad in ("30", "121", "6x"):
            r = subprocess.run([sys.executable, str(HERE / "drive.py"), COMMIT, "t1", "1", "S6", bad],
                               env=dict(os.environ, DRIVE_TEST_HARNESS="1"), capture_output=True, text=True)
            self.assertEqual(r.returncode, 2, bad)

    def test_a_vm_that_answers_late_still_gets_its_inputs(self):
        rc, state, log, tmp, _ = self.run_drive("mkdir_fail_once")
        self.assertEqual(rc, 0, log)
        self.assertIn("VM not ready for inputs (attempt 1)", log)

    def test_a_vm_that_never_answers_gets_no_uploads(self):
        rc, state, log, tmp, _ = self.run_drive("mkdir_never", minute="0.3")
        self.assertEqual(rc, 2, log)
        self.assertIn("the VM never answered", log)
        self.assertEqual(state["counts"].get("uploads_data", 0), 0)
        self.assertEqual(state["sessions"], {})

    def test_test_hooks_are_refused_without_the_harness(self):
        env = dict(os.environ, DRIVE_TEST_COLAB="/bin/true")
        env.pop("DRIVE_TEST_HARNESS", None)
        r = subprocess.run([sys.executable, str(HERE / "drive.py"), COMMIT, "t1"], env=env, capture_output=True,
                           text=True)
        self.assertEqual(r.returncode, 2)
        self.assertIn("refusing a real run with test hooks", r.stderr)


if __name__ == "__main__":
    unittest.main()
