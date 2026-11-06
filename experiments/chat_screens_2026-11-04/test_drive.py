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
    if mode == "bad_st_always" and src.name == "model.safetensors.part00":
        data = b"z" + data[1:]
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
        stb = (s * 900).encode()
        st_parts = []
        for i in range(0, len(stb), 1000):
            p = d / f"model.safetensors.part{i // 1000:02d}"; p.write_bytes(stb[i:i + 1000])
            st_parts.append({"name": p.name, "sha256": sha(p.read_bytes()), "bytes": p.stat().st_size})
        (d / "train.log").write_text('{"event": "step", "step": 256, "loss": 2.5}\n')
        (out / f"spec_{s}.json").write_text('{"name": "screen"}')
        (out / f"train_{s}_check.json").write_text('{"ok": true}')
        (d / "result.json").write_text('{"final_step": 256}')
        (d / "manifest.json").write_text(json.dumps({"screen": s, "run_id": run, "gguf_sha256": sha(gguf),
                                                     "gguf_bytes": len(gguf), "parts": parts,
                                                     "st_sha256": sha(stb), "st_bytes": len(stb),
                                                     "st_parts": st_parts,
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
    m = re.match(r".*claude_screens_(.+?)_(early_read|mkdir_in|vm_hashes\d|hf_fetch|launch_retry|launch_probe\d|launch|poll|"
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
        if os.environ.get("MOCK_TRAIN_LOG"):
            [(out / f"train_{s}.log").write_text(os.environ["MOCK_TRAIN_LOG"]) for s in ("S35", "S54", "S55")]
        if mode == "launch_lost_started" and n == 1:
            sys.exit(1)
        print("LAUNCHED"); sys.exit(0)
    if label == "early_read":
        if count("early_read") <= int(os.environ.get("MOCK_EARLY_FAIL", "0")):
            sys.exit(1)
    if label == "mkdir_in" and (mode == "mkdir_never" or (mode == "mkdir_fail_once" and count("mkdir") == 1)):
        sys.exit(1)
    if label.startswith("launch_probe"):
        print("STARTED" if (content / f"vm_{run}.pid").exists() else "ABSENT"); sys.exit(0)
    if label == "hf_fetch" and mode == "hf_fail_once" and not st["counts"].get("hf_fetch"):
        count("hf_fetch"); sys.exit(1)
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


def write_inputs(tmp, bad_pin=False, recipes=None):
    """Small stand-ins for every pinned input, and their pins (the base weights span several parts)."""
    d = tmp / "inputs"
    d.mkdir()
    pins = {}
    names = ["sfull_sft1500.safetensors", "pk_sft_v1.safetensors", "S54.jsonl", "S55.jsonl", "vocab.gguf", "S0.jsonl", "S1.jsonl", "S2.jsonl", "S3.jsonl",
             "S4.jsonl", "S5.jsonl", "S6.jsonl", "S7.jsonl", "S8.jsonl", "S9.jsonl", "S10.jsonl",
             "S11.jsonl", "S12.jsonl", "S13.jsonl", "S14.jsonl", "S15.jsonl", "S16.jsonl", "S24.jsonl", "S25.jsonl", "S26.jsonl", "S27.jsonl", "S21.jsonl", "S22.jsonl", "S23.jsonl", "S28.jsonl", "S29.jsonl", "S30.jsonl", "S31.jsonl", "S32.jsonl", "S33.jsonl", "S34.jsonl", "S35.jsonl", "S36.jsonl", "S37.jsonl", "S38.jsonl", "S39.jsonl", "S40.jsonl", "S41.jsonl", "S42.jsonl", "S43.jsonl", "S44.jsonl", "S45.jsonl", "S46.jsonl", "S47.jsonl", "S48.jsonl", "S49.jsonl", "S50.jsonl", "tc_e1_L2_r1.safetensors", "tc_e1_L2_r2.safetensors", "tc_e1_L4_r1.safetensors", "tc_e1_L4_r2.safetensors", "tc_e5_s35_r1.safetensors", "tc_e5_s35_r3.safetensors", "tc_e1_L16_r1.safetensors", "tc_e1_L16_r2.safetensors", "tc_e1_L32_r1.safetensors", "tc_e1_L32_r2.safetensors", "tc_sft_B_r1.safetensors", "tc_e1_60m_L8_r1.safetensors", "S51.jsonl", "S52.jsonl", "S53.jsonl", "s34_a16.safetensors", "s30_a15.safetensors", "recipes.json", "arm_a_lr3e5.json"]
    for i, n in enumerate(names):
        data = (n * (300 if n.endswith(".safetensors") else 3)).encode()
        if n == "recipes.json" and recipes is not None:
            data = json.dumps(recipes).encode()
        (d / n).write_bytes(data)
        pins[n] = [str(d / n), hashlib.sha256(data).hexdigest()]
    if bad_pin:
        pins[bad_pin if isinstance(bad_pin, str) else "S2.jsonl"][1] = "0" * 64
    return pins


def seed_hub(tmp, pins, missing=()):
    """The HF repo stand-in: every data file under its VM name, base weights as the driver's parts."""
    hub = tmp / "hub"
    hub.mkdir()
    for n, (path, _) in pins.items():
        if n.endswith(".safetensors"):
            data = Path(path).read_bytes()
            stem = "base" if n == "sfull_sft1500.safetensors" else n[:-len(".safetensors")]
            for i in range(0, len(data), 2000):
                (hub / f"{stem}.part{i // 2000:02d}").write_bytes(data[i:i + 2000])
        else:
            (hub / n).write_bytes(Path(path).read_bytes())
    for n in missing:
        (hub / n).unlink()
    return hub


class DriveTests(unittest.TestCase):
    def run_drive(self, mode, run="t1", bad_pin=False, hf=False, hub_missing=(), minute="0.1", extra=None,
                  pre=None, seed=None, screens=None, recipes=None):
        use = screens or SCREENS
        gates = sorted(["provenance", "checkout", "inputs", "configure", "build", "specs"]
                       + [f"{g}_{s}" for s in use for g in ("train", "package")])
        tmp = Path(tempfile.mkdtemp())
        mock = tmp / "colab"
        mock.write_text(MOCK)
        mock.chmod(0o755)
        (tmp / "content").mkdir()
        pins = write_inputs(tmp, bad_pin, recipes)
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

    def test_the_safetensors_comes_home_verified(self):
        rc, state, log, tmp, _ = self.run_drive("normal")
        self.assertEqual(rc, 0, log)
        for s in SCREENS:
            st = tmp / "art" / "t1" / s / "model.safetensors"
            self.assertEqual(st.read_bytes(), (s * 900).encode(), s)
            self.assertFalse(list(st.parent.glob("model.safetensors.part*")), s)

    def test_with_little_time_left_the_safetensors_is_skipped_after_home(self):
        rc, state, log, tmp, _ = self.run_drive("normal", extra={"DRIVE_TEST_ST_MIN_LEFT": "1000"})
        self.assertEqual(rc, 0, log)
        self.assert_home(tmp, SCREENS)
        self.assertFalse(list((tmp / "art").glob("*/*/model.safetensors")))
        self.assertIn("the safetensors stays on the VM", log)

    def test_an_unverified_safetensors_leaves_the_gguf_home_without_it(self):
        rc, state, log, tmp, _ = self.run_drive("bad_st_always")
        self.assertEqual(rc, 0, log)
        self.assert_home(tmp, SCREENS)
        self.assertFalse(list((tmp / "art").glob("*/*/model.safetensors")))
        self.assertIn("the GGUF comes home without its safetensors", log)

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
        self.assertEqual(sorted(man["files"]), sorted(["sfull_sft1500.safetensors", "vocab.gguf", "arm_a_lr3e5.json", "recipes.json",
                                                       "S0.jsonl", "S2.jsonl", "S4.jsonl"]))

    def test_a_status_with_another_seed_is_not_validated(self):
        rc, state, log, tmp, _ = self.run_drive("normal", seed=2, screens=["S0", "S2", "S4"],
                                                extra={"MOCK_SEED": "3"})
        self.assertEqual(rc, 1, log)
        self.assertIn("seed or screens differ", log)

    def test_bad_seed_or_screens_are_refused(self):
        for extra in (["0", "S0"], ["2", "S0,S99"], ["2", "S0,S0"], ["x", "S0"]):
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
        self.assertEqual(sorted(man["files"]), sorted(["sfull_sft1500.safetensors", "vocab.gguf", "arm_a_lr3e5.json", "recipes.json",
                                                       "S5.jsonl"]))

    def test_a_capped_diversity_session(self):
        rc, state, log, tmp, _ = self.run_drive("normal", seed=2, screens=["S8", "S6", "S7"], hf=True,
                                                extra={"DRIVE_TEST_CAP": "55"}, minute="0.3")
        self.assertEqual(rc, 0, log)
        self.assert_home(tmp, ["S6", "S7", "S8"], universe=["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"])

    def test_the_cap_sets_the_deadlines_and_is_bounded(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("drive_mod4", HERE / "drive.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertEqual(mod.capped(60), (50, 52, 55, 57, 60))
        self.assertEqual(mod.capped(35), (25, 27, 30, 32, 35))
        self.assertEqual((mod.launch_need(1), mod.launch_need(3)), (12, 30))
        self.assertEqual(mod.launch_need(["S0", "S5"]), 21)  # no recipes: 256 steps each
        for bad in ("29", "121", "6x"):
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

    def test_a_capped_session_never_uploads_the_base_weights_directly(self):
        rc, state, log, tmp, _ = self.run_drive("hf_exec_fails", hf=True, extra={"DRIVE_TEST_CAP": "60"}, minute="0.3",
                                                seed=2, screens=["S5"])
        self.assertEqual(rc, 2, log)
        self.assertIn("base weights are not uploaded directly", log)
        self.assertEqual(state["counts"]["hf_fetch"], 2)
        self.assertFalse([k for k in state["counts"] if k.startswith("upload:base.part")])
        self.assertEqual(state["counts"].get("launch", 0), 0)
        self.assertFalse((tmp / "content" / "screens_in" / ".hf_token").exists())
        self.assertEqual(state["sessions"], {})

    def test_a_failed_hf_fetch_is_retried_once(self):
        rc, state, log, tmp, _ = self.run_drive("hf_fail_once", hf=True, extra={"DRIVE_TEST_CAP": "60"}, minute="0.3",
                                                seed=2, screens=["S5"])
        self.assertEqual(rc, 0, log)
        self.assertEqual(state["counts"]["hf_fetch"], 2)
        self.assertEqual(state["counts"].get("uploads_data", 0), 0)
        self.assertNotIn("hf_fixture_token", log)

    TWO_ARMS = {"S51": {"base": "s34_a16", "train.steps": 1024}, "S52": {"base": "s34_a16", "train.steps": 1024}}
    ONE_BASE = {"S51": {"base": "s34_a16"}, "S52": {"base": "s34_a16"}}
    TWO_BASES = {"S51": {"base": "s34_a16"}, "S53": {"base": "s30_a15"}}

    def members(self, tmp, run="t1"):
        import tarfile
        with tarfile.open(tmp / "art" / run / "logs.tgz") as t:
            return [m.name for m in t.getmembers()]

    def test_two_arms_from_one_named_base_under_a_95_minute_cap(self):
        rc, state, log, tmp, _ = self.run_drive("normal", seed=2, screens=["S51", "S52"], hf=True, recipes=self.TWO_ARMS,
                                                extra={"DRIVE_TEST_CAP": "95", "DRIVE_TEST_LAUNCH_NEED": "1"}, minute="0.3")
        self.assertEqual(rc, 0, log)
        self.assert_home(tmp, ["S51", "S52"], universe=["S51", "S52"])
        man = json.loads((tmp / "receipts" / "t1" / "input_manifest.json").read_text())
        self.assertEqual(sorted(man["bases"]), ["s34_a16"])
        self.assertEqual(man["screen_bases"], {"S51": "s34_a16", "S52": "s34_a16"})
        self.assertEqual(sorted(man["files"]), sorted(["s34_a16.safetensors", "vocab.gguf", "arm_a_lr3e5.json", "recipes.json", "S51.jsonl", "S52.jsonl"]))
        self.assertEqual(man["bases"]["s34_a16"]["sha256"], man["files"]["s34_a16.safetensors"])
        self.assertTrue(all(p["name"].startswith("s34_a16.part") for p in man["bases"]["s34_a16"]["parts"]))
        self.assertEqual(state["counts"].get("uploads_data", 0), 0)

    def test_two_bases_in_one_session_upload_under_distinct_names(self):
        rc, state, log, tmp, _ = self.run_drive("normal", seed=2, screens=["S51", "S53"], recipes=self.TWO_BASES)
        self.assertEqual(rc, 0, log)
        man = json.loads((tmp / "receipts" / "t1" / "input_manifest.json").read_text())
        self.assertEqual(sorted(man["bases"]), ["s30_a15", "s34_a16"])
        names = [p["name"] for v in man["bases"].values() for p in v["parts"]]
        self.assertEqual(len(names), len(set(names)))
        for n in names:
            self.assertEqual(state["counts"][f"upload:{n}"], 1)
        self.assertNotIn("upload:base.part00", state["counts"])

    def test_the_default_base_keeps_its_part_names_beside_a_named_base(self):
        rc, state, log, tmp, _ = self.run_drive("normal", seed=2, screens=["S0", "S51"], recipes={"S51": {"base": "s34_a16"}})
        self.assertEqual(rc, 0, log)
        man = json.loads((tmp / "receipts" / "t1" / "input_manifest.json").read_text())
        self.assertEqual(man["screen_bases"], {"S0": "sfull_sft1500", "S51": "s34_a16"})
        self.assertTrue(all(p["name"].startswith("base.part") for p in man["bases"]["sfull_sft1500"]["parts"]))

    def test_a_flipped_byte_in_a_named_base_stops_before_any_session(self):
        rc, state, log, tmp, _ = self.run_drive("normal", bad_pin="s34_a16.safetensors", seed=2, screens=["S51", "S52"], recipes=self.TWO_ARMS)
        self.assertEqual(rc, 2, log)
        self.assertIn("s34_a16.safetensors hashes", log)
        self.assertEqual(state["counts"].get("new", 0), 0)

    def test_a_recipe_naming_an_unpinned_base_stops_before_any_session(self):
        rc, state, log, tmp, _ = self.run_drive("normal", seed=2, screens=["S51"], recipes={"S51": {"base": "nosuch"}})
        self.assertEqual(rc, 2, log)
        self.assertIn("not pinned", log)
        self.assertEqual(state["counts"].get("new", 0), 0)

    def test_a_cap_refuses_a_direct_upload_of_a_named_base(self):
        """The refusal once tested a fixed prefix: renamed parts would have gone over the home link under a cap."""
        rc, state, log, tmp, _ = self.run_drive("hf_exec_fails", hf=True, extra={"DRIVE_TEST_CAP": "95", "DRIVE_TEST_LAUNCH_NEED": "1"}, minute="0.3",
                                                seed=2, screens=["S51", "S52"], recipes=self.TWO_ARMS)
        self.assertEqual(rc, 2, log)
        self.assertIn("base weights are not uploaded directly", log)
        self.assertFalse([k for k in state["counts"] if ".part" in k and k.startswith("upload:")])
        self.assertEqual(state["counts"].get("launch", 0), 0)
        self.assertEqual(state["sessions"], {})

    def test_a_cap_refuses_one_missing_part_of_a_named_base(self):
        rc, state, log, tmp, _ = self.run_drive("normal", hf=True, hub_missing=["s34_a16.part01"], extra={"DRIVE_TEST_CAP": "95", "DRIVE_TEST_LAUNCH_NEED": "1"}, minute="0.3",
                                                seed=2, screens=["S51", "S52"], recipes=self.TWO_ARMS)
        self.assertEqual(rc, 2, log)
        self.assertIn("(1 part(s) missing or wrong", log)
        self.assertEqual(state["counts"].get("launch", 0), 0)

    def test_retrieval_leaves_out_the_safetensors_of_arms_already_home(self):
        rc, state, log, tmp, _ = self.run_drive("normal", seed=2, screens=["S51", "S52"], recipes=self.ONE_BASE)
        self.assertEqual(rc, 0, log)
        names = self.members(tmp)
        self.assertFalse([n for n in names if "model.safetensors.part" in n], names)
        self.assertFalse([n for n in names if "model.gguf.part" in n])
        for s in ("S51", "S52"):
            self.assertTrue((tmp / "art" / "t1" / s / "model.safetensors").is_file())

    def test_retrieval_keeps_the_safetensors_of_an_arm_not_home(self):
        rc, state, log, tmp, _ = self.run_drive("bad_st_always", seed=2, screens=["S51", "S52"], recipes=self.ONE_BASE)
        names = self.members(tmp)
        for s in ("S51", "S52"):
            self.assertTrue([n for n in names if f"screens/{s}/model.safetensors.part" in n], log)

    def test_the_a19_recipes_and_budget(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("drive_mod", HERE / "drive.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        rec = json.loads((HERE / "recipes.json").read_text())
        self.assertEqual({k: v for k, v in rec["S51"].items()}, {"base": "s34_a16", "train.steps": 1024, "train.lr": 0.0001, "train.lr_schedule.warmup_steps": 64, "train.eval_examples": 64})
        self.assertEqual(rec["S52"], rec["S51"])
        self.assertEqual(rec["S53"], {**rec["S51"], "base": "s30_a15"})
        self.assertEqual(mod.launch_need(["S51", "S52"]), 75)
        self.assertEqual(mod.capped(95), (85, 87, 90, 92, 95))
        self.assertTrue({"S51", "S52", "S53"} <= set(mod.ALL_SCREENS))
        self.assertEqual(mod.bases_of(["S51", "S52"]), ("s34_a16",))
        self.assertEqual(mod.bases_of(["S51", "S53"]), ("s30_a15", "s34_a16"))
        self.assertEqual(mod.PINS["S52.jsonl"], mod.PINS["S34.jsonl"])

    DEPTH = {"S35": {"base": "tc_e5_s35_r1", "train.steps": 2048}, "S36": {"base": "tc_e1_L32_r1", "train.steps": 2048}, "S37": {"base": "tc_e1_L2_r1", "train.steps": 2048}}
    ARMS = ["S35", "S36", "S37"]

    def depth(self, log, mode="normal", extra=None):
        env = {"MOCK_TRAIN_LOG": log, "DRIVE_TEST_LAUNCH_NEED": "1", **(extra or {})}
        return self.run_drive(mode, seed=1, screens=self.ARMS, recipes=self.DEPTH, extra=env, minute="0.3")

    @staticmethod
    def evals(*rows):
        return "".join('{"eval_device":true,"eval_s":2.4,"event":"eval","step":%d,"val_loss":%s}\n' % r for r in rows)

    def test_a_session_below_the_bar_runs_to_the_end(self):
        rc, state, log, tmp, _ = self.depth(self.evals((256, "3.1"), (512, "2.5"), (768, "2.2"), (1024, "1.88"), (1280, "2.9")))
        self.assertEqual(rc, 0, log)
        self.assert_home(tmp, self.ARMS, universe=self.ARMS)
        self.assertIn("S35: R(1024) = 1.88, bar 1.88: continues", log)
        self.assertEqual(state["counts"]["early_read"], 1)
        self.assertEqual(state["counts"].get("stop_workload", 0), 0)

    def test_above_the_bar_stops_after_s35_comes_home_with_its_weights(self):
        rc, state, log, tmp, _ = self.depth(self.evals((512, "1.5"), (1024, "1.881")))
        self.assert_home(tmp, ["S35"], universe=self.ARMS)
        self.assertTrue((tmp / "art" / "t1" / "S35" / "model.safetensors").is_file())
        self.assertIn("stop once S35 is home", log)
        self.assertIn("early stop: S35 is home", log)
        self.assertEqual(state["counts"]["stop_workload"], 1)
        self.assertEqual(rc, 1, log)
        self.assertEqual(state["sessions"], {})
        self.assertLess(log.index("S35: HOME"), log.index("early stop: S35 is home"))

    def test_a_null_reading_is_a_stop(self):
        for token in ("null", "nan", "inf", "-nan"):
            rc, state, log, tmp, _ = self.depth(self.evals((1024, token)))
            self.assertIn("R(1024) = None", log, token)
            self.assertEqual(state["counts"]["stop_workload"], 1, token)
            self.assert_home(tmp, ["S35"], universe=self.ARMS)

    def test_the_first_eval_at_or_past_the_step_is_the_one_read(self):
        rc, state, log, tmp, _ = self.depth(self.evals((768, "9.9"), (1280, "1.2"), (1536, "9.9")))
        self.assertEqual(rc, 0, log)
        self.assertIn("S35: R(1280) = 1.2", log)

    def test_no_eval_yet_is_read_again_and_is_not_a_stop(self):
        rc, state, log, tmp, _ = self.depth(self.evals((512, "9.9")))
        self.assertEqual(rc, 0, log)
        self.assertGreater(state["counts"]["early_read"], 1)
        self.assertNotIn("R(", log)

    def test_a_failed_read_is_logged_is_no_stop_and_is_tried_again(self):
        rc, state, log, tmp, _ = self.depth(self.evals((1024, "2.5")), extra={"MOCK_EARLY_FAIL": "1"})
        self.assertIn("early read failed: not a stop", log)
        self.assertIn("R(1024) = 2.5", log)
        self.assertEqual(state["counts"]["early_read"], 2)

    def test_a_read_that_always_fails_lets_the_session_finish(self):
        rc, state, log, tmp, _ = self.depth(self.evals((1024, "2.5")), extra={"MOCK_EARLY_FAIL": "99"})
        self.assertEqual(rc, 0, log)
        self.assert_home(tmp, self.ARMS, universe=self.ARMS)
        self.assertEqual(state["counts"].get("stop_workload", 0), 0)

    def test_no_cancel_until_s35_is_home(self):
        rc, state, log, tmp, _ = self.depth(self.evals((1024, "2.5")), mode="bad_part_always")
        self.assertNotIn("early stop:", log)
        self.assertEqual(state["counts"].get("stop_workload", 0), 0)
        self.assert_home(tmp, [], universe=self.ARMS)

    def test_cancel_with_the_safetensors_not_home_keeps_it_in_the_archive(self):
        rc, state, log, tmp, _ = self.depth(self.evals((1024, "2.5")), mode="bad_st_always")
        self.assertIn("early stop: S35 is home", log)
        self.assertFalse((tmp / "art" / "t1" / "S35" / "model.safetensors").exists())
        self.assertTrue([n for n in self.members(tmp) if "screens/S35/model.safetensors.part" in n])

    def test_a_session_without_s35_never_reads(self):
        rc, state, log, tmp, _ = self.run_drive("normal", seed=1, screens=["S38", "S39"], recipes={"S38": {"base": "tc_e1_L16_r1"}, "S39": {"base": "tc_e1_L4_r1"}}, extra={"DRIVE_TEST_LAUNCH_NEED": "1"})
        self.assertEqual(rc, 0, log)
        self.assertEqual(state["counts"].get("early_read", 0), 0)

    def test_s35_alone_is_not_cancelled_after_it_is_home(self):
        rc, state, log, tmp, _ = self.run_drive("normal", seed=1, screens=["S35"], recipes={"S35": self.DEPTH["S35"]},
                                                extra={"MOCK_TRAIN_LOG": self.evals((1024, "2.5")), "DRIVE_TEST_LAUNCH_NEED": "1"})
        self.assertEqual(rc, 0, log)
        self.assertNotIn("early stop:", log)

    def test_two_named_bases_in_one_a18_session_and_a_recipe_naming_an_unpinned_base(self):
        rc, state, log, tmp, _ = self.run_drive("normal", seed=1, screens=["S35", "S36"], recipes=self.DEPTH, extra={"DRIVE_TEST_LAUNCH_NEED": "1"})
        self.assertEqual(rc, 0, log)
        man = json.loads((tmp / "receipts" / "t1" / "input_manifest.json").read_text())
        self.assertEqual(sorted(man["bases"]), ["tc_e1_L32_r1", "tc_e5_s35_r1"])
        rc, state, log, tmp, _ = self.run_drive("normal", seed=1, screens=["S35"], recipes={"S35": {"base": "tc_e1_L3_r1"}}, extra={"DRIVE_TEST_LAUNCH_NEED": "1"})
        self.assertEqual(rc, 2, log)
        self.assertIn("not pinned", log)
        self.assertEqual(state["counts"].get("new", 0), 0)

    def test_an_arm_home_keeps_its_spec_and_check_when_the_logs_are_lost(self):
        rc, state, log, tmp, _ = self.run_drive("logs_lost", seed=1, screens=["S35"], recipes={"S35": self.DEPTH["S35"]}, extra={"DRIVE_TEST_LAUNCH_NEED": "1"})
        self.assertIn("LOGS NOT RETRIEVED", log)
        d = tmp / "art" / "t1" / "S35"
        self.assertEqual((d / "spec_S35.json").read_text(), '{"name": "screen"}')
        self.assertEqual((d / "train_S35_check.json").read_text(), '{"ok": true}')
        self.assertTrue((d / "train.log").is_file())

    def test_an_earlier_screen_fetches_neither(self):
        rc, state, log, tmp, _ = self.run_drive("normal")
        self.assertEqual(rc, 0, log)
        self.assertFalse((tmp / "art" / "t1" / "S0" / "spec_S0.json").exists())

    def test_the_last_depth_arm_may_fetch_its_safetensors_up_to_the_retrieval_deadline(self):
        extra = {"DRIVE_TEST_LAUNCH_NEED": "1", "DRIVE_TEST_ST_MIN_LEFT": "78"}
        rc, state, log, tmp, _ = self.run_drive("normal", seed=1, screens=["S35"], recipes={"S35": self.DEPTH["S35"]}, extra=extra, minute="1")
        self.assertEqual(rc, 0, log)
        self.assertTrue((tmp / "art" / "t1" / "S35" / "model.safetensors").is_file(), log)

    def test_an_arm_before_the_last_keeps_the_polling_deadline(self):
        extra = {"DRIVE_TEST_LAUNCH_NEED": "1", "DRIVE_TEST_ST_MIN_LEFT": "78"}
        rc, state, log, tmp, _ = self.run_drive("normal", seed=1, screens=["S35", "S36"], recipes=self.DEPTH, extra=extra, minute="1")
        self.assertFalse((tmp / "art" / "t1" / "S35" / "model.safetensors").exists(), log)
        self.assertIn("S35: under 78.0 minutes left", log)

    def test_the_a18_recipes_pins_and_budget(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("drive_mod", HERE / "drive.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        rec = json.loads((HERE / "recipes.json").read_text())
        shapes = {"S35": (448, 8, 7, 1248), "S36": (192, 32, 3, 832), "S37": (896, 2, 14, 2528), "S38": (320, 16, 5, 864), "S39": (640, 4, 10, 1760), "S40": (896, 2, 14, 2528), "S41": (448, 8, 7, 1248),
                  "S42": (192, 32, 3, 832), "S43": (320, 16, 5, 864), "S44": (640, 4, 10, 1760), "S45": (768, 8, 12, 2240), "S46": (768, 8, 12, 2240), "S47": (896, 2, 14, 2528), "S48": (192, 32, 3, 832),
                  "S49": (192, 32, 3, 832), "S50": (192, 32, 3, 832)}
        for s, (d, layers, heads, ff) in shapes.items():
            r = rec[s]
            self.assertEqual((r["arch.custom.d"], r["arch.custom.layers"], r["arch.custom.heads"], r["arch.custom.d_ff"]), (d, layers, heads, ff), s)
            self.assertEqual(d, heads * 64, s)
            self.assertIsNone(r["arch.state_tokens"])
            self.assertEqual((r["train.steps"], r["train.batch"], r["train.accum"], r["train.checkpoint_every"], r["train.eval_examples"], r["train.eval_every"], r["train.lr_schedule.warmup_steps"]), (2048, 8, 2, 256, 680, 256, 64), s)
            self.assertEqual(r["train.lr"], 0.0002 if s in ("S49", "S50") else 0.0001, s)
            self.assertEqual(r["base"].startswith("tc_"), True, s)
            self.assertIn(s, mod.ALL_SCREENS)
            self.assertIn(s, mod.SCREEN_MINUTES)
        self.assertEqual(rec["S47"]["base"], rec["S37"]["base"])
        self.assertEqual(rec["S48"]["base"], rec["S36"]["base"])
        self.assertEqual((rec["S49"]["base"], rec["S50"]["base"]), (rec["S36"]["base"], rec["S42"]["base"]))
        for s in ("S35", "S36", "S37", "S38", "S39", "S40", "S41", "S42", "S43", "S44", "S49", "S50"):
            self.assertEqual(mod.PINS[f"{s}.jsonl"], mod.PINS["S34.jsonl"], s)
            self.assertIn(f"{rec[s]['base']}.safetensors", mod.PINS, s)
        for s in ("S45", "S46", "S47", "S48"):   # pinned only when their branch opens, by patch_a18_driver.py
            if f"{s}.jsonl" in mod.PINS:
                self.assertEqual(mod.PINS[f"{s}.jsonl"], mod.PINS["S14.jsonl" if s in ("S47", "S48") else "S34.jsonl"], s)
        self.assertEqual(mod.launch_need(["S35", "S36", "S37"]), 103)
        self.assertEqual(mod.launch_need(["S51", "S52"]), 75)
        self.assertEqual(mod.launch_need(["S34"]), 75)
        self.assertEqual(mod.part_stem("tc_e1_L2_r1"), "tc_e1_L2_r1")
        self.assertEqual(mod.part_stem("sfull_sft1500"), "base")
        self.assertEqual(mod.EARLY_STOP, {"S35": (1024, 1.88)})

    def test_every_a18_session_plan_fits_the_driver(self):
        """Each plan's need is under its polling end by at least the 4 minutes a fetch takes, and every cap is one main() accepts."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("drive_mod", HERE / "drive.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        plans = ((("S35", "S36", "S37"), 120), (("S36", "S37"), 85), (("S38", "S39", "S40"), 110), (("S41", "S42"), 100), (("S43", "S44"), 85), (("S45",), 85), (("S46",), 85), (("S47", "S48"), 85), (("S49", "S50"), 110))
        for screens, cap in plans:
            self.assertTrue(30 <= cap <= 120, screens)
            self.assertLessEqual(mod.launch_need(list(screens)) + 4, mod.capped(cap)[0], screens)

    def test_a20_read_is_logged_and_stops_nothing(self):
        env = {"MOCK_TRAIN_LOG": self.evals((512, "3.0"), (1024, "2.5")), "DRIVE_TEST_LAUNCH_NEED": "1"}
        rc, state, log, tmp, _ = self.run_drive("normal", seed=1, screens=["S54"], recipes={"S54": {"base": "pk_sft_v1", "train.steps": 2048}}, extra=env, minute="0.3")
        self.assertEqual(rc, 0, log)
        self.assert_home(tmp, ["S54"], universe=["S54"])
        self.assertIn("S54: R(1024) = 2.5, bar 2.09, read only: over the bar", log)
        self.assertNotIn("early stop", log)
        self.assertEqual(state["counts"].get("stop_workload", 0), 0)

    def test_a20_null_read_stops_nothing(self):
        env = {"MOCK_TRAIN_LOG": self.evals((512, "null")), "DRIVE_TEST_LAUNCH_NEED": "1"}
        rc, state, log, tmp, _ = self.run_drive("normal", seed=1, screens=["S55"], recipes={"S55": {"base": "pk_sft_v1", "train.steps": 1024}}, extra=env, minute="0.3")
        self.assertEqual(rc, 0, log)
        self.assertIn("S55: R(512) = None, bar 2.65, read only: over the bar", log)

    def test_the_a20_recipes_pins_and_budget(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("drive_mod", HERE / "drive.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        rec = json.loads((HERE / "recipes.json").read_text())
        for s, steps, cap, need in (("S54", 2048, 115, 89), ("S55", 1024, 75, 46)):
            r = rec[s]
            self.assertEqual((r["arch.custom.d"], r["arch.custom.layers"], r["arch.custom.heads"], r["arch.custom.d_ff"]), (768, 12, 12, 2048))
            self.assertIsNone(r["arch.state_tokens"])
            self.assertEqual((r["train.steps"], r["train.batch"], r["train.accum"], r["train.checkpoint_every"], r["train.eval_examples"], r["train.eval_every"], r["train.lr_schedule.warmup_steps"], r["train.lr"]), (steps, 8, 2, 256, 680, 256, 64, 0.0001))
            self.assertEqual(r["base"], "pk_sft_v1")
            self.assertIn(s, mod.ALL_SCREENS)
            self.assertEqual(mod.launch_need([s]), need)
            self.assertLessEqual(mod.launch_need([s]) + 4, mod.capped(cap)[0])
        self.assertEqual(mod.PINS["S54.jsonl"], mod.PINS["S34.jsonl"])
        self.assertEqual(mod.PINS["S55.jsonl"], mod.PINS["S30.jsonl"])
        self.assertEqual(mod.part_stem("pk_sft_v1"), "pk_sft_v1")
        self.assertEqual(mod.EARLY_STOP, {"S35": (1024, 1.88)})
        self.assertEqual(set(mod.EARLY_READ) & set(mod.EARLY_STOP), set())

    def test_a_late_session_is_not_launched(self):
        rc, state, log, tmp, _ = self.run_drive("normal", extra={"DRIVE_TEST_LAUNCH_NEED": "100000"})
        self.assertEqual(rc, 2, log)
        self.assertIn("too late to launch", log)
        self.assertEqual(state["counts"].get("launch", 0), 0)
        self.assertEqual(state["sessions"], {})
        self.assertIn("CLEANUP OK", log)

    def test_a_four_screen_a7_session(self):
        rc, state, log, tmp, _ = self.run_drive("normal", seed=1, screens=["S9", "S11", "S10", "S8"], hf=True,
                                                extra={"DRIVE_TEST_CAP": "70"}, minute="0.3")
        self.assertEqual(rc, 0, log)
        self.assert_home(tmp, ["S8", "S9", "S10", "S11"],
                         universe=["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11"])

    def test_test_hooks_are_refused_without_the_harness(self):
        env = dict(os.environ, DRIVE_TEST_COLAB="/bin/true")
        env.pop("DRIVE_TEST_HARNESS", None)
        r = subprocess.run([sys.executable, str(HERE / "drive.py"), COMMIT, "t1"], env=env, capture_output=True,
                           text=True)
        self.assertEqual(r.returncode, 2)
        self.assertIn("refusing a real run with test hooks", r.stderr)


if __name__ == "__main__":
    unittest.main()
