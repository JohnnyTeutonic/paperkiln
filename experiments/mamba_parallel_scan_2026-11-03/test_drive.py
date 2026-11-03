"""Tests of drive.py against a mock colab CLI (no network, no Colab). python3 -m unittest test_drive -v"""
import importlib.util
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
GATES = {
    "forward": ["build", "checkout", "configure", "host_test", "numpy", "replay_2", "replay_3", "replay_unset",
                "scan_bench", "strict_2", "strict_3", "strict_unset", "switch_refuses_malformed"],
    "backward": ["build", "build_base", "checkout", "configure", "default_identical", "host_test", "host_test_bwd",
                 "numpy", "provenance", "replay_2", "replay_3", "replay_unset", "scan_bwd_bench", "strict_2",
                 "strict_3", "strict_unset", "switch_refuses_malformed"],
    "prefill": ["build", "checkout", "configure", "engine_unchanged", "host_test", "prefill_gate",
                "prefill_gate_cublas", "prefill_profile", "prefill_profile_cublas", "prefill_time",
                "prefill_time_cublas", "provenance"],
}

# The mock keeps its state in MOCK_STATE: sessions, call counts. exec is dispatched on the label in the -f file name
# (drive.py writes /tmp/claude_scan_RUN_LABEL.py) and simulates the VM side.
MOCK = r'''#!/usr/bin/env python3
import json, os, re, sys, tarfile, time
from pathlib import Path
state_p = Path(os.environ["MOCK_STATE"]); content = Path(os.environ["MOCK_CONTENT"]); mode = os.environ["MOCK_MODE"]
st = json.loads(state_p.read_text()) if state_p.exists() else {"sessions": {}, "counts": {}}
save = lambda: state_p.write_text(json.dumps(st))
def count(k):
    st["counts"][k] = st["counts"].get(k, 0) + 1; save(); return st["counts"][k]
args = sys.argv[1:]
cmd = args[0]
if cmd == "sessions":
    if mode == "sessions_fail" and st.get("created"):
        print("error: auth", file=sys.stderr); sys.exit(1)
    if mode == "banner":
        print("[colab] Using credentials from the keyring"); print("[colab] 1 session")
    for n, ep in st["sessions"].items(): print(f"[{n}] {ep} L4")
    sys.exit(0)
name = args[args.index("-s") + 1]
if cmd == "new":
    if mode == "orphan_new":
        st["sessions"]["?"] = "ep-orphan"; st["created"] = True; save(); sys.exit(1)
    st["sessions"][name] = "ep-" + name; st["created"] = True; save()
    if mode == "new_timeout": time.sleep(60)
    sys.exit(0)
if cmd == "stop":
    if mode != "stop_ignored": st["sessions"].pop(name, None); save()
    sys.exit(0)
def start_script(run, protocol):
    out = content / f"run_{run}" / "out"
    out.mkdir(parents=True, exist_ok=True)
    (content / f"vm_{run}.pid").write_text("4242")
    gates = {g: 0 for g in os.environ["MOCK_GATES"].split()}
    if mode == "gate_fail": gates[sorted(gates)[-1]] = 1
    commit = os.environ["MOCK_COMMIT"] if mode != "wrong_commit" else "f" * 40
    proto = protocol if mode != "wrong_protocol" else "sideways"
    s = {"run_id": run, "protocol": proto, "commit": commit, "head": commit, "exit_status": 0,
         "pass": mode != "gate_fail", "gates": gates}
    if mode not in ("never_done", "stop_unconfirmed"):
        (out / "status.json").write_text(json.dumps(s)); (out / "DONE").write_text(run)
    st["status"] = s; save()
if cmd == "exec":
    f = args[args.index("-f") + 1]
    code = Path(f).read_text()
    m = re.match(r".*claude_scan_(.+?)_(launch_retry|launch_probe\d|launch|poll|pack\d|stop_workload\d)\.py$", f)
    run, label = m.group(1), m.group(2)
    out = content / f"run_{run}" / "out"
    if label in ("launch", "launch_retry"):
        n = count("launch")
        protocol = re.search(r"vm_run_\S+\.sh \S+ \S+ (\w+) ", code).group(1)
        if mode == "launch_lost_started" and n == 1:
            start_script(run, protocol); sys.exit(1)        # the script started; the acknowledgement was lost
        if mode == "launch_lost_absent" and n == 1:
            sys.exit(1)                                     # nothing started
        start_script(run, protocol)
        print("LAUNCHED")
    elif label.startswith("launch_probe"):
        print("STARTED" if (content / f"vm_{run}.pid").exists() or (content / f"run_{run}").is_dir() else "ABSENT")
    elif label == "poll":
        print("DONE" if (out / "DONE").exists() else "RUNNING")
    elif label.startswith("stop_workload"):
        count("stop_workload")
        if mode == "stop_unconfirmed":
            print('STOP_REPLY {"result": "unconfirmed"}'); sys.exit(0)
        s = st["status"]; s["gates"][sorted(s["gates"])[-1]] = 143; s["pass"] = False; s["exit_status"] = 1
        (out / "status.json").write_text(json.dumps(s))
        (out / "stop.json").write_text(json.dumps({"step": "x", "step_started": True, "step_group_gone": True}))
        (out / "DONE").write_text(run)
        print('STOP_REPLY {"result": "interrupted"}')
    else:
        if mode == "pack_flaky" and count("pack") == 1:
            sys.exit(1)
        with tarfile.open(content / f"scan_{run}.tgz", "w:gz") as t: t.add(out, arcname=f"run_{run}/out")
        print("PACKED")
    sys.exit(0)
if cmd == "download":
    src, dst = args[-2], args[-1]
    Path(dst).write_bytes((content / Path(src).name).read_bytes()); sys.exit(0)
sys.exit(2)
'''


class DriveTests(unittest.TestCase):
    def run_drive(self, mode, run="t1", extra=None, protocol=None):
        tmp = Path(tempfile.mkdtemp())
        mock = tmp / "colab"
        mock.write_text(MOCK)
        mock.chmod(0o755)
        (tmp / "content").mkdir()
        env = dict(os.environ, DRIVE_TEST_HARNESS="1", DRIVE_TEST_COLAB=str(mock), DRIVE_TEST_CLI_PY=sys.executable,
                   DRIVE_TEST_REAP="true", DRIVE_TEST_ADOPT="true", DRIVE_TEST_ARTIFACTS=str(tmp / "art"),
                   DRIVE_TEST_RECEIPTS=str(tmp / "receipts"), DRIVE_TEST_MINUTE_SECONDS="0.3",
                   MOCK_STATE=str(tmp / "state.json"), MOCK_CONTENT=str(tmp / "content"), MOCK_MODE=mode,
                   MOCK_GATES=" ".join(GATES[protocol or "forward"]), MOCK_COMMIT=COMMIT)
        env.update(extra or {})
        t0 = time.monotonic()
        argv = [sys.executable, str(HERE / "drive.py"), COMMIT, run] + ([protocol] if protocol else [])
        r = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=120)
        state = json.loads((tmp / "state.json").read_text()) if (tmp / "state.json").exists() else {"sessions": {}}
        log = (tmp / "art" / run / "driver.log").read_text() if (tmp / "art" / run / "driver.log").exists() else ""
        return r.returncode, state, log, tmp, time.monotonic() - t0

    def test_normal_run_validates_and_stops(self):
        for protocol in (None, "backward", "prefill"):
            rc, state, log, tmp, _ = self.run_drive("normal", protocol=protocol)
            self.assertEqual(rc, 0, log)
            self.assertEqual(state["sessions"], {})
            self.assertIn("CLEANUP OK", log)
            self.assertIn("validation: PASS", log)
            self.assertIn(f"protocol {protocol or 'forward'}", log)
            for f in ("status.json", "driver.log", "archive.sha256"):
                self.assertTrue((tmp / "receipts" / "t1" / f).is_file(), f)
            self.assertIn("status.json", (tmp / "receipts" / "t1" / "archive.sha256").read_text())

    def test_failed_gate_is_exit_1_with_receipts(self):
        rc, state, log, tmp, _ = self.run_drive("gate_fail", protocol="backward")
        self.assertEqual(rc, 1, log)
        self.assertEqual(state["sessions"], {})
        self.assertTrue((tmp / "receipts" / "t1" / "status.json").is_file())

    def test_wrong_commit_or_protocol_fails_validation(self):
        for mode in ("wrong_commit", "wrong_protocol"):
            rc, _, log, _, _ = self.run_drive(mode, protocol="backward")
            self.assertEqual(rc, 2, log)
            self.assertIn("identity mismatch", log)

    def test_prefill_paths_reuse_the_hardening(self):
        rc, state, log, _, _ = self.run_drive("launch_lost_started", protocol="prefill")
        self.assertEqual((rc, state["counts"]["launch"]), (0, 1), log)
        rc, state, log, tmp, _ = self.run_drive("never_done", protocol="prefill")
        self.assertEqual(rc, 1, log)
        self.assertIn("workload stop: confirmed (interrupted)", log)
        self.assertEqual(state["sessions"], {})
        rc, _, log, _, _ = self.run_drive("gate_fail", protocol="prefill")
        self.assertEqual(rc, 1, log)
        rc, _, log, _, _ = self.run_drive("wrong_protocol", protocol="prefill")
        self.assertEqual(rc, 2, log)

    def test_gate_sets_do_not_validate_each_other(self):
        for protocol, other in (("prefill", "backward"), ("backward", "prefill"), (None, "prefill")):
            rc, _, log, _, _ = self.run_drive("normal", protocol=protocol, extra={"MOCK_GATES": " ".join(GATES[other])})
            self.assertEqual(rc, 1, log)
            self.assertIn("gate set differs", log)

    def test_forward_gate_set_does_not_validate_a_backward_run(self):
        rc, _, log, _, _ = self.run_drive("normal", protocol="backward", extra={})
        self.assertEqual(rc, 0, log)
        rc, _, log, _, _ = self.run_drive("normal", protocol=None,
                                          extra={"MOCK_GATES": " ".join(GATES["backward"])})
        self.assertEqual(rc, 1, log)
        self.assertIn("gate set differs", log)

    def test_banner_rows_are_not_sessions(self):
        rc, state, log, _, _ = self.run_drive("banner")
        self.assertEqual(rc, 0, log)
        self.assertIn('preflight sessions: []', log)

    def test_lost_launch_acknowledgement_is_reconciled_not_repeated(self):
        rc, state, log, _, _ = self.run_drive("launch_lost_started", protocol="backward")
        self.assertEqual(rc, 0, log)
        self.assertEqual(state["counts"]["launch"], 1)
        self.assertIn("the VM script is started", log)

    def test_absent_launch_is_retried_once(self):
        rc, state, log, _, _ = self.run_drive("launch_lost_absent", protocol="backward")
        self.assertEqual(rc, 0, log)
        self.assertEqual(state["counts"]["launch"], 2)
        self.assertIn("the VM script is absent", log)

    def test_unfinished_run_is_stopped_before_retrieval(self):
        rc, state, log, tmp, _ = self.run_drive("never_done", protocol="backward")
        self.assertEqual(rc, 1, log)  # interrupted: a failed experiment with its receipts
        self.assertEqual(state["counts"]["stop_workload"], 1)
        self.assertIn("workload stop: confirmed (interrupted)", log)
        self.assertTrue((tmp / "receipts" / "t1" / "stop.json").is_file())
        self.assertEqual(state["sessions"], {})

    def test_unconfirmed_stop_is_at_least_exit_2(self):
        rc, state, log, _, _ = self.run_drive("stop_unconfirmed", protocol="backward")
        self.assertGreaterEqual(rc, 2, log)
        self.assertIn("NOT CONFIRMED", log)
        self.assertEqual(state["sessions"], {})

    def test_flaky_pack_is_retried(self):
        rc, state, log, _, _ = self.run_drive("pack_flaky", protocol="backward")
        self.assertEqual(rc, 0, log)
        self.assertEqual(state["counts"]["pack"], 2)

    def test_unnamed_allocation_is_never_certified(self):
        rc, state, log, _, _ = self.run_drive("orphan_new")
        self.assertEqual(rc, 3, log)
        self.assertIn("CLEANUP UNRESOLVED", log)
        self.assertNotIn("CLEANUP OK", log)
        self.assertEqual(state["sessions"], {"?": "ep-orphan"})  # not ours by name or endpoint: not stopped

    def test_log_failure_does_not_stop_the_shutdown(self):
        rc, state, log, _, _ = self.run_drive("normal", extra={"DRIVE_TEST_LOG_FAIL_AFTER": "4"})
        self.assertEqual(state["sessions"], {})
        self.assertNotEqual(rc, 3)

    def test_cleanup_command_start_failure_is_retried(self):
        rc, state, log, _, _ = self.run_drive("normal", extra={"DRIVE_TEST_RAISE_ON": "stop"})
        self.assertEqual(state["sessions"], {}, log)
        self.assertIn("could not run", log)
        self.assertIn("CLEANUP OK", log)
        self.assertEqual(rc, 0, log)

    def test_test_hooks_refused_without_harness(self):
        env = dict(os.environ, DRIVE_TEST_COLAB="/bin/true")
        env.pop("DRIVE_TEST_HARNESS", None)
        r = subprocess.run([sys.executable, str(HERE / "drive.py"), COMMIT, "t9"], env=env, capture_output=True,
                           text=True)
        self.assertEqual(r.returncode, 2)
        self.assertIn("refusing", r.stderr)

    def test_listing_failure_is_never_reported_as_clean(self):
        rc, state, log, _, _ = self.run_drive("sessions_fail")
        self.assertEqual(rc, 3, log)
        self.assertIn("CLEANUP UNRESOLVED", log)
        self.assertNotIn("CLEANUP OK", log)

    def test_ignored_stop_is_unresolved(self):
        rc, state, log, _, _ = self.run_drive("stop_ignored")
        self.assertEqual(rc, 3, log)
        self.assertIn("t1", json.dumps(state["sessions"]))
        self.assertIn("CLEANUP UNRESOLVED", log)

    def test_creation_timeout_still_stops_the_allocated_session(self):
        rc, state, log, _, elapsed = self.run_drive("new_timeout")
        self.assertEqual(state["sessions"], {}, log)  # the VM the timed-out call allocated was found and stopped
        self.assertIn("CLEANUP OK", log)
        self.assertIn(rc, (0, 2), log)
        self.assertLess(elapsed, 0.3 * 90 + 30)  # bounded by the shutdown deadline (plus inventory slack)

    def test_bad_arguments_are_refused(self):
        for argv in (["main", "x"], [COMMIT, "t1", "sideways"]):
            r = subprocess.run([sys.executable, str(HERE / "drive.py")] + argv, capture_output=True, text=True)
            self.assertEqual(r.returncode, 2)


class LaunchCodeTests(unittest.TestCase):
    def test_launch_code_publishes_the_script_atomically_and_starts_it(self):
        spec = importlib.util.spec_from_file_location("drive", HERE / "drive.py")
        drive = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(drive)
        tmp = Path(tempfile.mkdtemp())
        src = tmp / "vm_run.sh"
        src.write_bytes(b'echo "$1 $2 $3" > "$(dirname "$0")/started"\r\n')
        old = tmp / "vm_run_r9.sh"
        old.write_text("an earlier launch's copy\n")
        code = drive.launch_code(src.as_uri(), COMMIT, "r9", "backward", root=str(tmp))
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
        self.assertIn("LAUNCHED", r.stdout, r.stderr)
        self.assertEqual(old.read_bytes(), b'echo "$1 $2 $3" > "$(dirname "$0")/started"\n')  # CRLF normalised
        self.assertEqual([p.name for p in tmp.iterdir() if p.name.endswith(".tmp")], [])
        deadline = time.monotonic() + 10
        while not (tmp / "started").exists() and time.monotonic() < deadline:
            time.sleep(0.1)
        self.assertEqual((tmp / "started").read_text().strip(), f"{COMMIT} r9 backward")

    def test_prefill_deadlines_fit_its_60_minute_cap(self):
        spec = importlib.util.spec_from_file_location("drive", HERE / "drive.py")
        drive = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(drive)
        self.assertEqual(set(drive.DEADLINES), set(drive.REQUIRED_GATES))
        for protocol, cap in (("forward", 90), ("backward", 90), ("prefill", 60)):
            t0 = time.monotonic()
            d = drive.Driver(COMMIT, "r9", protocol)
            ends = [d.work_end, d.retrieve_end, d.shutdown_end, d.inventory_end]
            self.assertEqual(ends, sorted(ends), protocol)  # each phase ends after the one before
            self.assertLessEqual(d.inventory_end - t0, cap * drive.MINUTE + 1, protocol)
            self.assertGreater(d.inventory_end - t0, (cap - 1) * drive.MINUTE, protocol)
        self.assertEqual(drive.DEADLINES["prefill"], (42, 50, 56, 60))


if __name__ == "__main__":
    unittest.main()
