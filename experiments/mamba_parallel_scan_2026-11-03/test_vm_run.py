"""Tests of vm_run.sh's gate bookkeeping, run-directory claim and stop handling with stub tools on PATH (no GPU, no
network). python3 -m unittest test_vm_run -v"""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest

HERE = Path(__file__).resolve().parent
COMMIT = "0123456789abcdef0123456789abcdef01234567"
BASE = "d0390e392006a383f8be4565b1b67fcfef513ff6"   # vm_run.sh's BASE
FORWARD = {"build", "checkout", "configure", "host_test", "numpy", "replay_2", "replay_3", "replay_unset",
           "scan_bench", "strict_2", "strict_3", "strict_unset", "switch_refuses_malformed"}
BACKWARD = {"build", "build_base", "checkout", "configure", "default_identical", "host_test", "host_test_bwd",
            "numpy", "provenance", "replay_2", "replay_3", "replay_unset", "scan_bwd_bench", "strict_2", "strict_3",
            "strict_unset", "switch_refuses_malformed"}

# Stubs: git clones by creating the tree and adds the base worktree; cmake makes the test binaries in the build
# directory it is given; nvcc makes the bench named by -o; python3 replay and pip succeed. FAIL_AT names one step to
# fail; the test binary refuses a malformed switch like the real one and prints 220 case lines.
STUBS = {
    "git": r'''#!/usr/bin/env bash
if [ "$1" = clone ]; then mkdir -p "${@: -1}"; [ "$FAIL_AT" = clone ] && exit 7; exit 0; fi
dir=.
if [ "$1" = -C ]; then dir=$2; shift 2; fi
case "$1" in
  rev-parse) case "$dir" in *base) echo "${STUB_BASE_HEAD:-d0390e392006a383f8be4565b1b67fcfef513ff6}";;
             *) echo "${STUB_HEAD:-0123456789abcdef0123456789abcdef01234567}";; esac;;
  status) [ "$FAIL_AT" = dirty ] && echo " M x"; true;;
  worktree) [ "$FAIL_AT" = worktree ] && exit 8; mkdir -p "${@: -2:1}";;
  *) [ "$FAIL_AT" = checkout ] && exit 5; true;;
esac''',
    "cmake": r'''#!/usr/bin/env bash
if [ "$1" = --build ]; then
  dir=$2
  [ "$FAIL_AT" = build ] && [ "$dir" = build ] && exit 4
  [ "$FAIL_AT" = build_base ] && [ "$dir" = base_build ] && exit 4
  mkdir -p "$dir"
  for t in test_selective_scan_chunked test_selective_scan_chunked_backward; do
    cat > "$dir/$t" <<'X'
#!/usr/bin/env bash
case "$0" in *backward) [ "$FAIL_AT" = host_test_bwd ] && exit 3;; *) [ "$FAIL_AT" = host_test ] && exit 3;; esac
echo '{"pass": true}'
X
  done
  cat > "$dir/test_selective_scan_cuda" <<'X'
#!/usr/bin/env bash
case "$MICROTORCH_SCAN_CHUNK" in
  " 2") echo "what(): MICROTORCH_SCAN_CHUNK must be a decimal count up to 2^31:  2" >&2; exit 134;;
esac
[ "$FAIL_AT" = "strict_${MICROTORCH_SCAN_CHUNK:-unset}" ] && exit 1
v=1; case "$0" in *base_build*) [ "$FAIL_AT" = base_differs ] && v=2;; esac
for i in $(seq 220); do echo "{\"raw_case\":true,\"case\":$i,\"dx\":[$v]}"; done
echo '{"pass":true}'
X
  chmod +x "$dir"/*; exit 0
fi
case "$*" in *base_build*) [ "$FAIL_AT" = configure_base ] && exit 6;; *) [ "$FAIL_AT" = configure ] && exit 6;; esac
exit 0''',
    "nvcc": r'''#!/usr/bin/env bash
if [ "$1" = --version ]; then echo "stub nvcc release 13.0"; exit 0; fi
out=${@: -1}
cat > "$out" <<'X'
#!/usr/bin/env bash
[ "$FAIL_AT" = hang ] && exec sleep 120
case "$0" in *scan_bwd_bench) [ "$FAIL_AT" = scan_bwd_bench ] && exit 1;; *) [ "$FAIL_AT" = scan_bench ] && exit 1;; esac
echo '{"summary": true, "pass": true}'
X
chmod +x "$out"''',
    "nvidia-smi": r'''#!/usr/bin/env bash
echo "NVIDIA L4, 580.82.07"
[ "$FAIL_AT" = two_gpus ] && echo "NVIDIA L4, 580.82.07"
true''',
    "python3": r'''#!/usr/bin/env bash
case "$*" in *replay_selective_scan.py*) [ "$FAIL_AT" = replay ] && exit 2; exit 0;; esac
case "$*" in *pip*) [ "$FAIL_AT" = numpy ] && exit 1; exit 0;; esac
[ "$1" = --version ] && echo "Python 3.13.0"
exit 0''',
    "nproc": "#!/usr/bin/env bash\necho 2",
}


class VmRunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        bin_ = self.tmp / "bin"
        bin_.mkdir()
        for name, body in STUBS.items():
            (bin_ / name).write_text(body + "\n")
            (bin_ / name).chmod(0o755)
        self.env = dict(os.environ, PATH=f"{bin_}:/usr/bin:/bin", VM_TEST_HARNESS="1", VM_ROOT=str(self.tmp))

    def run_vm(self, fail_at="", run="r1", head=None, protocol=None, extra=None):
        env = dict(self.env, FAIL_AT=fail_at, **(extra or {}))
        if head:
            env["STUB_HEAD"] = head
        argv = ["bash", str(HERE / "vm_run.sh"), COMMIT, run] + ([protocol] if protocol else [])
        r = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=90)
        out = self.tmp / f"run_{run}" / "out"
        status = json.loads((out / "status.json").read_text()) if (out / "status.json").exists() else None
        done = (out / "DONE").read_text().strip() if (out / "DONE").exists() else None
        return r.returncode, status, done

    # ---- forward protocol (the default): the gate set of r1-20261103
    def test_success_records_exactly_the_required_gates(self):
        rc, s, done = self.run_vm()
        self.assertEqual(rc, 0)
        self.assertEqual(done, "r1")
        self.assertTrue(s["pass"])
        self.assertEqual(set(s["gates"]), FORWARD)
        self.assertTrue(all(v == 0 for v in s["gates"].values()))
        self.assertEqual((s["exit_status"], s["commit"], s["head"], s["protocol"]), (0, COMMIT, COMMIT, "forward"))

    def test_prerequisite_failures_stop_and_record_the_real_status(self):
        for fail_at, gate, code in (("checkout", "checkout", 5), ("configure", "configure", 6), ("build", "build", 4),
                                    ("host_test", "host_test", 3)):
            rc, s, done = self.run_vm(fail_at, run=f"r_{fail_at}")
            self.assertEqual(rc, 1, fail_at)
            self.assertEqual(done, f"r_{fail_at}")
            self.assertFalse(s["pass"])
            self.assertEqual(s["gates"][gate], code, fail_at)
            self.assertNotIn("scan_bench", s["gates"], fail_at)  # no further paid work

    def test_failed_ordered_control_stops(self):
        rc, s, _ = self.run_vm("strict_unset")
        self.assertEqual(rc, 1)
        self.assertNotIn("strict_2", s["gates"])
        self.assertFalse(s["pass"])

    def test_failed_chunked_gate_continues_but_fails(self):
        rc, s, _ = self.run_vm("strict_2")
        self.assertEqual(rc, 1)  # the script exit follows pass
        self.assertNotEqual(s["gates"]["strict_2"], 0)
        self.assertEqual(s["gates"]["scan_bench"], 0)  # diagnostics still collected
        self.assertFalse(s["pass"])

    def test_failed_bench_fails(self):
        rc, s, _ = self.run_vm("scan_bench")
        self.assertEqual(rc, 1)
        self.assertFalse(s["pass"])

    def test_dirty_tree_or_wrong_head_fails_checkout(self):
        for i, kw in enumerate(({"fail_at": "dirty"}, {"head": "f" * 40})):
            rc, s, _ = self.run_vm(run=f"r{i}", **kw)
            self.assertEqual(rc, 1)
            self.assertEqual(s["gates"]["checkout"], 99)

    def test_bad_arguments_refused_without_a_run_directory(self):
        for argv in (["main", "r1"], [COMMIT, "r1", "sideways"]):
            r = subprocess.run(["bash", str(HERE / "vm_run.sh")] + argv, env=self.env, capture_output=True,
                               text=True)
            self.assertEqual(r.returncode, 2)
            self.assertFalse((self.tmp / "run_r1").exists())

    # ---- backward protocol
    def test_backward_success_records_exactly_its_gates(self):
        rc, s, done = self.run_vm(protocol="backward")
        self.assertEqual(rc, 0, s)
        self.assertEqual(done, "r1")
        self.assertTrue(s["pass"])
        self.assertEqual(set(s["gates"]), BACKWARD)
        self.assertTrue(all(v == 0 for v in s["gates"].values()))
        self.assertEqual((s["protocol"], s["head"]), ("backward", COMMIT))
        out = self.tmp / "run_r1" / "out"
        self.assertTrue((out / "scan_bwd_bench.jsonl").is_file())
        self.assertEqual(len((out / "cases_base.txt").read_text().splitlines()), 220)

    def test_backward_prerequisites_stop(self):
        for fail_at, gate in (("two_gpus", "provenance"), ("worktree", "checkout"), ("host_test_bwd", "host_test_bwd"),
                              ("numpy", "numpy"), ("strict_unset", "strict_unset")):
            rc, s, _ = self.run_vm(fail_at, run=f"b_{fail_at}", protocol="backward")
            self.assertEqual(rc, 1, fail_at)
            self.assertFalse(s["pass"], fail_at)
            self.assertNotEqual(s["gates"][gate], 0, fail_at)
            self.assertNotIn("build_base", s["gates"], fail_at)
            self.assertNotIn("scan_bwd_bench", s["gates"], fail_at)

    def test_wrong_base_head_fails_checkout(self):
        rc, s, _ = self.run_vm(protocol="backward", extra={"STUB_BASE_HEAD": "e" * 40})
        self.assertEqual(rc, 1)
        self.assertEqual(s["gates"]["checkout"], 99)

    def test_default_path_differing_from_master_fails(self):
        rc, s, _ = self.run_vm("base_differs", protocol="backward")
        self.assertEqual(rc, 1)
        self.assertEqual(s["gates"]["default_identical"], 1)
        self.assertEqual(s["gates"]["scan_bwd_bench"], 0)  # diagnostics still collected
        self.assertFalse(s["pass"])

    def test_failed_base_build_or_bench_fails(self):
        for fail_at, gate in (("build_base", "build_base"), ("scan_bwd_bench", "scan_bwd_bench"),
                              ("strict_3", "strict_3")):
            rc, s, _ = self.run_vm(fail_at, run=f"f_{fail_at}", protocol="backward")
            self.assertEqual(rc, 1, fail_at)
            self.assertNotEqual(s["gates"][gate], 0, fail_at)
            self.assertFalse(s["pass"], fail_at)

    # ---- hardening: run-directory claim, PID, stop
    def test_second_launch_of_a_run_is_refused_and_changes_nothing(self):
        rc, s, _ = self.run_vm(protocol="backward")
        self.assertEqual(rc, 0)
        before = (self.tmp / "run_r1" / "out" / "status.json").read_bytes()
        rc2, s2, _ = self.run_vm(protocol="backward")
        self.assertEqual(rc2, 2)
        self.assertEqual((self.tmp / "run_r1" / "out" / "status.json").read_bytes(), before)

    def test_pid_is_published(self):
        self.run_vm(protocol="backward")
        self.assertTrue((self.tmp / "vm_r1.pid").read_text().strip().isdigit())

    def test_stop_during_a_step_stops_its_group_and_publishes(self):
        t0 = time.monotonic()
        rc, s, done = self.run_vm("hang", protocol="backward", extra={"VM_TEST_SIGNAL_AT": "run_scan_bwd_bench"})
        self.assertLess(time.monotonic() - t0, 60)  # the hanging bench did not run out its 120 s
        self.assertEqual(rc, 1)
        self.assertEqual(done, "r1")
        self.assertEqual(s["gates"]["scan_bwd_bench"], 143)
        self.assertFalse(s["pass"])
        stop = json.loads((self.tmp / "run_r1" / "out" / "stop.json").read_text())
        self.assertEqual((stop["step"], stop["step_started"], stop["step_group_gone"]),
                         ("scan_bwd_bench", True, True))
        self.assertFalse((self.tmp / "run_r1" / "out" / "QUIESCENCE_UNCONFIRMED").exists())
        ps = subprocess.run(["ps", "-eo", "args="], capture_output=True, text=True).stdout
        self.assertNotIn("sleep 120", ps)

    def test_external_sigterm_is_handled_promptly(self):
        env = dict(self.env, FAIL_AT="hang")
        p = subprocess.Popen(["bash", str(HERE / "vm_run.sh"), COMMIT, "r1", "backward"], env=env,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        bench = self.tmp / "run_r1" / "scan_bwd_bench"
        deadline = time.monotonic() + 30
        while not bench.exists() and time.monotonic() < deadline:
            time.sleep(0.2)
        time.sleep(1.0)  # the hanging bench is running
        pid = int((self.tmp / "vm_r1.pid").read_text())
        os.kill(pid, 15)
        rc = p.wait(timeout=60)
        out = self.tmp / "run_r1" / "out"
        self.assertEqual(rc, 1)
        self.assertEqual((out / "DONE").read_text().strip(), "r1")
        self.assertEqual(json.loads((out / "stop.json").read_text())["step_group_gone"], True)
        self.assertEqual(json.loads((out / "status.json").read_text())["gates"]["scan_bwd_bench"], 143)


if __name__ == "__main__":
    unittest.main()
