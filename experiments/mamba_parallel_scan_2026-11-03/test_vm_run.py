"""Tests of vm_run.sh's gate bookkeeping with stub tools on PATH (no GPU, no network).
python3 -m unittest test_vm_run -v"""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
COMMIT = "0123456789abcdef0123456789abcdef01234567"
REQUIRED = {"build", "checkout", "configure", "host_test", "numpy", "replay_2", "replay_3", "replay_unset",
            "scan_bench", "strict_2", "strict_3", "strict_unset", "switch_refuses_malformed"}

# Stubs: git clones by creating the tree; cmake makes the two test binaries; nvcc makes scan_bench; python3 replay
# and pip succeed. FAIL_AT names one step to fail; the test binary refuses a malformed switch like the real one.
STUBS = {
    "git": r'''#!/usr/bin/env bash
if [ "$1" = clone ]; then mkdir -p "${@: -1}"; [ "$FAIL_AT" = clone ] && exit 7; exit 0; fi
if [ "$1" = -C ]; then shift 2; fi
case "$1" in
  rev-parse) echo "${STUB_HEAD:-0123456789abcdef0123456789abcdef01234567}";;
  status) [ "$FAIL_AT" = dirty ] && echo " M x"; true;;
  *) [ "$FAIL_AT" = checkout ] && exit 5; true;;
esac''',
    "cmake": r'''#!/usr/bin/env bash
if [ "$1" = --build ]; then
  [ "$FAIL_AT" = build ] && exit 4
  mkdir -p build
  cat > build/test_selective_scan_chunked <<'X'
#!/usr/bin/env bash
[ "$FAIL_AT" = host_test ] && exit 3; echo '{"pass": true}'
X
  cat > build/test_selective_scan_cuda <<'X'
#!/usr/bin/env bash
case "$MICROTORCH_SCAN_CHUNK" in
  " 2") echo "what(): MICROTORCH_SCAN_CHUNK must be a decimal count up to 2^31:  2" >&2; exit 134;;
esac
[ "$FAIL_AT" = "strict_${MICROTORCH_SCAN_CHUNK:-unset}" ] && exit 1; echo '{"pass":true}'
X
  chmod +x build/*; exit 0
fi
[ "$FAIL_AT" = configure ] && exit 6; exit 0''',
    "nvcc": r'''#!/usr/bin/env bash
if [ "$1" = --version ]; then echo "stub nvcc"; exit 0; fi
cat > scan_bench <<'X'
#!/usr/bin/env bash
[ "$FAIL_AT" = scan_bench ] && exit 1; echo '{"summary": true, "pass": true}'
X
chmod +x scan_bench''',
    "nvidia-smi": "#!/usr/bin/env bash\necho stub-L4",
    "python3": r'''#!/usr/bin/env bash
case "$*" in *replay_selective_scan.py*) [ "$FAIL_AT" = replay ] && exit 2; exit 0;; esac
exit 0''',
    "nproc": "#!/usr/bin/env bash\necho 2",
}


class VmRunTests(unittest.TestCase):
    def run_vm(self, fail_at="", run="r1", head=None):
        tmp = Path(tempfile.mkdtemp())
        bin_ = tmp / "bin"
        bin_.mkdir()
        for name, body in STUBS.items():
            (bin_ / name).write_text(body + "\n")
            (bin_ / name).chmod(0o755)
        env = dict(os.environ, PATH=f"{bin_}:/usr/bin:/bin", VM_ROOT=str(tmp), FAIL_AT=fail_at)
        if head:
            env["STUB_HEAD"] = head
        r = subprocess.run(["bash", str(HERE / "vm_run.sh"), COMMIT, run], env=env, capture_output=True, text=True,
                           timeout=60)
        out = tmp / f"run_{run}" / "out"
        status = json.loads((out / "status.json").read_text()) if (out / "status.json").exists() else None
        done = (out / "DONE").read_text().strip() if (out / "DONE").exists() else None
        return r.returncode, status, done

    def test_success_records_exactly_the_required_gates(self):
        rc, s, done = self.run_vm()
        self.assertEqual(rc, 0)
        self.assertEqual(done, "r1")
        self.assertTrue(s["pass"])
        self.assertEqual(set(s["gates"]), REQUIRED)
        self.assertTrue(all(v == 0 for v in s["gates"].values()))
        self.assertEqual((s["exit_status"], s["commit"], s["head"]), (0, COMMIT, COMMIT))

    def test_prerequisite_failures_stop_and_record_the_real_status(self):
        for fail_at, gate, code in (("checkout", "checkout", 5), ("configure", "configure", 6), ("build", "build", 4),
                                    ("host_test", "host_test", 3)):
            rc, s, done = self.run_vm(fail_at)
            self.assertEqual(rc, 1, fail_at)
            self.assertEqual(done, "r1")
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
        for kw in ({"fail_at": "dirty"}, {"head": "f" * 40}):
            rc, s, _ = self.run_vm(**kw)
            self.assertEqual(rc, 1)
            self.assertEqual(s["gates"]["checkout"], 99)

    def test_bad_arguments_refused_without_a_run_directory(self):
        r = subprocess.run(["bash", str(HERE / "vm_run.sh"), "main", "r1"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)


if __name__ == "__main__":
    unittest.main()
