"""Tests of vm_run.sh's gate bookkeeping, run-directory claim and stop handling with stub tools on PATH (no GPU, no
network). python3 -m unittest test_vm_run -v"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
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
PREFILL = {"build", "checkout", "configure", "engine_unchanged", "host_test", "prefill_gate", "prefill_gate_cublas",
           "prefill_profile", "prefill_profile_cublas", "prefill_time", "prefill_time_cublas", "provenance",
           "resources"}

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
  merge-base) [ "$FAIL_AT" = not_descendant ] && exit 1; exit 0;;
  diff) [ "$FAIL_AT" = engine_changed ] && exit 1; echo "$*" > "$STUB_LOG/git_diff_args"; exit 0;;
  *) [ "$FAIL_AT" = checkout ] && exit 5; true;;
esac''',
    "cmake": r'''#!/usr/bin/env bash
if [ "$1" = --build ]; then
  dir=$2
  echo "$*" > "$STUB_LOG/cmake_build_args_$dir"
  [ "$FAIL_AT" = build ] && [ "$dir" = build ] && exit 4
  [ "$FAIL_AT" = build_base ] && [ "$dir" = base_build ] && exit 4
  mkdir -p "$dir"
  for t in test_selective_scan_chunked test_selective_scan_chunked_backward test_mamba_prefill_bench; do
    cat > "$dir/$t" <<'X'
#!/usr/bin/env bash
case "$0" in *backward) [ "$FAIL_AT" = host_test_bwd ] && exit 3;; *prefill*) [ "$FAIL_AT" = host_test_prefill ] && exit 3;;
  *) [ "$FAIL_AT" = host_test ] && exit 3;; esac
echo '{"pass": true}'
X
  done
  # the prefill bench: refuses the scan switch set from outside, and the profile outside --mode profile (as the
  # real tool); prints the configuration it was given
  cat > "$dir/mamba_prefill_bench" <<'X'
#!/usr/bin/env bash
mode=""; prev=""
for a in "$@"; do [ "$prev" = --mode ] && mode=$a; prev=$a; done
key=prefill_$mode; [ "$MICROTORCH_CUBLAS" = 1 ] && key=${key}_cublas
[ -n "$MICROTORCH_SCAN_CHUNK$MICROTORCH_TF32" ] && exit 2
case "$mode" in profile) [ "$MICROTORCH_GPU_PROFILE" = 1 ] || exit 2;; *) [ -z "$MICROTORCH_GPU_PROFILE" ] || exit 2;; esac
[ "$FAIL_AT" = "hang_$mode" ] && exec sleep 120
[ "$FAIL_AT" = "$key" ] && exit 1
echo "{\"mode\": \"$mode\", \"cublas\": \"$MICROTORCH_CUBLAS\", \"profile\": \"$MICROTORCH_GPU_PROFILE\", \"args\": \"$*\"}"
X
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
case "$*" in *memory.free*nounits*) [ "$FAIL_AT" = low_device ] && echo 2000 || echo 22500; exit 0;; esac
case "$*" in *memory.free*) echo "name, memory.total [MiB], memory.used [MiB], memory.free [MiB]"
  echo "NVIDIA L4, 23034 MiB, 0 MiB, 22500 MiB"; exit 0;; esac
echo "NVIDIA L4, 580.82.07"
[ "$FAIL_AT" = two_gpus ] && echo "NVIDIA L4, 580.82.07"
true''',
    "python3": r'''#!/usr/bin/env bash
case "$*" in *replay_selective_scan.py*) [ "$FAIL_AT" = replay ] && exit 2; exit 0;; esac
case "$*" in *pip*) [ "$FAIL_AT" = numpy ] && exit 1; exit 0;; esac
# measure.py OUT -- CMD...: run CMD and record it, as the real helper does (tested on its own below)
case "$1" in *measure.py) out=$2; shift 3; "$@"; rc=$?
  case "$FAIL_AT" in no_receipt) ;; no_receipt_*) [ "$out" = "${out%/${FAIL_AT#no_receipt_}.time.json}" ] && echo "{\"exit\": $rc}" > "$out";;
    *) echo "{\"exit\": $rc}" > "$out";; esac
  exit $rc;; esac
[ "$1" = --version ] && echo "Python 3.13.0"
exit 0''',
    "nproc": "#!/usr/bin/env bash\necho ${STUB_NPROC:-2}",
}


class VmRunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        bin_ = self.tmp / "bin"
        bin_.mkdir()
        for name, body in STUBS.items():
            (bin_ / name).write_text(body + "\n")
            (bin_ / name).chmod(0o755)
        self.env = dict(os.environ, PATH=f"{bin_}:/usr/bin:/bin", VM_TEST_HARNESS="1", VM_ROOT=str(self.tmp),
                        STUB_LOG=str(self.tmp))
        for k in ("MICROTORCH_SCAN_CHUNK", "MICROTORCH_CUBLAS", "MICROTORCH_GPU_PROFILE", "MICROTORCH_TF32"):
            self.env.pop(k, None)
        # the prefill resources check reads these instead of the machine's own: a 50 GiB host, and by default a v2
        # process in the root of a cgroup2 mount with no limit
        self.meminfo(50 << 20)
        self.cgroup_v2("/", "/")
        self.env.update(VM_MEMINFO=str(self.tmp / "meminfo"), VM_PROC_CGROUP=str(self.tmp / "proc_cgroup"),
                        VM_MOUNTINFO=str(self.tmp / "mountinfo"))

    def meminfo(self, available_kib):
        (self.tmp / "meminfo").write_text(f"MemTotal:       {available_kib + 1024} kB\n"
                                          f"MemFree:        {available_kib} kB\n"
                                          f"MemAvailable:   {available_kib} kB\n")

    def cgroup_v2(self, path, mount_root):
        """This process in v2 cgroup PATH, under a cgroup2 mount (at tmp/cg2) whose root is MOUNT_ROOT; returns the
        mount directory. The cgroup's directories exist, with no limit files until a test writes them."""
        mnt = self.tmp / "cg2"
        (self.tmp / "proc_cgroup").write_text(f"0::{path}\n")
        (self.tmp / "mountinfo").write_text(
            "22 1 8:1 / / rw,relatime shared:1 - ext4 /dev/sda1 rw\n"
            f"30 22 0:26 {mount_root} {mnt} rw,nosuid,nodev,noexec shared:9 - cgroup2 cgroup2 rw,nsdelegate\n")
        rel = path[len(mount_root):] if mount_root != "/" and path.startswith(mount_root) else path
        mnt.mkdir(parents=True, exist_ok=True)
        if ".." not in rel.split("/"):
            (mnt / rel.strip("/")).mkdir(parents=True, exist_ok=True)
        return mnt

    def cgroup_v1(self, path, mount_root):
        """As cgroup_v2, for a v1 memory controller mounted at tmp/cg1 (the unified line is present too, as on a
        hybrid host; the memory controller's line wins)."""
        mnt = self.tmp / "cg1"
        (self.tmp / "proc_cgroup").write_text(f"12:pids:/x\n4:memory:{path}\n0::/\n")
        (self.tmp / "mountinfo").write_text(
            "22 1 8:1 / / rw,relatime shared:1 - ext4 /dev/sda1 rw\n"
            f"31 22 0:27 / {self.tmp}/cg1pids rw,nosuid shared:10 - cgroup cgroup rw,pids\n"
            f"32 22 0:28 {mount_root} {mnt} rw,nosuid shared:11 - cgroup cgroup rw,memory\n")
        rel = path[len(mount_root):] if mount_root != "/" and path.startswith(mount_root) else path
        mnt.mkdir(parents=True, exist_ok=True)
        if ".." not in rel.split("/"):
            (mnt / rel.strip("/")).mkdir(parents=True, exist_ok=True)
        return mnt

    @staticmethod
    def limit(d, limit_name, limit, usage_name=None, usage=None):
        d.mkdir(parents=True, exist_ok=True)
        (d / limit_name).write_text(f"{limit}\n")
        if usage_name:
            (d / usage_name).write_text(f"{usage}\n")

    def resources(self, run):
        return json.loads((self.tmp / f"run_{run}" / "out" / "resources_start.json").read_text())

    def assert_resources_stop(self, run, why):
        rc, s, done = self.run_vm(run=run, protocol="prefill")
        self.assertEqual((rc, done), (1, run), why)
        self.assertEqual(s["gates"], {"provenance": 0, "resources": 1}, why)  # no checkout, build or gate
        r = self.resources(run)
        self.assertFalse(r["ok"], why)
        return r

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

    # ---- prefill protocol
    def test_prefill_success_records_exactly_its_gates(self):
        rc, s, done = self.run_vm(protocol="prefill")
        self.assertEqual(rc, 0, s)
        self.assertEqual(done, "r1")
        self.assertTrue(s["pass"])
        self.assertEqual(set(s["gates"]), PREFILL)
        self.assertTrue(all(v == 0 for v in s["gates"].values()))
        self.assertEqual((s["protocol"], s["head"]), ("prefill", COMMIT))
        out = self.tmp / "run_r1" / "out"
        # each bench step ran its mode with its GEMM backend and profile switch, and --require-cuda
        for step, mode, cublas, profile in (("prefill_gate", "gate", "", ""), ("prefill_gate_cublas", "gate", "1", ""),
                                            ("prefill_time", "time", "", ""), ("prefill_time_cublas", "time", "1", ""),
                                            ("prefill_profile", "profile", "", "1"),
                                            ("prefill_profile_cublas", "profile", "1", "1")):
            row = json.loads((out / f"{step}.jsonl").read_text())
            self.assertEqual((row["mode"], row["cublas"], row["profile"]), (mode, cublas, profile), step)
            self.assertIn("--require-cuda", row["args"], step)
        # the engine and the model are compared with the prefill base, and nothing else is
        diff = (self.tmp / "git_diff_args").read_text().split()
        self.assertEqual(diff[-4:], ["--", "src", "include", "tools/parity_model.hpp"])
        self.assertIn("da5febf2f0a4578bbf17384123c8962e8c623bf5", diff)
        self.assertFalse((out / "scan_bench.jsonl").exists())  # no other protocol's work

    def test_prefill_records_resources_and_step_measurements(self):
        rc, s, _ = self.run_vm(protocol="prefill")
        self.assertEqual(rc, 0, s)
        out = self.tmp / "run_r1" / "out"
        for label in ("start", "before_gate", "end"):
            r = json.loads((out / f"resources_{label}.json").read_text())
            self.assertEqual((r["mem_available_kib"], r["device_free_mib"], r["ok"]), (50 << 20, 22500, True), label)
            self.assertEqual((r["cgroup_version"], r["cgroup_headroom_kib"], r["cgroup_error"]), ("v2", None, ""))
            self.assertTrue((out / f"meminfo_{label}.txt").is_file() and (out / f"gpu_memory_{label}.txt").is_file())
        for step in ("build", "host_test", "prefill_gate", "prefill_gate_cublas", "prefill_time", "prefill_time_cublas",
                     "prefill_profile", "prefill_profile_cublas"):
            self.assertEqual(json.loads((out / f"{step}.time.json").read_text())["exit"], 0, step)

    def test_prefill_build_jobs_are_bounded_by_host_memory(self):
        self.meminfo(17 << 20)  # 17 GiB: 5 jobs of 3 GiB, though nproc says 16
        rc, s, _ = self.run_vm(protocol="prefill", extra={"STUB_NPROC": "16"})
        self.assertEqual(rc, 0, s)
        self.assertIn("--parallel 5", (self.tmp / "cmake_build_args_build").read_text())
        rc, s, _ = self.run_vm(run="r2", protocol="prefill", extra={"STUB_NPROC": "3"})
        self.assertIn("--parallel 3", (self.tmp / "cmake_build_args_build").read_text())  # nproc bounds it too

    def test_prefill_short_host_or_device_memory_stops_before_any_heavy_work(self):
        self.meminfo(15 << 20)
        self.assert_resources_stop("m1", "host")
        self.meminfo(50 << 20)
        rc, s, _ = self.run_vm("low_device", run="m2", protocol="prefill")
        self.assertEqual((rc, s["gates"]), (1, {"provenance": 0, "resources": 1}))

    # cgroup v2: this process's own cgroup, resolved from /proc/self/cgroup and mountinfo, and its ancestors
    def test_v2_limit_on_an_ancestor_of_a_nested_cgroup_is_found(self):
        mnt = self.cgroup_v2("/a/b", "/")
        self.limit(mnt / "a", "memory.max", 24 << 30, "memory.current", 12 << 30)  # 12 GiB of room above us
        self.limit(mnt / "a" / "b", "memory.max", "max", "memory.current", 1 << 30)
        r = self.assert_resources_stop("n1", "ancestor limit")
        self.assertEqual((r["cgroup_version"], r["cgroup_path"], r["headroom_kib"]), ("v2", "/a/b", 12 << 20))
        self.assertIn(f"{mnt}/a=", r["cgroup_levels"])
        # with room at the ancestor too, the smallest headroom over the levels is the one that counts
        self.limit(mnt / "a", "memory.max", 40 << 30, "memory.current", 4 << 30)       # 36 GiB
        self.limit(mnt / "a" / "b", "memory.max", 30 << 30, "memory.current", 2 << 30)  # 28 GiB
        rc, s, _ = self.run_vm(run="n2", protocol="prefill")
        self.assertEqual(rc, 0, s)
        r = self.resources("n2")
        self.assertEqual((r["cgroup_headroom_kib"], r["headroom_kib"], r["ok"]), (28 << 20, 28 << 20, True))

    def test_v2_limit_on_the_process_own_nested_cgroup_is_found(self):
        mnt = self.cgroup_v2("/system.slice/colab.scope", "/")
        self.limit(mnt / "system.slice" / "colab.scope", "memory.max", 12 << 30, "memory.current", 0)
        r = self.assert_resources_stop("o1", "own limit")
        self.assertEqual(r["headroom_kib"], 12 << 20)

    def test_v2_mount_root_inside_the_hierarchy_maps_the_path(self):
        # a container's cgroup2 mount whose root is /docker/abc: /proc/self/cgroup's /docker/abc/app is mnt/app
        mnt = self.cgroup_v2("/docker/abc/app", "/docker/abc")
        self.limit(mnt, "memory.max", 12 << 30, "memory.current", 1 << 30)  # the container's limit at the mount
        r = self.assert_resources_stop("p1", "limit at the mount root")
        self.assertEqual(r["headroom_kib"], 11 << 20)

    def test_finite_limit_without_valid_usage_is_never_zero_usage(self):
        for i, usage in enumerate((None, "abc", "", "-5")):
            mnt = self.cgroup_v2("/a", "/")
            if usage is None:
                self.limit(mnt / "a", "memory.max", 24 << 30)
                (mnt / "a" / "memory.current").unlink(missing_ok=True)
            else:
                self.limit(mnt / "a", "memory.max", 24 << 30, "memory.current", usage)
            r = self.assert_resources_stop(f"u{i}", f"usage {usage!r}")
            self.assertIn("usage missing or malformed", r["cgroup_error"])

    def test_unmappable_membership_is_an_error_not_no_limit(self):
        self.cgroup_v2("/a", "/")
        (self.tmp / "mountinfo").write_text("22 1 8:1 / / rw - ext4 /dev/sda1 rw\n")  # no cgroup2 mount
        r = self.assert_resources_stop("q1", "no mount")
        self.assertIn("no matching mount", r["cgroup_error"])
        mnt = self.cgroup_v2("/gone", "/")
        (mnt / "gone").rmdir()
        r = self.assert_resources_stop("q3", "missing directory")
        self.assertIn("missing", r["cgroup_error"])
        mnt = self.cgroup_v2("/a", "/")
        self.limit(mnt / "a", "memory.max", "12G", "memory.current", 0)
        r = self.assert_resources_stop("q4", "unreadable limit")
        self.assertIn("unreadable limit", r["cgroup_error"])

    def test_missing_mount_directory_stops_even_outside_the_namespace(self):
        for run, path in (("m1", "/a"), ("m2", "/../../jupyter-children")):
            mnt = self.cgroup_v2(path, "/")
            shutil.rmtree(mnt)
            r = self.assert_resources_stop(run, f"mount directory missing, {path}")
            self.assertIn(f"cgroup mount {mnt} missing", r["cgroup_error"])

    # Colab's layout (run p1-20261103): the notebook's cgroup sits above the container's cgroup-namespace root
    def test_colab_layout_outside_the_namespace_passes_on_mem_available(self):
        mnt = self.cgroup_v2("/../../jupyter-children", "/")
        rc, s, _ = self.run_vm(run="c1", protocol="prefill")
        self.assertEqual(rc, 0, s)
        r = self.resources("c1")
        self.assertEqual((r["cgroup_version"], r["cgroup_path"], r["cgroup_scope"]),
                         ("v2", "/../../jupyter-children", "namespace_root"))
        self.assertIn("outside the visible cgroup namespace", r["cgroup_note"])
        self.assertEqual((r["cgroup_headroom_kib"], r["headroom_kib"], r["cgroup_error"]), (None, 50 << 20, ""))
        self.assertEqual(r["cgroup_levels"], f"{mnt}=no-limit-file;")
        self.assertFalse((self.tmp.parent / "jupyter-children").exists())  # nothing was looked up above the mount

    def test_colab_layout_applies_a_finite_limit_at_the_visible_mount_root(self):
        mnt = self.cgroup_v2("/../../jupyter-children", "/")
        self.limit(mnt, "memory.max", 12 << 30, "memory.current", 1 << 30)
        r = self.assert_resources_stop("c2", "mount-root limit")
        self.assertEqual((r["cgroup_scope"], r["headroom_kib"]), ("namespace_root", 11 << 20))
        self.limit(mnt, "memory.max", 40 << 30, "memory.current", 4 << 30)
        rc, s, _ = self.run_vm(run="c3", protocol="prefill")
        self.assertEqual(rc, 0, s)
        self.assertEqual(self.resources("c3")["headroom_kib"], 36 << 20)
        self.limit(mnt, "memory.max", 40 << 30)
        (mnt / "memory.current").unlink()
        r = self.assert_resources_stop("c4", "mount-root limit without usage")
        self.assertIn("usage missing or malformed", r["cgroup_error"])
        self.limit(mnt, "memory.max", "40G", "memory.current", 0)
        r = self.assert_resources_stop("c5", "unreadable mount-root limit")
        self.assertIn("unreadable limit", r["cgroup_error"])

    def test_paths_outside_the_mounted_subtree_use_the_visible_root(self):
        # a climb that ends elsewhere, and a path not under a bind mount's root, both leave the visible namespace
        for i, (path, root) in enumerate((("/a/../../b", "/"), ("/elsewhere/a", "/docker/abc"))):
            self.cgroup_v2(path, root)
            rc, s, _ = self.run_vm(run=f"e{i}", protocol="prefill")
            self.assertEqual(rc, 0, (path, s))
            self.assertEqual(self.resources(f"e{i}")["cgroup_scope"], "namespace_root", path)
        # a climb that stays inside the mount is an ordinary path, and its missing directory still stops
        self.cgroup_v2("/a/../b", "/")
        r = self.assert_resources_stop("e2", "inside, missing")
        self.assertEqual(r["cgroup_scope"], "process")
        self.assertIn("missing", r["cgroup_error"])
        mnt = self.cgroup_v2("/a/../b", "/")
        (mnt / "b").mkdir()
        self.limit(mnt / "b", "memory.max", 12 << 30, "memory.current", 0)
        r = self.assert_resources_stop("e3", "inside, own limit")
        self.assertEqual(r["headroom_kib"], 12 << 20)

    def test_v1_outside_the_namespace(self):
        mnt = self.cgroup_v1("/../x", "/")
        self.limit(mnt, "memory.limit_in_bytes", 9223372036854771712, "memory.usage_in_bytes", 1 << 30)
        rc, s, _ = self.run_vm(run="g1", protocol="prefill")
        self.assertEqual(rc, 0, s)
        r = self.resources("g1")
        self.assertEqual((r["cgroup_version"], r["cgroup_scope"], r["cgroup_levels"]),
                         ("v1", "namespace_root", f"{mnt}=unlimited;"))

    def test_v2_all_max_levels_are_recorded_and_unlimited(self):
        mnt = self.cgroup_v2("/a/b", "/")
        self.limit(mnt / "a", "memory.max", "max", "memory.current", 3 << 30)
        self.limit(mnt / "a" / "b", "memory.max", "max", "memory.current", 2 << 30)
        rc, s, _ = self.run_vm(run="x1", protocol="prefill")
        self.assertEqual(rc, 0, s)
        r = self.resources("x1")
        self.assertEqual((r["cgroup_scope"], r["cgroup_headroom_kib"], r["headroom_kib"]), ("process", None, 50 << 20))
        self.assertEqual(r["cgroup_levels"], f"{mnt}/a/b=unlimited;{mnt}/a=unlimited;{mnt}=no-limit-file;")

    def test_unreadable_or_unparsable_proc_cgroup_stops(self):
        (self.tmp / "proc_cgroup").unlink()
        r = self.assert_resources_stop("k1", "missing /proc/self/cgroup")
        self.assertIn("cannot read", r["cgroup_error"])
        for i, text in enumerate(("", "garbage\n", "0::/\nnot a line\n", "x:memory:/a\n")):
            (self.tmp / "proc_cgroup").write_text(text)
            r = self.assert_resources_stop(f"k{i + 2}", repr(text))
            self.assertIn("unparsable", r["cgroup_error"])

    def test_no_memory_cgroup_membership_uses_mem_available(self):
        (self.tmp / "proc_cgroup").write_text("12:pids:/x\n")  # no memory controller, no unified line
        rc, s, _ = self.run_vm(run="z1", protocol="prefill")
        self.assertEqual(rc, 0, s)
        r = self.resources("z1")
        self.assertEqual((r["cgroup_version"], r["headroom_kib"]), ("none", 50 << 20))

    # cgroup v1: the memory controller's line, under the memory mount
    def test_v1_nested_and_ancestor_limits(self):
        mnt = self.cgroup_v1("/docker/x/y", "/docker")
        unlimited = 9223372036854771712
        self.limit(mnt / "x" / "y", "memory.limit_in_bytes", unlimited, "memory.usage_in_bytes", 1 << 30)
        self.limit(mnt / "x", "memory.limit_in_bytes", 20 << 30, "memory.usage_in_bytes", 6 << 30)  # 14 GiB
        r = self.assert_resources_stop("v1", "v1 ancestor limit")
        self.assertEqual((r["cgroup_version"], r["cgroup_path"], r["headroom_kib"]), ("v1", "/docker/x/y", 14 << 20))
        self.limit(mnt / "x", "memory.limit_in_bytes", 40 << 30, "memory.usage_in_bytes", 4 << 30)
        self.limit(mnt / "x" / "y", "memory.limit_in_bytes", 13 << 30, "memory.usage_in_bytes", 0)  # its own: 13
        r = self.assert_resources_stop("v2", "v1 own limit")
        self.assertEqual(r["headroom_kib"], 13 << 20)
        self.limit(mnt / "x" / "y", "memory.limit_in_bytes", 30 << 30)
        (mnt / "x" / "y" / "memory.usage_in_bytes").unlink()
        r = self.assert_resources_stop("v3", "v1 limit without usage")
        self.assertIn("usage missing or malformed", r["cgroup_error"])
        self.limit(mnt / "x" / "y", "memory.limit_in_bytes", 30 << 30, "memory.usage_in_bytes", "n/a")
        r = self.assert_resources_stop("v4", "v1 malformed usage")
        self.assertIn("usage missing or malformed", r["cgroup_error"])
        self.limit(mnt / "x" / "y", "memory.limit_in_bytes", unlimited, "memory.usage_in_bytes", 1 << 30)
        rc, s, _ = self.run_vm(run="v5", protocol="prefill")  # x has 36 GiB, y no limit
        self.assertEqual(rc, 0, s)
        self.assertEqual(self.resources("v5")["headroom_kib"], 36 << 20)

    def test_missing_measurement_receipt_fails_the_step(self):
        rc, s, _ = self.run_vm("no_receipt", run="w1", protocol="prefill")  # every receipt missing: the build fails
        self.assertEqual((rc, s["gates"]["build"]), (1, 96))
        self.assertNotIn("host_test", s["gates"])
        rc, s, _ = self.run_vm("no_receipt_prefill_time", run="w2", protocol="prefill")
        self.assertEqual((rc, s["gates"]["prefill_time"]), (1, 96))
        self.assertEqual(s["gates"]["prefill_profile"], 0)  # the other steps still ran
        self.assertFalse(s["pass"])

    def test_measure_records_time_rss_and_the_exit_status(self):
        out = self.tmp / "m.json"
        r = subprocess.run([sys.executable, str(HERE / "measure.py"), str(out), "--", "sh", "-c", "echo hi; exit 3"],
                           capture_output=True, text=True, timeout=30)
        self.assertEqual((r.returncode, r.stdout), (3, "hi\n"))  # its own status, its output passed through
        m = json.loads(out.read_text())
        self.assertEqual((m["exit"], m["cmd"][0]), (3, "sh"))
        self.assertGreater(m["max_rss_kib"], 0)
        self.assertGreaterEqual(m["elapsed_s"], 0)
        r = subprocess.run([sys.executable, str(HERE / "measure.py"), str(out), "--", "sh", "-c", "kill -TERM $$"],
                           capture_output=True, text=True, timeout=30)
        self.assertEqual((r.returncode, json.loads(out.read_text())["exit"]), (143, 143))
        r = subprocess.run([sys.executable, str(HERE / "measure.py"), str(out)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
        # a receipt that cannot be written: the command's own status stands, and the failure is reported
        bad = self.tmp / "no_such_dir" / "m.json"
        r = subprocess.run([sys.executable, str(HERE / "measure.py"), str(bad), "--", "sh", "-c", "exit 3"],
                           capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 3)
        self.assertIn("could not write", r.stderr)
        self.assertFalse(bad.exists())
        r = subprocess.run([sys.executable, str(HERE / "measure.py"), str(bad), "--", "true"], capture_output=True,
                           text=True, timeout=30)
        self.assertEqual(r.returncode, 0)  # a success without its receipt; vm_run.sh fails that step (test above)

    def test_prefill_runs_with_the_switch_unset_even_if_the_vm_has_it(self):
        rc, s, _ = self.run_vm(protocol="prefill", extra={"MICROTORCH_SCAN_CHUNK": "64", "MICROTORCH_TF32": "1"})
        self.assertEqual(rc, 0, s)  # the stub, like the bench, refuses them; vm_run.sh unsets both

    def test_prefill_prerequisites_stop_before_timing(self):
        for fail_at, gate in (("two_gpus", "provenance"), ("engine_changed", "engine_unchanged"),
                              ("not_descendant", "engine_unchanged"), ("build", "build"),
                              ("host_test_prefill", "host_test"), ("prefill_gate", "prefill_gate"),
                              ("prefill_gate_cublas", "prefill_gate_cublas")):
            rc, s, done = self.run_vm(fail_at, run=f"p_{fail_at}", protocol="prefill")
            self.assertEqual(rc, 1, fail_at)
            self.assertEqual(done, f"p_{fail_at}")
            self.assertFalse(s["pass"], fail_at)
            self.assertNotEqual(s["gates"][gate], 0, fail_at)
            for later in ("prefill_time", "prefill_time_cublas", "prefill_profile", "prefill_profile_cublas"):
                self.assertNotIn(later, s["gates"], fail_at)  # no timing without the correctness gates

    def test_prefill_timing_failure_fails_and_still_collects_the_rest(self):
        for fail_at in ("prefill_time", "prefill_time_cublas", "prefill_profile", "prefill_profile_cublas"):
            rc, s, _ = self.run_vm(fail_at, run=f"t_{fail_at}", protocol="prefill")
            self.assertEqual(rc, 1, fail_at)
            self.assertFalse(s["pass"], fail_at)
            self.assertEqual(s["gates"][fail_at], 1, fail_at)
            self.assertEqual(set(s["gates"]), PREFILL, fail_at)  # the other steps still ran

    def test_prefill_stop_during_timing_stops_its_group(self):
        t0 = time.monotonic()
        rc, s, done = self.run_vm("hang_time", protocol="prefill", extra={"VM_TEST_SIGNAL_AT": "run_prefill_time"})
        self.assertLess(time.monotonic() - t0, 60)
        self.assertEqual((rc, done), (1, "r1"))
        self.assertEqual(s["gates"]["prefill_time"], 143)
        self.assertNotIn("prefill_profile", s["gates"])
        stop = json.loads((self.tmp / "run_r1" / "out" / "stop.json").read_text())
        self.assertEqual((stop["step"], stop["step_group_gone"]), ("prefill_time", True))
        ps = subprocess.run(["ps", "-eo", "args="], capture_output=True, text=True).stdout
        self.assertNotIn("sleep 120", ps)

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
