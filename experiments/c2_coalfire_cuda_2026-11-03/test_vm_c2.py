"""Tests of vm_c2.sh with stub tools (no GPU, no network): python3 -m unittest test_vm_c2 -v

The stubs: nvidia-smi and nvcc report one L4 (or STUB_GPUS); git clones by copying this packet (src) or making a
coalfire tree with a small corpus; cmake writes compile_commands.json (with or without -DUSE_CUDA) and the library;
g++ writes a coalfire_dump stand-in that behaves as the real tool does with and without a visible device. The
packet copy in src has inputs.sha256 and corpus.sha256 for the stub files, and a compare_cuda.py stand-in that exits
with STUB_COMPARE_RC (compare_cuda.py itself is tested in test_compare_cuda.py)."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest

HERE = Path(__file__).resolve().parent
COMMIT = "0123456789abcdef0123456789abcdef01234567"
COALFIRE = "605685e8e212eecdf86d01630f76b9ac47b5d5a8"
GATES = {"build", "checkout", "compare", "configure", "corpus", "device_required", "dump_build", "dump_cuda",
         "dump_cuda_repeat", "inputs", "numpy", "provenance"}
UPLOADS = ["pol_checkpoint_0.ckpt", "model.safetensors", "config.json", "ids.txt", "coalfire.f32", "paperkiln.f32"]
CORPUS = "".join(f"word{i} | other{i}\n" for i in range(40))

STUBS = {
    "nvidia-smi": '#!/usr/bin/env bash\nfor i in $(seq ${STUB_GPUS:-1}); do echo "${STUB_GPU_NAME:-NVIDIA L4}"; done\n',
    "nvcc": '#!/usr/bin/env bash\necho "Cuda compilation tools, release 13.0, V13.0.88"\n',
    "git": r'''#!/usr/bin/env bash
if [ "$1" = clone ]; then
    dst=${@: -1}
    case "$3" in
        *paperkiln*) mkdir -p "$dst/experiments" && cp -r "$STUB_PACKET" "$dst/experiments/c2_coalfire_cuda_2026-11-03" ;;
        *coalfire*) mkdir -p "$dst/data" && printf '%s' "$STUB_CORPUS" > "$dst/data/training_pairs.txt" ;;
    esac
    exit 0
fi
if [ "$1" = -C ]; then
    d=$2; shift 2
    [ "$1" = -c ] && shift 2
    case "$1" in
        checkout) exit 0 ;;
        rev-parse)
            case "$d" in
                *coalfire) echo "${STUB_COALFIRE_HEAD:-605685e8e212eecdf86d01630f76b9ac47b5d5a8}" ;;
                *) echo "${STUB_HEAD:-0123456789abcdef0123456789abcdef01234567}" ;;
            esac; exit 0 ;;
        status) [ "${STUB_DIRTY:-}" = "$(basename "$d")" ] && echo " M x"; exit 0 ;;
    esac
fi
exit 1
''',
    "cmake": r'''#!/usr/bin/env bash
if [ "$1" = -S ]; then
    b=$4; mkdir -p "$b"
    flags='-DUSE_CUDA -DCUDA_AVAILABLE -I/x/include -O3 -std=gnu++17 -fopenmp'
    [ "${STUB_NO_USE_CUDA:-}" = 1 ] && flags='-I/x/include -O3 -std=gnu++17 -fopenmp'
    printf '[{"directory": "%s", "command": "/usr/bin/c++ %s -o CMakeFiles/t.dir/src/transformer.cpp.o -c %s/coalfire/src/transformer.cpp", "file": "%s/coalfire/src/transformer.cpp"}]\n' \
        "$PWD/$b" "$flags" "$PWD" "$PWD" > "$b/compile_commands.json"
    exit 0
fi
[ "$1" = --build ] && { [ "${STUB_BUILD_FAIL:-}" = 1 ] && exit 2; : > "$2/libtransformer_core.a"; exit 0; }
exit 1
''',
    "g++": r'''#!/usr/bin/env bash
out=""; prev=""
for a in "$@"; do [ "$prev" = -o ] && out=$a; prev=$a; done
echo "$*" > "$out.args"
cat > "$out" <<'DUMP'
#!/usr/bin/env bash
[ "${NVIDIA_TF32_OVERRIDE:-}" = 0 ] || { echo "TF32 override missing" >&2; exit 7; }
if [ "${CUDA_VISIBLE_DEVICES-unset}" = "" ] && [ "${STUB_NODEVICE_RUNS:-}" != 1 ]; then
    echo '{"use_cuda": true, "devices": 0, "device": "", "cc": "0.0", "error": "cudaErrorNoDevice"}'
    echo "coalfire_dump: CUDA build without a visible device" >&2; exit 3
fi
echo "{\"use_cuda\": true, \"devices\": 1, \"device\": \"${STUB_DEVICE:-NVIDIA L4}\", \"cc\": \"8.9\", \"error\": \"\"}"
printf 'logits' > "$6"
DUMP
chmod +x "$out"
''',
}


class VmC2Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.bin = self.tmp / "cuda" / "bin"
        self.bin.mkdir(parents=True)
        for name, body in STUBS.items():
            (self.bin / name).write_text(body)
            (self.bin / name).chmod(0o755)
        # the packet as the VM's clone holds it, with hashes for the stub inputs and corpus
        self.packet = self.tmp / "packet"
        shutil.copytree(HERE, self.packet, ignore=shutil.ignore_patterns("receipts", "__pycache__"))
        (self.packet / "compare_cuda.py").write_text(
            "import os, sys\nopen(sys.argv[7], 'w').write('{}')\nsys.exit(int(os.environ.get('STUB_COMPARE_RC', '0')))\n")
        content = self.tmp / "content"
        content.mkdir()
        self.uploads = content / "upload_r1"

    def write_inputs(self, run="r1", alter=None, extra=None):
        up = self.tmp / "content" / f"upload_{run}"
        up.mkdir(exist_ok=True)
        for n in UPLOADS:
            (up / n).write_text(f"stub {n}\n")
        lines = subprocess.run(["sha256sum"] + UPLOADS, cwd=up, capture_output=True, text=True).stdout
        (self.packet / "inputs.sha256").write_text(lines)
        if alter:
            (up / alter).write_text("altered\n")
        if extra:
            (up / extra).write_text("extra\n")
        n = CORPUS.count("\n")
        rows = CORPUS.splitlines(keepends=True)
        c0 = "".join(rows[:n // 2]).replace("|", " ")
        c1 = "".join(rows[n // 2:]).replace("|", " ")
        cdir = self.tmp / "corpus"
        cdir.mkdir(exist_ok=True)
        (cdir / "train-0-small.txt").write_text(c0)
        (cdir / "train-1-small.txt").write_text(c1)
        (self.packet / "corpus.sha256").write_text(subprocess.run(
            ["sha256sum", "train-0-small.txt", "train-1-small.txt"], cwd=cdir, capture_output=True, text=True).stdout)

    def run_vm(self, run="r1", extra_env=None, args=None, write=True, **kw):
        if write:
            self.write_inputs(run, **kw)
        env = dict(os.environ, VM_TEST_HARNESS="1", VM_ROOT=str(self.tmp / "content"),
                   PATH=f"{self.bin}:{os.environ['PATH']}", STUB_PACKET=str(self.packet), STUB_CORPUS=CORPUS)
        env.update(extra_env or {})
        argv = ["bash", str(HERE / "vm_c2.sh")] + (args if args is not None else [COMMIT, run])
        r = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=120)
        out = self.tmp / "content" / f"run_{run}" / "out"
        s = json.loads((out / "status.json").read_text()) if (out / "status.json").exists() else None
        done = (out / "DONE").read_text().strip() if (out / "DONE").exists() else None
        return r.returncode, s, done, out, r

    def test_success_records_exactly_the_required_gates(self):
        rc, s, done, out, r = self.run_vm()
        self.assertEqual(rc, 0, r.stderr + json.dumps(s))
        self.assertEqual(done, "r1")
        self.assertEqual(set(s["gates"]), GATES)
        self.assertTrue(all(v == 0 for v in s["gates"].values()))
        self.assertEqual((s["protocol"], s["commit"], s["head"], s["coalfire_head"], s["pass"], s["exit_status"]),
                         ("c2cuda", COMMIT, COMMIT, COALFIRE, True, 0))
        self.assertIn('"devices": 0', (out / "dump_nodevice.log").read_text())
        self.assertIn('"device": "NVIDIA L4"', (out / "dump_cuda_repeat.log").read_text())
        self.assertIn("-DUSE_CUDA", (out / "dump_flags.txt").read_text())
        args = (self.tmp / "content" / "run_r1" / "coalfire_dump.args").read_text()
        self.assertIn("-DUSE_CUDA", args)                          # the library's own flags
        self.assertIn(f"-Wl,-rpath,{self.tmp}/cuda/lib64", args)   # and the toolkit beside nvcc
        self.assertIn("start compare", (out / "steps.log").read_text())

    def test_altered_or_extra_input_stops_before_the_build(self):
        for run, kw in (("a1", {"alter": "ids.txt"}), ("a2", {"extra": "stray.bin"})):
            rc, s, done, _, r = self.run_vm(run=run, **kw)
            self.assertEqual((rc, done), (1, run), r.stderr)
            self.assertEqual(s["gates"]["inputs"], 1)
            self.assertNotIn("configure", s["gates"])

    def test_corpus_mismatch_stops(self):
        rc, s, _, _, _ = self.run_vm(extra_env={"STUB_CORPUS": CORPUS + "one more | line\n"})
        self.assertEqual((rc, s["gates"]["corpus"]), (1, 1))
        self.assertNotIn("configure", s["gates"])

    def test_library_without_use_cuda_is_refused(self):
        rc, s, _, _, _ = self.run_vm(extra_env={"STUB_NO_USE_CUDA": "1"})
        self.assertEqual((rc, s["gates"]["dump_build"]), (1, 95))
        self.assertNotIn("dump_cuda", s["gates"])

    def test_a_dump_that_runs_without_a_device_fails_the_control(self):
        rc, s, _, _, _ = self.run_vm(extra_env={"STUB_NODEVICE_RUNS": "1"})
        self.assertEqual((rc, s["gates"]["device_required"]), (1, 1))
        self.assertNotIn("dump_cuda", s["gates"])

    def test_wrong_device_fails_the_cuda_leg(self):
        rc, s, _, _, _ = self.run_vm(extra_env={"STUB_DEVICE": "Tesla T4"})
        self.assertEqual((rc, s["gates"]["dump_cuda"]), (1, 94))

    def test_failed_comparison_fails_the_run(self):
        rc, s, done, _, _ = self.run_vm(extra_env={"STUB_COMPARE_RC": "1"})
        self.assertEqual((rc, done, s["pass"], s["gates"]["compare"]), (1, "r1", False, 1))
        self.assertEqual(set(s["gates"]), GATES)

    def test_wrong_head_or_dirty_tree_fails_checkout(self):
        for run, env in (("h1", {"STUB_HEAD": "f" * 40}), ("h2", {"STUB_COALFIRE_HEAD": "e" * 40}),
                         ("h3", {"STUB_DIRTY": "coalfire"}), ("h4", {"STUB_DIRTY": "src"})):
            rc, s, _, _, _ = self.run_vm(run=run, extra_env=env)
            self.assertEqual((rc, s["gates"]["checkout"]), (1, 99), run)

    def test_build_failure_and_a_build_that_dirties_the_tree(self):
        rc, s, _, _, _ = self.run_vm(run="b1", extra_env={"STUB_BUILD_FAIL": "1"})
        self.assertEqual((rc, s["gates"]["build"]), (1, 2))

    def test_provenance_needs_exactly_one_l4(self):
        for run, env in (("p1", {"STUB_GPUS": "2"}), ("p2", {"STUB_GPU_NAME": "Tesla T4"})):
            rc, s, done, _, _ = self.run_vm(run=run, extra_env=env)
            self.assertEqual((rc, done, s["gates"]), (1, run, {"provenance": 1}), run)

    def test_second_launch_of_a_run_is_refused_and_changes_nothing(self):
        rc, s, _, out, _ = self.run_vm()
        self.assertEqual(rc, 0)
        before = (out / "status.json").read_text()
        rc2, _, _, _, r = self.run_vm(write=False)
        self.assertEqual(rc2, 2)
        self.assertIn("refusing to reuse", r.stderr)
        self.assertEqual((out / "status.json").read_text(), before)

    def test_bad_arguments_refused_without_a_run_directory(self):
        for args in ([COMMIT, "r1", "prefill"], ["abc", "r1"], [COMMIT, "bad id!"]):
            rc, _, _, _, _ = self.run_vm(args=args, write=False)
            self.assertEqual(rc, 2, args)
        self.assertFalse((self.tmp / "content" / "run_r1").exists())

    def test_stop_during_a_step_stops_its_group_and_publishes(self):
        self.write_inputs("s1")
        env = dict(os.environ, VM_TEST_HARNESS="1", VM_ROOT=str(self.tmp / "content"),
                   PATH=f"{self.bin}:{os.environ['PATH']}", STUB_PACKET=str(self.packet), STUB_CORPUS=CORPUS,
                   VM_TEST_SIGNAL_AT="run_configure")
        t0 = time.monotonic()
        r = subprocess.run(["bash", str(HERE / "vm_c2.sh"), COMMIT, "s1"], env=env, capture_output=True, text=True,
                           timeout=120)
        out = self.tmp / "content" / "run_s1" / "out"
        s = json.loads((out / "status.json").read_text())
        stop = json.loads((out / "stop.json").read_text())
        self.assertEqual(r.returncode, 1)
        self.assertEqual(s["gates"]["configure"], 143)
        self.assertEqual((stop["step"], stop["step_group_gone"]), ("configure", True))
        self.assertEqual((out / "DONE").read_text().strip(), "s1")
        self.assertLess(time.monotonic() - t0, 60)


if __name__ == "__main__":
    unittest.main()
