"""Tests of vm_exit.sh with stub tools (no GPU, no network): python3 -m unittest test_vm_exit -v

The stubs are test_vm_c2.py's (the C2 CUDA leg), with two coalfire clones (cf_base, cf_fix) and a dump stand-in
per build that honours COALFIRE_DUMP_NORMAL_EXIT and can crash after its markers (STUB_CRASH=base|fix|both), skip
the normal-exit marker, write other logits or hang. VM_TEST_WANT is the stub logits' sha256, VM_TEST_REPS and
VM_TEST_TIMEOUT keep the runs short."""
import hashlib
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
BASE = "642de84c85669938ae66b067fa3823629016f1fb"
FIX = "72b1f7509f731f7c06346e73367bbd1e94300cbb"
GATES = {"build", "checkout", "configure", "corpus", "device_required", "dump_build", "exit_base", "exit_fix",
         "inputs", "provenance"}
UPLOADS = ["pol_checkpoint_0.ckpt", "model.safetensors", "config.json", "ids.txt", "coalfire.f32", "paperkiln.f32"]
CORPUS = "".join(f"word{i} | other{i}\n" for i in range(40))
WANT = hashlib.sha256(b"logits").hexdigest()

STUBS = {
    "nvidia-smi": '#!/usr/bin/env bash\nfor i in $(seq ${STUB_GPUS:-1}); do echo "${STUB_GPU_NAME:-NVIDIA L4}"; done\n',
    "nvcc": '#!/usr/bin/env bash\necho "Cuda compilation tools, release 13.0, V13.0.88"\n',
    "git": r'''#!/usr/bin/env bash
if [ "$1" = clone ]; then
    dst=${@: -1}
    case "$3" in
        *paperkiln*) mkdir -p "$dst/experiments" && cp -r "$STUB_PACKET" "$dst/experiments/coalfire_exit_2026-11-03" ;;
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
                *cf_base) echo "${STUB_BASE_HEAD:-642de84c85669938ae66b067fa3823629016f1fb}" ;;
                *cf_fix) echo "${STUB_FIX_HEAD:-72b1f7509f731f7c06346e73367bbd1e94300cbb}" ;;
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
    printf '[{"directory": "%s", "command": "/usr/bin/c++ %s -o t.o -c %s/src/transformer.cpp", "file": "%s/src/transformer.cpp"}]\n' \
        "$PWD/$b" "$flags" "$PWD" "$PWD" > "$b/compile_commands.json"
    exit 0
fi
[ "$1" = --build ] && { [ "${STUB_BUILD_FAIL:-}" = 1 ] && exit 2; : > "$2/libtransformer_core.a"; exit 0; }
exit 1
''',
    "g++": r'''#!/usr/bin/env bash
out=""; prev=""
for a in "$@"; do [ "$prev" = -o ] && out=$a; prev=$a; done
tag=${out##*dump_}
echo "$*" > "$out.args"
cat > "$out" <<DUMP
#!/usr/bin/env bash
TAG=$tag
DUMP
cat >> "$out" <<'DUMP'
[ "${NVIDIA_TF32_OVERRIDE:-}" = 0 ] || { echo "TF32 override missing" >&2; exit 7; }
if [ "${CUDA_VISIBLE_DEVICES-unset}" = "" ] && [ "${STUB_NODEVICE_RUNS:-}" != 1 ]; then
    echo '{"use_cuda": true, "devices": 0, "device": "", "cc": "0.0", "error": "cudaErrorNoDevice"}'; exit 3
fi
echo '{"use_cuda": true, "devices": 1, "device": "NVIDIA L4", "cc": "8.9", "error": ""}'
[ "${STUB_HANG:-}" = "$TAG" ] && sleep 30
if [ "${STUB_OTHER_LOGITS:-}" = "$TAG" ]; then printf 'other' > "$6"; else printf 'logits' > "$6"; fi
echo '{"logits_written": true}'
[ "${COALFIRE_DUMP_NORMAL_EXIT:-}" = 1 ] || { echo '{"cuda_shutdown": "ok"}'; exit 0; }
[ "${STUB_NO_NORMAL:-}" = "$TAG" ] || echo '{"exit": "normal"}'
case "${STUB_CRASH:-}" in "$TAG"|both) kill -SEGV $$ ;; esac
exit 0
DUMP
chmod +x "$out"
''',
}


class VmExitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.bin = self.tmp / "cuda" / "bin"
        self.bin.mkdir(parents=True)
        for name, body in STUBS.items():
            (self.bin / name).write_text(body)
            (self.bin / name).chmod(0o755)
        self.packet = self.tmp / "packet"
        shutil.copytree(HERE, self.packet, ignore=shutil.ignore_patterns("receipts", "__pycache__"))
        (self.tmp / "content").mkdir()

    def write_inputs(self, run="r1", alter=None):
        up = self.tmp / "content" / f"upload_{run}"
        up.mkdir(exist_ok=True)
        for n in UPLOADS:
            (up / n).write_text(f"stub {n}\n")
        lines = subprocess.run(["sha256sum"] + UPLOADS, cwd=up, capture_output=True, text=True).stdout
        (self.packet / "inputs.sha256").write_text(lines)
        if alter:
            (up / alter).write_text("altered\n")
        rows = CORPUS.splitlines(keepends=True)
        n = len(rows)
        cdir = self.tmp / "corpus"
        cdir.mkdir(exist_ok=True)
        (cdir / "train-0-small.txt").write_text("".join(rows[:n // 2]).replace("|", " "))
        (cdir / "train-1-small.txt").write_text("".join(rows[n // 2:]).replace("|", " "))
        (self.packet / "corpus.sha256").write_text(subprocess.run(
            ["sha256sum", "train-0-small.txt", "train-1-small.txt"], cwd=cdir, capture_output=True, text=True).stdout)

    def run_vm(self, run="r1", extra_env=None, args=None, write=True, **kw):
        if write:
            self.write_inputs(run, **kw)
        env = dict(os.environ, VM_TEST_HARNESS="1", VM_ROOT=str(self.tmp / "content"),
                   PATH=f"{self.bin}:{os.environ['PATH']}", STUB_PACKET=str(self.packet), STUB_CORPUS=CORPUS,
                   VM_TEST_REPS="3", VM_TEST_WANT=WANT, VM_TEST_TIMEOUT="5")
        env.update(extra_env or {})
        argv = ["bash", str(HERE / "vm_exit.sh")] + (args if args is not None else [COMMIT, run])
        r = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=120)
        out = self.tmp / "content" / f"run_{run}" / "out"
        s = json.loads((out / "status.json").read_text()) if (out / "status.json").exists() else None
        done = (out / "DONE").read_text().strip() if (out / "DONE").exists() else None
        return r.returncode, s, done, out, r

    def summary(self, out):
        return json.loads((out / "summary.json").read_text())

    def test_base_crashes_fix_exits_cleanly_passes(self):
        rc, s, done, out, r = self.run_vm(extra_env={"STUB_CRASH": "base"})
        self.assertEqual(rc, 0, r.stderr + json.dumps(s))
        self.assertEqual((done, set(s["gates"]), s["pass"]), ("r1", GATES, True))
        self.assertTrue(all(v == 0 for v in s["gates"].values()))
        self.assertEqual((s["protocol"], s["head"], s["coalfire_base_head"], s["coalfire_fix_head"]),
                         ("cfexit", COMMIT, BASE, FIX))
        sm = self.summary(out)
        self.assertEqual((sm["base"]["crashed"], sm["base"]["exit_codes"]), (3, [139]))
        self.assertEqual((sm["fix"]["exit_0"], sm["fix"]["logits_match_c2"]), (3, 3))
        rows = [json.loads(line) for line in (out / "exit_base.jsonl").read_text().splitlines()]
        self.assertTrue(all(x["written"] and x["normal_exit_reached"] and x["matches_c2"] for x in rows))

    def test_base_that_never_crashes_is_recorded_not_failed(self):
        rc, s, _, out, _ = self.run_vm()
        self.assertEqual(rc, 0)
        self.assertEqual(self.summary(out)["base"]["crashed"], 0)

    def test_fix_that_crashes_fails(self):
        rc, s, _, out, _ = self.run_vm(extra_env={"STUB_CRASH": "both"})
        self.assertEqual((rc, s["gates"]["exit_fix"], s["pass"]), (1, 1, False))
        self.assertEqual(self.summary(out)["fix"]["crashed"], 3)

    def test_fix_with_other_logits_or_without_the_normal_exit_marker_fails(self):
        for run, env in (("o1", {"STUB_OTHER_LOGITS": "fix"}), ("o2", {"STUB_NO_NORMAL": "fix"})):
            rc, s, _, _, _ = self.run_vm(run=run, extra_env=env)
            self.assertEqual((rc, s["gates"]["exit_fix"]), (1, 1), run)

    def test_a_hung_dump_is_killed_and_counted(self):
        rc, s, _, out, _ = self.run_vm(extra_env={"STUB_HANG": "fix", "VM_TEST_REPS": "1", "VM_TEST_TIMEOUT": "2"})
        self.assertEqual((rc, s["gates"]["exit_fix"]), (1, 1))
        self.assertEqual(self.summary(out)["fix"]["exit_codes"], [137])

    def test_wrong_heads_or_dirty_trees_fail_checkout(self):
        for run, env in (("h1", {"STUB_HEAD": "f" * 40}), ("h2", {"STUB_BASE_HEAD": "e" * 40}),
                         ("h3", {"STUB_FIX_HEAD": "d" * 40}), ("h4", {"STUB_DIRTY": "cf_fix"})):
            rc, s, _, _, _ = self.run_vm(run=run, extra_env=env)
            self.assertEqual((rc, s["gates"]["checkout"]), (1, 99), run)

    def test_altered_input_stops_before_the_builds(self):
        rc, s, _, _, _ = self.run_vm(alter="ids.txt")
        self.assertEqual((rc, s["gates"]["inputs"]), (1, 1))
        self.assertNotIn("configure", s["gates"])

    def test_a_dump_that_runs_without_a_device_fails_the_control(self):
        rc, s, _, _, _ = self.run_vm(extra_env={"STUB_NODEVICE_RUNS": "1"})
        self.assertEqual((rc, s["gates"]["device_required"]), (1, 1))
        self.assertNotIn("exit_base", s["gates"])

    def test_build_failure_stops(self):
        rc, s, _, _, _ = self.run_vm(extra_env={"STUB_BUILD_FAIL": "1"})
        self.assertEqual((rc, s["gates"]["build"]), (1, 2))

    def test_provenance_needs_exactly_one_l4(self):
        rc, s, done, _, _ = self.run_vm(extra_env={"STUB_GPUS": "2"})
        self.assertEqual((rc, done, s["gates"]), (1, "r1", {"provenance": 1}))

    def test_second_launch_is_refused_and_bad_arguments_too(self):
        rc, _, _, out, _ = self.run_vm()
        before = (out / "status.json").read_text()
        rc2, _, _, _, r = self.run_vm(write=False)
        self.assertEqual(rc2, 2)
        self.assertIn("refusing to reuse", r.stderr)
        self.assertEqual((out / "status.json").read_text(), before)
        for args in ([COMMIT, "r9", "extra"], ["abc", "r9"]):
            self.assertEqual(self.run_vm(args=args, write=False)[0], 2, args)
        self.assertFalse((self.tmp / "content" / "run_r9").exists())

    def test_stop_during_a_step_stops_its_group_and_publishes(self):
        self.write_inputs("s1")
        env = dict(os.environ, VM_TEST_HARNESS="1", VM_ROOT=str(self.tmp / "content"),
                   PATH=f"{self.bin}:{os.environ['PATH']}", STUB_PACKET=str(self.packet), STUB_CORPUS=CORPUS,
                   VM_TEST_SIGNAL_AT="run_configure")
        t0 = time.monotonic()
        r = subprocess.run(["bash", str(HERE / "vm_exit.sh"), COMMIT, "s1"], env=env, capture_output=True, text=True,
                           timeout=120)
        out = self.tmp / "content" / "run_s1" / "out"
        s = json.loads((out / "status.json").read_text())
        stop = json.loads((out / "stop.json").read_text())
        self.assertEqual((r.returncode, s["gates"]["configure"]), (1, 143))
        self.assertEqual((stop["step"], stop["step_group_gone"]), ("configure", True))
        self.assertEqual((out / "DONE").read_text().strip(), "s1")
        self.assertLess(time.monotonic() - t0, 60)


if __name__ == "__main__":
    unittest.main()
