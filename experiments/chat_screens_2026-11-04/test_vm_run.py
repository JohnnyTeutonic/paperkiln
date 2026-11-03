"""Tests of the screening vm_run.sh with stand-ins for git's remote, cmake and mtstudio (no GPU, no network).
python3 -m unittest test_vm_run -v

The source is a fixture git repository whose HEAD stands in for the numerical source (VM_TEST_SOURCE). The fake cmake
makes build/mtstudio; the fake mtstudio refuses to run without arm A's backend environment, prints 256 step events
and writes one GGUF named after the spec. FAKE_* variables make it fail, stop short or run slowly.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
COMMIT = "0123456789abcdef0123456789abcdef01234567"
SCREENS = ["S0", "S1", "S2", "S3"]
GATES = sorted(["provenance", "checkout", "inputs", "configure", "build", "specs"]
               + [f"{g}_{s}" for s in SCREENS for g in ("train", "package")])

FAKE_CMAKE = r'''#!/usr/bin/env python3
import os, sys
from pathlib import Path
a = sys.argv[1:]
if os.environ.get("FAKE_CMAKE_FAIL"):
    sys.exit(3)
if "-S" in a:
    Path(a[a.index("-B") + 1]).mkdir(parents=True, exist_ok=True); sys.exit(0)
if "--build" in a:
    b = Path(a[a.index("--build") + 1]); b.mkdir(parents=True, exist_ok=True)
    m = b / "mtstudio"; m.write_text("#!/bin/sh\nexit 0\n"); m.chmod(0o755); sys.exit(0)
sys.exit(2)
'''

FAKE_MTSTUDIO = r'''#!/usr/bin/env python3
import json, os, sys, time
from pathlib import Path
need = {"MICROTORCH_DEVICE": "cuda", "MICROTORCH_DEVICE_OPS": "1", "MICROTORCH_ATTN_BLOCKED": "1",
        "MICROTORCH_STEP_RESIDENCY": "1", "MICROTORCH_DEFER_DOWNLOADS": "1", "MICROTORCH_CUBLAS": "1",
        "OMP_NUM_THREADS": "4"}
if any(os.environ.get(k) != v for k, v in need.items()):
    print("backend environment missing"); sys.exit(4)
assert sys.argv[1] == "run"
spec = json.load(open(sys.argv[2]))
screen = spec["name"].split("_")[-1]
for p in (spec["data"]["vocab"], spec["data"]["sft_examples"], spec["train"]["init_from"]):
    assert os.path.isfile(p), p
if os.environ.get("FAKE_SLOW") == screen:
    time.sleep(30)
if os.environ.get("FAKE_FAIL") == screen:
    print("trainer error"); sys.exit(1)
last = 200 if os.environ.get("FAKE_SHORT") == screen else 256
for s in range(1, last + 1):
    print(json.dumps({"event": "step", "step": s, "loss": 3.0 - s / 1000}))
out = Path(spec["out_dir"]); out.mkdir(parents=True, exist_ok=True)
(out / (spec["name"] + ".gguf")).write_bytes((screen * 1500).encode())
(out / "result.json").write_text(json.dumps({"final_step": last}))
'''


def sha(b):
    return hashlib.sha256(b).hexdigest()


class VmRunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        repo = self.tmp / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        (repo / "README").write_text("fixture source\n")
        subprocess.run(["git", "-C", str(repo), "add", "README"], check=True)
        subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m",
                        "fixture"], check=True)
        self.source = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                                     check=True).stdout.strip()
        self.repo = repo
        for name, body in (("cmake", FAKE_CMAKE), ("mtstudio", FAKE_MTSTUDIO)):
            p = self.tmp / name
            p.write_text(body)
            p.chmod(0o755)
        self.root = self.tmp / "root"
        self.inputs = self.root / "screens_in"
        self.inputs.mkdir(parents=True)
        base = b"weights" * 700
        parts = []
        for i in range(0, len(base), 2000):
            n = f"base.part{i // 2000:02d}"
            (self.inputs / n).write_bytes(base[i:i + 2000])
            parts.append({"name": n, "sha256": sha(base[i:i + 2000])})
        files = {"sfull_sft1500.safetensors": sha(base)}
        for n in ["vocab.gguf"] + [f"{s}.jsonl" for s in SCREENS + ["S4"]]:
            data = (n * 5).encode()
            (self.inputs / n).write_bytes(data)
            files[n] = sha(data)
        spec = (HERE / "arm_a_lr3e5.json").read_bytes()
        (self.inputs / "arm_a_lr3e5.json").write_bytes(spec)
        files["arm_a_lr3e5.json"] = sha(spec)
        self.manifest = {"files": files, "base_parts": parts, "arm_a_spec_sha256": sha(spec), "source": self.source,
                         "run_id": "v1", "seed": 1, "screens": SCREENS}
        self.write_manifest()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write_manifest(self):
        (self.inputs / "manifest.json").write_text(json.dumps(self.manifest, indent=2, sort_keys=True))
        self.manifest_sha = sha((self.inputs / "manifest.json").read_bytes())

    def run_vm(self, run="v1", manifest_sha=None, extra=None, args=None, seed_screens=None):
        env = dict(os.environ, VM_TEST_HARNESS="1", VM_ROOT=str(self.root), VM_TEST_SOURCE=self.source,
                   VM_TEST_CLONE_URL=str(self.repo), VM_TEST_CMAKE=str(self.tmp / "cmake"),
                   VM_TEST_MTSTUDIO=str(self.tmp / "mtstudio"), VM_TEST_PROVENANCE="0")
        env.update(extra or {})
        argv = ["bash", str(HERE / "vm_run.sh")] + (args if args is not None else
                                                   [COMMIT, run, manifest_sha or self.manifest_sha]
                                                   + (seed_screens or []))
        r = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=120)
        out = self.root / f"run_{run}" / "out"
        status = json.loads((out / "status.json").read_text()) if (out / "status.json").exists() else None
        return r, status, out

    def ready(self, out):
        return [s for s in SCREENS if (out / "screens" / s / "READY").exists()]

    def test_normal_run_trains_and_packages_every_screen(self):
        r, s, out = self.run_vm()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(s["pass"])
        self.assertEqual(sorted(s["gates"]), GATES)
        self.assertTrue(all(v == 0 for v in s["gates"].values()))
        self.assertEqual((s["commit"], s["source"], s["head"], s["manifest_sha256"]),
                         (COMMIT, self.source, self.source, self.manifest_sha))
        self.assertEqual(s["packaged"], SCREENS)
        self.assertEqual((out / "DONE").read_text().strip(), "v1")
        for scr in SCREENS:
            d = out / "screens" / scr
            man = json.loads((d / "manifest.json").read_text())
            joined = b"".join((d / p["name"]).read_bytes() for p in man["parts"])
            self.assertEqual(sha(joined), man["gguf_sha256"])
            self.assertEqual(joined, (scr * 1500).encode())
            self.assertEqual((d / "READY").read_text().strip(), "v1")
        arm_a = json.loads((HERE / "arm_a_lr3e5.json").read_text())
        spec = json.loads((out / "spec_S2.json").read_text())
        self.assertEqual(spec["train"]["lr"], arm_a["train"]["lr"])
        self.assertEqual(spec["train"]["steps"], 256)
        self.assertTrue(spec["data"]["sft_examples"].endswith("/upload/S2.jsonl"))

    def test_a_wrong_manifest_hash_stops_before_the_build(self):
        r, s, out = self.run_vm(manifest_sha="0" * 64)
        self.assertNotEqual(r.returncode, 0)
        self.assertNotEqual(s["gates"]["inputs"], 0)
        self.assertNotIn("build", s["gates"])
        self.assertFalse(s["pass"])

    def test_a_tampered_input_fails_the_inputs_gate(self):
        (self.inputs / "S1.jsonl").write_bytes(b"tampered")
        r, s, out = self.run_vm()
        self.assertNotEqual(s["gates"]["inputs"], 0)
        self.assertNotIn("train_S0", s["gates"])

    def test_a_spec_pin_mismatch_fails_the_specs_gate(self):
        self.manifest["arm_a_spec_sha256"] = "1" * 64
        self.write_manifest()
        r, s, out = self.run_vm()
        self.assertEqual(s["gates"]["inputs"], 0)
        self.assertNotEqual(s["gates"]["specs"], 0)
        self.assertNotIn("train_S0", s["gates"])

    def test_a_failed_screen_stops_the_session_and_keeps_earlier_screens(self):
        r, s, out = self.run_vm(extra={"FAKE_FAIL": "S2"})
        self.assertEqual(r.returncode, 1)
        self.assertNotEqual(s["gates"]["train_S2"], 0)
        self.assertNotIn("train_S3", s["gates"])
        self.assertEqual(self.ready(out), ["S0", "S1"])
        self.assertEqual(s["packaged"], ["S0", "S1"])
        self.assertTrue((out / "DONE").exists())

    def test_a_run_that_stops_short_of_256_fails_its_check(self):
        r, s, out = self.run_vm(extra={"FAKE_SHORT": "S1"})
        self.assertEqual(s["gates"]["train_S1"], 96)
        self.assertEqual(self.ready(out), ["S0"])
        check = json.loads((out / "train_S1_check.json").read_text())
        self.assertEqual(check["last_step"], 200)

    def test_a_stop_during_a_screen_is_recorded_and_published(self):
        r, s, out = self.run_vm(extra={"FAKE_SLOW": "S1", "VM_TEST_SIGNAL_AT": "run_train_S1"})
        self.assertEqual(r.returncode, 1)
        self.assertEqual(s["gates"]["train_S1"], 143)
        stop = json.loads((out / "stop.json").read_text())
        self.assertEqual(stop["step"], "train_S1")
        self.assertTrue(stop["step_group_gone"])
        self.assertEqual(self.ready(out), ["S0"])
        self.assertTrue((out / "DONE").exists())
        self.assertTrue((self.root / "vm_v1.pid").exists())

    def test_inherited_backend_variables_are_cleared_and_recorded(self):
        r, s, out = self.run_vm(extra={"MICROTORCH_STRAY": "1", "MICROTORCH_DEVICE": "cpu"})
        self.assertEqual(r.returncode, 0, r.stderr)
        env = (out / "env_S0.txt").read_text()
        self.assertNotIn("MICROTORCH_STRAY", env)
        self.assertIn("MICROTORCH_DEVICE=cuda", env)

    def test_screen_manifests_hash_the_log_and_result(self):
        r, s, out = self.run_vm()
        man = json.loads((out / "screens" / "S1" / "manifest.json").read_text())
        for name in ("train.log", "result.json"):
            self.assertEqual(man[name.replace(".", "_") + "_sha256"],
                             sha((out / "screens" / "S1" / name).read_bytes()))
        self.assertTrue((out / "result_S1.json").is_file())

    def test_a_wave_2_session_applies_its_seed_to_every_spec(self):
        self.manifest.update(seed=2, screens=["S0", "S2", "S4"])
        self.write_manifest()
        r, s, out = self.run_vm(seed_screens=["2", "S0,S2,S4"])
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual((s["seed"], s["screens"], s["packaged"]), (2, ["S0", "S2", "S4"], ["S0", "S2", "S4"]))
        self.assertIn("train_S4", s["gates"])
        self.assertNotIn("train_S1", s["gates"])
        for scr in ("S0", "S2", "S4"):
            spec = json.loads((out / f"spec_{scr}.json").read_text())
            self.assertEqual(spec["train"]["seed"], 2)
            self.assertEqual(spec["name"], f"screen_{scr}")

    def test_a_seed_that_differs_from_the_manifest_fails_the_specs_gate(self):
        r, s, out = self.run_vm(seed_screens=["3", "S0,S1,S2,S3"])
        self.assertNotEqual(s["gates"]["specs"], 0)
        self.assertNotIn("train_S0", s["gates"])

    def test_a_second_launch_of_the_same_run_is_refused(self):
        self.run_vm()
        r, _, _ = self.run_vm()
        self.assertEqual(r.returncode, 2)
        self.assertIn("refusing to reuse a run directory", r.stderr)

    def test_bad_arguments_are_refused(self):
        for args in ([COMMIT[:39], "v1", "0" * 64], [COMMIT, "bad run", "0" * 64], [COMMIT, "v1", "xyz"],
                     [COMMIT, "v1", "0" * 64, "0", "S0"], [COMMIT, "v1", "0" * 64, "2", "S0;S1"]):
            r, _, _ = self.run_vm(args=args)
            self.assertEqual(r.returncode, 2, args)
        self.assertFalse((self.root / "run_v1").exists())


if __name__ == "__main__":
    unittest.main()
