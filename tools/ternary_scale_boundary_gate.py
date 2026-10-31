"""UNLAUNCHED prospective public ternary_scale old/fixed boundary gate."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import tarfile
import time
import zipfile
from pathlib import Path


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(upload, manifest, out):
    upload, out = Path(upload), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("immutable output already populated")
    root = Path("/content/ternary_boundary_source")
    build = Path("/content/ternary_boundary_build")
    deadline = time.monotonic() + 2400
    report = {
        "pass": False,
        "errors": [],
        "commands": [],
        "scope": "public ternary_scale direct helper only; full-object old/fixed CPU and actual CUDA; no QAT/model/speed claim",
    }
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MICROTORCH_DEVICE="cuda",
        MICROTORCH_CUBLAS="0",
        MICROTORCH_TF32="0",
    )

    def command(name, args, seconds, expected_rc=0):
        remain = deadline - time.monotonic()
        if remain <= 0:
            raise ValueError("fixed gate deadline expired")
        with (out / (name + ".log")).open("w") as log:
            p = subprocess.Popen(
                list(map(str, args)),
                cwd=root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                rc = p.wait(timeout=min(seconds, remain))
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
                raise ValueError(name + " timed out") from None
        report["commands"].append(
            {"name": name, "argv": list(map(str, args)), "returncode": rc}
        )
        if rc != expected_rc:
            raise ValueError(name + " failed")

    try:
        if sha(upload / "inputs.json") != manifest:
            raise ValueError("manifest differs")
        m = json.loads((upload / "inputs.json").read_text())
        report["inputs"] = m
        for n, h in m["files"].items():
            if Path(n).name != n or sha(upload / n) != h:
                raise ValueError("uploaded file drift")
        if sha(__file__) != m["files"]["ternary_scale_boundary_gate.py"]:
            raise ValueError("executing wrapper drift")
        root.mkdir(exist_ok=True)
        if any(root.iterdir()):
            raise ValueError("source root not empty")
        with tarfile.open(upload / "source.tgz") as t:
            for x in t.getmembers():
                if (
                    x.name.startswith("/")
                    or ".." in Path(x.name).parts
                    or not (x.isfile() or x.isdir())
                ):
                    raise ValueError("unsafe archive")
            t.extractall(root)
        for n, h in m["source_files"].items():
            if sha(root / n) != h:
                raise ValueError("source drift")
        command(
            "hardware",
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            30,
        )
        hw = (out / "hardware.log").read_text().strip().splitlines()
        if len(hw) != 1 or not hw[0].startswith("NVIDIA L4,"):
            raise ValueError("single actual L4 required")
        fixed_ops = (root / "src/ops.cpp").read_bytes()
        (root / "src/ops.cpp").write_bytes((upload / "old_ops.cpp").read_bytes())
        command(
            "configure",
            [
                "cmake",
                "-S",
                root,
                "-B",
                build,
                "-DCMAKE_BUILD_TYPE=Release",
                "-DMICROTORCH_CUDA=ON",
                "-DMICROTORCH_DEVCHECK=ON",
                "-DMICROTORCH_BUILD_PYTHON=OFF",
                "-DCMAKE_CUDA_ARCHITECTURES=89",
            ],
            180,
        )
        import sys

        sys.path.insert(0, str(root / "tools"))
        from replay_ternary_scale_boundary import replay

        report["replays"] = []
        for mode in ["old", "fixed"]:
            if mode == "fixed":
                (root / "src/ops.cpp").write_bytes(fixed_ops)
            command(
                mode + "_build",
                [
                    "cmake",
                    "--build",
                    build,
                    "--target",
                    "test_ternary_scale_boundary",
                    "--parallel",
                    "4",
                ],
                1000,
            )
            for cuda in [False, True]:
                name = mode + ("_cuda" if cuda else "_cpu")
                args = [build / "test_ternary_scale_boundary"] + (
                    ["--cuda"] if cuda else []
                )
                expected_rc = 1 if mode == "old" and cuda else 0
                command(name, args, 60, expected_rc)
                report["replays"].append(
                    replay(out / (name + ".log"), mode, cuda, expected_rc)
                )
        report["pass"] = True
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
    ) as error:
        report["errors"].append(str(error))
    finally:
        report["artifact_sha256"] = {
            str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file()
        }
        (out / "gate_summary.json").write_text(json.dumps(report, indent=2) + "\n")
        with zipfile.ZipFile(
            out / "out.zip", "x", compression=zipfile.ZIP_DEFLATED
        ) as z:
            for p in out.rglob("*"):
                if p.is_file() and p.name != "out.zip":
                    z.write(p, str(p.relative_to(out)))
    print(json.dumps({"pass": report["pass"], "errors": report["errors"]}))
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--upload", required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    raise SystemExit(run(a.upload, a.manifest_sha256, a.out))
