"""One-shot source-bound L4 validation of explicit transpose GEMM loads, NT-only prospective adoption criterion."""

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


def run(upload, manifest_sha, out):
    upload = Path(upload)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Immutable evidence destination is nonempty")
    root = Path("/content/gemm_transpose_source")
    build = Path("/content/gemm_transpose_build")
    report = {
        "pass": False,
        "scope": "NT-only GEMM load change; bitwise comparison and prospective kernel-only criterion; no model speed claim",
        "commands": [],
        "errors": [],
    }
    deadline = time.monotonic() + 1500
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MICROTORCH_GPU_PROFILE": "1",
        "MICROTORCH_DEVICE": "cuda",
    }
    logs = []

    def command(name, argv, seconds=300):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError("Gate deadline expired")
        log = out / (name + ".log")
        logs.append(log)
        with log.open("w") as f:
            p = subprocess.Popen(
                list(map(str, argv)),
                cwd=root,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                rc = p.wait(timeout=min(seconds, remaining))
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
                raise ValueError(name + " timeout") from None
        report["commands"].append(
            {"name": name, "argv": list(map(str, argv)), "returncode": rc}
        )
        if rc:
            raise ValueError(name + " failed with exit " + str(rc))

    try:
        if sha(upload / "inputs.json") != manifest_sha:
            raise ValueError("Manifest pin differs")
        m = json.loads((upload / "inputs.json").read_text())
        report["inputs"] = m
        for f, h in m["files"].items():
            if Path(f).name != f or sha(upload / f) != h:
                raise ValueError("Input hash differs: " + f)
        if sha(__file__) != m["files"]["gemm_transpose_gate.py"]:
            raise ValueError("Executing wrapper differs")
        root.mkdir(parents=True, exist_ok=True)
        if any(root.iterdir()):
            raise ValueError("Nonempty source destination")
        with tarfile.open(upload / "source.tgz") as t:
            for member in t.getmembers():
                if (
                    member.name.startswith("/")
                    or ".." in Path(member.name).parts
                    or not (member.isfile() or member.isdir())
                ):
                    raise ValueError("Unsafe source archive")
            t.extractall(root)
        for f, h in m["source_files"].items():
            if sha(root / f) != h:
                raise ValueError("Extracted source differs: " + f)
        command(
            "hardware",
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            30,
        )
        hw = (out / "hardware.log").read_text().strip().splitlines()
        if len(hw) != 1 or not hw[0].startswith("NVIDIA L4,"):
            raise ValueError("Requires actual single L4")
        command("nvcc", ["nvcc", "--version"], 30)
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
        command(
            "extract",
            ["python3", root / "tools/prepare_gemm_transpose.py", "--out", build],
            30,
        )
        command(
            "build",
            [
                "nvcc",
                "-std=c++17",
                "-O3",
                "-arch=sm_89",
                "-DMICROTORCH_DEVCHECK",
                "-I" + str(build),
                "-I" + str(root / "tests"),
                root / "tests/test_gemm_transpose_loads.cu",
                "-o",
                build / "test_gemm_transpose_cuda",
            ],
            240,
        )
        command("strict", [build / "test_gemm_transpose_cuda"], 120)
        command("timing", [build / "test_gemm_transpose_cuda", "--timing"], 120)
        report["binary_sha256"] = sha(build / "test_gemm_transpose_cuda")
        command(
            "numpy_install", ["python3", "-m", "pip", "install", "numpy==1.26.4"], 90
        )
        command(
            "independent_replay",
            [
                "python3",
                "-O",
                upload / "replay_gemm_transpose.py",
                out / "strict.log",
                "--timing",
                out / "timing.log",
                "--adoption",
            ],
            60,
        )
        result = json.loads((out / "independent_replay.log").read_text())
        if result.get("pass") is not True or result.get("cases") != 96:
            raise ValueError("Missing strict replay proof")
        if result.get("adoption", {}).get("pass") is not True:
            raise ValueError("Missing prospective adoption proof")
        report["independent_replay"] = result
        report["pass"] = True
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as e:
        report["errors"].append(str(e))
    finally:
        report["logs"] = {
            p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in logs
        }
        (out / "gate_summary.json").write_text(json.dumps(report, indent=2) + "\n")
        with zipfile.ZipFile(
            out / "out.zip", "x", compression=zipfile.ZIP_DEFLATED
        ) as z:
            for p in out.iterdir():
                if p.is_file() and p.name != "out.zip":
                    z.write(p, p.name)
    print(json.dumps({"pass": report["pass"], "errors": report["errors"]}))
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upload", required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    raise SystemExit(run(a.upload, a.manifest_sha256, a.out))
