"""One-shot source-bound L4 validation of explicit composed Mamba1, not adoption timing."""

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
    root = Path("/content/mamba1_cli_source")
    build = Path("/content/mamba1_cli_build")
    report = {
        "pass": False,
        "scope": "tiny actual CLI Mamba1 LM training/save/reload/F32 export; frozen host NumPy/Ember replay required after home; no quality/speed/full residency claim",
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
        "MICROTORCH_ATTN_BLOCKED": "0",
        "MICROTORCH_CUBLAS": "0",
        "MICROTORCH_TF32": "0",
        "MAMBA1_CLI_SOURCE_ROOT": str(root),
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
        if sha(__file__) != m["files"]["mamba1_cli_gate.py"]:
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
            "build",
            [
                "cmake",
                "--build",
                build,
                "--target",
                "mtstudio",
                "--parallel",
                "4",
            ],
            900,
        )
        report["binary_sha256"] = sha(build / "mtstudio")
        command(
            "numpy_install", ["python3", "-m", "pip", "install", "numpy==1.26.4"], 90
        )
        command(
            "lifecycle",
            [
                "python3",
                upload / "mamba1_cli_lifecycle.py",
                "produce",
                "--mtstudio",
                build / "mtstudio",
                "--out",
                out / "runs",
                "--require-cuda",
            ],
            300,
        )
        result = json.loads((out / "runs/producer.json").read_text())
        if (
            result.get("remote_or_CPU_producer_pass") is not True
            or result.get("actual_cli_cuda") is not True
            or len(result.get("cases", [])) != 2
        ):
            raise ValueError("Missing actual CLI lifecycle proof")
        report["producer"] = result
        report["host_independent_NumPy_Ember_replay_required"] = True
        report["artifact_sha256"] = {
            str(p.relative_to(out)): sha(p)
            for p in (out / "runs").rglob("*")
            if p.is_file()
        }
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
            for p in out.rglob("*"):
                if p.is_file() and p.name != "out.zip":
                    z.write(p, str(p.relative_to(out)))
    print(json.dumps({"pass": report["pass"], "errors": report["errors"]}))
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upload", required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    raise SystemExit(run(a.upload, a.manifest_sha256, a.out))
