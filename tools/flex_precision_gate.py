"""UNLAUNCHED four-case plain Flex model BF16 correctness gate. No CLI CUDA claim."""

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


def validate_manifest(m):
    if m.get("schema") != "plain-flex-bf16-l4-v1":
        raise ValueError("manifest schema")
    if m.get("science_commit") != "3db619690a0f08051224fa174484a2a22dc1f98a":
        raise ValueError("science commit")
    if m.get("limits") != {
        "max_usd": 1,
        "max_hours": 1,
        "gate_seconds": 1500,
        "gpu": "NVIDIA L4",
        "gpus": 1,
    }:
        raise ValueError("registered limits")
    if m.get("bounds") != {
        "normalized_element": 0.0003,
        "tensor_relative_norm": 0.03,
        "norm_floor": 1e-10,
    }:
        raise ValueError("registered bounds")
    if set(m.get("files", {})) != {
        "source.tgz",
        "flex_precision_gate.py",
        "protocol.json",
        "cpu_fixture.json",
    }:
        raise ValueError("uploaded inventory")
    required = {
        "CMakeLists.txt",
        "studio/chat.html",
        "tools/parity_model.hpp",
        "tools/mtstudio.cpp",
        "tests/flex_precision_gate.cpp",
        "tools/replay_flex_precision.py",
        "tools/flex_precision_gate.py",
    }
    if not required <= set(m.get("source_files", {})):
        raise ValueError("critical source inventory")
    for name, digest in list(m["source_files"].items()) + list(m["files"].items()):
        if Path(name).is_absolute() or ".." in Path(name).parts or "\\" in name:
            raise ValueError("unsafe source name")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("invalid digest")


def run(upload, manifest, out):
    upload, out = Path(upload), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("immutable output already populated")
    root = Path("/content/flex_precision_source")
    build = Path("/content/flex_precision_build")
    deadline = time.monotonic() + 1500
    report = {
        "pass": False,
        "errors": [],
        "commands": [],
        "scope": "four tiny canonical plain Flex model cases; full gradients against explicit BF16 surrogate; no CUDA CLI lifecycle, quality, speed or adoption claim",
    }
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MICROTORCH_DEVICE="cuda",
        MICROTORCH_CUBLAS="0",
        MICROTORCH_TF32="0",
        MICROTORCH_GPU_PROFILE="1",
        MICROTORCH_DEVICE_OPS="1",
    )

    def command(name, args, seconds):
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
        if rc:
            raise ValueError(name + " failed")

    try:
        if sha(upload / "inputs.json") != manifest:
            raise ValueError("manifest differs")
        m = json.loads((upload / "inputs.json").read_text())
        report["inputs"] = m
        validate_manifest(m)
        for n, h in m["files"].items():
            if Path(n).name != n or sha(upload / n) != h:
                raise ValueError("uploaded file drift")
        if sha(__file__) != m["files"]["flex_precision_gate.py"]:
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
        actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
        if actual != set(m["source_files"]):
            raise ValueError("archive inventory differs")
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
                "flex_precision_gate",
                "--parallel",
                "2",
            ],
            900,
        )
        command("numpy", ["python3", "-m", "pip", "install", "numpy==1.26.4"], 90)
        for mode in ["cpu", "cuda"]:
            args = [build / "flex_precision_gate", "--out", out / (mode + ".json")]
            if mode == "cuda":
                args.append("--require-cuda")
            command(mode, args, 60)
            args = [
                "python3",
                "-O",
                root / "tools/replay_flex_precision.py",
                out / (mode + ".json"),
            ]
            if mode == "cuda":
                args.append("--require-cuda")
            command(mode + "_replay", args, 60)
        for n, h in m["source_files"].items():
            if sha(root / n) != h:
                raise ValueError("post-run source drift")
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
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--upload", required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    raise SystemExit(run(a.upload, a.manifest_sha256, a.out))
