"""One-shot source-bound L4 validation of train-only GPT2 attention model, not adoption timing."""

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
    root = Path("/content/gpt2_precision_source")
    build = Path("/content/gpt2_precision_build")
    report = {
        "pass": False,
        "scope": "experimental train-only GPT2; overall completion also requires host Ember replay; no speed or quality claim",
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
        "MICROTORCH_DEVICE": "cuda",
        "MICROTORCH_DEVICE_OPS": "1",
        "MICROTORCH_STEP_RESIDENCY": "1",
        "MICROTORCH_DEFER_DOWNLOADS": "1",
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
        report["manifest_sha256"] = manifest_sha
        (out / "inputs.json").write_bytes((upload / "inputs.json").read_bytes())
        for f, h in m["files"].items():
            if Path(f).name != f or sha(upload / f) != h:
                raise ValueError("Input hash differs: " + f)
        if sha(__file__) != m["files"]["gpt2_precision_gate.py"]:
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
                "gpt2_precision_gate",
                "--parallel",
                "4",
            ],
            900,
        )
        binary = build / "gpt2_precision_gate"
        report["binary_sha256"] = sha(binary)
        command(
            "packages",
            ["python3", "-m", "pip", "install", "numpy==1.26.4", "pytest==9.0.3"],
            120,
        )
        command(
            "cpu_model", [binary, "--gate-model", "cpu", out / "cpu_model.json"], 90
        )
        command(
            "cuda_model", [binary, "--gate-model", "cuda", out / "cuda_model.json"], 90
        )
        command(
            "model_replay",
            [
                "python3",
                "-O",
                root / "tools/replay_gpt2_precision_gate.py",
                "--cpu",
                out / "cpu_model.json",
                "--cuda",
                out / "cuda_model.json",
                "--out",
                out / "model_replay.json",
            ],
            60,
        )
        env["MTSTUDIO_BIN"] = str(binary)
        env["GPT2_PRECISION_PROOF_DIR"] = str(out / "cli")
        env.pop("EMBER_IR_LOGITS_BIN", None)
        command(
            "actual_cli",
            [
                "python3",
                "-m",
                "pytest",
                "-q",
                root / "tests/test_gpt2_attention_precision_cli.py",
                "-k",
                "training_reloads or sft_recipe",
                "--basetemp",
                out / "pytest",
            ],
            180,
        )
        for label, spec in (
            ("lm", out / "cli/bf16_cli/bf16.json"),
            ("sft", out / "cli/sft_cli/sft.json"),
        ):
            command(
                label + "_inference",
                [
                    binary,
                    "--gate-inference",
                    "cuda",
                    spec,
                    out / (label + "_inference.json"),
                ],
                60,
            )
            command(
                label + "_replay",
                [
                    "python3",
                    "-O",
                    root / "tools/replay_gpt2_precision_gate.py",
                    "--cpu",
                    out / "cpu_model.json",
                    "--cuda",
                    out / "cuda_model.json",
                    "--inference",
                    out / (label + "_inference.json"),
                    "--out",
                    out / (label + "_replay.json"),
                ],
                60,
            )
            saved = json.loads(spec.read_text())
            events = [
                json.loads(line)
                for line in (Path(saved["out_dir"]) / "events.jsonl")
                .read_text()
                .splitlines()
            ]
            steps = [e for e in events if e["event"] == "step"]
            if len(steps) != (6 if label == "lm" else 3):
                raise ValueError("CLI step count differs")
            for e in steps:
                counts = e.get("gpu_n", {})
                if (
                    counts.get("gemm_bf16", 0) <= 0
                    or counts.get("gemm", 0) + counts.get("gemm_head", 0) <= 0
                ):
                    raise ValueError(
                        "CLI did not prove BF16 attention and FP32 projection GEMMs"
                    )
            report[label + "_cli_training_counts"] = [e["gpu_n"] for e in steps]
        report["overall_gate_status"] = (
            "VM checks passed; REQUIRED pinned host Ember+NumPy replay still pending"
        )
        report["pass"] = True
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as e:
        report["errors"].append(str(e))
    finally:
        report["logs"] = {
            p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in logs
        }
        report["artifacts"] = {
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upload", required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    raise SystemExit(run(a.upload, a.manifest_sha256, a.out))
