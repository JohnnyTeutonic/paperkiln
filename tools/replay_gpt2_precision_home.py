"""Required pinned host Ember/NumPy replay after the bounded CUDA model gate lands."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import replay_gpt2_precision_gate as model_gate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "spec/tools"))
import ir_check  # noqa: E402
import ir_numpy  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_ember_rows(actual, expected):
    """Reject broadcasting and nonfinite values before any reduction or comparison."""
    require = model_gate.require
    expected = np.asarray(expected, dtype=np.float64)
    require(
        expected.ndim == 2 and expected.size > 0 and np.isfinite(expected).all(),
        "invalid expected inference rows",
    )
    errors = {}
    for mode in ("prefill", "decode"):
        rows = np.asarray(actual[mode], dtype=np.float64)
        require(rows.shape == expected.shape, "host Ember " + mode + " shape differs")
        require(np.isfinite(rows).all(), "host Ember " + mode + " has nonfinite values")
        error = float(np.max(np.abs(rows - expected)))
        require(
            np.isfinite(error) and error <= 3e-5,
            "host Ember " + mode + " FP32 inference bound",
        )
        errors[mode] = error
    return errors


def replay(home, manifest_sha, ember, ember_sha):
    require = model_gate.require
    require(sha(ember) == ember_sha, "pinned host Ember binary changed")
    require(sha(home / "inputs.json") == manifest_sha, "frozen VM manifest changed")
    summary = json.loads((home / "gate_summary.json").read_text())
    require(
        summary["pass"] is True and summary["manifest_sha256"] == manifest_sha,
        "VM gate did not pass",
    )
    require(
        (home / "hardware.log").read_text().strip().startswith("NVIDIA L4,"),
        "actual L4 evidence missing",
    )
    for name, expected in summary["artifacts"].items():
        rel = Path(name)
        target = home / rel
        require(
            not rel.is_absolute()
            and ".." not in rel.parts
            and target.resolve().is_relative_to(home.resolve()),
            "artifact path escapes home",
        )
        require(sha(target) == expected, "landed artifact hash differs: " + name)

    def load(p):
        return json.loads(p.read_text())

    cpu, cuda = load(home / "cpu_model.json"), load(home / "cuda_model.json")
    model_replays = {
        label: model_gate.replay(cpu, cuda, load(home / (label + "_inference.json")))
        for label in ("lm", "sft")
    }
    inference = {}
    for label, directory, rowfile in [
        ("lm", home / "cli/bf16_cli", "trained_recipe.f32"),
        ("sft", home / "cli/sft_cli", "sft_recipe.f32"),
    ]:
        gguf = directory / "run/precision.gguf"
        ir = directory / "run/precision.ir.json"
        meta, tensors, offset, raw, _ = ir_check.read_gguf(gguf)
        require(
            meta["paperkiln.train.attention_precision"][1] == "bf16",
            "training recipe provenance missing",
        )
        doc = json.loads(meta["paperkiln.ir"][1])
        require(json.loads(ir.read_text()) == doc, "external IR differs")
        weights = {
            name: np.frombuffer(
                raw, dtype="<f4", count=int(np.prod(dims)), offset=offset + location
            ).reshape(tuple(reversed(dims)))
            for name, dims, kind, location in tensors
            if kind == 0
        }
        expected = ir_numpy.forward(doc, weights, [1, 2, 3])
        rows = np.fromfile(directory / rowfile, dtype="<f4").reshape(expected.shape)
        require(np.isfinite(rows).all(), "nonfinite CLI inference")
        cli_error = float(np.max(np.abs(rows - expected)))
        require(cli_error <= 3e-5, "landed CLI FP32 inference vs NumPy bound")
        profile_rows = model_gate.matrix(
            load(home / (label + "_inference.json"))["logits"]
        )
        require(
            profile_rows.shape == rows.shape
            and float(np.max(np.abs(profile_rows - rows))) <= 3e-5,
            "profile probe and CLI scores differ",
        )
        completed = subprocess.run(
            [str(ember), str(gguf)],
            input=json.dumps([[1, 2, 3]]),
            capture_output=True,
            text=True,
            timeout=90,
            env={
                **os.environ,
                "EMBER_IR": str(ir),
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            },
        )
        require(completed.returncode == 0, "pinned CPU Ember replay failed")
        actual = json.loads(completed.stdout)[0]
        errors = validate_ember_rows(actual, expected)
        inference[label] = {
            "gguf_sha256": sha(gguf),
            "ir_sha256": sha(ir),
            "cli_numpy_max_abs": cli_error,
            "ember_numpy_max_abs": errors,
            "raw_ember_rows": actual,
            "ember_stderr": completed.stderr,
        }
    return {
        "pass": True,
        "actual_cuda_model_gate": True,
        "required_host_replay_complete": True,
        "model_replays": model_replays,
        "inference": inference,
        "ember_sha256": ember_sha,
        "manifest_sha256": manifest_sha,
        "quality_or_speed_claim": False,
        "provider_calls": 0,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--home", type=Path, required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--ember", type=Path, required=True)
    p.add_argument("--ember-sha256", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    result = replay(a.home, a.manifest_sha256, a.ember, a.ember_sha256)
    with a.out.open("x") as f:
        json.dump(result, f, indent=1)
        f.write("\n")
    print(
        json.dumps(
            {
                "pass": True,
                "inference": {
                    k: {
                        n: v
                        for n, v in r.items()
                        if n not in ("raw_ember_rows", "ember_stderr")
                    }
                    for k, r in result["inference"].items()
                },
            },
            indent=1,
        )
    )
