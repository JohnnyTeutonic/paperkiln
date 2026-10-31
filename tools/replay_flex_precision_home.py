"""Post-home FP32 exported-model replay against independent NumPy and pinned CPU Ember."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "spec/tools"))
import ir_check  # noqa: E402
import ir_numpy  # noqa: E402

EMBER_SHA = "1f877524c3b59e1cdbf3ca29ab6d4e0ed216900516e1bcb28d5ec7e7e21b9a0f"
SEQUENCES = [[1], [1, 2, 3], [6, 5, 4, 3]]
BOUND = 1e-5


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(actual, expected):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    need(a.shape == b.shape and a.shape == (8, 7), "complete registered score shape")
    need(np.isfinite(a).all() and np.isfinite(b).all(), "finite full rows")
    error = float(np.max(abs(a - b) / (1 + abs(b))))
    need(
        error <= BOUND and np.array_equal(a.argmax(1), b.argmax(1)),
        "full-row normalized bound/argmax",
    )
    return error


def replay(artifact, ember, output):
    artifact, ember, output = Path(artifact), Path(ember), Path(output)
    need(not output.exists(), "immutable host output exists")
    need(sha(ember) == EMBER_SHA, "pinned CPU Ember binary identity")
    need(np.__version__ == "1.26.4", "frozen host NumPy identity")
    pins = {
        str(ROOT / "spec/tools" / n): sha(ROOT / "spec/tools" / n)
        for n in ["ir_check.py", "ir_numpy.py"]
    }
    output.mkdir(parents=True)
    cases = []
    for objective, depth, hidden in [("lm", 1, 12), ("sft", 3, 20)]:
        source = artifact / objective
        recipe = json.loads((source / "bf16.json").read_text())
        need(
            recipe["train"]
            == {
                "steps": 3,
                "batch": 2,
                "seed": 37,
                "lr": 0.001,
                "eval_every": 3,
                "checkpoint_every": 3,
                "eval_windows": 2,
                "gradmap_every": 0,
                "attention_precision": "bf16",
                "objective": objective,
            },
            "exact registered training recipe",
        )
        need(recipe["data"]["T"] == 8, "registered context length")
        need(
            all(
                type(recipe["train"][k]) is int
                for k in [
                    "steps",
                    "batch",
                    "seed",
                    "eval_every",
                    "checkpoint_every",
                    "eval_windows",
                    "gradmap_every",
                ]
            ),
            "integer registered training counters",
        )
        custom = recipe["arch"]["custom"]
        need(
            (
                custom["d"],
                custom["heads"],
                custom["layers"],
                custom["d_ff"],
                custom["norm"],
            )
            == (8, 2, depth, hidden, "layernorm"),
            "registered recipe geometry",
        )
        need(
            recipe["train"]["attention_precision"] == "bf16"
            and recipe["train"]["objective"] == objective,
            "registered training recipe",
        )
        trained = source / "trained"
        gguf = trained / "flex_precision.gguf"
        gguf_before = sha(gguf)
        meta, tensors, offset, raw, _ = ir_check.read_gguf(gguf)
        need(
            meta["paperkiln.train.attention_precision"][1] == "bf16",
            "training provenance metadata",
        )
        doc = json.loads(meta["paperkiln.ir"][1])
        need(
            doc["model"]
            == {"vocab": 7, "d": 8, "n_layers": depth, "ctx": 8, "dtype": "f32"},
            "registered exported model geometry",
        )
        need(
            json.loads((trained / "flex_precision.ir.json").read_text()) == doc,
            "IR sidecar identity",
        )
        need(all(kind == 0 for _, _, kind, _ in tensors), "F32 inference tensors only")
        weights = {
            name: np.frombuffer(
                raw, dtype="<f4", count=int(np.prod(dims)), offset=offset + location
            ).reshape(tuple(reversed(dims)))
            for name, dims, _, location in tensors
        }
        expected = np.concatenate(
            [ir_numpy.forward(doc, weights, ids) for ids in SEQUENCES]
        )
        actual = np.fromfile(source / "bf16_recipe.f32", dtype="<f4").reshape(8, 7)
        cli_error = compare(actual, expected)
        env = {
            **os.environ,
            "EMBER_DEVICE": "cpu",
            "EMBER_IR": str(trained / "flex_precision.ir.json"),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
        p = subprocess.run(
            [str(ember), str(gguf)],
            input=json.dumps(SEQUENCES),
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )
        (output / (objective + ".ember.stdout")).write_text(p.stdout)
        (output / (objective + ".ember.stderr")).write_text(p.stderr)
        need(p.returncode == 0, "CPU Ember execution")
        rows = json.loads(p.stdout)
        need(len(rows) == len(SEQUENCES), "complete Ember cases")
        errors = {
            mode: compare(
                np.concatenate([np.asarray(row[mode]) for row in rows]), expected
            )
            for mode in ["prefill", "decode"]
        }
        need(sha(gguf) == gguf_before, "exported model changed during host replay")
        cases.append(
            {
                "objective": objective,
                "layers": depth,
                "d_ff": hidden,
                "F32_export_sha256": gguf_before,
                "CLI_numpy_normalized_error": cli_error,
                "Ember_numpy_normalized_errors": errors,
                "all_8x7_rows_and_argmax": True,
            }
        )
    need(
        all(sha(name) == digest for name, digest in pins.items())
        and sha(ember) == EMBER_SHA,
        "host source/binary pins changed",
    )
    result = {
        "pass": True,
        "bound": BOUND,
        "numpy": np.__version__,
        "CPU_Ember_sha256": EMBER_SHA,
        "host_source_pins_before_after": pins,
        "cases": cases,
        "scope": "Saved FP32 inference contract only; no CUDA, model quality, speed or adoption claim.",
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--artifact", required=True)
    p.add_argument("--ember", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    print(json.dumps(replay(args.artifact, args.ember, args.out)))
