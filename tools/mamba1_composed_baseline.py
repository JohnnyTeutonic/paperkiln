"""Bounded CPU Mamba1 CLI/save/export/independent-engine baseline; no device/adoption claim."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "spec/tools"))
import ir_check  # noqa: E402
import ir_numpy  # noqa: E402

TOKENS = ["<unk>", "the", "cat", "sat", "on", "mat", "."]
SEQUENCES = [[1], [1, 2, 3], [6, 5, 4, 3, 2, 1]]


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def vocab(path):
    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b

    metadata = [
        s("tokenizer.ggml.model") + struct.pack("<I", 8) + s("word"),
        s("tokenizer.ggml.tokens")
        + struct.pack("<IIQ", 9, 8, len(TOKENS))
        + b"".join(s(x) for x in TOKENS),
    ]
    path.write_bytes(
        struct.pack("<IIQQ", 0x46554747, 3, 0, len(metadata)) + b"".join(metadata)
    )


def compare(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    need(
        a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(),
        "nonfinite or shape differs",
    )
    e = float(np.max(np.abs(a - b))) / max(1, float(np.max(np.abs(b))))
    need(e <= 1e-5, "full-row numeric bound fails: " + str(e))
    need(np.array_equal(a.argmax(axis=1), b.argmax(axis=1)), "argmax differs")
    return e


def run(binary, ember, out):
    binary, ember, out = (
        Path(binary).resolve(),
        Path(ember).resolve(),
        Path(out).resolve(),
    )
    out.mkdir(parents=True, exist_ok=False)
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "MICROTORCH_DEVICE": "cpu",
        "MICROTORCH_DEVICE_OPS": "0",
        "MICROTORCH_STEP_RESIDENCY": "0",
        "MICROTORCH_DEFER_DOWNLOADS": "0",
        "EMBER_DEVICE": "cpu",
        "EMBER_IR_DEVICE": "cpu",
    }
    identity = {
        "mtstudio": sha(binary),
        "ember": sha(ember),
        "sources": {
            f: sha(ROOT / f)
            for f in [
                "tools/parity_model.hpp",
                "tools/mtstudio.cpp",
                "src/ops.cpp",
                "src/selective_scan_kernels.cuh",
                "tools/mamba1_composed_baseline.py",
            ]
        },
    }
    summary = {
        "pass": False,
        "scope": "two tiny CPU Mamba1 CLI training/save/export/reload and NumPy/Ember forward baselines; no CUDA, speed or model-quality claim",
        "identity": identity,
        "cases": [],
    }

    def command(name, argv, directory, stdin=None, environment=None):
        r = subprocess.run(
            list(map(str, argv)),
            input=stdin,
            capture_output=True,
            text=True,
            timeout=120,
            env=environment or env,
        )
        (directory / (name + ".stdout")).write_text(r.stdout)
        (directory / (name + ".stderr")).write_text(r.stderr)
        need(
            r.returncode == 0,
            name + " failed with exit " + str(r.returncode) + ": " + r.stderr[-1000:],
        )
        return r

    for depth in (1, 2):
        folder = out / ("depth" + str(depth))
        folder.mkdir()
        vocab(folder / "vocab.gguf")
        (folder / "corpus.txt").write_text("the cat sat on the mat . " * 100)
        spec = {
            "name": "mamba1_cpu_depth" + str(depth),
            "arch": {
                "preset": "mamba-tiny",
                "custom": {
                    "d": 8,
                    "layers": depth,
                    "d_state": 2,
                    "ssm": "mamba1",
                    "d_conv": 3,
                    "expand": 2,
                    "dt_rank": 1,
                },
            },
            "data": {
                "vocab": str(folder / "vocab.gguf"),
                "corpus": str(folder / "corpus.txt"),
                "T": 8,
            },
            "train": {
                "steps": 3,
                "batch": 2,
                "seed": 9,
                "lr": 0.001,
                "eval_every": 3,
                "checkpoint_every": 3,
                "gradmap_every": 0,
            },
            "export": {"formats": ["safetensors", "gguf"]},
            "out_dir": str(folder / "trained"),
        }
        specpath = folder / "spec.json"
        specpath.write_text(json.dumps(spec, indent=2) + "\n")
        command("train", [binary, "run", specpath], folder)
        events = [
            json.loads(x)
            for x in (folder / "trained/events.jsonl").read_text().splitlines()
        ]
        model = next(x for x in events if x["event"] == "model")
        done = next(x for x in events if x["event"] == "done")
        steps = [x for x in events if x["event"] == "step"]
        need(
            model["family"] == "mamba"
            and model["ssm"]["kind"] == "mamba1"
            and model["layers"] == depth,
            "wrong model path",
        )
        need(done["final_step"] == 3 and len(steps) == 3, "wrong training step count")
        need(
            all(
                np.isfinite(x["loss"])
                and np.isfinite(x["grad_norm"])
                and x["grad_norm"] > 0
                for x in steps
            ),
            "training diagnostics invalid",
        )
        gguf = Path(
            next(
                x["path"]
                for x in events
                if x.get("event") == "export" and x.get("format") == "gguf"
            )
        )
        before = sha(gguf)
        command("reexport", [binary, "export", specpath], folder)
        need(sha(gguf) == before, "reloaded FP32 export bytes differ")
        command(
            "ir_check", [sys.executable, ROOT / "spec/tools/ir_check.py", gguf], folder
        )
        meta, tensors, offset, raw, _ = ir_check.read_gguf(gguf)
        need(meta["general.architecture"][1] == "paperkiln", "wrong architecture")
        weights = {}
        for name, dims, kind, location in tensors:
            need(kind == 0, "non-F32 export")
            weights[name] = np.frombuffer(
                raw, dtype="<f4", count=int(np.prod(dims)), offset=offset + location
            ).reshape(tuple(reversed(dims)))
        doc = json.loads(meta["paperkiln.ir"][1])
        need(
            doc["family_hint"] == "mamba1" and doc["block"]["repeat"] == depth,
            "wrong IR family or depth",
        )
        need(
            {x["op"] for x in doc["block"]["state"]}
            == {"conv1d_causal", "selective_scan"},
            "wrong recurrent states",
        )
        refs = [ir_numpy.forward(doc, weights, ids) for ids in SEQUENCES]
        req = folder / "score.in.jsonl"
        res = folder / "score.out.jsonl"
        rows = folder / "score.rows.f32"
        req.write_text(
            "".join(
                json.dumps({"prompt_ids": ids[:1], "continuation_ids": ids[1:] + [1]})
                + "\n"
                for ids in SEQUENCES
            )
        )
        command(
            "source_score",
            [binary, "score", specpath, "--in", req, "--out", res, "--logits", rows],
            folder,
        )
        source = np.fromfile(rows, dtype="<f4").reshape(-1, len(TOKENS))
        native_error = compare(source, np.concatenate(refs))
        engine = command(
            "ember",
            [ember, gguf],
            folder,
            json.dumps(SEQUENCES),
            {**env, "EMBER_IR": str(gguf.with_suffix(".ir.json"))},
        )
        actual = json.loads(engine.stdout)
        need(len(actual) == len(SEQUENCES), "wrong Ember case count")
        errors = {
            mode: max(compare(row[mode], ref) for row, ref in zip(actual, refs))
            for mode in ("prefill", "decode")
        }
        record = {
            "depth": depth,
            "ssm": "mamba1",
            "steps": 3,
            "batch": 2,
            "source_numpy": native_error,
            "ember_numpy": errors,
            "f32_export_sha256": before,
            "reexport_bytes_equal": True,
            "parameters": model["params"],
            "IR_states": ["conv1d_causal", "selective_scan"],
            "score_rows": len(source),
        }
        (folder / "result.json").write_text(json.dumps(record, indent=2) + "\n")
        summary["cases"].append(record)
    need(
        sha(binary) == identity["mtstudio"] and sha(ember) == identity["ember"],
        "binary changed during baseline",
    )
    summary["pass"] = True
    (out / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mtstudio", required=True)
    p.add_argument("--ember", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    run(a.mtstudio, a.ember, a.out)
