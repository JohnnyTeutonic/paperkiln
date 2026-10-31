"""Tiny actual CLI lifecycle producer and frozen host NumPy/Ember replay."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import numpy as np

ROOT = Path(
    os.environ.get("MAMBA1_CLI_SOURCE_ROOT", Path(__file__).resolve().parents[1])
).resolve()
sys.path.insert(0, str(ROOT / "spec/tools"))
import ir_check  # noqa: E402
import ir_numpy  # noqa: E402

TOKENS = ["<unk>", "the", "cat", "sat", "on", "mat", "."]
SEQUENCES = [[1], [1, 2, 3], [6, 5, 4, 3]]
BOUND = 1e-5
EXPECTED_EMBER_SHA256 = (
    "1f877524c3b59e1cdbf3ca29ab6d4e0ed216900516e1bcb28d5ec7e7e21b9a0f"
)
CORPUS = "the cat sat on the mat . " * 100


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def vocab_bytes():
    def text(value):
        data = value.encode()
        return struct.pack("<Q", len(data)) + data

    metadata = [
        text("tokenizer.ggml.model") + struct.pack("<I", 8) + text("word"),
        text("tokenizer.ggml.tokens")
        + struct.pack("<IIQ", 9, 8, len(TOKENS))
        + b"".join(text(t) for t in TOKENS),
    ]
    return struct.pack("<IIQQ", 0x46554747, 3, 0, len(metadata)) + b"".join(metadata)


def environment(gpu=False):
    return {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "MICROTORCH_DEVICE": "cuda" if gpu else "cpu",
        "MICROTORCH_DEVICE_OPS": "1" if gpu else "0",
        "MICROTORCH_STEP_RESIDENCY": "1" if gpu else "0",
        "MICROTORCH_DEFER_DOWNLOADS": "1" if gpu else "0",
        "MICROTORCH_GPU_PROFILE": "1" if gpu else "0",
        "MICROTORCH_CUBLAS": "0",
        "MICROTORCH_TF32": "0",
        "EMBER_DEVICE": "cpu",
        "EMBER_IR_DEVICE": "cpu",
    }


def command(name, argv, folder, env, stdin=None):
    p = subprocess.run(
        list(map(str, argv)),
        input=stdin,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        timeout=120,
    )
    (folder / (name + ".stdout")).write_text(p.stdout)
    (folder / (name + ".stderr")).write_text(p.stderr)
    need(p.returncode == 0, name + " failed: " + p.stderr[-800:])
    return p


def check_events(events, depth, gpu):
    model = [e for e in events if e.get("event") == "model"]
    done = [e for e in events if e.get("event") == "done"]
    steps = [e for e in events if e.get("event") == "step"]
    need(len(model) == len(done) == 1 and len(steps) == 3, "event inventory")
    m = model[0]
    for key, value in {
        "family": "mamba",
        "d": 8,
        "layers": depth,
        "vocab": 7,
        "T": 4,
        "batch": 2,
        "seed": 37,
        "params": 120 + 328 * depth,
    }.items():
        need(m.get(key) == value, "registered model " + key)
        if type(value) is int:
            need(type(m[key]) is int, "integer model " + key)
    need(
        m.get("ssm")
        == {"kind": "mamba1", "d_conv": 3, "d_state": 2, "expand": 1, "dt_rank": 2},
        "actual Mamba1 path",
    )
    need(
        done[0].get("final_step") == 3 and [e.get("step") for e in steps] == [1, 2, 3],
        "three completed LM steps",
    )
    need(
        type(done[0].get("final_step")) is int
        and all(type(e.get("step")) is int for e in steps),
        "integer step counters",
    )
    for step in steps:
        need(
            all(
                isinstance(step.get(k), (int, float))
                and not isinstance(step[k], bool)
                and math.isfinite(step[k])
                for k in ["loss", "grad_norm"]
            )
            and step["grad_norm"] > 0,
            "finite nonzero training diagnostics",
        )
        if gpu:
            for op in [
                "conv1d_causal_fwd",
                "conv1d_causal_bwd",
                "selective_scan_fwd",
                "selective_scan_bwd",
            ]:
                count = step.get("gpu_n", {}).get(op)
                ms = step.get("gpu_ms", {}).get(op)
                need(
                    type(count) is int
                    and count >= depth
                    and isinstance(ms, (int, float))
                    and not isinstance(ms, bool)
                    and math.isfinite(ms)
                    and ms >= 0,
                    "actual CLI CUDA profile " + op,
                )
        else:
            need(
                not step.get("gpu_n") and not step.get("gpu_ms"),
                "CPU control unexpectedly has CUDA profile",
            )
    return steps


def produce(binary, out, gpu=False):
    binary = Path(binary).resolve()
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = environment(gpu)
    identity = sha(binary)
    records = []
    for depth in [1, 2]:
        folder = out / f"depth{depth}"
        folder.mkdir()
        (folder / "vocab.gguf").write_bytes(vocab_bytes())
        (folder / "corpus.txt").write_text(CORPUS)
        spec = {
            "name": f"mamba1_cli_depth{depth}",
            "arch": {
                "preset": "mamba-tiny",
                "custom": {
                    "d": 8,
                    "layers": depth,
                    "d_state": 2,
                    "ssm": "mamba1",
                    "d_conv": 3,
                    "expand": 1,
                    "dt_rank": 2,
                },
            },
            "data": {
                "vocab": str(folder / "vocab.gguf"),
                "corpus": str(folder / "corpus.txt"),
                "T": 4,
            },
            "train": {
                "steps": 3,
                "batch": 2,
                "seed": 37,
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
        command("train", [binary, "run", specpath], folder, env)
        events = [
            json.loads(x)
            for x in (folder / "trained/events.jsonl").read_text().splitlines()
        ]
        steps = check_events(events, depth, gpu)
        gguf = folder / f"trained/mamba1_cli_depth{depth}.gguf"
        need(gguf.is_file(), "missing F32 export")
        before = sha(gguf)
        for name in ["model.safetensors", "optim.safetensors", "state.txt"]:
            need(
                (folder / "trained" / name).is_file(),
                "missing saved lifecycle artifact " + name,
            )
        command("reload_export", [binary, "export", specpath], folder, env)
        need(sha(gguf) == before, "saved reload F32 export bytes differ")
        requests = [
            {"prompt_ids": ids[:1], "continuation_ids": ids[1:] + [1]}
            for ids in SEQUENCES
        ]
        req = folder / "score.in.jsonl"
        req.write_text("".join(json.dumps(x) + "\n" for x in requests))
        command(
            "saved_score",
            [
                binary,
                "score",
                specpath,
                "--in",
                req,
                "--out",
                folder / "score.out.jsonl",
                "--logits",
                folder / "score.rows.f32",
            ],
            folder,
            env,
        )
        rows = np.fromfile(folder / "score.rows.f32", dtype="<f4")
        need(
            rows.size == 8 * 7 and np.isfinite(rows).all(),
            "saved score exact shape and finite",
        )
        records.append(
            {
                "depth": depth,
                "actual_cli_cuda": gpu,
                "LM_steps": 3,
                "parameter_elements": 120 + 328 * depth,
                "F32_export_sha256": before,
                "reload_export_byte_identical": True,
                "score_shape": [8, 7],
                "training_steps": steps,
            }
        )
    need(sha(binary) == identity, "CLI binary changed")
    result = {
        "remote_or_CPU_producer_pass": True,
        "actual_cli_cuda": gpu,
        "binary_sha256": identity,
        "cases": records,
        "artifact_sha256": {
            str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file()
        },
        "scope": "LM three-step CLI lifecycle only; host independent logits/Ember replay still required, no quality/speed/full residency claim",
    }
    (out / "producer.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def compare(actual, reference, shape):
    a = np.asarray(actual, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    need(
        a.shape == b.shape == shape and np.isfinite(a).all() and np.isfinite(b).all(),
        "exact full-row shape and finite",
    )
    error = float(np.max(abs(a - b))) / max(1, float(np.max(abs(b))))
    need(error <= BOUND, "full-row normalized bound")
    need(np.array_equal(a.argmax(axis=1), b.argmax(axis=1)), "argmax agreement")
    return error


def replay(artifact, ember, out):
    artifact = Path(artifact).resolve()
    ember = Path(ember).resolve()
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    summary = json.loads((artifact / "producer.json").read_text())
    need(
        summary.get("remote_or_CPU_producer_pass") is True
        and len(summary.get("cases", [])) == 2,
        "producer completion",
    )
    gpu = summary.get("actual_cli_cuda")
    need(type(gpu) is bool, "producer backend")
    need(
        set(summary["artifact_sha256"])
        == {
            str(p.relative_to(artifact))
            for p in artifact.rglob("*")
            if p.is_file() and p != artifact / "producer.json"
        },
        "complete artifact hash inventory",
    )
    need(
        sha(ember) == EXPECTED_EMBER_SHA256 and np.__version__ == "1.26.4",
        "frozen host Ember/NumPy identity",
    )
    for f, h in summary["artifact_sha256"].items():
        path = Path(f)
        need(
            not path.is_absolute()
            and ".." not in path.parts
            and sha(artifact / path) == h,
            "artifact pin " + f,
        )
    identity = {
        "ember_binary_sha256": sha(ember),
        "host_reference_sources": {
            f: sha(ROOT / f)
            for f in [
                "spec/tools/ir_numpy.py",
                "spec/tools/ir_check.py",
                "tools/mamba1_cli_lifecycle.py",
            ]
        },
        "numpy": np.__version__,
    }
    records = []
    for depth, producer in zip([1, 2], summary["cases"]):
        folder = artifact / f"depth{depth}"
        local = out / f"depth{depth}"
        local.mkdir()
        need(
            producer.get("depth") == depth and producer.get("actual_cli_cuda") is gpu,
            "case/backend",
        )
        need(
            producer.get("LM_steps") == 3
            and producer.get("parameter_elements") == 120 + 328 * depth
            and producer.get("score_shape") == [8, 7],
            "producer registered counters",
        )
        need(
            (folder / "vocab.gguf").read_bytes() == vocab_bytes()
            and (folder / "corpus.txt").read_text() == CORPUS,
            "registered tokenizer/corpus",
        )
        spec = json.loads((folder / "spec.json").read_text())
        need(
            spec["arch"]["custom"]
            == {
                "d": 8,
                "layers": depth,
                "d_state": 2,
                "ssm": "mamba1",
                "d_conv": 3,
                "expand": 1,
                "dt_rank": 2,
            }
            and spec["data"]["T"] == 4
            and spec["train"]["steps"] == 3
            and spec["train"]["batch"] == 2
            and spec["train"]["seed"] == 37,
            "registered spec",
        )
        events = [
            json.loads(x)
            for x in (folder / "trained/events.jsonl").read_text().splitlines()
        ]
        check_events(events, depth, gpu)
        gguf = folder / f"trained/mamba1_cli_depth{depth}.gguf"
        need(
            sha(gguf) == producer["F32_export_sha256"]
            and producer.get("reload_export_byte_identical") is True,
            "F32 bytes proof",
        )
        command(
            "ir_check",
            [sys.executable, ROOT / "spec/tools/ir_check.py", gguf],
            local,
            environment(),
        )
        meta, tensors, offset, raw, _ = ir_check.read_gguf(gguf)
        doc = json.loads(meta["paperkiln.ir"][1])
        need(
            doc.get("family_hint") == "mamba1"
            and doc["block"]["repeat"] == depth
            and doc["model"]
            == {"vocab": 7, "d": 8, "n_layers": depth, "ctx": 4, "dtype": "f32"},
            "registered IR model",
        )
        need(
            {x["op"] for x in doc["block"]["state"]}
            == {"conv1d_causal", "selective_scan"},
            "two recurrent state kinds",
        )
        weights = {}
        for name, dims, kind, location in tensors:
            need(kind == 0, "all F32 tensors")
            array = np.frombuffer(
                raw, dtype="<f4", count=int(np.prod(dims)), offset=offset + location
            ).reshape(tuple(reversed(dims)))
            need(np.isfinite(array).all(), "finite exported weights")
            weights[name] = array
        refs = [ir_numpy.forward(doc, weights, ids) for ids in SEQUENCES]
        reference = np.concatenate(refs)
        need(reference.shape == (8, 7), "reference shape")
        source = np.fromfile(folder / "score.rows.f32", dtype="<f4")
        need(source.size == 56, "score raw byte inventory")
        source = source.reshape(8, 7)
        source_error = compare(source, reference, (8, 7))
        score = [
            json.loads(x) for x in (folder / "score.out.jsonl").read_text().splitlines()
        ]
        need(len(score) == 3, "score case inventory")
        cursor = 0
        for i, (ids, row) in enumerate(zip(SEQUENCES, score)):
            n = len(ids)
            need(
                row.get("i") == i
                and row.get("continuation_ids") == ids[1:] + [1]
                and row.get("logits_rows") == n
                and row.get("argmax")
                == (source[cursor : cursor + n, 1:].argmax(axis=1) + 1).tolist(),
                "score registered rows/argmax",
            )
            need(
                len(row.get("logprobs", [])) == n
                and all(
                    isinstance(v, (int, float)) and math.isfinite(v)
                    for v in row["logprobs"]
                ),
                "finite score probabilities",
            )
            cursor += n
        engine = command(
            "ember",
            [ember, gguf],
            local,
            {**environment(), "EMBER_IR": str(gguf.with_suffix(".ir.json"))},
            json.dumps(SEQUENCES),
        )
        actual = json.loads(engine.stdout)
        need(
            isinstance(actual, list) and len(actual) == 3, "Ember exact case inventory"
        )
        errors = {
            mode: max(
                compare(row[mode], ref, (len(ids), 7))
                for ids, row, ref in zip(SEQUENCES, actual, refs)
            )
            for mode in ["prefill", "decode"]
        }
        records.append(
            {
                "depth": depth,
                "source_numpy_normalized_error": source_error,
                "ember_numpy_normalized_error": errors,
                "all_full_rows_argmax_agree": True,
                "score_shape": [8, 7],
                "F32_export_sha256": sha(gguf),
                "actual_cli_cuda": gpu,
            }
        )
    need(
        sha(ember) == identity["ember_binary_sha256"],
        "Ember binary changed during replay",
    )
    result = {
        "pass": True,
        "actual_cli_cuda": gpu,
        "identity": identity,
        "bound": BOUND,
        "cases": records,
        "scope": "two tiny LM three-step CLI training/save/reload/F32 export and full-row NumPy/Ember prefill/decode replay; no quality/speed/full residency claim",
    }
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("produce")
    a.add_argument("--mtstudio", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--require-cuda", action="store_true")
    a = sub.add_parser("replay")
    a.add_argument("--artifact", required=True)
    a.add_argument("--ember", required=True)
    a.add_argument("--out", required=True)
    args = p.parse_args()
    print(
        json.dumps(
            produce(args.mtstudio, args.out, args.require_cuda)
            if args.mode == "produce"
            else replay(args.artifact, args.ember, args.out)
        )
    )
