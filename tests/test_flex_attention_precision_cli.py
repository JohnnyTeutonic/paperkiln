"""Actual tiny CLI controls: train-only BF16 selection and FP32 saved inference."""

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "spec/tools"))
from test_mtstudio_chat import write_vocab_gguf  # noqa: E402
import ir_check  # noqa: E402
import ir_numpy  # noqa: E402

BIN = os.environ.get("MTSTUDIO_BIN")
pytestmark = pytest.mark.skipif(not BIN, reason="set MTSTUDIO_BIN to actual CPU binary")
TOKENS = ["<unk>", "the", "cat", "sat", "on", "mat", "."]


def execute(directory, label, *args):
    env = {
        **os.environ,
        "MICROTORCH_DEVICE": "cpu",
        "MICROTORCH_DEVICE_OPS": "0",
        "MICROTORCH_STEP_RESIDENCY": "0",
        "MICROTORCH_DEFER_DOWNLOADS": "0",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    p = subprocess.run(
        [BIN, *map(str, args)], text=True, capture_output=True, timeout=120, env=env
    )
    (directory / (label + ".stdout")).write_text(p.stdout)
    (directory / (label + ".stderr")).write_text(p.stderr)
    return p


def save(directory, name, spec):
    path = directory / (name + ".json")
    path.write_text(json.dumps(spec, indent=2) + "\n")
    return path


def fixture(directory, layers=1, hidden=12):
    directory.mkdir(parents=True, exist_ok=True)
    write_vocab_gguf(directory / "v.gguf", TOKENS)
    (directory / "corpus.txt").write_text("the cat sat on the mat . " * 400)
    return {
        "name": "flex_precision",
        "arch": {
            "preset": "gpt2-nano",
            "custom": {
                "d": 8,
                "heads": 2,
                "layers": layers,
                "d_ff": hidden,
                "norm": "layernorm",
            },
        },
        "data": {
            "vocab": str(directory / "v.gguf"),
            "corpus": str(directory / "corpus.txt"),
            "T": 8,
        },
        "train": {
            "steps": 3,
            "batch": 2,
            "seed": 37,
            "lr": 0.001,
            "eval_every": 3,
            "checkpoint_every": 3,
            "eval_windows": 2,
            "gradmap_every": 0,
        },
        "export": {"formats": ["safetensors", "gguf"]},
        "out_dir": str(directory / "trained"),
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("attention", "swa"),
        ("position", "rope"),
        ("position", "sinusoidal"),
        ("position", "t5_relative"),
        ("position", "shaw_relative"),
        ("activation", "relu"),
        ("activation", "gelu_erf"),
        ("activation", "swiglu"),
        ("norm", "rmsnorm"),
        ("residual", "highway"),
        ("residual", "plain"),
        ("tie_head", True),
        ("qkv_conv", 3),
        ("qk_norm", "per_head"),
        ("final_norm", False),
        ("norm_placement", "post"),
        ("embed", "expr:x"),
        ("logits", "expr:x"),
    ],
)
def test_unsupported_flex_rejected_before_output(tmp_path, field, value):
    s = fixture(tmp_path)
    s["arch"]["custom"][field] = value
    s["train"]["attention_precision"] = "bf16"
    p = execute(tmp_path, "refusal", "run", save(tmp_path, "bad", s))
    assert p.returncode != 0 and not Path(s["out_dir"]).exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("attention_kernel", "flash"),
        ("precision", "bf16"),
        ("checkpoint_activations", True),
        ("fused_head", True),
        ("objective", "grpo"),
        ("attention_precision", "fp16"),
    ],
)
def test_unsupported_training_combination_rejected(tmp_path, field, value):
    s = fixture(tmp_path)
    s["train"]["attention_precision"] = "bf16"
    s["train"][field] = value
    p = execute(tmp_path, "refusal", "run", save(tmp_path, "bad", s))
    assert p.returncode != 0 and not Path(s["out_dir"]).exists()


def test_default_omitted_and_explicit_fp32_training_bit_equal(tmp_path):
    proof = os.environ.get("FLEX_PRECISION_PROOF_DIR")
    if proof:
        tmp_path = Path(proof) / "default"
        tmp_path.mkdir(parents=True, exist_ok=False)
    s = fixture(tmp_path)
    paths = []
    for label, explicit in [("omitted", False), ("explicit", True)]:
        spec = copy.deepcopy(s)
        spec["out_dir"] = str(tmp_path / label)
        if explicit:
            spec["train"]["attention_precision"] = "fp32"
        p = execute(tmp_path, label, "run", save(tmp_path, label, spec))
        assert p.returncode == 0, p.stdout + p.stderr
        paths.append(Path(spec["out_dir"]))
    for name in ["model.safetensors", "optim.safetensors", "flex_precision.gguf"]:
        assert (paths[0] / name).read_bytes() == (paths[1] / name).read_bytes()


@pytest.mark.parametrize("objective,layers,hidden", [("lm", 1, 12), ("sft", 3, 20)])
def test_train_only_precision_saved_scoring_export_and_independent_rows(
    tmp_path, objective, layers, hidden
):
    proof = os.environ.get("FLEX_PRECISION_PROOF_DIR")
    directory = Path(proof) / objective if proof else tmp_path
    if proof:
        directory.mkdir(parents=True, exist_ok=False)
    s = fixture(directory, layers, hidden)
    s["train"].update(attention_precision="bf16", objective=objective)
    if objective == "sft":
        examples = directory / "examples.jsonl"
        examples.write_text(
            "".join(
                json.dumps(
                    {
                        "segments": [
                            {"text": "the cat", "loss": False},
                            {"text": "sat on the mat .", "loss": True},
                        ]
                    }
                )
                + "\n"
                for _ in range(30)
            )
        )
        s["data"]["sft_examples"] = str(examples)
    sp = save(directory, "bf16", s)
    p = execute(directory, "training", "run", sp)
    assert p.returncode == 0, p.stdout + p.stderr
    trained = Path(s["out_dir"])
    result = json.loads((trained / "result.json").read_text())
    assert result["final_step"] == 3 and result["train_attention_precision"] == "bf16"
    assert np.isfinite(result["best_val"])
    events = [
        json.loads(x) for x in (trained / "events.jsonl").read_text().splitlines()
    ]
    model = next(x for x in events if x["event"] == "model")
    assert model["family"] == "flex" and model["train_attention_precision"] == "bf16"
    assert len([x for x in events if x["event"] == "step"]) == 3
    inference = copy.deepcopy(s)
    inference["train"]["attention_precision"] = "fp32"
    fp = save(directory, "fp32_inference", inference)
    sequences = [[1], [1, 2, 3], [6, 5, 4, 3]]
    requests = directory / "requests.jsonl"
    requests.write_text(
        "".join(
            json.dumps({"prompt_ids": [ids[0]], "continuation_ids": ids[1:] + [2]})
            + "\n"
            for ids in sequences
        )
    )
    for label, spec in [("bf16_recipe", sp), ("fp32_recipe", fp)]:
        p = execute(
            directory,
            label,
            "score",
            spec,
            "--in",
            requests,
            "--out",
            directory / (label + ".jsonl"),
            "--logits",
            directory / (label + ".f32"),
        )
        assert p.returncode == 0, p.stdout + p.stderr
    assert (directory / "bf16_recipe.f32").read_bytes() == (
        directory / "fp32_recipe.f32"
    ).read_bytes()
    actual = np.fromfile(directory / "bf16_recipe.f32", dtype="<f4").reshape(
        8, len(TOKENS)
    )
    gguf = trained / "flex_precision.gguf"
    meta, tensors, offset, raw, _ = ir_check.read_gguf(gguf)
    assert meta["paperkiln.train.attention_precision"][1] == "bf16"
    doc = json.loads(meta["paperkiln.ir"][1])
    assert all(kind == 0 for _, _, kind, _ in tensors)
    weights = {
        name: np.frombuffer(
            raw, dtype="<f4", count=int(np.prod(dims)), offset=offset + location
        ).reshape(tuple(reversed(dims)))
        for name, dims, _, location in tensors
    }
    oracle = np.concatenate([ir_numpy.forward(doc, weights, ids) for ids in sequences])
    assert (
        actual.shape == oracle.shape
        and np.isfinite(actual).all()
        and np.isfinite(oracle).all()
    )
    error = float(np.max(abs(actual - oracle) / (1 + abs(oracle))))
    assert error <= 1e-5 and np.array_equal(actual.argmax(1), oracle.argmax(1))
    exported = directory / "reloaded_export"
    p = execute(
        directory,
        "reload_export",
        "export",
        sp,
        "--formats",
        "gguf",
        "--out-dir",
        exported,
    )
    assert p.returncode == 0, p.stdout + p.stderr
    assert gguf.read_bytes() == (exported / "flex_precision.gguf").read_bytes()
    plain = directory / "explicit_fp32_export"
    p = execute(
        directory, "fp32_export", "export", fp, "--formats", "gguf", "--out-dir", plain
    )
    assert p.returncode == 0, p.stdout + p.stderr
    m2, t2, o2, r2, _ = ir_check.read_gguf(plain / "flex_precision.gguf")
    assert "paperkiln.train.attention_precision" not in m2
    assert (
        doc == json.loads(m2["paperkiln.ir"][1])
        and tensors == t2
        and raw[offset:] == r2[o2:]
    )
    (directory / "replay.json").write_text(
        json.dumps(
            {
                "pass": True,
                "objective": objective,
                "layers": layers,
                "d_ff": hidden,
                "actual_cuda": False,
                "steps": 3,
                "all_8x7_rows_error": error,
                "all_argmax_agree": True,
                "reload_F32_GGUF_byte_identical": True,
                "saved_inference_recipe_precision_bit_equal": True,
                "binary_sha256": hashlib.sha256(Path(BIN).read_bytes()).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("layers", -1),
        ("layers", 0),
        ("layers", 1.5),
        ("layers", True),
        ("d", -8),
        ("heads", -2),
        ("d_ff", -12),
        ("T", -8),
    ],
)
def test_bf16_geometry_conversion_refused(tmp_path, field, value):
    recipe = fixture(tmp_path)
    recipe["train"]["attention_precision"] = "bf16"
    if field == "T":
        recipe["data"][field] = value
    else:
        recipe["arch"]["custom"][field] = value
    p = execute(tmp_path, "invalid_geometry", "run", save(tmp_path, "bad", recipe))
    assert p.returncode != 0 and not Path(recipe["out_dir"]).exists()
