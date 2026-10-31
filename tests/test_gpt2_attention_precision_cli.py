"""Tiny CPU training recipe, fixed defaults and unchanged FP32 inference contract."""

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


def run(args, directory, label):
    p = subprocess.run(
        [str(x) for x in args],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
    )
    (directory / (label + ".stdout")).write_text(p.stdout)
    (directory / (label + ".stderr")).write_text(p.stderr)
    return p


def fixture(directory):
    directory.mkdir(parents=True, exist_ok=True)
    write_vocab_gguf(directory / "v.gguf", TOKENS)
    (directory / "c.txt").write_text("the cat sat on the mat . " * 400)
    return {
        "name": "precision",
        "arch": {"preset": "gpt2-nano", "custom": {"d": 8, "heads": 2, "layers": 2}},
        "data": {
            "vocab": str(directory / "v.gguf"),
            "corpus": str(directory / "c.txt"),
            "T": 4,
            "max_tokens": 0,
        },
        "train": {
            "steps": 6,
            "batch": 2,
            "seed": 17,
            "lr": 0.01,
            "eval_every": 3,
            "checkpoint_every": 3,
            "eval_windows": 2,
        },
        "export": {"formats": ["safetensors", "gguf"]},
        "out_dir": str(directory / "run"),
    }


def save(directory, spec, name):
    p = directory / (name + ".json")
    p.write_text(json.dumps(spec, indent=1) + "\n")
    return p


@pytest.mark.parametrize(
    "change",
    [
        "invalid",
        "llama",
        "flex",
        "depth",
        "swa",
        "kimi",
        "flash",
        "projections",
        "checkpoint",
        "objective",
        "state",
        "custom",
    ],
)
def test_unsupported_is_rejected_before_output(tmp_path, change):
    s = fixture(tmp_path)
    s["train"]["attention_precision"] = "bf16"
    if change == "invalid":
        s["train"]["attention_precision"] = "fp16"
    elif change == "llama":
        s["arch"]["preset"] = "llama-tiny"
    elif change == "flex":
        s["arch"]["custom"]["activation"] = "relu"
    elif change == "depth":
        s["arch"]["custom"]["layers"] = 3
    elif change in ("swa", "kimi"):
        s["arch"]["custom"]["attention"] = change
    elif change == "flash":
        s["train"]["attention_kernel"] = "flash"
    elif change == "projections":
        s["train"]["precision"] = "bf16"
    elif change == "checkpoint":
        s["train"]["checkpoint_activations"] = True
    elif change == "objective":
        s["train"]["objective"] = "rejection_sft"
        s["data"]["sft_examples"] = "unused.jsonl"
    elif change == "state":
        s["arch"]["state_tokens"] = {"k": 1, "mask": "turn_state"}
    elif change == "custom":
        s["arch"]["custom"]["position"] = "sinusoidal"
    p = run([BIN, "run", save(tmp_path, s, "bad")], tmp_path, "bad")
    assert p.returncode != 0 and not Path(s["out_dir"]).exists()
    assert "attention_precision" in p.stderr or "state_tokens" in p.stderr


def test_bf16_training_reloads_as_fp32_inference_and_exports(tmp_path):
    proof = os.environ.get("GPT2_PRECISION_PROOF_DIR")
    directory = Path(proof) / "bf16_cli" if proof else tmp_path
    directory.mkdir(parents=True, exist_ok=False) if proof else None
    s = fixture(directory)
    s["train"]["attention_precision"] = "bf16"
    sp = save(directory, s, "bf16")
    p = run([BIN, "run", sp], directory, "train")
    assert p.returncode == 0, p.stdout + p.stderr
    out = Path(s["out_dir"])
    events = [json.loads(x) for x in (out / "events.jsonl").read_text().splitlines()]
    model = next(x for x in events if x["event"] == "model")
    assert model["train_attention_precision"] == "bf16"
    assert (
        json.loads((out / "result.json").read_text())["train_attention_precision"]
        == "bf16"
    )
    s2 = copy.deepcopy(s)
    s2["train"].pop("attention_precision")
    fp = save(directory, s2, "inference_fp32")
    req = directory / "req.jsonl"
    req.write_text(
        json.dumps({"prompt_ids": [1], "continuation_ids": [2, 3, 4]}) + "\n"
    )
    for label, path in [("trained_recipe", sp), ("explicit_fp32", fp)]:
        p = run(
            [
                BIN,
                "score",
                path,
                "--in",
                req,
                "--out",
                directory / (label + ".scores.jsonl"),
                "--logits",
                directory / (label + ".f32"),
            ],
            directory,
            label,
        )
        assert p.returncode == 0, p.stdout + p.stderr
    assert (directory / "trained_recipe.f32").read_bytes() == (
        directory / "explicit_fp32.f32"
    ).read_bytes()
    expected = np.fromfile(directory / "trained_recipe.f32", dtype="<f4").reshape(
        -1, len(TOKENS)
    )
    g = out / "precision.gguf"
    meta, tensors, offset, raw, _ = ir_check.read_gguf(g)
    assert meta["paperkiln.train.attention_precision"][1] == "bf16"
    doc = json.loads(meta["paperkiln.ir"][1])
    weights = {
        name: np.frombuffer(
            raw, dtype="<f4", count=int(np.prod(dims)), offset=offset + location
        ).reshape(tuple(reversed(dims)))
        for name, dims, kind, location in tensors
        if kind == 0
    }
    oracle = ir_numpy.forward(doc, weights, [1, 2, 3])
    err = float(np.max(np.abs(expected - oracle)))
    assert err < 3e-5
    p = run(
        [
            BIN,
            "export",
            fp,
            "--formats",
            "gguf",
            "--out-dir",
            directory / "export_fp32",
        ],
        directory,
        "export",
    )
    assert p.returncode == 0, p.stdout + p.stderr
    plain = directory / "export_fp32/precision.gguf"
    m2, t2, o2, r2, _ = ir_check.read_gguf(plain)
    assert "paperkiln.train.attention_precision" not in m2
    assert doc == json.loads(m2["paperkiln.ir"][1]) and tensors == t2
    assert (
        raw[offset:] == r2[o2:]
    )  # exactly unchanged F32 tensors and inference semantics
    ember = os.environ.get("EMBER_IR_LOGITS_BIN")
    ember_error = None
    if ember:
        env = {
            **os.environ,
            "EMBER_IR": str(out / "precision.ir.json"),
            "OMP_NUM_THREADS": "1",
        }
        q = subprocess.run(
            [ember, str(g)],
            input=json.dumps([[1, 2, 3]]),
            text=True,
            capture_output=True,
            env=env,
            timeout=120,
        )
        (directory / "ember.json").write_text(q.stdout)
        (directory / "ember.stderr").write_text(q.stderr)
        assert q.returncode == 0, q.stderr
        e = json.loads(q.stdout)[0]
        ember_error = {
            mode: float(np.max(np.abs(np.asarray(e[mode]) - oracle)))
            for mode in ("prefill", "decode")
        }
        assert max(ember_error.values()) < 3e-5
    report = {
        "pass": True,
        "bf16_recipe_score_equals_fp32_score_bitwise": True,
        "fp32_export_tensor_bytes_identical": True,
        "numpy_max_abs": err,
        "ember_max_abs": ember_error,
        "actual_cuda": False,
        "binary_sha256": hashlib.sha256(Path(BIN).read_bytes()).hexdigest(),
    }
    (directory / "replay.json").write_text(json.dumps(report, indent=1) + "\n")


def test_sft_recipe_trains_and_saves(tmp_path):
    proof = os.environ.get("GPT2_PRECISION_PROOF_DIR")
    if proof:
        tmp_path = Path(proof) / "sft_cli"
        tmp_path.mkdir(parents=True, exist_ok=False)
    s = fixture(tmp_path)
    examples = tmp_path / "examples.jsonl"
    examples.write_text(
        "".join(
            json.dumps(
                {
                    "segments": [
                        {"text": "the cat", "loss": False},
                        {"text": "sat", "loss": True},
                    ]
                }
            )
            + "\n"
            for _ in range(20)
        )
    )
    s["data"]["sft_examples"] = str(examples)
    s["data"]["T"] = 8
    s["train"].update(
        objective="sft", attention_precision="bf16", steps=3, eval_every=3
    )
    p = run([BIN, "run", save(tmp_path, s, "sft")], tmp_path, "sft")
    assert p.returncode == 0, p.stdout + p.stderr
    out = Path(s["out_dir"])
    result = json.loads((out / "result.json").read_text())
    assert result["final_step"] == 3 and np.isfinite(result["best_val"])
    assert result["train_attention_precision"] == "bf16"
    assert (out / "precision.safetensors").is_file()

    same = copy.deepcopy(s)
    same["train"].pop("attention_precision")
    fp = save(tmp_path, same, "sft_inference_fp32")
    req = tmp_path / "req.jsonl"
    req.write_text(
        json.dumps({"prompt_ids": [1], "continuation_ids": [2, 3, 4]}) + "\n"
    )
    for label, spec_path in [("sft_recipe", tmp_path / "sft.json"), ("sft_fp32", fp)]:
        scored = run(
            [
                BIN,
                "score",
                spec_path,
                "--in",
                req,
                "--out",
                tmp_path / (label + ".jsonl"),
                "--logits",
                tmp_path / (label + ".f32"),
            ],
            tmp_path,
            label,
        )
        assert scored.returncode == 0, scored.stdout + scored.stderr
    assert (tmp_path / "sft_recipe.f32").read_bytes() == (
        tmp_path / "sft_fp32.f32"
    ).read_bytes()
