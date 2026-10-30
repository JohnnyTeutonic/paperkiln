"""Actual tiny-model probe integration; CPU only, no provider/model API calls."""

import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest
from test_thirdparty_ir import spec_of

BIN = os.environ.get("MTSTUDIO_BIN")
pytestmark = pytest.mark.skipif(not BIN, reason="MTSTUDIO_BIN required")


def invoke(*args):
    return subprocess.run(
        [BIN, *map(str, args)],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=120,
    )


def stage():
    return {
        "max_new_tokens": 3,
        "cases": [
            {
                "id": "needle",
                "kind": "needle",
                "prompt": "the cat",
                "contract": "contains",
                "expected": "cat",
            },
            {
                "id": "behaviour",
                "kind": "behaviour",
                "prompt": "the cat",
                "contract": "not_contains",
                "expected": "not_a_semantic_test",
            },
        ],
    }


@pytest.fixture
def trained(tmp_path):
    path = spec_of(
        tmp_path,
        "probes",
        "llama-tiny",
        {"d": 8, "layers": 1, "heads": 2, "d_ff": 16},
        steps=1,
        formats=("safetensors", "gguf"),
    )
    spec = json.loads(path.read_text())
    spec["evaluation_probes"] = stage()
    path.write_text(json.dumps(spec))
    result = invoke("run", path)
    assert result.returncode == 0, result.stdout + result.stderr
    return path, spec


def test_real_model_probe_replay_identity_and_immutable_output(trained):
    path, spec = trained
    out = Path(spec["out_dir"])
    receipt = json.loads((out / "evaluation_probes.json").read_text())
    assert receipt["status"] == "COMPLETE" and receipt["final_step"] == 1
    assert (
        receipt["checkpoint"]["sha256"]
        == hashlib.sha256((out / "probes.safetensors").read_bytes()).hexdigest()
    )
    assert (
        receipt["tokenizer"]["sha256"]
        == hashlib.sha256(Path(spec["data"]["vocab"]).read_bytes()).hexdigest()
    )
    assert receipt["configuration"] == stage()
    for row in receipt["cases"]:
        assert row["generated_tokens"] == len(row["generated_ids"]) <= 3
        assert row["prompt_tokens"] == len(row["prompt_ids"]) == 2
        assert row["status"] in {"PASS", "FAIL"}
    manual = out / "manual.json"
    result = invoke("probes", path, "--out", manual)
    assert result.returncode == 0, result.stderr
    again = json.loads(manual.read_text())
    assert again["checkpoint"]["sha256"] == receipt["checkpoint"]["sha256"]
    assert again["cases"] == receipt["cases"]
    before = manual.read_bytes()
    assert invoke("probes", path, "--out", manual).returncode != 0
    assert manual.read_bytes() == before
    events = (out / "events.jsonl").read_bytes()
    assert invoke("run", path).returncode != 0
    assert (out / "events.jsonl").read_bytes() == events


@pytest.mark.parametrize(
    "damage",
    [
        "unknown",
        "bool",
        "huge",
        "duplicate",
        "needle_unfounded",
        "negative_needle",
        "formats",
    ],
)
def test_bad_config_refused_before_training(tmp_path, damage):
    path = spec_of(
        tmp_path, "bad", "llama-tiny", {"d": 8, "layers": 1, "heads": 2}, steps=1
    )
    spec = json.loads(path.read_text())
    config = stage()
    if damage == "unknown":
        config["shell"] = "whoami"
    elif damage == "bool":
        config["max_new_tokens"] = True
    elif damage == "huge":
        config["max_new_tokens"] = 4294967297
    elif damage == "duplicate":
        config["cases"].append(config["cases"][0])
    elif damage == "needle_unfounded":
        config["cases"][0]["expected"] = "missing"
    elif damage == "negative_needle":
        config["cases"][0]["contract"] = "not_contains"
    elif damage == "formats":
        spec["export"]["formats"] = ["gguf"]
    spec["evaluation_probes"] = config
    path.write_text(json.dumps(spec))
    assert invoke("run", path).returncode != 0
    assert not (Path(spec["out_dir"]) / "events.jsonl").exists()


def test_oversized_prompt_error_before_training(tmp_path):
    path = spec_of(
        tmp_path, "oversize", "llama-tiny", {"d": 8, "layers": 1, "heads": 2}, steps=1
    )
    spec = json.loads(path.read_text())
    spec["evaluation_probes"] = stage()
    spec["evaluation_probes"]["cases"][0]["prompt"] = "the cat " * 20
    path.write_text(json.dumps(spec))
    result = invoke("run", path)
    assert result.returncode != 0
    out = Path(spec["out_dir"])
    receipt = json.loads((out / "evaluation_probes.json").read_text())
    assert receipt["status"] == "ERROR" and receipt["global_errors"] == 1
    assert not list(out.glob("*.safetensors"))


def test_actual_native_bpe_probe_path(tmp_path):
    import bpe

    corpus = "the red key is amber. the blue key is jade.\n" * 160
    tok = bpe.train(corpus, 300)
    bpe.write_gguf(str(tmp_path / "v.gguf"), tok)
    (tmp_path / "corpus.txt").write_text(corpus)
    spec = {
        "name": "native",
        "arch": {
            "preset": "llama-tiny",
            "custom": {"d": 8, "layers": 1, "heads": 2, "d_ff": 16},
        },
        "data": {
            "vocab": str(tmp_path / "v.gguf"),
            "corpus": str(tmp_path / "corpus.txt"),
            "T": 32,
        },
        "train": {
            "steps": 1,
            "batch": 1,
            "eval_every": 10,
            "checkpoint_every": 10,
            "gradmap_every": 0,
        },
        "export": {"formats": ["safetensors"]},
        "out_dir": str(tmp_path / "native"),
        "evaluation_probes": {
            "max_new_tokens": 4,
            "cases": [
                {
                    "id": "native",
                    "kind": "needle",
                    "prompt": "red key amber",
                    "expected": "amber",
                    "contract": "contains",
                }
            ],
        },
    }
    path = tmp_path / "native.json"
    path.write_text(json.dumps(spec))
    result = invoke("run", path)
    assert (tmp_path / "native/evaluation_probes.json").exists(), (
        result.stdout + result.stderr
    )
    receipt = json.loads((tmp_path / "native/evaluation_probes.json").read_text())
    assert receipt["tokenizer"]["model"] == "bpe"
    row = receipt["cases"][0]
    assert row["prompt_ids"] == tok.encode("red key amber")
    assert row["generated_tokens"] == len(row["generated_ids"]) <= 4
    if row["status"] == "ERROR":
        assert result.returncode != 0 and receipt["summary"]["errors"] == 1
        assert row.get("generated_bytes_hex") or row.get("error")
    else:
        assert result.returncode == 0
        assert row["generated_text"] == tok.decode(row["generated_ids"])


def test_manual_resolved_embedded_vocab_identity(trained):
    path, spec = trained
    vocab = Path(spec["data"]["vocab"])
    hidden = vocab.with_suffix(".hidden")
    vocab.rename(hidden)
    try:
        output = Path(spec["out_dir"]) / "fallback.json"
        result = invoke("probes", path, "--out", output)
        assert result.returncode == 0, result.stderr
        report = json.loads(output.read_text())
        actual = Path(spec["out_dir"]) / "probes.gguf"
        assert report["tokenizer"]["source"] == str(actual)
        assert (
            report["tokenizer"]["sha256"]
            == hashlib.sha256(actual.read_bytes()).hexdigest()
        )
    finally:
        hidden.rename(vocab)


def test_http_invalid_probe_preserves_armed_spec(trained):
    import socket
    import time
    import urllib.error
    import urllib.request

    path, spec = trained
    out = Path(spec["out_dir"])
    armed = out / "train_spec.json"
    armed.write_bytes(path.read_bytes())
    before = armed.read_bytes()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = dict(
        os.environ,
        MTSTUDIO_UI=str(Path(__file__).resolve().parents[1] / "studio/index.html"),
    )
    server = subprocess.Popen(
        [BIN, "serve", str(out), str(port), str(path)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(100):
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/evaluation-probes", timeout=1
                ) as response:
                    assert json.load(response)["status"] == "COMPLETE"
                break
            except OSError:
                time.sleep(0.03)
        else:
            pytest.fail("server did not start")
        bad = dict(spec)
        bad["evaluation_probes"] = {**stage(), "max_new_tokens": True}
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/train",
            data=json.dumps(bad).encode(),
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request, timeout=2)
        assert error.value.code == 400
        assert armed.read_bytes() == before
    finally:
        server.terminate()
        server.wait(timeout=10)


def test_manual_nonfinite_checkpoint_records_errors(trained):
    import struct

    path, spec = trained
    out = Path(spec["out_dir"])
    weights = out / "probes.safetensors"
    data = bytearray(weights.read_bytes())
    offset = 8 + struct.unpack("<Q", data[:8])[0]
    data[offset:] = struct.pack("<f", float("nan")) * ((len(data) - offset) // 4)
    weights.write_bytes(data)
    target = out / "nonfinite.json"
    result = invoke("probes", path, "--out", target)
    assert result.returncode != 0
    report = json.loads(target.read_text())
    assert report["status"] == "COMPLETE_WITH_ERRORS"
    assert report["summary"] == {"total": 2, "passed": 0, "failed": 0, "errors": 2}
    assert all(
        r["status"] == "ERROR" and "nonfinite" in r["error"] for r in report["cases"]
    )


def test_manual_applies_declared_execution_settings(trained):
    path, spec = trained
    spec["train"]["precision"] = "bf16"
    spec["train"]["attention_kernel"] = "flash"
    path.write_text(json.dumps(spec))
    output = Path(spec["out_dir"]) / "recipe.json"
    result = invoke("probes", path, "--out", output)
    assert result.returncode == 0, result.stdout + result.stderr
    generation = json.loads(output.read_text())["generation"]
    assert generation["requested_precision"] == "bf16"
    assert generation["requested_attention_kernel"] == "flash"
    assert generation["runtime_attention_kernel_setting"] == "flash"
    assert generation["execution_device"] == "cpu"
