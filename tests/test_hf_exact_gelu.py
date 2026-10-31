"""Only explicitly recorded erf-GELU Flex maps to HF exact GELU."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from test_hf_export import flex_gpt2_fixture
import hf_export


@pytest.mark.parametrize("bad", [None, "tanh", "exact", 0, False, {}, []])
def test_exact_gelu_requires_explicit_none_approximation(bad):
    m, _ = flex_gpt2_fixture()
    m.update(activation="gelu_erf", gelu_approx=bad)
    with pytest.raises(ValueError):
        hf_export.validate_gpt2(m)


def test_exact_gelu_missing_unknown_activation_and_tanh_default():
    m, _ = flex_gpt2_fixture()
    hf_export.validate_gpt2(m)
    with pytest.raises(ValueError):
        hf_export.validate_gpt2({**m, "activation": "gelu_erf"})
    for activation in ["erf", "gelu_exact", "unknown", None, 0]:
        with pytest.raises(ValueError):
            hf_export.validate_gpt2(
                {**m, "activation": activation, "gelu_approx": "none"}
            )
    with pytest.raises(ValueError):
        hf_export.validate_gpt2({**m, "gelu_approx": "none"})
    hf_export.validate_gpt2({**m, "activation": "gelu_erf", "gelu_approx": "none"})


@pytest.mark.parametrize("tied", [False, True])
def test_actual_exact_gelu_flex_transformers_roundtrip(tmp_path, tied):
    binary = os.environ.get("MTSTUDIO_BIN")
    if not binary:
        pytest.skip("MTSTUDIO_BIN required for actual CPU parity")
    pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM
    from test_thirdparty_ir import spec_of

    root = Path(__file__).resolve().parents[1]
    custom = {
        "d": 16,
        "layers": 2,
        "heads": 2,
        "d_ff": 24,
        "tie_head": tied,
        "norm": "layernorm",
        "activation": "gelu_erf",
        "position": "learned",
    }
    spec = spec_of(
        tmp_path, "erf", "gpt2-nano", custom, steps=3, formats=("safetensors", "gguf")
    )
    out = tmp_path / "erf"

    def run(*args):
        p = subprocess.run(
            list(map(str, args)),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=120,
        )
        assert p.returncode == 0, p.stdout + p.stderr

    run(binary, "run", spec)
    events = [
        json.loads(line) for line in (out / "events.jsonl").read_text().splitlines()
    ]
    model_event = next(e for e in events if e.get("event") == "model")
    assert (
        model_event["family"] == "flex"
        and model_event["activation"] == "gelu_erf"
        and model_event["gelu_approx"] == "none"
    )
    run(sys.executable, root / "tools/hf_export.py", out)
    config = json.loads((out / "hf/config.json").read_text())
    assert (
        config["activation_function"] == "gelu"
        and config["tie_word_embeddings"] is tied
    )
    loaded = AutoModelForCausalLM.from_pretrained(
        out / "hf", attn_implementation="eager"
    )
    assert (
        loaded.get_input_embeddings().weight.data_ptr()
        == loaded.get_output_embeddings().weight.data_ptr()
    ) is tied
    for i, prompt in enumerate(["the cat", "cat .\nthe", "the unknown café"]):
        receipt = tmp_path / f"erf-{i}.json"
        run(
            sys.executable,
            root / "tools/hf_export_verify.py",
            out / "hf",
            out,
            "--spec",
            spec,
            "--prompt",
            prompt,
            "--tokens",
            8,
            "--mtstudio",
            binary,
            "--receipt",
            receipt,
        )
        run(sys.executable, root / "tools/hf_export_verify.py", "--replay", receipt)
