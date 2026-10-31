"""Plain Flex tied heads map strictly and survive actual Transformers loading."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from test_hf_export import flex_gpt2_fixture, flex_llama_fixture
import hf_export


@pytest.mark.parametrize("fixture", [flex_gpt2_fixture, flex_llama_fixture])
def test_tied_inventory_and_aliases(fixture):
    model, state = fixture()
    model["tie_head"] = True
    state.pop("head.weight")
    got = hf_export.map_weights(state, model)
    embed = (
        "transformer.wte.weight"
        if hf_export.hf_model_type(model) == "gpt2"
        else "model.embed_tokens.weight"
    )
    np.testing.assert_array_equal(got["lm_head.weight"], got[embed])
    for invalid in [None, 0, 1, "true", [], {}]:
        with pytest.raises(ValueError):
            hf_export.map_weights(state, {**model, "tie_head": invalid})
    for invalid in [False, 1, "true", None]:
        with pytest.raises(ValueError):
            hf_export.map_weights(state, {**model, "tie_embeddings": invalid})
    hf_export.map_weights(state, {**model, "tie_embeddings": True})
    for bad in [
        dict(
            state,
            **{"head.weight": np.zeros((model["d"], model["vocab"]), dtype=np.float32)},
        ),
        {k: v for k, v in state.items() if k != "wte.weight"},
    ]:
        with pytest.raises(ValueError):
            hf_export.map_weights(bad, model)
    for extra in [
        {"embed": {"expr": "expr:x*2"}},
        {"embed_before_positions": True},
        {"activation": "gelu_erf"},
        {"qk_norm": "full"},
        {"bitlinear": True},
        {"norm_placement": {"attn": "post", "ffn": "post"}},
    ]:
        with pytest.raises(ValueError):
            hf_export.map_weights(state, {**model, **extra})


@pytest.mark.parametrize("flavour", ["gpt2", "llama"])
def test_actual_trained_tied_flex_transformers_roundtrip(tmp_path, flavour):
    binary = os.environ.get("MTSTUDIO_BIN")
    if not binary:
        pytest.skip("MTSTUDIO_BIN required for real native model comparison")
    pytest.importorskip("transformers")
    from test_thirdparty_ir import spec_of

    root = Path(__file__).resolve().parents[1]
    custom = {
        "d": 16,
        "layers": 2,
        "heads": 2,
        "d_ff": 24,
        "tie_head": True,
        "norm": "layernorm" if flavour == "gpt2" else "rmsnorm",
        "activation": "gelu" if flavour == "gpt2" else "swiglu",
        "position": "learned" if flavour == "gpt2" else "rope",
    }
    path = spec_of(
        tmp_path, "tied", "gpt2-nano", custom, steps=3, formats=("safetensors", "gguf")
    )
    out = tmp_path / "tied"

    def run(*args):
        p = subprocess.run(
            list(map(str, args)),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=120,
        )
        assert p.returncode == 0, p.stdout + p.stderr
        return p

    before = hashlib.sha256(Path(binary).read_bytes()).hexdigest()
    run(binary, "run", path)
    run(sys.executable, root / "tools/hf_export.py", out)
    config = json.loads((out / "hf/config.json").read_text())
    assert config["tie_word_embeddings"] is True
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        out / "hf", attn_implementation="eager"
    )
    assert (
        model.get_input_embeddings().weight.data_ptr()
        == model.get_output_embeddings().weight.data_ptr()
    )
    for i, prompt in enumerate(["the cat", "cat .\nthe", "the unknown café"]):
        receipt = tmp_path / f"{flavour}-{i}.json"
        run(
            sys.executable,
            root / "tools/hf_export_verify.py",
            out / "hf",
            out,
            "--spec",
            path,
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
    assert hashlib.sha256(Path(binary).read_bytes()).hexdigest() == before
