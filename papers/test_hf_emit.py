"""Installed-package HF config emission must not silently replace known mechanisms."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paperkiln_fetch/src"))
from paperkiln_fetch import Arch, Finding  # noqa: E402
from paperkiln_fetch.hf_emit import build_hf_config  # noqa: E402


def paper(**fields):
    base = {"d_model": 24, "n_layers": 2, "n_heads": 4, "d_ff": 48,
            "vocab_size": 64, "context_length": 32}
    base.update(fields)
    return Arch("fixture", fields={key: Finding(value, f"quoted {key}={value}",
                                               verdict="used" if isinstance(value, str) else None)
                                   for key, value in base.items()})


def test_llama_dimensions_gqa_and_quoted_provenance():
    arch = paper(norm="rmsnorm", positional="rope", activation="swiglu", n_kv_heads=2)
    before = copy.deepcopy(arch)
    cfg, family, defaults = build_hf_config(arch)
    assert family == "llama" and cfg["num_key_value_heads"] == 2
    assert cfg["hidden_act"] == "silu"
    assert cfg["_paperkiln"]["fields"]["n_kv_heads"]["evidence"] == "quoted n_kv_heads=2"
    assert defaults["rms_norm_eps"] == 1e-5 and defaults["rope_theta"] == 10000
    assert arch == before


@pytest.mark.parametrize("activation,hf_name", [("relu", "relu"), ("gelu", "gelu")])
def test_gpt2_activation_is_not_silently_replaced(activation, hf_name):
    cfg, family, _ = build_hf_config(paper(activation=activation))
    assert family == "gpt2" and cfg["activation_function"] == hf_name


def test_geglu_uses_gelu_gate_not_silu():
    cfg, _, _ = build_hf_config(paper(norm="rmsnorm", positional="rope", activation="geglu"))
    assert cfg["hidden_act"] == "gelu"


@pytest.mark.parametrize("fields", [
    {"norm": "rmsnorm", "positional": "learned", "activation": "swiglu"},
    {"norm": "layernorm", "positional": "rope", "activation": "swiglu"},
    {"positional": "rope"}, {"activation": "swiglu"},
    {"attention": "swa"}, {"residual": "highway"}, {"window": 16}, {"n_sinks": 4},
    {"n_kv_heads": 2},
])
def test_unsupported_known_mechanisms_refuse(fields):
    with pytest.raises(ValueError, match="cannot represent"):
        build_hf_config(paper(**fields))


@pytest.mark.parametrize("key,value", [("d_model", 0), ("n_heads", True),
                                       ("d_ff", -1), ("n_layers", 2.5),
                                       ("vocab_size", "64"), ("context_length", 0)])
def test_invalid_dimensions_refuse(key, value):
    with pytest.raises(ValueError, match="positive integer"):
        build_hf_config(paper(**{key: value}))


@pytest.mark.parametrize("fields", [{"d_model": 25},
                                   {"norm": "rmsnorm", "positional": "rope", "n_kv_heads": 3},
                                   {"norm": "rmsnorm", "positional": "rope", "d_model": 20}])
def test_invalid_head_geometry_refuses(fields):
    with pytest.raises(ValueError, match="divisible|even head"):
        build_hf_config(paper(**fields))


@pytest.mark.parametrize("verdict", ["contested", "family", "unverified", "quote_only", "rejected"])
def test_unverified_values_retained_as_evidence_but_not_applied(verdict):
    arch = paper()
    arch.fields["activation"] = Finding("swiglu", "not established", verdict=verdict)
    cfg, family, defaults = build_hf_config(arch)
    assert family == "gpt2" and defaults["activation"] == "gelu"
    assert cfg["_paperkiln"]["fields"]["activation"]["verdict"] == verdict


def test_missing_architecture_assumptions_are_declared():
    cfg, _, defaults = build_hf_config(paper())
    assert defaults["norm"] == "layernorm" and defaults["positional"] == "learned"
    assert defaults["activation"] == "gelu"
    assert cfg["_paperkiln"]["defaults_applied"] == defaults


def test_cli_refuses_incompatible_mechanism_before_writing(tmp_path):
    import os
    tex, out = tmp_path / "paper.tex", tmp_path / "config.json"
    tex.write_text("We use sliding-window attention with a window of 16 tokens.")
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "paperkiln_fetch/src")}
    run = subprocess.run([sys.executable, "-m", "paperkiln_fetch.cli", "fixture", "--tex", str(tex),
                          "--emit-hf", str(out)], env=env, capture_output=True, text=True)
    assert run.returncode == 2 and "cannot represent" in run.stderr
    assert not out.exists()


def test_serialized_configs_load_with_transformers(tmp_path):
    transformers = pytest.importorskip("transformers")
    for name, arch in (("gpt2", paper()),
                       ("llama", paper(norm="rmsnorm", positional="rope", activation="swiglu"))):
        cfg, family, _ = build_hf_config(arch)
        folder = tmp_path / name
        folder.mkdir()
        (folder / "config.json").write_text(json.dumps(cfg))
        loaded = transformers.AutoConfig.from_pretrained(folder, local_files_only=True)
        assert loaded.model_type == family
        assert loaded._paperkiln["fields"]["d_model"]["value"] == 24
