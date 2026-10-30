"""HF interop must not mistake training master weights for effective forward weights."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import hf_export as export  # noqa: E402


def model(family="llama"):
    return {"family": family, "d": 4, "heads": 2, "layers": 2, "vocab": 8, "T": 16,
            "rope_heads": "all", "norm": "rmsnorm", "activation": "swiglu", "position": "rope"}


@pytest.mark.parametrize("value", [{"weights": "ternary", "act_bits": 8, "in_norm": True},
                                    {"weights": "ternary", "act_bits": 0, "in_norm": False},
                                    {}, None, True, 0, "ternary"])
@pytest.mark.parametrize("family", ["llama", "gpt2", "flex"])
def test_bitlinear_active_or_malformed_never_maps_master_weights(value, family):
    record = {**model(family), "bitlinear": value}
    with pytest.raises(ValueError, match="BitLinear"):
        export.map_weights({}, record)
    for validate in (export.validate_model, export.validate_gpt2, export.validate_flex_llama):
        with pytest.raises(ValueError, match="BitLinear"):
            validate(record)


@pytest.mark.parametrize("value", [1., 0.5, -1, float("nan"), float("inf"), None, "0", False, {}])
@pytest.mark.parametrize("nested", [False, True])
def test_temperature_nonzero_or_malformed_never_maps_master_weights(value, nested):
    record = model()
    if nested:
        record["thirdparty"] = {"qk_temp": value}
    else:
        record["qk_temp"] = value
    with pytest.raises(ValueError, match="QK-temperature"):
        export.map_weights({}, record)
    with pytest.raises(ValueError, match="QK-temperature"):
        export.validate_model(record)


@pytest.mark.parametrize("extra", [{}, {"bitlinear": False}, {"qk_temp": 0}, {"qk_temp": 0.0},
                                   {"tie_head": True, "tie_embeddings": True}])
def test_plain_absent_or_explicit_disabled_defaults_keep_original_mapping(extra):
    record = {**model(), **extra}
    export.validate_model(record)
    assert export.hf_model_type(record) == "llama"
    assert export.map_weights({}, record) == {}


@pytest.mark.parametrize("extra", [{"tie_embeddings": False}, {"tie_head": False},
                                   {"tie_embeddings": True, "tie_head": False}])
def test_native_head_alias_cannot_silently_hide_untied_or_conflicting_semantics(extra):
    with pytest.raises(ValueError, match="untied or conflicting"):
        export.map_weights({}, {**model(), **extra})


@pytest.mark.parametrize("modifier", [{"bitlinear": {"weights": "ternary", "act_bits": 8, "in_norm": True}},
                                      {"thirdparty": {"qk_norm": True, "qk_temp": 1.0}}])
def test_actual_recorded_modifier_survives_event_parse_and_cli_refuses_before_stale_gguf(modifier, tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    record = {**model(), **modifier, "event": "model"}
    (run / "events.jsonl").write_text(json.dumps({"event": "start", "name": "toy"}) + "\n" + json.dumps(record) + "\n")
    assert export.model_event(run) == record
    (run / "toy.gguf").write_bytes(b"stale pre-quantization model file")
    target = run / "hf"
    target.mkdir()
    for name in export.EXPORT_FILES:
        (target / name).write_bytes(("previous:" + name).encode())
    before = {p.name: p.read_bytes() for p in target.iterdir()}
    def stale_read(*args):
        pytest.fail("unsupported model reached stale GGUF/tokenizer read")
    monkeypatch.setattr(export, "gguf_tokens", stale_read)
    monkeypatch.setattr(sys, "argv", ["hf_export.py", str(run), "--overwrite"])
    with pytest.raises(ValueError, match="BitLinear|QK-temperature"):
        export.main()
    assert {p.name: p.read_bytes() for p in target.iterdir()} == before
    assert not list(run.glob(".hf.export-*"))


@pytest.mark.parametrize("value", [{"weights": "ternary"}, {}])
def test_nested_bitlinear_metadata_also_refuses_direct_mapping(value):
    with pytest.raises(ValueError, match="BitLinear"):
        export.map_weights({}, {**model(), "thirdparty": {"bitlinear": value}})
