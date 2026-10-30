"""CLI integration of the merged BitLinear/untied-head architecture flags."""
import json
from pathlib import Path
import struct
import subprocess

import pytest

from test_thirdparty_ir import BIN, events, run, spec_of

pytestmark = pytest.mark.skipif(BIN is None, reason="set MTSTUDIO_BIN")


def test_bitlinear_cli_records_modifier_without_misleading_gguf(tmp_path):
    spec = spec_of(tmp_path, "bit", "llama-tiny",
                   {"d": 16, "layers": 1, "heads": 2, "d_ff": 32, "bitlinear": {}}, steps=2)
    run(spec)
    rows = events(spec)
    model = next(row for row in rows if row["event"] == "model")
    assert model["bitlinear"] == {"weights": "ternary", "act_bits": 8, "in_norm": True}
    skipped = [row for row in rows if row["event"] == "export_skipped"]
    assert any("BitLinear" in row["reason"] for row in skipped)
    out = Path(json.loads(spec.read_text())["out_dir"])
    assert list(out.glob("*.safetensors"))
    assert not list(out.glob("*.gguf")) and not list(out.glob("*.ir.json"))


def test_untied_llama_cli_saves_distinct_head(tmp_path):
    spec = spec_of(tmp_path, "untied", "llama-tiny",
                   {"d": 16, "layers": 1, "heads": 2, "d_ff": 32, "tie_embeddings": False},
                   steps=2, formats=("safetensors",))
    run(spec)
    model = next(row for row in events(spec) if row["event"] == "model")
    assert model["tie_embeddings"] is False
    out = Path(json.loads(spec.read_text())["out_dir"])
    weights = next(out.glob("*.safetensors"))
    with weights.open("rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(length))
    assert "lm_head.weight" in header
    assert header["lm_head.weight"]["shape"][0] == 16


@pytest.mark.parametrize("preset,bitlinear,message", [
    ("llama-tiny", {"act_bits": 1}, "act_bits must be 0 or in 2..8"),
    ("llama-tiny", {"weights": "int4"}, "weights must be"),
    ("llama-tiny", {"unknown": 1}, "unknown key unknown"),
    ("gpt2-nano", {}, "bitlinear and tie_embeddings need the llama family"),
])
def test_bitlinear_cli_refuses_unsupported_settings(tmp_path, preset, bitlinear, message):
    spec = spec_of(tmp_path, "bad", preset,
                   {"d": 16, "layers": 1, "heads": 2, "bitlinear": bitlinear}, steps=2)
    result = subprocess.run([BIN, "plan", str(spec)], capture_output=True, text=True, timeout=60)
    assert result.returncode != 0 and message in result.stderr
