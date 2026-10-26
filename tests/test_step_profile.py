"""tools/step_profile.py: the per-class breakdown, the config refusal and the resumable profile.json,
against a fake mtstudio that writes step events the way tools/mtstudio.cpp does."""
import json
import os
import struct
import sys
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import step_profile as sp  # noqa: E402

FAKE = r'''#!/usr/bin/env python3
import json, os, sys
spec = json.load(open(sys.argv[2]))
out = spec["out_dir"]
os.makedirs(out, exist_ok=True)
tr = spec["train"]
model = {"event": "model"}
if os.environ.get("FAKE_RECORDS") == "1":
    for k in ("precision", "attention_kernel", "fused_head"):
        if k in tr:
            model[k] = tr[k]
prof = os.environ.get("MICROTORCH_GPU_PROFILE") == "1"
with open(os.path.join(out, "events.jsonl"), "w") as f:
    f.write(json.dumps(model) + "\n")
    for s in range(1, tr["steps"] + 1):
        e = {"event": "step", "step": s, "loss": 9.0 - 0.01 * s}
        if prof:
            e["t_ms"] = {"forward": 40.0, "backward": 50.0, "optimizer": 10.0, "other": 500.0}
            e["gpu_ms"] = {"gemm": 50.0, "attn_heads_bwd": 20.0, "rmsnorm_fwd": 5.0, "mystery_op": 5.0}
            e["peak_mb"] = 1234.0
        line = json.dumps(e)
        f.write(line + "\n")
        print(line, flush=True)
'''


@pytest.fixture
def fake(tmp_path, monkeypatch):
    b = tmp_path / "mtstudio"
    b.write_text(FAKE)
    b.chmod(0o755)
    monkeypatch.setattr(sp, "BIN", str(b))
    if os.name == "nt":  # no shebang execution on Windows: run the fake through this interpreter
        real = sp.subprocess.Popen

        def popen(argv, **kw):
            return real([sys.executable] + list(argv), **kw)
        monkeypatch.setattr(sp.subprocess, "Popen", popen)
    return b


def test_synth_corpus_header(tmp_path):
    p = sp.synth_corpus(str(tmp_path / "s.ids"), vocab=16384, n=1000)
    raw = open(p, "rb").read()
    assert raw[:8] == b"PKIDS001"
    width, nv, count = struct.unpack("<IIQ", raw[8:24])
    assert (width, nv, count) == (2, 16384, 1000)
    ids = struct.unpack("<1000H", raw[24:])
    assert max(ids) < 16384 and len(raw) == 24 + 2000


def test_breakdown_classes_and_idle():
    evs = [{"event": "step", "step": s, "t_ms": {"forward": 60.0, "backward": 40.0, "other": 900.0},
            "gpu_ms": {"gemm": 30.0, "attn_heads_fwd": 10.0, "axpy": 5.0, "new_kernel": 5.0}} for s in range(1, 21)]
    b = sp.breakdown(evs, warm=10)
    assert b["steps"] == 10
    assert b["wall_ms"] == pytest.approx(100.0)  # "other" (eval, logging) is outside the step
    assert b["gpu_ms"] == pytest.approx(50.0) and b["idle_ms"] == pytest.approx(50.0)
    assert b["class_ms"] == {"attention_fwd": 10.0, "elementwise": 5.0, "gemm": 30.0, "other": 5.0}


def test_run_profiles_and_refuses_unrecorded_config(tmp_path, fake, monkeypatch):
    monkeypatch.delenv("FAKE_RECORDS", raising=False)
    a = types.SimpleNamespace(out=str(tmp_path / "o"), vocab="v.gguf", shapes="20m_L2", configs="fp32,bf16",
                              steps=12, warm=4)
    assert sp.cmd_run(a) == 0
    res = json.load(open(tmp_path / "o" / "profile.json"))
    fp = res["runs"]["20m_L2/fp32"]
    assert fp["profile"]["class_ms"]["gemm"] == pytest.approx(50.0)
    assert fp["profile"]["idle_ms"] == pytest.approx(20.0) and fp["profile"]["peak_mb"] == 1234.0
    assert fp["tokens_per_step"] == 16384 and len(fp["losses"]) == 12
    # A binary that ignores train.precision would time FP32 under the name bf16: refused.
    assert "refused" in res["runs"]["20m_L2/bf16"] and "profile" not in res["runs"]["20m_L2/bf16"]
    spec = json.load(open(tmp_path / "o" / "runs" / "20m_L2_bf16_clean_spec.json"))
    assert spec["train"]["precision"] == "bf16" and spec["export"] == {"formats": []}


def test_run_resumes_and_accepts_recorded_config(tmp_path, fake, monkeypatch):
    a = types.SimpleNamespace(out=str(tmp_path / "o"), vocab="v.gguf", shapes="pk_base_v1", configs="bf16_flash",
                              steps=12, warm=4)
    monkeypatch.setenv("FAKE_RECORDS", "1")
    assert sp.cmd_run(a) == 0
    res = json.load(open(tmp_path / "o" / "profile.json"))
    cell = res["runs"]["pk_base_v1/bf16_flash"]
    assert cell["profile"] and cell["tokens_per_step"] == 16 * 512 and cell["arch"]["d"] == 768
    monkeypatch.setattr(sp, "run_once", lambda *a, **k: (_ for _ in ()).throw(AssertionError("reran a done cell")))
    assert sp.cmd_run(a) == 0
    assert "| pk_base_v1 | bf16_flash |" in sp.table(res)


def test_fused_head_config_is_refused_unless_recorded(tmp_path, fake, monkeypatch):
    a = types.SimpleNamespace(out=str(tmp_path / "o"), vocab="v.gguf", shapes="20m_L4", configs="fused_head",
                              steps=12, warm=4)
    monkeypatch.delenv("FAKE_RECORDS", raising=False)
    assert sp.cmd_run(a) == 0
    assert "refused" in json.load(open(tmp_path / "o" / "profile.json"))["runs"]["20m_L4/fused_head"]
    spec = json.load(open(tmp_path / "o" / "runs" / "20m_L4_fused_head_clean_spec.json"))
    assert spec["train"]["fused_head"] is True
    b = types.SimpleNamespace(**dict(vars(a), out=str(tmp_path / "p")))
    monkeypatch.setenv("FAKE_RECORDS", "1")
    assert sp.cmd_run(b) == 0
    assert json.load(open(tmp_path / "p" / "profile.json"))["runs"]["20m_L4/fused_head"]["profile"]
    assert sp.OP_CLASS["linear_ce"] == "head_fused"
