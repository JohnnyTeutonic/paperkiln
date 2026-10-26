"""Tests of gguf_to_hf.py: the layout transforms are exact inverses, every suite file of a supported
architecture round-trips GGUF -> HF -> GGUF bit for bit (Mamba's A within one ulp, see below), and, where
transformers is installed, each export loads with no missing or unexpected keys and matches the reference.

    python -m pytest adapters/common -q
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "tools"))
import gguf_to_hf as g2h  # noqa: E402

MODELS = ROOT / "conformance" / "models"


def suite_files():
    out = []
    for d in sorted(MODELS.iterdir()):
        g = d / (d.name + ".gguf")
        if g.is_file():
            meta, _ = g2h.read(g)
            out.append((d.name, g, meta.get("general.architecture")))
    return out


SUPPORTED = [(n, g) for n, g, a in suite_files() if a in g2h.FAMILIES]


def test_some_of_every_family():
    arches = {a for _, _, a in suite_files() if a in g2h.FAMILIES}
    assert arches >= {"llama", "mistral", "qwen3", "gemma3", "gptneox", "gpt2", "mamba"}


@pytest.mark.parametrize("n_head,kv", [(4, 4), (4, 2), (2, 1)])
def test_rope_permutation_inverse(n_head, kv):
    rng = np.random.default_rng(1)
    hd, d = 8, 16
    for heads in (n_head, kv):
        w = rng.standard_normal((heads * hd, d)).astype(np.float32)
        assert np.array_equal(g2h.permute_rope(g2h.unpermute_rope(w, heads), heads), w)
        assert np.array_equal(g2h.unpermute_rope(g2h.permute_rope(w, heads), heads), w)


def test_rope_permutation_meaning():
    """Interleaved pairs (2j, 2j+1) of llama.cpp's rows become HF's half-split pairs (j, j + hd/2)."""
    hd = 8
    w = np.arange(hd, dtype=np.float32)[:, None]  # one head, row r holds r
    u = g2h.unpermute_rope(w, 1)[:, 0]
    assert list(u) == [0, 2, 4, 6, 1, 3, 5, 7]


def test_qkv_interleave_inverse_and_meaning():
    rng = np.random.default_rng(2)
    H, hd, d = 2, 4, 8
    w = rng.standard_normal((3 * H * hd, d)).astype(np.float32)
    assert np.array_equal(g2h.deinterleave_qkv(g2h.interleave_qkv(w, H), H), w)
    b = np.arange(3 * H * hd, dtype=np.float32)  # q rows 0..7, k 8..15, v 16..23
    hf = g2h.interleave_qkv(b, H)
    assert list(hf[:12]) == [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19]  # head 0: q0 k0 v0


@pytest.mark.parametrize("name,gguf", SUPPORTED, ids=[n for n, _ in SUPPORTED])
def test_round_trip(name, gguf):
    meta, W = g2h.read(gguf)
    bad = g2h.round_trip(gguf)
    if meta["general.architecture"] != "mamba":
        assert bad == {}, bad
        return
    # HF stores A_log; -exp(A_log) cannot hit every float32 A (exp spreads neighbouring A_log values more than
    # an ulp of A apart where |A| > 1). The exporter picks the exact A_log where one exists, the nearest
    # otherwise: only ssm_a may differ, by at most one ulp.
    assert set(bad) <= {k for k in W if k.endswith("ssm_a")}, bad
    _, _, c, S, _ = g2h.convert(meta, W)
    back = g2h.back(meta, c, S)
    for k in bad:
        ulps = np.abs(back[k].view(np.int32).astype(np.int64) - W[k].view(np.int32).astype(np.int64))
        assert int(ulps.max()) <= 1


def test_a_log_exact_where_reachable():
    a = -np.exp(np.linspace(-1.0, 1.0, 101)).astype(np.float32)
    al, miss = g2h.a_log_of(a)
    hit = (-np.exp(al)).astype(np.float32) == a
    assert int(np.sum(~hit)) == miss
    # float32 exp is neither correctly rounded nor onto, so some A have no exact A_log: the nearest is one ulp off
    err = np.abs((-np.exp(al)).astype(np.float32).view(np.int32).astype(np.int64) - a.view(np.int32).astype(np.int64))
    assert int(err.max()) <= 1 and miss < len(a) // 4


def test_gemma_norm_offset_exact():
    _, W = g2h.read(MODELS / "tp_gemma3" / "tp_gemma3.gguf")
    for k, v in W.items():
        if k.endswith("norm.weight"):
            assert np.all((v >= 0.5) & (v <= 2.0)), k  # Sterbenz: s - 1 exact, so 1 + (s - 1) == s
            assert np.array_equal((np.float32(1) + (v - np.float32(1))).astype(np.float32), v)


def test_unsupported():
    with pytest.raises(g2h.Unsupported):
        g2h.convert({"general.architecture": "paperkiln"}, {})


@pytest.mark.parametrize("name,gguf", SUPPORTED, ids=[n for n, _ in SUPPORTED])
def test_loads_in_transformers(name, gguf, tmp_path):
    torch = pytest.importorskip("torch")
    tf = pytest.importorskip("transformers")
    import ir_conformance as rc
    import ir_numpy
    g2h.export(gguf, tmp_path)
    torch.set_num_threads(2)
    m, info = tf.AutoModelForCausalLM.from_pretrained(str(tmp_path), dtype=torch.float32, output_loading_info=True)
    assert not any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys")), info
    rep = json.loads((tmp_path / "keramos_export.json").read_text(encoding="utf-8"))
    assert set(rep["round_trip_not_exact"]) <= {k for k in rep["round_trip_not_exact"] if k.endswith("ssm_a")}
    doc, W, _, _ = rc.load_model(gguf.parent)
    meta, _ = g2h.read(gguf)
    nodes = doc["block"]["nodes"]
    if {n["op"] for n in nodes} & {"linear_attention", "recurrence"} or (
            meta["general.architecture"] == "gpt2" and any(n.get("window") or n.get("sinks") for n in nodes)):
        pytest.skip("the file's document says more than its architecture (IR-only semantics: llama.cpp K1)")
    ids = [int(x) for x in np.random.default_rng(3).integers(1, doc["model"]["vocab"], size=8)]
    with torch.no_grad():
        got = m(input_ids=torch.tensor([ids])).logits[0].float().numpy()
    ref = ir_numpy.forward(doc, W, ids)
    ok, msg = rc.close(got, ref, "ROUNDING", 1e-5)
    assert ok, msg
