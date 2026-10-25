"""The IR conformance suite (conformance/, IR_SPEC.md section 11) and its runner
(tools/ir_conformance.py): the tensor hex format round-trips bit for bit; the GGUF writer's files read back
through tools/ir_validate.py; the eviction rule's legal row sets (6.8.3) on a worked sequence; every case
passes against the numpy reference (a FAIL is a regression of the reference or of the suite)."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import ir_conformance as conf  # noqa: E402
import ir_validate  # noqa: E402

HAVE_SUITE = (conf.SUITE / "manifest.json").is_file()


def test_hex_round_trip_keeps_every_bit():
    a = np.array([[1.0, -0.0, np.inf, -np.inf], [np.float32(1e-45), 3.4e38, np.nan, 0.1]], np.float32)
    t = conf.enc(a)
    assert t["f32"][:8] == "3f800000" and t["f32"][8:16] == "80000000"
    assert np.array_equal(conf.dec(t).view(np.uint32), a.view(np.uint32))


def test_exact_compares_bits_but_any_nan_matches():
    a = np.array([0.0, np.nan], np.float32)
    assert conf.close(a, np.array([0.0, -np.nan], np.float32), "EXACT", 0)[0]
    assert not conf.close(np.array([-0.0], np.float32), np.array([0.0], np.float32), "EXACT", 0)[0]


def test_gguf_writer_reads_back(tmp_path):
    doc = json.loads((ROOT / "examples" / "chat_sftfinal.ir.json").read_text(encoding="utf-8"))
    W = {"a.weight": np.arange(6, dtype=np.float32).reshape(2, 3), "b": np.ones(5, np.float32)}
    p = tmp_path / "m.gguf"
    conf.write_gguf(p, {"general.architecture": (8, "paperkiln"), "x.n": (4, 7),
                        "tokenizer.ggml.tokens": (9, ["<unk>", "a"])}, W, doc)
    meta, tensors, data_start, raw, _ = ir_validate.read_gguf(p)
    assert meta["x.n"] == (4, 7) and meta["tokenizer.ggml.tokens"][1] == ["<unk>", "a"]
    assert json.loads(meta["paperkiln.ir"][1]) == doc and meta["paperkiln.ir_version"] == (4, 0)
    assert list(meta)[-2:] == ["paperkiln.ir", "paperkiln.ir_version"]
    dims = {t[0]: t[1] for t in tensors}
    assert dims == {"a.weight": [3, 2], "b": [5]}


def test_eviction_rows_eager_and_lazy():
    V, k = 10, 2
    S0, S1 = V, V + 1
    ids = [2, 5, 3, 6, 1, S0, S1]  # one exchange and its block: the call ends where block 0 completes
    allowed = conf.evict_allowed(ids, V, k)
    assert allowed == [[5, 6], [0, 1, 2, 3, 4, 5, 6]]  # eager: the block; lazy: segment 0's text too
    allowed = conf.evict_allowed(ids + [2], V, k)  # the next text position: eager only
    assert allowed == [[5, 6, 7]]
    allowed = conf.evict_allowed([2, 5, 3, 6, 1, S0], V, k)  # a block left open: nothing completes
    assert allowed == [list(range(6))]


@pytest.mark.skipif(not HAVE_SUITE, reason="conformance/ has no manifest yet")
def test_suite_passes_against_the_reference():
    counts = conf.run_suite(conf.NumpyAdapter())
    assert counts["FAIL"] == 0 and counts["PASS"] > 0, counts


@pytest.mark.skipif(not HAVE_SUITE, reason="conformance/ has no manifest yet")
def test_versions_0_and_1_read_alike(tmp_path):
    """IR_SPEC 2.1: a version 0 document is a version 1 document. A suite model written with each version
    validates and gives bit-identical reference logits."""
    import ir_numpy
    doc, W, meta, _ = conf.load_model(conf.SUITE / "models" / "pk_llama")
    ids = [1, 4, 2, 7, 3]
    out = {}
    for ver in (0, 1):
        d = dict(doc, ir_version=ver)
        p = tmp_path / f"v{ver}" / f"v{ver}.gguf"
        p.parent.mkdir()
        conf.write_gguf(p, meta, W, d)
        assert ir_validate.file_errors(p) == []
        d2, W2, _, _ = conf.load_model(p.parent)
        assert d2["ir_version"] == ver
        out[ver] = ir_numpy.forward(d2, W2, ids)
    assert np.array_equal(out[0].view(np.uint32), out[1].view(np.uint32))
