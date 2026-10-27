"""Revision 1.3's rules on the file (IR_SPEC.md 3.1, rule 38) and on T5's bias (5.15, rule 39): every suite model's
architecture names a graph its document is, or "paperkiln"; a llama.cpp label on an IR-only document, a label
whose metadata disagrees with the document and a label section 9.3 does not define are refused; rule 38 reads
the embedded document only; tools/gguf_relabel.py relabels a pre-1.3 file without touching its document or
tensors; gguf_ir.f32 (standard library) is numpy's shortest float32 decimal; position_bias's table is checked.
Needs numpy for the f32 comparison only."""
import json
from pathlib import Path
import random
import struct
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "conformance"
sys.path.insert(0, str(ROOT / "tools"))
import gguf_ir  # noqa: E402
import gguf_relabel  # noqa: E402
import ir_conformance  # noqa: E402
import ir_validate as v  # noqa: E402

MODELS = sorted(p for p in (SUITE / "models").iterdir() if (p / f"{p.name}.gguf").is_file())


@pytest.mark.parametrize("md", MODELS, ids=lambda p: p.name)
def test_every_suite_model_is_labelled_by_its_graph(md):
    meta, tensors, *_ = v.read_gguf(md / f"{md.name}.gguf")
    assert v.architecture_errors(meta, tensors) == []
    arch = meta["general.architecture"][1]
    hint = json.loads(meta["paperkiln.ir"][1])["family_hint"]
    if hint in ("gpt2_swa", "kimi", "recurrence"):  # gpt2's tensors, IR-only semantics
        assert arch == "paperkiln"


def _relabelled(meta, arch):
    old = meta["general.architecture"][1]
    out = {(arch + k[len(old):] if k.startswith(old + ".") and not k.startswith("paperkiln.ir") else k): val
           for k, val in meta.items()}
    out["general.architecture"] = (8, arch)
    return out


@pytest.mark.parametrize("name", ["pk_kimi", "pk_gpt2_swa", "pk_recurrence"])
def test_ir_only_model_under_gpt2_is_refused(name):
    meta, tensors, *_ = v.read_gguf(SUITE / "models" / name / f"{name}.gguf")
    errs = v.architecture_errors(_relabelled(meta, "gpt2"), tensors)
    assert len(errs) == 1 and "(rule 38)" in errs[0] and "IR_SPEC 3.1" in errs[0], errs


def test_metadata_that_disagrees_and_unknown_architectures_are_refused():
    meta, tensors, *_ = v.read_gguf(SUITE / "models" / "pk_llama" / "pk_llama.gguf")
    theta = dict(meta, **{"llama.rope.freq_base": (6, 500000.0)})
    assert "theta" in " ".join(v.architecture_errors(theta, tensors))
    eps = dict(meta, **{"llama.attention.layer_norm_rms_epsilon": (6, 1e-6)})
    assert v.architecture_errors(eps, tensors)
    assert "neither" in v.architecture_errors(_relabelled(meta, "falcon"), tensors)[0]


def test_rule_38_reads_the_embedded_document_only(tmp_path):
    # A rewritten document read in place of tp_mamba1's own (recurrence nodes under architecture mamba) is
    # accepted: it is not in the file a program without the IR reads.
    errs = v.check_errors(SUITE / "models" / "tp_mamba1" / "tp_mamba1.gguf", SUITE / "recurrence" / "tp_mamba1.ir.json")
    assert errs == []


def test_relabel_keeps_document_and_tensors(tmp_path):
    doc, W, meta, _ = ir_conformance.load_model(SUITE / "models" / "pk_kimi")
    old = tmp_path / "old.gguf"
    ir_conformance.write_gguf(old, _relabelled(dict(meta), "gpt2"), W, doc)
    assert v.file_errors(old)  # rule 38
    new = tmp_path / "new.gguf"
    assert gguf_relabel.main([str(old), str(new)]) == 0
    m1, t1, d1, raw1, _ = v.read_gguf(old)
    m2, t2, d2, raw2, _ = v.read_gguf(new)
    assert v.file_errors(new) == []
    assert m2["general.architecture"][1] == "paperkiln" and m2["paperkiln.ir"] == m1["paperkiln.ir"]
    assert {k.replace("gpt2.", "paperkiln.", 1) for k in m1 if k != "general.architecture"} == \
        {k for k in m2 if k != "general.architecture"}
    assert t1 == t2 and raw1[d1:] == raw2[d2:]
    # A file rule 38 accepts is left alone without --force.
    assert gguf_relabel.main([str(SUITE / "models" / "pk_gpt2" / "pk_gpt2.gguf"), str(tmp_path / "x.gguf")]) == 0
    assert not (tmp_path / "x.gguf").exists()


def test_builder_f32_is_numpys_shortest_decimal():
    np = pytest.importorskip("numpy")
    rng = random.Random(13)
    vals = [1e-5, 1e-6, 10000.0, 1e6, 0.35355339059327373, 1e-8, 3.4e38, 1e-45]
    vals += [rng.choice([-1, 1]) * 10 ** rng.uniform(-40, 38) for _ in range(20000)]
    vals += [struct.unpack("<f", struct.pack("<I", rng.getrandbits(31)))[0] for _ in range(20000)]
    for x in (x for x in vals if x == x):
        assert gguf_ir.f32(x) == float(str(np.float32(x))), x


def test_position_bias_table_shape_is_checked():
    doc, W, meta, path = ir_conformance.load_model(SUITE / "models" / "pk_llama")
    _, tensors, *_ = v.read_gguf(path)
    dims = {t[0]: t[1] for t in tensors}
    att = [n for n in doc["block"]["nodes"] if n["op"] == "attention"][0]
    att["position_bias"] = {"weight": "token_embd.weight", "buckets": [0, 1]}
    assert v.doc_errors(doc) == []
    errs = v.position_bias_gguf_errors(doc, dims)
    assert errs and all("(rule 39)" in e for e in errs)
    att["position_bias"] = {"weight": "token_embd.weight", "buckets": [0, -1]}
    assert any("(rule 39)" in e for e in v.doc_errors(doc))
