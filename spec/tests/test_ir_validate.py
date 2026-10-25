"""tools/ir_validate.py, the shared validator (IR_SPEC.md section 10): the worked documents and every
stored document pass; one-fault mutations of the worked documents are refused under the section the rule
names; the schema evaluator agrees with the jsonschema package (where installed) on all of them; a version 0
document is a version 1 document; tools/ir_check.py's semantic checks are the validator's.
Needs numpy only (the stored corpus outside the repository is used where present)."""
import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import ir_check  # noqa: E402
import ir_validate as v  # noqa: E402

WORKED = sorted((ROOT / "examples").glob("*.ir.json"))


def load(name):
    return json.loads((ROOT / "examples" / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("path", WORKED, ids=lambda p: p.name)
def test_worked_documents_pass(path):
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert v.doc_errors(doc) == []
    assert v.doc_errors(dict(doc, ir_version=1)) == []  # version 0 documents are version 1 documents


def test_stored_corpus_passes():
    paths = v.corpus()
    bad = {p: v.file_errors(p) for p in paths}
    bad = {p: e for p, e in bad.items() if e}
    assert not bad, json.dumps(bad, indent=1)[:3000]


def node(doc, op, k=0):
    return [n for n in doc["block"]["nodes"] if n["op"] == op][k]


def kv(doc):
    return [s for s in doc["block"]["state"] if s["kind"] == "kv"][0]


def mor(doc):
    """chat_sftfinal as a MoR model: a depth_route first, kv_rows active, a weight_index (IR_SPEC 7.4)."""
    d = copy.deepcopy(doc)
    L = 5
    d["model"]["n_layers"] = d["block"]["repeat"] = L
    d["block"]["weight_index"] = {"per_layer": [0, 1, 1, 1, 2]}
    d["block"]["nodes"].insert(0, {"op": "depth_route", "in": ["x"], "weight": "rec.0.router.weight",
                                   "route": "token_choice", "n_rec": 3, "gate": "score",
                                   "step": {"per_layer": [-1, 0, 1, 2, -1]}, "out": "rec_depth"})
    kv(d)["kv_rows"] = "active"
    return d


LLAMA = "chat_sftfinal.ir.json"
CASES = [  # (name, base, mutation, (section, rule) the refusal must name)
    ("unknown op", LLAMA, lambda d: d["block"]["nodes"].append({"op": "warp", "in": ["x"], "out": "x"}), ("3.5", 4)),
    ("unknown attribute", LLAMA, lambda d: node(d, "attention").update(temperature=2.0), ("3.5", 4)),
    ("unknown top-level key", LLAMA, lambda d: d.update(profiles=["core"]), ("3.2", 1)),
    ("prose keys are free", LLAMA, lambda d: d.update(_anything={"x": 1}), None),
    ("version 2", LLAMA, lambda d: d.update(ir_version=2), ("2.3", 2)),
    ("repeat != n_layers", LLAMA, lambda d: d["block"].update(repeat=3), ("3.3", 3)),
    ("required attribute missing", LLAMA, lambda d: node(d, "rope").pop("pairing"), ("3.5", 8)),
    ("per-layer wrong length", LLAMA, lambda d: node(d, "rope").update(theta={"per_layer": [1.0]}), ("3.8", 6)),
    ("per-layer structural", LLAMA, lambda d: node(d, "linear").update(
        weight={"per_layer": ["a"] * d["block"]["repeat"]}), ("3.8", 6)),
    ("per-layer in the head", LLAMA, lambda d: d["head"]["nodes"][0].update(eps={"per_layer": [1e-5]}), ("3.8", 6)),
    ("add with one input", LLAMA, lambda d: node(d, "add").update(**{"in": ["x"]}), ("3.5", 7)),
    ("rope n_rot odd", LLAMA, lambda d: node(d, "rope").update(n_rot=3), ("5.14", 10)),
    ("rope reads another position", LLAMA, lambda d: node(d, "rope")["in"].__setitem__(2, "p2"), ("3.5", "17a")),
    ("a node writes pos", LLAMA, lambda d: node(d, "add").update(out="pos"), ("3.5", "17a")),
    ("attention not causal", LLAMA, lambda d: node(d, "attention").update(causal=False), ("5.15", 14)),
    ("attention window differs from its slot", LLAMA, lambda d: node(d, "attention").update(window=4), ("5.15", 12)),
    ("attention kv_heads differs from its slot", LLAMA, lambda d: node(d, "attention").update(kv_heads=3), ("5.15", 12)),
    ("slot declared twice", LLAMA, lambda d: d["block"]["state"].append(copy.deepcopy(kv(d))), ("6.1", 11)),
    ("slot read by no node", LLAMA, lambda d: d["block"]["state"].append(dict(copy.deepcopy(kv(d)), name="kv2")),
     ("6.1", 11)),
    ("two attention nodes on one slot", LLAMA, lambda d: d["block"]["nodes"].append(
        copy.deepcopy(node(d, "attention"))), ("6.1", "17a")),
    ("evict under the causal mask", LLAMA, lambda d: kv(d).update(evict="turn_state"), ("6.8.3", 16)),
    ("turn_state without state_weight", LLAMA, lambda d: node(d, "attention").update(mask="turn_state"), ("6.8.2", 15)),
    ("tensor_index without weight_index", LLAMA, lambda d: node(d, "linear").update(tensor_index="application"),
     ("7.2", 19)),
    ("{i} outside the block", LLAMA, lambda d: d["head"]["nodes"][0].update(weight="blk.{i}.x"), ("3.7", "17a")),
    ("{i} twice", LLAMA, lambda d: node(d, "linear").update(weight="blk.{i}.{i}.w"), ("3.7", "17a")),
    ("kv_from", LLAMA, lambda d: kv(d).update(kv_from={"per_layer": [0] * d["block"]["repeat"]}), ("7.6", 22)),
    ("weight_index unused layer", LLAMA, lambda d: d["block"].update(weight_index={"per_layer": [0, 2, 2, 2, 2, 2]}),
     ("7.1", 18)),
    ("mor is valid", "mor", lambda d: None, None),
    ("mor route not first", "mor", lambda d: d["block"]["nodes"].insert(0, d["block"]["nodes"].pop(1)), ("7.4", 20)),
    ("mor expert choice", "mor", lambda d: node(d, "depth_route").update(route="expert_choice"), ("7.7", 5)),
    ("mor unequal steps", "mor", lambda d: node(d, "depth_route").update(step={"per_layer": [-1, 0, 0, 1, -1]}),
     ("7.4", 20)),
    ("mor no kv_rows", "mor", lambda d: kv(d).pop("kv_rows"), ("7.5", 21)),
    ("mor router with {i}", "mor", lambda d: node(d, "depth_route").update(weight="rec.{i}.router.weight"), ("7.7", 20)),
    ("mor out read", "mor", lambda d: node(d, "add").update(**{"in": ["x", "rec_depth"]}), ("7.4", "17a")),
    ("mor with lists", "mor", lambda d: d["embed"]["nodes"].append({"op": "list", "in": ["x"], "out": "bank"}),
     ("7.4", 23)),
    # v1.2: rule 36 (3.5, 7.4) and rule 20's contiguous steps.
    ("a read of a value nothing writes", LLAMA, lambda d: node(d, "add")["in"].__setitem__(1, "nowhere"), ("3.5", 36)),
    ("the head writes no logits", LLAMA, lambda d: d["head"]["nodes"][-1].update(out="y"), ("3.4", 36)),
    ("mor reads an embed value", "mor", lambda d: (d["embed"]["nodes"].append(
        {"op": "affine", "in": ["x"], "scale": 1.0, "shift": 0.0, "out": "e"}),
        d["block"]["nodes"].insert(1, {"op": "add", "in": ["x", "e"], "out": "x"})), ("7.4", 36)),
    ("mor steps with a gap", "mor", lambda d: node(d, "depth_route").update(step={"per_layer": [-1, 0, -1, 1, 2]}),
     ("7.4", 20)),
]


@pytest.mark.parametrize("name,base,f,cited", CASES, ids=[c[0] for c in CASES])
def test_one_fault_is_refused_under_its_rule(name, base, f, cited):
    doc = mor(load(LLAMA)) if base == "mor" else load(base)
    f(doc)
    errs = v.doc_errors(doc)
    if cited is None:
        assert errs == []
    else:
        section, rule = cited
        assert any(e.startswith(f"IR_SPEC {section} (rule {rule}):") for e in errs), errs


@pytest.mark.parametrize("text", ['{"a": 1, "a": 1}', '{"x": {"b": [1, {"c": 0, "c": 1}]}}', '{"_n": {"a": 1, "a": 2}}'])
def test_duplicate_member_names_are_refused(text):
    """IR_SPEC 3.2, rule 35: at any depth, '_' members included, whatever the members hold."""
    errs = v.text_errors(text)
    assert len(errs) == 1 and errs[0].startswith("IR_SPEC 3.2 (rule 35):"), errs
    with pytest.raises(v.DocumentError):
        v.load_document(text)


def test_every_worked_document_reads_as_a_text():
    for p in WORKED:
        assert v.text_errors(p.read_text(encoding="utf-8")) == []


def test_schema_agrees_with_jsonschema():
    js = pytest.importorskip("jsonschema")
    if not hasattr(js, "Draft202012Validator"):
        pytest.skip("jsonschema without draft 2020-12")
    val = js.Draft202012Validator(v.schema())
    js.Draft202012Validator.check_schema(v.schema())
    docs = [json.loads(p.read_text(encoding="utf-8")) for p in WORKED]
    for name, base, f, _ in CASES:
        d = mor(load(LLAMA)) if base == "mor" else load(base)
        f(d)
        docs.append(d)
    for d in docs:
        assert (not list(val.iter_errors(d))) == (not v.schema_errors(d))


def test_ir_check_uses_the_validator():
    doc = load(LLAMA)
    assert ir_check.state_errors(doc) == v.state_errors(doc) == []
    bad = mor(doc)
    kv(bad).pop("kv_rows")
    assert ir_check.depth_route_errors(bad) == v.depth_route_errors(bad) and v.depth_route_errors(bad)


SUITE_MODEL = ROOT / "conformance" / "models" / "pk_llama_attnres"


@pytest.mark.skipif(not SUITE_MODEL.is_dir(), reason="the conformance suite's models are absent")
def test_s1_pseudo_query_absent_is_an_exporter_warning(tmp_path):
    """IR_SPEC 5.18: a consumer may accept a missing S = 1 pseudo-query; exporters must write it, so the
    validator warns (exporter rule) and still accepts the file. A missing S >= 2 query is refused."""
    import ir_conformance as conf
    doc, W, meta, _ = conf.load_model(SUITE_MODEL)
    assert v.depth_attend_sources(doc)["blk.0.attnres_attn_q.weight"] == {1}
    for name, refused in (("blk.0.attnres_attn_q.weight", False), ("blk.1.attnres_attn_q.weight", True)):
        p = tmp_path / f"{refused}.gguf"
        conf.write_gguf(p, meta, {k: a for k, a in W.items() if k != name}, doc)
        errs, warns = v.file_report(p)
        if refused:
            assert any(name in e and e.startswith("IR_SPEC 3.7 (rule 25)") for e in errs), errs
        else:
            assert errs == [] and len(warns) == 1 and "exporter rule" in warns[0] and name in warns[0], (errs, warns)


@pytest.mark.skipif(not SUITE_MODEL.is_dir(), reason="the conformance suite's models are absent")
def test_version_key_follows_the_embedded_document():
    """IR_SPEC 11.2: a document read in place of the file's own is governed by its own ir_version (rule 2);
    rule 24 compares paperkiln.ir_version with the embedded document, as ember does."""
    import ir_conformance as conf
    doc, W, meta, path = conf.load_model(SUITE_MODEL)
    sub = dict(doc, ir_version=1 - doc["ir_version"])
    meta2, tensors, *_ = v.read_gguf(path)
    assert v.gguf_errors(sub, meta2, tensors) == [] and v.doc_errors(sub) == []
    bad = dict(meta2, **{"paperkiln.ir_version": (4, 7)})
    assert any(e.startswith("IR_SPEC 3.1 (rule 24)") for e in v.gguf_errors(sub, bad, tensors))
