"""tools/ir_validate.py, the shared validator (spec/IR_SPEC.md section 10): the worked documents and every
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

WORKED = sorted((ROOT / "ir").glob("*.ir.json"))


def load(name):
    return json.loads((ROOT / "ir" / name).read_text(encoding="utf-8"))


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
CASES = [  # (name, base, mutation, section the refusal must name)
    ("unknown op", LLAMA, lambda d: d["block"]["nodes"].append({"op": "warp", "in": ["x"], "out": "x"}), "10.1 rule 1"),
    ("unknown attribute", LLAMA, lambda d: node(d, "attention").update(temperature=2.0), "10.1 rule 1"),
    ("unknown top-level key", LLAMA, lambda d: d.update(profiles=["core"]), "10.1 rule 1"),
    ("prose keys are free", LLAMA, lambda d: d.update(_anything={"x": 1}), None),
    ("version 2", LLAMA, lambda d: d.update(ir_version=2), "2.3"),
    ("repeat != n_layers", LLAMA, lambda d: d["block"].update(repeat=3), "3.3"),
    ("required attribute missing", LLAMA, lambda d: node(d, "rope").pop("pairing"), "10.1 rule 1"),
    ("per-layer wrong length", LLAMA, lambda d: node(d, "rope").update(theta={"per_layer": [1.0]}), "3.8"),
    ("per-layer structural", LLAMA, lambda d: node(d, "linear").update(
        weight={"per_layer": ["a"] * d["block"]["repeat"]}), "3.8"),
    ("per-layer in the head", LLAMA, lambda d: d["head"]["nodes"][0].update(eps={"per_layer": [1e-5]}), "3.8"),
    ("add with one input", LLAMA, lambda d: node(d, "add").update(**{"in": ["x"]}), "10.1 rule 7"),
    ("rope n_rot odd", LLAMA, lambda d: node(d, "rope").update(n_rot=3), "5.14"),
    ("rope reads another position", LLAMA, lambda d: node(d, "rope")["in"].__setitem__(2, "p2"), "3.5"),
    ("a node writes pos", LLAMA, lambda d: node(d, "add").update(out="pos"), "3.5"),
    ("attention not causal", LLAMA, lambda d: node(d, "attention").update(causal=False), "5.15"),
    ("attention window differs from its slot", LLAMA, lambda d: node(d, "attention").update(window=4), "5.15"),
    ("attention kv_heads differs from its slot", LLAMA, lambda d: node(d, "attention").update(kv_heads=3), "5.15"),
    ("slot declared twice", LLAMA, lambda d: d["block"]["state"].append(copy.deepcopy(kv(d))), "6.1"),
    ("slot read by no node", LLAMA, lambda d: d["block"]["state"].append(dict(copy.deepcopy(kv(d)), name="kv2")),
     "6.1"),
    ("two attention nodes on one slot", LLAMA, lambda d: d["block"]["nodes"].append(
        copy.deepcopy(node(d, "attention"))), "6.1"),
    ("evict under the causal mask", LLAMA, lambda d: kv(d).update(evict="turn_state"), "6.8.3"),
    ("turn_state without state_weight", LLAMA, lambda d: node(d, "attention").update(mask="turn_state"), "6.8.2"),
    ("tensor_index without weight_index", LLAMA, lambda d: node(d, "linear").update(tensor_index="application"),
     "7.2"),
    ("{i} outside the block", LLAMA, lambda d: d["head"]["nodes"][0].update(weight="blk.{i}.x"), "3.7"),
    ("{i} twice", LLAMA, lambda d: node(d, "linear").update(weight="blk.{i}.{i}.w"), "3.7"),
    ("kv_from", LLAMA, lambda d: kv(d).update(kv_from={"per_layer": [0] * d["block"]["repeat"]}), "7.6"),
    ("weight_index unused layer", LLAMA, lambda d: d["block"].update(weight_index={"per_layer": [0, 2, 2, 2, 2, 2]}),
     "7.1"),
    ("mor is valid", "mor", lambda d: None, None),
    ("mor route not first", "mor", lambda d: d["block"]["nodes"].insert(0, d["block"]["nodes"].pop(1)), "7.4"),
    ("mor expert choice", "mor", lambda d: node(d, "depth_route").update(route="expert_choice"), "7.7"),
    ("mor unequal steps", "mor", lambda d: node(d, "depth_route").update(step={"per_layer": [-1, 0, 0, 1, -1]}),
     "7.4"),
    ("mor no kv_rows", "mor", lambda d: kv(d).pop("kv_rows"), "7.5"),
    ("mor router with {i}", "mor", lambda d: node(d, "depth_route").update(weight="rec.{i}.router.weight"), "7.7"),
    ("mor out read", "mor", lambda d: node(d, "add").update(**{"in": ["x", "rec_depth"]}), "7.4"),
    ("mor with lists", "mor", lambda d: d["embed"]["nodes"].append({"op": "list", "in": ["x"], "out": "bank"}),
     "7.4"),
]


@pytest.mark.parametrize("name,base,f,section", CASES, ids=[c[0] for c in CASES])
def test_one_fault_is_refused_under_its_section(name, base, f, section):
    doc = mor(load(LLAMA)) if base == "mor" else load(base)
    f(doc)
    errs = v.doc_errors(doc)
    if section is None:
        assert errs == []
    else:
        assert any(e.startswith(f"IR_SPEC {section}:") for e in errs), errs


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
