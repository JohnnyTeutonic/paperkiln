"""tools/ir_recurrence.py and the digest (spec/IR_SPEC.md 6.10 and 3.10): the four rewrites reproduce the
hand-written ops (bit for bit where 6.10.6 says so), the validator's recurrence rules, finite-difference
gradients against a closed form, and the canonical serialisation."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import ir_conformance as C  # noqa: E402
import ir_numpy  # noqa: E402
import ir_recurrence as R  # noqa: E402
import ir_validate as V  # noqa: E402

SUITE = ROOT / "spec" / "conformance"
BITS = {"linear_attention": True, "ssm_scan": True, "conv1d_causal": True, "selective_scan": False}


def bits(a, b):
    return np.array_equal(np.asarray(a, np.float32).view(np.uint32), np.asarray(b, np.float32).view(np.uint32))


@pytest.mark.parametrize("op", sorted(BITS))
def test_rewrite_reproduces_the_hand_written_op(op):
    case = json.loads((SUITE / "ops" / f"{op}.json").read_text())["cases"][0]
    W = {k: C.dec(v) for k, v in case["weights"].items()}
    node = R.REWRITES[op](case["node"])
    env, st = {k: C.dec(v) for k, v in case["inputs"].items()}, {}
    ir_numpy.run_nodes([node], env, W, i=0, states=st)
    ref = C.dec(case["expect"][case["node"]["out"]])
    if BITS[op]:
        assert bits(env[node["out"]], ref)
        for t, v in list(case["expect_state"].values())[0].items():
            assert bits(list(st.values())[0][0][t], C.dec(v))
    else:
        assert np.max(np.abs(env[node["out"]] - ref)) <= 1e-6 * max(1.0, np.max(np.abs(ref)))
    assert R.compile_node(node).exact == (op in ("ssm_scan", "conv1d_causal"))


@pytest.mark.parametrize("name", ["pk_kimi", "pk_srd", "pk_mamba", "tp_mamba1"])
def test_rewritten_models(name):
    doc, W, meta, path = C.load_model(SUITE / "models" / name)
    new = R.rewrite(doc)
    m, t, *_ = V.read_gguf(path)
    assert V.doc_errors(new) == [] and V.gguf_errors(new, m, t) == []
    ids = [3, 1, 4, 1, 5, 9, 2, 6, 5]
    a, b = ir_numpy.forward(doc, W, ids), ir_numpy.forward(new, W, ids)
    if name == "tp_mamba1":
        assert np.max(np.abs(a - b)) < 1e-6 * max(1.0, np.max(np.abs(a)))
    else:
        assert bits(a, b)


def node(**kw):
    n = {"op": "recurrence", "in": ["x"], "state": "s", "index": {"i": 2}, "inputs": ["x[i]"],
         "update": ["m[i] = m[i] + x[i]"], "readout": "y[i] = m[i]", "out": "y"}
    n.update(kw)
    return n


@pytest.mark.parametrize("kw,rule", [
    ({"update": ["m[i] = m[i] +"]}, 31), ({"index": {"i": 0}}, 31), ({"inputs": []}, 31),
    ({"update": ["x[i] = x[i]"]}, 31), ({"readout": "y[i] = rowsum(m[i])"}, 31), ({"index": {"sum": 2}}, 31),
    ({"readout": "y[i] = m[j]"}, 32), ({"readout": "y[i] = sum(i, m[i])"}, 32), ({"readout": "y[i] = m[i, i]"}, 32),
    ({"init": ["m[i] = x[i]"]}, 32), ({"let": ["t[i] = t[i]"]}, 32), ({"init": ["z[i] = 1"]}, 33),
    ({"init": ["m[i] = 1", "m[i] = 2"]}, 33)])
def test_refusals(kw, rule):
    errs = R.errors(node(**kw))
    assert errs and f"(rule {rule})" in errs[0], errs


def test_reads_and_scope_that_are_legal():
    # a diagonal read, an index value, an offset read past the end (zero padding), a constant index
    n = node(index={"i": 3, "j": 3}, weights={"W[i,j]": "w"}, update=["m[i] = W[i,i] * m[i] + x[i] + i"],
             readout="y[i] = m[i + 1] + m[0]", inputs=["x[i]"])
    p = R.compile_node(n)
    W = np.arange(9, dtype=np.float32).reshape(3, 3)
    y, st = R.run(p, [np.ones((2, 3), np.float32)], {"W": W})
    m1 = np.array([1, 2, 3], np.float32)
    m2 = np.diag(W) * m1 + 1 + np.arange(3)
    assert np.array_equal(st["m"], m2)
    assert np.array_equal(y[1], np.array([m2[1], m2[2], 0]) + m2[0])


def test_validator_slot_rules():
    doc, *_ = C.load_model(SUITE / "models" / "pk_kimi")
    new = R.rewrite(doc)
    assert V.doc_errors(new) == []
    bad = copy.deepcopy(new)
    [s for s in bad["block"]["state"] if s["op"] == "recurrence"][0]["tensors"][0]["init"] = "body"
    assert any("rule 33" in e for e in V.doc_errors(bad))
    two = copy.deepcopy(new)
    two["block"]["nodes"].append(dict([n for n in new["block"]["nodes"] if n["op"] == "recurrence"][0], out="a2"))
    assert any("read by 2 nodes" in e for e in V.doc_errors(two))
    per = copy.deepcopy(new)
    [n for n in per["block"]["nodes"] if n["op"] == "recurrence"][0]["index"] = {"per_layer": [1, 1]}
    assert V.doc_errors(per)


def test_fd_grads_against_a_closed_form():
    # s_t = a s_{t-1} + x_t, y_t = s_t: dL/dx_t = sum_{u >= t} g_u a^(u - t), dL/da = sum_u g_u ds_u/da
    n = node(index={"i": 1}, weights={"a[i]": "a"}, update=["m[i] = a[i] * m[i] + x[i]"])
    p = R.compile_node(n)
    x = np.array([[0.5], [-1.0], [2.0], [0.25]], np.float32)
    g = np.array([[1.0], [0.5], [-2.0], [3.0]], np.float32)
    a = np.array([0.75], np.float32)
    fd = R.fd_grads(p, [x], {"a": a}, g)
    T = 4
    dx = [sum(float(g[u, 0]) * 0.75 ** (u - t) for u in range(t, T)) for t in range(T)]
    assert np.allclose(fd["x"][:, 0], dx, rtol=1e-8, atol=1e-9)
    s, ds, da = 0.0, 0.0, 0.0
    for t in range(T):
        ds = s + 0.75 * ds  # d s_t / d a
        s = 0.75 * s + float(x[t, 0])
        da += float(g[t, 0]) * ds
    assert abs(fd["a"][0] - da) < 1e-7


def test_digest_canonical_form():
    assert V.canonical('{"b": 1e-05, "a": [1.0, 2.5E+16, "x\\u0001"], "_note": 1}') == \
        '{"a":[1,25000000000000000,"x\\u0001"],"b":0.00001}'
    assert V.digest('{"a": 1}') == V.digest('{ "a" : 1.0, "_c": "prose" }')
    for bad in ('{"x": -0.0}', '{"a": 1, "a": 2}', '9007199254740993', '"\\ud800"'):
        with pytest.raises(V.DigestError):
            V.digest(bad)
    vec = json.loads((SUITE / "digest" / "vectors.json").read_text(encoding="utf-8"))["vectors"]
    for v in vec:
        if "refused" in v:
            with pytest.raises(V.DigestError):
                V.digest(v["text"])
        else:
            assert V.digest(v["text"], v.get("prose", False)) == v["sha256"]
