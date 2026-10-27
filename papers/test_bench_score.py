"""Offline tests for papers/bench_score.py: the block graph's structure by data flow, the size
matching, and the truth file's integrity (every label quote verbatim in its paper's LaTeX).

    python -m pytest papers/test_bench_score.py -q
"""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_score as B  # noqa: E402
import llm_extract as L  # noqa: E402


def n(op, ins, out, role="", new_op="", sub="block", attrs=()):
    return {"subgraph": sub, "op": op, "role": role, "inputs": list(ins), "output": out, "new_op": new_op,
            "attributes": [{"name": k, "value": v} for k, v in attrs]}


def attn(x, out):
    return [n("linear", [x], "qkv"), n("split", ["qkv"], "q,k,v"), n("rope", ["q", "k"], "qr,kr"),
            n("attention", ["qr", "kr", "v"], "a"), n("linear", ["a"], out)]


def ffn(x, out, gated=False):
    if gated:
        return [n("linear", [x], "g"), n("linear", [x], "u"), n("silu", ["g"], "gs"), n("mul", ["gs", "u"], "m"),
                n("linear", ["m"], out)]
    return [n("linear", [x], "u"), n("gelu", ["u"], "ua"), n("linear", ["ua"], out)]


def test_pre_norm_llama_block():
    ops = [n("rmsnorm", ["x"], "h"), *attn("h", "o"), n("add", ["x", "o"], "x2"), n("rmsnorm", ["x2"], "h2"),
           *ffn("h2", "f", gated=True), n("add", ["x2", "f"], "x")]
    g = B.graph_structure(ops)
    assert g == {"attn": "pre", "ffn": "pre", "parallel": False, "gated": True, "rope": True, "abs_pos": False}


def test_post_sandwich_branch_post_and_none():
    post = [*attn("x", "o"), n("add", ["x", "o"], "r1"), n("layernorm", ["r1"], "h1"), *ffn("h1", "f"),
            n("add", ["h1", "f"], "r2"), n("layernorm", ["r2"], "x")]
    assert (B.graph_structure(post)["attn"], B.graph_structure(post)["ffn"]) == ("post", "post")
    sandwich = [n("rmsnorm", ["x"], "h"), *attn("h", "o"), n("rmsnorm", ["o"], "on"), n("add", ["x", "on"], "x1"),
                n("rmsnorm", ["x1"], "h2"), *ffn("h2", "f"), n("rmsnorm", ["f"], "fn"), n("add", ["x1", "fn"], "x")]
    g = B.graph_structure(sandwich)
    assert (g["attn"], g["ffn"]) == ("sandwich", "sandwich")
    # Primer: the norm before attention, the FFN's after its transformation, inside the branch.
    primer = [n("expr", ["x"], "h", new_op="pnorm"), *attn("h", "o"), n("add", ["x", "o"], "x1"),
              n("linear", ["x1"], "u"), n("expr", ["u"], "u2", new_op="sq"), n("linear", ["u2"], "f"),
              n("expr", ["f"], "fn", new_op="pnorm"), n("add", ["x1", "fn"], "x")]
    g = B.graph_structure(primer, {"pnorm"}, {"sq"})
    assert (g["attn"], g["ffn"], g["gated"]) == ("pre", "branch_post", False)
    bare = [*attn("x", "o"), n("add", ["x", "o"], "x1"), *ffn("x1", "f"), n("add", ["x1", "f"], "x")]
    assert (B.graph_structure(bare)["attn"], B.graph_structure(bare)["ffn"]) == ("none", "none")


def test_parallel_block_and_moe_gate():
    par = [n("layernorm", ["x"], "h"), *attn("h", "o"), *ffn("h", "f", gated=True), n("add", ["x", "o"], "x1"),
           n("add", ["x1", "f"], "x")]
    g = B.graph_structure(par)
    assert g["parallel"] is True and (g["attn"], g["ffn"]) == ("pre", "pre")
    moe = [n("rmsnorm", ["x"], "h"), *attn("h", "o"), n("add", ["x", "o"], "x1"), n("rmsnorm", ["x1"], "h2"),
           n("moe_ffn", ["h2"], "f", attrs=[("activation", "swiglu")]), n("add", ["x1", "f"], "x")]
    assert B.graph_structure(moe)["gated"] is True
    assert B.graph_structure([n("embedding", ["tokens"], "x", sub="embed")]) is None


def test_scored_against_the_labels():
    g = {"attn": "pre", "ffn": "branch_post", "parallel": False, "gated": False, "rope": False, "abs_pos": True}
    rows = B.score_graph(g, {"norm_placement": {"value": {"attn": "pre", "ffn": "branch_post"}},
                             "parallel_block": {"value": True}, "ffn_gated": None}, "learned")
    assert [(r["field"], r["score"]) for r in rows] == [("placement.attn", "correct"), ("placement.ffn", "correct"),
                                                        ("parallel_block", "WRONG"), ("abs_position", "correct")]
    assert all(r["score"] == "abstain" for r in B.score_graph(None, {"ffn_gated": {"value": True}}, "rope"))


def test_sizes_matched_per_config():
    truth = [{"name": "small", "d_model": {"value": 512}, "n_layers": {"value": 6}, "d_ff": {"value": 2048}},
             {"name": "big", "d_model": {"value": 1024}, "n_layers": {"value": 6}, "vocab_size": {"value": 32000,
                                                                                               "accept": [32000, 32768]}}]
    got = [{"_name": "big", "d_model": 1024.0, "n_layers": 6.0, "vocab_size": 32768.0, "n_heads": 16.0},
           {"_name": "small", "d_model": 512.0, "n_layers": 6.0, "d_ff": 4096.0},
           {"_name": "extra", "d_model": 2048.0}]
    s = B.score_sizes(truth, got)
    assert s["tally"] == {"correct": 5, "WRONG": 1, "abstain": 0, "unlabelled": 2}
    assert s["misses"] == [{"config": "small", "field": "d_ff", "truth": 2048, "got": 4096.0, "score": "WRONG",
                            "matched": "small"}]
    assert s["unmatched_extracted"] == ["extra"]


def test_sizes_match_a_config_by_name_when_no_size_agrees():
    # Mamba's headline configs state only a context length: matched by name, not left unmatched.
    truth = [{"name": "Mamba-130M", "context_length": {"value": 2048}}]
    s = B.score_sizes(truth, [{"_name": "Mamba-130M", "context_length": 2048.0}])
    assert s["tally"] == {"correct": 1, "WRONG": 0, "abstain": 0, "unlabelled": 0}
    s = B.score_sizes(truth, [{"_name": "other", "context_length": 2048.0}])
    assert s["tally"] == {"correct": 0, "WRONG": 0, "abstain": 1, "unlabelled": 1}


def test_accept_lists_for_context_length_and_qk_norm():
    # DeepSeek-V2: the 4K pre-training length or the extended 128K (main session, 27 Oct 2026).
    truth = [{"name": "DeepSeek-V2", "d_model": {"value": 5120},
              "context_length": {"value": 4096, "accept": [4096, 131072, 128000]}}]
    for got, verdict in ((131072.0, "correct"), (4096.0, "correct"), (32768.0, "WRONG")):
        s = B.score_sizes(truth, [{"_name": "DeepSeek-V2", "d_model": 5120.0, "context_length": got}])
        assert s["tally"][verdict] == (2 if verdict == "correct" else 1)
    # OLMo 2's q/k norm reads as full or unspecified.
    nf = {"new_fields": {"qk_norm": {"value": "full", "accept": ["full", "unspecified"]}}}
    for got, verdict in (("full", "correct"), ("unspecified", "correct"), ("per_head", "WRONG")):
        rows = {r["field"]: r["score"] for r in B.score_new_fields("x", _rep({"qk_norm": got}), {}, nf)}
        assert rows["qk_norm"] == verdict


def test_truth_file_quotes_occur_in_the_sources():
    truth = json.loads(open(B.TRUTH_FILE, encoding="utf-8").read())
    assert len(truth["papers"]) == 43  # the 40 + Mamba, Qwen3, OLMo 2 (28 Oct 2026)
    checked = 0
    for pid, p in truth["papers"].items():
        path = os.path.join(HERE, ".cache", f"{pid}.tex")
        if not os.path.exists(path):
            pytest.skip("the benchmark's LaTeX cache is not on this machine")
        V = L.Verifier(L.prepare_source(open(path, encoding="utf-8", errors="replace").read()))
        for c in p["configs"]:
            for f in B.SIZE_FIELDS:
                e = c.get(f)
                if e:
                    assert V.quote_found(e["quote"]), (pid, c["name"], f)
                    assert L.number_in_quote(float(e["value"]), e["quote"]), (pid, c["name"], f)
                    checked += 1
        for k, e in (p.get("structure") or {}).items():
            if e:
                assert V.quote_found(e["quote"]), (pid, k)
    assert checked > 500


def _rep(flavours=None, new_ops=None):
    fl = {k: {"value": "", "check": "not_stated"} for k in L.FLAVOUR_FIELDS}
    for k, v in (flavours or {}).items():
        fl[k] = {"value": v, "check": "verified"}
    return {"flavours": fl, "new_ops": new_ops or []}


def _op(name, replaces, roles, kind="expr", inits=None):
    inits = inits or {}
    return {"name": name, "kind": kind, "replaces": replaces, "check": "verified",
            "inputs": [{"name": "xabc"[i], "role": r, "init": {"value": inits.get(i, "")},
                        "init_check": "verified" if i in inits else "not_stated"} for i, r in enumerate(roles)]}


def test_new_fields_scored_against_their_labels():
    truth = {"structure": {"norm_placement": {"value": {"attn": "pre", "ffn": "branch_post"}}},
             "new_fields": {"qk_norm": {"value": "absent"}, "ssm": {"value": "absent"},
                            "positional": {"value": "t5_bias", "rel_buckets": {"value": 32},
                                           "rel_max_distance": {"value": 128}},
                            "new_ops": [{"name": "sq", "category": "activation", "scored": True,
                                         "inputs": [{"role": "input"}]},
                                        {"name": "ln", "category": "norm", "scored": True,
                                         "inputs": [{"role": "input"}]},
                                        {"name": "mup", "category": "embed", "scored": False,
                                         "inputs": [{"role": "input"}]}]}}
    rep = _rep({"norm_placement": "other:per sublayer", "norm_placement_attn": "pre",
                "norm_placement_ffn": "branch_post", "qk_norm": "per_head", "positional": "t5_bias",
                "rel_buckets": "32"},
               [_op("squared_relu", "FFN activation", ["input"]),
                _op("primer_norm", "layer normalization", ["input", "learned_vector", "learned_vector"]),
                _op("temp", "attention temperature", ["input", "learned_scalar"], inits={1: "1.0"})])
    rows = {(r["field"], r.get("op")): r["score"] for r in B.score_new_fields("x", rep, {}, truth)}
    assert rows[("placement_field.attn", None)] == "correct"
    assert rows[("placement_field.ffn", None)] == "correct"
    assert rows[("qk_norm", None)] == "WRONG"  # asserted where the paper never mentions one
    assert rows[("ssm", None)] == "correct"
    assert rows[("positional_ext", None)] == "correct"
    assert rows[("rel_buckets", None)] == "correct"
    assert rows[("rel_max_distance", None)] == "abstain"
    assert rows[("new_op_inputs", "sq")] == "correct"
    assert rows[("new_op_inputs", "ln")] == "correct"  # a norm's gain and bias are allowed
    assert ("new_op_inputs", "mup") not in rows  # not scored
    assert rows[("learned_inputs", None)] == "WRONG"  # the learned temperature no label has


def test_new_op_roles_wrong_or_unstated():
    truth = {"new_fields": {"new_ops": [{"name": "cap", "category": "final_logits", "scored": True,
                                         "inputs": [{"role": "input"}, {"role": "other"}]}]}}
    for roles, verdict in ((["input", "learned_scalar"], "WRONG"), ([None], "abstain"), (["input"], "correct")):
        rep = _rep(new_ops=[_op("final_logit_softcap", "final logits", roles)])
        r = [x for x in B.score_new_fields("x", rep, {}, truth) if x["field"] == "new_op_inputs"]
        assert r[0]["score"] == verdict, roles
    rep = _rep(new_ops=[_op("final_logit_softcap", "final logits", ["input"], kind="needs_new_primitive")])
    assert [x["score"] for x in B.score_new_fields("x", rep, {}, truth) if x["field"] == "new_op_inputs"] == ["WRONG"]


def test_new_field_label_quotes_occur_in_the_sources():
    truth = json.loads(open(B.TRUTH_FILE, encoding="utf-8").read())
    checked = 0

    def walk(V, pid, x):
        nonlocal checked
        if isinstance(x, dict):
            q = x.get("quote")
            if isinstance(q, str) and q:
                assert V.quote_found(q), (pid, q[:60])
                if isinstance(x.get("value"), (int, float)) and not isinstance(x.get("value"), bool):
                    assert L.number_in_quote(float(x["value"]), q), (pid, q[:60])
                checked += 1
            for v in x.values():
                walk(V, pid, v)
        elif isinstance(x, list):
            for v in x:
                walk(V, pid, v)

    for pid, p in truth["papers"].items():
        assert "new_fields" in p, pid
        path = os.path.join(HERE, ".cache", f"{pid}.tex")
        if not os.path.exists(path):
            pytest.skip("the benchmark's LaTeX cache is not on this machine")
        walk(L.Verifier(L.prepare_source(open(path, encoding="utf-8", errors="replace").read())), pid, p["new_fields"])
    assert checked >= 40


def test_affine_node_and_attention_primitive_count_as_the_op():
    truth = {"new_fields": {"new_ops": [
        {"name": "embed scale", "category": "embed", "scored": True, "inputs": [{"role": "input"}, {"role": "other"}]},
        {"name": "attn cap", "category": "attention_logits", "scored": True,
         "inputs": [{"role": "input"}, {"role": "other"}]}]}}
    out = {"block_ops": [n("embedding", ["tok"], "e", sub="embed"),
                         n("affine", ["e"], "x", sub="embed", attrs=[("scale", "22.627417"), ("shift", "0")])]}
    rep = _rep(new_ops=[_op("softcapped_attention", "attention", [], kind="needs_new_primitive")])
    rows = {r["op"]: r["score"] for r in B.score_new_fields("x", rep, out, truth) if r["field"] == "new_op_inputs"}
    assert rows == {"embed scale": "correct", "attn cap": "correct"}
    rows = {r["op"]: r["score"] for r in B.score_new_fields("x", _rep(), {"block_ops": []}, truth)
            if r["field"] == "new_op_inputs"}
    assert rows == {"embed scale": "abstain", "attn cap": "abstain"}
