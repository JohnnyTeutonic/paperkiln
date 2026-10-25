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


def test_truth_file_quotes_occur_in_the_sources():
    truth = json.loads(open(B.TRUTH_FILE, encoding="utf-8").read())
    assert len(truth["papers"]) == 40
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
