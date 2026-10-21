"""Offline tests for papers/llm_extract.py: the expr grammar, the verifier on
crafted good and bad replies, the extraction.json hand-off to
tools/arxiv_to_chat.py, and the API path with a mocked client (receipts,
cost, cache, budget, refusals, splitting). No network, no key.

    python -m pytest papers/test_llm_extract.py -q
"""
import json
import math
import os
import re
import sys
from types import SimpleNamespace

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))
import llm_extract as L  # noqa: E402

# --------------------------------------------------------------------------
# fixtures

LLAMA_TEX = r"""
\title{A Small Open Model}
\section{Architecture}
\label{sec:arch}
Our network is based on the transformer architecture. We use the RMSNorm normalizing
function to normalize the input of each transformer sub-layer. % we once used LayerNorm here
We replace the ReLU non-linearity by the SwiGLU activation function.
We remove the absolute positional embeddings, and instead, add rotary positional
embeddings (RoPE) at each layer of the network.
\begin{tabular}{cccc}
params & dimension & $n$ heads & $n$ layers \\
1.3B & 2048 & 16 & 24 \\
7B & 4096 & 32 & 32 \\
\end{tabular}
The feed-forward dimension is 5504 for the smallest model, the vocabulary has 32{,}000
tokens and the context length is 2048 tokens. We use $\epsilon = 10^{-6}$ in every norm.
"""

# Verbatim excerpts of Primer (arXiv:2109.08668), whose activation is outside
# every lattice (squared ReLU) and whose attention adds a convolution over time.
PRIMER_TEX = r"""
\section{Primer}
\paragraph{Squared ReLU:} The most effective modification is the improvement from a ReLU activation to a squared ReLU activation in the Transformer's feed forward block. Rectified polynomials of varying degrees have been studied in the context of neural network activation functions~\cite{Krotov2016DenseAM}, but are not commonly used.

\paragraph{Multi-DConv-Head Attention (MDHA):} Another effective modification is adding 3x1 depthwise convolutions after each of the multi-head projections for query $Q$, key $K$ and value $V$ in self-attention. These depthwise convolutions are performed over the spatial dimension of each dense projection's output.

it is necessary to initialize the search population with copies of the Transformer~\cite{so2019evolved} (input embedding size $d_{model}=512$, feed forward upwards projection size $d_{ff}=2048$, and number of layers $L=6$)
"""

GPT_TEX = r"""
\section{Model and Architectures}
We use the same model and architecture as GPT-2 \cite{radford2019language}, including the
modified initialization, pre-normalization, and reversible tokenization described therein.
"""


def ev(value="", status="not_stated", quote="", where=""):
    return {"value": value, "status": status, "quote": quote, "where": where}


def config(name, **fields):
    c = {"name": name, **{k: ev() for k in L.CONFIG_FIELDS}}
    c.update(fields)
    return c


def base_output(**kw):
    out = {"title": "", "model_name": "", "family": ev(),
           "inherits": {"ancestor": "", "quote": "", "where": ""}, "configs": [], "primary_config": "",
           "flavours": {k: ev() for k in L.FLAVOUR_FIELDS}, "block_ops": [], "new_ops": [],
           "mechanisms_outside_ir": [], "notes": ""}
    out.update(kw)
    return out


def node(op, role, ins, out, attrs=(), new_op="", subgraph="block"):
    return {"subgraph": subgraph, "op": op, "role": role, "inputs": list(ins), "output": out,
            "attributes": [dict(name=n, **ev(*a)) for n, a in attrs], "new_op": new_op}


ROW_SMALL = "1.3B & 2048 & 16 & 24"
NORM_Q = "We use the RMSNorm normalizing function to normalize the input of each transformer sub-layer."


def llama_output():
    fl = {k: ev() for k in L.FLAVOUR_FIELDS}
    fl.update(
        norm=ev("rmsnorm", "stated", NORM_Q, "Sec. Architecture"),
        norm_placement=ev("pre", "stated", NORM_Q, "Sec. Architecture"),
        norm_eps=ev("1e-6", "stated", r"We use $\epsilon = 10^{-6}$ in every norm.", "sec:arch"),
        activation=ev("swiglu", "stated", "We replace the ReLU non-linearity by the SwiGLU activation function.",
                      "Sec. Architecture"),
        positional=ev("rope", "stated", "add rotary positional embeddings (RoPE) at each layer of the network",
                      "Sec. Architecture"),
    )
    return base_output(
        title="A Small Open Model", model_name="small",
        family=ev("decoder_only", "stated", "Our network is based on the transformer architecture.", "sec:arch"),
        configs=[
            config("1.3B", d_model=ev("2048", "stated", ROW_SMALL, "Table 1"),
                   n_heads=ev("16", "stated", ROW_SMALL, "Table 1"),
                   n_layers=ev("24", "stated", ROW_SMALL, "Table 1"),
                   d_ff=ev("5504", "stated", "The feed-forward dimension is 5504 for the smallest model", "sec:arch"),
                   vocab_size=ev("32000", "stated", "the vocabulary has 32{,}000 tokens", "sec:arch"),
                   context_length=ev("2048", "stated", "the context length is 2048 tokens", "sec:arch")),
            config("7B", d_model=ev("4096", "stated", "7B & 4096 & 32 & 32", "Table 1"),
                   n_heads=ev("32", "stated", "7B & 4096 & 32 & 32", "Table 1"),
                   n_layers=ev("32", "stated", "7B & 4096 & 32 & 32", "Table 1")),
        ],
        primary_config="1.3B",
        flavours=fl,
        block_ops=[
            node("embedding", "token embedding", ["tokens"], "x", subgraph="embed"),
            node("rmsnorm", "attn_norm", ["x"], "h", [("eps", ("1e-6", "stated",
                                                              r"We use $\epsilon = 10^{-6}$ in every norm.", ""))]),
            node("attention", "self-attention", ["q", "k", "v"], "a",
                 [("heads", ("16", "stated", ROW_SMALL, "Table 1")), ("window", ("", "not_stated", "", ""))]),
            node("add", "residual", ["x", "a"], "x"),
        ],
    )


def primer_output():
    fl = {k: ev() for k in L.FLAVOUR_FIELDS}
    sq = ("The most effective modification is the improvement from a ReLU activation to a squared ReLU "
          "activation in the Transformer's feed forward block.")
    md = ("Another effective modification is adding 3x1 depthwise convolutions after each of the multi-head "
          "projections for query $Q$, key $K$ and value $V$ in self-attention.")
    fl["activation"] = ev("new_op:squared_relu", "stated", sq, "Squared ReLU")
    return base_output(
        title="Primer", family=ev("decoder_only", "stated", sq, "Primer"),
        configs=[config("seed", d_model=ev("512", "stated", "input embedding size $d_{model}=512$", "Sec. search"),
                        d_ff=ev("2048", "stated", "feed forward upwards projection size $d_{ff}=2048$", ""),
                        n_layers=ev("6", "stated", "and number of layers $L=6$", ""))],
        primary_config="seed", flavours=fl,
        new_ops=[
            {"name": "squared_relu", "kind": "expr", "expr": "relu(x)^2",
             "inputs": [{"name": "x", "meaning": "FFN up-projection output"}], "replaces": "FFN activation",
             "reason": "", "quote": sq, "where": "Squared ReLU"},
            {"name": "mdha_dconv", "kind": "needs_new_primitive", "expr": "",
             "inputs": [{"name": "x", "meaning": "per-head Q, K or V projection"}],
             "replaces": "nothing (added after the Q, K, V projections)",
             "reason": "a causal 3x1 depthwise convolution over the time axis: mixes tokens, so it needs a "
                       "conv1d primitive with a state of the last two positions",
             "quote": md, "where": "Multi-DConv-Head Attention (MDHA)"},
        ],
        block_ops=[node("linear", "ffn_up", ["h"], "u"), node("expr", "ffn activation", ["u"], "u2",
                                                                   new_op="squared_relu"),
                   node("unsupported", "q dconv", ["q"], "q", new_op="mdha_dconv")],
    )


def src(tex):
    return L.prepare_source(tex)


# --------------------------------------------------------------------------
# grammar

@pytest.mark.parametrize("expr,x,want", [
    ("relu(x)^2", 3.0, 9.0), ("-x^2", 2.0, -4.0), ("2^3^2", 0.0, 512.0), ("x^-1", 4.0, 0.25),
    ("1 - 2 - 3", 0.0, -4.0), ("8 / 4 / 2", 0.0, 1.0), ("where(x, x, 0.01*x)", -2.0, -0.02),
    ("max(x, 0) + min(x, 0)", -1.5, -1.5), ("sqrt(pi)^2", 0.0, math.pi), ("abs(-x)", 2.0, 2.0),
    ("x*sigmoid(1.702*x)", 0.0, 0.0), ("0.5*x*(1+erf(x/sqrt(2)))", 0.0, 0.0), (".5e1", 0.0, 5.0),
])
def test_grammar_evaluates(expr, x, want):
    assert L.expr_eval(L.parse_expr(expr), {"x": x}) == pytest.approx(want)


@pytest.mark.parametrize("expr,msg", [
    ("", "empty"), ("max(x, 0", "expected ')'"), ("foo(x)", "unknown function"), ("relu", "without arguments"),
    ("x > 0", "unexpected character"), ("max(x)", "takes 2"), ("x +", "end of input"), ("+x", "unexpected '+'"),
    ("e^x", None), ("x y", "after a complete expression"), ("beta*x", "unknown name"),
])
def test_grammar_rejects(expr, msg):
    if msg is None:  # e is an input name, not Euler's number: legal, but undeclared below
        assert L.check_expr(expr, ["x"])["errors"] == ["uses undeclared input(s) ['e']"]
        return
    with pytest.raises(L.ExprError, match=re.escape(msg)):
        L.parse_expr(expr)


def test_grammar_canonical_and_rows():
    sm = "exp(x - rowmax(x)) / rowsum(exp(x - rowmax(x)))"
    r = L.check_expr(sm, ["x"])
    assert r["ok"] and r["canonical"] == "(exp((x-rowmax(x)))/rowsum(exp((x-rowmax(x)))))"
    assert sum(r["samples"]["output"]) == pytest.approx(1.0)
    assert L.expr_canonical(L.parse_expr("a*x + b")) == "((a*x)+b)"
    bad = L.check_expr("relu(a)*x", ["a", "x"])
    assert any("first input" in e for e in bad["errors"])
    assert L.check_expr("relu(a)*x", ["x", "a"])["ok"]
    assert L.check_expr("x", ["x", "a"])["warnings"]  # declared, unused
    assert L.check_expr("log(x)", ["x"])["warnings"]  # log of a negative sample


def test_numbers_in_quote():
    assert L.number_in_quote(32000, "32{,}000 tokens")
    assert L.number_in_quote(32000, "32,000 tokens")
    assert L.number_in_quote(1e-6, r"$\epsilon = 10^{-6}$")
    assert L.number_in_quote(1e-5, r"$1\times 10^{-5}$")
    assert L.number_in_quote(1e-5, "eps 1e-5")
    assert L.number_in_quote(2048, "a 2K context") and L.number_in_quote(2000, "a 2K context")
    assert L.number_in_quote(1048576, "$2^{20}$ tokens")
    assert L.number_in_quote(12, "twelve layers")
    assert not L.number_in_quote(1e-5, r"$10^{-6}$")
    assert not L.number_in_quote(4096, "1.3B & 2048 & 16 & 24")


def test_prepare_source_strips_comments_only():
    s = src(LLAMA_TEX + "\n50\\% of the data\n\\begin{comment}secret\\end{comment}\n")
    assert "we once used LayerNorm" not in s and "secret" not in s
    assert "50\\% of the data" in s and "RMSNorm normalizing" in s


# --------------------------------------------------------------------------
# verifier

def test_good_reply_verifies():
    rep = L.verify(llama_output(), src(LLAMA_TEX))
    s = rep["summary"]
    assert s["unverified"] == [] and s["new_op_errors"] == [] and s["node_errors"] == []
    f = rep["flavours"]
    assert f["norm"]["check"] == f["activation"]["check"] == f["positional"]["check"] == "verified"
    assert f["norm_eps"]["check"] == "verified"
    assert rep["configs"][0]["vocab_size"]["check"] == "verified"
    assert rep["configs"][0]["consistency"][0]["ok"]
    assert f["moe"]["check"] == "not_stated"


def test_quote_not_in_source_is_unverified():
    out = llama_output()
    out["flavours"]["norm"]["quote"] = "We use the RMSNorm normalizing function to normalise every input."
    rep = L.verify(out, src(LLAMA_TEX))
    assert rep["flavours"]["norm"]["check"] == "unverified"
    assert "quote not found" in rep["flavours"]["norm"]["reasons"][0]


def test_commented_out_text_cannot_be_quoted():
    out = llama_output()
    out["flavours"]["norm"] = ev("layernorm", "stated", "we once used LayerNorm here", "")
    assert L.verify(out, src(LLAMA_TEX))["flavours"]["norm"]["check"] == "unverified"


def test_whitespace_differences_tolerated():
    out = llama_output()
    out["flavours"]["norm"]["quote"] = "We  use the RMSNorm\nnormalizing   function"
    assert L.verify(out, src(LLAMA_TEX))["flavours"]["norm"]["check"] == "verified"


def test_number_must_be_in_its_quote():
    out = llama_output()
    out["configs"][0]["d_ff"] = ev("11008", "stated", "The feed-forward dimension is 5504 for the smallest model", "")
    rep = L.verify(out, src(LLAMA_TEX))
    assert rep["configs"][0]["d_ff"]["check"] == "unverified"
    assert "does not occur in its quote" in rep["configs"][0]["d_ff"]["reasons"][0]


def test_flavour_quote_must_name_the_value():
    out = llama_output()
    out["flavours"]["activation"]["value"] = "gelu"  # quote names ReLU and SwiGLU, not GELU
    rep = L.verify(out, src(LLAMA_TEX))
    assert rep["flavours"]["activation"]["check"] == "unverified"


def test_value_outside_enum_and_bad_bool():
    out = llama_output()
    out["flavours"]["norm"]["value"] = "batchnorm"
    out["flavours"]["qkv_bias"] = ev("no", "stated", NORM_Q, "")
    rep = L.verify(out, src(LLAMA_TEX))
    assert rep["flavours"]["norm"]["check"] == "unverified"
    assert rep["flavours"]["qkv_bias"]["check"] == "unverified"


def test_size_arithmetic_demotes():
    out = llama_output()
    out["configs"][0]["n_kv_heads"] = ev("24", "stated", ROW_SMALL, "Table 1")  # 24 does not divide 16
    rep = L.verify(out, src(LLAMA_TEX))
    c = rep["configs"][0]
    assert c["n_kv_heads"]["check"] == "unverified" and c["n_heads"]["check"] == "unverified"
    assert not all(x["ok"] for x in c["consistency"])


def test_inherited_values_checked_against_curated_table():
    out = base_output(inherits={"ancestor": "GPT-2", "where": "Sec. 2",
                                "quote": "We use the same model and architecture as GPT-2"})
    q = out["inherits"]["quote"]
    out["flavours"]["norm"] = ev("layernorm", "inherited", q, "Sec. 2")
    out["flavours"]["positional"] = ev("learned", "inherited", q, "Sec. 2")
    out["flavours"]["activation"] = ev("swiglu", "inherited", q, "Sec. 2")  # GPT-2 is gelu
    rep = L.verify(out, src(GPT_TEX))
    assert rep["inherits"]["resolved"] == "gpt-2"
    assert rep["flavours"]["norm"]["check"] == "verified"
    assert rep["flavours"]["positional"]["check"] == "verified"
    assert rep["flavours"]["activation"]["check"] == "unverified"
    ext = L.to_extraction("x", out, rep, {})
    assert ext["fields"]["norm"]["verdict"] == "inherited"
    assert ext["inherits"]["ancestor"] == "gpt-2"


def test_inherited_from_unknown_ancestor_is_unverified():
    tex = r"\section{Model} We follow the Chinchilla architecture exactly."
    out = base_output(inherits={"ancestor": "Chinchilla", "quote": "We follow the Chinchilla architecture",
                                "where": ""})
    out["flavours"]["norm"] = ev("rmsnorm", "inherited", "We follow the Chinchilla architecture", "")
    rep = L.verify(out, src(tex))
    assert rep["flavours"]["norm"]["check"] == "unverified"
    assert "curated table" in rep["flavours"]["norm"]["reasons"][0]


def test_new_ops_expr_and_primitive():
    rep = L.verify(primer_output(), src(PRIMER_TEX))
    ops = {o["name"]: o for o in rep["new_ops"]}
    assert ops["squared_relu"]["check"] == "verified"
    assert ops["squared_relu"]["expr_check"]["canonical"] == "(relu(x)^2)"
    assert ops["squared_relu"]["expr_check"]["samples"]["output"] == [0.0, 0.0, 0.0, 0.25, 4.0]
    assert ops["mdha_dconv"]["check"] == "verified"
    assert rep["flavours"]["activation"]["check"] == "quote_only"
    assert rep["summary"]["node_errors"] == []


@pytest.mark.parametrize("mutate,err", [
    (lambda o: o["new_ops"][0].update(expr="relu(x)**2"), "does not parse"),
    (lambda o: o["new_ops"][0].update(expr="relu(y)^2"), "undeclared"),
    (lambda o: o["new_ops"][0].update(expr="squash(x)"), "unknown function"),
    (lambda o: o["new_ops"][1].update(reason=""), "without a reason"),
    (lambda o: o["new_ops"][1].update(expr="x"), "carries an expr"),
    (lambda o: o["new_ops"][0].update(quote="squared ReLU is the best activation ever found"), "quote not found"),
])
def test_bad_new_ops_flagged(mutate, err):
    out = primer_output()
    mutate(out)
    rep = L.verify(out, src(PRIMER_TEX))
    msgs = [e for o in rep["summary"]["new_op_errors"] for e in o["errors"]]
    assert any(err in m for m in msgs), msgs


def test_dangling_new_op_references():
    out = primer_output()
    out["flavours"]["activation"]["value"] = "new_op:cubed_relu"
    out["block_ops"][1]["new_op"] = "cubed_relu"
    out["block_ops"][2]["op"] = "expr"  # names a needs_new_primitive op
    rep = L.verify(out, src(PRIMER_TEX))
    assert rep["flavours"]["activation"]["check"] == "unverified"
    errs = {n["node"]: n["errors"] for n in rep["summary"]["node_errors"]}
    assert "not defined" in errs[1][0] and "needs_new_primitive" in errs[2][0]


# --------------------------------------------------------------------------
# extraction.json -> tools/arxiv_to_chat.py

SIZE = {"d": 64, "layers": 2, "heads": 2, "d_ff": 0, "T": 64}


def test_extraction_feeds_arxiv_to_chat_llama():
    import arxiv_to_chat as A
    out = llama_output()
    rep = L.verify(out, src(LLAMA_TEX))
    ext = L.to_extraction("2302.13971", out, rep, {"extractor": "papers/llm_extract.py"})
    assert ext["fields"]["d_model"] == {"value": 2048, "evidence": "1.3B & 2048 & 16 & 24 [Table 1]",
                                        "verdict": "used"}
    assert ext["fields"]["vocab_size"]["value"] == 32000
    assert {"residual", "attention", "n_kv_heads"} <= set(ext["unresolved"])
    assert ext["mechanism_scan"] == []
    assert ext["variants"][1] == {"d_model": 4096, "n_layers": 32, "n_heads": 32}
    family, arch, report = A.build_spec(ext, SIZE)
    assert family == "llama" and arch["custom"]["position"] == "rope"


def test_extraction_refuses_primer():
    import arxiv_to_chat as A
    out = primer_output()
    ext = L.to_extraction("2109.08668", out, L.verify(out, src(PRIMER_TEX)), {})
    assert ext["fields"]["activation"]["value"] == "expr:(relu(x)^2)"
    mechs = [m["mechanism"] for m in ext["mechanism_scan"]]
    assert any("squared_relu = (relu(x)^2)" in m for m in mechs)
    assert any("mdha_dconv" in m and "new primitive" in m for m in mechs)
    with pytest.raises(A.Refused) as r:
        A.build_spec(ext, SIZE)
    reasons = " | ".join(r.value.args[0])
    assert "mdha_dconv" in reasons and "no engine support" in reasons and "norm is unresolved" in reasons


def test_unverified_flavour_is_refused_by_name():
    import arxiv_to_chat as A
    out = llama_output()
    out["flavours"]["norm"]["quote"] = "a sentence the paper never wrote"
    ext = L.to_extraction("2302.13971", out, L.verify(out, src(LLAMA_TEX)), {})
    assert ext["fields"]["norm"]["verdict"] == "unverified"
    with pytest.raises(A.Refused, match="verdict 'unverified'"):
        A.build_spec(ext, SIZE)
    fam, _, rep = A.build_spec(ext, SIZE, {"norm": "rmsnorm"})  # the operator settles it
    assert fam == "llama"


def test_grouped_query_and_moe_block():
    out = llama_output()
    out["configs"][0]["n_kv_heads"] = ev("16", "stated", ROW_SMALL, "Table 1")
    out["configs"][0]["n_heads"] = ev("16", "stated", ROW_SMALL, "Table 1")
    out["flavours"]["moe"] = ev("true", "stated", NORM_Q, "")
    ext = L.to_extraction("x", out, L.verify(out, src(LLAMA_TEX)), {})
    mechs = [m["mechanism"] for m in ext["mechanism_scan"]]
    assert "mixture of experts" in mechs and "grouped or multi-query attention" not in mechs


# --------------------------------------------------------------------------
# the API path, mocked

class FakeStream:
    def __init__(self, msg):
        self.msg, self.request_id = msg, "req_test"

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return self.msg


class FakeMessages:
    def __init__(self, replies, n_tokens=1000):
        self.replies, self.n_tokens = list(replies), n_tokens
        self.calls, self.counts = [], 0

    def count_tokens(self, **kw):
        self.counts += 1
        n = self.n_tokens(kw) if callable(self.n_tokens) else self.n_tokens
        return SimpleNamespace(input_tokens=n)

    def stream(self, **kw):
        self.calls.append(kw)
        return FakeStream(self.replies.pop(0))


def fake_client(replies, n_tokens=1000):
    return SimpleNamespace(beta=SimpleNamespace(messages=FakeMessages(replies, n_tokens)))


def reply(output, usage=None, stop="end_turn", model="claude-opus-5-5", details=None):
    content = [SimpleNamespace(type="thinking", thinking=""),
               SimpleNamespace(type="text", text=json.dumps(output) if not isinstance(output, str) else output)]
    return SimpleNamespace(content=content, usage=usage or {"input_tokens": 1000, "output_tokens": 2000},
                           model=model, stop_reason=stop, stop_details=details)


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "LLM_CACHE", tmp_path / "llm")
    return tmp_path / "ledger.jsonl"


def test_request_shape():
    p = L.request_params("TEX", "1234.5678")
    assert p["model"] == "claude-opus-5-5" and p["thinking"] == {"type": "adaptive"}
    assert p["output_config"]["effort"] == "high"
    assert "format" not in p["output_config"]  # the schema is too large to compile: it rides in the prompt
    assert json.dumps(L.SCHEMA, separators=(",", ":")) in p["system"]
    assert p["betas"] == [L.FALLBACK_BETA] and p["fallbacks"] == [{"model": L.FALLBACK_MODEL}]
    assert "temperature" not in p and "budget_tokens" not in json.dumps(p)
    assert '<paper_latex arxiv_id="1234.5678">\nTEX</paper_latex>' in p["messages"][0]["content"]
    assert "fallbacks" not in L.request_params("T", "x", fallback=False)


def test_schema_is_strict_everywhere():
    def walk(s):
        if isinstance(s, dict):
            if s.get("type") == "object":
                assert s["additionalProperties"] is False
                assert sorted(s["required"]) == sorted(s["properties"])
            for v in s.values():
                walk(v)
        elif isinstance(s, list):
            for v in s:
                walk(v)
    walk(L.SCHEMA)
    assert set(L.SCHEMA["properties"]["flavours"]["properties"]) == set(L.FLAVOUR_FIELDS)


def test_mocked_extraction_receipt_and_cache(isolated):
    client = fake_client([reply(llama_output())])
    ext = L.run_extraction("2302.13971", LLAMA_TEX, client, "t", 5.0, ledger=isolated)
    assert ext["fields"]["norm"]["verdict"] == "used"
    assert ext["source"]["cost_usd"] == pytest.approx(1000 * 4 / 1e6 + 2000 * 20 / 1e6)
    assert ext["source"]["sent_sha256"] == L.sha256_text(L.prepare_source(LLAMA_TEX))
    row = json.loads(isolated.read_text().splitlines()[0])
    assert row["request_id"] == "req_test" and row["served_by"] == "claude-opus-5-5"
    assert row["usage"]["output_tokens"] == 2000 and row["counted_input_tokens"] == 1000
    assert client.beta.messages.counts == 1  # counted once, reused for the budget check
    # second run: the cached reply, no stream call, no new spend
    client2 = fake_client([])
    ext2 = L.run_extraction("2302.13971", LLAMA_TEX, client2, "t", 5.0, ledger=isolated)
    assert ext2["fields"] == ext["fields"] and client2.beta.messages.calls == []
    rows = [json.loads(x) for x in isolated.read_text().splitlines()]
    assert rows[1]["cache_hit"] and rows[1]["cost_usd"] == 0.0
    assert L.ledger_spent("t", isolated) == pytest.approx(0.044)


def test_budget_refuses_before_sending(isolated):
    L.ledger_append({"tag": "pilot", "cost_usd": 4.5}, isolated)
    L.ledger_append({"tag": "other", "cost_usd": 100.0}, isolated)
    client = fake_client([reply(llama_output())], n_tokens=50_000)
    with pytest.raises(SystemExit, match="over the \\$5.00 cap"):
        L.run_extraction("2302.13971", LLAMA_TEX, client, "pilot", 5.0, ledger=isolated)
    assert client.beta.messages.calls == []
    # unknown cost (a dropped stream) counts at its worst case
    L.ledger_append({"tag": "z", "cost_usd": None, "worst_case_usd": 1.25}, isolated)
    assert L.ledger_spent("z", isolated) == 1.25


@pytest.mark.parametrize("msg,err", [
    (reply("{}", stop="refusal", details=SimpleNamespace(category="cyber", explanation="x")), "refusal"),
    (reply('{"title": "trunc', stop="max_tokens"), "truncated"),
    (reply("not json"), "not JSON"),
])
def test_bad_replies_are_receipted_and_fail(isolated, msg, err):
    with pytest.raises(SystemExit, match=err):
        L.run_extraction("2302.13971", LLAMA_TEX, fake_client([msg]), "t", None, ledger=isolated)
    row = json.loads(isolated.read_text().splitlines()[-1])
    assert err in row["error"] and row["cost_usd"] > 0
    assert not list((L.LLM_CACHE).glob("*.json"))  # a failed reply is never cached


def test_fallback_iterations_billed_per_model():
    usage = {"input_tokens": 0, "output_tokens": 0, "iterations": [
        {"type": "message", "model": "claude-opus-5-5", "input_tokens": 1000, "output_tokens": 10,
         "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
        {"type": "fallback_message", "model": "claude-opus-4-8", "input_tokens": 1000, "output_tokens": 2000,
         "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}]}
    want = (1000 * 4 + 10 * 20) / 1e6 + (1000 * 5 + 2000 * 25) / 1e6
    assert L.cost_usd(usage, "claude-opus-5-5") == pytest.approx(want)
    assert L.worst_case_usd(1000, L.request_params("x", "y")) == pytest.approx(
        (1000 * 4 + 32000 * 20) / 1e6 + (1000 * 5 + 32000 * 25) / 1e6)


def test_oversized_paper_is_split_and_merged(isolated):
    big = LLAMA_TEX + "\n\\section{Appendix}\n" + ("filler text. " * 1000)
    a = llama_output()
    b = base_output(flavours={k: ev() for k in L.FLAVOUR_FIELDS})
    b["flavours"]["qkv_bias"] = ev("false", "stated", NORM_Q, "")
    b["configs"] = [config("1.3B", n_kv_heads=ev("16", "stated", ROW_SMALL, "Table 1"))]
    # tokens ~ chars / 3; a 2,000-token "context" forces at least two parts
    client = fake_client([reply(a), reply(b), reply(b), reply(b), reply(b), reply(b)],
                         n_tokens=lambda kw: len(kw["messages"][0]["content"]) // 3)
    ext = L.run_extraction("2302.13971", big, client, "t", None, ledger=isolated, context_tokens=2000 + 32000 + 20000,
                           use_cache=False)
    calls = client.beta.messages.calls
    assert len(calls) >= 2 and "part 1 of" in calls[0]["messages"][0]["content"]
    rep = ext["llm"]["verification"]
    assert rep["flavours"]["norm"]["check"] == "verified"         # from part 1
    assert rep["flavours"]["qkv_bias"]["check"] == "quote_only"   # from a later part
    assert rep["configs"][0]["n_kv_heads"]["check"] == "verified"
    assert ext["source"]["n_calls"] == len(calls)


def test_key_comes_from_the_environment_only(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY is not set"):
        L.make_client()


# --------------------------------------------------------------------------
# CLI and comparison

def test_cli_verify(tmp_path, capsys):
    (tmp_path / "p.tex").write_text(PRIMER_TEX, encoding="utf-8")
    (tmp_path / "r.json").write_text(json.dumps(primer_output()), encoding="utf-8")
    rc = L.main(["verify", str(tmp_path / "r.json"), "--tex", str(tmp_path / "p.tex"), "--id", "2109.08668",
                 "--out", str(tmp_path / "e.json")])
    assert rc == 0
    ext = json.loads((tmp_path / "e.json").read_text(encoding="utf-8"))
    assert ext["arxiv_id"] == "2109.08668" and "squared_relu" in capsys.readouterr().out
    bad = primer_output()
    bad["new_ops"][0]["expr"] = "relu(x)**2"
    (tmp_path / "r.json").write_text(json.dumps(bad), encoding="utf-8")
    assert L.main(["verify", str(tmp_path / "r.json"), "--tex", str(tmp_path / "p.tex")]) == 1


def test_compare_with_rule_based():
    out = primer_output()
    cmp = L.compare("2109.08668", PRIMER_TEX, L.verify(out, src(PRIMER_TEX)))
    rows = {r["field"]: r for r in cmp["rows"]}
    # truth: activation outside the lattice (squared ReLU), norm rmsnorm
    assert rows["activation"]["llm_score"] == "correct"
    assert rows["activation"]["llm"] == "new_op:squared_relu"
    assert rows["norm"]["llm_score"] == "abstain"
    assert L.score("gelu", "relu") == "WRONG" and L.score(L.OUT, "rope") == "WRONG"
    assert L.score(("family", "gated-glu"), "swiglu") == "correct"


def test_validate_schema_rejects_malformed_replies():
    assert L.validate_schema({"value": "x", "status": "stated", "quote": "q", "where": "w"},
                             {"$ref": "#/$defs/ev"}, defs=L.SCHEMA["$defs"]) == []
    bad = {"value": "x", "status": "guessed", "quote": "q", "extra": 1}
    errs = L.validate_schema(bad, {"$ref": "#/$defs/ev"}, defs=L.SCHEMA["$defs"])
    assert any("guessed" in e for e in errs) and any("missing 'where'" in e for e in errs)
    assert any("unexpected 'extra'" in e for e in errs)
    assert L.validate_schema({"title": "t"}, L.SCHEMA)  # most properties missing
    assert L.parse_reply('Here it is:\n```json\n{"a": 1}\n```') == {"a": 1}
