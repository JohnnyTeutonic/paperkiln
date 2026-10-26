"""Offline tests for tools/arxiv_to_chat.py: cached extraction -> spec ->
refusal, and the stage bookkeeping with mtstudio mocked (no training, no
network, no servers). Runs in about a second.

    python3 -m pytest tools/test_arxiv_to_chat.py -q
"""
import json
import os
import pathlib
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import arxiv_to_chat as A  # noqa: E402

CACHE = os.path.join(HERE, "arxiv_chat")
SIZE = {"d": 64, "layers": 2, "heads": 2, "d_ff": 0, "T": 64}


def cached(pid):
    return os.path.join(CACHE, f"{pid}.extraction.json")


def run_cli(tmp_path, pid, *extra):
    # The spec/stage tests start from cached extractions; the relevance gate has its own tests below.
    return A.main([pid, "--run-dir", str(tmp_path / pid), "--extraction", cached(pid),
                   "--skip-relevance-gate", *extra])


def test_llama_paper_maps_to_llama_family(tmp_path):
    assert run_cli(tmp_path, "2302.13971", "--stop-after", "spec", "--d", "64", "--heads", "2") == 0
    run = tmp_path / "2302.13971"
    rep = json.loads((run / "spec_report.json").read_text())
    assert rep["family"] == "llama"
    pre = json.loads((run / "pretrain_spec.json").read_text())
    sft = json.loads((run / "sft_spec.json").read_text())
    assert pre["arch"]["rope_heads"] == "all"
    assert pre["arch"]["custom"] == {"position": "rope", "d": 64, "layers": 2, "heads": 2}
    assert sft["train"]["objective"] == "sft"
    assert pathlib.Path(sft["train"]["init_from"]).parts[-2:] == ("pretrain", "arxiv_2302_13971_base.safetensors")
    buckets = {b for b, _ in rep["report"]}
    assert {"MATCHED", "DEFAULT", "SIZED"} <= buckets
    man = json.loads((run / "manifest.json").read_text())
    assert man["stages"]["spec"]["family"] == "llama" and len(man["stages"]["extract"]["extraction_sha256"]) == 64


def test_gpt3_like_paper_maps_to_flex(tmp_path):
    # Cerebras-GPT inherits GPT-3's flavours and says it uses dense attention
    # ("unlike GPT-3 ..."): the sparse-attention cue is noted, not blocking.
    assert run_cli(tmp_path, "2304.03208", "--stop-after", "spec") == 0
    rep = json.loads((tmp_path / "2304.03208" / "spec_report.json").read_text())
    assert rep["family"] == "flex"
    custom = json.loads((tmp_path / "2304.03208" / "pretrain_spec.json").read_text())["arch"]["custom"]
    assert (custom["norm"], custom["activation"], custom["position"], custom["attention"]) == \
        ("layernorm", "gelu", "learned", "exact")
    assert any(b == "NOTED" and "sparse" in d for b, d in rep["report"])


@pytest.mark.parametrize("pid,why", [("2312.00752", "state-space model"),
                                     ("2005.14165", "sparse or banded attention")])
def test_refusals(tmp_path, pid, why, capsys):
    assert run_cli(tmp_path, pid) == A.EXIT_REFUSED
    ref = json.loads((tmp_path / pid / "refusal.json").read_text())
    assert any(why in r for r in ref["reasons"])
    assert not (tmp_path / pid / "pretrain_spec.json").exists()
    assert "REFUSED" in capsys.readouterr().out
    assert json.loads((tmp_path / pid / "manifest.json").read_text())["stages"]["spec"]["refused"]


def ext_with(**fields):
    f = {k: {"value": v, "evidence": "x", "verdict": "used", "score": 3.0} for k, v in fields.items()}
    return {"arxiv_id": "0000.00000", "fields": f, "mentions": {}, "mechanism_scan": []}


def test_unresolved_flavour_refused_until_asserted():
    ext = ext_with(activation="gelu", positional="learned")
    with pytest.raises(A.Refused, match="norm is unresolved"):
        A.build_spec(ext, SIZE)
    fam, arch, rep = A.build_spec(ext, SIZE, {"norm": "layernorm"})
    assert fam == "flex" and arch["custom"]["norm"] == "layernorm"
    assert any(b == "ASSERTED" for b, _ in rep)


def test_contested_and_unsupported_refused():
    ext = ext_with(norm="layernorm", activation="gelu", positional="learned")
    ext["fields"]["norm"]["verdict"] = "contested"
    with pytest.raises(A.Refused, match="contested"):
        A.build_spec(ext, SIZE)
    with pytest.raises(A.Refused, match="no engine support"):
        A.build_spec(ext_with(norm="rmsnorm", activation="geglu", positional="rope"), SIZE)
    # RoPE with flavours the llama block lacks: flex with position rope.
    fam, arch, _ = A.build_spec(ext_with(norm="layernorm", activation="gelu", positional="rope"), SIZE)
    assert fam == "flex" and arch["custom"]["position"] == "rope" and "rope_heads" not in arch
    # Grouped-query attention: the llama family keeps the paper's ratio when the run's heads allow it.
    with pytest.raises(A.Refused, match="n_kv_heads = 8 of n_heads = 32.*multiple of 4"):
        A.build_spec(ext_with(norm="rmsnorm", activation="swiglu", positional="rope", n_heads=32,
                              n_kv_heads=8), SIZE)


def test_grouped_query_and_rope_variants_build_on_llama():
    ext = ext_with(norm="rmsnorm", activation="swiglu", positional="rope", n_heads=32, n_kv_heads=8)
    ext["mechanism_scan"] = [{"mechanism": "grouped or multi-query attention", "where": "abstract", "blocking": True,
                              "evidence": "we use grouped-query attention"}]
    fam, arch, rep = A.build_spec(ext, dict(SIZE, heads=8))
    assert fam == "llama" and arch["custom"]["kv_heads"] == 2
    assert any(b == "NOTED" and "built from n_kv_heads" in d for b, d in rep)
    # Multi-query (one kv head) stays one at any head count.
    fam, arch, _ = A.build_spec(ext_with(norm="rmsnorm", activation="swiglu", positional="rope", n_heads=16,
                                         n_kv_heads=1), SIZE)
    assert arch["custom"]["kv_heads"] == 1
    # The GQA cue without n_kv_heads cannot be sized: still refused.
    ext = ext_with(norm="rmsnorm", activation="swiglu", positional="rope")
    ext["mechanism_scan"] = [{"mechanism": "grouped or multi-query attention", "where": "abstract", "blocking": True,
                              "evidence": "we use grouped-query attention"}]
    with pytest.raises(A.Refused, match="grouped or multi-query attention"):
        A.build_spec(ext, SIZE)
    # A stated theta, NEOX pairs and a partial rotary fraction on the llama family.
    ext = ext_with(norm="rmsnorm", activation="swiglu", positional="rope")
    ext["llm"] = {"output": {"flavours": {}}, "verification": {"flavours": {
        "rope_theta": {"value": "500000", "check": "verified", "status": "stated", "where": "Sec. 3"},
        "rope_pairing": {"value": "neox", "check": "quote_only", "status": "stated", "where": "Sec. 3"},
        "rope_fraction": {"value": "0.25", "check": "verified", "status": "stated", "where": "Table 1"}}}}
    fam, arch, rep = A.build_spec(ext, dict(SIZE, d=128, heads=4))
    assert fam == "llama"
    c = arch["custom"]
    assert (c["rope_theta"], c["rope_pairing"], c["rope_dims"]) == (500000.0, "neox", 8)
    # flex rotates interleaved with theta 10000 only: the same statements refuse there.
    ext["fields"]["norm"]["value"] = "layernorm"
    with pytest.raises(A.Refused, match="only the llama family builds it"):
        A.build_spec(ext, dict(SIZE, d=128, heads=4))


def test_swa_window_scaled_and_sinks_explicit():
    ext = ext_with(norm="layernorm", activation="gelu", positional="learned", attention="swa", window_size=4096)
    fam, arch, rep = A.build_spec(ext, SIZE)
    assert fam == "flex" and arch["custom"]["window"] == 16 and arch["custom"]["sinks"] == 0
    assert any(b == "SCALED" for b, _ in rep)
    del ext["fields"]["window_size"]
    with pytest.raises(A.Refused, match="window"):
        A.build_spec(ext, SIZE)


def test_missing_mechanism_scan_refused_unless_skipped():
    ext = ext_with(norm="rmsnorm", activation="swiglu", positional="rope")
    del ext["mechanism_scan"]
    with pytest.raises(A.Refused, match="mechanism scan"):
        A.build_spec(ext, SIZE)
    _, _, rep = A.build_spec(ext, SIZE, skip_scan=True)
    assert any(b == "UNCHECKED" for b, _ in rep)


def test_stages_with_mtstudio_mocked_and_restartable(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello world\n")
    sft_src = tmp_path / "sft.jsonl"
    sft_src.write_text("".join(json.dumps({"prompt": f"<|user|>q{i}<|assistant|>", "completion": "a"}) + "\n"
                               for i in range(5)))
    calls = []

    def fake(a, args, log_path):
        calls.append(args[0:2])
        spec = json.loads(open(args[1]).read())
        os.makedirs(spec["out_dir"], exist_ok=True)
        open(os.path.join(spec["out_dir"], spec["name"] + ".safetensors"), "wb").write(b"w" + spec["name"].encode())
        json.dump({"steps": spec["train"]["steps"]}, open(os.path.join(spec["out_dir"], "result.json"), "w"))
        json.dump(spec, open(os.path.join(spec["out_dir"], "spec.json"), "w"))

    monkeypatch.setattr(A, "mtstudio", fake)
    if not os.path.exists(VOCAB):
        pytest.skip("the SFT slice is tokenised with the bpe8k vocabulary, not on disk")
    args = ["--corpus", str(corpus), "--vocab", VOCAB, "--sft-examples", str(sft_src), "--sft-limit", "3",
            "--stop-after", "sft", "--emit-jobs", str(tmp_path / "jobs.json")]
    assert run_cli(tmp_path, "2302.13971", *args) == 0
    run = tmp_path / "2302.13971"
    assert len((run / "sft_examples.jsonl").read_text().splitlines()) == 3
    man = json.loads((run / "manifest.json").read_text())
    assert len(man["stages"]["sft"]["sft_slice_sha256"]) == 64 and man["stages"]["pretrain"]["corpus_sha256"]
    assert calls == [["run", str(run / "pretrain_spec.json")], ["run", str(run / "sft_spec.json")]]
    # Restart: every stage's output exists, so nothing runs again.
    assert run_cli(tmp_path, "2302.13971", *args) == 0
    assert len(calls) == 2
    jobs = json.loads((tmp_path / "jobs.json").read_text())["jobs"]
    assert [j["name"] for j in jobs] == ["base_2302_13971", "sft_2302_13971"]
    assert jobs[1]["after"] == ["base_2302_13971"] and jobs[1]["init_job"] == "base_2302_13971"
    assert "after" not in jobs[0]
    assert jobs[0]["spec"]["data"]["corpus"].startswith("/content/pk/")
    assert "init_from" not in jobs[1]["spec"]["train"] and "out_dir" not in jobs[0]["spec"]
    # Riding in another launch: absolute VM paths, an .ids corpus read whole, extra dependencies.
    args2 = [a if a != str(tmp_path / "jobs.json") else str(tmp_path / "jobs2.json") for a in args]
    assert run_cli(tmp_path, "2302.13971", *args2, "--vm-corpus", "/content/chat/data/mix.main.ids",
                   "--vm-sft-upload", "/content/chat/lever1/lever1_sft_v1.jsonl", "--vm-after", "data",
                   "--vm-sft-after", "lever1") == 0
    j2 = json.loads((tmp_path / "jobs2.json").read_text())["jobs"]
    assert j2[0]["after"] == ["data"] and j2[1]["after"] == ["base_2302_13971", "lever1"]
    assert j2[0]["spec"]["data"]["corpus"] == "/content/chat/data/mix.main.ids"
    assert j2[0]["spec"]["data"]["max_tokens"] == 0
    assert j2[1]["spec"]["data"]["sft_examples"] == "/content/chat/lever1/lever1_sft_v1.jsonl"


VOCAB = os.path.join(os.path.dirname(HERE), "corpus", "data", "vocab_fineweb_tinychat_bpe8k_v1",
                     "fineweb_tinychat_bpe8k.gguf")


@pytest.mark.skipif(not os.path.exists(VOCAB), reason="the bpe8k vocabulary is not on disk")
def test_sft_examples_longer_than_T_dropped_and_counted(tmp_path):
    import bpe

    tok = bpe.read_gguf(VOCAB)
    rows = [{"prompt": "<|user|>hi<|assistant|>", "completion": "hello there"},
            {"prompt": "<|user|>" + "tell me a long story " * 20 + "<|assistant|>", "completion": "no"},
            {"prompt": "<|user|>ok<|assistant|>", "completion": "fine " * 30},
            {"prompt": "<|user|>bye<|assistant|>", "completion": "goodbye"},
            {"prompt": "<|user|>one more<|assistant|>", "completion": "sure"}]
    src = tmp_path / "sft.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    T = 16
    lens = [len(tok.encode(r["prompt"])) + len(tok.encode(r["completion"])) + 1 for r in rows]
    fits = [n <= T + 1 for n in lens]
    assert fits == [True, False, False, True, True], lens
    fit = A.sft_fit(str(src), VOCAB, T, 2, str(tmp_path / "out.jsonl"))
    kept = [json.loads(x) for x in (tmp_path / "out.jsonl").read_text(encoding="utf-8").splitlines()]
    assert kept == [rows[0], rows[3]]  # file order, up to the limit, the long ones skipped
    assert (fit["seen"], fit["kept"], fit["dropped_longer_than_T"]) == (4, 2, 2)
    assert fit["longest_kept_tokens"] <= T + 1 < fit["longest_dropped_tokens"]
    fit = A.sft_fit(str(src), VOCAB, T, 0, str(tmp_path / "all.jsonl"))
    assert (fit["seen"], fit["kept"], fit["dropped_longer_than_T"]) == (5, 3, 2)


# --------------------------------------------------------------------------
# LLM extractions (papers/llm_extract.py): a paper's new op as an expr

RECEIPTS = os.path.join(os.path.dirname(HERE), "experiments", "arxiv_chain_2026-10-23")


def receipt(pid):
    with open(os.path.join(RECEIPTS, f"{pid}.extraction.json"), encoding="utf-8") as f:
        return json.load(f)


def llm_ext(expr="relu(x)^2", kind="expr", slot="activation", inputs=("x",), ok=True, errors=(),
            stated=None, inherits=None):
    """An LLM extraction as to_extraction writes it: one new op in `slot`,
    the other flavours stated (or as `stated` gives them)."""
    stated = {"norm": "layernorm", "positional": "learned"} if stated is None else stated
    canonical = expr if not ok else f"({expr})" if kind == "expr" else ""
    v_flav = {k: {"value": v, "check": "verified", "status": "stated"} for k, v in stated.items()}
    v_flav[slot] = {"value": "new_op:op1", "check": "verified", "status": "stated"}
    fields = {k: {"value": v, "evidence": "q", "verdict": "used"} for k, v in stated.items()}
    fields[slot] = {"value": ("expr:" + canonical) if kind == "expr" and ok else "new_primitive:op1",
                    "evidence": "q", "verdict": "used"}
    op = {"name": "op1", "kind": kind, "replaces": "FFN activation", "reason": "" if kind == "expr" else
          "a causal convolution over time", "quote": "the paper's sentence", "where": "Sec. 2",
          "inputs": [{"name": n, "meaning": "m"} for n in inputs], "errors": list(errors), "quote_found": True,
          "check": "verified" if not errors else "unverified"}
    if kind == "expr":
        op["expr_check"] = {"expr": expr, "ok": ok, "errors": list(errors), "canonical": canonical}
    out_flav = {k: {"status": "stated"} for k in v_flav}
    return {"arxiv_id": "0000.00001", "fields": fields, "mentions": {},
            "mechanism_scan": [{"mechanism": f"new op op1 = {canonical} (expr)" if kind == "expr" else
                                "new op op1 (needs a new primitive: a causal convolution over time)",
                                "where": "llm extraction, verified quote", "blocking": True, "evidence": "q"}],
            "llm": {"output": {"flavours": out_flav, "family": {"status": "stated"}},
                    "verification": {"new_ops": [op], "flavours": v_flav, "inherits": inherits}}}


def test_llm_expr_activation_hosted_on_flex():
    fam, arch, rep = A.build_spec(llm_ext("relu(x)^2"), SIZE)
    assert fam == "flex" and arch["custom"]["activation"] == "expr:(relu(x)^2)"
    assert (arch["custom"]["norm"], arch["custom"]["position"]) == ("layernorm", "learned")
    assert any(b == "EXPR" and "finite in f32" in d for b, d in rep)
    # The new op's blocking scan entry is judged from its record, not refused as a mechanism.
    assert not any("outside the extractor's lattice" in d for _, d in rep)


def test_llm_expr_bad_expressions_refused():
    with pytest.raises(A.Refused, match="unverified.*does not parse"):
        A.build_spec(llm_ext("x + * 2", ok=False, errors=["does not parse: unexpected '*'"]), SIZE)
    # Verified by the extractor's sampler but not finite in f32 over the probe: never trained.
    with pytest.raises(A.Refused, match="non-finite in f32 at x = -20"):
        A.build_spec(llm_ext("sqrt(x)"), SIZE)
    with pytest.raises(A.Refused, match="non-finite"):
        A.build_spec(llm_ext("exp(x^2)"), SIZE)
    # A second input whose role is unknown ("m", no role): neither a gate nor a learned input.
    with pytest.raises(A.Refused, match="reads 2 inputs.*a = m \\(other\\)"):
        A.build_spec(llm_ext("relu(a)*x", inputs=("x", "a")), SIZE)
    # A norm expr whose extra input is neither learned nor the row (here "m").
    with pytest.raises(A.Refused, match="reads 2 inputs.*the row \\(input\\) and learned"):
        A.build_spec(llm_ext("(x-rowmean(x))*a", slot="norm", inputs=("x", "a"),
                             stated={"activation": "gelu", "positional": "learned"}), SIZE)
    # An expr value with no new-op record behind it (a hand-written extraction).
    ext = ext_with(norm="layernorm", activation="expr:(relu(x)^2)", positional="learned")
    with pytest.raises(A.Refused, match="no verified new-op record"):
        A.build_spec(ext, SIZE)


def test_llm_needs_new_primitive_refused():
    with pytest.raises(A.Refused, match="needs a new primitive.*causal convolution"):
        A.build_spec(llm_ext(kind="needs_new_primitive"), SIZE)


def test_llm_expr_norm_hosted_on_flex():
    # One input: flex's e(x) * g + b over the row.
    fam, arch, rep = A.build_spec(llm_ext("x/sqrt(rowmean(x^2)+1e-5)", slot="norm",
                                          stated={"activation": "gelu", "positional": "learned"}), SIZE)
    assert fam == "flex" and arch["custom"]["norm"] == "expr:(x/sqrt(rowmean(x^2)+1e-5))"
    assert any(b == "NORM-AFFINE" for b, _ in rep)
    # Primer's form: the row expression, then the paper's own learned per-channel shift a and gain b,
    # (core + a) * b = core * b + a * b: flex's gain and bias, another parametrisation.
    primer = "((x-rowmean(x))/sqrt(abs(rowmean(x*(x-rowmean(x)))))+a)*b"
    ext = llm_ext(primer, slot="norm", inputs=("x", "a", "b"), stated={"activation": "gelu", "positional": "rope"})
    for i in ext["llm"]["verification"]["new_ops"][0]["inputs"][1:]:
        i["meaning"] = "learned per-channel vector"
    fam, arch, rep = A.build_spec(ext, SIZE)
    assert fam == "flex" and arch["custom"]["position"] == "rope"
    assert arch["custom"]["norm"] == "expr:((x-rowmean(x))/sqrt(abs(rowmean((x*(x-rowmean(x)))))))"
    assert any(b == "NORM-AFFINE" and "(core + a)*g" in d for b, d in rep)
    # A learned vector inside the row expression stays one of its inputs (flex norm_inputs); the trailing
    # gain b is flex's gain.
    ext = llm_ext("(x-rowmean(x)*a)*b", slot="norm", inputs=("x", "a", "b"), stated={"activation": "gelu",
                                                                                    "positional": "learned"})
    for i in ext["llm"]["verification"]["new_ops"][0]["inputs"][1:]:
        i["meaning"] = "learned per-channel vector"
    fam, arch, rep = A.build_spec(ext, SIZE)
    assert arch["custom"]["norm"] == "expr:(x-(rowmean(x)*a))"
    assert arch["custom"]["norm_inputs"] == ["row", {"kind": "channel", "init": 1.0}]
    assert any(b == "NORM-AFFINE" and "core*g" in d and "bias b" in d for b, d in rep)
    # A learned vector both inside the expression and in its affine tail cannot be split.
    ext = llm_ext("(x*b)*b", slot="norm", inputs=("x", "b"), stated={"activation": "gelu", "positional": "learned"})
    ext["llm"]["verification"]["new_ops"][0]["inputs"][1]["meaning"] = "learned per-channel vector"
    with pytest.raises(A.Refused, match="also enters the row expression"):
        A.build_spec(ext, SIZE)


def test_llm_expr_activation_with_rope_is_flex():
    fam, arch, rep = A.build_spec(llm_ext("relu(x)^2", stated={"norm": "rmsnorm", "positional": "rope"}), SIZE)
    assert fam == "flex" and arch["custom"] == {"norm": "rmsnorm", "activation": "expr:(relu(x)^2)",
                                                "position": "rope", "residual": "residual",
                                                "attention": "exact", "d": 64, "layers": 2, "heads": 2}
    # A stated rope variant the families here do not rotate with refuses.
    ext = llm_ext("relu(x)^2", stated={"norm": "rmsnorm", "positional": "rope"})
    ext["llm"]["verification"]["flavours"]["rope_theta"] = {"value": "500000", "check": "verified",
                                                           "status": "stated", "where": "Sec. 3"}
    with pytest.raises(A.Refused, match="rope_theta = 500000"):
        A.build_spec(ext, SIZE)
    # A stated post-norm builds on flex (both sublayers); a placement flex does not build refuses.
    ext = llm_ext("relu(x)^2", stated={"norm": "rmsnorm", "positional": "rope"})
    ext["llm"]["verification"]["flavours"]["norm_placement"] = {"value": "post", "check": "quote_only",
                                                               "status": "stated", "where": "Sec. 3"}
    fam, arch, rep = A.build_spec(ext, SIZE)
    assert fam == "flex" and arch["custom"]["norm_placement"] == "post"
    ext["llm"]["verification"]["flavours"]["norm_placement"]["value"] = "other:between the two sublayers"
    with pytest.raises(A.Refused, match="norm_placement = other:between.*no family here builds that placement"):
        A.build_spec(ext, SIZE)


def test_relu2wins_receipt_builds_flex_rope():
    fam, arch, rep = A.build_spec(receipt("2402.03804"), SIZE)
    assert fam == "flex"
    assert {k: arch["custom"][k] for k in ("norm", "activation", "position", "residual", "attention")} == {
        "norm": "rmsnorm", "activation": "expr:(relu(x)^2)", "position": "rope", "residual": "residual",
        "attention": "exact"}
    # Residual and attention: inherited without a curated entry to check them, and the defaults.
    assert sorted(d.split(" ")[0] for b, d in rep if b == "DEFAULT") == ["attention", "residual"]
    assert any(b == "NOTED" and d.startswith("qkv_bias = false") for b, d in rep)


def test_primer_receipt_refused_with_every_reason(tmp_path):
    ext = receipt("2109.08668")
    with pytest.raises(A.Refused) as r:
        A.build_spec(ext, SIZE)
    reasons = " | ".join(r.value.args[0])
    assert "causal_depthwise_conv3 needs a new primitive" in reasons
    assert "relative" in reasons  # T5's relative attention positions: no engine support
    # The cached reply (22 Oct 2026) describes the placement in free text ("other:..."), which nothing builds;
    # a reply with the per-sublayer fields builds it (test_primer_placement_builds_from_per_sublayer_fields).
    assert "norm_placement = other:" in reasons
    # Its norm and squared ReLU are hostable now; the paper is refused for the rest.
    assert "primer_norm" not in reasons and "squared_relu" not in reasons


def test_golu_receipt_builds_flex_with_inherited_flavours():
    fam, arch, rep = A.build_spec(receipt("2502.03654"), SIZE)
    assert fam == "flex" and arch["custom"]["activation"] == "expr:(x*exp((-exp((-x)))))"
    assert (arch["custom"]["norm"], arch["custom"]["position"]) == ("layernorm", "learned")
    inh = [d for b, d in rep if b == "INHERITED"]
    assert len(inh) == 2 and all("fetch.ANCESTORS['gpt-2']" in d for d in inh)


def test_ancestor_fill_needs_a_verified_inheritance_the_reply_uses():
    inh = {"ancestor": "GPT-2", "resolved": "gpt-2", "quote": "we train GPT-2 small", "where": "Sec. 3",
           "quote_found": True}
    ext = llm_ext("relu(x)^2", stated={}, inherits=inh)
    # The reply names the ancestor but marks nothing inherited from it: no lookup.
    with pytest.raises(A.Refused, match="norm is unresolved"):
        A.build_spec(ext, SIZE)
    ext["llm"]["output"]["family"]["status"] = "inherited"
    fam, arch, _ = A.build_spec(ext, SIZE)
    assert fam == "flex" and arch["custom"]["norm"] == "layernorm" and arch["custom"]["position"] == "learned"
    # An inheritance quote not found in the source: no lookup.
    ext["llm"]["verification"]["inherits"] = dict(inh, quote_found=False)
    with pytest.raises(A.Refused, match="positional is unresolved"):
        A.build_spec(ext, SIZE)


# --------------------------------------------------------------------------
# LOCAL_LLM_STACK 11 item 4.2 (26 Oct 2026): expr inputs, placements, T5's bias, q/k norm, Mamba-1

def with_roles(ext, *roles, inits=None):
    """Give the extraction's one new op its inputs' roles (and stated, verified inits)."""
    for k, (i, r) in enumerate(zip(ext["llm"]["verification"]["new_ops"][0]["inputs"], roles)):
        i["role"] = r
        v = (inits or {}).get(i["name"])
        if v is not None:
            i["init"] = {"value": str(v), "status": "stated", "quote": f"we initialise it to {v}", "where": "Sec. 3"}
            i["init_check"] = "verified"
    return ext


def test_gated_and_parameterised_activations_build_on_flex():
    # A gate projection: ReGLU as relu(a) * x.
    ext = with_roles(llm_ext("relu(a)*x", inputs=("x", "a")), "input", "gate")
    fam, arch, rep = A.build_spec(ext, SIZE)
    assert fam == "flex" and arch["custom"]["activation"] == "expr:(relu(a)*x)"
    assert arch["custom"]["activation_inputs"] == ["up", "gate"]
    assert any(b == "EXPR-INPUTS" and "mlp.c_gate" in d for b, d in rep)
    # A learned scalar inside the expression (Swish-beta), its stated init kept.
    ext = with_roles(llm_ext("x*sigmoid(a*x)", inputs=("x", "a")), "input", "learned_scalar", inits={"a": 1.5})
    fam, arch, rep = A.build_spec(ext, SIZE)
    assert arch["custom"]["activation_inputs"] == ["up", {"kind": "scalar", "init": 1.5}]
    # Unstated: 1.0, reported.
    ext = with_roles(llm_ext("x*sigmoid(a*x)", inputs=("x", "a")), "input", "learned_scalar")
    _, arch, rep = A.build_spec(ext, SIZE)
    assert arch["custom"]["activation_inputs"][1] == {"kind": "scalar", "init": 1.0}
    assert any(b == "EXPR-INPUTS" and "not stated: 1.0" in d for b, d in rep)
    # Two up-projections, or an input of role other, refuse.
    with pytest.raises(A.Refused, match="reads 2 inputs"):
        A.build_spec(with_roles(llm_ext("x*a", inputs=("x", "a")), "input", "input"), SIZE)
    with pytest.raises(A.Refused, match="reads 2 inputs"):
        A.build_spec(with_roles(llm_ext("x*a", inputs=("x", "a")), "input", "other"), SIZE)
    # The probe holds a learned input at its init: x^a at a = 0.5 is not finite for x < 0.
    with pytest.raises(A.Refused, match="non-finite.*learned inputs at their init"):
        A.build_spec(with_roles(llm_ext("x^a", inputs=("x", "a")), "input", "learned_scalar", inits={"a": 0.5}),
                     SIZE)


def test_dyt_norm_builds_with_its_learned_alpha():
    # DyT (arXiv 2503.10622): gamma * tanh(alpha x) + beta; alpha a learned scalar, gamma and beta per channel.
    ext = llm_ext("tanh(a*x)*g+b", slot="norm", inputs=("x", "a", "g", "b"),
                  stated={"activation": "gelu", "positional": "learned"})
    with_roles(ext, "input", "learned_scalar", "learned_vector", "learned_vector", inits={"a": 0.5})
    fam, arch, rep = A.build_spec(ext, SIZE)
    assert fam == "flex" and arch["custom"]["norm"] == "expr:tanh((a*x))"
    assert arch["custom"]["norm_inputs"] == ["row", {"kind": "scalar", "init": 0.5}]
    assert any(b == "NORM-AFFINE" and "core*g + a" in d and "the same form" in d for b, d in rep)


TONIGHT = os.path.join(os.path.dirname(HERE), "experiments", "arxiv_chain_2026-10-26")


def tonight(pid):
    with open(os.path.join(TONIGHT, f"{pid}.extraction.json"), encoding="utf-8") as f:
        return json.load(f)


def test_dyt_receipt_builds_with_per_place_alpha_and_the_embedding_scalar():
    # arXiv 2503.10622 (26 Oct 2026): DyT on LLaMA, alpha 0.8 in the attention sublayer's norm, 0.2 in
    # the FFN's and the head's (a second new op, the same expression), a learned scalar after the
    # embedding initialised to sqrt(d). Its activation is inherited without a quote naming LLaMA.
    ext = tonight("2503.10622")
    with pytest.raises(A.Refused, match="activation = swiglu has verdict 'unverified'"):
        A.build_spec(ext, SIZE)
    fam, arch, rep = A.build_spec(ext, SIZE, A.parse_asserts(["activation=swiglu"]))
    c = arch["custom"]
    assert fam == "flex" and c["norm"] == "expr:tanh((a*x))" and c["activation"] == "swiglu" and c["position"] == "rope"
    assert c["norm_inputs"] == ["row", {"kind": "scalar", "init": 0.8, "init_ffn": 0.2, "init_final": 0.2}]
    assert c["embed"] == "expr:(x*a)" and c["embed_inputs"] == ["row", {"kind": "scalar", "init": 8.0}]  # sqrt(64)
    assert any(b == "NORM-VARIANT" and "dyt_other" in d for b, d in rep)


def test_tonights_other_receipts_refuse_by_name():
    reasons = {}
    for pid in ("2109.08668", "2312.00752", "2505.09388", "1710.05941"):
        with pytest.raises(A.Refused) as r:
            A.build_spec(tonight(pid), SIZE)
        reasons[pid] = " | ".join(r.value.args[0])
    # Primer: its placement and norm build now; its depthwise convolutions and relative positions do not.
    assert "block op conv1d_causal" in reasons["2109.08668"] and "positional = relative" in reasons["2109.08668"]
    assert "norm_placement" not in reasons["2109.08668"] and "primer_layernorm" not in reasons["2109.08668"]
    # Mamba: built as Mamba-1 from N = 16 and E = 2, but the conv width is not stated and the paper's own
    # block adds a LayerNorm the block here lacks.
    assert "d_conv is unresolved" in reasons["2312.00752"] and "LayerNorm" in reasons["2312.00752"]
    assert "state-space model" not in reasons["2312.00752"].split("Optional")[0]
    # Qwen3: the headline is the MoE model; its QK-norm's width is not stated.
    assert "mixture of experts" in reasons["2505.09388"] and "does not say over which width" in reasons["2505.09388"]
    # Swish: hosted on convolutional networks (BatchNorm), outside every family here.
    assert "BatchNorm" in reasons["1710.05941"] or "convolutional" in reasons["1710.05941"]


def test_norm_placements_per_sublayer():
    # Primer's: normalisation before attention, the second one after the FFN (inside its branch).
    ext = llm_ext("relu(x)^2", stated={"norm": "layernorm", "positional": "learned"})
    fl = ext["llm"]["verification"]["flavours"]
    fl["norm_placement"] = {"value": "other:pre attention, after the FFN", "check": "quote_only", "status": "stated"}
    fl["norm_placement_attn"] = {"value": "pre", "check": "quote_only", "status": "stated", "where": "Sec. 2"}
    fl["norm_placement_ffn"] = {"value": "branch_post", "check": "quote_only", "status": "stated", "where": "Sec. 2"}
    fam, arch, rep = A.build_spec(ext, SIZE)
    assert fam == "flex" and arch["custom"]["norm_placement"] == {"attn": "pre", "ffn": "branch_post"}
    # The llama block's flavours with a sandwich placement are flex, not llama.
    ext = ext_with(norm="rmsnorm", activation="swiglu", positional="rope")
    ext["llm"] = {"verification": {"flavours": {"norm_placement": {"value": "sandwich", "check": "verified",
                                                                   "status": "stated"}}}}
    fam, arch, _ = A.build_spec(ext, SIZE)
    assert fam == "flex" and arch["custom"]["norm_placement"] == "sandwich"


def test_t5_relative_bias_refused_until_ember_runs_it():
    ext = ext_with(norm="layernorm", activation="relu", positional="t5_bias")
    with pytest.raises(A.Refused, match="t5_bias.*ember does not run position_bias yet"):
        A.build_spec(ext, SIZE)


def qk_ext(value, positional="rope"):
    ext = ext_with(norm="rmsnorm", activation="swiglu", positional=positional)
    ext["llm"] = {"verification": {"flavours": {"qk_norm": {"value": value, "check": "verified", "status": "stated",
                                                            "where": "Sec. 2"}}}}
    return ext


def test_qk_norm_builds_on_llama():
    fam, arch, rep = A.build_spec(qk_ext("per_head"), SIZE)
    assert fam == "llama" and arch["custom"]["qk_norm"] is True
    assert any(b == "MATCHED" and "qk_norm = per_head" in d for b, d in rep)
    with pytest.raises(A.Refused, match="does not say over which width.*--assert qk_norm=per_head"):
        A.build_spec(qk_ext("unspecified"), SIZE)
    fam, arch, rep = A.build_spec(qk_ext("unspecified"), SIZE, A.parse_asserts(["qk_norm=per_head"]))
    assert arch["custom"]["qk_norm"] is True and any("by the operator" in d for _, d in rep)
    with pytest.raises(A.Refused, match="whole width"):
        A.build_spec(qk_ext("full"), SIZE)
    fam, arch, _ = A.build_spec(qk_ext("none"), SIZE)
    assert "qk_norm" not in arch["custom"]
    # On flex (learned positions here) only the llama family builds it.
    with pytest.raises(A.Refused, match="qk_norm = per_head.*only the llama family builds it"):
        A.build_spec(qk_ext("per_head", positional="learned"), SIZE)


def mamba_ext(**sizes):
    ext = ext_with()
    ext["mechanism_scan"] = [{"mechanism": "state-space model", "where": "title/abstract", "blocking": True,
                              "evidence": "Mamba: linear-time sequence modelling with selective state spaces"}]
    fl = {"ssm": {"value": "mamba1", "check": "verified", "status": "stated", "quote": "selective SSM"}}
    for k, v in sizes.items():
        fl[k] = {"value": str(v), "check": "verified", "status": "stated", "quote": f"{k} {v}"}
    ext["llm"] = {"verification": {"family": {"value": "state_space", "check": "verified"}, "flavours": fl}}
    return ext


def test_mamba1_builds_from_its_sizes():
    fam, arch, rep = A.build_spec(mamba_ext(d_state=16, d_conv=4, expand=2, dt_rank="ceil(d_model/16)"), SIZE)
    assert fam == "mamba" and arch["preset"] == "mamba-tiny"
    assert arch["custom"] == {"ssm": "mamba1", "d": 64, "layers": 2, "d_state": 16, "d_conv": 4, "expand": 2,
                              "dt_rank": 4}
    assert any(b == "NOTED" and "built as Mamba-1" in d for b, d in rep)
    # The Mamba paper states N and E but not the conv width: refused until asserted; dt_rank by default.
    with pytest.raises(A.Refused, match="d_conv is unresolved.*--assert d_conv=VALUE"):
        A.build_spec(mamba_ext(d_state=16, expand=2), SIZE)
    fam, arch, rep = A.build_spec(mamba_ext(d_state=16, expand=2), SIZE, A.parse_asserts(["d_conv=4"]))
    assert arch["custom"]["d_conv"] == 4 and "dt_rank" not in arch["custom"]
    assert any(b == "DEFAULT" and d.startswith("dt_rank = ceil(d/16)") for b, d in rep)
    # A stated norm other than the block's RMSNorm refuses.
    ext = mamba_ext(d_state=16, d_conv=4, expand=2)
    ext["fields"]["norm"] = {"value": "layernorm", "evidence": "we choose LayerNorm", "verdict": "used"}
    with pytest.raises(A.Refused, match="normalises with RMSNorm"):
        A.build_spec(ext, SIZE)
    with pytest.raises(SystemExit, match="positive integer"):
        A.parse_asserts(["d_conv=four"])


# --------------------------------------------------------------------------
# the relevance gate before extraction (papers/llm_extract.py run_gate), client stubbed

GOLU_META = {"arxiv_id": "2502.03654", "title": "Gompertz Linear Units",
             "abstract": "We introduce GoLU, a novel self-gated activation function defined as GoLU(x) = x "
                         "Gompertz(x). Across language modelling and image classification it outperforms GELU."}


class _Stream:
    def __init__(self, msg):
        self.msg, self.request_id = msg, "req_gate"

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return self.msg


def gate_client(text):
    from types import SimpleNamespace as NS
    sent = []

    def stream(**kw):
        sent.append(kw)
        return _Stream(NS(content=[NS(type="text", text=text)], model="claude-opus-5-5", stop_reason="end_turn",
                          stop_details=None, usage={"input_tokens": 600, "output_tokens": 200}))
    msgs = NS(count_tokens=lambda **kw: NS(input_tokens=600), stream=stream, sent=sent)
    return NS(beta=NS(messages=msgs))


@pytest.fixture
def gated(tmp_path, monkeypatch):
    """The real gate with the API stubbed: metadata, client, cache and ledger in tmp_path."""
    import llm_extract as L
    monkeypatch.setattr(L, "LLM_CACHE", tmp_path / "llm")
    monkeypatch.setattr(L, "LEDGER", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(L, "arxiv_metadata", lambda aid, offline=False: dict(GOLU_META, arxiv_id=aid))
    state = {}

    def use(verdict, evidence="a novel self-gated activation function", raw=None):
        reply = raw if raw is not None else json.dumps({
            "verdict": verdict, "reason": f"judged {verdict}", "architecture_kind": "FFN activation function",
            "evidence": evidence})
        state["client"] = gate_client(reply)
        monkeypatch.setattr(L, "make_client", lambda: state["client"])
        return state["client"]

    def run(*extra):
        return A.main(["2502.03654", "--run-dir", str(tmp_path / "run"), "--extraction",
                       os.path.join(RECEIPTS, "2502.03654.extraction.json"), *extra])
    return SimpleNamespace(use=use, run=run, run_dir=tmp_path / "run", ledger=tmp_path / "ledger.jsonl")


from types import SimpleNamespace  # noqa: E402


def test_gate_in_scope_lets_extraction_run(gated):
    client = gated.use("in_scope")
    assert gated.run("--stop-after", "spec") == 0
    man = json.loads((gated.run_dir / "manifest.json").read_text())
    g = man["stages"]["gate"]
    assert g["verdict"] == "in_scope" and g["passes"] and g["evidence_found"]
    assert g["gate"]["cost_usd"] == pytest.approx((600 * 4 + 200 * 20) / 1e6)
    assert man["relevance_gate"] == "in_scope" and "spec" in man["stages"]
    assert (gated.run_dir / "extraction.json").exists() and not (gated.run_dir / "refusal.json").exists()
    assert len(client.beta.messages.sent) == 1
    assert json.loads(gated.ledger.read_text().splitlines()[0])["kind"] == "gate"


@pytest.mark.parametrize("verdict", ["out_of_scope", "spurious"])
def test_gate_refuses_before_extraction(gated, verdict, capsys):
    gated.use(verdict, evidence="Gompertz(x)")
    assert gated.run() == A.EXIT_REFUSED
    ref = json.loads((gated.run_dir / "refusal.json").read_text())
    assert ref["stage"] == "relevance_gate" and ref["verdict"] == verdict
    assert ref["reason"] == f"judged {verdict}" and ref["evidence"] == "Gompertz(x)" and ref["evidence_found"]
    assert any(verdict in r for r in ref["reasons"])
    assert not (gated.run_dir / "extraction.json").exists()  # nothing extracted, nothing trained
    man = json.loads((gated.run_dir / "manifest.json").read_text())
    assert man["stages"]["gate"]["refused"] and "extract" not in man["stages"] and man["refused"]
    assert "REFUSED" in capsys.readouterr().out


def test_gate_evidence_not_in_abstract_refuses(gated):
    gated.use("in_scope", evidence="a sparse mixture of experts router")
    assert gated.run() == A.EXIT_REFUSED
    ref = json.loads((gated.run_dir / "refusal.json").read_text())
    assert not ref["evidence_found"] and any("evidence phrase not found" in r for r in ref["reasons"])
    assert not (gated.run_dir / "extraction.json").exists()


def test_gate_malformed_reply_stops_the_chain(gated):
    gated.use(None, raw="I think this paper is in scope.")
    with pytest.raises(SystemExit, match="not JSON"):
        gated.run()
    assert not (gated.run_dir / "extraction.json").exists()
    assert not (gated.run_dir / "relevance.json").exists()


def test_gate_skip_is_explicit_and_recorded(gated):
    client = gated.use("out_of_scope")
    assert gated.run("--skip-relevance-gate", "--stop-after", "extract") == 0
    man = json.loads((gated.run_dir / "manifest.json").read_text())
    assert man["relevance_gate"].startswith("SKIPPED") and man["stages"]["gate"]["skipped"]
    assert client.beta.messages.sent == []
