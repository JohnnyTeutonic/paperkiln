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
    with pytest.raises(A.Refused, match="n_kv_heads"):
        A.build_spec(ext_with(norm="rmsnorm", activation="swiglu", positional="rope", n_heads=32,
                              n_kv_heads=8), SIZE)


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
    # A two-input (gated) form: flex's expr activation reads only the up-projection.
    with pytest.raises(A.Refused, match="reads 2 inputs"):
        A.build_spec(llm_ext("relu(a)*x", inputs=("x", "a")), SIZE)
    # A norm expr whose extra inputs are not learned per-channel vectors (here "m").
    with pytest.raises(A.Refused, match="not all learned per-channel vectors"):
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
    # A learned vector inside the row expression is not a trailing affine.
    ext = llm_ext("(x-rowmean(x)*a)*b", slot="norm", inputs=("x", "a", "b"), stated={"activation": "gelu",
                                                                                    "positional": "learned"})
    for i in ext["llm"]["verification"]["new_ops"][0]["inputs"][1:]:
        i["meaning"] = "learned per-channel vector"
    with pytest.raises(A.Refused, match="trailing"):
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
    # So does a stated norm placement other than pre-norm.
    ext = llm_ext("relu(x)^2", stated={"norm": "rmsnorm", "positional": "rope"})
    ext["llm"]["verification"]["flavours"]["norm_placement"] = {"value": "post", "check": "quote_only",
                                                               "status": "stated", "where": "Sec. 3"}
    with pytest.raises(A.Refused, match="norm placement = post"):
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
    assert "norm placement" in reasons  # no norm before the FFN, the FFN output normalised
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
