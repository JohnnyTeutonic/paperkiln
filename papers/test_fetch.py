"""Offline tests for papers/fetch.py — no network, fixture LaTeX only.

    python papers/test_fetch.py
"""
import sys

from fetch import (Arch, detex, emit_cpp, emit_spec, extract, parse_tables,
                   score_flavors, split_sections)

FIXTURE_PROSE = r"""
\title{A Tiny Transformer}
We stack $N=4$ identical layers with $d_{\text{model}}=256$ and employ
$h=8$ parallel attention layers, or heads. The inner-layer has
dimensionality $d_{ff}=1024$. We use a vocabulary of 8K tokens and a
context length of 512 tokens. Sub-layers are followed by layer
normalization with a ReLU activation. % comment should vanish
"""

FIXTURE_TABLE = r"""
We use the RMSNorm normalizing function with SwiGLU and rotary positional
embeddings (RoPE).
\begin{tabular}{cccc}
params & dimension & n heads & n layers \\
1.3B & 2048 & 16 & 24 \\
7B & 4096 & 32 & 32 \\
\end{tabular}
"""


def test_prose() -> None:
    arch = extract("0000.00000", FIXTURE_PROSE)
    got = {k: f.value for k, f in arch.fields.items()}
    assert got["d_model"] == 256, got
    assert got["n_layers"] == 4, got
    assert got["n_heads"] == 8, got
    assert got["d_ff"] == 1024, got
    assert got["vocab_size"] == 8000, got
    assert got["context_length"] == 512, got
    assert got["norm"] == "layernorm", got
    assert got["activation"] == "relu", got
    assert arch.title == "A Tiny Transformer", arch.title
    assert "% comment" not in detex(FIXTURE_PROSE)
    cpp = emit_cpp(arch)
    assert "cfg.d        = 256" in cpp and "GPT2" in cpp
    print("prose extraction ok:", got)


def test_table() -> None:
    arch = extract("0000.00001", FIXTURE_TABLE)
    got = {k: f.value for k, f in arch.fields.items()}
    assert got["norm"] == "rmsnorm", got
    assert got["activation"] == "swiglu", got
    assert got["positional"] == "rope", got
    # Row 1 (smallest model) is the chosen config; both rows are variants.
    assert got["d_model"] == 2048, got
    assert got["n_heads"] == 16, got
    assert got["n_layers"] == 24, got
    assert len(arch.variants) == 2, arch.variants
    assert arch.variants[1]["d_model"] == 4096
    cpp = emit_cpp(arch)
    assert "LlamaExportConfig" in cpp and "cfg.block_count        = 24" in cpp
    print("table extraction ok:", got, "variants:", len(arch.variants))


FIXTURE_COMPACT = r"""
The model has 16 transformer layers of dimension 1024. Each decoder is
110M parameters ($d_{model}=768$, $d_{ff}=3072$, L=12).
"""


def test_compact_notation() -> None:
    # Real phrasings that used to slip through: ALiBi's "layers of
    # dimension 1024" and Primer's "(d_model=768, d_ff=3072, L=12)".
    arch = extract("0000.00003", FIXTURE_COMPACT)
    got = {k: f.value for k, f in arch.fields.items()}
    assert got["n_layers"] == 16, got   # first match wins (prose order)
    assert got["d_ff"] == 3072, got
    assert got["d_model"] in (768, 1024), got
    # The L=12 pattern alone, without the ALiBi sentence:
    arch2 = extract("0000.00004", r"Each decoder is ($d_{model}=768$, L=12).")
    got2 = {k: f.value for k, f in arch2.fields.items()}
    assert got2["n_layers"] == 12, got2
    print("compact-notation extraction ok:", got, got2)


FIXTURE_MENTION_ONLY = r"""
Prior work has explored alternatives such as SwiGLU \cite{a} \cite{b},
compared against strong baselines. Our search discovers squaring ReLU
activations, a novel modification.
"""

FIXTURE_REPLACE = r"""
We replace LayerNorm with RMSNorm in every block of our model.
"""

FIXTURE_CONTESTED = r"""
We use LayerNorm for the encoder blocks. We use RMSNorm for the decoder
blocks.
"""


def test_contribution_vs_mention() -> None:
    # Mention-only: nothing asserted, mentions reported.
    a1 = extract("0000.00005", FIXTURE_MENTION_ONLY)
    assert "activation" in a1.unresolved, a1.fields
    assert any(c["value"] == "swiglu" for c in a1.mentions.get("activation", []))
    # Replacement: the target is used, the source is not.
    a2 = extract("0000.00006", FIXTURE_REPLACE)
    f = a2.fields["norm"]
    assert f.value == "rmsnorm" and f.verdict == "used", (f.value, f.verdict)
    # Symmetric usage: contested, runner-up carried, nothing asserted
    # silently.
    a3 = extract("0000.00007", FIXTURE_CONTESTED)
    f3 = a3.fields.get("norm")
    assert f3 is not None and f3.verdict == "contested", f3
    assert f3.runner_up is not None
    # Explicit rejection is a VETO: "we choose not to adopt X" anywhere
    # makes X ineligible for "used", whatever its best sentence scored
    # (the Falcon failure, 2026-08-01).
    a4 = extract("0000.00008", r"""
We evaluate gated units extensively and we use SwiGLU in early runs.
After ablations, we choose not to adopt SwiGLU. Our final model uses
the GELU activation throughout.""")
    f4 = a4.fields.get("activation")
    assert f4 is None or f4.value != "swiglu", f4
    print("contribution-vs-mention ok: mention-only abstains, "
          f"replace->{a2.fields['norm'].value}, symmetric->contested, "
          "explicit rejection vetoes")


FIXTURE_INHERIT = r"""
\section{Model}
We use the same model and architecture as GPT-2, with one change: we use
RMSNorm for all normalization layers.
"""

FIXTURE_INHERIT_GENERIC = r"""
\section{Model}
Our network is based on the transformer architecture \cite{vaswani}.
"""

FIXTURE_INHERIT_NOTOURS = r"""
\section{Data}
Training dedicated classifiers based on BERT models often resulted in
over-fitting, so we filtered with heuristics instead.
"""


def test_inheritance() -> None:
    # Strong claim on a SPECIFIC ancestor: unstated fields inherit, and
    # the paper's own delta overrides the base.
    a = extract("0000.00009", FIXTURE_INHERIT)
    assert a.inherits and a.inherits["ancestor"] == "gpt-2", a.inherits
    assert a.fields["activation"].value == "gelu", a.fields["activation"]
    assert a.fields["activation"].verdict == "inherited"
    assert a.fields["positional"].value == "learned"
    # delta wins over base
    assert a.fields["norm"].value == "rmsnorm", a.fields["norm"]
    assert a.fields["norm"].verdict != "inherited"
    # Generic "based on the Transformer" is recorded but NEVER fills
    # fields (BERT is the counterexample: it silently changes two).
    b = extract("0000.00010", FIXTURE_INHERIT_GENERIC)
    assert b.inherits and b.inherits["ancestor"] == "transformer"
    assert not any(f.verdict == "inherited" for f in b.fields.values()), b.fields
    # A data-pipeline aside is not an architecture claim (self-reference
    # must PRECEDE the claim; "so we" trails it).
    c = extract("0000.00011", FIXTURE_INHERIT_NOTOURS)
    assert c.inherits is None, c.inherits
    print("inheritance ok: specific ancestor fills + delta overrides, "
          "generic transformer refuses, data aside rejected")


def test_glu_family() -> None:
    tex = r"""
\section{Model}
We use SwiGLU in the feed-forward layers. Our GeGLU variant uses the same
gating. We use SwiGLU and we use GeGLU throughout our model.
"""
    a = extract("0000.00012", tex)
    f = a.fields.get("activation")
    assert f is not None and f.value == "gated-glu", f
    assert f.verdict == "family" and f.runner_up is not None
    print(f"GLU family ok: contested swiglu/geglu -> {f.value}")


def test_inheritance_outranks_third_party_attribution() -> None:
    # Megatron-LM shape (KNOWN_WRONG fixed 2026-10-11): the paper declares
    # a specific ancestor, never states its own activation in the first
    # person, and the contrasted alternative sits in a bare declarative
    # clause about ANOTHER model. The inheritance must win.
    tex = r"""
\section{Model}
We train a transformer language model similar to GPT-2 at scale. It is
worthwhile to mention that both GPT-2 and BERT apply layer normalization
to the input of each layer, whereas the original transformer
\citep{vaswani} uses ReLU nonlinearities and normalizes outputs.
"""
    a = extract("0000.00020", tex)
    assert a.inherits and a.inherits["ancestor"] == "gpt-2", a.inherits
    f = a.fields["activation"]
    assert (f.value, f.verdict) == ("gelu", "inherited"), (f.value, f.verdict)
    assert any(c["value"] == "relu" for c in a.mentions["activation"])
    # Control: the same ancestor, but a FIRST-PERSON delta. The paper's
    # own statement still overrides the base.
    ctrl = extract("0000.00021", r"""
\section{Model}
We train a transformer language model similar to GPT-2 at scale. Unlike
GPT-2, we use ReLU nonlinearities throughout.
""")
    f = ctrl.fields["activation"]
    assert (f.value, f.verdict) == ("relu", "used"), (f.value, f.verdict)
    # Control: Llama 2 shape. "We" opens the sentence far outside the
    # cue window and a citation precedes the verb; still first person,
    # so no third-party mark.
    llama2 = score_flavors(split_sections(r"""
\section{Model}
We use the standard transformer architecture \citep{vaswani}, apply
pre-normalization using RMSNorm \citep{zhang}, use the SwiGLU activation
function \citep{shazeer}, and rotary positional embeddings.
"""))
    sw = [c for c in llama2["activation"] if c["value"] == "swiglu"][0]
    assert "third-party-attribution" not in sw["cues"], sw
    print("precedence ok: inheritance beats third-party attribution; "
          "first-person delta and Llama-2 shape unchanged")


def test_future_work_vetoes_and_like_inheritance() -> None:
    # Cerebras-GPT shape (KNOWN_WRONG fixed 2026-10-11): the ancestor is
    # named as "X-like architecture" with the paper's own model name as
    # subject, and the wrong flavors appear only as future work.
    tex = r"""
\title{Foo-LM: Open Compute-Optimal Models}
\section{Model Architecture}
Foo-LM models have a GPT-3-like architecture, an autoregressive
transformer decoder model. The main difference is that we use dense
attention in all decoder blocks.
\section{Limitations}
Model features worth exploring in future work include position
embeddings, such as RoPE and ALiBi, and activation functions, like
SwiGLU.
\section{Appendix}
GPT-J and Pythia models use rotary positional embeddings, which show
modest improvements.
"""
    a = extract("0000.00022", tex)
    assert a.inherits and a.inherits["ancestor"] == "gpt-3", a.inherits
    for fieldname, want in (("positional", "learned"), ("activation", "gelu")):
        f = a.fields[fieldname]
        assert (f.value, f.verdict) == (want, "inherited"), (fieldname, f)
    # Control: a deferral VETOES only the deferred flavor; the adopted
    # one in the same sentence still reads as used.
    ctrl = extract("0000.00023", r"""
\section{Model}
We use RoPE in every layer. We leave ALiBi for future work.
""")
    f = ctrl.fields.get("positional")
    assert f is not None and (f.value, f.verdict) == ("rope", "used"), f
    # Same sentence: the veto still lands on the deferred flavor only.
    cands = score_flavors(split_sections(r"""
\section{Model}
We use RoPE in every layer and leave ALiBi for future work.
"""))["positional"]
    cues = {c["value"]: c["cues"] for c in cands}
    assert "rejection-elsewhere" in cues["alibi"], cues
    assert "rejection-elsewhere" not in cues["rope"], cues
    # Control: a "-like" sentence with no self-reference is not this
    # paper's inheritance.
    other = extract("0000.00024", r"""
\section{Model}
Several open models have a GPT-3-like architecture.
""")
    assert other.inherits is None, other.inherits
    print("future-work veto + X-like inheritance ok; adopted flavor in a "
          "deferral sentence still used")


def test_gated_compound_names() -> None:
    # LaMDA shape (KNOWN_WRONG fixed 2026-10-11): "gated-GELU" IS GeGLU,
    # and the bare GELU inside it must not match on its own.
    tex = r"""
\section{Model}
The Transformer has 64 layers, relative attention as described in T5,
and gated-GELU activation as described in Raffel et al.
"""
    cands = score_flavors(split_sections(tex))["activation"]
    assert [c["value"] for c in cands] == ["geglu"], cands
    a = extract("0000.00025", tex)
    assert a.fields["activation"].value == "geglu", a.fields["activation"]
    # Spelling variants normalise the same way; gated-ReLU (ReGLU, out of
    # the lattice) blocks its inner ReLU and yields NO candidate.
    for spelling, want in (("Gated GELU", ["geglu"]), ("gated-SiLU", ["swiglu"]),
                           ("gated-ReLU", [])):
        c = score_flavors(split_sections(
            "\\section{Model}\nWe use the " + spelling + " activation."))
        got = [x["value"] for x in c.get("activation", [])]
        assert got == want, (spelling, got)
    # Control: a bare GELU / ReLU / GeGLU is untouched.
    for spelling, want in (("GELU", "gelu"), ("ReLU", "relu"), ("GeGLU", "geglu")):
        a = extract("0000.00026", "\\section{Model}\nWe use the " + spelling
                    + " activation.")
        assert a.fields["activation"].value == want, (spelling, a.fields)
    print("compound names ok: gated-X -> XGLU, longest match wins")


def test_unresolved_reported() -> None:
    arch = extract("0000.00002", r"A paper with no architecture at all.")
    assert "d_model" in arch.unresolved and "norm" in arch.unresolved
    assert not arch.fields
    print("unresolved reporting ok")


def test_emit_html() -> None:
    from fetch import emit_html
    arch = extract("0000.00000", FIXTURE_PROSE)
    html = emit_html(arch)
    # Every extracted field appears with its evidence; unresolved fields
    # are labelled as such; the page is self-contained (no external refs).
    for k, f in arch.fields.items():
        assert k in html, f"field {k} missing from html"
        assert str(f.value) in html
    for k in arch.unresolved:
        assert k in html
    assert "unresolved &mdash; reported, not guessed" in html or not arch.unresolved
    assert "http" not in html.split("</style>")[1], "external reference leaked"
    assert "diff-to-paper" in html
    print("emit_html ok "
          f"({len(arch.fields)} fields, {len(arch.unresolved)} unresolved)")


FIXTURE_SWA = r"""
\section{Model}
We use sliding window attention with a window size of 4096 tokens to
bound the per-token cost. We keep 4 attention sink tokens at the start
of the sequence.
"""

FIXTURE_HIGHWAY = r"""
\section{Model}
We replace residual connections with highway layers, using transform
gates to modulate information flow across depth.
"""

FIXTURE_MISTRAL = r"""
\section{Model}
We use the same architecture as Mistral-7B and train on our corpus.
"""


def test_swa_highway() -> None:
    # SWA prose: mechanism + its numeric companions extract together.
    a = extract("0000.00013", FIXTURE_SWA)
    f = a.fields.get("attention")
    assert f is not None and f.value == "swa" and f.verdict == "used", f
    assert a.fields["window_size"].value == 4096, a.fields.get("window_size")
    assert a.fields["n_sinks"].value == 4, a.fields.get("n_sinks")
    spec = emit_spec(a)
    c = spec["base"]["arch"]["custom"]
    assert c.get("attention") == "swa" and c.get("sinks") == 4, c
    # window >= T is degenerate (swa == exact): scaled, loudly.
    assert 0 < c["window"] < spec["base"]["data"]["T"], c
    assert "4096" in spec["_comment"], spec["_comment"]

    # Highway replacement: target used, mapped to the residual knob.
    b = extract("0000.00014", FIXTURE_HIGHWAY)
    fb = b.fields.get("residual")
    assert fb is not None and fb.value == "highway" and fb.verdict == "used", fb
    cb = emit_spec(b)["base"]["arch"]["custom"]
    assert cb.get("residual") == "highway", cb

    # Named-swa ancestor: Mistral inheritance carries the mechanism AND
    # its window; emit still applies + scales it.
    m = extract("0000.00015", FIXTURE_MISTRAL)
    assert m.inherits and m.inherits["ancestor"] == "mistral", m.inherits
    fm = m.fields.get("attention")
    assert fm is not None and fm.value == "swa" and fm.verdict == "inherited", fm
    cm = emit_spec(m)["base"]["arch"]["custom"]
    assert cm.get("attention") == "swa" and cm.get("window", 0) > 0, cm

    # swa WITHOUT a window is refused, not guessed.
    n = extract("0000.00016",
                r"\section{Model} We use sliding window attention.")
    sn = emit_spec(n)
    assert "attention" not in sn["base"]["arch"]["custom"], sn
    assert "window unresolved" in sn["_comment"], sn["_comment"]
    print("swa/highway extraction ok: swa+window+sinks, highway replace, "
          "mistral inheritance, windowless swa refused")


def test_emit_spec() -> None:
    arch = extract("0000.00000", FIXTURE_PROSE)
    # Paper-faithful dims: extracted numbers land in arch.custom.
    spec = emit_spec(arch, corpus="c.txt", vocab="v.gguf", steps=40)
    custom = spec["base"]["arch"]["custom"]
    assert custom["d"] == 256 and custom["layers"] == 4 and custom["heads"] == 8
    assert custom["d_ff"] == 1024
    assert spec["base"]["train"]["steps"] == 40
    assert spec["base"]["data"]["corpus"] == "c.txt"
    # Flavors only when verdict says used/inherited; whatever is skipped
    # must be named in the comment, never silently dropped.
    for src, f in arch.fields.items():
        if src in ("norm", "activation", "positional"):
            dst = {"norm": "norm", "activation": "activation",
                   "positional": "position"}[src]
            if f.verdict in ("used", "inherited"):
                assert custom.get(dst) == str(f.value)
            else:
                assert str(f.value) in spec["_comment"]
    # House dims keep the mechanism, swap the scale.
    house = emit_spec(arch, house_dims=True)["base"]["arch"]["custom"]
    assert house["d"] == 128 and house["layers"] == 2
    assert "d_ff" not in house  # ratio falls to the engine default
    # Unresolved fields surface in the comment.
    assert (not arch.unresolved or
            all(u in spec["_comment"] for u in arch.unresolved))
    print("emit_spec ok (paper-faithful + house dims, contract preserved)")


if __name__ == "__main__":
    test_compact_notation()
    test_contribution_vs_mention()
    test_inheritance()
    test_glu_family()
    test_inheritance_outranks_third_party_attribution()
    test_future_work_vetoes_and_like_inheritance()
    test_gated_compound_names()
    test_prose()
    test_table()
    test_unresolved_reported()
    test_emit_html()
    test_emit_spec()
    test_swa_highway()
    print("\n[PASS] all fetcher tests")
    sys.exit(0)
