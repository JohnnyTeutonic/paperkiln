"""Offline tests for papers/fetch.py — no network, fixture LaTeX only.

    python papers/test_fetch.py
"""
import sys

from fetch import Arch, detex, emit_cpp, emit_spec, extract, parse_tables

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
    test_prose()
    test_table()
    test_unresolved_reported()
    test_emit_html()
    test_emit_spec()
    print("\n[PASS] all fetcher tests")
    sys.exit(0)
