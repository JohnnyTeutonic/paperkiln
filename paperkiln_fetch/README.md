# paperkiln-fetch

**Drop an arXiv id. Get the paper's architecture back — with receipts.**

```bash
pip install paperkiln-fetch

paperfetch 1706.03762
# arXiv:1706.03762  Attention Is All You Need
#   d_model         = 512       [d_model = 512]
#   n_layers        = 6         [N = 6 identical layers]
#   n_heads         = 8         [h = 8 parallel attention layers]
#   ...

paperfetch 2302.13971 --emit-hf llama_config.json
# a config.json that transformers.AutoConfig.from_pretrained opens
# directly — with every extracted value's evidence snippet and every
# declared default riding inside it under the "_paperkiln" key
```

To compare a checkpoint's explicitly supplied local configuration with the paper extraction:

```bash
paperfetch 2302.13971 --tex paper.tex --hf-config checkpoint/config.json --json comparison.json --emit-html comparison.html
```

The comparison reports explicit numeric differences and missing values, with every paper-size
variant shown separately. It never selects a checkpoint/variant or fills or overwrites a quoted
paper field. Conflicting config aliases and invalid numeric values are rejected. Only top-level
config keys are compared; a nested `text_config` is not silently selected. Numeric agreement
alone does not establish architecture equivalence or checkpoint identity.

The rule-based extractor can report explicit MoE adoption and expert/top-k counts, with quotes.
It abstains on conflicting counts, preliminary/retracted choices and ambiguous model variants.
These simple emitters do not implement MoE: requesting C++, mtsweep or HF-config output for
stated or unresolved MoE fails before writing a dense substitute. JSON/HTML extraction reports
remain available. This bounded support does not infer expert-gate normalization.

HF output is a **configuration skeleton**, not a weight conversion or a verified reconstruction
of the paper. The two supported skeletons cannot represent every extracted mechanism: explicit
sliding windows, sinks, highway or absent residuals, incompatible norm/position/activation combinations,
and GPT-2 grouped-query attention refuse rather than silently change the architecture.
Dimensions and head geometry are checked. Unverified or contested findings remain in provenance
but are not applied. Missing dimensions, architecture assumptions and emitted constants are named
in `_paperkiln.defaults_applied`; an unqualified GELU uses the exact form as a declared assumption.
Bare SiLU does not establish a gated FFN, so Llama emission requires explicit SwiGLU or GEGLU.

## Why this exists

Most transformer papers are deltas over a known skeleton — dims, depth,
heads, norm flavor, activation, position encoding. That makes extraction
tractable. This package preserves **provenance**:

- every extracted value carries the **evidence snippet** it was read
  from (the sentence, the table row);
- a paper *mentions* many architectures but *uses* one — the
  contribution-vs-mention scorer separates them, and close calls are
  reported as **contested** instead of auto-applied;
- fields the sweep cannot resolve are **declared unresolved**, never
  silently guessed — and when `--emit-hf` must fill a default, the
  default is named in the output;
- explicit rejections ("we choose not to adopt X") **veto** a candidate
  outright.

The [paperkiln repository](https://github.com/JohnnyTeutonic/paperkiln) contains
its paper corpus, ground truth, scoring code and dated benchmark receipts. Those
records identify the extractor and protocol that produced each result. The full
stack can also train supported extracted architectures on its C++ engine.

## The full loop lives upstream

This package is the rule-based extractor, standalone, zero build step and
no API key. The mother repo also has an LLM extractor
(`papers/llm_extract.py`): one Claude call per paper returns the
architecture as an op graph, new operations as formulas the engines run,
and a verbatim quote for every value, each quote checked against the
LaTeX before use. The mother repo adds: constructor-real training of the extracted architecture
(`--emit-cpp`, mtsweep specs), GGUF export, in-browser chat with the
model you just trained, and the Architecture Atlas — a pre-registered
findings registry for architecture claims.

MIT. Source of truth: `papers/fetch.py` in the paperkiln repo; this
package vendors it verbatim (checked by `tools/sync_fetch_pkg.py`).
