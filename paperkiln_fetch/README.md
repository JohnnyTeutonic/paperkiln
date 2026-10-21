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

## Why this exists

Most transformer papers are deltas over a known skeleton — dims, depth,
heads, norm flavor, activation, position encoding. That makes extraction
tractable. What nobody else ships is **provenance**:

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

On a 40-paper ground-truth benchmark the scorer runs at grouped AUROC
0.905, abstention-first, with **no wrong assertion in 92 verdicts** (72
correct, the rest abstentions). Every wrong assertion the benchmark has
found is registered with its root cause and fixed. The benchmark,
its confidence intervals, and its growth protocol ship in the
[paperkiln repo](https://github.com/JohnnyTeutonic/paperkiln), where the
extracted architectures also *train* — as real models, on a readable
C++ stack, with the paper's actual hyperparameters.

## The full loop lives upstream

This package is the rule-based extractor, standalone, zero build step and
no API key. The mother repo also has an LLM extractor
(`papers/llm_extract.py`): one Claude call per paper returns the
architecture as an op graph, new operations as formulas the engines run,
and a verbatim quote for every value, each quote checked against the
LaTeX before use. On the same 40-paper benchmark it scores 82 correct,
9 abstentions and 1 wrong, against this package's 72 / 20 / 0. The mother
repo adds: constructor-real training of the extracted architecture
(`--emit-cpp`, mtsweep specs), GGUF export, in-browser chat with the
model you just trained, and the Architecture Atlas — a pre-registered
findings registry for architecture claims.

MIT. Source of truth: `papers/fetch.py` in the paperkiln repo; this
package vendors it verbatim (checked by `tools/sync_fetch_pkg.py`).
