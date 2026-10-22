# arXiv paper with a new op -> chat, no human edits (23 Oct 2026)

The chain (LOCAL_LLM_STACK.md section 5): `papers/llm_extract.py` (one Claude call) ->
`tools/arxiv_to_chat.py` (paperkiln 96cc444e) -> mtstudio trains a tiny flex model on the CPU -> GGUF
with the IR embedded -> ember.cpp's CPU server (main 8aec241) answers /chat from the IR. An
engineering demonstration: reply quality is not the point.

## Positive: GoLU, arXiv 2502.03654

Gompertz Linear Units (Das et al.). The extraction (`2502.03654.extraction.json`, $0.26) gives
the new op as an expr, verified against the LaTeX:

    golu = x*exp(-exp(-x))         canonical (x*exp((-exp((-x)))))   replaces the FFN activation

The paper trains babyGPT and GPT2-S from nanoGPT. Norm and positions are not restated; the reply
names GPT-2 as the ancestor (verified quote) and marks the family inherited, so the chain takes
`layernorm` and `learned` from the curated ancestor table (`fetch.ANCESTORS['gpt-2']`) and reports
them as INHERITED. Residual and attention default (none stated). Spec: flex, d 128, 2 layers, 4
heads, T 64, `custom.activation = "expr:(x*exp((-exp((-x)))))"` (`golu_run/spec_report.json`).

- mtstudio's start-up gradient check: autograd vs central finite differences, worst rel 2.6e-5
  (tolerance 2e-3), value and gradient finite on [-20, 20].
- Pretrain 600 steps (batch 4 x 64, TinyChat sample): train loss 9.05 -> 4.22, val 4.33, 708 s.
- SFT 200 steps on 712 TinyChat examples that fit 64 tokens (`sft_fit.py`): val 3.65, 196 s.
- Export: GGUF `general.architecture = paperkiln`, IR block `layernorm, linear, split, attention,
  linear, add, layernorm, linear, expr, linear, add`; the expr node is
  `{"op": "expr", "in": ["u"], "expr": "(x*exp((-exp((-x)))))", "out": "u"}` (`golu_run/expr_node.json`).
  ember has no builder for `paperkiln`, so it runs the embedded IR on its interpreter.
- ember vs mtstudio /chat, greedy, 10 frozen prompts: 10/10 identical replies, token counts, stop
  reasons and prompt token counts. Chat transcript: `golu_run/chat.jsonl`.

Run: `chain.sh` through `scripts/cpu_budget.py` (2 slots), CPU only, about 18 minutes in all.

## Negatives (refused, nothing trained)

- Primer, 2109.08668 (`2109.08668.refusal.json`): the depthwise convolution over time is
  `needs_new_primitive`; its modified layer norm is an expr in the norm slot, which flex does not
  host; positions are T5's relative scheme. Its squared ReLU alone would have been hosted.
- ReLU^2 Wins, 2402.03804 (`2402.03804.refusal.json`): RoPE with an expr activation (the llama
  family is swiglu only; flex has no rope), and residual and attention inherited from LLaMA without
  a curated value.

API: $0.543 of the $5 budget (`api_cost.json`; Primer was a cache hit at $0).
