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

## Positive: ReLU^2 Wins, arXiv 2402.03804 (flex with RoPE)

(This run, the expr-norm gate and the Primer rerun: paperkiln 80ffca45.)

The first run refused this paper: RoPE with an expr activation had no family (llama is SwiGLU only,
flex had no RoPE). flex now takes `position: "rope"` (paperkiln, tools/parity_model.hpp: every head,
ops::apply_rope between the fused qkv projection and attention, exported as the llama block's rope
node). Same cached extraction (`2402.03804.extraction.json`, no new API call), same scale as GoLU:

- Spec (`relu2wins/spec_report.json`): flex, d 128, 2 layers, 4 heads, T 64,
  `norm rmsnorm, activation expr:(relu(x)^2), position rope, residual residual, attention exact`.
  rmsnorm, rope and the expr are the paper's verified values; residual and attention are inherited
  from LLaMA-2 without a curated entry to check them, so the chain's defaults for an unstated
  residual stream and attention apply (DEFAULT, with the reason); `qkv_bias = false` (inherited,
  unverified) is NOTED: flex's projections carry biases.
- Gradient check at start-up: `relu(x)^2` 1.6e-5 vs finite differences.
- Pretrain 600 steps: train loss 9.05 -> 4.00, val 4.15, 558 s. SFT 200 steps: val 3.70, 267 s.
  The SFT set is TinyChat unfiltered (`sft_tinychat.jsonl`); arxiv_to_chat itself kept the first
  2,000 examples that fit T and dropped 5,697 of the 7,697 read as longer (`relu2wins/sft_fit.json`;
  no `sft_fit.py` this time).
- Export: GGUF (sha256 68e399e0...) with IR block `rmsnorm, linear, split, rope, attention, linear,
  add, rmsnorm, linear, expr, linear, add` (`relu2wins/ir_nodes.json`), no position table.
- ember main 8aec241 (binary sha256 fea592b0...) vs mtstudio /chat, greedy, 40 frozen prompts: 40/40
  identical replies, token counts, stop reasons and prompt token counts (`relu2wins/parity.jsonl`).
- Chat (`relu2wins/chat.jsonl`, greedy):
  - user: Hello! What is your name? / assistant: I understand, I am feeling quite frustrating when
    they are not not not not not not not go.
  - user: Can you tell me a short story about a cat? / assistant: I understand, I am surprised to
    see you too, it is a bit sad.

Run: `chain_relu2.sh` through `scripts/cpu_budget.py` (2 slots), about 15 minutes.

## ember runs flex's expr norm (gate)

flex now takes `norm: "expr:<e>"`: y = e(x) * g + b, e over the row (row reductions allowed), g and
b learned per feature, exported as an expr node and a linear with weight diag(g) and bias b
(ir.hpp, Families: flex). `expr_norm_gate.py` trains a tiny flex model with Primer's modified layer
norm `(x - rowmean(x)) / sqrt(abs(rowmean(x * (x - rowmean(x)))))`, RoPE and squared ReLU (d 128,
L 2, T 64; 400 + 100 steps, val 4.37 / 3.92), start-up gradient checks norm 3.8e-5, activation
1.6e-5. IR block: `expr, linear, linear, split, rope, attention, linear, add, expr, linear, linear,
expr, linear, add`; head `expr, linear, linear`. ember (same binary) vs mtstudio /chat greedy on the
40 frozen prompts: 40/40 (`expr_norm_gate/gate.json`, `parity.jsonl`). The engine-level checks are
in paperkiln's tests: test_flex 7-8 (layernorm written as an expr equals layernorm within 8.9e-8 on
logits of range 0.35; whole-model gradient checks: rope 185 entries worst 3.7e-4, the layernorm expr
447 entries 3.3e-4, Primer's norm with rope 441 entries 7.2e-4) and test_ir_export's flex cases.

## Negatives (refused, nothing trained)

- Primer, 2109.08668 (`2109.08668.refusal.json`, rerun on the cached extraction): its squared ReLU
  and its modified layer norm are both hosted now (the norm's learned per-channel shift a and gain b,
  (core + a) * b, map onto flex's gain and bias: the same functions, another parametrisation). It
  still refuses, for three reasons the engines cannot honour: the depthwise convolution over time
  (`needs_new_primitive`), a norm placement that is not pre-norm (no norm before the FFN, the FFN
  output normalised; every family here is pre-norm), and T5's relative attention positions.

API: $0.543 of the $5 budget (`api_cost.json`; Primer was a cache hit at $0). The reruns made no
API call.
