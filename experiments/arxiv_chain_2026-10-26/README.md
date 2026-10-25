# The extractor's gaps (26 Oct 2026, LOCAL_LLM_STACK.md section 11 item 4.2)

Five live extractions under the prompt of 26 Oct 2026 (papers/llm_extract.py; prompt sha256 starts
e6c6c505), the chain's spec stage on each, and one paper trained and exported end to end. API: $2.00
in all (`api_cost.json`): five extractions ($1.99) and two relevance-gate calls ($0.015). The other
three gate verdicts and all 69 gate-bench verdicts were cache hits.

## The relevance gate benchmark, rerun

`gate_bench_v1_result_2026-10-26.json`: every verdict was cached (the gate's prompt did not change),
so it cost $0 and gives the same result as `../arxiv_chain_2026-10-23/gate_bench_v1_result_prompt2.json`:
0 of 48 in-scope papers falsely refused, 0 of 14 out-of-scope or spurious ones falsely passed.

## Tonight's papers (`<id>.extraction.json`; the gate verdicts in `relevance_gate.json`)

| arXiv | paper | the chain's verdict |
|---|---|---|
| 2503.10622 | DyT | builds on flex once `--assert activation=swiglu` is given: the paper says it leaves the base model's activation untouched but never names SwiGLU. Built from the reply: the norm tanh(alpha x) * g + b, alpha 0.8 at the attention sublayer's norm and 0.2 at the FFN's and the head's (two new ops, one expression, `NORM-VARIANT`); the learned scalar after the embedding, initialised to sqrt(d); RoPE from LLaMA. |
| 2109.08668 | Primer | refuses: the depthwise convolutions after q, k and v (conv1d_causal block ops, which no family wires there; before the guard added tonight these passed silently), and positional = relative (Shaw et al.'s). Its placement (pre attention, branch_post FFN: the new per-sublayer fields) and its modified layer norm now build. |
| 2312.00752 | Mamba | refuses: d_conv is not stated, and its block has an optional inner LayerNorm ("we choose LayerNorm"), which the Mamba-1 block here lacks. It is recognised as Mamba-1 with N = 16 and E = 2, both verified. |
| 2505.09388 | Qwen3 | refuses: the headline model is the MoE (235B-A22B), and its QK-norm is `unspecified` (the paper does not say per head). |
| 1710.05941 | Swish | refuses: its hosts are convolutional networks with BatchNorm. Its expr, x * sigmoid(a x) with a learned scalar, is extracted with its roles. |

## DyT trained and exported (`dyt_run/`)

`tools/arxiv_to_chat.py 2503.10622 --extraction 2503.10622.extraction.json --assert activation=swiglu
--d 64 --layers 2 --heads 4 --T 64 --pretrain-steps 60 --sft-steps 20 --stop-after export` (paperkiln
~/ex_build, CPU, 2 slots, through scripts/cpu_budget.py). The spec is in `pretrain_spec.json`:
norm `expr:tanh((a*x))`, norm_inputs `["row", {"kind": "scalar", "init": 0.8, "init_ffn": 0.2,
"init_final": 0.2}]`, embed `expr:(x*a)` with init 8 (sqrt 64), swiglu, rope. Validation loss was 6.31
after pretraining (93 s) and 5.96 after SFT (30 s). The GGUF (sha256 eee098d2...,
not committed) passes `spec/tools/ir_check.py` (`ir_check.txt`). Its IR (`arxiv_2503_10622.ir.json`):
embed `embedding, expr, scale, expr`; block `expr, scale, expr, scale, linear, split, rope, attention,
linear, add, expr, scale, expr, scale, linear, silu, linear, mul, linear, add`; head `expr, scale, expr,
scale, linear`.

## Engine checks (paperkiln, `test_flex.log`, `test_flex_ir.log`)

- test_flex 11/11 (9: expr inputs; 10: placements; 11: T5's bias).
- tests/test_flex_ir.py 17/17. Ten models have their numpy IR within 0.9e-7 to 3.2e-7 of `mtstudio
  score`, greedy 40/40: a gated unit, Swish-beta, DyT (scalar, per channel, the LLaMA recipe), post,
  sandwich, Primer's placement and norm, none + highway, and T5's bias under the proposed IR (2.62e-7).
- The existing IR export and thirdparty tests pass 36/36.
