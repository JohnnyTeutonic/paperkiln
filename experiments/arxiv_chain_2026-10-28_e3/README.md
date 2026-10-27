# The chain's remaining refusals and the bench's verifier gaps (28 Oct 2026, LOCAL_LLM_WORK E3)

Three live extractions under the current prompt (papers/llm_extract.py; ledger tag `e3_2026-10-28`,
$1.41 in all: the Transformer 1706.03762 $0.33, OLMo 2 2501.00656 $0.63, Primer 2109.08668 $0.45) and the
chain's spec stage on E3's papers, each extraction re-verified by today's verifier against its cached LaTeX
(`spec_results.py` -> `spec_results.json`; offline). The 40-paper bench was not rerun: its scores stay those
of the 26 Oct prompt, rescored offline by today's verifier and scorer (`papers/llm_extract_receipts/
2026-10-28_bench_scores.json`).

## Each refusal's outcome (LOCAL_LLM_STACK.md 5.4's triage)

| refusal | outcome | how |
|---|---|---|
| Primer's positions | permanent refusal by rule | The paper names relative attention only for the codebases it runs in ("T2T uses positional embeddings ..., while T5 and Lingvo use relative attention~\cite{Shaw2018...}") and states no clipping distance and no head sharing. Under today's prompt the reply gives `shaw_relative` with `rel_max_distance` and `rel_per_head` not stated, and the chain refuses both by name. An unstated value is never supplied by the chain; `--assert` is the operator's way in. |
| q/k norm on flex, or over the full width | built: wiring (a) | flex `qk_norm` "per_head" or "full": an RMSNorm on q and k after the projection (and MDHA's convolution), before RoPE (parity::FlexConfig). Exported as the thirdparty profile's `rmsnorm` after the split, `group` head_dim per head, no group over the row (IR_SPEC 5.6): no keramos change. A per-head q/k norm stays on llama when the block is llama's. A LayerNorm on q and k in the block graph refuses (the IR's layernorm has no group). A stated RoPE base also builds on flex (OLMo 2's 500,000). |
| Mamba-1's optional inner LayerNorm | permanent refusal by rule | "we additionally use an optional normalization layer (we choose LayerNorm), motivated by RetNet's usage of a normalization layer in a similar location": no place in the block and no statement of which models use it. `--assert ssm_inner_norm=none` builds the block without it, on the operator's word (with `--assert d_conv=4`, which the paper also leaves out). |
| MoE headline models | built: wiring (a) | The llama family's `arch.moe` (E SwiGLU experts, top k, gates renormalised over the k), from stated `n_experts`, `top_k` and `moe_gate_norm`. An unstated gate normalisation refuses until asserted; `raw` refuses (paperkiln trains it, but llama.cpp's MoE renormalises, so mtstudio exports no GGUF and nothing serves it). Shared experts and other MoE structure stay refusals by name. `--waive TEXT` lets the operator report a blocking mechanism that is not part of the network (Qwen3's load-balancing loss, its inference-time YaRN and DCA, a stated absence) instead of refusing it. |
| an affine scale given as a formula | built: wiring (a) and (b) | The verifier accepts a scale written as a formula in d_model (the expr grammar over that one name, sqrt and arithmetic: every number in the quote, a root named, a width named) and the chain evaluates it at the run's width. A constant affine on the embedding becomes flex's embed expr, one on the logits flex's new logits expr (`custom.logits`, which also hosts Gemma 2's final soft cap). An affine before an additive position table (the Transformer's sqrt(d_model)) refuses by name: flex's embed expr acts after the table. Live: today's prompt returns the Transformer's scale as `sqrt(d_model)`, verified (the 26 Oct reply gave 22.627417, which its quote does not contain). |

## The chain on E3's papers (`spec_results.json`, size d 64, 2 layers, 4 heads)

- Mamba: refuses (inner norm by rule, d_conv unstated); with `--assert d_conv=4 ssm_inner_norm=none` builds
  the mamba preset (d_state 16, expand 2 verified).
- Qwen3 (headline MoE 235B-A22B): refuses (gate normalisation, q/k-norm width, four mechanism entries);
  with `--assert qk_norm=per_head moe_gate_norm=renorm` and four `--waive` texts, at 16 heads for the paper's
  16:1 query to key/value ratio, builds llama + MoE (128 experts, top 8) + per-head q/k norm + RoPE base 1e6.
- Mixtral: refuses (norm and positions inherited from Mistral 7B, which the curated table lacks); with
  `--assert norm=rmsnorm positional=rope` builds llama + MoE (8 experts, top 2, renormalised).
- OLMo 2 (today's extraction): its q/k norm comes back `unspecified`; with `--assert qk_norm=full` builds flex
  with the full-width q/k norm, branch_post placement, SwiGLU, RoPE base 500,000.
- PaLM: the logit scale 1/sqrt(d_model) is hosted; still refuses for its parallel block and its bias-free
  LayerNorm.
- Gemma 2: the final soft cap is hosted on the logits; still refuses for its attention soft cap (a new
  primitive) and GeGLU.
- The Transformer (today's extraction, `--assert family=decoder_only`): refuses for the embedding scale
  before the sinusoidal table (above) and the reply's sinusoidal table as a needs_new_primitive.

## OLMo 2 through the chain, end to end (`olmo2_run/`)

`tools/arxiv_to_chat.py 2501.00656 --extraction 2501.00656.extraction.json --reverify --assert qk_norm=full
--d 64 --layers 2 --heads 4 --T 64 --pretrain-steps 400 --sft-steps 100 --engine numpy` (a clean ext4 clone,
mtstudio built there, 2 slots through scripts/cpu_budget.py): relevance gate in_scope ($0.0070) -> spec flex
(full-width q/k norm, branch_post on both sublayers, SwiGLU, RoPE base 500,000) -> pretraining 400 steps
(val 4.63, 1,003 s) -> SFT 100 steps (283 s) -> GGUF (sha256 6ee21d54...) that keramos's `ir_check` accepts ->
keramos's numpy interpreter against `mtstudio chat`, greedy 11/11 identical replies. The IR is
`olmo2_run/arxiv_2501_00656.ir.json`. The API spend of the folder is $1.415 (three extractions and this gate).

## Engine checks (paperkiln, a clean clone at HEAD with these changes)

- tests/test_flex.cpp 15/15: 14, the q/k norm: per head invariant to a head's query scale (2.9e-7), full
  width invariant to the whole row's (4.0e-7) and not to one head's (4.0e-2); gradient checks 2.2e-4 and
  2.4e-4; exported as rmsnorm after the split. 15, the logits expr: x*0.125 equals the logits times 0.125 bit
  for bit; 30*tanh(x/30) gradient check 3.2e-4; exported as an expr node after the head.
- tests/test_flex_ir.py: numpy IR against `mtstudio score` on 40 prompts, greedy 40/40 each: per-head q/k norm
  3.0e-7, full width with RoPE base 500,000 and branch_post 2.5e-7, the logit scale 7.2e-8, Gemma 2's soft cap
  3.5e-7; with test_thirdparty_ir.py 36 passed, 1 skipped.
- tools/test_arxiv_to_chat.py, papers/test_llm_extract.py, papers/test_bench_score.py: 141 passed, 2 skipped.
