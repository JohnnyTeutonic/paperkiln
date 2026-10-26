# Changelog

Notable changes to microtorch. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versions are tagged `v*`.

## [Unreleased]

### September–October 2026

- **The fused LM head + cross-entropy, `train.fused_head`** (26 Oct; LOCAL_LLM_STACK.md section 11 item 5.1;
  off by default, and off is the step as before).
  - `ops::linear_cross_entropy(h, W, layout, bias, targets, mask, upstream, chunk_rows)`: the mean NLL of
    softmax(h W + b) over the kept rows, never holding the [tokens, vocab] logits. The design is Liger-Kernel's
    FusedLinearCrossEntropy (arXiv 2410.10989): chunks of tokens, gradients computed in the forward, three GEMMs
    per chunk. As in Cut Cross-Entropy (arXiv 2411.09009) no logits are stored, and no gradient entry is
    filtered. It takes an untied head (`nn::Linear` W [d, vocab]) or a tied embedding ([vocab, d]), a bias, and
    the 0/1 row mask of `cross_entropy_masked` (SFT under loss policy all, final or each; state tokens; padding).
    The default chunk is Liger's rule (logits chunk about the size of the hidden states, at least 32 rows).
    `nn::Llama::forward_hidden` is `forward` up to its head, and `forward` is it plus the head, the same ops.
  - Host numerics against the unfused path (matmul -> cross_entropy[_masked] -> backward): the loss, dh and db
    are bit-identical wherever `ops::host_gemm_rows_free` holds (the vendored blocked GEMM's row split fixed by
    vocab and width alone). The head weight's gradient is summed in ascending row order, within the
    sequential-sum bound (n + 2) u sum |h dl| (worst 0.13 of it; the unfused 0.06) and 2.6e-7 to 4.3e-7 of the
    unfused one normwise. Every chunk size (1, 7, 32, default, all rows) and every OpenMP thread count from 1 to
    8 gives the same bits.
  - `tests/test_fused_head.cpp`: six cases (tied and untied, a bias, no mask, SFT, all-ones, scattered ignored
    rows, one kept row, upstream 1 and 1/2); a shape where the GEMM's rows are not free (dh 3.1e-7, dW 7.0e-7);
    finite differences (worst 2.9e-5); NoGrad; an unstated upstream (rescaled, 4e-7). On `nn::Llama` (tied and
    untied, with and without an SFT mask) the loss is bitwise, and so is every parameter's gradient but the
    head weight's (1.7e-7 to 2.3e-7).
  - mtstudio `train.fused_head` (llama family, lm and sft objectives; `train.fused_head_chunk` sets the chunk)
    is recorded in the model event and result.json when on. `tests/test_mtstudio_fused_head.py` (6/6): with
    the option off, an lm run (accum 2) and an SFT run train byte-identically to mtstudio at c6365b88 (every
    step's loss and grad norm, every eval, the safetensors and GGUF). With it on, the first step's loss is
    bitwise and the run tracks the unfused one; another chunk size gives the same run bit for bit.
  - `tools/fused_head_gate.py` on FineWeb-Edu text (d64 L2, T 64, batch 8, accum 2, 60 steps, the 16k llama3
    vocabulary, CPU): off vs c6365b88 bitwise, explicit off and a repeat bitwise; fused vs unfused first step
    bitwise, worst step 6.2e-8, final val 3.9e-9 relative, PASS
    (`docs/receipts/fused_head_gate_cpu_20261026.json`). The host path is the reference, not a speed path: 0.87x.
  - CUDA (`devops::linear_ce`; compiled with nvcc, not yet run on a GPU): the unfused CE kernels in place on
    each chunk, the GEMMs through the step-resident path. A new `gemm_tiled_acc` continues each element's FMA
    chain from the previous chunk, so with the tiled GEMM every output should be the unfused device path's bit
    for bit; under cuBLAS, float rounding. Device legs: `test_fused_head` leg 9 in
    `tools/colab_cuda_validate.sh` (tiled and cuBLAS), `tools/colab_chat/fused_head_cuda_v1.json` (the CUDA
    gate, `fused_head_gate.py` on 20m_L4 against c6365b88, `step_profile.py --configs fp32,fused_head` at 20m_L4,
    20m_L16, 60m_L32 and pk_base_v1; class head_fused).

- **Standard small benchmarks, IF-tiny and compliance by turn distance** (26 Oct;
  `experiments/benchmarks_2026-10-26/RESULTS.md`).
  - `tools/bench_build.py` builds HellaSwag, ARC-Easy/Challenge, PIQA, WinoGrande, LAMBADA and BLiMP from
    pinned revisions; `sources.json` holds the licences and the sha256s, and the data are never committed.
  - `tools/bench_eval.py` scores log-prob forced choice in lm-evaluation-harness's zero-shot formats, with
    Wilson intervals and chance. Its engines are `mtstudio score`, HF transformers, `hfgguf` (a paperkiln GGUF
    through transformers' GGUF loader, batched; within 3.2e-5 of mtstudio, argmax 20,581/20,581), ember's
    POST /score and the numpy IR. `compare` holds two engines against each other on the same requests.
  - On the benchmarks, pk_base_v1 is level with Pythia-160m except on LAMBADA and BLiMP (HellaSwag 0.292, ARC-E
    0.437, PIQA 0.612).
  - IF-tiny v1 (`corpus/eval/iftiny_v1.jsonl`, 574 items, 15 families) and turn distance v1
    (`corpus/eval/ifdist_v1.jsonl`, 1,344 conversations) come from `tools/iftiny_build.py`. Their checkers are
    `tools/iftiny_checks.py`, with 64 unit tests in `tests/test_iftiny_checks.py`.
  - `tools/iftiny_eval.py` generates greedily through ember's /chat or HF. It replays a second pass, checks
    replies against mtstudio's argmax, and checks the server's prompt ids against schema v1.
  - On IF-tiny our models sit near their base rates (pk_sft_v1 macro 0.123, bare 0.062). Over distance their
    compliance rises with the compliant history, not the instruction.

- **The extractor's gaps, closed or refused by name** (26 Oct; LOCAL_LLM_STACK.md section 11 item 4.2, formerly 4.3).
  - **Flex options** (`tools/parity_model.hpp` FlexConfig, mtstudio `arch.custom`):
    - An expr activation or norm may read more inputs (`activation_inputs` / `norm_inputs`): a gate projection
      (`mlp.c_gate`), learned scalars and per-channel vectors with their inits. A norm's inits can differ by
      place (`init_attn`, `init_ffn`, `init_final`).
    - An expr on the embed output (`embed`, `embed_inputs`).
    - Norm placement per sublayer (`norm_placement`: pre, post, sandwich, branch_post, none, or
      `{"attn": .., "ffn": ..}`), with no output norm when both sublayers are post-norm unless `final_norm`.
    - `position: "t5_relative"`: T5's bucketed relative bias (`rel_buckets`, `rel_max_distance`).
  - **IR**: the learned inputs export as `scale` nodes over a row of ones (`expr` "1"), with tensors
    `blk.{i}.<slot>_p_<letter>.weight`, `embd_p_<letter>.weight` and `output_norm_p_<letter>.weight`. The
    sandwich norms are `attn_post_norm` and `ffn_post_norm`. None of this needs a keramos change.
    T5's bias has no keramos op: its GGUF export refuses by name. Only with
    `PAPERKILN_IR_UNRATIFIED=t5_relative` does the export write the proposed
    `attention.position_bias {weight, buckets}`, and only for the test of the proposal.
  - **Ops**: `ops::t5_position_buckets` (T5's causal table, equal to HF's f32 computation at five
    configurations) and `ops::relpos_attention` (forward and backward). At their defaults the options give the
    old model bit for bit (test_flex's equivalence pin).
  - **Checks**: `tests/test_flex.cpp` 9-11 cover gradient checks of every new input, every one of the 25
    placement pairs against the block composed by hand, and the bias op against causal attention and finite
    differences. `tests/test_flex_ir.py` runs ten models through ir_check, and the numpy IR matches
    `mtstudio score` within 2.7e-7 to 3.2e-7, greedy 40/40. The T5 model is checked against the proposed
    numpy lines.
  - **Extractor** (`papers/llm_extract.py`): new fields `norm_placement_attn` / `_ffn` (with `branch_post`),
    `qk_norm` (per_head, full, unspecified, none), positional `t5_bias` with `rel_buckets` and
    `rel_max_distance`, `ssm` (mamba1, s4) with `d_state`, `d_conv`, `expand` and `dt_rank`, and each new op
    input's role and init. The IR ops list gains conv1d_causal and selective_scan. Older replies read the new
    fields as not stated.
  - **Chain** (`tools/arxiv_to_chat.py`): it builds the above from the roles, including a norm's variants with
    other inits at other places. A per-head q/k norm builds on the llama family, and Mamba-1 builds from its
    sizes on the mamba preset. `--assert` takes `qk_norm`, `d_state`, `d_conv`, `expand` and `dt_rank`. A
    structural op in the block graph that no family wires there now refuses: Primer's conv1d_causal on q, k
    and v used to pass silently.
  - **Scoring** (`papers/bench_score.py`, `papers/bench_truth_structure.json`): sizes and the block graph by
    data flow, against labels read off the 40 papers' LaTeX. `flavor_bench.TRUTH` now gives Primer's norm as
    its modified layer norm (out of lattice), no longer the T5 baseline's RMSNorm.

- **The third-party ops train here** (26 Oct; LOCAL_LLM_STACK.md section 11 item 4.4, which this closes). The
  ops of the keramos thirdparty profile that paperkiln had only in numpy now have forward and backward passes in
  its autograd. A model that uses them trains, exports its GGUF with its IR, and runs in ember.
  - **Ops** (`include/microtorch/ops.hpp`):
    - `ops::apply_rope` takes `pairing` (interleaved, or NEOX's (x[j], x[j + n_rot/2])) and `n_rot` (partial
      rotary). At its defaults it is the original loop, statement for statement.
    - `ops::rmsnorm_group`: a per-head RMSNorm with a weight shared by every group.
    - `ops::repeat_heads`: grouped-query attention's kv sharing. The backward sums each group in ascending
      head order.
    - Mamba-1's `ops::conv1d_causal` and `ops::selective_scan`: the backward runs through time over the kept
      states, and every sequence of a stacked batch starts from a zero state.
  - **CUDA**: the rope kernel takes the pairing and the rotated width, still rounding as the host does.
    repeat_heads has kernels that are bit-exact with the host. rmsnorm_group runs the rmsnorm kernels over
    rows of the group's width. The two Mamba ops run on the host; paperkiln's CUDA path has no recurrent
    kernels.
  - **Checks** (`tests/test_thirdparty_ops.cpp`, ctest `thirdparty_ops`):
    - Every op against a float64 statement of its IR_SPEC formula: 1.0e-7 to 2.3e-7.
    - Every input's gradient against central finite differences: worst 3.5e-4.
    - RoPE equals a float re-statement of the host loop bit for bit, and rmsnorm_group over the whole row
      equals ops::rmsnorm bit for bit.
    - The llama options at their default values give the plain block bit for bit.
    - A grouped-query model equals the multi-head model whose k and v columns repeat, bit for bit.
    - KV-cached decode is within 6.0e-8 of the full forward, and a Mamba-1 batch equals its sequences bit for
      bit.
  - **Numpy agreement.** `tools/ir_eval` runs rope, rmsnorm, attention, conv1d_causal and selective_scan
    nodes through these training ops, and mtstudio's conformance adapter routes them there. Every keramos
    single-op case they express passes against numpy's expected values: rope interleaved, neox, neox_partial
    and grouped_query; rmsnorm row and group; attention causal and grouped_query; conv1d_causal; selective_scan.
    That is 10 pass, 0 fail. Three cases are skipped (freq_scale, weight_offset, a window), none of which
    paperkiln trains.
  - **mtstudio**:
    - The llama family takes `arch.custom.kv_heads`, `rope_pairing`, `rope_dims`, `qk_norm` and `rope_theta`.
      Each left out gives the plain model, unchanged, with an unchanged model event.
    - The mamba preset takes `arch.custom.ssm: "mamba1"` with `d_state`, `d_conv`, `expand` and `dt_rank`.
      The graph is keramos `tools/gguf_ir.py`'s mamba graph node for node, with llama.cpp's tensor names and
      A stored as the forward's -exp(A_log).
    - GQA alone exports architecture llama with head_count_kv. NEOX, partial rotary and q/k norms export as
      "paperkiln".
    - muon refuses grouped-query attention, because its per-head split assumes one kv head per query head.
      attn_res refuses the options, because llama_attnres runs its own attention.
  - **End to end** (`tests/test_thirdparty_ir.py`): five models train 40 steps on CPU and export.
    - The models: tp_gqa (4 q / 2 kv heads); tp_qwen (GQA + NEOX + q/k norms, theta 1e6); tp_mqa_partial (one
      kv head, rope_dims 4 of 8); tp_neox_partial; tp_mamba1.
    - keramos's checker accepts every file. Numpy IR against `mtstudio score` is within 2.5e-7 to 4.3e-7 of
      max(1, |ref|), greedy 40/40 on each.
    - The gate files for ember are in `C:/ml_artifacts/thirdparty_train_gate/`.

- **The host GELU gives the same bits at every OpenMP thread count** (26 Oct). `ops::gelu` on the CPU runs
  the vendored `Matrix::apply_gelu` on one thread (`apply_gelu_one_chunk`, src/ops.cpp). The core compiles that
  loop with -O3 -march=native -ffast-math, so GCC vectorises it with libmvec's tanhf: 8 lanes over each thread's
  chunk, then 4 lanes and scalar tanhf for the chunk's last chunk_len % 8 elements, and the three round
  differently. An element's bits therefore depended on where the chunks fell. At 3, 5, 6 or 7 threads a
  stacked batch differed from its sequences run alone, which is the deep_swa batch pin's failure with
  `OMP_NUM_THREADS=3`, present since the test was written.
  - Every thread count now gives the one-thread bits. Those are also the 2- and 4-thread bits wherever those
    chunks were multiples of 8 elements, which covers every gate's sizes. A probe hashed `ops::gelu` on seven
    shapes and FlexLM logits on three: at 1-5 threads every hash now equals the old one-thread hash.
  - mtstudio gpt2 and flex runs at 1-5 threads give weights byte-identical to the old one-thread run. Before
    the change the flex run differed at 5 threads.
  - `tests/test_deep_swa.cpp` section 4b runs the batch pin at 1-8 threads in process and checks each count's
    logits against one thread's. ctest gives 42/42 at 2 and at 3 threads.

- **Tiny chat, arm X5: Attention Residuals on the llama base** (26 Oct; `experiments/tiny_chat_2026-10/PREREGISTRATION.md`
  amendment 5). Jonathan's cross-layer idea on Carry-5. The arm is 20m_L16 with `llama_attnres` (full form at
  sublayer level: 33 depth sources, zero-initialised pseudo-queries, scale "sources"), against E1's three home L16
  runs paired by seed, at E1 L16's learning rate. HX5-main is an item-level paired test (3 x 888 items) with Stage
  L0's verdicts. The isoplot against the plain 20M runs is reported beside it.
  - The port's checks: `tests/test_llama_attnres_reference.py` (the fused op against
    `python/attn_res_reference.py` under torch autograd, worst 7.1e-6), and the `ar_x5` case of
    `tests/test_llama_attnres_ir.py` (the arm's exact shape: numpy IR vs mtstudio 2.7e-6, greedy 40/40, keramos
    v1.1.3's validator accepts it). The ember gate files are in `C:/ml_artifacts/attnres_gate_x5/`.
  - Jobs: `tools/colab_chat/tiny_chat_s0_x5.json` (CUDA gate, probes, a 500-step plain replay of E1's
    pt_e1_L16_r1 and an X5 smoke run), `tiny_chat_e1_x5_ctl.json` (E1's L16 post jobs re-run from their home
    weights for their per-item records) and `tiny_chat_e1_x5.json` (the gate needs the Stage 0 receipts; its post
    jobs bring home items, replies and the SFT model, so the pick runs at home when the seeds run on separate
    machines).
  - Tooling: `x5/bitid.py`, `analyze_x5.py` (in `analyze.py` as E1_x5_20m), `tools/carry5_attnres_trace.py` (the
    depth-attention weights at the answer token, through keramos's interpreter), and `tiny_chat_run.py gate
    --need-ok`.
  - Stage 0 (585206a8, one RunPod L4; A5.12):
    - CUDA gate 22/22;
    - X5 at 17,378 tokens/s against plain 20m_L16's 26,007 (0.668x, +15% peak memory);
    - a 500-step replay of E1's pt_e1_L16_r1 bit-identical to E1's log;
    - the smoke model numpy-vs-mtstudio 1.9e-6, greedy 40/40.
  - Ember e115fd3 is the post jobs' pin (`post --ember-commit`): ar_x5 40/40 on CPU, CUDA and WASM, and the
    control's model generated exactly as at cbdcb915.
  - Result (`experiments/tiny_chat_2026-10/x5/RESULTS.md`): **HX5-main EQUIVALENT**. D = Y(X5) - Y(L16) =
    -0.0101, 90% [-0.0299, +0.0096], over 2,664 item-level pairs; both cells pass the gate.
    - Resampled by item, frame or slot cluster, the 90% interval reaches below -0.03 through the knowledge
      family. The planted-fact families stay within 3 points, and every 95% upper bound is at most +0.021.
    - X5 lowers every loss by 0.03-0.05 nats and raises twin accuracy 2.8 points, with Y unchanged.
    - The isoplot through the 22 plain 20M runs is weak evidence (slope -0.019 Y per nat, unstable).
    - The control's re-runs reproduce E1's L16 summaries exactly, and the score audit leaves every Y unchanged.
    - The pick (seed 3) ran at home.

- **The general chat model's next step: pk_sft_v1** (25 Oct; paperkiln 0ad646d5..91f47fec, one RunPod L4,
  $3.19; `experiments/chat_general/RESULTS.md`). pk_base_v1 + lever 1's SFT set at T 512 as conversation rows
  (`tools/lever1_conversations.py`: one row per dialogue, the per-turn file's 463,695 replies in 313,606 rows)
  under `train.sft_loss` all, BF16 + flash, 18,620 steps, held-out loss 1.508: simple-QA 21/60 against big_sft
  11, ctrl_sft and moe_sft 7, the live model 2; ember aa50a61 greedy parity 40/40 on CUDA and CPU. The winner
  rule (PLAN.md, fixed before the results) names the live model as registered and pk_sft_v1 under its
  amendment A1. Tooling: `bpe.load_vocab` (chat_eval, qa_eval, chat_sample_ember and sample_candidates read the
  16k vocabulary), `tools/colab_chat/general_chat_v1.json`, `chat_eval_stage.py`, `ensure_ember_vm.py`,
  `chat_parity_vm.py`, `pref_sample_vm.py`, `experiments/chat_general/winner.py`; chat_eval deletes its
  reference-scoring runs (they filled a 40 GB volume).
- **The IR's specification is keramos** (25 Oct). The spec text, schema, validator, numpy reference, conformance
  suite (with engines' receipts), worked documents and prior-art survey are the keramos repository
  (github.com/JohnnyTeutonic/keramos, private; tags vR.S.P, IR_SPEC 2.6), vendored at the pinned tag in
  `spec/`, now v1.1.1 (6170ee05: revision 1.1 with the rulings on ember's review points):
  `spec/IR_SPEC.md`, `spec/conformance/`, the tools in `spec/tools/`, the worked documents in
  `spec/examples/`. `spec/KERAMOS_PIN.json` records the tag, its commit and the commit's manifest;
  `tests/test_keramos_pin.py` fails on any local edit and `tools/sync_keramos.sh TAG` refuses to overwrite
  one. paperkiln's side of the suite is `tools/ir_conformance_mtstudio.py` (mtstudio's adapter and the
  training of the pk_ models) and `tools/ir_corpus.py` (both repositories' stored documents).
  paperkiln's C++ follows v1.1.1: `ir::canonical` runs every digest check on the whole document before
  removing its `_` members, and `ops::recurrence` accepts a repeated index in a declaration (`W[c,c]`).
- **The CUDA mtstudio checked against the CPU build on trained models** (25 Oct; paperkiln ff00d60a,
  `tools/cpu_cuda_parity.py`, `tools/colab_chat/cpu_cuda_parity_vm.sh`, RunPod L4). Two builds of one
  commit, the CUDA one under the production environment, on the live chat model (dense, 14.7M), big_sft
  (dense, d768 L12, 91M) and fw_moe (MoE, E 8 top-2, 70M): `mtstudio chat` greedy on the 40 frozen
  prompts identical 40/40 on each (text, token count, stop reason); `mtstudio score --logits` on the
  CPU's replies: top-1 equal at every scored position (676, 677, 1,592), largest |dlogit| 7.4e-7,
  7.2e-7 and 1.15e-6 of the logit range, |dlogprob| under 1e-5; the reference-bits/byte SFT
  evaluation (on CUDA, the device evaluation) within 1.1e-5, 8.3e-6 and 3.0e-6 relative.
- **SFT evaluation on the device; deferred downloads for the gpt2 family (defect D1 fixed)** (25 Oct;
  paperkiln ff00d60a). When a run's own steps defer (CUDA, device op set, step residency, deferred
  downloads), mtstudio's SFT evaluation runs its forwards inside one step window: the parameters
  upload once per evaluation, and activations, logits and the softmax stay on the device. CUDA
  eval events carry `eval_device` and `eval_s`; `MICROTORCH_DEVICE_EVAL=0` keeps the windowless
  path; CPU builds and deferral-off runs are unchanged. D1 was LayerNorm's `xhat` forward cache
  dying stale inside the window (docs/CUDA_PHASE_B2.md, record D1); it is now a self-discarding
  cache, so gpt2-family and layernorm-flex runs can defer (`tools/arxiv_to_chat.py` no longer turns
  deferral off for them). Gates on an L4 (`tools/device_eval_gate.py`, `tests/test_gpt2_device.cpp`):
  val_loss identical to the windowless evaluation on llama (eval_batch 1 and 4), gpt2 and flex, and
  training losses bit-identical; 100 held-out examples of ~370 tokens evaluate in 0.55 s instead of
  8.37 s at d384 L6 and in 2.58 s instead of 40.0 s at d768 L12 (about 15x); gpt2 exact, swa, kimi,
  srd and 4-layer flex train 60 steps under deferral with glibc's heap checks on, every training and
  validation loss bit-identical to write-through. `mtstudio score --logits FILE` writes each scored
  position's logits row (float32), for the CUDA-vs-CPU gate.
- **Real k-quant blocks in the 4-bit exports** (25 Oct; paperkiln 255c9561). `gguf::quantize_q4_k`,
  `quantize_q5_k`, `quantize_q6_k` are llama.cpp b6000's (4762ad73) `quantize_row_q*_K_ref` with
  `make_qkx2_quants`, `make_qx_quants` and `nearest_int`, statement for statement, so `gguf_q4_k_m`,
  `gguf_q4_k_s` and `gguf_q4_0` store Q4_K / Q5_K / Q6_K wherever ne0 is a multiple of 256 instead of
  refusing (llama.cpp's fallbacks below 256 are unchanged). The 4-bit variants now write the IR keys
  before `general.quantization_version` and `general.file_type`, as llama.cpp's quantiser does (it
  copies the source keys, then appends those two): since the IR went into the GGUF the files had
  differed from llama.cpp's in that order alone. Gates: on pk_base_v1 (d 768, d_ff 2048, 12 layers)
  every one of the 111 tensors of q4_k_m (73 Q4_K, 13 Q6_K), q4_k_s (80 Q4_K, 5 Q5_K, 1 Q6_K) and q4_0
  is byte-identical to llama.cpp b6000's `llama-quantize` (built with GGML_NATIVE off) run on
  paperkiln's own F32 export, and so are the three whole files (sha256 78b3482f..., 7980aa54...,
  89d8581c...; `C:/ml_artifacts/next_base/base/pk_base_v1_q4_k_{m,s}.gguf`); `tests/test_gguf_q4.cpp` checks an 8-layer d 256 export's
  types and bytes and 9 patterns x 2 super-blocks per type against llama.cpp b6000's own reference
  rows (hashes of its bytes).
- **The build uses the vendored transformer_core by default** (25 Oct). `third_party/transformer_cpp`
  is canonical; a sibling `../coalfire.cpp` or `../transformer_cpp` is used only with
  `-DMICROTORCH_TCPP_SIBLING=ON` (or `-DMICROTORCH_TCPP_DIR=<path>`). A fresh configure from the
  workspace checkout had picked the lagging sibling and failed in `src/gguf.cpp`
  (`GGUFWriter::write_metadata_int32_array`); so had a clone placed in a home directory holding a
  stale `transformer_cpp`. `tools/sync_vendor.sh` refuses to overwrite the vendored copy while
  the two differ.
- **One conversation schema for instruction tuning and chat** (24 Oct; LOCAL_LLM_STACK.md section
  3, points 3-5). A conversation is `{"messages": [{role system | user | assistant, content,
  loss?}]}` (one leading system message, strict alternation, tool reserved); an instruction is a
  one-exchange conversation. The template is a versioned record (`paperkiln/1` = the old
  `<|user|>U<|assistant|>A<|endoftext|>` template byte for byte, system folded into the first
  user turn behind one constant; `paperkiln/legacy-word`; ember's ChatML Qwen2.5/Qwen3 and
  Gemma 3 records) in `tools/chat_template.py` and its header-only C++ twin
  `include/microtorch/chat_template.hpp` (render, Jinja, Ollama Modelfile, resolution). mtstudio:
  `{"messages"}` SFT rows under `train.sft_loss` {"turns": "all" | "final" | "each"} (default
  all), `chat.template` in the spec, `/chat` takes `system` (or a leading system history entry)
  and `thinking`, `/health` names the template, `score` and `sample-set` take `messages` rows,
  `sft-dump` writes the rows a spec trains on; every GGUF of a vocabulary with the v1 specials
  carries `paperkiln.chat_template` and the generated `tokenizer.chat_template`, with
  `<name>.Modelfile` beside the F32 file. The Python builders (chat_data, carry5_eval, chat_eval,
  carry12_build, `_common.render_dialogue`) render through it. Gates: v1 bit identity on 460
  frozen prompts, 9,288 Carry-5 and 24,684 Carry-12 prompt strings and 99,608 pinned dialogues
  (each corpus's pinned text.txt sha256 reproduced), ids per piece = whole under both
  vocabularies; mtstudio's conversation rows bit-identical to the segments and per-turn rows on
  600 pinned dialogues, T 96 and 512, state tokens k 0 and 4, both vocabularies; Jinja = record
  under jinja2 on 1,000 random conversations; the third-party records = their HF templates; 29
  edge cases in both languages (`tests/chat_template_cases.json`).
- **BF16 mixed precision and flash attention, opt-in** (24 Oct; `train.precision` "fp32" |
  "bf16", `train.attention_kernel` "tiled" | "flash", llama family; the defaults are the previous
  step bit for bit and write nothing new into events, results or GGUFs). BF16 rounds the inputs
  of every `ops::matmul` GEMM (forward and both backward products) to bfloat16, round to nearest
  even, and accumulates and returns FP32: on the device `cublasGemmEx` (CUDA_R_16BF in, CUDA_R_32F
  out, CUBLAS_COMPUTE_32F; sm80+), elsewhere a host emulation (rounded copies through the FP32
  path). Master weights, AdamW state, gradients, norms, softmax, RoPE, residual adds and the loss
  stay FP32; no loss scaling (bfloat16 has float32's exponent, and no gradient is stored in it).
  Recorded as `precision` in the model event and result.json and as `paperkiln.train.precision`
  in the F32/Q8_0 GGUFs. Flash attention (`ops::attention_heads_flash`, `src/cuda_flash.cu`):
  online-softmax forward that keeps the per-row log-sum-exp instead of the [H T, seq_len]
  weights, and a recompute backward (dK/dV per key tile, dQ per query tile, no atomics), FP32,
  on the full, causal and turn_state masks, head widths up to 64; host loops over the same tiles
  are the reference. CPU gates (`tests/test_mixed_precision.cpp`): rounding against a
  nearest-neighbour reference on 1e6 floats, BF16 GEMMs and `ops::matmul` equal to the rounded
  FP32 GEMMs bit for bit, flash vs tiled within 5.7e-7 (y) and 5.6e-7 (dqkv) relative on every
  mask, FD gradchecks of the flash op (worst 1.1e-3), a 60-step llama run: flash vs tiled 2.3e-7,
  BF16 vs FP32 4.5e-3 per-step loss. Through mtstudio on the CPU (`tools/precision_gate.py`, d 64
  L2, 150 steps on FineWeb-Edu text, 16k vocabulary): precision off vs the pre-change binary bit
  for bit (every step's loss and grad norm, the validation loss, the safetensors and GGUF sha256),
  as are the explicit-off spec and a rerun; 20-step moving-average loss vs FP32 tiled: flash
  1.9e-8, BF16 7.5e-5, both 1.0e-4 (receipts `docs/receipts/*_cpu_20261024.*`). Every CUDA
  target compiles (nvcc 12.6, sm_89; the flash kernels spill nothing). Device legs in `tools/colab_cuda_validate.sh`; the GPU
  gate and throughput run is `tools/colab_chat/bf16_gate_v1.json`. `tools/step_profile.py`
  (per-op-class step breakdown) and `tools/precision_gate.py` (off-bitwise and loss-curve legs
  through mtstudio) come with it.
- **Masked cross-entropy on the device** (24 Oct, the SFT loss): `ops::cross_entropy_masked`
  had no device path, so every SFT step downloaded the [rows, vocab] logits, ran the softmax on
  the host and uploaded the gradient (1 GiB each way at batch 32, T 512, 16k vocab; the tiny-chat
  pilot's SFT ran at 3,490 tok/s on an L4 against 25,113 for pretraining the same model). New
  `devops::ce_fwd_masked` / `ce_bwd_masked` run `k_ce_fwd` / `k_ce_bwd` instantiated with a mask
  (masked rows carry target -1: nll 0, gradient exactly 0); the unmasked instantiation compiles
  to the previous kernels, so `cross_entropy` is untouched, and an all-ones mask still returns
  `cross_entropy` itself. Kept rows' P and gradient are the unmasked kernels' bit for bit; the
  loss differs from the host op only by the sum's order. The host path is unchanged. Tests:
  `tests/test_cuda_ops.cpp` leg 9 (Colab, `tools/colab_cuda_validate.sh`).
- **Mixture-of-Recursions, causal token choice** (23 Oct, tiny-chat amendment 4):
  `arch.custom.recursion` schedule "mor" with {"mor": {route "token_choice", gate "score" |
  "none", balance 0.1, z_loss 0.001, kv_rows "active"}}: one linear router (state key
  `recursion.router.weight`, GGUF `rec.0.router.weight`) at step 0's input picks each token's
  depth 1..R by its largest logit; the token takes that many repeats, each x + g (S(x) - x) with
  g the chosen depth's probability, on its sequence's active rows only (gathered; attention causal
  among them at absolute positions, so prefill equals decode); the balancing loss and z-loss go
  through `MoEAux`. Step and eval events log the depth fractions, mean depth, entropy and router
  losses; `mtstudio score` writes each prompt's depth trace. IR: the `depth_route` node and the kv
  slot's `kv_rows: "active"` (normative in ir.hpp), family `llama_mor`; `tools/ir_numpy.py` and
  `ir_check.py` follow it. Tests: `tests/test_mor.cpp` (forced full depth with gate none = fixed
  recursion bit for bit, logits and gradients; prefix, decode and stacked-sequence causality with
  depths identical; per-application rows exactly the active tokens'; FD gradcheck worst 7.2e-3;
  the device-resident step for Colab) and `tests/test_mor_ir.py` (numpy IR vs mtstudio within
  4.0e-7, greedy 40/40, depth traces 40/40). Jobs: `tiny_chat_s0_mor.json`, `tiny_chat_e1_mor.json`.
- **Tied-block recursion** (23 Oct, the tiny-chat recursion arm): llama-family
  `arch.custom.recursion` {prelude P, block B, repeats R, coda C, schedule "fixed", inject
  "none" | "add"} stores P + B + C layers and applies P, then B R times, then C, each application
  with its own KV cache and full backprop through all of them (`nn::layer_schedule`). IR: the
  block key `weight_index` {"per_layer": [...]} (application i reads blk.{m_i}), family
  `llama_recursive`, GGUF architecture `paperkiln` with the shared layers stored once;
  `tools/ir_numpy.py` and `ir_check.py` follow it. Tests: `tests/test_recursion.cpp` (R = 1 bit
  for bit, tied vs untied bit for bit, FD gradcheck worst 2.7e-3, KV cache, AdamW once per
  tensor, the device-resident step for Colab) and `tests/test_recursion_ir.py` (numpy IR vs
  mtstudio within 3.8e-7, greedy 40/40).
- **Tiny-chat experiments, pre-registered and prepared** (23 Oct): `experiments/tiny_chat_2026-10/`
  (PREREGISTRATION.md, `make_jobs.py`, `analyze.py`, shape and SFT specs),
  `tools/colab_chat/tiny_chat_run.py` and `tiny_chat_data.py`, jobs files
  `tiny_chat_{s0,e5,e1,s0_60m,e1_60m}.json` (not launched; E5's dialogue levels follow a
  registered rule from the measured human corpus, `design.json`, provisional until the default set
  is home). mtstudio: SFT examples in a
  `segments` form (a whole multi-turn dialogue per row, every assistant turn trained; a one-turn
  record trains bit for bit like its `{prompt, completion}` twin), `train.eval_windows` (LM
  validation windows, default 8 as before), and a clear refusal of a corpus whose training part or
  held-out 5% is shorter than T + 2 tokens (the window draw used to wrap round and crash);
  `tools/chat_data.py sft --all-turns` writes the segments form. Tests:
  `tests/test_mtstudio_segments.py`, `test_tiny_chat_pipeline.py`, `test_tiny_chat_analyze.py`,
  `test_chat_data.py`.
- **Runtime-defined ops** (22 Oct, `1159d63e`): the `expr` op, a closed elementwise grammar
  with row reductions parsed at runtime into differentiable primitives (new tape ops exp, log,
  sqrt, tanh, erf, abs, div, pow, maximum, minimum, where, rowsum, rowmax, broadcast_cols);
  flex `activation: "expr:<e>"` with an IR `expr` node; `tools/ir_expr.py` numpy reference;
  `tools/expr_eval`; `tests/test_expr.cpp` and `tests/test_ir_expr.py`. C++ vs numpy within
  1.19e-7; ember.cpp vendors the parser.
- **Third-party architectures, reference side** (22 Oct, `daab3e1d`, `a5cddee1`, `09943ab4`,
  `93a9037e`): `tools/gguf_ir.py` builds the IR from llama.cpp metadata (qwen2, qwen3, gemma3,
  gpt2, gptneox, mamba, windowed llama/mistral); `tools/ir_numpy.py` gains grouped-query
  attention, NEOX and partial RoPE, per-head RMSNorm, per-layer attributes, weight_offset,
  freq_scale, conv1d_causal and selective_scan; `tools/thirdparty_parity.py`,
  `tools/thirdparty_synth.py`, `tools/thirdparty_hfcheck.py`. GPT-2 124M and Pythia-160m 40/40
  greedy vs HF; Gemma 3 and Mistral sliding window on random tiny models. `tools/ir_run.py`
  runs on `ir_numpy`'s kernels (`d2b6d402`).
- **LLM-based arXiv extractor** (21-22 Oct, `dd772abe` onwards): `papers/llm_extract.py`, one
  Claude call per paper, a deterministic verifier, receipts and a cost cap; 82 / 9 / 1 on the
  40-paper benchmark against the rule-based 72 / 20 / 0. The docs present it as the main path.
- **Qwen pre-tokeniser** (22 Oct, `a8510d93`): `llama3_tokenizer.hpp` `max_digits` for
  llama.cpp's qwen2 regex.
- **Chat-model data and mtstudio features** (20-21 Oct): lever 1's SFT set
  (`tools/lever1_sft.py`), lever 2's dialogue corpora (`corpus/fetchers/dialogue_pt.py`,
  `corpus/decontam.py`, `corpus/mix_pretrain.py`), the 16k Llama 3 scheme vocabulary and its
  exact C++ encoder, `.ids` corpora, `data.anneal`, the `/chat` repetition penalty, the KV cache
  (`chat --kv-cache`), batched SFT evaluation (`1f1f906b`); the next-base launch
  (`tools/colab_chat/make_next_base.py`); Colab data jobs, resumable uploads and weights-only
  resume (`runner_multi.py`, `jobs_vm.py`).
- **Atlas Stages 2-3 with `rope_heads=all`** (10 Oct): 84 runs; none of the nine affected
  findings overturned (`docs/ROPE_HEADS.md`).
- **sparse_s1_longbudget** (10 Oct): 20 runs, "not supported / not supported"; registry row
  S1f-longbudget-monotone.
- **The transfer paper** (submitted to Machine Learning 9 Oct; Study 3 done 11 Oct, reading
  mixed; registry row T3-rate-specific-disagreement) was retired on 20 Oct after the desk
  rejection.

- **Fixed: llama-family RoPE rotated only the first head** (6 Oct,
  `cb135a1`). `arch.rope_heads` selects `all` (the default for new runs) or
  `first` (every earlier llama run, which replays as recorded); exports and
  the browser engine follow the run's record. Nine Atlas Stage 2/3 findings
  are under review; see `docs/ROPE_HEADS.md`.
- **Chat quickstart** (6 Oct): `quickstart_chat.sh`; TinyChat corpus
  generator (v1 and v2) with an end-of-text token; `tinychat-quick` and
  `tinychat-better` specs; `mtstudio chat` and `serve` on cpp-httplib;
  `data.max_tokens`; `tools/model_card.py` (card with a fixed chat probe
  set); `tools/train_progress.py`; `tools/package_chat_model.sh`;
  `tools/publish_hf.py` (Hugging Face Hub); `web/chat/`, a zero-server
  browser chat engine in plain JavaScript with a parity test.
- **Extractor** (6 Oct): highway and sliding-window patterns, window and
  sink extraction (`AUX_PATTERNS`), Mistral as a named SWA ancestor.
- **`tools/synthesis/`** (16–17 Sep): architectural-synthesis generator,
  arXiv scoper with field-term backstops, hand check of the top OPEN ideas,
  a mathematics column, and the folk-claims scoping.
- **Fresh-clone path** (14 Sep): `tools/get_tinystories_data.py` builds a
  public corpus and vocabulary; `tools/hf_chat.py` chats with an exported
  model from Python.
- **transfer_s2** (11–16 Sep): pre-registered learning rate per width;
  stage 1 lr selection, stage 2 S, M and L arms and the exploratory Lx
  arm banked. The pre-registered falsifier fired (14 Sep): F1 0.627, not
  adopted; rho -0.60.
- **transfer_s1** (31 Aug–7 Sep): bridge gate passed, arms S (72), M (72)
  and L banked, licensed reading in `RESULTS_M.md` and `RESULTS_L.md`.
- **Checkpoint/resume with full optimiser state** (4 Sep, `884aa0f`),
  bit-identical on CUDA; events.jsonl trimmed to the checkpoint on resume.
- **`mtsweep --shard K/N`** (5 Sep) to split an arm across VMs.
- **Colab runner** (Sep): relays partial checkpoints across the
  60-minute session prune, verifies the resume restore before launch,
  chunked idempotent uploads, re-adopts orphaned VMs
  (`tools/colab_adopt.py`) and refreshes the session key.
- **Docs:** `ROADMAP.md` is the single list of open work; the in-flight,
  backlog and handoff pages are retired (9 Oct).

### Added
- **CUDA B2.2 (partial): masked attention softmax on-device** — fused
  (causal/block) and swa (window+sinks) forward kernels plus one shared
  backward (`ds = A .* (ds - rowdot(ds .* A)) * scale`; masked entries carry
  A == 0, so no mask bookkeeping), in place over the [T,T] scores through the
  B2.1b value-cache seam: under deferral the score/weight matrix never
  crosses the bus inside a step. Host loops retained as fallback and
  reference; `test_cuda_ops` leg 4 pins all three flavors x {off, on, defer}
  at the composed-tape tolerances.
- **CUDA B2.2 (complete): embedding + cross-entropy on-device** — embed
  gather (first activation born resident), CE forward as softmax + nll +
  on-device sum so the [R,vocab] logits never cross the bus and the host
  receives one float, CE backward (P - onehot) * g with the gradient first
  touching host at accumulate() (B2.3's choke). Leg 5 composes
  embedding -> CE and pins loss + scatter-add table grad across
  {off, on, defer}. **T4-VALIDATED 30 Aug** — 12/12 suites, 281 checks, 0
  fails (receipts docs/receipts/receipts_b22_t4_20260830.txt); the first run's leg-4
  heap-corruption crash (a deferred temporary dying stale, then step_end
  downloading into the freed pointer) is fixed by the new
  `device::discard()` primitive, confirmed both directions on T4. Next:
  B2.3 (optimizer + accumulate on-device).
- **CUDA Phase B COMPLETE — adoption gate passed (31 Aug)**: B2 beats CPU
  AVX **21x at d=256 and 30.5x at d=512** (T=512, L=4, Tesla T4) with
  identically-converging trajectories (final losses match at print
  precision). Rung C of the scale ladder runs on CUDA; a ~6-CPU-hour cell
  becomes ~12 minutes. Receipts docs/receipts/receipts_b2gate_t4_20260831.txt.
- **Fixed: deferred activations feeding a gemm uploaded stale host data.**
  `window_operand` consulted the B2.1b value cache only for slotless
  operands; every Variable's data carries a slot, so under deferral a gemm
  reading a deferred activation uploaded the untouched (zero) host buffer.
  Real models trained at exactly ln(vocab) loss while the 285-check suite
  passed — the bug needs "deferred activation -> slotted gemm with no
  intervening materialize", a shape no test had. The stale hit is now
  universal; regression leg 7 covers the shape. Found by the adoption
  benchmark, which was the first end-to-end run at realistic vocab/width.
- **CUDA B2.3 complete + T4-VALIDATED (30 Aug)** — full step residency:
  persistent device optimizer state (B2.3b), device-side gradient
  accumulation with the materialize choke moved to the step boundary
  (B2.3c, 27-site backward audit), 12/12 suites and 285 checks green with
  legs 4/5/6 identical between plain and fully-deferred configs. The gate
  caught a second dying-temporary heap corruption (gradient temps consumed
  by stale-hit axpy, then step_end writing their freed buffers); fixed by
  a self-discarding rvalue `accumulate(Matrix&&)` overload. The lifetime
  rule is now enforced three ways: ~Variable discards, backward()
  discards consumed non-leaf grads, rvalue accumulate discards temps.
  Remaining: the d=512 wall-clock adoption gate.
- **CUDA B2.3a: optimizer steps on device (write-through parity seam)** —
  `k_adamw_step`/`k_sgd_step` with host-computed bias corrections so the
  math matches nn.cpp verbatim; real SGD/AdamW try the devops path per
  param with host loops as reference; leg 6 drives 5-step trajectories.
  Persistent device state (B2.3b) and device-side accumulate (B2.3c) next.
- **Llama-family models** (`nn::Llama`): RMSNorm, RoPE, SwiGLU, GQA, tied
  embeddings; HF-native parameter names; module-level FD gradchecks.
- **mtstudio**: spec-driven run driver (JSON spec -> train -> eval -> export),
  JSONL event stream, gradient accumulation, early stopping, GGUF export with
  embedded vocabulary, `--serve` live dashboard mode.
- **Studio dashboard** (`studio/index.html`): loss/val curves, per-module
  gradient glow, SRD gate track, spec builder; renders a dropped
  `events.jsonl` or polls a live serve.
- **GGUF export** verified end-to-end: mtstudio-trained Llama generates
  coherent TinyStories text through tinyllama.cpp.
- **SRD (Surprise-Routed Density)** research tools: `srd_parity` (four-lane
  falsifier protocol), `srd_needle` (associative recall with batch, difficulty,
  and seed knobs); results ledger in docs/SPARSE_ATTENTION.md.
- **Mamba S4 through the tape**: `ops::ssm_scan` with real BPTT, FD-checked.
- **CUDA seam** (`MICROTORCH_CUDA`): `device::matmul` dispatch through the
  transformer_core kernel tree, validated on Colab T4.
- Vendored `transformer_cpp` build surface in `third_party/` so clones build
  without sibling checkouts (`tools/sync_vendor.sh` refreshes it).

### Fixed
- CI: artifact actions v3 -> v4, coverage package list, codecov v5, lcov 2.x
  strictness; whole-tree clang-format so the style gate reflects reality.

## 2026-08-29
- CUDA Phase B2.1b T4-VALIDATED: deferred downloads behind
  MICROTORCH_DEFER_DOWNLOADS=1 (value cache, materialize boundaries,
  accumulate choke point); defer-vs-write-through composed-tape diffs
  exactly 0.0. Receipts: docs/receipts/receipts_b21b_t4_20260829.txt.
- The seed lottery exhibit: atlas/SEED_LOTTERY.md + tools/seed_lottery.py
  render the banked boundary receipts as the case for multi-seed,
  pre-registered ablation (two one-seed labs contradict 48% of the time
  past 2000 steps). README gains the exhibit + the registry scoreboard.
- CITATION.cff added.
- sparse_s1_seeds pre-registered (B*(256): point or distribution; ten new
  seeds); sweep in flight.
- Rung B (sparse_s1_scale), the Highway registry pilot
  (experiments/registry_0001_highway + registry/0001_highway_networks),
  and their receipts brought under version control.

## 2026-08-28
- Repo root reorganised: results CSVs -> experiments/srd_needle_2026_07/,
  atlas docs -> atlas/, completed phase docs -> docs/history/, design and
  reference docs -> docs/; 38 files of cross-references rewritten.
- ROADMAP: uplift plan adopted (scale ladder as keystone, CUDA-B as its
  prerequisite, extractor decoupling, benchmark to D&B scale, mechanism
  freeze). README: extractor provenance framing corrected.

## [0.1.0] — Phase 1-3 foundation

- Reverse-mode autograd on a DAG tape; gradient checkpointing.
- SafeTensors loading; GPT-2 and Qwen parity verification.
- Novel mechanisms with unit + gradient tests: Kimi linear attention,
  cerebellum gating, Mamba state-space.
- GTest suite, benchmarks, Doxygen documentation workflow.
