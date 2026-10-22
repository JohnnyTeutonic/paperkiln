# Changelog

Notable changes to microtorch. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versions are tagged `v*`.

## [Unreleased]

### September–October 2026

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
