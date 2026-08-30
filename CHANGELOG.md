# Changelog

Notable changes to microtorch. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versions are tagged `v*`.

## [Unreleased]

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
  fails (receipts docs/receipts_b22_t4_20260830.txt); the first run's leg-4
  heap-corruption crash (a deferred temporary dying stale, then step_end
  downloading into the freed pointer) is fixed by the new
  `device::discard()` primitive, confirmed both directions on T4. Next:
  B2.3 (optimizer + accumulate on-device).
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
  exactly 0.0. Receipts: docs/receipts_b21b_t4_20260829.txt.
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
