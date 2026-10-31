# paperkiln roadmap

*The one list of what is left to do, as at 30 October 2026. Finished work
moves to [`CHANGELOG.md`](CHANGELOG.md) and comes out of here. Everything
else under `docs/` and `atlas/` is reference (design, records, results);
where one of those documents carries a plan, this file wins.*

## 1. Now

0. **Tiny chat from scratch (LOCAL_LLM_WORK.md E1;
   `experiments/tiny_chat_2026-10/`, `experiments/tiny_chat_long_2026-10/`).**
   E5, E1 at 20M/60M, recursion and MoR are home and analysed
   (`experiments/tiny_chat_2026-10/analysis_2026-10-28.json`). Stage L0 on the unchanged 150M
   pick is complete and home (single-pick descriptive result in the long-carry RESULTS).
   Carry-12's registered HS1 extension is complete and all four
   new B/S-gist runs are home: the five-seed S-gist minus B contrast is SUPPORTED
   (-12.668 points), with six recorded full-history parity mismatches now under review.
   Its registered S-full condition is numerically true, but no control has launched;
   parity evidence, recipe and spending require review. The broader programme remains open.
   The complete three-seed Carry-12 core is analysed in `experiments/tiny_chat_long_2026-10/RESULTS.md`:
   HT-SFT SUPPORTED, other primary tests unresolved at three seeds. The separate five-seed
   HS1 result now activates the S-full condition. The registered size
   extension is complete at five matched seeds (`experiments/tiny_chat_2026-10/size/RESULTS.md`):
   primary size slope EQUIVALENT within ±0.01 Y/doubling under this fixed-token protocol; secondary 60M→150M step unresolved.
1. **The chat model (LOCAL_LLM_WORK.md E2).** Full preference judging at 256 new tokens
   with the rubric unchanged, then DPO + NLL and the five-part gate in `experiments/chat_general/PLAN.md`.
   The winner, both pilots and full sampling are complete (3,020 candidate rows home in
   `C:/ml_artifacts/pref_full/sample/`). The read-only readiness plan needs 5,660 more rankings
   with the existing cache (`receipts/pref_full_readiness_20261030.json` in the chat experiment).
   The delegated 28 Oct decision is recorded in
   `experiments/chat_general/RESULTS.md`, "The full run" (00ecd0d4); judging and training remain pending,
   with the recorded $200 total cap including the cache and held-out side-by-side gate.

The 30 Oct publication integration passes all 49 CPU CTests and 65 targeted Python
integration tests. Changed C++ ranges pass clang-format 18 checks, but 17 files
retain whole-file formatting failures at or below the exact pre-existing baseline;
this is not an all-green CI claim. The publication's GitHub test workflow has since
succeeded (41048998212); that does not establish success of every workflow. Source-bound evidence is in
`docs/receipts/publication_lowbit_qktemp_20261030/`. The chat-v1 release serves the
quickstart's ready-made model. cppcheck and Valgrind still run as advisory steps
that cannot fail the build.

## 3b. Environment traps found 17 Oct 2026

- WSL git reports ~238 files modified in the OneDrive checkout (CRLF), so
  `--require-clean` runs must come from an ext4 clone.

## 4. Research

- **Post-training and corpus provenance** (charter 0001, order of work §11).
  Remaining preference execution is listed in section 1; completed corpus, verifier, SFT, GRPO,
  DPO, byte-BPE and chat-model milestones are recorded in CHANGELOG and the charter.
  Other open follow-ups whose completion is not established here:
  - remaining per-token host work needs a current representative profile. CUDA/device-op outputs already avoid host zero-fill (20 Oct); a bounded tiny CPU teacher-forced score audit found no justified zero-fill change and does not close the historical CUDA follow-up (`docs/receipts/host_matrix_cpu_audit_20261031/validation.json`).

- **Scale ladder, depth rung.** d=256, 4 layers, exact against SWA, as its
  own pre-registration. Deep SWA is built and gated.
- **Atlas Stages 5–6.** Architectural fingerprints, neighbours, and the
  Atlas surface in the studio (`atlas/ARCHITECTURE_ATLAS.md` §19).
- **Sparse attention** (`docs/SPARSE_ATTENTION.md`):
  - V2 sketch-state attention (no code yet);
  - the V3 bake-off;
  - the R2-efficiency row, still pending;
  - the optimiser-interaction follow-up.
- **Parked lines, resumable.**
  - consensus prefill, commit-gated TTT and the designer pilot, parked until the journal verdicts;
  - SRD rung 2b;
  - the sparse S1 depth rung;
  - V2 CoD;
  - a sparse_s1_seeds re-run under the provenance rules.
- **atlas_selection follow-up.** A prospective, position-diverse
  shortlist. So far it is a proposal only.
- **Synthesis tool.**
  - 37 ideas were never scoped, because the API credit ran out.
  - The maths-column OPEN verdicts need a hand check.
- **Archaeology registry** (`registry/`). After highway, the next entries
  are Grid LSTM, gMLP, an RWKV-v4 block and retention. Each one grows the
  spec grammar. Verdicts are scoped to protocol and scale.
- **Chimera** (designed-experiment search) and the backprop-free lane
  (`docs/STUDIO_PLAN.md` §11) are parked.
  - Chimera's output is findings rows under pre-registration, never
    autonomously written papers.
  - Falsifier discovery is the result that would lift it.

## 5. Extractor

- **The LLM extractor is the main path** (`papers/llm_extract.py`; the 40-paper bench under the 26 Oct
  prompt, 27 Oct, $18.08: flavours 81 correct / 11 abstain / 0 wrong vs the rule-based 73 / 19 / 0; sizes
  569 / 2 / 71 vs the rule-based 136 / 8 / 498; block graph 91 / 2 / 2; per-sublayer placement 34 / 0 / 0;
  qk_norm, ssm and learned inputs 40 / 0 / 0 each; positions outside the old lattice 5 / 0 / 2; new-op inputs
  6 / 0 / 1; `papers/bench_score.py` against `papers/bench_truth_structure.json`,
  `papers/llm_extract_receipts/2026-10-27_bench_scores.json`). The same replies rescored by the 28 Oct
  verifier and scorer, with Mamba, Qwen3 and OLMo 2 labelled (43 papers; OLMo 2 has no reply under the
  bench's prompt): `2026-10-28_bench_scores.json` (`experiments/arxiv_chain_2026-10-28_e3/`). Open: prompt optimisation.
- **Publish `paperkiln-fetch` 0.1.1.** Wheel and source archive are built, licensed, metadata-checked
  and smoke-tested from an isolated installation (`docs/receipts/fetch_release_011_build_20261030.json`).
  Publication is blocked by a missing configured PyPI API token;0.1.0 is still the public release.
- **Grow the benchmark from 40 to 60–100 papers**, with ground truth read
  off the fetched source and never recalled. Add a reconstruction-fidelity
  task (parameter-count error per reconstructed paper) for a datasets and
  benchmarks paper. Candidates next in line: StarCoder, ELECTRA, UL2,
  BigBird, Chinchilla.

## 6. Engine

- **CUDA** (records: `docs/CUDA_PHASE_B2.md`):
  - Finish the host-read boundary audit in ops.cpp/mtstudio and other unreviewed
    host loops. The parity-model export subset is guarded and CPU DEVCHECK-tested
    (docs/receipts/export_host_read_20261031/); its new device-negative leg still
    needs actual CUDA execution. The audited non-WIP boundaries are wired and
    pass actual L4 CUDA+DEVCHECK integration (`docs/receipts/devcheck_host_read_cuda_20261030/`);
    those completed guards leave this list.
  - Coalesce loads in the transposed GEMM paths.
  - Re-measure the speedup on the op-set configuration over a full run
    before any paper quotes it. The 21x and 30.5x figures were measured
    over nine steps with deferral on.
- **BF16 attention production integration.** The completed 40-cell descriptive profile
  (`docs/receipts/bf16_profile_20261030/RESULTS.md`) measures attention spans at 16–34% of
  baseline instrumented GPU-span time. The standalone dense MHA primitive now passes its fixed
  32-case L4 FP32/BF16 matrix and independent raw replay, with FP32 softmax/backward and an explicit
  BF16 straight-through surrogate (`docs/receipts/bf16_attention_20261031/RESULTS.md`).
  Production wiring, model-level forward/backward parity and loss, and incremental end-to-end
  benefit remain open. The primitive matrix, profile, noise-floor gate and launch-count work
  are complete; these results do not establish a production speed gain.
- **Mamba:** a parallel scan and device kernels for Mamba-1's `ops::conv1d_causal` and
  `ops::selective_scan` (sequential, host only; the mamba preset's `ssm: "mamba1"`).
- **int4/NF4 quantisation.**
- **HF export for modified flex flavours and unmapped GPT2 variants; quantised GGUFs outside the llama family.** Strictly plain GPT2- and Llama-flavour Flex now have verified HF mappings for separate or explicitly tied token/output heads (arbitrary depth/MLP width); those completed subsets leave this list. Actual saved/reloaded Transformers CPU parity is recorded in docs/receipts/hf_tied_flex_20261031/. Plain GPT2 Flex additionally supports explicitly recorded exact-erf GELU (HF gelu), tied or untied, with evidence in docs/receipts/hf_exact_gelu_20261031/. Plain GPT2 Flex ReLU also passes finite-execution HF CPU parity for tied/untied heads, with documented NaN/negative-zero differences (docs/receipts/hf_relu_flex_20261031/). Other modified flavours remain open. Every mtstudio family now exports F32 GGUF with its IR
  (`include/microtorch/ir.hpp`, Families): gpt2 and flex with the GPT-2
  flavour in llama.cpp's `gpt2` layout, as are kimi and plain-residual flex
  of that flavour (the IR alone carries what differs); srd, attnres, mamba
  and the other flex flavours as architecture `paperkiln`. Quantised
  variants are refused by name. The window and sinks live in the IR only
  (family_hint `gpt2_swa`), so llama.cpp loads a sliding-window file but
  computes dense attention.
- **The taxonomy's "stream" lattice slot.**

## 7. Studio, chat and distribution

- **Open-weight models on our own engines: speed.** Parity is done through the IR,
  on the CPU and on CUDA. CPU comparison is complete for SmolLM2-135M and Gemma3-270M
  at three lengths (strict full-row parity); paired AVX2 improves prefill 2.38-2.87x
  and decode 1.04-1.50x but remains slower than Transformers. Evidence:
  `../ember.cpp/experiments/hf_cpu_2026-10-30/`. The SmolLM2-135M CUDA subset is now complete:
  actual L4, FP32, one host thread and eager HF, with 96 full-row/greedy parity checks
  and 24 immutable ABBA timing trials independently replayed. Ember decode medians
  are 3.80-3.95x faster on these synthetic runtime workloads; the 512-token prefill
  medians are close and four observations do not establish a broad prefill advantage.
  Timers include synchronous CPU-ID to CPU-row transfers and runtime dispatch, rather
  than isolated kernels or chat. See the scoped results and exact receipt in
  `../ember.cpp/experiments/hf_cuda_2026-10-30/` (Ember commit 7b546d8).
  Remaining: Gemma3-270M has no CUDA comparison result and is not queued.
- **Browser performance and remaining device gates** (LOCAL_LLM_WORK.md E9, E10).
  The IR family and recurrence paths are built in WebAssembly/WebGPU. The 28 Oct hardware
  receipt (`../ember.cpp/experiments/webgpu_gpu_2026-10-27/`) covers the existing tiny families,
  Gemma-3-270m, Qwen2.5-0.5B, Qwen3-0.6B and the page chat path on an L4. This does not
  establish hardware WebGPU execution for operations currently delegated to WebAssembly.
  E4's later CUDA gate is complete (eight actual GPU families and two explicit CPU fallbacks). Small-IR browser speed remains a measured limitation
  (state4 24.7 replies/s WebGPU against 136 in single-threaded WebAssembly in that receipt).
- **Windows testing of the quickstart** (parked).

## 8. Ecosystem (coalfire.cpp, ember.cpp)

*Reference: [`docs/ECOSYSTEM.md`](docs/ECOSYSTEM.md).*

- **C2, the paperkiln/coalfire training logit-parity pin:** no receipt exists for that contract.
- **C3, a single GGUF writer** (coalfire side).
- **C5, tokenizer consolidation with coalfire.** Paperkiln already trains byte-level BPE models
  (`tools/bpe.py`, `bpe_tokenizer.hpp`); the coalfire consolidation contract remains open.
- **Technique transfer still open** (`docs/TECH_TRANSFER.md`):
  - SiTU-GLU and quantile balancing;
  - gated MLA/NoPE;
  - MXFP4;
  - mHC;
  - multi-token prediction.
  - The KDA reference that `python/attn_res_reference.py` mentions
    (`kda_reference.py`) is not in this repo.
