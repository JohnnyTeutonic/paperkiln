# paperkiln roadmap

*The one list of what is left to do, as at 30 October 2026. Finished work
moves to [`CHANGELOG.md`](CHANGELOG.md) and comes out of here. Everything
else under `docs/` and `atlas/` is reference (design, records, results);
where one of those documents carries a plan, this file wins.*

## 1. Now

0. **Tiny chat from scratch (LOCAL_LLM_WORK.md E1;
   `experiments/tiny_chat_2026-10/`, `experiments/tiny_chat_long_2026-10/`).**
   E5, E1 at 20M/60M, recursion and MoR are home and analysed
   (`experiments/tiny_chat_2026-10/analysis_2026-10-28.json`). Remaining: the registered size
   extension's 20M/60M seeds 4-5 (all five 150M runs are home), Stage L0 on the registered 150M
   pick, and Carry-12's full analysis after the remaining post summaries arrive. The size extension
   rule has fired; its pins and 150M pick inputs passed
   `experiments/tiny_chat_2026-10/size/readiness_2026-10-30.json`.
1. **The chat model (LOCAL_LLM_WORK.md E2).** Full preference judging at 256 new tokens
   with the rubric unchanged, then DPO + NLL and the five-part gate in `experiments/chat_general/PLAN.md`.
   The winner, both pilots and full sampling are complete (3,020 candidate rows home in
   `C:/ml_artifacts/pref_full/sample/`). The read-only readiness plan needs 5,660 more rankings
   with the existing cache (`receipts/pref_full_readiness_20261030.json` in the chat experiment).
   The delegated 28 Oct decision is recorded in
   `experiments/chat_general/RESULTS.md`, "The full run" (00ecd0d4); judging and training remain pending,
   with the recorded $200 total cap including the cache and held-out side-by-side gate.
   Release of the staged F16 browser model remains pending Jonathan's go (E2.2).

## 3. Broken now

- **`mtstudio score` on CPU is slow for `llama_attnres`.** Stage L0 on pick_e1_x5_X5 (20m_L16 + Attention
  Residuals) scored about 4.5 min per chunk, against about 0.8 min for plain 20m_L16's pick on the same laptop
  build (`~/pk_state_build/mtstudio`, four slots; the process sat at ~210% CPU). Not diagnosed. On the L4, training
  runs at 0.67x plain.

The last recorded CI status is green on all three workflows (Code Quality, the test
suite with its Python suites and the fetcher drift check, and the wheel
build with its smoke test), and the chat-v1 release serves the quickstart's
ready-made model. One open choice: cppcheck and Valgrind still run as
advisory steps that cannot fail the build.

## 3b. Environment traps found 17 Oct 2026

- WSL git reports ~238 files modified in the OneDrive checkout (CRLF), so
  `--require-clean` runs must come from an ext4 clone.

## 4. Research

- **Post-training and corpus provenance** (charter 0001, order of work §11).
  Remaining preference execution is listed in section 1; completed corpus, verifier, SFT, GRPO,
  DPO, byte-BPE and chat-model milestones are recorded in CHANGELOG and the charter.
  Other open follow-ups whose completion is not established here:
  - the remaining per-token host work (zero-filled Matrix construction in the profile);
  - a VM-side snapshot before each checkpoint relay, so relays stop racing saves;
  - HF-export round trip of an SFT'd model (the brief's Phase 3 export check).

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
  bench's prompt): `2026-10-28_bench_scores.json` (`experiments/arxiv_chain_2026-10-28_e3/`). Open: a
  studio button that runs it instead of `fetch.py` when a key is configured; prompt optimisation later.
- **Two lexical gaps found while fixing the wrong assertions:** the GELU
  pattern does not match the spelling "GeLU" (Megatron, BART), and a
  vetoed runner-up still marks a field as contested.
- **Publish `paperkiln-fetch` 0.1.1** with the extractor fixes (0.1.0 on
  PyPI predates them).
- **Grow the benchmark from 40 to 60–100 papers**, with ground truth read
  off the fetched source and never recalled. Add a reconstruction-fidelity
  task (parameter-count error per reconstructed paper) for a datasets and
  benchmarks paper. Candidates next in line: StarCoder, ELECTRA, UL2,
  BigBird, Chinchilla.
- **Fetcher v2 remainder:** MoE fields and the HF-config cross-check. The
  per-variant fields and `n_kv_heads` are done.

## 6. Engine

- **CUDA** (records: `docs/CUDA_PHASE_B2.md`):
  - The third-party ops' device legs (queued until the RunPod top-up): root `scripts/colab_driver.json`
    entry pk-tp-cuda, `tools/colab_chat/thirdparty_cuda_v1.json` (the CUDA gate with
    `tests/test_thirdparty_ops.cpp`, and `tests/test_thirdparty_ir.py`'s five models trained with the device
    op set against numpy's IR).
  - Wire `MT_DEVCHECK_HOST_READ` at real call sites; today it is defined
    and never called.
  - Coalesce loads in the transposed GEMM paths.
  - Re-measure the speedup on the op-set configuration over a full run
    before any paper quotes it. The 21x and 30.5x figures were measured
    over nine steps with deferral on.
- **BF16 mixed precision and flash attention** (`train.precision` bf16, `train.attention_kernel`
  flash; built opt-in and CPU-gated 24 Oct 2026, CHANGELOG; v1 on an L4: CUDA legs 19/19, precision
  off bitwise equal to dc29b5e6): the noise-floor gate `tools/colab_chat/bf16_gate_v2.json` (3 seeds,
  rule in `docs/receipts/bf16_gate_v2_prereg_20261025.md`) and the throughput per shape. Then, per the profile
  (`tools/step_profile.py`): BF16 inside the attention products only if attention shows up. (Launch count is
  not the limit at the tiny-chat shapes: the GPU idles 1-2% of a step; LOCAL_LLM_STACK.md section 10, 28 Oct.)
- **Mamba:** a parallel scan and device kernels for Mamba-1's `ops::conv1d_causal` and
  `ops::selective_scan` (sequential, host only; the mamba preset's `ssm: "mamba1"`).
- **int4/NF4 quantisation.**
- **Kimi linear attention:** the non-causal backward (`src/ops.cpp:658`
  throws).
- **HF export for the flex and gpt2 families; quantised GGUFs outside the
  llama family.** Every mtstudio family now exports F32 GGUF with its IR
  (`include/microtorch/ir.hpp`, Families): gpt2 and flex with the GPT-2
  flavour in llama.cpp's `gpt2` layout, as are kimi and plain-residual flex
  of that flavour (the IR alone carries what differs); srd, attnres, mamba
  and the other flex flavours as architecture `paperkiln`. Quantised
  variants are refused by name. The window and sinks live in the IR only
  (family_hint `gpt2_swa`), so llama.cpp loads a sliding-window file but
  computes dense attention.
- **Kimi and SRD in mtstudio beyond two-block parity models.**
- **The taxonomy's "stream" lattice slot.**

## 7. Studio, chat and distribution

- **More `paperkiln` wheels:** macOS and Windows builds (Linux x86-64
  only today).
- **Studio features:**
  - Research Mode: clone a run with one change, compare two runs;
  - a sweep heatmap;
  - a fit-to-VRAM budgeter;
  - an evaluation-probe stage in the spec (needle and behavioural
    probes).
- **Open-weight models on our own engines: speed.** Parity is done through the IR,
  on the CPU and on CUDA; benchmark ember against `transformers` for speed.
- **Browser performance and remaining device gates** (LOCAL_LLM_WORK.md E4, E9, E10).
  The IR family and recurrence paths are built in WebAssembly/WebGPU. The 28 Oct hardware
  receipt (`../ember.cpp/experiments/webgpu_gpu_2026-10-27/`) covers the existing tiny families,
  Gemma-3-270m, Qwen2.5-0.5B, Qwen3-0.6B and the page chat path on an L4. This does not close
  newer families' op-specific CUDA gates or establish hardware WebGPU execution for operations
  currently delegated to WebAssembly. Small-IR browser speed remains a measured limitation
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
