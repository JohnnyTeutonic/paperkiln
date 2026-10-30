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
   extension's two 60M post summaries (20M seeds 4/5 and all five 150M runs are home), Stage L0 on the registered 150M
   pick, and Carry-12's registered HS1 extension (B/S-gist seeds 4/5, training under the watchdog). The complete
   three-seed Carry-12 core is analysed in `experiments/tiny_chat_long_2026-10/RESULTS.md`:
   HT-SFT SUPPORTED, other primary tests unresolved; S-full does not trigger. The size extension
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
  - Finish the host-read boundary audit in author-WIP ops.cpp/mtstudio/parity_model
    and other unreviewed host loops. The audited non-WIP boundaries are wired and
    pass actual L4 CUDA+DEVCHECK integration (`docs/receipts/devcheck_host_read_cuda_20261030/`);
    those completed guards leave this list.
  - Coalesce loads in the transposed GEMM paths.
  - Re-measure the speedup on the op-set configuration over a full run
    before any paper quotes it. The 21x and 30.5x figures were measured
    over nine steps with deferral on.
- **BF16/flash per-shape throughput profile.** The three-seed noise-floor gate is complete
  (25 Oct): flash, bf16 and bf16+flash pass the pre-stated primary and curve rules; 60M is
  one-seed descriptive evidence. The full per-shape profile remains open: its original campaign
  did not finish. Historical loss arithmetic and third-party CUDA completion were rechecked
  on 30 Oct (`docs/receipts/roadmap_device_reconciliation_20261030.json`); do not rerun those
  completed gates. BF16 inside attention products is conditional on attention showing up in
  the profile (launch count is already closed, LOCAL_LLM_STACK.md section 10, 28 Oct).
- **Mamba:** a parallel scan and device kernels for Mamba-1's `ops::conv1d_causal` and
  `ops::selective_scan` (sequential, host only; the mamba preset's `ssm: "mamba1"`).
- **int4/NF4 quantisation.**
- **Kimi linear attention:** the non-causal backward (`src/ops.cpp:658`
  throws).
- **HF export for modified flex flavours and unmapped GPT2 variants; quantised GGUFs outside the llama family.** Strictly plain GPT2-flavour and untied Llama-flavour flex now have verified HF mappings (arbitrary depth/MLP width); those completed subsets leave this list. Tied-head flex remains unsupported until its C++ source lands. Every mtstudio family now exports F32 GGUF with its IR
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

- **Studio features:**
  - a fit-to-VRAM budgeter;
  - an evaluation-probe stage in the spec (needle and behavioural
    probes).
- **Open-weight models on our own engines: speed.** Parity is done through the IR,
  on the CPU and on CUDA. CPU comparison is complete for SmolLM2-135M and Gemma3-270M
  at three lengths (strict full-row parity); paired AVX2 improves prefill 2.38-2.87x
  and decode 1.04-1.50x but remains slower than Transformers. Evidence:
  `../ember.cpp/experiments/hf_cpu_2026-10-30/`. Remaining: matched CUDA speed comparison.
  A separate synchronized harness is prepared (`../ember.cpp/experiments/hf_cuda_2026-10-30/`),
  with 13 offline contracts and an actual CPU-only fallback-refusal check. No CUDA timing result yet.
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
