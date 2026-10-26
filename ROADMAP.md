# paperkiln roadmap

*The one list of what is left to do, as at 24 October 2026. Finished work
moves to [`CHANGELOG.md`](CHANGELOG.md) and comes out of here. Everything
else under `docs/` and `atlas/` is reference (design, records, results);
where one of those documents carries a plan, this file wins.*

## 1. Now

0. **Tiny chat from scratch (LOCAL_LLM_STACK.md section 11, item 1;
   `experiments/tiny_chat_2026-10/`, `experiments/tiny_chat_long_2026-10/`).**
   E5, E1 and the recursion arm run on Colab; E1 at 60M waits for an L4 (first use of the
   RunPod backend for `tools/colab_chat/runner_multi.py`, under validation); Carry-12 and its
   Stage L0 follow E5 and E1.
   - Arm X5 (amendment 5): HX5-main EQUIVALENT (`experiments/tiny_chat_2026-10/x5/RESULTS.md`). Left: Stage L0
     on `pick_e1_x5_X5` (13 of 91 chunks; it resumes; behind the main session's CPU queue).
1. **The chat model (LOCAL_LLM_STACK.md sections 8 and 11, item 2).**
   - Jonathan's pick between the winner rule's two verdicts (`experiments/chat_general/RESULTS.md`,
     "The winner": live as registered, pk_sft_v1 under amendment A1).
   - The Claude-judge pilot on pk_sft_v1, prepared, launch on Jonathan's go: candidates on a RunPod
     L4 (`tools/colab_chat/pref_pilot_v1.json`), then `experiments/chat_general/judge_pilot.sh`
     (200 prompts, $100 cap) at home; then the full ~3,000 prompts and DPO + NLL.
2. **Conversation schema follow-ups** (LOCAL_LLM_STACK.md section 3 point 3; section 11, item 3):
   instruction placement, measurable on a v2 model, and v2 (a reserved control-token block
   appended to the vocabulary) when Jonathan decides. His
   three decisions sit behind one constant each: v1's system mode (`SYSTEM_MODE_V1` /
   `kSystemModeV1`, fold or none), the default loss policy (`DEFAULT_SFT_TURNS` /
   `kDefaultSftTurns`), and when v2 is built.
3. **Engine:** BF16 mixed precision and a fused attention backward (opt-in; building).
4. **Colab runner: a re-provisioned driver pushes stale checkpoints over
   finished runs** (`tools/colab_transfer_runner.py`). In Study 3 it
   overwrote four M25 event logs at step ~2900; the runs were re-run
   (execution clarification 1). Fix: never relay a checkpoint into a run
   directory that already has `result.json`.

## 2. Decisions waiting on Jonathan

- The tiny-chat MoR block's nine runs (amendment 4, A4.7: his confirmation of the measured
  total) and R8 if A3.3's rule fires.
- Carry-12's MSC rulings (licence unstated; an episode as one conversation; ConvAI2 duplicates).

Settled on 25 Oct 2026 (the main session, under Jonathan's delegation): paperkiln's own translation
units are compiled with `-ffp-contract=off` (spec/IR_SPEC.md Q15; fc96e602).

Settled on 22 Oct 2026: up to four Colab sessions at once, one named session per job, each
driver under the Windows watchdog. Settled on 16 Oct 2026: the post-training and corpus charter
([`docs/decisions/0001-posttrain-charter.md`](docs/decisions/0001-posttrain-charter.md))
is approved, with the open calls delegated and recorded in its §12.

Settled on 10 Oct 2026:
- PyPI: `paperkiln-fetch` 0.1.0 and `paperkiln` 0.3.1 are published
  (`paperkiln`: portable x86-64-v3 Linux wheels for Python 3.10–3.12).
- Registry: Study 2 has its row (T2-structure-position-fragile), which
  supersedes T1-structure-transfers; T1-fixed-lr-mistunes stands. The
  highway pilot has its row (R0001-highway-depth2-null).
- Parked until the journal verdicts are in: the two synthesis sketches
  (consensus prefill, commit-gated TTT) and the designer pilot. Their
  code and plans are committed.
- The mechanism freeze is lifted; every new mechanism ships with a
  pre-registered falsifier (decision D10).
- No separate methodology paper: its argument was in the transfer paper,
  retired on 20 Oct 2026 after the Machine Learning desk rejection.
  `atlas/PAPER_PLAN.md` stays as reference.

## 3. Broken now

- **The benchmarks' HellaSwag scores** read 0.003-0.005 low in acc and 0.004-0.011 low in acc_norm: HellaSwag
  validation's `ind` is not unique (433 ids name two items), so `bench_v1.jsonl` stored the second item's
  results over the first's. `tools/bench_eval.py` is fixed (fafc4750: repeated ids become `id#n`, colliding
  keys refused); the published table is rescored by pk-bench2 (repinned to master). `mtstudio score` itself is
  correct on CUDA (`experiments/tiny_chat_2026-10/SCORE_AUDIT_2026-10-26.md`).
- **`mtstudio score` on CPU is slow for `llama_attnres`.** Stage L0 on pick_e1_x5_X5 (20m_L16 + Attention
  Residuals) scored about 4.5 min per chunk, against about 0.8 min for plain 20m_L16's pick on the same laptop
  build (`~/pk_state_build/mtstudio`, four slots; the process sat at ~210% CPU). Not diagnosed. On the L4, training
  runs at 0.67x plain.

CI is green on all three workflows (Code Quality, the test
suite with its Python suites and the fetcher drift check, and the wheel
build with its smoke test), and the chat-v1 release serves the quickstart's
ready-made model. One open choice: cppcheck and Valgrind still run as
advisory steps that cannot fail the build.

## 3b. Environment traps found 17 Oct 2026

- WSL git reports ~238 files modified in the OneDrive checkout (CRLF), so
  `--require-clean` runs must come from an ext4 clone.

## 4. Research

- **Post-training and corpus provenance** (charter 0001, order of work §11).
  Step 1 done (16 Oct 2026): `corpus/` registry, ledgers, pinned fetchers
  (`make corpus` rebuilds the research slice byte-exact), cards, and C0.
  Phase 1 done: `posttrain/verifier.hpp` (receipts mandatory; exact match,
  format, synthetic judge, cached LLM judge, human audit, Linux code sandbox)
  and `tools/posttrain/judge_sidecar.py` (round trip to C++ tested).
  Phase E done: `tinyarith` (levels R=10/20/50, splits disjoint by fact) with
  frozen dev/test/hack-probe sets, and frozen held-out narrative and dialogue text
  (`corpus/eval/`, pins enforced by tests). Probe: Common Crawl (FineWeb-Edu) and
  Simple Wikipedia lose 17-20% of tokens even to their own 4096-word vocabulary,
  so they are retired until there is a subword tokenizer. Next:
  Phases 2-3 done (17 Oct 2026): `mtstudio sample-set` (pure temperature,
  replayable), `rejection_sample.py`, `audit_sample.py`, masked SFT
  (`train.objective` sft / rejection_sft, `train.init_from`); the lm path is
  bit-identical to a banked Atlas receipt (`experiments/posttrain_bridge/`);
  pipeline smoke in `experiments/posttrain_smoke/`.
  Phases 4-5 done (17 Oct 2026): GRPO (`ops::policy_gradient_kl`; bandit lands on
  the analytic optimum to 2e-7) and provenance-weighted DPO (`ops::dpo_weighted`),
  both as mtstudio objectives with monitors and falsifier modes; how-to in
  `docs/POSTTRAIN.md`. Studies:
  - `posttrain_calib`: done, R = 20 (charter §13);
  - PT2 (filter provenance): done, null (charter §14);
  - PT4 (hack monitor, +1 channel): done, MONITOR_BLIND; PT4b (+2 channel): done,
    MONITOR_VALIDATED (charter §15–16);
  - PT5 (judge vs noise pairs; provenance weighting): done (18 Oct 2026),
    NOT_RESOLVED and WEIGHTING_INERT; the weight table is retired as the default
    (`train.po.weight_mode` is `uniform`). The fixed DPO recipe lowered truth in
    every arm, exact-ordered included (charter §17).
  - Byte-level BPE (`tools/bpe.py`, `bpe_tokenizer.hpp`, 8192-entry TinyChat vocab)
    and the chat corpus (TinyChat, ledger 0003) are in; GGUF exports of BPE models
    carry their merges.
  Next:
  - **TinyChat chat model: done (19 Oct 2026).** Base 12000 steps (val 1.989) +
    SFT, served by ember.cpp with BPE and 40/40 greedy parity
    (`experiments/chat_tinychat/RESULTS.md`). Chat improves with base steps; the
    simple-QA probe stays 0/50 on facts, science and arithmetic. SPIN preference
    pairs retired (repetition, collapse). SFT overfits after ~500 steps.
  - **General-text chat model** (running from 19 Oct 2026,
    `experiments/chat_general/PLAN.md`): FineWeb-Edu base d384 L6, ~195M tokens,
    47,500 steps, ~24 L4-hours (`tools/colab_chat/boot_fineweb.sh`, runner
    session pk-fw); then TinyChat SFT (~500 steps), then a Claude-judged
    preference stage (`tools/posttrain/chat_judge.py`, ~$25 per 1,000 prompts,
    needs Jonathan's OK), scored by `tools/chat_eval.py` and `tools/qa_eval.py`.
  - Engine, next: the remaining per-token host work (zero-filled Matrix
    construction on every op is the largest item in the profile); a VM-side
    snapshot before each checkpoint relay, so relays stop racing the saves.
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

- **The LLM extractor is the main path** (`papers/llm_extract.py`; on the 40-paper bench's cached replies,
  flavours 83 correct / 9 abstain / 0 wrong vs the rule-based 73 / 19 / 0; sizes 580 correct / 0 wrong / 62
  abstain vs the rule-based 136 / 8 / 498; block graph 91 / 2 wrong / 2 abstain; `papers/bench_score.py`
  against `papers/bench_truth_structure.json`). Open: re-extract the 40 under the 26 Oct prompt (its new
  fields: per-sublayer placement, q/k norm, Mamba-1 sizes, T5's bias, new-op input roles and inits; about
  $18); check every `expr` with `spec/tools/ir_expr.py` (the parser both engines use; landed in 1159d63e); a
  studio button that runs it instead of `fetch.py` when a key is configured; prompt optimisation later.
- **Chain gaps still refusing** (LOCAL_LLM_STACK.md section 11 item 4.2): Primer's MDHA (conv1d_causal on q, k, v:
  the op exists, flex does not wire it) and Shaw-style relative positions; T5's bucketed bias trains
  (flex `position: "t5_relative"`) but exports only once keramos has the op; a q/k norm on flex or over the
  full width; Mamba-1's inner LayerNorm (the paper's optional norm); MoE headline models.
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

- **Benchmarks** (`experiments/benchmarks_2026-10-26/`):
  - pk-bench2 (`tools/colab_chat/bench_v2.json`; root `scripts/colab_driver.json`, disabled until the RunPod
    top-up):
    - scores the picks that came home after pk-bench (pick_e1_L4, pick_e1_60m_L32, pick_e5_s0);
    - runs ember 580e108's POST /score on the full sets for pk_sft_v1, live and those picks, for
      `tools/bench_eval.py compare` at home.
  - Batch `mtstudio score`: it scores one request at a time on the host (~75 ms each for the 14.6M live model on
    an L4).
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
  (`tools/step_profile.py`): launch-count work (CUDA graphs or fused elementwise ops) if the idle share
  dominates; BF16 inside the attention products only if attention shows up.
- **The fused LM head + cross-entropy** (`train.fused_head`, off by default; CPU-gated 26 Oct 2026, CHANGELOG):
  the device legs, queued until the RunPod top-up (root `scripts/colab_driver.json` entry pk-fused-head,
  disabled, $3 cap; `tools/colab_chat/fused_head_cuda_v1.json`: the CUDA gate with `tests/test_fused_head.cpp`
  leg 9, the gate on 20m_L4 against c6365b88, and the throughput against fp32 at 20m_L4, 20m_L16, 60m_L32 and
  pk_base_v1).
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

- **Publish the default chat model** (see Broken now) and put it on the
  Hugging Face Hub with `tools/publish_hf.py`.
- **More `paperkiln` wheels:** macOS and Windows builds (Linux x86-64
  only today).
- **Studio features:**
  - Research Mode: clone a run with one change, compare two runs;
  - a sweep heatmap;
  - a fit-to-VRAM budgeter;
  - an evaluation-probe stage in the spec (needle and behavioural
    probes).
- **Browser chat** for SWA, Kimi, SRD and attnres models. It supports
  exact attention only.
- **Open-weight models on our own engines: speed.** Parity is done through the IR,
  on the CPU and on CUDA; benchmark ember against `transformers` for speed.
- **WebGPU for the IR families** (ember's browser chat runs llama and MoE GGUFs on
  WebGPU).
- **Windows testing of the quickstart** (parked).

## 8. Ecosystem (coalfire.cpp, ember.cpp)

*Reference: [`docs/ECOSYSTEM.md`](docs/ECOSYSTEM.md).*

- **C2, the cross-engine logit-parity pin:** no receipt exists.
- **C3, a single GGUF writer** (coalfire side).
- **C5, a shared BPE tokenizer.** Training is still word-level.
- **Technique transfer still open** (`docs/TECH_TRANSFER.md`):
  - SiTU-GLU and quantile balancing;
  - gated MLA/NoPE;
  - MXFP4;
  - mHC;
  - multi-token prediction.
  - The KDA reference that `python/attn_res_reference.py` mentions
    (`kda_reference.py`) is not in this repo.
