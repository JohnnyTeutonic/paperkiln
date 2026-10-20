# paperkiln roadmap

*The one list of what is left to do, as at 9 October 2026. Finished work
moves to [`CHANGELOG.md`](CHANGELOG.md) and comes out of here. Everything
else under `docs/` and `atlas/` is reference (design, records, results);
where one of those documents carries a plan, this file wins.*

## 1. Now

1. **The transfer paper is under submission at *Machine Learning*
   (Springer), submitted 9 Oct 2026.** Nothing to do until the editor
   replies.
   - **Study 3, width x rate crossed: done (11 Oct 2026). Reading MIXED.**
     H-POS supported at 5e-4 (S5 vs M5) but not at 2.5e-4 (S25 vs M25,
     0.967 through t4); H-RATE supported at width 256, not at 512.
     Descriptively, every declining pair includes M5 (d512 at 5e-4): the
     paper's "decay along the loss curve" account does not hold at a
     shared 2.5e-4. Results in `experiments/transfer_s3/RESULTS.md`;
     registry row T3-rate-specific-disagreement. Goes into any MLJ
     revision.
2. **Atlas Stages 2–3 with `rope_heads=all`: done (10 Oct 2026).** The
   two designs re-run with every head rotated (84 runs,
   `experiments/atlas_stage{2,3}_rope_all/`). None of the nine affected
   findings is overturned: the six machine checks replicate and the other
   three match on main effects. Registry notes record the numbers; statuses
   unchanged ([`docs/ROPE_HEADS.md`](docs/ROPE_HEADS.md)).
3. **sparse_s1_longbudget: done (10 Oct 2026).** All 20 runs banked; the
   frozen analysis returned "not supported / not supported": no second
   crossing through 12000 steps, the overfit-order condition fails (2/10),
   and the undetermined zone closes at b=4000. Results in
   `experiments/sparse_s1_longbudget/RESULTS.md`; registry row
   S1f-longbudget-monotone.

4. **Colab runner: a re-provisioned driver pushes stale checkpoints over
   finished runs** (`tools/colab_transfer_runner.py`). In Study 3 it
   overwrote four M25 event logs at step ~2900; the runs were re-run
   (execution clarification 1). Fix: never relay a checkpoint into a run
   directory that already has `result.json`.

## 2. Decisions waiting on Jonathan

None open. Settled on 16 Oct 2026: the post-training and corpus charter
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
- No separate methodology paper: its argument is in the transfer paper
  at Machine Learning. `atlas/PAPER_PLAN.md` stays as reference.

## 3. Broken now

Nothing known. CI is green on all three workflows (Code Quality, the test
suite with its Python suites and the fetcher drift check, and the wheel
build with its smoke test), and the chat-v1 release serves the quickstart's
ready-made model. One open choice: cppcheck and Valgrind still run as
advisory steps that cannot fail the build.

## 3b. Environment traps found 17 Oct 2026

- A clone placed directly in `~` builds against the stale `~/transformer_cpp`
  (CMake prefers a sibling to the vendored copy) and fails on missing headers.
  Put clean clones in their own parent directory, or add a CMake switch that
  forces `third_party/transformer_cpp`.
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
  - Fix the `MICROTORCH_DEFER_DOWNLOADS` crash for the gpt2 family (defect
    record D1; deferral works for llama runs and every production run
    uses it), and add an end-to-end mtstudio leg under deferral.
  - SFT evaluation is the slow part of a fine-tune: on the Colab L4 an
    `objective: sft` run spends ~10-20 minutes at each `eval_every`
    checkpoint with the process on the CPU (capacity session, 20 Oct 2026),
    while its training steps are fast. Put the SFT eval on the device path.
  - Wire `MT_DEVCHECK_HOST_READ` at real call sites; today it is defined
    and never called.
  - Coalesce loads in the transposed GEMM paths.
  - Re-measure the speedup on the op-set configuration over a full run
    before any paper quotes it. The 21x and 30.5x figures were measured
    over nine steps with deferral on.
- **Mixed precision** (fp16/bf16 on the tape).
- **Mamba:** a selective SSM (input-dependent A and B) and a parallel scan.
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
- **Serve open-weight HF models on our own engine**, and benchmark them
  against `transformers` for parity and speed.
- **WebGPU backend,** after CUDA, through the same dispatch seam.
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
