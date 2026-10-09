# paperkiln roadmap

*The one list of what is left to do, as at 9 October 2026. Finished work
moves to [`CHANGELOG.md`](CHANGELOG.md) and comes out of here. Everything
else under `docs/` and `atlas/` is reference (design, records, results);
where one of those documents carries a plan, this file wins.*

## 1. Now

1. **The transfer paper is under submission at *Machine Learning*
   (Springer), submitted 9 Oct 2026.** Nothing to do until the editor
   replies. The named follow-up is Study 3, a crossed width x learning-rate
   design (about 45 L4-hours), which is also the natural material for a
   revision.
2. **Atlas Stages 2–3 with `rope_heads=all`: running (started 10 Oct
   2026).** The two designs re-run unchanged except for RoPE on every head
   (`experiments/atlas_stage{2,3}_rope_all/sweep.json`; 84 runs, local WSL
   CPU, 3 jobs x 1 thread, about 18–19 hours; outputs in
   `~/atlas_rope_all/`, resumable with `~/atlas_rope_all.sh`). Nine findings
   carry an UNDER REVIEW note ([`docs/ROPE_HEADS.md`](docs/ROPE_HEADS.md));
   S2-heads-null and S3-ctx-null are directly suspect. When it finishes:
   - copy the rows and receipts into those experiment folders;
   - verdict each finding with `tools/reproduce.py <id> --check-only --rows
     <new atlas_rows.jsonl>`, then supersede or confirm each row;
   - add the under-review status to `atlas/FINDINGS.md` and to the Stage 2
     and Stage 3 write-ups.
3. **sparse_s1_longbudget: running (launched 10 Oct 2026, due 20 Oct).**
   Amendment 1 (pre-data, author-approved) set `checkpoint_every` to 400;
   nothing else changed. 20 runs on three L4 sessions (`lb0`–`lb2`, the
   sweep sharded by run index), drivers' logs in
   `/mnt/c/ml_artifacts/transfer/longbudget*_driver.log`; about 1 s/step,
   so about 3.3 hours per run and two waves. When all 20 are banked: copy
   receipts into `experiments/sparse_s1_longbudget/receipts/`, run the
   frozen `analyze.py`, write `RESULTS.md`, add the registry row.

## 2. Decisions waiting on Jonathan

- **PyPI names.** `paperkiln` and `paperkiln-fetch` are unclaimed (404 on
  9 Oct 2026). Claiming them needs his credentials and takes two minutes.
- **The longbudget checkpoint amendment** (Now, item 3).
- **The transfer_s1 registry rows after study 2.**
  T1-structure-transfers and T1-fixed-lr-mistunes still read "supported",
  although study 2's pre-registered falsifier fired. There are no study 2
  rows yet.
- **The highway pilot verdict.** It is recorded in
  `registry/0001_highway_networks/ENTRY.md` §7 and has no
  `findings.jsonl` row. Should it get one?
- **The two synthesis study sketches, go or no-go.**
  - `experiments/consensus_prefill/SKETCH.md` §7 lists four decisions
    (whether to run Phase A, the 4096-context model, the gate default, and
    the twelve-seed arm).
  - `experiments/commit_ttt/SKETCH.md` §6 lists what that sketch leaves
    open.
- **The designer pilot, go or no-go.**
  `experiments/atlas_designer_pilot/PILOT_PLAN.md` is a draft Colab pilot.
- **Commit the uncommitted research work, or not.** Nothing below is in
  git yet:
  - the experiment designer: `tools/atlas_designer*.py`,
    `tests/test_atlas_designer.py`, `atlas/EXPERIMENT_DESIGNER.md`,
    `atlas/designer_demo_v1/`;
  - `experiments/atlas_designer_pilot/`;
  - the post-hoc decision audit `experiments/atlas_selection/`;
  - the Jev judge evaluation and the maths ranking under
    `tools/synthesis/`.
- **The mechanism freeze of 28 Aug 2026.** It was tied to the scale
  ladder, which the transfer studies have now climbed. Lift it or keep it?
- **Whether the methodology paper stays a separate paper**
  (`atlas/PAPER_PLAN.md`, aimed at JMLR, not drafted). The transfer paper
  absorbed much of its argument, including gap G1.

## 3. Broken now

Nothing known. CI is green on all three workflows (Code Quality, the test
suite with its Python suites and the fetcher drift check, and the wheel
build with its smoke test), and the chat-v1 release serves the quickstart's
ready-made model. One open choice: cppcheck and Valgrind still run as
advisory steps that cannot fail the build.

## 4. Research

- **Scale ladder, depth rung.** d=256, 4 layers, exact against SWA, as its
  own pre-registration. Deep SWA is built and gated.
- **Atlas Stages 5–6.** Architectural fingerprints, neighbours, and the
  Atlas surface in the studio (`atlas/ARCHITECTURE_ATLAS.md` §19).
- **Methodology paper gaps** (if it stays separate): G2, an outside
  contributor; G4, the prior-art sweep on pre-registration in ML.
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

- **Fix the three registered wrong assertions** (`papers/flavor_bench.py`
  `KNOWN_WRONG`):
  - Megatron-LM, attributed adoption: inheritance should outrank
    third-party attribution.
  - Cerebras-GPT: future-work mentions should veto, and "X-like" should
    count as an inheritance cue.
  - LaMDA, compound-name shadowing: longest match should win, with a
    `gated-X → XGLU` normalisation.
- **Grow the benchmark from 40 to 60–100 papers**, with ground truth read
  off the fetched source and never recalled. Add a reconstruction-fidelity
  task (parameter-count error per reconstructed paper) for a datasets and
  benchmarks paper. Candidates next in line: StarCoder, ELECTRA, UL2,
  BigBird, Chinchilla.
- **Fetcher v2 remainder:** MoE fields and the HF-config cross-check. The
  per-variant fields and `n_kv_heads` are done.

## 6. Engine

- **CUDA** (records: `docs/CUDA_PHASE_B2.md`):
  - Fix the `MICROTORCH_DEFER_DOWNLOADS` crash in mtstudio (defect
    record D1), and add an end-to-end mtstudio leg under deferral.
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
- **GGUF and HF export for the flex and gpt2 families** (llama only today).
- **Kimi and SRD in mtstudio beyond two-block parity models.**
- **The taxonomy's "stream" lattice slot.**

## 7. Studio, chat and distribution

- **Publish the default chat model** (see Broken now) and put it on the
  Hugging Face Hub with `tools/publish_hf.py`.
- **`pip install paperkiln`:** publish the pybind11 wheel once the names
  are claimed and the wheel CI passes.
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
