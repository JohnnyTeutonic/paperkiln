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
2. **Re-run Atlas Stages 2–3 with `rope_heads=all`.** Nine findings carry
   an UNDER REVIEW note (record: [`docs/ROPE_HEADS.md`](docs/ROPE_HEADS.md)).
   S2-heads-null and S3-ctx-null are directly suspect.
   - Run the corrected design with `tools/reproduce.py --rope-heads all`.
   - Then supersede or confirm each row.
   - Add the under-review status to `atlas/FINDINGS.md` and to the
     Stage 2 and Stage 3 write-ups, which do not show it yet.
3. **sparse_s1_longbudget, due 20 Oct 2026.** It is pre-registered and
   parked, at 0 of 20 runs. As configured, 12000-step runs with
   `checkpoint_every: 1000000` cannot survive Colab's session cap.
   - The author writes an amendment that enables checkpoints and changes
     nothing else.
   - Then launch per `AGENTS.md`.
   - Budget from the CUDA measurements, not from the pre-registration's
     estimate (~16 min per run, ~5 GPU-hours in all). The measured pace at
     this shape is about 0.8 s/step, both for one cell on a T4 and for
     four concurrent cells on an L4. That makes each 12000-step run about
     2.7 hours, and the whole study five waves of four runs, roughly 14
     hours of sessions relayed through checkpoints.

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

- **The chat quickstart's default route returns 404 until the `chat-v1`
  release exists.** The bundle is built and verified
  (`~/release/tinychat-v2-web.tar.gz` in WSL: llama family, RoPE on every
  head; the quickstart's default route runs end to end against it). It
  needs publishing as release `chat-v1` on GitHub, which is the author's
  call.
- **Pending a push.** The fixes below are committed locally and verified
  locally; CI confirms them only once pushed: clang-format clean; the wheel
  builds and passes its smoke test (binding argument fix, position-independent
  libraries, install rule, distribution `paperkiln` 0.3.0); the
  documentation check points at `ROADMAP.md`; the Python suites and the
  `paperkiln-fetch` drift check run in the test workflow; `test_highway` and
  `test_deep_swa` run under ctest (22 suites).
- **Advisory only.** cppcheck (`|| true`) and Valgrind
  (`continue-on-error`) cannot fail the build. Whether to make them gate is
  the author's call.

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
