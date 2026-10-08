# paperkiln roadmap

*The one list of what is left to do, as at 9 October 2026. Finished work
moves to [`CHANGELOG.md`](CHANGELOG.md) and comes out of here. Everything
else under `docs/` and `atlas/` is reference (design, records, results);
where one of those documents carries a plan, this file wins.*

## 1. Now

1. **The transfer paper needs a journal.** The manuscript and supplement
   live in the root repository at `AI_ML/transfer_prereg/`. TMLR rejected
   it without review on 8 Oct 2026, and TMLR is now closed as a venue. No
   new experiments are needed. Remaining:
   - Pick the journal from its recent issues, not from memory.
   - Strip TMLR style, its anonymity conventions, and the "TMLR's own
     stated criterion" sentence (`main.tex` line 71).
   - Cut the abstract (295 words) to the venue's limit.
   - Run the Codex `hostile-referee` and `claims-vs-evidence` reviews, save
     them in a `reviews/` folder, and answer every major point.
   - Re-run the gate suite, and rebuild and verify the supplement.
   - Commit the untracked inputs the supplement cites:
     - `experiments/transfer_s2/posthoc/bootstrap_partial.py` in this repo;
     - the `transfer_prereg` scripts and records in the root repo
       (`reachability_report.py`, `reproduce.py`, `verify_supplement.py`,
       `SUBMISSION_CHECKLIST`, `PACKAGE_CHECK`, `DATA_AND_BUILD.md`,
       `REGISTRATION_TIMELINE.json`).
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

## 3. Broken now (fix before anything new)

- **The chat quickstart's default route returns 404.** `quickstart_chat.sh`
  downloads `releases/download/chat-v1/tinychat-v2-web.tar.gz`, but the
  only GitHub release is `v0.1.1`, with no assets. Publish the `chat-v1`
  release with the bundle that `tools/package_chat_model.sh` builds. The
  train-it-yourself route works.
- **CI.**
  - Code Quality fails on every run. The clang-format check fails, and
    `.github/workflows/code-quality.yml:80` requires `PHASE_3_SUMMARY.md`
    at the root, but the file lives in `docs/history/`. `release.yml`
    cites the same path and still builds `microtorch-*` tarballs.
  - Python Wheels fails. The distribution is still named `microtorch`, and
    the versions disagree (`pyproject.toml` 0.2.0; `setup.py` and
    `__init__` 0.3.0).
  - No Python test runs in CI. That includes `papers/test_fetch.py`,
    `tests/*.py`, `tools/test_*.py` and
    `experiments/atlas_selection/test_selection.py`.
  - `test_highway` and `test_deep_swa` are built but never registered with
    ctest (`CMakeLists.txt` lines 96–100).
  - cppcheck (`|| true`) and Valgrind (`continue-on-error`) cannot fail
    the build.
- **`paperkiln_fetch` has drifted from `papers/fetch.py`.** It lacks the
  6 Oct residual and attention patterns and `AUX_PATTERNS`. Re-sync with
  `tools/sync_fetch_pkg.py` and put `--check` in CI.
- **The studio's ▶ Train button ignores the browser's spec.**
  `tools/mtstudio.cpp` lines 1258–1270 train the spec armed at launch,
  not the one edited or extracted in the page.
- **README.** It disagrees with the code in about seventeen places,
  chiefly:
  - It calls CUDA Phase B "next" and says Stage 3 "is running".
  - It counts 16 test suites; there are 20.
  - It counts 19 registry claims; there are 22, nine of them under review.
  - It promises a ready-made model that is not published.
  - It says the extractor has "zero wrong assertions"; there are three
    documented ones.
  - It calls the SSM "Mamba", although it is not selective and has no
    parallel scan.
  - It says `paperkiln_fetch` vendors the fetcher verbatim.
  - It omits `web/`, `atlas/`, `registry/`, `experiments/`,
    `paperkiln_fetch/` and `tools/synthesis/` from the layout.
- **Smaller documentation fixes.**
  - `include/microtorch/mamba.hpp` (lines 3 and 12) claims a parallel scan
    that does not exist.
  - `docs/CHAT_WITH_A_PAPERKILN_MODEL.md` describes a pre-fix llama run
    and omits `--allow-legacy-rope`.
  - `specs/README.md` omits `tinychat-quick` and `tinychat-better`.
  - `web/chat/README.md` still calls tinychat-better gpt2-family.
  - `tools/coalfire_spec.py` calls itself C2; it is C4.
- **Line endings.** About 180 files show as modified on Windows checkouts
  with no real change. A `.gitattributes` pass would settle it; at present
  it covers only `*.sh`.

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
