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

## 2. Decisions waiting on Jonathan

None open. Settled on 10 Oct 2026:
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

## 4. Research

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
