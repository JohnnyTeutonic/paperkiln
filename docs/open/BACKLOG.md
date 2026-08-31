# Backlog — ordered

*What is outstanding, roughly in the order it should be done. The
long-arc vision lives in [`../../ROADMAP.md`](../../ROADMAP.md); this
file is the near-term queue. Anything finished moves to the CHANGELOG
and comes out of here.*

## 1. Finish the transfer study (the flagship)

Pre-registered at anchor `3fa55ae`. Order fixed: **bridge → S → M → L.**

- [ ] bridge gate verdict (in flight — see [`NOW.md`](NOW.md))
- [ ] arm S — d=256, 12 seeds, 6 lanes, 72 runs, ~6 T4-hours
- [ ] arm M — d=512, 12 seeds, 6 lanes, 72 runs, ~13 T4-hours
- [ ] arm L — d=1024, 3 seeds, 2 lanes, 6 runs, ~3.6 T4-hours
      (preliminary by design; carries no inference)
- [ ] run `analyze.py` exactly as committed, write `RESULTS.md`, bank
      receipts, add the registry row

**If the gate fails**, the study halts and the finding is: *the compute
backend changes the conclusion.* That slots into the existing arc — S1c
says budget changes the sign, S1e says seed changes the sign, this would
say venue does too. Three axes on which a single-number ablation claim
is under-specified. Write it up rather than treating it as a setback.

## 2. sparse_s1_longbudget — the theorem's falsifier

Pre-registered, ready, queued. 12000 steps, 10 seeds, both lanes,
~5 GPU-hours. Tests whether a SECOND crossing appears once the larger
class overfits — which would falsify assumption (iii) of
[`../../atlas/THEOREM_CROSSING.md`](../../atlas/THEOREM_CROSSING.md).
Condition and consequence are registered separately, so all four cells
of the outcome table are informative.

## 3. Extractor bugs — three, each with a named one-line fix

All in `papers/fetch.py`, all registered in
`papers/flavor_bench.py::KNOWN_WRONG` with diagnoses. The benchmark
gate still fails on any NEW wrong assertion, and reports a fixed entry
as FIXED so the list cannot rot into an excuse list.

- [ ] **Attributed adoption** (Megatron-LM). A paper stating its own
      flavor only as a property of its ancestors loses to a contrasted
      alternative in a bare declarative clause. Fix: inheritance
      outranks third-party attribution. Narrowed by the Qwen2 positive
      control — attribution *with* a first-person verb already works, so
      the failing case is syntactically well-defined.
- [ ] **Future-work mention** (Cerebras-GPT). "Worth exploring in future
      work" read as adoption. Fix: future-work mentions should VETO,
      exactly as explicit rejections already do. Second gap in the same
      paper: "X-like architecture" does not register as an inheritance
      cue.
- [ ] **Compound-name shadowing** (LaMDA). "gated-GELU" IS GeGLU; bare
      `GELU` matched inside it. Fix: longest-match-wins over the flavor
      lattice plus a `gated-X → XGLU` normalisation.

## 4. Extractor benchmark — grow past 40

40 papers, grouped AUROC 0.905, CI-band target met. Growth is
mechanical but slow by design: **every entry is read off the fetched
source, never recalled, and a field a paper does not state is omitted
rather than inferred.** Next candidates that scanned clean but were not
yet added: StarCoder, ELECTRA, UL2, BigBird, Chinchilla (several state
no flavors at all — correctly skipped).

## 5. CUDA — what is left

Phase B is complete and adopted (21× at d=256, 30.5× at d=512). Open:

- [ ] Wire `MT_DEVCHECK_HOST_READ` at actual call sites. The macro and
      `devcheck_host_read` exist; nothing calls them. A DEVCHECK build
      would turn a silent stale read into a loud assert.
- [ ] Coalesced loads in the transposed GEMM paths (stated non-goal
      during B2, correctness first — now fair game).

## 6. Housekeeping

- [ ] `paperkiln-fetch` and `paperkiln` are not yet claimed on PyPI. The
      wheel builds; only Jonathan can upload.
- [ ] `tools/sync_fetch_pkg.py --check` reports drift while
      `papers/fetch.py` has uncommitted changes. Expected, not a defect.
- [ ] The repo carries a large CRLF-only diff surface on Windows
      checkouts (~180 files show modified with zero real changes;
      `git diff --ignore-cr-at-eol` is empty). Cosmetic, but it makes
      `git status` useless at a glance. A `.gitattributes` pass would
      settle it.
