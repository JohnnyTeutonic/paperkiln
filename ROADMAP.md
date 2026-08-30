# paperkiln engineering roadmap

*Ordered backlog. Research (atlas/sparse attention) is the flagship;
these are the engineering items that unlock it.*

---

# UPLIFT PLAN — adopted 28 Aug 2026 (supersedes ordering below where they conflict)

1. **THE SCALE LADDER IS THE KEYSTONE.** The one reviewer objection that
   matters against Atlas is "does any of this hold past d=128/T=256/CPU?"
   The strong claim available: *do cheap designed screens predict
   expensive outcomes?* Same factor set at 3–4 rungs (d=128/256/512/768,
   token-matched), report rank correlation (Kendall's τ) of effect
   estimates across rungs. Factors that survive → screen architecture
   decisions at ~1/1000 compute and trust the ranking (industrial value).
   Factors that invert → the field's ablations are scale-sensitive and
   currently run wrong (arguably the better paper). **No failure branch;
   either outcome publishes.** The seed-noise finding stops being a
   curiosity and becomes the setup for this result.
   **LICENSED AND RUNNING 31 Aug 2026** — experiments/transfer_s1/
   PREREGISTRATION.md, licence anchor 3fa55ae, committed together with
   its analyze.py before any run existed. The realised design is
   stronger than this sketch: the transferred object is the COMPLETE
   PAIRWISE SIGN MATRIX over six lanes (15 edges) at three widths
   (256/512/1024), not a rank correlation of effect estimates — because
   S1e proved the scalars themselves are seed-distributions, so ranking
   them across arms is the weak read and the sign structure is the
   strong one. A numerics-bridge gate runs FIRST and halts the study if
   the CUDA venue disagrees with the banked CPU cohort. S1e also
   supplied the power numbers (12 seeds/arm) and killed one candidate
   position rule outright (overfit-onset: unreachable in 14/15 seeds).
2. **CUDA Phase B is the PREREQUISITE for the ladder, not
   infrastructure.** Resident device tensors, params uploaded once,
   activations on-device (B2.1b → B2.2 → B2.3 below). Without it the
   ladder cannot be climbed; prioritise it as the enabling step for the
   flagship research, ahead of everything else clamouring in the repo.
3. **DECOUPLE THE EXTRACTOR — highest-leverage day of work in the
   list.** papers/fetch.py has the widest possible audience and is
   trapped behind a C++ build most of that audience will never run.
   Ship it as a standalone pip package: `pip install paperkiln-fetch`;
   `paperfetch 2302.13971 --emit hf` → a HuggingFace config handed
   straight to `transformers`. hf_export.py already holds the layout
   knowledge; the change is mostly plumbing. This is the artifact that
   gets stars, citations, and drags readers back to the rest of the
   repo — the only piece not adoption-gated behind a build step.
4. **Extractor benchmark → proper D&B submission.** 26 papers is small.
   Grow to 60–100 with ground truth, publish the annotation protocol,
   and add a second task: architecture reconstruction fidelity
   (systematise the Primer 110M→114M check: reconstruct N papers,
   report parameter-count error, taxonomise failures). Target shape:
   "we reconstruct 47 of 60 papers to within 5% of reported parameters,
   and here is why the other 13 fail." Zero wrong assertions is the
   headline.
5. **MECHANISM FREEZE.** Kimi Linear, cerebellum gating, Mamba/S4, SRD,
   LoRA/QLoRA/int8: individually good, collectively they read as
   breadth, and breadth is what reviewers discount. **Nothing new goes
   in until the ladder is done.** SRD continues only via its
   pre-registered test (experiments/SRD_PREREG_R2.md); otherwise it stands as the
   honest partial negative it is. New mechanisms are still wanted —
   later, not now.

---

1. **CUDA past the dispatch seam.** Phase A, Phase B1, and now Phase
   **B2.0 (T4-validated 13 Aug 2026)**: step-residency plumbing —
   Variable-owned device state, transpose-flag GEMM, epoch-scoped
   caches; CUDA training pin matched CPU to 2.33e-07, staleness probe
   green (docs/CUDA_PHASE_B2.md). Remaining: **B2.1a
   T4-VALIDATED 21 Aug 2026** — full device op set (src/cuda_ops.cu) +
   attention transpose-kill, kernel parity <= 3.8e-06 worst / bitwise
   elementwise, gradcheck+nn green with ops live (receipts in docs/);
   then
   **B2.1b T4-VALIDATED 29 Aug 2026** (deferred downloads: value cache,
   materialize boundaries, defer-vs-writethrough EXACTLY 0.0 on the
   composed tape; receipts docs/receipts_b21b_t4_20260829.txt); next
   **B2.2 T4-VALIDATED 30 Aug 2026** (fused + swa masked attention
   softmax on-device — the biggest forced materialize gone — plus
   embedding gather and CE with the host receiving ONE float; 12/12
   suites, 281 checks; receipts docs/receipts_b22_t4_20260830.txt; on
   the way it surfaced and fixed the deferred-temporary-dies-stale
   heap-corruption class via the new device::discard() primitive); next
   **B2.3 T4-VALIDATED 30 Aug 2026** (persistent device optimizer
   state + device-side accumulate: the step's downloads are now the
   loss scalar and param grads at the boundary; 12/12 suites, 285
   checks; receipts docs/receipts_b23_t4_20260830.txt; the gate also
   caught+fixed dying-temporary corruption round two — see the phase
   doc's standing lifetime rule).
   **ADOPTION GATE PASSED 31 Aug 2026 — CUDA PHASE B IS COMPLETE.**
   B2 beats CPU AVX by **21x at d=256 and 30.5x at d=512** (T=512,
   L=4, T4) on an identically-converging computation (final losses
   match at print precision: 5.2485 and 4.9382). Rung C RUNS ON
   CUDA: a cell that cost ~6 CPU-hours costs ~12 minutes. Receipts
   docs/receipts_b2gate_t4_20260831.txt. The gate run also caught
   the deferred-gemm staleness bug the 285-check suite missed —
   the benchmark is now part of the correctness gate, not a
   postscript (see docs/CUDA_PHASE_B2.md).

   1a. **Deep SWA — DONE 12 Aug 2026 (same night it was discovered).**
   FlexLM takes attention=exact|swa with window/sinks at any depth;
   mtstudio promotes swa+depth to flex; taxonomy allows it. Gated by
   tests/test_deep_swa.cpp: the SWA PIN (FlexLM(swa, L=2) ==
   ParityLM(SWA) bitwise), depth-4 grads + FD, depth-4 batch/masking
   pin, swa+highway composition. The DEPTH AXIS of the scale ladder is
   now open: a depth rung (d=256, layers=4, exact vs swa) is runnable
   as its own pre-registration once width Rung B lands.
2. **Serve open-weights HF models through our own inference engine**
   (ember.cpp lineage). Flagged by Jonathan 11 Aug 2026 during the
   referee-project design (the 7-vendor panel currently uses
   `transformers` for local models). IMPORTANT: this is the item that
   makes paperkiln a self-contained lab — download weights via
   `referee/src/fetch_models.py`-style tooling, run them on our
   engine, benchmark against `transformers` for parity + speed.
   Depends partly on (1) for anything beyond ~7B on GPU.
3. LoRA + quantisation (existing backlog order).
4. arXiv LaTeX fetcher: **emit-spec landed 12 Aug 2026** —
   `papers/fetch.py <id> --emit-spec` turns a paper into a runnable
   mtsweep spec (paper-faithful or --house-dims; unresolved fields
   omitted loudly per the module contract). Proven live: 1706.03762
   fetched, extracted with evidence, and trained (flex family,
   layernorm/relu/sinusoidal). **Highway/SWA patterns landed 13 Aug
   2026**: residual + attention flavor fields (attention one-sided by
   design — only swa has a positive signature), window/sinks as aux
   numerics that never swell `unresolved`, Mistral as a named-swa
   ancestor, and emit_spec refuses windowless swa + scales degenerate
   windows loudly. Emitted specs pass mtsweep --dry-run — registry
   entries for both new mechanisms can auto-seed from papers.

Reference docs: atlas/ARCHITECTURE_ATLAS.md (the lab charter),
atlas/PAPER_PLAN.md (G1-G3 gaps), docs/SPARSE_ATTENTION.md (research
state). Scale ladder Rung B: experiments/sparse_s1_scale/.

---

# Long-arc programmes (adopted 12 Aug 2026)

*Origin: three proposals surfaced via Gemini, assessed and re-scoped.
Two are convergent re-derivations of plans already in this repo
(studio vision; Chimera) — adopted with the re-scopes below. These are
the between-times arc: the active fronts (G1 ladder, CUDA, deep-SWA,
the agentic month) always take precedence.*

## P1. ML Archeology Registry ("de-extinction")

A curated, reproducible registry of historical/forgotten
architectures: each entry = the paper's mechanism translated into the
spec grammar, a token-matched standardized run on the fixed corpus,
and an Atlas row (gradient behaviour, param efficiency, loss curve,
scope-labelled per the scale-ladder doctrine).
- RE-SCOPE: curated translation now; automated paper->spec compilation
  is the north star, not the entry ticket. Every entry grows the
  engine grammar, which also feeds P3.
- Depends on: spec grammar (atlas plan step 2), arXiv fetcher (item 4
  above); engine features per family as needed.
- **PILOT (#0): Highway Networks (arXiv 1505.00387)** — small,
  pre-residual gated depth; crisp mechanism (transform/carry gates);
  bounded engine addition; the verdict-at-tiny-scale is genuinely
  interesting (does gating beat residuals when both are tiny?).
  Candidate shortlist after the pilot: Grid LSTM, early relative-
  position attention variants, gMLP, RWKV-v4 block, retention.
- Claim discipline: every registry verdict is scoped to protocol +
  scale; trends up the ladder, never absolutes about the paper.

## P2. Browser + accelerator targets (split from "paper-to-silicon")

- P2a **WASM runtime — SPIKE PROVEN 12 Aug 2026** (ember.cpp/WASM.md):
  the inference core compiled to wasm UNMODIFIED in one em++ command
  and ran coherent GGUF chat inference under node. Remaining for the
  browser page: MEMFS model loading + a small JS API (a day, when demo
  value calls — e.g. a certain Melbourne inference lab).
- P2b **WebGPU backend**: bounded kernel-backend project, AFTER CUDA
  (item 1) — same dispatch seam, second target.
- P2c Silicon/Verilog: DECLINED as a goal (HLS is its own field).
  One-line note kept for the record; revisit only if a collaborator
  with hardware chops appears.

## P3. Chimera: autonomous designed-experiment search (existing plan)

Gemini's "evolutionary search + self-writing papers" = the Chimera
design doc, independently re-derived. Mutation proposals over the
typed grammar, designed experiments via mtsweep, early-kill on
anomaly, rediscovery gauntlet as the validation gate.
- RE-SCOPE (firm): outputs are findings-registry rows and
  auto-generated results artifacts under pre-registration discipline.
  NO autonomously written papers — verification precedes prose,
  always (this repo's own MBS history is the cautionary tale).
- Substrate = atlas plan steps 2-5 + P1's grammar growth; the
  LangGraph agentic month builds the orchestration skill.
- Tier levers (existing memory): falsifier discovery is the preferred
  result that lifts this to JMLR/JAIR class.
