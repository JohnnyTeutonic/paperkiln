# paperkiln engineering roadmap

*Ordered backlog. Research (atlas/sparse attention) is the flagship;
these are the engineering items that unlock it.*

1. **CUDA past the dispatch seam.** Phase A, Phase B1, and now Phase
   **B2.0 (T4-validated 13 Aug 2026)**: step-residency plumbing —
   Variable-owned device state, transpose-flag GEMM, epoch-scoped
   caches; CUDA training pin matched CPU to 2.33e-07, staleness probe
   green (docs/CUDA_PHASE_B2.md). Remaining: **B2.1a
   T4-VALIDATED 21 Aug 2026** — full device op set (src/cuda_ops.cu) +
   attention transpose-kill, kernel parity <= 3.8e-06 worst / bitwise
   elementwise, gradcheck+nn green with ops live (receipts in docs/);
   then
   B2.1b (deferred downloads + DEVCHECK), B2.2 (embedding + CE
   scalar-only), B2.3 (optimizer on device). Adoption gated on a d=512 wall-clock
   win vs AVX. That is what actually GATES Rung C (d=512, T=512).

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

Reference docs: ARCHITECTURE_ATLAS.md (the lab charter),
atlas/PAPER_PLAN.md (G1-G3 gaps), SPARSE_ATTENTION.md (research
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
