# The paperkiln ecosystem — coherence plan

*2026-08-01. The three-repo family after the rename: **paperkiln** (this
repo — tape autograd, studio, Atlas), **coalfire.cpp** (the original
hard-coded C++/CUDA trainer), **ember.cpp** (inference). This document
answers: what makes them ONE ecosystem instead of three projects, what
gets built next, and how the sparse-attention research phase sits on
top.*

## 1. Honest roles (stop pretending they compete)

| Engine | Is | Is not |
|---|---|---|
| **paperkiln** | The research studio: any architecture the taxonomy can express, every gradient FD-checked, every run an Atlas row. Flexibility and receipts. | The fastest trainer (tape overhead, CPU-first, fp32-only today) |
| **coalfire.cpp** | The **kernel authority and scale trainer**: the hand-written CUDA corpus (pre-LLM, learned off the manuals — stride layouts, cuBLAS conventions, fp16 mixed precision, the alignment lessons) that paperkiln already links as `transformer_core`, plus the mature hard-coded training loop that burns vanilla architectures fast. | Flexible — it cannot train kimi/srd/attnres/flex, and should not learn to |
| **ember.cpp** | The serving half: GGUF + safetensors, quantized types, HTTP `/chat`, web UI. | A training system |

### 1.1 coalfire's genuine roles (not a legacy shrine — load-bearing)

The build system already tells the truth: paperkiln's CMake links
`transformer_core` as a static library, `MICROTORCH_CUDA` routes every
`device::matmul` through coalfire's kernels, and `tools/sync_vendor.sh`
exists to keep the vendored copy fresh. Four roles, all structural:

1. **Kernel authority.** New CUDA work — phase B resident tensors, fp16
   on the tape, and eventually sparse kernels — lands in coalfire's
   kernel tree FIRST and vendor-syncs into paperkiln. The flagship's
   GPU engine is, permanently, the corpus that was debugged by hand
   before code assistants existed. That provenance is also a
   portfolio differentiator no green-field repo can claim.
2. **Scale trainer.** When paperkiln graduates a vanilla-shaped
   architecture, coalfire trains it at CUDA/fp16 speed (the C4
   translator + C1 adapter make this a button, not a story).
3. **Independent witness.** The C2 golden pin is symmetric: two
   independently written trainers agreeing on logits is a verification
   asset neither has alone. coalfire checks paperkiln as much as the
   reverse — the same discipline that made ember's argmax-parity
   serving trustworthy.
4. **Sparse kernel forge.** The sparse-attention phase (§5) needs
   fast kernels, not just taped reference implementations: research is
   *designed* in paperkiln (taxonomy, Atlas cells, falsifiers) and its
   winning kernels are *forged* in coalfire — sliding-window CUDA
   belongs there. Work in this direction already exists in the tree.

The story that makes this coherent: **search and verify small in the
kiln, burn the winner at scale in the coalfire, keep it glowing in
ember.** An architecture graduates from paperkiln (where variants are
cheap and receipted) to coalfire (where the blessed shape trains fast)
to ember (where it serves). Today only the kiln→ember edge is real;
the kiln→coalfire edge is aspirational until the contracts below exist.

## 2. The four contracts (the actual ecosystem spine)

Coherence is not shared code — it is shared **contracts**. Four, all
already born in paperkiln, to be spoken by all three engines:

1. **`spec.json`** — the run description (arch/data/train/export). One
   schema, versioned. Today: mtstudio only.
2. **`events.jsonl`** — the run telemetry (`start/step/eval/export/done`).
   Anything that emits it gets the studio dashboard, the Atlas
   extractor, and mtsweep aggregation FOR FREE. Today: mtstudio only.
3. **Artifacts** — safetensors + GGUF with embedded vocab, byte-exact,
   one blessed writer. Today: paperkiln's `gguf.hpp` (alignment-audited)
   and coalfire's own exporter coexist — a bug written twice is a bug
   shipped twice.
4. **Atlas rows** — the structural echo + behavioural metrics per run.
   Today: computed from events.jsonl, so it follows contract 2.

## 3. Coherence workstreams, ranked by value/effort

**C1. Teach coalfire to emit `events.jsonl`.** ~an afternoon: one
logging shim in its training loop. Payoff is outsized: the studio
dashboard tails coalfire runs, coalfire runs become Atlas rows, and
mtsweep can sweep coalfire configs. One dashboard, two engines — the
single cheapest unification available.

**C2. The cross-engine golden pin.** Load the same tiny checkpoint into
paperkiln's GPT-2 and coalfire's transformer; assert logit parity on a
fixed input. This is the receipt that the two trainers are *the same
mathematics* — the ecosystem's equivalent of the Block(S=1)==Full pin.
Any kernel drift between engines becomes a test failure instead of a
mystery. (ember already has this discipline via argmax-parity serving.)

**C3. One GGUF writer.** coalfire adopts paperkiln's `gguf.hpp`
(vendored the same way paperkiln vendors coalfire's CUDA kernels — the
vendor flow already exists, run it in reverse). The 32-byte alignment
bug class gets exactly one home.

**C4. spec → coalfire config translator.** A ~100-line script mapping
the mtstudio spec's vanilla-transformer subset onto coalfire's config
JSON. Combined with C1, the studio's ▶ TRAIN button could offer an
"engine: kiln | coalfire" dropdown for vanilla architectures — the
graduation edge made real.

**C5. Shared tokenizer.** Both trainers use word-level-from-GGUF-vocab;
ember parses real BPE tokenizers already. Porting ember's BPE reader
into the training side upgrades both trainers' data pipeline at once
and closes the biggest realism gap vs the small-LM literature.

**Non-goal:** porting flex/kimi/srd/attnres into coalfire — flexibility
lives on the tape. But the converse is NOT a retirement plan: CUDA
phase B and fp16 in paperkiln are built ON coalfire's kernels (§1.1
role 1), so paperkiln getting faster makes coalfire MORE load-bearing,
not less. The division of labor is permanent: coalfire owns the
kernels and the scale burns; paperkiln owns the architectures and the
receipts.

## 4. Feature candidates (paperkiln-side), ranked

1. **CUDA phase B: resident device tensors.** Params uploaded once,
   activations on-device — built on coalfire's kernel corpus (new
   kernels land there first, per §1.1). THE unlock: makes fp16/bf16
   worth building (the deferred STUDIO_PLAN item; coalfire's
   half_precision_kernels.cu is waiting) and turns the T4-validated
   seam into real speed.
2. **In-studio sampler.** `mtstudio sample ckpt --prompt ...` + a
   generate box in the dashboard fed by a `sample` event. Closes the
   train→poke loop without leaving the page (ember stays the real
   server; this is the quick-look).
3. **Eval-probe stage in the spec.** Beyond held-out loss: needle
   retrieval (the SRD harness already exists) and a couple of tiny
   behavioural probes, emitted as events → the Atlas behavioural
   vector grows real task axes, which the sparse-attention phase needs
   as its measuring stick.
4. **HF-format export.** `config.json` + tokenizer files next to the
   safetensors so `transformers.from_pretrained` opens a paperkiln
   model directly. Small effort, large interop credibility.
5. **Studio Research Mode.** "Clone run with one knob changed" button +
   two-run comparison view (loss curves overlaid, grad-glow diff).
   This is ARCHITECTURE_ATLAS section 15's ablation UX made concrete.
6. **Sweep heatmap.** mtsweep cells rendered as a d×lr (etc.) grid in
   the studio — Stage-2/3 style results readable at a glance.
7. int4/NF4, Mamba parallel scan, fetcher GQA/MoE fields — as already
   roadmapped.

## 5. The sparse-attention phase (the flagship), Atlas-native

The aspiration only works if it is run as science, not as a pile of
implementations. The Atlas is the instrument that makes it so:

- **Phase S0 — taxonomy.** Survey and classify: windowed/local
  (sliding, dilated), global-token hybrids (Longformer/BigBird),
  content-based routing (Reformer LSH, Routing Transformer), kernel/
  linear approximations (Performer; Kimi linear already in-repo),
  KV-eviction at inference (H2O, StreamingLLM sinks), and the
  learned-density lineage SRD belongs to. Each becomes a row in the
  taxonomy's attention lattice with its compatibility constraints.
- **Phase S1 — verified baselines.** Implement 2–3 cheap, honest
  baselines as spec-expressible attention alternatives: sliding-window
  (+attention-sink) first — it is simple, strong, and the field's
  default control — then one routing/block-sparse variant. Same
  receipts as everything else: FD gradchecks, batch pins, equivalence
  pins where exact cases exist (window ≥ T must equal full attention).
  Reference implementations live on the tape; their FAST kernels are
  forged in coalfire (§1.1 role 4) and vendor-synced back.
- **Phase S2 — the measuring stick.** A fixed evaluation cell: LM loss
  + needle retrieval + long-context degradation, ≥3 seeds, run through
  mtsweep. Every attention variant faces the same cell. The SRD
  retraction taught the protocol: single-seed results are
  uninformative; falsifiers ship with mechanisms.
- **Phase S3 — original variants.** Only now: SRD's successor ideas and
  new mechanisms, each entering the same cell against the S1 baselines,
  negatives published. The Atlas fingerprint work (which behaviours a
  variant trades away) is the novel analytical angle nobody else's
  "we beat full attention" tables have.

## 6. Suggested order

C1 → C2 (coherence receipts, days) → feature 2 (sampler, the demo gets
its last missing beat) → CUDA phase B (the long pole, start early) →
C3/C4/C5 alongside → features 3–6 as the Atlas needs them → sparse
phase S0 once Stage 3/4 of the Atlas programme is humming. Everything
above stays true to the house rule: receipts first, negatives
published, no claim without its pin.
