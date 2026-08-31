> **SUPERSEDED 31 Aug 2026 by PREREGISTRATION.md in this directory.**
> Kept as the audit trail: it records which choices were OPEN and
> when each was closed. Every `[OPEN]` flag below is resolved there,
> with its reason — and two of them were settled by the seeds data
> rather than by preference (the overfit-onset position rule was
> dropped as unevaluable; the seed count came from measured SD).
> Nothing here licenses a claim; that document does.

# DRAFT pre-registration — transfer_s1: scalars don't transfer; does structure?

**STATUS: DRAFT — NOT LICENSED.** Nothing here licenses a directional
claim. The licence will be the future commit of `PREREGISTRATION.md`
in this directory, which happens only after (a) sparse_s1_seeds
RESULTS.md lands and its seed-variance numbers fill the power section,
(b) the open design choices flagged **[OPEN]** below are fixed by
Jonathan, and (c) the d=512 wall-clock gate (CUDA B2.2/B2.3) prices
the mid arm. Committed as a draft precisely so the final
pre-registration has an audit trail showing which choices were made
before ANY cross-scale data existed — no cross-scale run may start
before the licence commit.

## The question

The registry's own strongest results are the objection to the whole
programme: S1c-budget-reversal (the sign of exact-vs-swa depends on
where you stop), NEEDLE-scale-negative (a task family that fails to
resolve at 2 blocks), S3-d-unpaid (a width crossover above our scale),
and the seed lottery (single-seed signs contradict 48% of the time
past 2000 steps). If effects reverse with budget and vanish with
scale, why would any tiny-scale measurement inform a larger model?

This experiment turns that objection into a measured quantity:

**When width grows, is the STRUCTURE of the comparison — the sign
pattern over an intervention panel, the shape class of each Δ(budget)
trajectory, the location behavior of the crossover B\* — preserved,
even where the scalar rankings are not?**

Three findings are possible and ALL are bankable: (1) structure
transfers while scalars don't → tiny-scale fingerprinting is licensed,
scalars-don't-transfer-structure-does is the headline; (2) neither
transfers → tiny-scale ablation practice measures protocol noise, and
the registry that demonstrated it with receipts is the contribution;
(3) both transfer → tiny-scale screening is validated outright. The
only failure mode is a result too blurry to distinguish these — which
is a power problem, addressed pre-data in the power section.

## The three scales

| arm | d | params (≈) | seeds | role |
|-----|-----|-----------|-------|------|
| S (base) | 256 | 3.7M | 15 (BANKED: boundary {1..5} + seeds {6..15}) | reference fingerprint, zero new compute |
| M (mid) | 512 | ~13M | 10 paired, new | primary transfer contrast |
| L (upper) | 1024 | ~48M | 3 paired, new | preliminary trend point, claims marked as such |

Fixed across arms: gpt2-nano family, layers=2, heads scale with d
(d/32), T=256, batch 4, TinyStories slice + chat7b vocab (cap 4096),
steps 3600-equivalent at S **[OPEN: budget scaling rule — same steps,
same tokens, or same estimated FLOPs across arms; the matched-position
rule below reduces but does not remove this choice's bite]**. No 7B
claims anywhere: the claim is the measured TREND over a 13× parameter
range, with extrapolation explicitly labeled extrapolation.

## The intervention panel

The fingerprint needs a panel, not one comparison. Candidate panel,
all with banked S-scale precedent in the registry:

- **P1** exact vs swa(w=64, sinks=1) — the flagship pair (boundary+seeds).
- **P2** window dose within swa: w=32 vs w=64 vs w=128 (sparse_s1_window
  precedent).
- **P3** sinks within swa(w=64): sinks=0 vs sinks=1 (sparse_s1_sinks
  precedent).

Each pairwise comparison contributes one signed edge per budget slice;
the **fingerprint of an arm** = the per-seed joint sign vector over
the panel's edges at the pre-committed budget slices, i.e. a
DISTRIBUTION over sign vectors (the seed lottery forbids treating a
single seed's vector as the arm's fingerprint). **[OPEN: freeze the
exact edge list and slice grid at licence time.]**

## The fingerprint layers (pre-committed order of claim strength)

**F1 — sign-pattern concordance (primary).** For each edge and slice,
the S-arm seed-majority sign vs the M-arm seed-majority sign.
Concordance = fraction of (edge, slice) cells agreeing. Inference by
seed bootstrap **[OPEN: exact resampling scheme and the concordance
threshold that counts as "transfers" — to be fixed with real variance
numbers from sparse_s1_seeds]**.

**F2 — shape-class concordance (secondary).** Each seed's Δ(budget)
trajectory per edge classified by a fixed rule into
{monotone−, monotone+, single-crossing, flat} **[OPEN: the
classification rule, including the flatness band, fixed at licence
time against banked S-arm receipts only]**. Concordance = agreement of
modal class across arms per edge.

**F3 — B\* location behavior (the transfer law; exploratory unless
sparse_s1_seeds resolves B\*(256)).** If the crossing exists at both
scales: does B\*(512) sit at larger budget than B\*(256) (the
S3-d-unpaid "width raises the crossover" direction)? Reported as a
direction with per-seed dispersion, not a fitted law, at n=2 widths.
The L arm adds a third point at preliminary weight.

**H-SCALAR (the foil, committed now):** Spearman rank correlation of
scalar Δ(final) across arms over the panel's edges. The programme's
framing PREDICTS this is weak or sign-unstable; if it is strong, that
is finding (3) above, reported as such. The headline
"scalars don't transfer, structure does" is licensed ONLY if F1 clears
its threshold AND H-SCALAR fails its **[OPEN]** threshold — both
halves measured, neither assumed.

## The matched-position rule (design invariant, not open)

Cross-scale comparisons are made at matched RELATIVE positions, never
matched absolute steps — S1c-budget-reversal makes absolute-step
matching a known confound. Candidate operationalizations, one to be
fixed at licence time **[OPEN — decide before any M-arm run]**:

- (a) budget as fraction of the arm's overfit onset (best_val step,
  the boundary regime-check object), per seed;
- (b) budget at matched val-loss milestones per lane-pair;
- (c) budget as fraction of total steps (weakest; kept only as the
  robustness appendix, never the primary).

Whichever is chosen, the OTHER two are reported descriptively as
robustness — divergence between them is itself reportable.

## Power (inputs now REAL — from sparse_s1_seeds RESULTS.md, 30 Aug 2026)

Measured S-arm variance at d=256, n=15 paired seeds (S1e-bstar-
distribution): SD(Δ1200) = 0.0214, SD(Δ3600) = 0.0377 (SD roughly
doubles over the budget range); sign split at 3600 = 9+/6−; per-seed
persistent-crossing budgets span 1600→never (6/15 never cross by
3600); shrink effect +0.0541 with per-seed SD ≈ 0.0414 (t=4.13 at
n=10).

Consequences already forced on the design:
- **F3 must be distribution-to-distribution**: a point B\* comparison
  is refuted at the base scale itself. F3 compares the b0_s
  DISTRIBUTIONS (including the never-crossed mass) across arms.
- **The shrink, not the sign, is the powered primary within-arm
  object**: at S-scale the shrink resolves at t=4.13 with n=10 while
  the sign fails at n=15. F1's edge signs at late budgets are
  therefore expected to be seed-lottery-dominated at M too — the
  concordance target must be defined over the objects that resolve
  (early-budget signs, shrink directions, shape classes), and the
  late-budget sign cells reported descriptively. **[OPEN: final cell
  list, chosen on exactly this criterion at licence time.]**
- Sizing: detecting a shrink of the S-scale magnitude (+0.054, SD
  0.041) at one-tailed α=.05, power .80 needs n ≈ 9; power .90 needs
  n ≈ 12. **Draft recommendation: M arm at 12 paired seeds, not 10.**
  If M-scale SD comes in larger (the S-scale trend says dispersion
  grows), the bridge cells (Threat 1) provide the first M-variance
  estimate BEFORE the panel launches — a go/grow gate at licence
  time, fixed pre-data, never patched post-hoc.

## Threat checks (fixed in draft, tightened at licence)

1. **Numerics bridge.** The S arm ran CPU; M/L will run CUDA
   (B2.1b-validated stack). Before any M-arm cell: 3 bridge seeds of
   the S-arm flagship cells on the CUDA path; per-seed Δ trajectories
   must be consistent with the CPU-banked cohort (protocol-equivalence
   check in the style of boundary Threat 3, 5/5 precedent). Bridge
   fails → the transfer study halts and the numerics discrepancy is
   the finding.
2. **LR/width confound.** Fixed lr=0.001 across widths confounds
   width with effective step size (the muP objection). **[OPEN:
   commit to fixed-lr-as-protocol-property (matching every banked S
   receipt) with the confound stated in scope, OR add a matched
   lr-swept mini-grid at M. Draft recommendation: fixed-lr primary +
   a 3-seed lr=0.0005 sensitivity cell at M, reported descriptively.]**
3. **Protocol-drift guard.** M-arm exact-lane loss curves must land in
   the qualitative regime of the S arm (still-training at final
   budget in ≥ 8/10 seeds), else arms are not comparable and no
   transfer claim is licensed.
4. **Refuse-to-run guard.** Every run's model event must record its
   d, layers, heads, seed, and lane fields; lanes read from model
   events only; every receipt must pass
   `tools/validate_events.py` (docs/EVENTS_SPEC.md) before analysis.
5. **No peeking.** analyze.py is committed WITH the licence commit,
   before any M/L run exists. The S-arm fingerprint is computed from
   already-banked receipts and frozen into the licence commit itself.

## What this cannot show

One architecture family, one corpus, layers=2, one house protocol,
widths 256→1024 (13× params, far from 7B). A transfer result here is a
property of THIS protocol's fingerprints across THIS width range. What
it CAN show — for the first time with seed distributions and
pre-registration on both sides of the scale gap — is whether the
things tiny-scale labs actually publish (signs, shapes, crossovers)
are the kind of quantity that survives a width change at all.

## Compute price (the CUDA gate's whole purpose — measured 31 Aug 2026)

CUDA Phase B is complete and adopted: B2 beats CPU AVX **30.5x at
d=512** (21x at d=256) on an identically-converging computation
(docs/receipts/receipts_b2gate_t4_20260831.txt). That measurement is at
T=512/L=4; this study's arms run T=256/L=2, so the per-step cost is
roughly a quarter of it (attention is quadratic in T, depth halves the
rest). ESTIMATE, not a measurement — the M arm's true ms/step gets
measured in the bridge cells (Threat 1) before the panel launches:

| arm | est. ms/step | 3600 steps | runs (P1, 12 seeds x 2 lanes) | est. T4-hours |
|---|---|---|---|---|
| M (d=512) | ~180 | ~11 min | 24 | ~4.3 |
| L (d=1024) | ~600 | ~36 min | 6 (3 seeds x 2 lanes) | ~3.6 |

Under the 4-concurrent-session cap that is roughly **2-3 hours of wall
clock for the P1 spine at both new scales**, and the full three-edge
panel at M is ~10 T4-hours. On CPU the M arm alone would have been
~130 hours — which is why the ladder was gated on Phase B and not
merely helped by it. Colab discipline: tools/colab_supervisor
(checkpoint interval well under reclaim interval, events.jsonl +
resume relayed every tick) plus tools/colab_reap_orphans.py against
leaked sessions.

## Execution sketch (priced at licence time)

S arm: banked. Bridge: 6 runs (3 seeds × 2 lanes) — CUDA, cheap.
M arm: panel edges × 10 seeds (P1 alone: 20 runs; full panel ≈ 50).
L arm: P1 × 3 seeds = 6 runs. M/L run on Colab under
tools/colab_supervisor (admission rule: checkpoint interval well under
reclaim interval; events.jsonl + resume relayed every tick), receipts
copied into receipts/ here immediately on completion.
