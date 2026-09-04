# Pre-registration — transfer_s1: do fingerprints transfer across scale?

*Written and committed BEFORE any run of this experiment. 31 Aug 2026.
The licence anchor for every directional claim below is THIS commit,
which also carries `analyze.py` in its final form. Supersedes
PREREG_DRAFT.md (kept beside this file as the audit trail of which
choices were open and when they were closed).*

## The question

The registry's own strongest results are the standing objection to
tiny-scale work — and they are ours: **S1c-budget-reversal** (the sign
of a sparse-vs-dense comparison depends on where you stop),
**S1e-bstar-distribution** (the crossover budget is a distribution over
seeds, not a point), **NEEDLE-scale-negative** (a task family that does
not resolve at 2 blocks), **S3-d-unpaid** (a width crossover above our
scale). If effects reverse with budget and vary with seed, why would
any tiny-scale measurement inform a larger model?

This experiment turns that objection into a measured quantity:

**When width grows 4x and 16x, is the STRUCTURE of a comparison panel —
the pattern of pairwise dependencies — preserved, even where the scalar
rankings are not?**

Three outcomes, all bankable: structure transfers while scalars do not
(tiny-scale fingerprinting is licensed, and "scalars don't transfer,
structure does" is the headline); neither transfers (the field's
tiny-scale ablation practice measures protocol noise, demonstrated with
receipts); both transfer (tiny-scale screening validated outright). The
only bad outcome is one too blurry to distinguish these, which is a
power problem, addressed pre-data below.

## The lane set and the fingerprint object

Six lanes, every one with registry precedent (S1b window-monotone,
S1c, S1d sink-inert, S1e):

| lane | attention |
|---|---|
| L1 | exact |
| L2 | swa w=16, sinks=1 |
| L3 | swa w=32, sinks=1 |
| L4 | swa w=64, sinks=1 |
| L5 | swa w=128, sinks=1 |
| L6 | swa w=64, sinks=0 |

**The fingerprint of an arm is the COMPLETE PAIRWISE SIGN MATRIX over
these six lanes** — all C(6,2) = 15 edges — evaluated at the
pre-committed positions below, as a DISTRIBUTION over seeds (S1e
forbids treating any single seed's matrix as the arm's fingerprint).
This is the operational reading of "the pattern of dependencies, not
the scalar ranking": an edge is a signed claim "lane A beats lane B
here", and the fingerprint is all of them at once.

## Arms

| arm | d | heads | seeds | lanes | role |
|---|---|---|---|---|---|
| S | 256 | 8 | 12 | all 6 | reference fingerprint |
| M | 512 | 16 | 12 | all 6 | primary transfer contrast (4x params) |
| L | 1024 | 32 | 3 | L1, L4 only | preliminary trend point (16x) |

Head dim is held at 32 (heads = d/32), so width scales head COUNT, not
head size — the axis the sparse-attention literature actually varies.
Everything else identical across arms and identical to the banked S1
protocol: gpt2-nano family, layers=2, T=256, batch=4, lr=1e-3,
steps=3600, eval_every=100, TinyStories slice + chat7b vocab (cap
4096). Venue: CUDA (Phase B, adopted 31 Aug 2026 at 30.5x over CPU AVX
with identically-converging trajectories).

**DECISION — budget scaling: SAME STEPS (3600), same batch, same T, at
every arm.** Rejected alternatives: token-matched (identical here,
since batch and T are fixed) and FLOP-matched (would change the
budget with width, confounding the very axis under test with the axis
S1c proved is decisive). Width is the ONLY factor that moves. The
"but the arms are at different points in training" objection is
answered by the matched-position rule, not by moving the budget.

**DECISION — the banked CPU data is the BRIDGE, not the S arm.**
sparse_s1_boundary + sparse_s1_seeds hold 15 CPU seeds of L1-vs-L4 at
d=256. Reusing them as the S arm would mix venues (CPU S vs CUDA M)
inside the primary comparison. Instead the S arm is re-run in full on
CUDA, and the banked CPU cohort becomes the numerics-bridge reference
(Threat 1) — a stronger use of it: it converts "did the engine change
anything?" from an assumption into a measurement, against 15 seeds of
pre-registered prior data.

## The matched-position rule (fixed now)

Cross-arm comparisons are made at matched RELATIVE positions, never at
matched absolute steps — S1c makes absolute-step matching a known
confound.

**PRIMARY: matched validation-loss milestones on the exact lane (L1).**
Milestones are the L1 val-loss values at the S arm's per-seed median
trajectory at slices {800, 1600, 2400, 3200, 3600}, restricted to the
band every arm actually reaches (an arm that never reaches a milestone
contributes no cell there, and that omission is reported, never
interpolated). For each arm and seed, an edge's sign at a milestone is
evaluated at the first eval where that seed's L1 val loss crosses it.

**DROPPED, with an empirical reason: fraction-of-overfit-onset.** The
draft's option (a) is undefined at this protocol — sparse_s1_seeds
Threat 2 found best_val landing within the last 3 evals in **14 of 15**
seeds, i.e. overfit onset is not reached inside 3600 steps. A rule that
cannot be evaluated on the reference arm cannot anchor the study.

**ROBUSTNESS: fraction-of-total-steps** (the nine 400-step slices the
banked analyses already use). Reported alongside the primary in full;
divergence between the two readings is itself a finding and is
reported, not reconciled.

## Fingerprint layers and decision rules

**F1 — sign-pattern concordance (PRIMARY).** For each (edge, position)
cell, take each arm's seed-majority sign. Concordance = fraction of
cells where S and M agree. Inference by seed bootstrap: 10,000
resamples of seeds with replacement WITHIN each arm, recomputing the
majority sign matrices and the concordance each draw; report the point
estimate and the 2.5/97.5 percentiles.
- **"Structure transfers" is adopted iff concordance >= 0.75 AND the
  bootstrap 2.5th percentile > 0.50.**
- 0.50 is the coin (the seed lottery's own number past 2000 steps);
  0.75 is a substantive majority fixed pre-data.
- **Stated limitation, not discovered later:** cells within an edge
  are correlated across positions, so the bootstrap interval is a
  seed-noise band, NOT a p-value, and is reported as such.

**F2 — shape-class concordance (SECONDARY).** Each seed's Delta(budget)
trajectory per edge is classified by a rule fixed now, over the nine
400-step slices:
- **flat** iff max |Delta(b)| < **0.02** across all slices (the band =
  one SD of the tightest S-arm slice, SD(Delta@400) = 0.0179, measured
  in sparse_s1_seeds);
- **monotone+** iff non-flat, Delta rises at >= 7 of 8 adjacent slice
  steps, and Delta(3600) > 0; **monotone-** symmetric;
- **single-crossing** iff non-flat with exactly one persistent sign
  change (the S1e b0 definition: smallest b with Delta(b') of one sign
  for all b' >= b);
- **other** otherwise.
Concordance = agreement of the MODAL class per edge across arms;
reported with the full class distribution, no threshold test.

**F3 — B\* as a distribution (the transfer law; EXPLORATORY).** For
every edge, the per-seed persistent-crossing budget b0 (S1e's
definition), INCLUDING the never-crossed mass as a category rather than
a missing value. Compare distributions across arms (median, spread,
never-crossed fraction). S1e already refuted a point-valued B\* at the
base scale, so a point-to-point comparison is not run. At two widths
this is a direction, not a fitted law; the L arm adds a third point at
preliminary weight.

**H-SCALAR (the committed foil).** Spearman rank correlation of the
scalar mean Delta at the final matched milestone across arms, over the
15 edges. "Scalars don't transfer" is adopted iff |rho| < 0.5 or its
95% CI includes 0. **Honest power statement, fixed now:** n = 15 edges
is a weak basis for a rank correlation, so H-SCALAR is a DESCRIPTIVE
foil and can never carry the headline alone.

**Headline licence.** "Scalars don't transfer, structure does" may be
claimed ONLY if F1 clears its threshold AND H-SCALAR meets its
condition. Both halves measured; neither assumed. If F1 clears and
H-SCALAR also shows strong correlation, the claim is the weaker and
still-useful "tiny-scale screening transfers, scalars included".

## Power (from measured S-arm variance, sparse_s1_seeds RESULTS.md)

Measured at d=256, n=15: SD(Delta@1200) = 0.0214, SD(Delta@3600) =
0.0377, shrink effect +0.0541 with per-seed SD 0.0414, sign split 9+/6-
at 3600, b0 spread 1600 -> never (6/15 never crossing).

- Within-arm shrink at alpha=0.05 one-tailed: n=9 gives power 0.80,
  **n=12 gives 0.90**. Seeds fixed at **12 per arm** for S and M.
- L is 3 seeds and carries NO inference — it is a trend point, labelled
  preliminary everywhere it appears.
- If the M arm's measured SD exceeds the S arm's by more than 1.5x (the
  bridge cells measure this BEFORE the panel launches), the M seed
  count rises to 18 before any panel cell runs. This escalation is
  fixed here, pre-data, and may not be revised after seeing panel
  results.

## Threat checks (fixed now)

1. **NUMERICS BRIDGE (gate — runs first, panel blocked on it).** 5
   seeds of L1 and L4 at d=256 on CUDA, compared against the banked CPU
   cohort (sparse_s1_boundary/seeds, same seeds). **AMENDED 31 Aug 2026,
   pre-data — see Amendment 1 below.** Criteria as amended: per-run
   relative val-loss difference at step 100 <= 1e-3 for every seed and
   lane (PRIMARY), and the pooled mean Delta(3600) within 2 SE of the
   banked pooled mean (retained). Per-seed sign agreement is reported
   but no longer gates. Failure HALTS the study and the discrepancy is
   written up as its own finding — an engine that changes conclusions is
   a bigger result than this experiment.
2. **Regime check.** Per lane per arm, best_val must land within the
   last 3 evals in >= 9/12 seeds, scoping every statement to the
   still-training regime (the S-arm condition measured in
   sparse_s1_seeds).
3. **Protocol-drift guard.** M-arm L1-vs-L4 Delta(1200) must be
   negative on >= 8/12 seeds (S arm: 15/15 across boundary + seeds).
   Fewer voids cross-arm pooling of that edge and the discrepancy is
   reported as its own finding.
4. **LR/width confound.** lr = 1e-3 fixed at every width, declared as a
   PROTOCOL PROPERTY and stated in every scope line (the muP objection
   is acknowledged, not answered). Plus a 3-seed lr=5e-4 sensitivity
   cell at M on L1/L4 only, reported descriptively; it cannot change
   any primary reading.
5. **Refuse-to-run guard.** Every run's model event must record d,
   layers, heads, seed, and its lane's attention fields; lanes are read
   from model events, never directory names; every receipt must pass
   `tools/validate_events.py` (docs/EVENTS_SPEC.md) before analysis.
6. **No peeking.** `analyze.py` is committed WITH this file, before any
   run exists. The S-arm fingerprint is computed by that script from
   runs made after this commit, not chosen after inspection.

## AMENDMENTS (a licensed rule WAS changed — read this before trusting the gate)

### Amendment 1 — the bridge gate's sign criterion was confounded
**Made 31 Aug 2026, with the bridge arm at 0/10 runs. No bridge data
existed when this was written; the driver log timestamps establish it.**

The original criterion required per-seed SIGN agreement on Delta(3600),
>= 4/5. That criterion is wrong, and the arithmetic is not marginal.

A backend change perturbs a trajectory at roughly 1e-7 per operation —
different reduction order, different kernels, FMA contraction. Training
is a chaotic system, so by 3600 steps that perturbation has been
amplified to macroscopic scale. **A backend change therefore behaves
like a reseed**: "seed 3 on CUDA" is not a rerun of "seed 3 on CPU", it
is an independent draw from the same distribution. This is the same
phenomenon the seed lottery documents, entering through a different
door.

S1e measured that distribution: 9+/6- at b=3600, so p(+) = 0.60. Two
independent draws therefore agree with probability
p^2 + (1-p)^2 = **0.52**, and P(>= 4/5 agreements) = **0.21**.

**The gate as licensed would have halted the study four times in five
on a perfectly correct engine.**

**Replacement.** Loosening a gate is the amendment most deserving of
suspicion, so the replacement is deliberately STRICTER about what the
bridge exists to test. The bridge asks "do the kernels compute the same
thing", not "do chaotic trajectories reconverge" — and that is answered
EARLY, before chaos amplifies:

- **PRIMARY (new, strict):** per-run relative val-loss difference at
  step 100 <= 1e-3, every seed, every lane. A broken kernel is gross
  here — the CPU-only regression of 30 Aug produced loss = ln(vocab)
  from the first step — while a correct kernel is invisible.
- **RETAINED unchanged:** pooled mean Delta(3600) within 2 SE of the
  banked pooled mean. This was always distribution-level and is not
  affected by the confound.
- **DEMOTED to descriptive:** per-seed sign agreement, printed with its
  ~52% null expectation beside it so a low value cannot be misread as a
  fault.

Both directions are re-verified by `gate_test.py`: identical data gives
0.00e+00 early divergence and PASSES; a swa lane shifted by +0.15 gives
3.41e-02 (34x the tolerance) and FAILS on the early check.

**What this amendment does NOT do:** it does not touch F1, F2, F3,
H-SCALAR, the matched-position rule, the seed counts, or any threshold
carrying a directional claim. It replaces one threat-check criterion
that could not have done its job.

## Execution clarifications (added after the licence commit; NO rule changed)

*Recorded here rather than edited silently. Both were found on
31 Aug 2026 by testing the analysis before any run existed, and
neither alters a hypothesis, threshold, or decision rule.*

1. **Legacy lane identification, bridge reference arm only.** mtstudio
   began emitting `window`/`sinks` in the model event only with the
   deep-SWA work; the banked CPU cohort predates it and records bare
   `attention="swa"`. Those experiments had exactly ONE swa
   configuration by construction (w=64, sinks=1, fixed in their
   committed sweep.json), so for that cohort the lane is read from the
   manifest and the substitution is ANNOUNCED in the analysis output.
   The panel arms (S/M/L) run on the new binary and are read strictly —
   a legacy receipt reaching the strict path is dropped, never merged
   into a real lane, because five swa lanes are otherwise
   indistinguishable. This preserves Threat 5 (lanes from model events,
   never directory names) for every arm that carries a claim.
2. **The gate is tested in both directions before it is trusted.**
   `gate_test.py` beside this file synthesises a bridge arm from the
   banked receipts and runs the real gate twice: identical data must
   PASS (observed 5/5 sign agreement, pooled means equal), and a swa
   lane shifted by +0.15 must FAIL (observed 3/5, mean outside 2 SE,
   "STUDY HALTS"). A gate that cannot fail is not a gate.

3. **H-SCALAR's `|rho|` treats anti-correlation as transfer.** The
   licensed rule adopts "scalars don't transfer" iff `|rho| < 0.5`.
   A strongly NEGATIVE rho therefore fails the condition — i.e. counts
   as scalars transferring — even though a reversed ranking is the
   worst possible outcome for anyone screening at small scale. The rule
   is NOT amended (it is licensed, and the case is unlikely given four
   replications of a consistent direction), but `analyze.py` now emits
   an explicit warning whenever rho <= -0.5, so the threshold can never
   speak for that case unchallenged. Found by the smoke test's
   scrambled regime before any data existed.

4. **CUDA execution config: `MICROTORCH_DEVICE_OPS=1` (31 Aug 2026,
   commit `697e281`), decided BEFORE any run completed.** The first
   bridge attempt ran gemm-only because the device op set OOMed at
   step ~95. That leak is now fixed (`In::owned` clobbered by member
   initialization order), re-measured flat over 400 steps, and the op
   set is 1.31x faster. **No bridge run had completed when this
   changed** — zero receipts existed, so no data informed the choice
   and nothing is being re-run after seeing a result.

   This is a config change, not a rule change: no hypothesis,
   threshold, or decision rule is touched. It is recorded here because
   the whole point of the bridge gate is that **the backend can change
   the conclusion**, so which backend the panel ran on is part of the
   result. Every arm that carries a claim (S/M/L) runs on this same
   config, and the gate compares it against the banked CPU cohort —
   which is exactly the comparison the gate was written to make.

5. **GPU: L4, 4 cells concurrent (1 Sep 2026), again with ZERO runs
   banked.** The T4 runtime has 2 vCPU and is CPU-saturated (load 2.1
   while the GPU idles at 11%), giving ~52 min per run against an
   observed ~52 min Colab reclaim interval. Only COMPLETED runs are
   banked, so runs were dying just short of the line and the arm could
   have made no progress indefinitely — one run was lost that way at
   00:31. Measured on an L4 (12 vCPU), 200 steps per cell: 4 concurrent
   cells give 3.4x the throughput at 48 min per run, which fits inside
   a vm lifetime; 8 cells give barely more throughput at 71 min per
   run, which does not.

   **This is a numerics-relevant change and it is applied uniformly.**
   A different GPU can reorder floating-point reductions, which by this
   study's own thesis is enough to move a result — see
   `../../atlas/THEOREM_CROSSING.md` on non-confluence. So **every arm
   that carries a claim runs on L4**: bridge, S, M and L. No arm mixes
   GPU types, and no partially-completed T4 work was kept — the one
   T4 run in flight was discarded rather than banked, precisely to
   avoid a mixed-backend cohort.

   Concurrency itself is numerically inert: each cell is a separate
   process with its own seed and its own output directory, and cells
   never share state. It changes only how many runs a vm banks before
   it is reclaimed.

## What this cannot show

One architecture family, one corpus, layers=2, T=256, one house
protocol, widths 256 -> 1024 (16x params, far from 7B). A transfer
result here is a property of THIS protocol's fingerprints across THIS
width range, at fixed lr. It does not establish that any published
result is wrong. What it CAN show — for the first time with seed
distributions and pre-registration on both sides of a scale gap — is
whether the things tiny-scale labs actually publish (signs, shapes,
crossovers) are the kind of quantity that survives a width change.

5. **Checkpoint interval for arms M and L: `checkpoint_every` 200 (M) and
   100 (L), was 1000000 (4 Sep 2026), decided BEFORE any M or L run
   existed.** Colab prunes every session at 60 minutes and an M run takes
   74 to 100 minutes, so M and L can only complete by checkpointing and
   resuming across sessions. Checkpoints now carry the full optimizer
   state (AdamW moments and timesteps, Muon momentum) and the batch
   stream is replayed exactly on resume. Proof on the CUDA path: an
   M-shaped probe cell (d=512, 16 heads, exact, seed 21, 900 steps) was
   run uninterrupted and again with its session killed by hand after
   step 100 and resumed on a fresh VM; every post-resume step loss and
   eval matched the uninterrupted run to the float (`tools/test_resume.sh`
   proves the same on CPU for AdamW and Muon, with a negative control).
   A resumed run is therefore the same run, not an approximation of it,
   and no hypothesis, threshold, decision rule, or lane changes. Arm S
   ran with checkpointing disabled and is unaffected. Zero M or L
   receipts existed when this changed.

## Execution

`sweep_S.json`, `sweep_M.json`, `sweep_L.json`, `sweep_bridge.json`
beside this file. Runner `tools/mtsweep.py` on CUDA (resumable;
completed cells skip). Order is fixed: **bridge -> S -> M -> L**, with
the bridge gate evaluated before any panel cell runs. Colab under
tools/colab_supervisor (checkpoint interval well under reclaim
interval; events.jsonl + resume relayed every tick;
tools/colab_reap_orphans.py against leaked sessions). Receipts
(events.jsonl, result.json, driver.log) copy into `receipts/` here
immediately on completion — never left in VM-only storage.

Estimated cost from the 31 Aug adoption gate (extrapolated to
T=256/L=2, measured for real by the bridge): S ~6 T4-hours, M ~13,
L ~3.6, bridge ~1.
