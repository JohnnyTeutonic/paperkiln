# Pre-registration — sparse_s1_boundary: locating B*(256)

*Written and committed BEFORE any run. 21 Aug 2026. The licence anchor
for every directional claim below is THIS commit.*

## The object

The programme's accumulated results define a crossover budget
**B\*(d)**: the training budget at which the sign of
Delta(b) = bestval(swa) − bestval(exact) flips from negative (sparse
ahead) to positive (dense ahead), at fixed window w=64, sinks=1, T=256,
layers=2, house protocol.

What is already established, each under its own pre-registration:

- **B\*(128) ∈ (400, 1200]** — S1-c: Delta flips −0.0472 → +0.0123,
  shrink t = 5.09 (df=4), 5/5 seeds same direction.
- **B\*(256) > 1200** — Rung B: shrink replicates (t = 4.03, all five
  shrink_s positive) but Delta(1200) = −0.0279 with 5/5 seeds still
  negative. The crossover has not arrived at this width by this budget.

This experiment extends the d=256 budget axis 1200 → **3600** on the
same five paired seeds to either LOCATE B\*(256) or BOUND it below.

## Design

Identical to Rung B (as amended) except `train.steps`: 3600.
d=256, heads=8, layers=2 (gpt2-nano family pin), window=64, sinks=1,
T=256, batch=4, lr=0.001, eval_every=100 (so the b=400 and b=1200
slices are extracted by exactly the loader Rung B used), seeds
{1,2,3,4,5}, both lanes as factor cells in one grid, TinyStories
slice + chat7b vocab (cap 4096). Local CPU execution via mtsweep
(the Rung B venue); out_root is persistent storage, never /tmp
(the S1-d incident rule).

## Hypotheses and decision rules (fixed now)

**H-SHRINK-CONT (committed direction — one-tailed licensed by this
commit):** within-seed paired
`shrink2_s = Delta_s(3600) − Delta_s(1200) > 0`.
Paired t over 5 seeds; one-tailed crit 2.132 (df=4); two-tailed 2.776
reported alongside. This is the twice-replicated effect; its direction
is committed pre-data for the third time.

**H-CROSS vs H-NO-CROSS (NO direction committed — two-tailed only):**
the sign of Delta(3600).
- If Delta(3600) > 0 with paired two-tailed t clearing 2.776:
  **B\*(256) ∈ (1200, 3600]** — the boundary point is located and the
  phase-boundary claim gains its second width.
- If Delta(3600) < 0 clearing the same bar: **B\*(256) > 3600** — the
  boundary is bounded below; compatible with both "arrives later" and
  "never arrives at this width". No stronger claim is licensed.
- If neither clears: the sign at 3600 is undetermined; report the
  estimate and interval, claim nothing about B\*.

Named rival, stated before data: **B\*(256) may be effectively
infinite** — sparse may simply win at this width on this task. That
outcome is a RESULT (the boundary terminates or bends sharply), not a
failure of the experiment. Neither outcome gets a one-tailed rescue.

**D-SHAPE (descriptive only, no test):** Delta at every 400-step slice,
for the trajectory plot. Candidate input to a future functional-form
question about B\*(d); no claim licensed here.

## Threat checks (fixed now)

1. **Regime check:** both lanes' eval loss still decreasing at 3600
   (no early stop, no upturn). If either lane has entered overfitting
   (best_val achieved ≥ 3 evals before 3600 in ≥ 3 seeds), the
   crossover statement is scoped "in the still-training regime up to
   the overfit onset" and D-SHAPE carries the report.
2. **Guard (refuse-to-run):** every run's model event must record
   d=256, layers=2, and its own lane's attention field; lanes are read
   from model events, never directory names (Rung B loader rule).
3. **Comparability guard:** Delta_s(1200) recomputed here must agree
   in sign with Rung B's stored Delta_s(1200) on ≥ 4/5 seeds; gross
   disagreement voids the extension (different effective protocol)
   rather than licensing a new claim.

## What this cannot show

Scope: TinyStories slice, one architecture family, w=64 only, 2
layers, five seeds, CPU numerics. B\*(d) here is a property of THIS
protocol. No claim about published sparse-attention results at scale;
the general-methods claim remains the budget-reporting discipline.

## Execution

`sweep.json` beside this file; runner `tools/mtsweep.py` (resumable —
completed cells skip). Analysis `analyze.py` beside this file, written
and committed with this pre-registration, run only after all 10
result.json exist.
