# transfer_s1, arm L banked: the preliminary 16x point (7 Sep 2026)

*Six runs (d = 1024, 32 heads, seeds 21 to 23, lanes L1 exact and L4
swa w64 s1), L4 GPU, checkpoint-resume across prunes. Receipts in
`receipts/L/`; the deterministic three-arm analysis is
`receipts/L/ANALYSIS_SML_20260907.txt`. Three seeds carry no inference
(PREREGISTRATION.md, power section): this is a trend point and is
labelled preliminary everywhere it appears. Nothing here amends the
licensed S-vs-M reading in RESULTS_M.md.*

## What L shows

**L does not reach the S-arm milestones.** The matched val-loss
milestones run from 3.847 (S slice 800) down to 3.349 (S slice 3600).
Every L run bottoms out between 3.85 and 3.93 and then rises: best val
at steps 2400 to 3100, final val 3.87 to 3.95, six of six. At lr = 1e-3,
fixed across widths as the declared protocol property, the 16x model
trains worse than the 4x model, not better, and turns over before the
budget ends. So the primary matched-milestone comparison is **undefined
at L: 0 cells for S vs L and for M vs L**, and Threat 2 (regime) fails
outright, 0/6, in the overfitting direction rather than the plateau seen
at M.

**At fixed steps, M and L agree: 1.000 over 9 cells; S vs L 0.444.**
On the one edge L carries (exact vs swa64s1), the sign of the comparison
at every 400-step slice matches M exactly and matches S on fewer than
half. With three seeds and one edge that is a direction, not a finding:
the 4x and 16x widths look alike at fixed position while the base scale
looks different, which is what the matched-position rule predicts when
the base scale is at a different place on its curve.

## What it means for the reading

- The licensed claim stays as written: sign structure transfers at
  matched position from d = 256 to d = 512, scalars do not. L neither
  confirms nor contradicts it, because L cannot be placed on the
  milestone scale at all.
- What L does show is the muP objection materialising exactly where the
  pre-registration said it might: lr = 1e-3 fixed across widths is a
  protocol property, and at 16x it is the wrong learning rate. The
  Threat 4 sensitivity cell at M (lr = 5e-4, L1 and L4, 3 seeds,
  `sweep_M_lr.json`, launched 7 Sep 09:10) is the pre-registered probe of
  that property; it is descriptive and cannot change the primary reading.
- Any L-arm claim in a write-up must carry: three seeds, one edge,
  milestone scale unreachable, regime failed 6/6.

## Housekeeping found on the way

F2's modal-class tie-break was nondeterministic (set order); fixed and
recorded as execution clarification 7. The deterministic F2 reading for
S vs M is 9/15 edges (ties go to the alphabetically first class, which
is "other"); the two earlier outputs read 7/15 and 9/15 and are kept.
