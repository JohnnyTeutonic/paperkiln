# transfer_s3 — results (11 October 2026)

*Analysis by `analyze.py` exactly as committed at the licence anchor
(f61e253), unchanged since, run after all 144 new runs were banked, with
all six cells. Receipts: `receipts/s3_S5/` and `receipts/s3_M25/` (72 runs
each, 144/144 pass `tools/validate_events.py`), the binary bridge in
`receipts/bridge/` (bit-identical), and the reused cells' receipts under
`../transfer_s1/` and `../transfer_s2/`. One execution clarification
(PREREGISTRATION.md): four M25 runs were re-run after a relay overwrote
their event logs; each re-run's result is identical to the original's.
Printed output: `receipts/ANALYSIS_20261011.txt`.*

## Verdict

**Reading: MIXED.** The two pairs of each hypothesis disagree, so the
registered table names no mechanism.

| Hypothesis | Pair | Per target, t1 to t4 | Over t1–t4 [band] | Verdict |
|---|---|---|---|---|
| H-POS (shared rate) | S25 vs M25 | 0.933, 1.000, 1.000, 0.933 | 0.967 [0.850, 0.983] | not supported |
| H-POS (shared rate) | S5 vs M5 | 0.933, 0.733, 0.733, 0.733 | 0.783 [0.600, 0.850] | supported |
| H-RATE (shared width) | S25 vs S5 | 0.933, 1.000, 0.933, 0.733 | 0.900 [0.767, 0.950] | supported |
| H-RATE (shared width) | M25 vs M5 | 0.933, 0.733, 0.667, 0.533 | 0.717 [0.600, 0.850] | not supported |

Every target was admitted for every licensed pair (S5 reaches t4 in 11 of
12 seeds; every other cell reaches every target in 12 of 12).

**Descriptive pairs.** Study 1's pair (S10 vs M10) reaches only t1 at
these targets: 0.867. Study 2's pair (S25 vs M5): 0.933, 0.733, 0.667,
0.467, over t1–t4 0.700 [0.583, 0.833], Study 2's result reproduced at
the fixed targets.

## What the pattern is (post-hoc, descriptive, licenses nothing)

Over every pair of the four cells that reach all four targets:

| Pair | t1 | t2 | t3 | t4 | over t1–t4 |
|---|---|---|---|---|---|
| S25 vs S5 | 0.933 | 1.000 | 0.933 | 0.733 | 0.900 |
| S25 vs M25 | 0.933 | 1.000 | 1.000 | 0.933 | 0.967 |
| S5 vs M25 | 1.000 | 1.000 | 0.933 | 0.800 | 0.933 |
| S25 vs M5 | 0.933 | 0.733 | 0.667 | 0.467 | 0.700 |
| S5 vs M5 | 0.933 | 0.733 | 0.733 | 0.733 | 0.783 |
| M25 vs M5 | 0.933 | 0.733 | 0.667 | 0.533 | 0.717 |

The three pairs that decline are exactly the three that include M5,
the width-512 panel trained at 5e-4. The three without it stay at 0.900
or above over the band. At a shared rate of 2.5e-4 the panel's ordering
carries from width 256 to width 512 through all four targets, with one
reversal in sixty cells.

## What this means for the transfer paper

The paper's account of Study 2 is that agreement decays along the loss
curve, and that the shared rate of Study 1 had confined the comparison to
the early band where agreement is high. Study 3 does not support that
account as a general one: with both widths at 2.5e-4, agreement holds
through the whole band. The late disagreement in Study 2 belongs to the
width-512 panel at 5e-4, the rate Study 2's rule selected for that width.
What the paper reports as measured stands. What changes is the
mechanism: on this evidence, the transfer verdict depends on the rate the
larger width is trained at, and the per-width rule chose a rate under
which the larger width orders the panel differently late in training.
This is a descriptive reading of one study, licensed by none of its
registered rules.

## Scope

One model family, two layers, widths 256 and 512, rates 2.5e-4 and 5e-4
(1e-3 descriptive only), one corpus slice, twelve seeds per cell, four
fixed loss targets.
