# transfer_s3 — PRE-REGISTRATION: width and learning rate, crossed

*Written 11 October 2026 before any transfer_s3 run existed, and licensed
by Jonathan the same day. **LICENCE ANCHOR: the commit that introduces this
file together with `analyze.py`, `smoke_analyze.py`, `bridge_check.py` and
the sweeps `sweep_S5.json`, `sweep_M25.json` and `sweep_bridge.json`.**
Anything changed after that commit is a dated amendment recorded at the
end of this file, never a silent edit.*

## Why this study exists

Studies 1 and 2 compared a six-lane attention panel across widths 256 and
512. Study 1 held the learning rate at 1e-3 for both widths and found
every edge's orientation preserved, at the one loss milestone the larger
width reached. Study 2 set the rate per width (2.5e-4 at 256, 5e-4 at 512)
and found agreement high at the first milestone and falling through the
later ones; the registered transfer criterion failed. The two studies
change the rate and the reachable band together, so they cannot say
whether the decline is a property of position on the loss curve or of the
rate mismatch between the two widths. This study separates them.

## Design

**Cells: width x rate.** Six cells, each the full panel (six lanes: exact;
swa with w = 16, 32, 64, 128 and one sink; swa with w = 64 and no sink),
twelve seeds (21 to 32), 3600 steps, gpt2-nano, two layers, T = 256,
batch 4, everything else as Studies 1 and 2.

| | 2.5e-4 | 5e-4 | 1e-3 |
|---|---|---|---|
| d = 256 | S25: Study 2 arm S (banked) | **S5: new** | S10: Study 1 arm S (banked) |
| d = 512 | **M25: new** | M5: Study 2 arm M (banked) | M10: Study 1 arm M (banked) |

The two new cells are `sweep_S5.json` and `sweep_M25.json`, identical to
Study 2's sweeps in every field except the learning rate and the output
directory. Four cells are reused, and their data have been analysed
before. Every licensed comparison below pairs a reused cell with a new
one, so no licensed verdict can be computed from data that exist today.

**Positions: four fixed loss targets.** The exact lane's validation loss
at 3.7539, 3.5152, 3.3491 and 3.2392 (t1 to t4): Study 2's first four
milestones, the band Study 2 admitted. They are fixed here and identical
for every cell, so two cells are compared at the same loss whatever
their rate. For each seed, the exact lane's first evaluation at or below
a target sets the step at which every lane of that seed is read, as in
Studies 1 and 2.

**Reachability, fixed now.** A target is admitted for a pair iff each
cell reaches it in at least 9 of 12 seeds. A hypothesis whose target is
not admitted is reported as untested, not as supported or unsupported.

## Hypotheses and decision rules (fixed now)

F1 is the share of the fifteen edges whose seed-majority orientation
agrees between the two cells of a pair, per target and over the admitted
targets. Its band comes from a seed bootstrap (10 000 draws, seed 12345)
that redraws seed labels per cell and reads each edge-and-target cell
from the drawn seeds that have a value there, which is the correction of
Study 2's frozen bootstrap.

**H-POS (position), one per shared-rate pair, S25 vs M25 and S5 vs M5.**
With the rate held equal across widths, agreement is high early and falls
by the last target: supported iff t1 and t4 are admitted, F1(t1) >= 0.75
and F1(t4) < 0.75.

**H-RATE (rate), one per shared-width pair, S25 vs S5 and M25 vs M5.**
With width held equal, changing the rate does not reorder the panel at
matched loss: supported iff F1 over the admitted targets >= 0.75 and the
band's 2.5th percentile > 0.50.

**Joint reading, applied mechanically by `analyze.py`.** A reading names a
mechanism only when both pairs of each hypothesis agree.

| H-POS (both pairs) | H-RATE (both pairs) | reading |
|---|---|---|
| supported | supported | **POSITION**: agreement decays along the loss curve at a shared rate, and rate alone does not reorder the panel. Study 2's decline is a position effect. |
| supported | not supported | **BOTH**: position decays agreement at a shared rate, and rate alone also reorders the panel. |
| not supported | not supported | **RATE**: agreement holds at a shared rate, and rate alone reorders the panel. Study 2's decline is a rate effect. |
| not supported | supported | **NEITHER**: agreement holds at a shared rate and across rates at a shared width; Study 2's decline needs both factors to move together. |
| pairs disagree | | **MIXED**: reported pair by pair, no mechanism named. |
| any untested | | **UNTESTED**: reported pair by pair. |

**Descriptive, licensing nothing:** Study 1's pair (S10 vs M10) and
Study 2's pair (S25 vs M5) at the fixed targets; reversals and ties per
target for every pair.

**Predictions written before running.** The account in the transfer
paper predicts POSITION. Every other row is a bankable outcome, and none
is chosen in advance.

## Threats (fixed now)

1. **Refuse-to-run.** `analyze.py` refuses any cell with a run whose model
   event does not record the cell's d, layers = 2 and learning rate, or
   whose lane is not one of the six. Lanes come from model events only.
2. **Binary bridge.** The banked cells ran on the CUDA binary built on
   12 September; the new cells run on a binary built from the current
   master. Before the panel launches, one banked run (S25, seed 21, exact
   lane) is re-run on the new binary for 800 steps on an L4. Its
   evaluation losses must match the receipt at every evaluation to a
   relative 1e-5. If they do not, the panel does not launch and the
   discrepancy is resolved first.
3. **Venue.** Every cell runs on L4 GPUs; no cell mixes GPU types.
4. **Exact-lane matching.** Only the exact lane is matched to the targets;
   other lanes are read at its crossing step. Accepted and disclosed, as
   in Studies 1 and 2.

## What this cannot show

One model family, two layers, two widths, three rates, one corpus slice.
POSITION would show the decline survives a shared rate in this family; it
would not show how general that is.

## Cost

144 new runs (72 per cell) on three L4 sessions with checkpoint relay:
about 18 session-hours for S5 and about 30 for M25, roughly 48 L4
session-hours in all and about 16 hours of wall time. The bridge check
adds under an hour.

## Analysis

`analyze.py`, committed with this file, run only after all 144 new
`result.json` exist, with all six cells: `python3 analyze.py --cells
S25=<root> M5=<root> S10=<root> M10=<root> S5=<root> M25=<root>`.
`smoke_analyze.py` exercises every decision branch on fabricated data
(SMOKE PASS on 11 October 2026, before any run).

## Amendments after the anchor

None.
