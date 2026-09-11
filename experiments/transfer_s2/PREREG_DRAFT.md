# transfer_s2 — pre-registration DRAFT: the same panel with the learning rate set per width

*Drafted 11 September 2026, before any transfer_s2 run existed. This is a
DRAFT, not the licence anchor. The anchor is the commit Jonathan nominates
after reading it, with the analysis script in its final form. Stage 1
below is a calibration step whose rule is fixed here and whose outputs
select a hyperparameter; it reads no panel data and licenses no claim.
House protocol as in transfer_s1: pre-registered hypotheses, receipts for
every run, analysis run verbatim from the anchor commit.*

## Why this study exists

transfer_s1 (RESULTS_M.md, RESULTS_L.md) licensed "sign structure
transfers, scalars do not" from d = 256 to d = 512, but only at the one
val-loss milestone both arms reach. The fixed learning rate lr = 1e-3,
declared there as a protocol property, mis-tunes the larger widths: M
bottoms at val 3.72 against S's 3.35 and never reaches the four later
milestones; L overfits from about step 2700 and reaches none. The Threat 4
cell showed M at lr = 5e-4 reaching all five milestones, and showed the
edge sign pattern is itself lr-sensitive. So the claim is narrow because
the curves were never aligned, and the remedy is to align them by a rule
fixed before the panel runs.

## Stage 1: per-width learning-rate selection (calibration, not evidence)

**Runs.** For each width d in {256, 512, 1024}: the exact lane (L1) only,
seeds 21 to 23, 3600 steps, everything else exactly as the transfer_s1
sweeps, at a four-point learning-rate grid:

| width | grid | already banked (reused, identical spec) | new runs |
|---|---|---|---|
| 256 | 2e-3, 1e-3, 5e-4, 2.5e-4 | 1e-3 (arm S, seeds 21-23) | 9 |
| 512 | 2e-3, 1e-3, 5e-4, 2.5e-4 | 1e-3 (arm M), 5e-4 (M_lr cell) | 6 |
| 1024 | 1e-3, 5e-4, 2.5e-4, 1.25e-4 | 1e-3 (arm L) | 9 |

The 1024 grid is shifted one step down because arm L showed 1e-3 already
past its optimum; that is a design choice made from transfer_s1's banked
receipts and disclosed here, not from any transfer_s2 run. Reuse is
permitted only where the spec is identical apart from the checkpoint
interval, which does not touch numerics (transfer_s1, clarification 6).

**Selection rule, fixed now.** For each width, among the grid points that
pass the regime check (best val within the last three evals in at least
2 of 3 seeds), lr*(d) is the one with the lowest median best val over
seeds. If no grid point passes, lr*(d) is the one whose median best-val
step is latest. Ties go to the larger learning rate. The rule is applied
by `select_lr.py`, committed with this file, and its output is recorded
in RESULTS_STAGE1.md with every number that fed it.

**What stage 1 may not do.** It may not look at any lane other than
exact, any seed beyond 21 to 23, or any transfer quantity. It selects
three numbers and stops.

## Stage 2: the panel (the licensed study)

Identical to transfer_s1 in every respect except the learning rate,
which is lr*(d) per width: six lanes, twelve seeds at S and M, L at
twelve seeds if budget allows and otherwise at three and labelled
preliminary. If lr*(256) is not 1e-3, arm S is re-run at lr*(256); the
transfer_s1 S arm is not reused, so that every arm is on the same footing.

**Reachability condition, fixed now.** Milestones are the L1 val-loss
values at the S arm's per-seed median trajectory at slices {800, 1600,
2400, 3200, 3600}, as in transfer_s1. The panel is fit for its purpose
only if every arm reaches every milestone in at least 9 of 12 seeds
(3 of 3 at a three-seed L). If an arm fails this, the study reports the
failure as a finding about the protocol and the primary reading is
confined, as in transfer_s1, to the band reached, with the omission
printed by the analysis.

**Hypotheses and thresholds: unchanged from transfer_s1.** F1
sign-pattern concordance at matched milestones, adopted iff >= 0.75 with
seed-bootstrap 2.5th percentile > 0.50, now evaluated at up to five
positions; H-SCALAR Spearman |rho| < 0.5 as the committed foil; F2 shape
classes and F3 crossover distributions descriptive. Headline "scalars
don't transfer, structure does" claimable only if F1 clears at EVERY
reached milestone and H-SCALAR meets its condition.

**Falsifier, fixed now.** F1 below 0.75 at any later milestone means the
sign structure is position-fragile even with aligned curves, and the
paper becomes the receipts-backed "protocol noise" outcome. Both are
bankable; neither is chosen in advance.

**Predictions written before running.** F1 >= 0.75 at every milestone;
|rho| < 0.5; F2 and F3 remain metric-fragile; the fixed-step robustness
variant still fails, because matched steps are still different places on
different curves.

## Threats carried over, and one new one

Numerics bridge: satisfied by the same binary and GPU (L4) as
transfer_s1; no arm mixes GPU types. Regime check per lane, drift guard
on the L1-vs-L4 edge at 1200 steps, lanes read from model events, no
peeking: all as before. **New:** lr selection on the exact lane could
favour the exact lane in the panel. This is accepted and disclosed; the
alternative, selecting per lane, would change what a lane is. The
Threat 4 cell's finding that sign patterns are lr-sensitive is the reason
the selection rule is fixed and mechanical.

## Cost

Stage 1: 24 new runs. About 2.5 GPU-hours at S, 10 at M, 30 at L on L4
with checkpoint-resume across prunes. Stage 2 at twelve seeds per arm:
roughly transfer_s1 again (S 6 h, M 30 h, L 3 to 4 days at 2 cells) plus
S again if its rate changes.

## Analysis

`analyze.py` from transfer_s1 at commit d88584f (milestone band printed,
deterministic F2), invoked with the transfer_s2 artefact roots; any
change to it before the anchor is listed here with a reason.
