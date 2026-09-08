# Paper sketch: topology transfers, metric does not (8 Sep 2026)

*A sketch, not a draft. The author's rules apply: no draft until the
lr-per-width arm is pre-registered, run and banked; journal only; venue
chosen by reading recent issues, not from memory. Everything marked †
depends on that arm.*

## The frame

The object measured is the **comparison graph**: six attention lanes,
fifteen edges, each edge oriented at a training position by which lane
has the lower validation loss (seed-majority sign). Two layers live on
that graph:

- **Topology**: the orientation of every edge. What a tiny-scale panel
  is asked to deliver when it says "mechanism A beats mechanism B".
- **Metric**: everything with a magnitude. Effect sizes (Delta), the
  shape of Delta over budget, the crossover budget B*.

Claim: width preserves the topology and destroys the metric. The
invariant is **fibre-wise**: it holds at matched position on the loss
curve, and the lr cell shows a fibre can reorient under a different
learning rate. The protocol's job is to align the curves; once aligned,
the orientations carry.

Define the invariant precisely once (edge orientation at matched
val-loss position, seed-majority), then use "topology" and "metric" as
the names of the two layers throughout. A referee who prefers "ordinal"
has nothing to object to because the definition is explicit.

## Evidence in hand (S and M, licensed 6 Sep; corrected 7 Sep)

- F1 primary: concordance 1.000 over 15 cells at the one matched
  milestone both arms reach; seed band [0.867, 1.000]. Adopted.
- H-SCALAR: Spearman rho = -0.082 over 15 edges. Scalars do not transfer.
- F1 robustness at fixed steps: 0.681, not adopted (step-matching is the
  confound; this is the matched-position rule's prediction).
- F2 shape classes: 9/15 modal agreement; at M every edge is "other".
- F3 crossover: S crossed 11/12 (median b0 2000); M crossed 4/12, never
  8/12. The crossover moves late with width.
- Threat 2 (regime): M plateaus (best val at median 0.89 of run).
- The milestone band: S reaches all five, M reaches one, L none, at the
  fixed lr = 1e-3. M bottoms at val 3.72 vs S 3.35. This is the
  protocol flaw the protocol flagged (Threat 4).
- Threat 4 cell: at lr = 5e-4, M reaches all five milestones and trains
  below S; the exact-vs-swa64s1 sign pattern agrees between learning
  rates on 16/27 fixed slices. The fingerprint is lr-sensitive at fixed
  width.
- L (16x, 3 seeds, one edge): overfits from ~2700 at lr 1e-3; reaches no
  milestone; fixed-step M vs L 1.000 over 9 cells. Trend point only.

## The missing arm †

**transfer_s2 (pre-register before any run):** the same six-lane panel
at d in {256, 512, 1024}, learning rate set per width by a rule fixed
in advance (candidates: a 3-point lr sweep on the exact lane per width
with the winner locked before the panel runs; or a stated scaling
rule). Success condition, fixed now: every arm reaches every milestone
in >= 9/12 seeds, so F1 is evaluated at five positions. Keep the S arm
as is if its lr survives its own sweep; otherwise re-run S under the
rule so all arms are on the same footing. Twelve seeds at S and M; L at
twelve seeds if budget allows, else labelled preliminary again.

Predictions to write down before running: F1 >= 0.75 at every
milestone; H-SCALAR |rho| < 0.5; F2 and F3 remain metric-fragile.
Falsifier: F1 below 0.75 at any later milestone means the topology is
itself position-fragile, and the paper becomes the "protocol noise"
outcome with receipts.

## Sections (working order)

1. The objection: budget reversal and B* as a distribution (our own
   results) make tiny-scale conclusions suspect.
2. Turning the objection into a measurement: the comparison graph,
   topology vs metric, matched position vs matched step.
3. Protocol and pre-registration (thresholds, foil, threats, the
   amendment history, all dated).
4. Results: topology (F1 at five positions †), metric (H-SCALAR, F2,
   F3), the width-fixed-lr failure and the lr cell as its diagnosis.
5. What a tiny-scale panel can license: which mechanism wins, not by
   how much, and only at the same place on the curve.
6. Limits: one family, two layers, one corpus slice, three widths; the
   invariant is a property of this protocol across this range.

## Assets already banked

Receipts S, M, L, M_lr; analyze.py with the milestone band and the
deterministic F2; RESULTS_M.md and RESULTS_L.md; the checkpoint-resume
and sharded Colab runner (a methods footnote, not a section).
