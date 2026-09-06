# transfer_s1, arm M banked: the pre-registered S-vs-M reading (6 Sep 2026)

*Analysis: `analyze.py` at its licence-anchored form, run verbatim on the
S and M artefact roots (72 runs each); output banked as
`receipts/M/ANALYSIS_M_20260906.txt`. Arm L (16x) is running and adds a
third, preliminary point; the lr = 5e-4 sensitivity cell at M (Threat 4)
is still to run. Nothing below is amended by either.*

## The licensed reading

**F1, primary (matched val-loss milestones): concordance 1.000 over 15
cells, seed-bootstrap band [0.867, 1.000]. Adopted (threshold 0.75, band
low > 0.50).** Every one of the fifteen pairwise lane comparisons carries
the same seed-majority sign at d = 512 as at d = 256, at every matched
milestone.

**H-SCALAR (committed foil): Spearman rho = -0.082 over 15 edges.
"Scalars don't transfer" condition met (|rho| < 0.5).** The rank order of
the scalar effect sizes at the final milestone is uncorrelated across
widths.

**Headline licence: BOTH halves measured, both met. "Scalars don't
transfer, structure does" is claimed for this protocol across d in
{256, 512}.** This is outcome one of the three named in the
pre-registration, the one that licenses tiny-scale fingerprinting.

## What does not transfer, and the qualifiers

- **F1 robustness (fixed 400-step slices): 0.681, band [0.615, 0.748],
  not adopted.** Position matched by step rather than by val loss does
  not carry the sign pattern, which is exactly the matched-position
  rule's prediction: the widths train at different speeds, so equal
  steps are different places on the curve.
- **F2 shape-class: modal agreement 9/15 edges** (deterministic
  tie-break, clarification 7; the first run read 7/15 by set order). At M the modal class
  is "other" on every edge; S's single-crossing and monotone edges do not
  reappear as classes. Shape is protocol-fragile even where sign is not.
- **F3 crossing budget, exact vs swa64s1: S crossed 11/12 (median b0
  2000); M crossed 4/12, never-crossed 8/12.** The crossing moves late
  and mostly off the end of the budget at 2x width: the direction is
  "the sparse lane needs proportionally more budget to catch up", but
  two widths make this a direction, not a law.
- **Threat 3 (protocol drift): 12/12, passes.**
- **Threat 2 (regime): FAILS at M in every lane** (best val within the
  last three evals: exact 7/12, swa128s1 6/12, swa16s1 4/12, swa32s1
  3/12, swa64s0 5/12, swa64s1 6/12; the criterion is >= 9/12). At S it
  passes in four lanes (exact 9, swa128s1 10, swa32s1 9, swa64s1 11) and
  fails in two (swa16s1 2/12, swa64s0 8/12). The best-val step sits at a
  median 0.89 of the run at M (0.94 at S) and never earlier than 0.67 in
  either arm: the curves plateau on the 400k-token slice, they do not
  turn up. Per the pre-registration this scopes every statement to the
  still-training regime. The primary reading is computed at matched
  val-loss milestones taken from S's descending curve, all of which M
  reaches before its plateau, so the licensed claim lives inside that
  regime; the late fixed-step slices do not, which is a second reason
  the robustness variant fails and why it is reported, not adopted.
- Scope line, verbatim from the analysis: gpt2-nano family, layers 2,
  T 256, batch 4, lr 1e-3 fixed across widths (a protocol property, not
  a muP answer), TinyStories slice with the chat7b vocab, CUDA venue.

## What this is worth

The result is the one the programme needed: the standing objection to
tiny-scale work (effects reverse with budget and vary by seed) is
answered by a measured quantity, not an argument. Sign structure at
matched position survives a 2x width change perfectly while scalar
rankings do not, under a protocol whose own instability was documented
first. The L arm (3 seeds, two lanes, preliminary by design) can only
add a trend point; the lr sensitivity cell can only qualify the scope
line. Neither can change the reading above.
