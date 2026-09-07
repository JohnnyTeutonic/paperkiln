# transfer_s1, arm M banked: the pre-registered S-vs-M reading (6 Sep 2026)

*Analysis: `analyze.py` at its licence-anchored form, run verbatim on the
S and M artefact roots (72 runs each); output banked as
`receipts/M/ANALYSIS_M_20260906.txt`. Arm L (16x) is running and adds a
third, preliminary point; the lr = 5e-4 sensitivity cell at M (Threat 4)
is still to run. Nothing below is amended by either.*

## The licensed reading

**F1, primary (matched val-loss milestones): concordance 1.000 over 15
cells, seed-bootstrap band [0.867, 1.000]. Adopted (threshold 0.75, band
low > 0.50).** The fifteen cells are the fifteen edges at ONE milestone:
the S slice-800 level (val 3.847), which M's exact lane reaches at about
step 1400. **M never reaches the other four milestones** (3.656, 3.505,
3.369, 3.349): at the protocol's fixed lr = 1e-3 the d = 512 model bottoms
out at a median best val of 3.72 (S: 3.35). The matched-position rule
says an arm that never reaches a milestone contributes no cell there and
the omission is reported, never interpolated; this is that report. The
licensed claim is therefore: at the one early position both arms share,
all fifteen pairwise signs agree. (An earlier version of this file said
"at every matched milestone"; that was wrong and is corrected here,
7 Sep.)

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

## Threat 4, the lr sensitivity cell (7 Sep 2026; descriptive)

`sweep_M_lr.json`: d = 512, lr = 5e-4, L1 and L4, seeds 21 to 23
(`receipts/M_lr/THREAT4_lr_sensitivity_20260907.txt`). Two facts.

1. **At lr = 5e-4 the 4x model reaches every S milestone** (exact lane,
   all three seeds, final val 3.18 to 3.23, below S's 3.35); at lr = 1e-3
   it reaches only the first. The fixed learning rate, declared a
   protocol property, is the wrong learning rate for d = 512 and, from
   RESULTS_L.md, for d = 1024. This is the muP objection materialising,
   and it is why the milestone band collapses to one position.
2. **The sign pattern on the exact-vs-swa64s1 edge at M depends on lr:**
   fixed-slice sign agreement between the two learning rates is 16/27
   (seed 21: 6/9, seed 22: 8/9, seed 23: 2/9). The fingerprint is
   lr-sensitive at the same width.

By the pre-registration this cell cannot change the primary reading, and
it does not. What it changes is the scope line's weight: "lr fixed
across widths" is not a neutral convention here but the reason the
comparison is confined to one early milestone, and the larger arms are
mis-tuned under it. Any write-up has to carry both facts next to the
1.000.

## What this is worth

The result is narrower than the headline sentence suggests and should be
stated at its true width: at the one matched position the two arms share,
sign structure survives a 2x width change on all fifteen edges while the
scalar rankings do not, under a protocol whose fixed learning rate
mis-tunes the larger width. That is a licensed, receipts-backed finding
and the pre-registered outcome one. It is not yet "fingerprints transfer
across the training curve", because the curve was not shared past its
first milestone. The obvious next study is pre-registered, not
improvised: the same panel with the learning rate set per width by a
rule fixed in advance, so that every arm reaches every milestone and the
claim can be tested at five positions instead of one. The L arm (3 seeds, two lanes, preliminary by design) can only
add a trend point; the lr sensitivity cell can only qualify the scope
line. Neither can change the reading above.
