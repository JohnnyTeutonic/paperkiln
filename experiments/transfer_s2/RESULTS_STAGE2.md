# transfer_s2 stage 2: the pre-registered reading, S and M (14 Sep 2026)

Analysis: `experiments/transfer_s2/analyze.py` (frozen at the licence
anchor 90791ed), run verbatim on 14 Sep 2026 10:45 with
`--arms S=<s2_S> M=<s2_M>`. Receipt: `receipts/ANALYSIS_stage2_SM_20260914.txt`;
run receipts (events.jsonl + result.json, 72 + 72) under `receipts/s2_S/`
and `receipts/s2_M/`. Rates: lr*(256) = 2.5e-4, lr*(512) = 5e-4
(`RESULTS_STAGE1.md`). Everything else identical to transfer_s1. The L
arm (lr*(1024) = 5e-4 by the fallback clause) is running and is read
separately when it lands.

## What the pre-registration says happens now

The falsifier fixed before the panel ran: "F1 below 0.75 at any later
milestone means the sign structure is position-fragile even with aligned
curves, and the paper becomes the receipts-backed 'protocol noise'
outcome." The prediction written before running was F1 >= 0.75 at every
milestone and |rho| < 0.5. **The prediction is falsified. The falsifier
fired.**

## The numbers (verbatim from the receipt)

Milestone band reached (seeds whose exact lane reaches each milestone):

```
  S: 800:12/12  1600:12/12  2400:12/12  3200:12/12  3600:8/12
  M: 800:12/12  1600:12/12  2400:12/12  3200:12/12  3600:12/12
```

The curves are now aligned: M reaches every S milestone in 12 of 12
seeds. That was the whole purpose of stage 1, and it worked. (S reaches
its own last milestone in only 8 of 12 seeds, one short of the
reachability condition's 9 of 12 at that one position; the analysis
prints the omission as the rule requires. The four earlier milestones
are fully reached by both arms.)

F1, sign-pattern concordance at matched val-loss milestones, PRIMARY:

```
  concordance S vs M = 0.627 over 75 cells
  seed bootstrap 95% band [0.467, 0.667]
  F1 VERDICT: not adopted (need >= 0.75 and band low > 0.5)
```

F1, robustness variant (fixed 400-step slices): 0.630 over 135 cells,
band [0.585, 0.719], not adopted. F2 modal shape-class agreement: 5 of
15 edges. F3, exact vs swa64s1 crossover: S crossed 9 of 12 (median
b0 1600), M crossed 11 of 12 (median b0 3200).

H-SCALAR, the committed foil:

```
  Spearman rho over 15 edges = -0.600
  'scalars don't transfer' condition: not met (|rho| < 0.5)
```

The script's own warning applies and is repeated here because the
threshold must not speak for it: rho is strongly NEGATIVE. The M-arm
ranking of effect sizes is close to the reverse of the S-arm ranking.
Screening at the small width would select among the worst candidates at
the larger one.

Guards: Threat 2 (regime), best val within the last three evals: S 47 of
72 runs, M 19 of 72. Threat 3 (protocol drift) at M: Delta(1200)
negative in 5 of 12 seeds against the 8 of 12 required (S-arm precedent
in transfer_s1 was 15 of 15). The drift guard was calibrated on the
fixed-rate protocol and the rate now differs by design; it is reported,
not used to discard anything.

## Reading

1. **Aligning the curves removed the transfer.** In transfer_s1 the two
   arms shared a rate and shared its mistuning, and at the one milestone
   both reached, all 15 edge signs agreed. With each width at its own
   selected rate, both arms reach all five positions and the signs agree
   in 63 percent of cells, with a noise band that includes 0.5. The
   transfer_s1 concordance is therefore not evidence of width-invariant
   structure; the candidate explanation the receipts support is that it
   was the shared learning-rate regime that the two arms had in common.
   That is a finding about the earlier result, and it is the finding
   transfer_s2 was designed to be able to produce.

2. **The scalar foil did worse than 'no transfer'.** rho = -0.60 is the
   ranking reversing, not decorrelating. The reversal, not the F1
   number, is the most consequential single figure in the receipt for
   anyone who screens attention variants at small width.

3. **The licensed outcome is the pre-registered negative**: with the
   protocol's curves aligned, neither the sign structure nor the scalar
   ordering of a six-lane attention panel carries from d = 256 to
   d = 512 in this family. Scope as declared: gpt2-nano, two layers,
   T = 256, batch 4, TinyStories slice, chat7b vocab, L4, per-width rates
   selected on the exact lane by the stage-1 rule (an accepted and
   disclosed favouring of that lane).

4. **Not yet read**: the L arm (d = 1024), whose rate came from the
   fallback clause (`RESULTS_STAGE1.md`). Its band will be narrower; it
   is a trend point, and it cannot rescue F1, which failed at S vs M.

## Post-hoc, descriptive (added 14 Sep 2026 13:10; `posthoc/posthoc.py`, receipt `receipts/POSTHOC_20260914.txt`)

Labelled post-hoc; the licensed figure remains the aggregate above.

**Per-milestone F1, study 2 (S vs M):** 800: 0.933; 1600: 0.733; 2400:
0.667; 3200: 0.467; 3600: 0.333 (15 cells each). The concordance decays
monotonically along the curve, from near-complete agreement at the first
milestone to below chance at the last. Study 1's single reached
milestone was 800, where study 2 also agrees in 14 of 15 cells (the
per-edge signs at 800 are identical between the studies except one tie).
So the study-1 positive was real at its position; what the fixed rate
did was freeze the comparison at the one position where the two widths
agree, by stalling the larger width there.

**Seeds matter:** F1 recomputed on random k-seed subsets of each arm
(2000 draws, milestones recomputed from the S subset):

```
study 2  k=3:  median 0.653, 2.5-97.5 pct [0.493, 0.853], draws with F1 >= 0.75: 16.3 %
         k=6:  median 0.600, [0.507, 0.733], 1.0 %
         k=12: 0.627 (the study)
study 1  k=3:  median 0.933, [0.733, 1.000], 96.9 %
         k=6:  median 0.933, [0.800, 1.000], 98.9 %
         k=12: 1.000 (the study)
```

A three-seed version of study 2 would have "passed" the pre-registered
threshold about one draw in six; the study-1 positive is robust to seed
count, which is the point: its problem was position, not noise.

**Reading for the paper:** the sign structure of the panel agrees across
width early in training and diverges as training proceeds, ending in
reversal. Any single-position screen inherits the position it happens to
land on; the fixed rate landed study 1 on the agreeing one.

## What is not in this file

No per-milestone breakdown of F1: the frozen script reports the
aggregate over the reached band, as pre-registered. A per-milestone
table would be a post-hoc, descriptive addition and is labelled as such
if it is ever produced. No paper text. The transfer_s1 registry rows
(`T1-structure-transfers`, `T1-fixed-lr-mistunes`) are not edited here;
their status is the author's call now that the sequel has reported.
