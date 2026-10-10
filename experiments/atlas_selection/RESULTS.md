# ATLAS selection audit: archival results

**Post-hoc; twelve seeds, one corpus, one width pair.**
No new training was run. No frozen transfer-study analysis was changed.

Report SHA-256: `2d541c92a5d78ca98c6ac75fc5a5d62310e45ceaa7ab3ed510d4af320df0f6cc`.

Input: 144 complete runs. All 924 six/six seed partitions; 108 source-position / target-position / shortlist cells.

Every selection uses calibration seeds; scoring uses different seed IDs at both widths. All losses are the logged validation-loss units.

## End-of-training decisions

Source step 3600, target step 3600. Negative contrasts favour the informed shortlist. Random is the exact expectation over uniform size-k subsets, using the same target calibration rule.

| k | Excess vs exact | Excess vs random | Excess vs direct target | Exclusion gap | Calibration gap |
|---:|---:|---:|---:|---:|---:|
| 1 | +0.129369 | +0.053855 | +0.126783 | +0.135431 | +0.000000 |
| 2 | +0.118044 | +0.097774 | +0.115457 | +0.124105 | +0.000000 |
| 3 | +0.000161 | -0.007089 | -0.002426 | +0.006222 | +0.000000 |
| 6 | +0.002587 | +0.000000 | +0.000000 | +0.000000 | +0.008648 |

## Source training position

Mean held-out loss difference versus random shortlisting, target step 3600. All source positions are shown; choosing the best row now is post-hoc tuning.

| Source step | k=1 | k=2 | k=3 | k=6 |
|---:|---:|---:|---:|---:|
| 400 | +0.183104 | +0.097774 | +0.037507 | +0.000000 |
| 800 | +0.183104 | +0.097774 | +0.036993 | +0.000000 |
| 1200 | +0.183104 | +0.097774 | +0.033450 | +0.000000 |
| 1600 | +0.183104 | +0.097774 | -0.007187 | +0.000000 |
| 2000 | +0.183104 | +0.094426 | -0.007534 | +0.000000 |
| 2400 | +0.183104 | +0.097774 | -0.007250 | +0.000000 |
| 2800 | +0.183104 | +0.097774 | -0.007239 | +0.000000 |
| 3200 | +0.180917 | +0.097774 | -0.007250 | +0.000000 |
| 3600 | +0.053855 | +0.097774 | -0.007089 | +0.000000 |

## Reference cell details

`{'source_step': 3600, 'target_step': 3600, 'k': 2}`

Split percentiles below describe overlapping partitions. **They are not confidence intervals.**

| Paired contrast | Mean | Split p05 | Split p95 |
|---|---:|---:|---:|
| loss_vs_random_shortlist | +0.097774 | +0.085089 | +0.110540 |
| loss_vs_direct_target | +0.115457 | +0.101629 | +0.128994 |

Selected-lane counts across the 924 overlapping splits:

- swa32s1: 924

Acquisition/scoring accounting (run-steps, separated by width):

- source_run_steps: 129,600
- target_run_steps: 43,200
- random_target_run_steps: 43,200
- direct_target_run_steps: 129,600
- heldout_scoring_run_steps_per_policy: 21,600

## What these results support

These are retrospective decisions in the existing six-lane transfer_s2 panel. Excess against exact and the paired policy contrasts quantify selection consequences. Exclusion plus calibration gap equals the gap to the held-out sample's best lane; that optimistic benchmark is not a population oracle.

The informed shortlist pays additional source cost. A favourable loss contrast alone does not establish a compute saving. There are no new data/corpus holdouts, fresh seeds, population confidence intervals, or confirmatory claims here. See README.md for the design, literature anchors and prospective next experiment.
