# sparse_s1_longbudget — results (10 October 2026)

*Analysis by `analyze.py` exactly as committed before any run (licence
commit 89760cd; the script's last change, c96d893, also predates every
run), executed after all 20 `result.json` existed, on the receipts in
`receipts/runs/` (20/20 pass `tools/validate_events.py`; every model
event records d=256, layers=2, its lane, w=64, sinks=1 and its seed).
Amendment 1 (`checkpoint_every` 400) is the only change since the licence
commit. No deviations. Printed output: `receipts/ANALYSIS_20261010.txt`.*

## Verdict

From the joint outcome table fixed before data: **H-OVERFIT-ORDER not
supported, H-SECOND-CROSS not supported.** The monotone regime extends;
the crossing theorem stands unfalsified to 12000 steps at this protocol,
and the zone result carries the report.

## Numbers

**Threat 3 (overfit reality): passed.** 10/10 seeds show a held-out
minimum before the final eval (the rule asks for 8), so the runs reached
the regime the study was built to probe and H-SECOND-CROSS is tested,
not untested.

**H-OVERFIT-ORDER: not supported.** The exact lane reaches its held-out
minimum before the swa lane in 2 of 10 seeds; the registered line is 8.
Read descriptively from the printed table: the swa lane reaches its
minimum first in 7 seeds and the two tie in one (seed 43, both at 3800);
every minimum lies between 3800 and 5800 steps.

**Δ(b) = L_swa(b) − L_exact(b)** (positive: exact ahead), pooled over
the ten seeds:

| b | mean Δ | SD | threshold t·SD/√10 | zone |
|---:|---:|---:|---:|---|
| 1000 | −0.0516 | 0.0244 | 0.0175 | |
| 2000 | −0.0108 | 0.0331 | 0.0237 | in zone |
| 3000 | +0.0146 | 0.0345 | 0.0247 | in zone |
| 4000 | +0.0377 | 0.0366 | 0.0262 | |
| 5000 | +0.0629 | 0.0307 | 0.0219 | |
| 6000 | +0.0839 | 0.0393 | 0.0281 | |
| 7000 | +0.1159 | 0.0665 | 0.0476 | |
| 8000 | +0.1555 | 0.0534 | 0.0382 | |
| 9000 | +0.2090 | 0.0589 | 0.0422 | |
| 10000 | +0.2087 | 0.0506 | 0.0362 | |
| 11000 | +0.2142 | 0.0859 | 0.0614 | |
| 12000 | +0.2561 | 0.0640 | 0.0458 | |

**H-SECOND-CROSS: not supported.** One significant sign change, between
b=1000 (swa ahead) and b=4000 (exact ahead), and none after it. Once
both lanes have passed their minima, the gap widens instead of
reversing, reaching +0.256 at 12000 steps.

**Z-CLOSE: the zone closes at b=4000** and stays closed through 12000.
At this shape the S1e study, with fifteen other seeds, found the zone
still open at its last slice, 3600; here it closes at the next reporting
slice.

**W-CHECK.** The corollary predicts a zone about 2610 steps wide (SD
0.0510, slope +2.80e-05 per step, n=10). The observed zone occupies the
2000 and 3000 slices and not 1000 or 4000, so on the 1000-step grid its
width lies between 1000 and 3000 steps: consistent with the prediction
at the grid's resolution.

## What this means for THEOREM_CROSSING

Assumption (iii) was not falsified at this protocol through 12000 steps.
Both lanes did overfit, but the larger class did not overfit first, and
no second crossing appeared: the gap kept widening in exact attention's
favour. The statement is bounded to gpt2-nano, d=256, two layers, T=256,
batch 4, AdamW at 1e-3, w=64 with one sink, the TinyStories slice and
12000 steps.

## Provenance

The runs executed on three L4 sessions with the CUDA binary built once
from paperkiln `master` at launch (commit e745760; build cache
`mtstudio_cuda_l4_c32730186da0`), checkpointing every 400 steps and
resuming through the runner's relay (resume restores the full optimiser
state and is bit-identical on the CUDA path). The relayed run
directories hold `events.jsonl` and `result.json`; per-run binary hashes
were not relayed, so the build fingerprint above is the record of the
binary. Driver logs: `/mnt/c/ml_artifacts/transfer/longbudget*_driver.log`.
