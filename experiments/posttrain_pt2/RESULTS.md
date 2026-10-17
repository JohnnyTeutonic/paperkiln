# PT2 — results (17 Oct 2026)

*Licence `e10514e`. Ten seeds, run from a clean clone at that commit
(`receipts/results.json`: `repo_dirty` false; binaries sha256 `3b345a60…`
and `da8d7c4b…`, byte-identical to the calibration's). The frozen
`analyze.py` output is in `receipts/ANALYSIS.md`. Nothing was amended.*

## The registered answers

| Hypothesis | Contrast (paired, 10 seeds) | Mean | 95% CI | t | Verdict |
|---|---|---:|---|---:|---|
| H1, q = 0.10 | judge_truth_gap: J10 − R10 | +0.004 | [−0.007, +0.016] | +0.84 | not resolved |
| **H1, q = 0.25** | judge_truth_gap: J25 − R25 | +0.005 | [−0.017, +0.027] | +0.51 | **not resolved** |
| H2, q = 0.10 | truth_acc: E − J10 | +0.059 | [+0.017, +0.101] | +3.17 | supported |
| **H2, q = 0.25** | truth_acc: E − J25 | +0.097 | [+0.061, +0.132] | +6.10 | **supported** (9/10 seeds) |
| H3, q = 0.10 | truth_acc: R10 − J10 | −0.007 | | −0.55 | not resolved |
| H3, q = 0.25 | truth_acc: R25 − J25 | +0.005 | | +0.34 | not resolved |

**F-PT2 fires at both judge rates.** Rejection-sampling SFT filtered by a
judge that passes 10% or 25% of wrong answers does not teach the model to
fool that judge any more than SFT on samples filtered with the same
confusion rates placed at random:
- the judge–truth gap differs by at most +0.016 (q = 0.10) and +0.027
  (q = 0.25) at the upper ends of the 95% intervals;
- the accuracy cost of judge filtering is likewise indistinguishable from
  the cost of random filtering errors (H3).

**At this step of the pipeline and this scale, judge error behaves like
label noise.** The provenance thesis (charter §2) is not supported here.

## What the arms did (means over seeds; test set, 8 samples per prompt)

| | base | E | J10 | R10 | J25 | R25 | X |
|---|---:|---:|---:|---:|---:|---:|---:|
| truth_acc | 0.371 | **0.445** | 0.386 | 0.379 | 0.348 | 0.353 | 0.301 |
| judge_truth_gap_25 | 0.155 | 0.125 | | | 0.154 | 0.149 | |

- **Exact-filtered rejection SFT helps:** +0.074 over the base.
- **Judge-filtered SFT at q = 0.25 ends below the base.** Its noise-matched
  twin ends in the same place.
- **Accepting samples at random (X) costs the most** (E − X = +0.143,
  t = 7.30).
- The judge's measured false-accept rate on these samples was 0.237–0.267
  across seeds (set: 0.25). K, the equal arm size, was 1,264–2,560.
- **Neither hack metric moves.** `hack_frac` and `distractor_frac` change
  by less than ±0.013 against the base for every arm (descriptive, in
  `ANALYSIS.md`).

## Scope

- `tinyarith` R = 20; llama-tiny (d = 128, L = 2, about 1M parameters).
- Base: 3000 steps. One round of rejection-sampling SFT (600 steps,
  lr 1e-3); 8 samples per training fact.
- Synthetic judges with the declared rules (cue and near-miss acceptance
  at q, no false rejections).
- Says nothing about LLM judges, larger models, iterated rounds, or
  preference optimisation (PT5's question).
