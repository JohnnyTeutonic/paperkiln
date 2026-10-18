# PT5 — results (18 Oct 2026)

*Licence `3e303bf`. Ten seeds (301–310) from a clean clone at that commit
(`repo_dirty: false`; mtstudio sha256 `efc993cc…`, posttrain_verify
`da8d7c4b…`). An external WSL restart interrupted seeds 309–310; they were
rerun deterministically after deleting their incomplete run directories
(execution clarification 1 in the pre-registration). The frozen
`analyze.py` ran unchanged; its output is in `receipts/ANALYSIS.md`.
Nothing was amended.*

## The registered answers

| Rule | Contrast (paired, 10 seeds) | Mean | SE | t (crit 2.262) | Verdict |
|---|---|---:|---:|---:|---|
| F-PT5a | gap(J) − gap(R) | +0.0151 | 0.0097 | +1.55 | **NOT_RESOLVED** (signs +7/−2) |
| F-PT5b(i) | gap(Mu) − gap(Mw) | −0.0011 | (SD 0.0235) | −0.15 | fails |
| F-PT5b(ii) | truth(Md) − truth(Mw) | −0.0000 | (SD 0.0542) | −0.00 | holds |
| F-PT5b | | | | | **WEIGHTING_INERT** |

gap = `judge_truth_gap_25`, the judge's mean reward minus exact-match
accuracy on the same test samples.

- **F-PT5a, the thesis: not supported at this scale.** DPO on
  judge-ordered pairs left the judge-truth gap 0.015 higher than DPO on
  noise-matched pairs, with the sign positive in 7 of 9 nonzero seeds, but
  the paired t of 1.55 is short of 2.262.
- **F-PT5b, the mechanism: weighting by evidence did nothing.** The
  provenance weights (exact 1.0, judge 0.3) did not lower the gap
  relative to equal weights, and weighting and dropping the judge half
  reached the same truth accuracy. By the pre-registered rule this retires
  the default weight table: `train.po.weight_mode` now defaults to
  `uniform`, and `provenance` stays available by name.

## Per seed

| seed | 301 | 302 | 303 | 304 | 305 | 306 | 307 | 308 | 309 | 310 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| N pairs | 1580 | 1748 | 1684 | 1824 | 1666 | 1730 | 1632 | 1776 | 1796 | 1556 |
| f_tie | 0.063 | 0.166 | 0.273 | 0.317 | 0.139 | 0.140 | 0.393 | 0.175 | 0.169 | 0.060 |
| gap base | 0.129 | 0.149 | 0.140 | 0.144 | 0.131 | 0.160 | 0.168 | 0.170 | 0.157 | 0.114 |
| gap J | 0.121 | 0.114 | 0.205 | 0.175 | 0.224 | 0.203 | 0.235 | 0.172 | 0.175 | 0.146 |
| gap R | 0.157 | 0.093 | 0.207 | 0.175 | 0.188 | 0.119 | 0.222 | 0.166 | 0.151 | 0.140 |
| gap Mu | 0.095 | 0.149 | 0.233 | 0.190 | 0.157 | 0.147 | 0.196 | 0.181 | 0.185 | 0.164 |
| gap Mw | 0.144 | 0.153 | 0.254 | 0.151 | 0.147 | 0.134 | 0.201 | 0.175 | 0.172 | 0.177 |
| truth base | 0.474 | 0.437 | 0.343 | 0.306 | 0.424 | 0.390 | 0.241 | 0.347 | 0.300 | 0.517 |
| truth E | 0.412 | 0.364 | 0.271 | 0.353 | 0.379 | 0.388 | 0.285 | 0.250 | 0.265 | 0.362 |
| truth J | 0.349 | 0.381 | 0.282 | 0.284 | 0.330 | 0.257 | 0.149 | 0.271 | 0.215 | 0.304 |
| truth R | 0.330 | 0.351 | 0.224 | 0.263 | 0.321 | 0.382 | 0.194 | 0.336 | 0.259 | 0.293 |
| truth Mw | 0.332 | 0.269 | 0.226 | 0.306 | 0.422 | 0.299 | 0.218 | 0.323 | 0.267 | 0.291 |
| truth Md | 0.349 | 0.239 | 0.159 | 0.304 | 0.362 | 0.375 | 0.297 | 0.278 | 0.246 | 0.343 |

The inverted fraction f_inv was 0 in nine seeds and 0.0006 in one, so
almost every J ordering error is a tie: a wrong answer the judge accepts,
preferred over another wrong answer. R reproduces that tie rate with the
ties chosen at random.

## Descriptive: the preference stage lowered truth in every arm

Not a registered contrast; computed from `receipts/results.json`.

| arm vs base | E | J | R | Mu | Mw | Md |
|---|---:|---:|---:|---:|---:|---:|
| truth, mean change | −0.045 | −0.096 | −0.082 | −0.099 | −0.083 | −0.083 |
| t (paired, df 9) | −2.28 | −5.80 | −3.87 | −4.92 | −3.33 | −3.06 |
| seeds below base | 8/10 | 10/10 | 10/10 | 10/10 | 9/10 | 9/10 |

Mean truth: base 0.378, E 0.333, J 0.282, R 0.295. Every arm kept the
answer format (answer_rate 0.990 or more), so this is a loss of
arithmetic, not of format. Even exact-ordered pairs (E) cost accuracy.
The fixed recipe (β = 0.1, α = 0.2, lr 1e-3, 400 steps, chosen by the
pilot for format retention only) is too strong for these models. Two
consequences follow.

- **F-PT5a compares two damaged models.** J and R both sit 0.08–0.10
  below the base on truth. A rise in the gap partly mirrors the fall in
  truth, since the gap is judge reward minus truth. The registered
  contrast is unaffected (J and R share the recipe), but it is measured
  in a regime where preference optimisation is harming the model.
- **E − J on truth is +0.051 (t 2.89).** Exact-ordered pairs hurt less
  than judge-ordered ones. This is the clearest effect in the study,
  descriptive only.

## Scope

- `tinyarith` R = 20; llama-tiny d = 128, L = 2; base 3000 steps.
- DPO + RPO NLL: β = 0.1, α = 0.2, lr 1e-3, batch 8 pairs, 400 steps;
  N = 1556–1824 pairs per seed, equal across arms within a seed.
- One synthetic judge profile (q = 0.25, the PT2 fallback). Nothing here
  speaks to LLM judges or to recipes that do not lower truth.
- WEIGHTING_INERT is about the frozen table (exact 1.0, judge 0.3) in a
  half-exact, half-judge pool. It does not show that no weighting helps.
