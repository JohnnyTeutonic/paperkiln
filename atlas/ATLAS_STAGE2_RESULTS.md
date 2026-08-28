# Atlas Stage 2 — the Plackett–Burman screening experiment

**Status: COMPLETE, 2026-08-01.** The first science through the Atlas
infrastructure: 36/36 runs, 7 factors screened, main effects with
seed-based standard errors. Raw receipts in
[`experiments/atlas_stage2/`](experiments/atlas_stage2/) (design
manifest, all 36 Atlas rows, per-cell aggregates, generated
main-effects report).

## Design

- **Plan:** PB12 design (`tools/mtsweep.py --design pb12`), the
  balanced orthogonal 12-row matrix verified by the mtsweep selftest.
  7 factors assigned; 12 cells × 3 seeds = **36 runs**.
- **Factors (low → high):** `d` 128→192, `layers` 2→4, `heads` 4→8,
  `T` 128→256, `lr` 1e-3→3e-3, `batch` 2→4, `optimizer` adamw→muon.
- **Base:** llama-tiny family, TinyStories corpus, GGUF vocab
  (cap 4096), 400 steps, no early stopping. Params ranged 0.95M–2.7M
  across cells (not parameter-matched — a screen limitation noted
  below).
- **Compute:** 6.9 h of serial CPU packed onto 2 workers overnight
  (OMP split 4+4, WAIT_POLICY=PASSIVE — the oversubscription lesson).
- **Analysis:** `tools/atlas_analyze.py` — main effect per factor =
  mean(high) − mean(low) over 18-vs-18 runs, SE from per-cell seed
  variance, |t| ≥ 2 flagged as a screen signal. The analyzer's
  selftest recovers a planted −0.5 effect and keeps a null factor
  null.

## Findings

Full tables: [`experiments/atlas_stage2/main_effects.md`](experiments/atlas_stage2/main_effects.md).

**1. The optimizer is the strongest factor in the screen — Muon buys
quality and pays in throughput.** Muon improves `best_val` by −0.34
(t = −6.2; level means 3.96 vs 4.30) and dominates the whole-run loss
integral `loss_auc_norm` at **t = −10.8**, the largest |t| anywhere in
the experiment. The cost side is measured, not asserted: −148
tokens/sec (t = −2.4) from the Newton–Schulz iterations. At this
scale the trade is clearly worth it — Muon's quality edge is 6–10
sigma while its speed cost is ~2.4.

**2. T = 256 improves final loss (t = −2.8) — with an honest alias.**
At fixed step count, doubling T doubles tokens seen per step, so this
effect bundles "longer context" with "more data". A screen can't
separate them; a token-matched follow-up can.

**3. Learning rate decouples speed from quality.** lr = 3e-3 reaches
the half-gap ~7 steps sooner (t = −8.8, the second-strongest signal)
**and triples gradient spikes** (t = +3.5), yet shows **no**
`best_val` advantage (t = +1.2, trending toward 1e-3). Fast early
descent is not better final loss, and the instability is visible in
the tape's own metrics. This is the behavioural-vector idea working:
one run, several axes, different verdicts.

**4. d = 192 is pure cost at this budget.** −280 tokens/sec
(t = −6.2), half-gap reached 4 steps sooner (t = −2.9), but the
`best_val` trend *favors d = 128* (t = +1.6, not significant).
Capacity accelerates early descent and hasn't paid for itself by step
400. A longer-budget run is the test of whether this inverts — the
classic small-scale/large-scale crossover, now with a concrete local
handle.

**5. The nulls are findings too.** `heads` (4 vs 8) is null on every
metric — |t| ≤ 0.5 across best_val, AUC, half-gap, spikes,
throughput. `layers` is null on all quality metrics and only shows
its throughput price (t = −3.5). At 400-step tiny scale, head count
simply does not matter; the Atlas records that instead of assuming
it.

**6. One metric-artifact candidate, flagged.** `batch` = 4 raises
`grad_spike_count` (t = +3.1). Stacked rows change the gradient-norm
scale, and the spike detector thresholds on those norms — so this may
be the metric, not the physics. Marked for a normalization fix in
`atlas_extract.py` before Stage 3 leans on that column.

**ADDENDUM 2026-08-01 — flag confirmed, metric fixed, effect resolved.**
Diagnosis on this corpus's raw events: **~85% of all >3×-median
exceedances sat in the first 5% of steps** (median spike position: the
1% mark) — the metric was measuring the *initialization transient*, not
training instability. `atlas_extract.py` now splits them:
`grad_spike_count` counts post-warmup steps against the post-warmup
median, and the new `grad_init_transient` (max warmup grad / post-warmup
median) carries the init story. Re-analysis of these same 36 runs
([spike_metric_v2.md](experiments/atlas_stage2/spike_metric_v2.md)):

- the batch→spike effect **vanishes exactly** (t = +3.13 → **0.00**) —
  finding #6 was pure init-transient artifact;
- batch reappears where it belongs: `grad_init_transient` t = +2.35
  (larger batches start more violently relative to steady state), with
  lr (t = +3.2), d (t = +3.0) and layers (t = +2.1) alongside;
- **lr survives on the cleaned instability metric** (t = +2.45), so
  finding #3's "3× the gradient spikes" stands — now measured on
  post-warmup steps only — and T = 256 shows a mild post-warmup spike
  cost (t = +2.19) worth watching in Stage 3.

The original tables above are left as reported; the corrected spike
columns live in the v2 file. Stage 3 extraction uses the fixed metric
from the start.

## Cell ranking vs main effects — the seed-lottery lesson, quantified

The per-cell aggregate says the **top-2 cell gap (0.021) is inside
mean seed noise (0.027)** — ranking individual cells is *not*
supported by this data. The main effects above are signal anyway,
because each is an 18-vs-18 contrast that averages seed noise down by
√18 per side. That is the entire argument for designed experiments
over leaderboard-style best-cell hunting, demonstrated in our own
data on night one.

## Caveats

- A screen estimates **main effects only**; two-way interactions are
  deliberately aliased and unmeasured. Stage 3 (resolution-V) exists
  for exactly that.
- One corpus, one family (llama), 400 steps, tiny scale. No claim
  transfers upward until the Stage 4 scale ladder says so.
- Cells are not parameter-matched (0.95M–2.7M); `best_val` effects
  for d/layers partially reflect parameter count. Optimizer, lr, and
  batch comparisons are parameter-clean (same models either side).
- Seeds n=3 per cell. SEs carry that honestly; nothing here rests on
  a single seed.

## What Stage 3 should do

Survivors by quality-relevance: **optimizer, T, lr, d** (batch enters
only through the suspect spike metric). A resolution-V design on
these four (2^4 = 16 cells full factorial is actually affordable —
better than fractional here) × 3 seeds at **1200+ steps** with
**token-matched T levels** would answer the three open questions this
screen created: does d invert with budget, does the lr quality gap
appear or stay null, and does Muon's edge persist when everything
else moves.
