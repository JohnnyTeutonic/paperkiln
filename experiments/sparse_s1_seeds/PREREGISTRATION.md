# Pre-registration — sparse_s1_seeds: B*(256) — point or distribution?

*Written and committed BEFORE any run. 28 Aug 2026. The licence anchor
for every directional claim below is THIS commit.*

## The object

sparse_s1_boundary (RESULTS.md, 23 Aug 2026) left B\*(256)
**undetermined at 3600 steps**: Delta(3600) = +0.0207, t = 0.90,
3 seeds positive / 2 negative. Its stated next discriminating step —
adopted here verbatim — is **more seeds at the 2000–3600 budgets**
(variance resolution), not a longer budget, because per-seed
trajectories disagree about the sign at every budget past 1600 and
between-seed variance grows with budget. The candidate framing named
there, now promoted to this experiment's question:

**Is B\*(256) a point five seeds resolve poorly, or is it — at this
width and protocol — better described as a DISTRIBUTION over seeds?**

## Design

**Ten NEW paired seeds {6..15}**, both lanes (exact, swa), otherwise
IDENTICAL to sparse_s1_boundary: gpt2-nano family, d=256, heads=8,
layers=2, window=64, sinks=1, T=256, batch=4, lr=0.001, steps=3600,
eval_every=100, TinyStories slice + chat7b vocab (cap 4096). Local CPU
via mtsweep (the Rung B venue — comparability is the point), out_root
`/home/jonat/seeds_out` (absolute; mtsweep does not expand `~` — the
boundary execution-note rule). 2 cells × 10 seeds = 20 runs;
`--jobs 2 --omp 4` to stay polite on a machine in active use.

Pooling: the 5 boundary seeds {1..5} at the same budgets are the SAME
effective protocol (boundary Threat 3 passed 5/5 against Rung B).
Primary analyses run on the POOLED 15 seeds; the shrink replication
runs on the 10 NEW seeds alone so it is a genuinely independent
fourth replication, not a re-test of data that already voted.

## Hypotheses and decision rules (fixed now)

**H-CROSS-15 (NO direction committed — two-tailed only):** the sign of
Delta(3600), paired t over the pooled n=15, two-tailed crit **2.145**
(df=14).
- Positive and clearing: **B\*(256) ∈ (1200, 3600]** — located; the
  phase-boundary claim gains its second width.
- Negative and clearing: **B\*(256) > 3600** — bounded below.
- Neither: undetermined at n=15 → V-DISPERSION (below) carries the
  report. Fifteen seeds failing to resolve a sign that three separate
  five-seed panels each leaned on IS evidence about dispersion, and
  will be reported as such — not as a failed experiment.

**H-SHRINK-4 (committed direction — one-tailed licensed by this
commit; NEW seeds only):** within-seed paired
`shrink2_s = Delta_s(3600) − Delta_s(1200) > 0` over seeds {6..15}.
Paired t, n=10, one-tailed crit **1.833** (df=9); two-tailed 2.262
reported alongside. The budget-conditionality direction is committed
pre-data for the FOURTH time, on seeds that have never been run.

**V-DISPERSION (pre-committed descriptive criterion — no t-test):**
over the pooled 15 seeds at b=3600, report mean, SD, and the sign
split of Delta_s(3600), plus each seed's persistent-crossing budget
b0_s (the smallest slice b such that Delta_s(b') > 0 for ALL b' ≥ b;
"none" if never). The reading **"B\*(256) is a distribution at this
protocol's resolution"** is adopted iff BOTH: |mean| < SD, AND the
minority sign holds ≥ 4/15 seeds. Otherwise the point-reading stands
and the failure to clear H-CROSS-15 (if it fails) is attributed to
power, not dispersion. Criterion fixed now so the framing cannot be
chosen after seeing the data.

**D-SHAPE (descriptive only, no test):** pooled mean Delta at every
400-step slice, n=15 per slice.

## Threat checks (fixed now)

1. **Protocol-drift guard (pooling licence):** the NEW seeds'
   Delta_s(1200) must be negative on ≥ 7/10 (boundary and Rung B were
   5/5 negative). Fewer than 7/10 negative VOIDS pooling: analyses run
   on the new 10 alone, the discrepancy is reported as its own
   finding, and no pooled claim is licensed.
2. **Regime check:** per lane over all 15 seeds, best_val achieved
   ≥ 3 evals before 3600 in ≥ 9/15 seeds scopes every statement "in
   the still-training regime up to the overfit onset".
3. **Refuse-to-run guard:** every run's model event must record
   d=256, layers=2, and its own lane's attention field; lanes are read
   from model events, never directory names.

## What this cannot show

Scope: TinyStories slice, one architecture family, w=64 only, 2
layers, CPU numerics, house protocol. B\*(d) here is a property of
THIS protocol. Fifteen seeds bound dispersion at one width; they say
nothing about whether dispersion grows with d (that is a future
question for the depth/width rungs). No claim about published
sparse-attention results at scale; the general-methods claim remains
the budget-reporting discipline.

## Execution

`sweep.json` beside this file (seeds 6..15, out_root
/home/jonat/seeds_out); runner `tools/mtsweep.py` (resumable —
completed cells skip; the driver may be killed and relaunched freely).
Analysis `analyze.py` beside this file, written and committed with
this pre-registration, run only after all 20 result.json exist. It
reads new seeds from $SPARSE_SEEDS_ROOT (default ~/seeds_out) and the
boundary seeds from $SPARSE_S1_ROOT (default ~/boundary_out).
Receipts (events.jsonl, result.json, driver.log) copy into receipts/
here immediately on completion — never left in WSL-only storage.
