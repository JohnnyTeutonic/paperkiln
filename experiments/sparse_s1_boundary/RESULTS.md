# sparse_s1_boundary RESULTS — 23 Aug 2026

Local CPU execution (Rung B venue, same release binary ~/mtrel/mtstudio,
same five paired seeds); receipts/ holds every run's events.jsonl +
result.json plus the driver log. Analysis run VERBATIM from the pre-data
commit (95ff4a9; execution-note amendment e57785c, also pre-data).

**Guards: ALL GREEN.**
- Threat 3 (comparability): Delta_s(1200) negatives 5/5 here vs 5/5 in
  Rung B — the extension is the same effective protocol.
- Threat 1 (regime): early-best in 1/5 (exact) and 0/5 (swa) seeds — no
  overfit scoping required; both lanes still training at 3600.

## H-SHRINK-CONT: SUPPORTED (one-tailed only)

shrink2_s = Delta_s(3600) − Delta_s(1200), per-seed
[+0.0483, +0.0110, −0.0061, +0.0871, +0.1027]
mean **+0.0486, t = 2.31, df = 4** (one-tailed crit 2.132; two-tailed
2.776 NOT cleared). The one-tailed licence was committed pre-data for
the third time; this is the **third replication** of the
budget-conditionality direction — and the weakest of the three (one
seed negative). Stated plainly: the effect recurs, with growing seed
heterogeneity at longer budgets.

## H-CROSS: SIGN UNDETERMINED AT 3600 — no B* claim licensed

Delta(3600) = **+0.0207, t = 0.90** (two-tailed crit 2.776), per-seed
[+0.0250, −0.0408, −0.0207, +0.0794, +0.0606] — 3 seeds positive,
2 negative. Per the pre-registered decision rule: **the sign at 3600 is
undetermined and no claim about B*(256) is licensed.**

## D-SHAPE (descriptive only — no test, no claim)

Mean Delta by budget slice: −0.061 (400) → −0.040 (800) → −0.028
(1200) → −0.025 (1600) → **+0.001 (2000)** → +0.001 (2400) → +0.006
(2800) → +0.018 (3200) → +0.021 (3600).

The MEAN crosses zero between b=1600 and b=2000 and drifts positive
thereafter — descriptively consistent with B*(256) ≈ 2000 against
B*(128) ∈ (400, 1200], i.e. the crossover moving right with width. But
the per-seed trajectories DISAGREE about the sign at every budget past
1600 (seed 2 remains negative through 3600; seeds 4–5 strongly
positive), which is precisely why H-CROSS came back undetermined: at
this scale **the crossover point itself appears to be seed-dependent**,
with between-seed variance growing as budget grows. A candidate framing
for the next pre-registration: B* is a distribution, not a point — and
five seeds resolve its mean poorly at d=256.

## What this does and does not license

Licensed: the budget-conditionality DIRECTION replicates at d=256 out
to 3× the original budget (one-tailed, third pre-committed direction).
NOT licensed: any statement that sparse loses (or wins) at d=256 by
3600 steps; any located B*(256). The phase-boundary programme's next
discriminating step is MORE SEEDS at the 2000–3600 budgets (variance
resolution), not a longer budget — and the depth rung remains queued
under its own future pre-registration.

Scope: TinyStories slice, gpt2-nano family, d=256, L=2, w=64+sink,
T=256, 5 paired seeds, CPU numerics, house protocol.
