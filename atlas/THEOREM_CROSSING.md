# The crossing theorem (sketch)

*Why S1c-budget-reversal looks the way it does, what it forbids, and
the one assumption that can break — which is a runnable experiment,
not a philosophical caveat. Written 31 Aug 2026 against the banked
S1c / S1e data; the long-budget test is
experiments/sparse_s1_longbudget/.*

## The object

For two attention lanes compared at training budget `b`:

    Delta(b) = L_swa(b) - L_exact(b)

on held-out loss, paired within seed. `Delta < 0` = sparse dominates,
`Delta > 0` = dense dominates. S1c registered the fact that this sign
REVERSES with `b`; S1e registered that the reversal budget is a
distribution over seeds, not a point.

## Decomposition

For a lane class `C`, write

    L_C(b) = L*_C + G_C(b)

- `L*_C` — the best held-out loss achievable in class `C` (the
  approximation term). A property of the class and the data, not of
  the run.
- `G_C(b) >= 0` — everything the finite run has not yet closed:
  optimization progress plus estimation error. `G_C(b) -> 0` as the
  run approaches the best the class can do.

Then

    Delta(b) = A + [G_swa(b) - G_exact(b)],     A := L*_swa - L*_exact.

## Assumption (i): NESTING — and it is a fact here, not an assumption

Sliding-window attention with window `w >= T` and `sinks = 0` computes
exactly what causal full attention computes: it visits the same range
in the same order. This repo pins that BITWISE
(`tests/test_swa.cpp`, the SWA equivalence pin). Hence the SWA family
is a subclass of the exact family, and

    A = L*_swa - L*_exact >= 0.

The asymptotic winner is therefore fixed **by construction, before any
experiment**: dense wins or ties in the limit. Any finite-budget
result in which sparse "wins" is a statement about `G`, not about
representational power.

## Assumptions (ii) and (iii)

- **(ii) Decay.** `G_swa(b), G_exact(b) -> 0`.
- **(iii) Ordered decay.** `D(b) := G_exact(b) - G_swa(b)` is
  non-increasing in `b`.

(iii) is the substantive one. It says the *larger* class starts with
the larger unclosed gap — more parameters to fit, more variance to
average down — and that this disadvantage only ever shrinks. It is
exactly the classical bias/variance intuition, stated as a monotonicity
condition on the difference rather than on each term.

## Theorem (sketch)

*Assume (i), (ii), (iii). Then:*

1. `Delta(b) = A - D(b)` is **non-decreasing** in `b`.
2. `Delta` has **at most one sign change**, and if it has one it runs
   **sparse-favoured -> dense-favoured**, never the reverse.
3. `Delta(b) -> A >= 0`: dense wins or ties in the limit.
4. Consequently **no basin exists** — no "sparse wins, then dense, then
   sparse again" shape is admissible.

*Proof sketch.* (1) is immediate: `A` is constant in `b` and `D` is
non-increasing by (iii), so `A - D` is non-decreasing. (2) follows
because a non-decreasing function crosses any level at most once, and
the crossing direction is forced by the sign of the difference. (3) is
(ii) applied to `Delta(b) = A + G_swa - G_exact`. (4) is (2) restated.
QED (modulo the population/expectation framing below).

**Match to data.** The pooled S1e trajectory at d=256, n=15, is
monotone across all nine slices — -0.0633, -0.0524, -0.0328, -0.0213,
-0.0006, +0.0042, +0.0135, +0.0173, +0.0195 — one crossing near 2000,
no return. Four independent pre-committed replications of the
direction (S1c, boundary, Rung B, S1e).

## Corollary (the undetermined zone)

Near the crossing `|E Delta|` passes through zero while the
between-seed SD does not. With `n` seeds and critical value `t`, an
effect is detectable only when `|E Delta| > t * SD / sqrt(n)`. Since
`Delta` is differentiable in `b` with slope `s = dDelta/db` near the
crossing, the budgets at which the comparison is UNDETERMINED form an
interval of width

    W  ~=  2 t SD / ( sqrt(n) * |s| ).

Measured at d=256: `SD ~ 0.035`, `s ~ 2.6e-5` per step, `n = 15`,
`t = 2.145` gives `W ~ 1500 steps`. Observed: the zone opens between
b=1200 and b=1600 and has NOT closed by b=3600.

Three consequences worth stating plainly:

- The zone is **structural, not sloppiness**. It exists for every
  finite seed budget.
- It closes only as `sqrt(n)`: **halving the zone costs four times the
  seeds.**
- A single-seed experiment has `W` inflated by `sqrt(n)` relative to a
  15-seed one and no way to know it is inside the zone. This is the
  seed lottery (atlas/SEED_LOTTERY.md) in closed form.

## What can break — and where to look

**(iii) must fail once the larger class overfits.** Held-out loss is
not monotone forever: past its own minimum, `G_exact` turns UPWARD
while `G_swa` (smaller class, later minimum) is still falling. Then
`D(b)` rises again, `Delta` turns back down, and a **second crossing**
appears — sparse retakes the lead not by being better but by
overfitting later.

So the theorem's falsifier is sharp and cheap:

> **A second sign change of `Delta(b)` at long budget falsifies (iii),
> and it should appear if and only if the exact lane reaches its
> held-out minimum at a smaller budget than the swa lane.**

That is a conditional prediction with real content — it names both the
consequence AND the condition, so it can fail in two distinguishable
ways. It is pre-registered and under test in
`experiments/sparse_s1_longbudget/` (12000 steps, ~30 epochs of the
corpus, both lanes, 10 seeds).

## Scope and honesty

This is a statement about EXPECTED held-out loss under a decomposition,
not a finite-sample guarantee about any single run — per-seed
trajectories are noisy and S1e shows their crossings scatter across
1600->never. It assumes the comparison is between nested classes; it
says nothing about non-nested mechanism swaps (Kimi, SSM, gating),
where `A` has no sign by construction and a basin IS admissible. And
it is a sketch: (1)-(4) are elementary once the decomposition is
granted, and the decomposition is the modelling step, not a derivation.

The contribution is not the algebra. It is that the algebra **forbids a
shape** (the basin), **predicts a specific reversal condition** (larger
class overfits first), and **quantifies the zone** where no experiment
of a given size can answer the question at all.
