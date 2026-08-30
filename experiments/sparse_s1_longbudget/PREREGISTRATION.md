# Pre-registration — sparse_s1_longbudget: does the zone close, and is there a second crossing?

*Written and committed BEFORE any run. 31 Aug 2026. The licence anchor
for every directional claim below is THIS commit, which also carries
`analyze.py`. This experiment is the falsification test of
atlas/THEOREM_CROSSING.md assumption (iii).*

## Why

Every S1 result so far stops at 3600 steps. Two things are therefore
unmeasured, and both are load-bearing:

1. **The undetermined zone never closed.** At d=256, n=15, the zone
   (|mean Delta| below the detection threshold) opens between b=1200
   and b=1600 and is still open at b=3600. We have never observed its
   far edge, so "dense eventually wins" is a limit statement we have
   not seen.
2. **The crossing theorem forbids a second crossing — under an
   assumption that must fail eventually.** Assumption (iii) (the
   gap difference `D(b) = G_exact - G_swa` is non-increasing) cannot
   hold past the point where the LARGER class overfits: there
   `G_exact` turns up while `G_swa` is still falling, `Delta` turns
   back down, and sparse retakes the lead by overfitting later rather
   than by fitting better.

## Design

Identical to sparse_s1_boundary/seeds in every respect except budget:
gpt2-nano, d=256, heads=8, layers=2, T=256, batch=4, lr=1e-3,
TinyStories slice + chat7b vocab (cap 4096). **steps = 12000**
(~30 epochs of the ~400k-token corpus — comfortably past the overfit
onset the 3600-step runs never reached), eval_every = 200 (60 evals).
Two lanes, exact and swa(w=64, sinks=1). **10 seeds {41..50}** —
new seeds, never used in any S1 experiment. Venue CUDA (Phase B,
adopted 31 Aug at 30.5x; this experiment is only affordable because of
it — 20 runs x 12000 steps is ~90 CPU-hours and ~5 GPU-hours).

Budget choice rationale, fixed now: 12000 rather than 20000 keeps a
single run at ~16 min, well under the observed Colab reclaim interval,
which is the supervisor admission rule (a reclaim must cost minutes,
not the run). If H-OVERFIT-ORDER (below) shows NEITHER lane has turned
up by 12000, the honest report is "onset not reached at 12000" and the
budget question is escalated in a NEW pre-registration — not extended
by patching this one.

## Hypotheses and decision rules (fixed now)

**H-OVERFIT-ORDER (committed direction, one-tailed).** The exact lane
reaches its held-out minimum at a SMALLER budget than the swa lane:
per-seed `argmin_b L_exact(b) < argmin_b L_swa(b)`. Paired sign test
over 10 seeds, one-tailed, reject at >= 8/10 (binomial p = 0.055 at
8/10, 0.011 at 9/10; the 8/10 line is fixed here). This is the
CONDITION half of the theorem's conditional prediction.

**H-SECOND-CROSS (committed direction, conditional).** `Delta(b)`
exhibits a second sign change (positive -> negative) at some
b in (B\*, 12000]. Detection rule fixed now: the pooled mean
`Delta(b)` must be positive at some slice, then negative at a later
slice, with BOTH excursions clearing the detection threshold
`t * SD(b) / sqrt(10)` (t = 2.262, two-tailed df=9) at their
respective slices. A sign wobble inside the threshold is NOT a
crossing and will be reported as "within the zone".

**The theorem's conditional prediction, stated as a joint outcome
table (fixed before data):**

| H-OVERFIT-ORDER | H-SECOND-CROSS | reading |
|---|---|---|
| supported | supported | **Theorem's escape hatch confirmed**: (iii) fails exactly where predicted; the reversal is an overfitting-order effect. Strongest outcome. |
| supported | not supported | Condition holds, consequence absent within 12000 — (iii) survives longer than the mechanism predicts; report the gap and the budget bound. |
| not supported | supported | A second crossing NOT explained by overfitting order — the decomposition is missing a term. Most interesting negative; would send the theorem back. |
| not supported | not supported | Monotone regime simply extends; the theorem stands unfalsified to 12000 and the zone question (below) carries the report. |

**Z-CLOSE (descriptive, pre-committed criterion).** For each slice,
the zone indicator `|mean Delta(b)| <= t * SD(b) / sqrt(10)`. Report
the first slice at which the zone CLOSES (indicator false and staying
false for all later slices up to any second crossing). If it never
closes by 12000, that is the reported result: **"the comparison remains
statistically undetermined through 12000 steps"**, which is itself a
finding about the cost of answering this question at all.

**W-CHECK (descriptive).** Compare the observed zone width against the
corollary's prediction `W ~= 2 t SD / (sqrt(n) |s|)`, using the
measured local slope `s` at the first crossing. Agreement supports the
corollary's use as a design tool (how many seeds does a given
resolution cost); disagreement bounds it.

## Threat checks (fixed now)

1. **Refuse-to-run guard.** Every run's model event must record d=256,
   layers=2, and its lane's attention fields; lanes are read from model
   events, never directory names; receipts must pass
   `tools/validate_events.py`.
2. **Venue guard.** This is the first S1 experiment run on CUDA. The
   transfer_s1 numerics bridge (same protocol, d=256, CPU-vs-CUDA on
   the banked seeds) gates it: if that bridge FAILS, this experiment's
   results are quarantined until the discrepancy is resolved, because
   they would then be measuring the engine.
3. **Overfit reality check.** At least 8/10 seeds must show a
   held-out minimum strictly before the final eval in AT LEAST ONE
   lane; otherwise the run did not reach the regime it was built to
   probe and H-SECOND-CROSS is reported as untested rather than
   unsupported.

## What this cannot show

One family, one corpus, layers=2, d=256, w=64, fixed lr. A second
crossing here would falsify assumption (iii) *at this protocol*, not
in general; its absence bounds the monotone regime to 12000 steps at
this scale, not forever. Nothing here speaks to non-nested mechanism
comparisons, where the theorem does not apply at all (A has no sign)
and a basin IS admissible.

## Execution

`sweep.json` beside this file; runner
`tools/colab_transfer_runner.py` (reclaim-resilient, binary cached,
one run relayed at a time). Receipts copy into `receipts/` here on
completion. Analysis `analyze.py`, committed with this file, run only
after all 20 result.json exist.
