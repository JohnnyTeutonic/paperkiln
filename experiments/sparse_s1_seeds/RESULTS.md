# sparse_s1_seeds — RESULTS (30 Aug 2026)

*Analysis run by `analyze.py` exactly as committed with
PREREGISTRATION.md (licence commit caabe19, 28 Aug 2026), only after
all 20 result.json existed. No deviations. Receipts for all 20 new
runs in `receipts/` (validated by `tools/validate_events.py`, 20/20
pass); driver.log, atlas_rows.jsonl, cells.jsonl included. mtsweep
exit 0, 2026-08-30 18:24 AEST.*

## Verdict

**B\*(256) is a DISTRIBUTION at this protocol's resolution.** The
pre-committed V-DISPERSION criterion was met on both prongs, and the
pooled n=15 sign test failed to resolve — fifteen seeds could not
settle a sign that three separate five-seed panels each leaned on.
The crossover budget is a property of the (protocol, seed) pair, not
of the protocol alone.

And the budget-conditionality direction — committed pre-data for the
**fourth** time, on ten seeds that had never been run — replicated
decisively (t = 4.13, clearing even the unlicensed two-tailed bar).

## Threat checks (gates first, as pre-registered)

- **Threat 1 (pooling licence):** new-seed Delta_s(1200) negative
  **10/10** (required ≥ 7/10; boundary + Rung B precedent 5/5).
  Pooling LICENSED → primary analyses on n=15.
- **Threat 2 (regime):** best_val reached ≥ 3 evals before 3600 in
  only **1/15** seeds per lane — i.e. 14/15 were still improving at
  cutoff. The pre-registered "up to the overfit onset" scoping is
  replaced by the stronger fact: **the entire result sits inside the
  still-training regime; overfit onset was not reached.**
- **Threat 3 (refuse-to-run):** all 20 model events carry d=256,
  layers=2, and their lane's attention field; lanes read from model
  events only.

## H-SHRINK-4 (new seeds {6..15} only; one-tailed licensed by caabe19)

shrink2_s = Delta_s(3600) − Delta_s(1200), per seed:
+0.0693, +0.0325, +0.0044, +0.0893, +0.0533, +0.0968, +0.0943,
−0.0072, +0.0982, +0.0103 → 9/10 positive.

mean **+0.0541**, t = **4.13**, df = 9 — clears the licensed
one-tailed crit 1.833 AND the two-tailed 2.262.
**SUPPORTED — fourth independent pre-committed replication** (after
S1c-budget-reversal, sparse_s1_boundary, Rung B).

## H-CROSS-15 (pooled n=15; two-tailed only, no direction committed)

Delta(3600) = **+0.0195**, SD 0.0377, t = **2.00** < crit 2.145.
**Sign undetermined at n=15** → per the pre-registration,
V-DISPERSION carries the report.

## V-DISPERSION (pre-committed criterion — both prongs required)

- |mean| = 0.0195 **<** SD = 0.0377 ✓
- minority sign 6/15 **≥** 4/15 ✓ (split 9+ / 6−)

Per-seed persistent-crossing budget b0_s (smallest b with
Delta_s(b′) > 0 for all b′ ≥ b):

| s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 | s10 | s11 | s12 | s13 | s14 | s15 |
|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
| 3200 | — | — | 2000 | 1600 | 2400 | — | — | 2400 | 3600 | 2000 | 2400 | — | 2800 | — |

Six of fifteen seeds **never cross by 3600**; the nine that do are
spread across 1600–3600. **READING ADOPTED: B\*(256) is a
distribution over seeds at this protocol's resolution.**

## D-SHAPE (descriptive, n=15 per slice)

| b | 400 | 800 | 1200 | 1600 | 2000 | 2400 | 2800 | 3200 | 3600 |
|---|-----|-----|------|------|------|------|------|------|------|
| mean Δ | −.0633 | −.0524 | −.0328 | −.0213 | −.0006 | +.0042 | +.0135 | +.0173 | +.0195 |
| SD | .0179 | .0296 | .0214 | .0228 | .0334 | .0415 | .0330 | .0405 | .0377 |

The mean marches monotonically upward across every slice (the shrink,
again) while the SD roughly doubles — the between-seed fan-out that
makes the sign unresolvable is itself budget-dependent. At b=400 all
15 seeds agree (SWA ahead, SD tight); by b=3600 the panel is 9/6.
**Early unanimity is the cheap, wrong signal; dispersion is what
budget buys.** This is the seed-lottery exhibit's mechanism, now at
n=15 under pre-registration.

## What this licenses (and doesn't)

Scope: TinyStories slice, gpt2-nano family, d=256, L=2, w=64+1 sink,
T=256, batch 4, lr 1e-3, CPU numerics, house protocol, still-training
regime. B\*(d) here is a property of THIS protocol. No claim about
published sparse-attention results at scale.

Programme consequences:
1. Registry row **S1e-bstar-distribution** (supersedes nothing;
   completes the arc S1 → S1b → S1c → boundary → here).
2. Any future single-number "crossover budget" claim, ours or
   anyone's, is under-specified without a seed distribution — this is
   now a measured fact at n=15, not a methodological preference.
3. The transfer study (experiments/transfer_s1) inherits its power
   inputs from this file: SD(Δ3600) = 0.0377, SD(Δ1200) = 0.0214,
   sign split 9/6 at 3600, b0 spread 1600→never. Fingerprint layer F3
   must treat B\* as a per-seed distribution at BOTH scales — a
   point-to-point B\* comparison is already refuted at the base scale.
