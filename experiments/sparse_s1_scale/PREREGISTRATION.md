# Pre-registration — sparse_s1_scale (G1 ladder, Rung B)

## EXECUTION NOTE (12 Aug 2026, PRE-DATA — venue only, design unchanged)

Colab execution abandoned after two loss events established a hard
~2-hour session lifetime for CLI tunnel sessions (with or without
keep-alive interaction; see the colab-sweep memory record): a 3.2-hour
cell cannot complete in a 2-hour session. The 10 runs of the amended
design execute LOCALLY on CPU via mtsweep, same spec, same seeds, same
analysis. Zero results existed at this note's commit; licences
unaffected.

## AMENDMENT 1 (12 Aug 2026, PRE-DATA) — width-only re-scope

The original Rung B below (d=256, **layers=4**) is not implementable
in the current engine, discovered at pre-flight on all four Colab
shards before any run produced any number: family=gpt2 pins
layers==2 (`atlas_taxonomy.py`), `mtstudio.cpp` throws for swa at
layers!=2, and FlexLM carries no window/sinks fields. The deep-SWA
capability is now a named engineering item (ROADMAP.md item 2a).

**Re-scope: Rung B becomes WIDTH-ONLY — d=128 -> 256 at layers=2,
heads=8, all else unchanged.** Zero data existed when this amendment
was committed (the shard logs show pre-flight SystemExit; 0/10
result.json), so H-SCALE-SHRINK and its committed direction carry
unchanged to the amended design, and the one-tailed licence is
re-anchored to THIS commit, which still predates every run. The
depth axis joins Rung C as explicitly gated engine work; no depth
claim is licensed by this experiment. The ladder table below is
retained unedited as the original registration; read "Rung B" as
d=256/layers=2 throughout the analysis.

Precedent for the axis: S3-d-unpaid walked width (d=192) at 2 blocks
under the same protocol family.

---

*Written and committed BEFORE any run. 12 Aug 2026. This is the first
rung of the Stage 4 scale ladder that PAPER_PLAN.md G1 names as
blocking: scale becomes an axis, not a constant, per
ARCHITECTURE_ATLAS §18.*

## What is being scaled, and why this claim first

`S1c-budget-reversal` (supported, t=+5.09, n=5 paired) is the paper's
central case study: at d=128/2-block, the sliding-window advantage
over exact attention at 400 steps REVERSES by 1200 steps, so the sign
of a sparse-vs-dense comparison depends on the training budget. The
first referee question (G1) is whether that behaviour is a tiny-scale
artefact. Rung B re-measures the same paired contrast at 8x the
compute scale.

## Rung definition

| Axis | Rung A (done, CPU) | Rung B (this prereg) | Rung C (GATED) |
|---|---|---|---|
| d | 128 | **256** | 512 |
| layers | 2 | **4** | 6-8 |
| heads | (nano default) | **8** (head_dim 32) | 8-16 |
| T | 256 | **256** (held) | 512 |
| corpus | TinyStories slice | same | same+ |

T is deliberately HELD at 256: one factor family (capacity) moves per
rung. Sequence-length scaling is a separate axis. **Rung C is gated
on CUDA kernels landing past the dispatch seam; it is out of scope
here and no claim about it is made.**

d_ff and any dims not set explicitly resolve server-side in mtstudio;
they are recorded from the `model` event at run start. The analysis
depends only on their being IDENTICAL across lanes, which the shared
spec guarantees.

## Cells

2 lanes x 5 seeds = **10 runs**, 1200 steps each, eval_every=100.
- Lane EXACT: full causal attention.
- Lane SWA: window 64, sinks 1 (identical to S1/S1c configuration).
- Seeds 1-5, paired across lanes (same seeds as S1c rows).
- Data/optimiser: TinyStories slice, vocab_cap 4096, batch 4,
  AdamW lr 1e-3 (the S3-lrxopt safe quadrant), T=256.
- Delta(b) := best-val-by-step-b(SWA) - best-val-by-step-b(EXACT),
  read from the eval trace at b=400 and b=1200, per seed, exactly as
  S1c computed it.

## Hypotheses and decision rules (committed direction)

**H-SCALE-SHRINK (primary).** The budget effect recurs in sign at
Rung B: Delta(1200) - Delta(400) > 0 (the sparse position worsens
with budget), paired across seeds.
- Test: paired t over the 5 per-seed values of
  [Delta(1200) - Delta(400)].
- The ONE-TAILED reading at alpha=0.05 (crit t=2.132, df=4) is
  licensed by THIS COMMIT, which predates the runs and fixes the
  direction. Two-tailed value reported alongside regardless.
- FALSIFIER: t below one-tailed crit, or sign against, and the
  budget-conditionality claim is SCOPED TO d=128 in FINDINGS.md and
  the paper reports a scale boundary at Rung B. That outcome is
  publishable content, not failure.

**D-SIGNS (secondary, DESCRIPTIVE ONLY - no committed direction).**
Whether Delta(400) < 0 (early sparse advantage) and Delta(1200) > 0
(late exact advantage) individually recur at Rung B. Reported as
estimates with per-seed values; no inferential entitlement is claimed
beyond H-SCALE-SHRINK. The reversal POINT may move with scale in
either direction; wherever it lands is measurement, not hypothesis.

**Trend framing (atlas doctrine).** The deliverable row reports the
Rung A -> Rung B TREND of the budget effect, not a new point claim.

## Execution

- Colab via the sharded runner `colab_shard.sh` (this directory):
  builds microtorch (cmake, CPU), executes assigned runs via
  `mtsweep.py --jobs 1`, relays a results zip after every run,
  touches `SPARSE_S1_SCALE_DONE` when its shard is complete.
- 4 shards (session cap), runs assigned round-robin by index.
- RESUME GRANULARITY: per RUN (mtsweep skips runs whose result.json
  exists). A VM reclaim costs at most the in-flight run.
- Runtime estimate: Rung A ran ~2-3 s/step on Colab-class CPU at
  d=128/2L/T=256; Rung B is ~8x parameters-compute => ~16-24 s/step
  => ~5.5-8 h per 1200-step run; 10 runs over 4 shards => roughly
  15-24 h wall-clock. The shard script logs measured s/step in its
  first chunk so the estimate is checked within the first hour.
- No number from any run enters FINDINGS.md except through
  analyze.py over the relayed result artifacts.

## What this prereg does NOT license

- No claim about Rung C or about deployment-scale models.
- No mechanism story for any observed reversal (S1b's amendment
  stands as the cautionary precedent).
- No reuse of these runs for window-ladder (S1b-type) claims; a
  window ladder at Rung B would be its own pre-registration.
