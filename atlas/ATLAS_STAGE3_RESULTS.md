# Atlas Stage 3 — the full factorial, and the interaction that unmasks Stage 2

**Status: COMPLETE, 2026-08-03.** 48/48 runs: a full 2⁴ factorial on the
Stage-2 survivors {optimizer, lr, d, context} × 3 seeds at 3× the
Stage-2 token budget, with **token-matched context** (T=128×1200 steps
vs T=256×600 steps, both 614,400 tokens at batch 4) — the de-aliasing
fix Stage 2's finding #2 demanded. Receipts in
[`experiments/atlas_stage3/`](experiments/atlas_stage3/) (manifest, 48
Atlas rows, 16 cell aggregates, generated effects + interactions
report). Interactions are unconfounded — that was the point of paying
for the full factorial.

## Findings

**1. THE HEADLINE — lr × optimizer is a real interaction, and it
rewrites a Stage 2 conclusion.** Signal on three metrics at once:
best_val t = −3.12, loss-AUC t = −2.69, post-warmup spikes t = −4.01.
Reading the cells: **lr = 3e-3 is the best setting under Muon and the
worst under AdamW.** The four best cells in the experiment are all
Muon (top two at 3e-3: 3.515, 3.524); the four worst are all
AdamW @ 3e-3 (3.976–4.184). Stage 2's screen reported "lr has zero
final-loss effect" — that main effect was two opposite conditional
effects averaging out. This is exactly what a screen cannot see and a
factorial can, and it lands on the mechanism the Muon literature
claims: **orthogonalized updates widen the stable learning-rate
range.** Our tiny-scale data reproduces it with seed-based error bars,
and the spike interaction says the *instability* side too: high lr
spikes AdamW, not Muon.

**2. Muon's main effect replicates at 3× budget.** best_val −0.31
(t = −7.4), loss-AUC t = −12.4 — same direction and magnitude as Stage
2. Two independent designed experiments, same answer. The measured
throughput cost, meanwhile, is not significant here (t = −1.6).

**3. The token-matched context null — Stage 2's "T=256 is better"
evaporates.** With tokens held equal, ctx best_val t = −1.48: no
signal. Stage 2's t = −2.77 for longer context was, as suspected,
**"more data" wearing "longer context" clothes.** The alias is dead;
this null is the payoff of the linked-factor design and is worth as
much as a positive.

**4. Capacity still hasn't paid at 3× budget.** d = 192: best_val
n.s. (trend still favors 128, t = +1.10), throughput −209 tok/s
(t = −6.6), more init transient (t = +3.64), more spikes (t = +2.18).
Stage 2 asked "does d invert with budget?" — at 3× the answer is *not
yet*. Any crossover lies beyond this budget scale (Stage 4's ladder
territory).

**5. Stability decomposition holds up.** The split metrics from the
Stage-2 addendum behave coherently here: `grad_init_transient` is
driven by lr (t = +5.07) and d (t = +3.64) — init violence scales
with step size and width — while post-warmup `grad_spike_count`
carries the lr×optimizer interaction (finding 1). One curiosity: a
d×lr interaction on tokens/sec (t = +2.90) — mild, unexplained,
recorded not narrated.

## Caveats

- **grad_spike_count is a COUNT, and token-matched ctx halves the step
  count at T=256** (600 vs 1200 steps): the ctx→fewer-spikes "signal"
  (t = −2.78) is partly mechanical. Fix queued: spikes per 1,000
  post-warmup steps before Stage 4 reads that column across budgets.
- **tokens_per_second spans two thermal regimes**: 16 runs executed
  during a 2-worker phase (with measured laptop throttling) and 32
  during 1-worker phases. The d effect (−209 tok/s, t = −6.6) is large
  and consistent with Stage 2, but treat throughput magnitudes as
  indicative, not clean — wall-clock metrics want a quiet machine.
- Cell-level ranking: top-2 gap (0.008) is again inside seed noise
  (~0.016–0.019); the *grouping* is still informative — the top four
  cells are all Muon @ T=256 — but per-cell ordering is not.
- One corpus, llama family, 2 blocks, tiny scale. Nothing here claims
  transfer upward; that is Stage 4's job.

## What this buys the programme

Stage 2 → Stage 3 is now a complete worked example of the Atlas
method: screen (main effects, one night) → factorial (interactions,
one day) → a conditional-effect discovery that the screen provably
could not have made, plus a de-aliasing null that kills a wrong
conclusion before it propagated. Stage 4 (the scale ladder) inherits
three sharp questions: where does d's crossover live, does the
Muon-widens-lr-range interaction grow or shrink with scale, and does
the context null hold when contexts get long enough to matter.
