# Pre-registration — registry #0001 pilot (Highway Networks, 2-block rung)

*Written and committed BEFORE any run. 12 Aug 2026. Dossier:
`registry/0001_highway_networks/ENTRY.md`. This is the archeology
registry's pilot: one historical mechanism, one protocol, one
pre-registered contrast, one Atlas row.*

## Design: three lanes, parameters matched

| Lane | residual | d_ff | params vs residual lane |
|---|---|---|---|
| RESIDUAL | residual | 512 | reference |
| HIGHWAY | highway (gate_bias_init −2) | 384 | **+256** (documented) |
| PLAIN | plain | 512 | equal |

Param matching: at d=128, the highway gates cost
2 layers x 2 gates x (128^2 + 128) = 66,048 parameters; reducing the
GELU MLP's d_ff by 128 saves 2 x (2*128*128 + 128) = 65,792. The
residual mismatch of +256 parameters (0.03%) is reported exactly, not
hidden. PLAIN shares the residual lane's architecture minus the skip,
so its parameter count is identical to RESIDUAL.

## Protocol (atlas house protocol, identical to the S1 family)

gpt2-nano dims (d=128, layers=2, heads=4), T=256, TinyStories slice,
vocab_cap 4096, batch 4, AdamW lr 1e-3, 1200 steps, eval_every 100,
seeds 1-5 paired across lanes. Exact attention. Execution: local CPU
via mtsweep (out_root on ext4 ~, receipts copied into the repo per
the /tmp-volatility rule).

## Questions and decision rules

**ESTIMATION FRAMING, stated up front: no direction is committed and
no one-tailed licence exists.** Neither the 2015 paper nor our priors
justify a directional bet on gating vs the hardwired residual at this
scale; the pilot MEASURES. All tests two-tailed, paired t, df=4,
crit 2.776.

- **Q1 (primary): Delta1 = bestval(HIGHWAY) − bestval(RESIDUAL)** at
  1200 steps, per seed, paired. Also read at 400 steps (the S1c
  budget-conditionality lesson is baked in: both budgets reported,
  neither privileged).
- **Q2 (secondary, descriptive): Delta2 = bestval(PLAIN) −
  bestval(RESIDUAL)** — what any skip connection is worth at this
  scale. Expected large and positive (skips help); reported as
  estimate.
- **Mechanism observable: per-layer mean transform-gate activation**
  E[T] over the eval batch, from initialisation (sigma(−2) ~= 0.119)
  to step 1200. Does the network open its gates, and how far?
  Descriptive only; extraction from checkpoints/events if available,
  else omitted with a note (no protocol change mid-run).

**Pre-declared readings:**
- |t| > 2.776 on Delta1: the registry row records the sign and size of
  the gating effect at this protocol and scale.
- |t| <= 2.776 on Delta1: the row records "no resolvable difference
  between learned gating and the residual prior at this protocol and
  scale" — a genuine registry verdict, not a failure.
- Every verdict is scoped: at THIS protocol, THIS scale, as a trend
  candidate for the ladder (depth-4/6 rung is future work, already
  capability-ready). No claim about the 2015 paper's regime.

## What this prereg does not license

- No depth claims (2 blocks only here).
- No gate-bias-init sensitivity claims (a bias-init cell would be its
  own registration).
- No mechanism story beyond the reported E[T] trajectory.
