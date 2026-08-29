# Registry entry 0001 (pilot #0): Highway Networks

Status: DOSSIER ONLY — no engine code exists for this entry yet. No runs.
Provenance: sections 1-2 verified against the fetched abs page and ar5iv full
text of 1505.00387 (2026-08-12). Statements about 1507.06228 are from prior
knowledge and are marked [not re-verified].

## 1. PAPER

Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber. "Highway Networks."
arXiv:1505.00387 (v1 3 May 2015, v2 3 Nov 2015). Presented at the ICML 2015
Deep Learning workshop. Follow-up full paper: "Training Very Deep Networks,"
arXiv:1507.06228, NeurIPS 2015 [not re-verified].

Claim as published: plain (unskipped) deep networks become hard to optimize
with depth. The paper introduces layers with learned gating units that
regulate how much of the layer's transformation vs. the unchanged input is
passed forward ("information highways"). Evidence (verified from full text):
on MNIST/CIFAR-10/CIFAR-100 with SGD+momentum, highway networks up to 100
layers train directly where plain networks with variance-preserving init (He
et al. 2015) degrade; networks as deep as 900 layers are reported optimizable
in preliminary experiments; a 100-layer highway network performs comparably
to much shallower plain nets. Hyperparameters via random search (40 runs).
This is the pre-residual gated-depth mechanism: ResNet (Dec 2015) later
displaced it with the parameter-free identity skip.

## 2. MECHANISM, EXACTLY

General form (paper Eq. 2), with transform gate T and carry gate C:

    y = H(x, W_H) · T(x, W_T) + x · C(x, W_C)

Simplified coupled form actually used (paper Eq. 3), C = 1 − T:

    y = H(x, W_H) · T(x, W_T) + x · (1 − T(x, W_T))

All products are elementwise. The transform gate:

    T(x) = σ(W_T^T x + b_T),   σ(z) = 1 / (1 + e^{−z})

Negative gate-bias initialization: b_T is initialized to a negative value
("e.g. -1, -3 etc." — paper's wording). At init T ≈ σ(b_T) is small, so every
layer starts close to the identity (y ≈ x): gradients flow through the whole
stack from step 0, and layers only "switch on" transformation as learning
finds use for it. The paper credits the idea to LSTM gate biasing for
bridging long-term dependencies early in training. This init is the load-
bearing trick — it is what lets 100-layer stacks train at all.

Dimensionality: x, y, H(x), T(x) must all have the same dimension. For
dimension changes the paper uses a plain (non-highway) layer, or
sub-sampling/zero-padding of x. In our transformer-block translation this
constraint is automatically satisfied (all sublayers are d -> d).

Contrast with the residual connection that displaced it: ResNet computes
y = x + F(x) — the skip is hardwired, parameter-free, and unconditional; both
paths contribute at full strength. Highway computes a learned, input-
dependent convex combination: T costs an extra d×d + d parameters per gated
unit, couples the two paths (more transform necessarily means less carry),
and can shut a layer off entirely (T -> 0 gives exact identity, which a
residual layer can only approximate by driving F(x) to 0). The follow-up
paper's lesioning analysis reports many trained layers sit near carry
behaviour [not re-verified]. The open registry question is whether the
learned gate buys anything at tiny scale, where its parameter cost is dear.

## 3. SPEC-GRAMMAR MAPPING

Grounding (read 2026-08-12): tools/coalfire_spec.py (spec paths
arch.custom.{d,layers,heads,d_ff,attention,window,sinks}, data.T,
train.{optimizer,lr,batch,steps}), tools/atlas_taxonomy.py (TAXONOMY,
CONSTRAINTS), tools/parity_model.hpp (ParityLM, FlexLM).

The taxonomy already reserves the right slot: "residual" is status "planned"
with lattice ["pre-norm", "post-norm", "attnres"]. Highway is a value of that
slot (a combine-rule substitution), not a new attention kind. Proposed
minimal spec keys:

    arch.custom.residual:        "residual" | "highway" | "plain"
                                 (path for the planned residual slot;
                                  "residual" = today's hardwired pre-norm add;
                                  "plain" = no skip at all, needed for lane C)
    arch.custom.gate_bias_init:  float, default -2.0
                                 (legal only when residual = "highway";
                                  add a CONSTRAINTS predicate for this)

If a flatter knob is preferred, `arch.custom.block: "highway"` is equivalent;
the residual-slot form is recommended because Atlas Deltas are defined as
substitutions within a slot lattice. Family: rides flex (FlexLM), which
already owns depth/norm/activation/position; the gpt2 family constraint pins
layers==2 and fixed flavors, so highway does not belong there.

Translation decisions to record with the entry (interpretation, not paper):
- T1: the 2015 H is one fully-connected layer + nonlinearity. Transformer
  translation: H = the existing sublayer (attn or MLP); each of the two
  sublayers per block gets its own gate (W_T: d×d, b_T: d). The residual add
  `h = x + sub(x)` becomes `h = T(x)·sub(x) + (1−T(x))·x`.
- T2: all lanes keep house pre-LN so the ONLY difference between lanes is
  the combine rule. The paper used no normalization; a paper-faithful
  no-norm lane is blocked on the norm="none" knob (taxonomy notes it waits
  for a knob) and is out of scope for the pilot.
- Cross-engine: highway is microtorch-only. coalfire_spec.py must add
  arch.custom.residual != default to its REFUSALS table (same shape as the
  sinks refusal: different mechanism, not a different parameter).

## 4. ENGINE GAP ANALYSIS

Already present (verified in ops.hpp / nn.hpp / parity_model.hpp):
- nn::Linear with optional bias (the gate's W_T, b_T)
- ops::sigmoid (ops.hpp:82; in-place sigmoid_ at :128)
- ops::mul (hadamard, :16), ops::sub (:15), ops::add, ops::scale (:26)
- residual add wiring in ParityLM/FlexBlock; FlexLM already takes arbitrary
  n_layers, so DEPTH for exact-attention highway needs no new engine work
- constant-fill Matrix(1, d, v) + reg() (used by FlexBlock rmsnorm weights),
  so initializing b_T = gate_bias_init is one constructor line

Not present — must be added:
- A highway combine in FlexBlock (or a HighwayBlock sibling): gate Linear per
  sublayer, sigmoid, elementwise mix. No new autograd op is needed: use the
  identity y = x + T·(H−x)  ==  ops::add(x, ops::mul(T, ops::sub(H, x))),
  which avoids materializing a ones tensor for (1−T).
- A way to set the gate bias to a constant: either write b_T's Matrix after
  constructing the Linear, or register a separate bias Var via reg().
- "plain" (no-skip) wiring for lane C — trivial (drop the add).
- Spec plumbing: mtstudio.cpp parse for arch.custom.residual /
  gate_bias_init; atlas_taxonomy.py flips the residual slot to implemented
  (add "highway", "plain" to the lattice) + the gate_bias_init constraint;
  coalfire_spec.py refusal entry.

Size estimate: ~50-70 lines in parity_model.hpp (a FlexBlock-shaped variant
plus gate telemetry accessor), ~30-50 in mtstudio.cpp parsing/validation,
~15 in atlas_taxonomy.py, ~10 in coalfire_spec.py, plus an equivalence test
(gate_bias_init -> −inf must reproduce... note: T -> 0 gives identity, NOT
the residual model, so the pin test is: residual lane bit-identical to
FlexLM today; highway lane with T forced to 1 reproduces "plain"). Order
150 lines total, no new kernels, no new autograd closures. Complexity: low.

ROADMAP placement: independent of CUDA (item 1) and of deep-SWA (item 1a) at
2 blocks — highway uses exact attention, and FlexLM depth already works, so
1a's window/sinks gap does not gate this entry at any depth. CUDA matters
only for the deeper rungs (depth 4-6+ on CPU is a time cost, not a
capability gap). Slots naturally after Rung B lands, before or alongside P2a.

## 5. PROTOCOL DRAFT (house protocol, atlas standard)

Fixed for all lanes: TinyStories, vocab_cap 4096, T=256, batch 4, AdamW,
flat lr 1e-3, 1200 steps, eval_every 100, seeds 1-5 paired across lanes,
fp32, no dropout. d=128, heads=4, exact attention, pre-LN, learned position,
gelu MLP. Telemetry: per-layer mean T(x) each eval (same pattern as the SRD
mean_gate) — the mechanism observable, one atlas row per lane.

PLANNED CONTRAST at 2 blocks, matched parameter count:
- Lane A residual: FlexLM as it exists today (d_ff 512).
- Lane B highway:  gate per sublayer, gate_bias_init −2.0. Gates cost
  4×(128×128+128) = 66,048 params; match by d_ff 512 -> 384 (removes
  65,792; residual mismatch +256 in lane B, ~0.02% — report exact counts in
  the run registration). Secondary width-matched variant (d_ff 512, params
  reported, not matched) may be run but is labelled as such.
- Lane C plain: no skip at all, d_ff 512 (param-identical to A).

Post deep-rung unlock (CUDA for time): repeat A/B/C at depth 4 and 6, same
protocol, same seeds. Depth is where the paper's mechanism is supposed to
earn its keep; depth 2 is the null-region check.

Scope discipline: any verdict is "at this protocol and scale" — TinyStories,
d=128, ≤6 blocks, 1200 steps, 5 seeds. Reported as a trend candidate for the
scale ladder, never as an absolute about the 2015 paper, whose claims live
at 10-900 fully-connected layers on image classification. A null at depth 2
does not contradict the paper; the paper makes no claim at depth 2.

## 6. WHAT WOULD MAKE THIS ENTRY INTERESTING

Q1 (headline): at tiny scale where every parameter is dear, does learned
gating beat the hardwired residual prior at matched params — or does paying
66K params out of d_ff for gates cost more than the gate is worth?
Pre-registered directional guess: A ≤ B at depth 2 (residual as good or
better); either outcome is a registry verdict.

Q2: does the negative-bias init matter at depth 2? Sweep gate_bias_init in
{0, −1, −2, −4}, lane B only. The paper's rationale is depth-driven (keep
early gradients flowing through MANY layers); at 2 blocks it may be inert.
A flat curve at depth 2 that steepens at depth 4-6 would be a clean
"mechanism requires depth" trend for the ladder.

Q3 (post-depth-rung): ordering and degradation with depth — plain is
expected to rot first (that is the 2015 claim's shape); does highway track
residual or fall behind it, and do trained gates drift toward carry
(mean T falling, layers self-pruning) as the follow-up's lesioning analysis
suggests [not re-verified]? The mean-T telemetry answers this for free.

## 7. REGISTRY ROW — depth-2 rung RESULT (13 Aug 2026)

Pre-registered analysis ran 13 Aug 2026 (analyze.py, committed pre-data;
receipts in experiments/registry_0001_highway/receipts/). All guards
green: params residual 1,478,144 / highway +256 as documented / plain
equal; no early stops; trace and result.json agree on every run.

**Q1 (primary, estimation framing, two-tailed df=4 crit 2.776):
NO RESOLVABLE DIFFERENCE between learned gating and the residual prior
at this protocol and scale.**
- Delta1(400)  = +0.0044, 95% CI [-0.0095, +0.0183], t = 0.88
- Delta1(1200) = -0.0101, 95% CI [-0.0305, +0.0104], t = -1.37
Per the prereg this is a genuine registry verdict, not a failure: at
d=128, 2 blocks, 1200 steps, the 66K parameters spent on transform
gates buy neither harm nor help against the hardwired skip.

**Q2 (descriptive): any skip connection is worth ~1.2-1.5 nats here.**
- Delta2(PLAIN - RESIDUAL) at 400: +1.208 [1.168, 1.249]
- Delta2 at 1200: +1.497 [1.457, 1.537] — the gap GROWS with budget.

Mechanism observable E[T]: omitted (event stream records no gate
signal; noted per prereg, not patched mid-run). Q2-bias-sweep and the
depth-4/6 rungs (section 5) remain open; depth is where the 2015
mechanism claims to earn its keep, and this depth-2 null is exactly
the null-region check the dossier predicted it might be.

Scope: TinyStories, d=128, 2 blocks, 1200 steps, 5 paired seeds.
Trend candidate for the ladder; no claim about the 2015 regime.
