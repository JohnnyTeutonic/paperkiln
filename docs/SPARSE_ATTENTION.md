# Sparse Attention Research Program

The flagship research phase. Goal: an original, *natively trainable* efficient-attention
mechanism validated in microtorch — not another commodity reimplementation.

Survey date: 2026-07-30 (live arXiv sweep; refresh before each design round).

> **CORRECTION — 2026-08-04 (rung 2, pre-registered:
> [experiments/SRD_PREREG_R2.md](experiments/SRD_PREREG_R2.md)).** Wherever this document
> describes the SRD gate as concentrating on "retrieval-critical
> positions", read: **distributionally novel positions**. The 2×2
> de-confounding experiment (12 runs, 3 seeds) showed the 5×-replicated
> concentration collapses to zero when needles are in-distribution
> (+0.59 → −0.003), never-queried decoys are gated at parity with or
> above true targets (DCI 0.97 / 1.89), and target-vs-non-target
> selectivity is microscopic. The concentration and the shuffle
> falsifier were TRUE results about a DIFFERENT quantity: the gate is
> an honest, information-dependent **novelty detector**, and the
> benchmark had made novelty and retrieval-criticality the same
> positions by construction. This also completes the recall
> retraction's post-mortem — firing on the needle buys no recall if
> firing tracks novelty rather than need. The matched-density
> efficiency question remains open (control lane failed the task at
> the rung-2 config; see the pre-reg's RESULTS for rung 2b).

> **S1 STATUS — 2026-08-05, SETTLED at n=10.** `ops::swa_attention`
> (sliding window + sinks, bitwise equivalence pin against full
> attention at window ≥ T) is a spec-expressible lane, and its
> measuring cell now carries a real result: **w=64+sink BEATS full
> attention at T=256** — best_val **3.861 vs 3.909**, paired
> **t = −10.35 (df = 9, two-tailed crit 2.262), swa better in 10/10
> seeds**, mean gap −0.048 nats. The cell is genuinely sparse: w=64
> against T=256 computes **44.3%** of full causal's attention entries
> (stated next to w because the equivalence pin makes w ≥ T degenerate
> by construction). Less attention, better loss, every seed.
>
> *How it got here, because the path is the point:* the first cut
> (n=3) was downgraded to `pending` on referee review — paired
> t = −3.39 at df = 2 sat below the 4.303 critical value, one paired
> diff was a visible outlier (−0.0208), and the direction had NOT been
> pre-specified, so no one-tailed rescue was licensed. The extension
> settles it two-tailed with the outlier explained as seed noise. The
> conservative unpaired analysis (t = −4.19, df = 18) clears its bar
> too; both are recorded in the registry row to keep the numbers
> unambiguous.
>
> **S1-b — 2026-08-06: the window sweep killed our own explanation.**
> Loss falls MONOTONICALLY as the window shrinks — w=16 (3.785) <
> w=32 (3.822) < w=64 (3.852) < w=128 (3.894) < exact (3.899), every
> adjacent contrast clearing its paired test, down to **12.3% density**.
> We had pre-registered two mutually exclusive predictions at the w=16
> rung: locality-as-inductive-prior required an interior optimum (too
> small a window should starve the model); capacity-artifact required
> monotone. **Monotone won.** Per the decision rule committed before
> the runs, `S1-swa-beats-exact` has been amended and its mechanism
> claim withdrawn — the effect is real and replicated, the explanation
> we gave for it is not supported. Registry: `S1b-window-monotone`.
>
> The harness verified itself on the way: swa@w=256 with sinks=1
> reproduced the exact lane **bitwise (diff 0.00e+00, all 5 seeds)**
> through 400 steps of real training, so the equivalence pin holds
> outside the unit test.
>
> Stated limitation, from the pre-registration rather than added after:
> monotone down to w=16 is ALSO consistent with an optimum below 16.
> The discriminating test is budget, not another window rung — and it
> was named as the follow-up before this result was known.
> `experiments/sparse_s1_budget/` (steps 400 → 1200) is pre-registered
> and running: capacity-artifact predicts the advantage shrinks;
> optimum-below-16 does not.
>
> **S1-c — 2026-08-07: THE ADVANTAGE REVERSES. The S1 positive was a
> short-budget artifact.** Paired Δ (swa@64 − exact) moves from
> **−0.0472 at 400 steps** to **+0.0123 at 1200 steps** — a change of
> **+0.0595 (t = 5.09, df = 4)**, in the same direction on every seed.
> swa was ahead in 5/5 seeds at 400 and 1/5 at 1200. Both lanes were
> still improving at the cutoff (no early stop, 12 evals, loss ~3.86 →
> ~3.59), so this is the still-training regime, not overfitting — the
> pre-registered threat was checked and did not fire.
>
> `S1-swa-beats-exact` is **SUPERSEDED** by `S1c-budget-reversal`, per
> the decision rule committed before these runs. The 400-step
> measurement remains correct; what it cannot support is the ordering.
>
> *Stated precisely:* the SHRINKAGE is established two-tailed
> (t = 5.09, crit 2.776). The REVERSAL itself (t = 2.74) clears the
> one-tailed crit 2.132 but **not** the two-tailed 2.776 — and the
> one-tailed reading is licensed **only** because P2 committed the
> direction to git before the runs. S1's surprise direction forbade
> exactly that test three days ago. Same lab, same week, opposite
> entitlements, decided entirely by what was written down first.
>
> **What this does and does not license.** At this scale the *sign* of
> a sparse-vs-dense comparison depends on where you stop, so a
> comparison reported at one short budget carries no information about
> the ordering at a longer one. It does NOT establish that any
> published result is wrong — our scope is 2-block, d=128, T=256,
> TinyStories. The general claim is about **reporting discipline**:
> state the budget, vary it, report both.
>
> **Consequence for V2/CoD: the bar changed shape.** "Beat
> sliding-window" is now under-specified, because which lane leads
> depends on the stopping point. Any V2 comparison must run at ≥2
> budgets and report both, fixed in its pre-registration before it
> runs.
>
> Remaining follow-ups: the sinks-vs-window decomposition (which also
> unlocks cross-engine comparison — coalfire has no sinks), and the
> optimizer interaction.

## 1. The landscape, mapped to our opening questions

### Q: "Split dot-product attention into blocks?"
Two distinct lineages, often confused:
- **Flash Attention** — blocked but **exact**. An IO-aware tiling of the same math.
  Not sparsity; nothing is skipped.
- **Block-sparse** — skip most blocks. Sparse Transformers/BigBird (2019-20, fixed
  patterns) → the modern trainable era: **NSA** (arXiv:2502.11089, compressed +
  selected + sliding tiers, end-to-end trainable), **MoBA** (top-k block routing via
  MoE-style gate), **InfLLM-V2** (arXiv:2509.24663, dense↔sparse switchable).
  Verdict: *whether blocks* is settled; **how to cheaply score blocks** is not
  (XAttention, arXiv:2503.16428, scores blocks by antidiagonal sums — the field is
  still guessing at proxies).

### Q: "Hot cache of frequently used tokens / speculative mechanism?"
- Inference-side is crowded: H2O heavy-hitters, StreamingLLM attention sinks,
  HashEvict (arXiv:2412.16187, SimHash Hamming distance as pre-attention proxy),
  Expected Attention (arXiv:2510.00636, estimates future-query attention).
- **Trainable** cache residency — the model *learning* what stays hot during
  pretraining — is much thinner. Our V1 is this idea in disguise: the surprise gate
  is a learned residency policy.

### Q: "LSH / SimHash as cheap dot-product approximation?"
Reformer (2020) used LSH as the attention itself: trains poorly (bucket imbalance,
causality pain). The revival (HashEvict) uses hashes as *selectors*, not replacements.
**Lesson: cheap approximations survive as routers, not as the attention.**

### Q: "Traditional algorithms not yet exploited?" (the Q2 vein)
- **Fast Multipole / H-matrices**: FMA (arXiv:2310.11960), Multipole Attention
  (arXiv:2506.13059). Exists but thin; mostly not natively trained at scale.
  N-body framing (near-field exact, far-field summarized) remains underexploited.
- **Leverage scores** (randomized NLA): Compactor (arXiv:2507.08143), pre-scoring
  (arXiv:2505.11040) — inference-side only so far. Training-time leverage-score
  routing: open.
- **Matrix sketching (Frequent Directions / Co-occurring Directions)**: FD is
  15 years of streaming theory with deterministic guarantees; CoD sketches the
  product of two matrix streams — which is literally linear attention's K^T V
  state. **No attention paper found using it.** → V2.
- **Discrepancy theory**: streaming attention approximation exists in theory
  (arXiv:2502.07861); no practical trained variant seen.

### Q: "Diagonalisation methods?"
Partially mined, from three directions: Performer's FAVOR+ (random-feature
factorization of the softmax kernel), Nyströmformer (low-rank spectral
approximation via landmarks), FNet (fixed Fourier mixing — diagonalization of
circulant structure), and S4/Mamba (state matrices *literally* run diagonalized,
DPLR/S4D). Open-ish corner: maintaining a cheap streaming eigen-sketch of the
attention Gram matrix and routing by spectral leverage — connects to the
leverage-score vein above. Keep on the shortlist; check for 2026 papers before
investing.

### Q: "Synthesis risk — degrading training?"
The central lesson of NSA: **sparsity must be trainable end-to-end**. Post-hoc
sparsification of a densely-trained model degrades; discrete/non-differentiable
routers starve gradients. Every microtorch variant must therefore:
1. keep gradients flowing through the router (soft gates, straight-through, or
   aux losses),
2. pass finite-difference gradcheck on every new op,
3. match the Kimi-linear baseline's training curve on TinyStories before claiming
   anything (the graduation gate).

## 2. Our three shots

### V1 — Surprise-Routed Density (SRD)  ← IN PROGRESS
NSA/MoBA route blocks by learned top-k affinity scores. We route **per-query
density by prediction residual** — the cerebellum mechanism already in-repo:

    predicted = RoutinePredictor(x)              (small MLP, trained in-loop)
    g[t]      = sigmoid(4 * rms(x[t] - predicted[t]) - 1)      in (0,1)
    out[t]    = g[t] * ExactAttention(x)[t] + (1-g[t]) * KimiLinear(x)[t]

Routine tokens (predictable) ride the O(n·d²) linear path; surprising tokens get
exact O(n²) attention. Q and K/V projections are SHARED between paths, so the
mechanism difference is isolated. Training is soft (both paths computed, fully
differentiable); inference hardens the gate (g > τ → exact for that query only),
giving O(ρ·n²·d + n·d²) with ρ = surprise rate. An aux loss mean(g) lets the
caller price density.

Novelty claim (as of the 2026-07 sweep): routing by *prediction error* rather
than by attention-affinity scores appears untried. Closest neighbors:
InfLLM-V2 (global dense/sparse switch, not per-token), MoE-style MoBA gates
(affinity-based), SSA (arXiv:2511.20102, aligns sparse & full outputs — a
useful *evaluation* idea for us).

Risks: (a) gate collapse (always open / always closed) — monitor gate histogram,
price with aux loss; (b) the residual signal may proxy token frequency rather
than contextual novelty — probe with synthetic sequences; (c) soft training
cost is exact+linear together — acceptable at research scale.

### V2 — Sketch-State Attention (CoD/FD)  ← **PROMOTED to lead candidate (2026-08-05)**
Replace linear attention's unbounded K^T V accumulation with a Co-occurring
Directions sketch: fixed sketch size ℓ, deterministic error bounds, streaming-
native, O(n·ℓ·d). The deterministic-guarantee counterpart to Performer's random
features. No prior attention use found. Design question: differentiating through
the SVD-based shrink step (options: straight-through the shrink, periodic
detached re-sketch, or replace SVD with a differentiable power-iteration).

**Why this outranks V1 as a novelty claim (assessment 2026-08-05, and the
lesson the SRD arc paid for).** SRD's claim was BEHAVIOURAL — "the gate
routes on retrieval-relevant surprise" — a hypothesis about what a learned
component would end up doing, which is exactly the kind of claim that
survives replication and still gets reclassified out from under you
(R2-novelty). V2's claim is STRUCTURAL: CoD sketches the product of two
matrix streams, and linear attention's state IS that product (K^T V). The
correspondence is a mathematical identity, checkable before a single model
trains — it cannot be reinterpreted by a later experiment the way a
behavioural story can. Three consequences that make it the better bet:

- **The novelty is defensible by inspection.** "CoD applied to linear
  attention state" is either prior art or it is not; the literature search
  found no attention paper using it (recheck at design time — names rot).
  Compare SRD, whose novelty was always contingent on the mechanism doing
  what we said it did.
- **It inherits deterministic error bounds.** Performer-class methods give
  probabilistic guarantees over random features; FD/CoD give worst-case
  bounds as a function of sketch size ℓ. A quality claim with a
  deterministic bound attached is a fundamentally stronger artifact than a
  benchmark table.
- **Its main risk is ENGINEERING, not epistemic.** The differentiation-
  through-the-shrink question has three named, testable options (above);
  each fails loudly and quickly. SRD's risk was that the mechanism might
  mean something other than we thought — a failure mode that took two
  rungs, a falsifier and a de-confounding 2×2 to even detect.

**Falsifiers to pre-register before implementation** (the S1 lesson: fix the
direction and the analysis first): (a) sketch-size sweep ℓ ∈ {8, 16, 32, 64,
d} — if quality only arrives at ℓ ≈ n the compression claim is dead and that
is a negative worth publishing; (b) a RANDOM-projection state of equal size
as the honest control (deterministic bounds must buy something over a random
sketch, or the theory is decorative); (c) the equivalence check — as ℓ grows
to full rank, the sketch state must converge to exact K^T V, giving a pin of
the same character as SWA's bitwise one. Baseline to beat is not full
attention alone: it is **sliding-window+sink** (S1), which is currently the
bar at our scale.

### V3 — Cheap-proxy block-scoring bake-off
Head-to-head, trained end-to-end NSA-lite: PQ codebook lookup vs SimHash Hamming
vs antidiagonal sums vs leverage-score sampling as the block router. Least novel
(Online VQ Attention, arXiv:2602.03922, is adjacent and *current* — move fast if
we pick this), but fastest to a publishable ablation and it builds the block
harness V1's hardened-inference mode needs anyway.

## 2b. V1 graduation run — RESULTS (2026-07-30)

Setup: tools/srd_parity.cpp, four parameter-matched 2-layer LMs on identical
TinyStories batches (chat7b's 20000-word vocab, unk 14.9%, T=64, d=128,
4 heads, AdamW 3e-3 + clip 1.0, SRD density price lambda=0.05, 300 steps,
one seed). Task loss (aux excluded) averaged per 100-step segment; paired
stats over the final 100 (identical windows make lanes directly pairable).

    steps 201-300 means:  exact 5.586 | kimi 5.397 | srd 5.515 | srd_f 5.584
    paired srd_f - srd  = +0.069  (SE 0.011, t = 6.3)
    paired srd  - exact = -0.071  (SE 0.011, t = -6.5)
    mean gate: srd 0.599 vs falsified 0.602  (same density budget)

**Falsifier verdict: PASSED.** With the same gate budget, the shuffled-
predictor lane is worse by ~0.07 nats at 6 sigma. The surprise gate is
doing real, query-aligned routing work -- not just acting as a stochastic
path mixer.

**Regime finding (honest):** at this scale the ordering is
kimi < srd < exact ~ srd_f -- pure linear attention WINS short-context
small-data TinyStories, and exact attention is the weak lane. SRD lands
between, tracking its gate (~0.6 toward linear). So V1's mechanism is
validated, but this regime cannot show the quality case for density
routing: the battleground is longer context, where linear attention's
state bottleneck bites and surprising tokens should need exact recall.

Next experiment (Colab, T=256-1024): same four lanes plus a
needle-in-haystack probe; expected signature if V1 is right: exact > kimi
at long T, with srd matching exact at a fraction of the exact-path budget,
and gate mass concentrating on retrieval-critical tokens. Caveats to
carry: single seed, 14.9% unk, optimizer moments reset at chunk
boundaries (identical across lanes).

### T=256 Colab run (2026-07-30, experiments/srd_needle_2026_07/results_srd_parity_T256.csv)

One T4, ~20 min, ~1 unit; vocab capped to 4096 (frequency-ordered), 300
steps, otherwise the T=64 protocol. Final-100 paired stats:

    kimi  - exact = -0.149 (t = -34)     srd - exact = -0.025 (t = -5.2)
    srd_f - srd   = +0.027 (t = +5.3)    gates: srd 0.611, srd_f 0.618

**Falsifier REPLICATES at T=256** (second independent 5-sigma result):
aligned surprise routing beats shuffled routing at the same density
budget, at both scales tested. The router mechanism is solid.

**Regime prediction FAILED (recorded as a negative):** exact does NOT
overtake kimi at T=256 -- the linear lane wins LM loss even more decisively
than at T=64. Diagnosis: TinyStories stories are only a few hundred words,
so a 256-token window spans ~1-2 stories and plain next-word loss never
rewards precise long-range retrieval; the linear path's compression is
simply the better bias for this task at any T we can reach with it.
Implication: the quality case for density routing will not come from
TinyStories LM loss at any T. It requires evals with genuine retrieval
structure:
  (a) synthetic needle-in-haystack (insert key-value pairs early, query
      late; measure recall loss per lane), and/or
  (b) a long-dependency corpus (code, wikitext-103 at T>=1024).
That is the next run. Two 5-sigma replications of the mechanism plus a
clean regime negative is the right foundation for it.

### Needle-in-haystack run (2026-07-30, experiments/srd_needle_2026_07/results_needle_{train,probe}.csv)

tools/srd_needle.cpp, 600 steps, T=256, 8 KV pairs / 64 keys / 240-token
gap, fresh sequences, fixed 32-probe eval every 25 steps.

**Recall: INCONCLUSIVE at this config.** No lane -- including exact
attention -- formed the recall circuit in 600 steps (answer CE converged
to ~log 64 = "right token class, uniform over values"; accuracy at noise).
The pre-registered exact>0.8 criterion failed for every lane, so the
discriminative comparison never activated. Plausible causes: 1 sequence/
step, 240-token retrieval distance, d=128 2-layer capacity, 600-step
budget. AMENDMENT (pre-registered before the rerun): T=128 with a 100-token
gap and/or 2000+ steps via checkpoint resume; batch >1 if needed.

**Gate profile: STRONG POSITIVE, replicated at all 24 probes.** SRD's
router separated retrieval-critical positions from filler WITHOUT recall
supervision ever succeeding:

    srd    tail_gate 0.91-0.95  vs  fill_gate 0.30-0.40  (open split
           from probe 2 onward, stable to step 600)
    srd_f  flat at every probe (0.76-0.92 both) -- the decoupled gate
           shows zero positional structure, as it must

The surprise signal identifies WHERE density belongs from the token
statistics alone (rare keys/query vs abundant filler), before and
independent of task success. That is the third independent confirmation
of the router mechanism, and the first evidence it concentrates density
at retrieval sites specifically -- the hardened-inference story's
prerequisite.

### Needle amendment run (2026-07-31, experiments/srd_needle_2026_07/results_needle128_{train,probe}.csv)

The pre-registered amendment: 2000 steps at T=128 (shorter retrieval
distance, 3.3x the training budget), same four lanes, 80 probes at
25-step cadence.

**Recall: INCONCLUSIVE AGAIN.** Every lane -- including exact -- sat at
the same plateau: best answer CE 4.16-4.19 vs log 64 = 4.159 (the "right
token class, uniform over values" attractor once more), max accuracy
0.062 = 1/16 (a single probe-batch hit, consistent with noise). Shorter
distance plus more steps did not form the circuit either, which narrows
the suspect list to throughput/capacity: 1 sequence/step or d=128
2-layer. Next escalation (pre-registered in the amendment): batch > 1.
The discriminative srd-vs-exact comparison still has not activated; no
lane comparison is being claimed from these runs.

**Gate profile: POSITIVE and srd-specific again -- fourth confirmation --
with new temporal structure.** Separation (tail_gate - fill_gate):

    srd    sep > 0.1 at 70/80 probes; peak +0.60 (0.91 vs 0.31) at
           step 475; decays to +0.07 by step 2000 as BOTH gates close
           (tail 0.95 -> 0.35) with falling LM surprise
    srd_f  sep > 0.1 at 0/80 probes -- flat at every single probe

The decay is the new observation: at T=128 the router's positional split
erodes late in training as the whole sequence becomes predictable and
the surprise signal shrinks globally. Two readings, currently
undistinguished: (a) designed behavior -- the density budget contracts
when nothing is surprising, which is the point of surprise routing; or
(b) the T=256 run (600 steps) simply never reached the late-training
regime, and its "stable to step 600" split would also erode by 2000.
A 2000-step T=256 run separates them; queued behind the batch>1 recall
escalation.

### Batch escalation run (B=4, 2026-07-31, experiments/srd_needle_2026_07/results_needleB4_{train,probe}.csv)

B=4 x 1500 steps x T=128: 6,000 sequences (3x the previous run) at 4x
per-step gradient SNR on the answer position.

**Recall: INCONCLUSIVE, third consecutive.** Every lane finished at the
plateau (exact final CE 4.162 vs floor log 64 = 4.159; best transient dip
4.10; accuracy at noise). With {1,4} seq/step and T in {128,256} all
tried, gradient noise and retrieval distance are excluded as the binding
constraint; what remains is model capacity (d=128, 2 layers) vs task
difficulty (8 pairs / 64 keys).

**Gate: fifth specificity confirmation, and the decay readings tighten.**
srd sep > 0.1 at 37/60 probes (peak +0.59 at step 250); srd_f 0/60.
Both T=128 runs ended at train CE ~4.67-4.68 with tail gates 0.28-0.35
despite different step/batch routes there -- gate closure tracks the
surprise LEVEL, not the step count. That is evidence for reading (a):
the density budget contracts as data stops being surprising, which is
the designed behavior.

**Instrument verdict: control-first calibration.** Three runs where the
control never passes means the comparison cannot activate at this
difficulty/capacity point; tuning lanes against a task the control fails
would be noise-mining. npairs/nkeys are now CLI knobs (defaults preserve
the original task): the calibration ladder starts at 2 pairs / 8 keys /
T=64 / B=4 and raises difficulty until exact stops forming the circuit --
the lane comparison then runs at the hardest rung the control passes.

### Calibration ladder + rung-1 breakthrough (2026-07-31, experiments/srd_needle_2026_07/results_needle_ladder1_{train,probe}.csv)

Ladder rungs (600 steps, B=4): 2p/8k/T64, 4p/16k/T64, 8p/32k/T128 -- all
four lanes floor-bound at every rung, accuracy oscillating with no trend.
Consistent with the induction-head literature: recall circuits form as an
abrupt phase change, and 600 steps is pre-jump. So rung 1 was extended to
3000 steps via checkpoint resume, and the phase change arrived -- with an
ordering nobody pre-registered:

    step   exact_ce  srd_ce  srd_f_ce   (floor = log 8 = 2.079)
    1800    2.201     2.143    2.357     all hovering
    2000    2.080     1.828    2.080     srd breaks
    2400    2.067     1.714    1.949     srd_f following
    3000    2.066     1.539    1.662     exact still floor-bound

    breakthrough (3 consecutive probes < floor-0.15):
      srd @ 2000, srd_f @ 2575, exact NONE, kimi NONE (through 3000)
    last-8 probes: srd ce 1.56 / acc 0.34;  srd_f 1.76 / 0.28;
                   exact 2.11 / 0.14;  kimi 2.11 / 0.15

**SRD formed the recall circuit first; exact attention has not formed it
at all within the budget.** The falsifier pair decomposes the effect:
gating AT ALL helps (srd_f eventually breaks while exact never does --
plausibly multiplicative gating suppresses the filler gradient that
drowns the rare answer-position signal in the exact lane), and
surprise-COUPLED gating helps beyond that (srd leads srd_f by ~575 steps
and ~0.2 nats). If this holds up it inverts the V1 framing: the gate is
not a tax on recall paid for efficiency -- it is an accelerant for
forming the recall circuit.

**Status: single seed (7), single config -- NOT yet a claim.**
Pre-registered replication: seeds 1,2,3 at rung 1, 3000 steps; criteria
(i) srd breakthrough strictly before exact per seed, (ii) last-8 CE
ordering srd < srd_f < exact. Also run: seed-7 extension to 6000 steps.
**Both completed 2026-07-31. See the next section -- the replication
FAILED and the claim above does not stand as written.**

### REPLICATION: FAILED (2026-07-31, experiments/srd_needle_2026_07/results_needle_r1s{1,2,3}_*.csv)

Seeds 1, 2, 3 at rung 1 (2p/8k/T64/B4, 3000 steps), identical protocol.

    seed  exact_bt  srd_bt  | last-8 CE: srd    srd_f   exact
      7      none    2000   |            1.559  1.761   2.112   <- original
      1      none    none   |            2.112  1.983   2.111
      2      none    none   |            2.112  2.111   2.113
      3      none    none   |            2.113  2.025   2.114
    (breakthrough = 3 consecutive probes below floor-0.15; floor = log 8
     = 2.079)

**Criterion (i): 0/3 pass. Criterion (ii): 0/3 pass.** SRD did not break
the floor in ANY replication seed within the 3000-step budget. The
seed-7 result was a seed lottery, not a mechanism, and the section above
must be read as describing one lucky initialisation.

**Worse for the specific hypothesis:** in seeds 1 and 3 the only lane to
move below the floor at all was **srd_f -- the falsifier**, whose gate is
deliberately decoupled from surprise. The claim that surprise-COUPLED
gating beats generic gating is therefore not merely unreplicated, it is
contradicted in 2 of 3 seeds. V1's distinctive ingredient is the part
that failed.

**What survives, stated at its true (weak) strength:** across all four
seeds, NO ungated lane ever broke the floor (exact 0/4, kimi 0/4), while
some gated lane broke in 3/4 (seed 7 both; seeds 1 and 3 srd_f only).
That is suggestive that multiplicative gating per se helps the recall
circuit form -- plausibly by suppressing the filler-token gradient that
drowns the rare answer signal -- but at n=4 with no effect in seed 2 it
is an observation, not a result, and it is a claim about GATING, not
about surprise routing.

**Consequences, recorded so they are not quietly forgotten:**
1. The seed-7 6000-step extension (srd CE 0.886, acc 0.531 while exact
   sat at 2.057) is a real measurement of ONE seed and is not evidence
   for the mechanism. Do not cite it as such.
2. Multi-seed is now mandatory before any needle claim; single-seed runs
   on this task are uninformative because the phase change is a lottery
   over inits.
3. Pre-registered next test, if this line is continued: 8 seeds x rung 1
   x 6000 steps, comparing only GATED vs UNGATED (srd+srd_f pooled vs
   exact+kimi pooled), with breakthrough-rate as the statistic. That
   tests the surviving hypothesis rather than the dead one.
4. The gate-profile result (the router concentrates density at
   retrieval-critical positions, srd 70/80 probes vs srd_f 0/80) is
   UNAFFECTED by this failure -- it was measured on gate values, not on
   recall, and it replicated five times. The mechanism does what it says
   on the tin; it just does not (yet) buy recall performance.

## 3. Protocol

- **Scale ladder**: (1) op-level gradchecks → (2) 2-layer model, TinyStories
  word-level (transformer_cpp corpus + tinyllama.cpp for inference sanity) →
  (3) Colab T4 for anything bigger ([[colab discipline: no local GPU]]).
- **Metrics**: perplexity-vs-FLOPs frontier against exact + Kimi-linear + NSA-lite
  baselines; needle-in-haystack retrieval for long context; gate statistics
  (V1); gradient norms through the router (the q4 guard).
- **Falsifiers first**: each variant ships with the experiment that would kill it
  (V1: shuffle the predictor's inputs — if quality holds, the gate wasn't using
  surprise; V2: sketch size sweep — if ℓ≈n needed, no win; V3: random router
  baseline — if proxies don't beat random block choice, stop).
- Venue discipline: journals only, strength over speed.

## 4. Bibliography (survey snapshot 2026-07-30)

- NSA: Native Sparse Attention — arXiv:2502.11089
- MoBA optimization — arXiv:2511.11571
- InfLLM-V2 dense-sparse switchable — arXiv:2509.24663
- XAttention antidiagonal block scoring — arXiv:2503.16428
- SSA: aligning sparse & full outputs — arXiv:2511.20102
- HashEvict (SimHash eviction) — arXiv:2412.16187
- Expected Attention — arXiv:2510.00636
- Online VQ Attention — arXiv:2602.03922
- Compactor (leverage-score KV compression) — arXiv:2507.08143
- Efficient Attention via Pre-Scoring — arXiv:2505.11040
- Fast Multipole Attention — arXiv:2310.11960
- Multipole Attention for long-context reasoning — arXiv:2506.13059
- Frequent Directions — arXiv:1501.01711
- Streaming attention via discrepancy theory — arXiv:2502.07861
