# The Architecture Atlas

**The Architecture Atlas is a cumulative science of neural architectures.**

Not a corpus, not a database, not a leaderboard — those are its instruments.
The manifesto is the noun: a *science*, in which every controlled run adds to a
body of evidence that outlives the paper it was run for, effects are estimated
with uncertainty and revised as data arrives, negatives are kept, and the next
experiment is chosen because the accumulated evidence cannot yet answer it.
Architecture research today restarts from zero with each paper; the Atlas
exists so that it never has to again.

**Status: living design note, begun 2026-07-31; stages 0–1 implemented.** The
proposal is Jonathan's, consolidated from two drafts. Sections marked
*Engineering note* are additions from working feasibility against the existing
codebase; several of them change the plan rather than merely costing it.

---

## 0. The intellectual move

> An architecture should not be represented only by its source code, family
> name, or final benchmark score. It should also be represented by the pattern
> of behaviours and dependencies it exhibits under controlled interventions.

That is the thesis. It turns ablation from a local diagnostic used inside a
single paper into the foundation of a comparative empirical science of model
architecture.

A leaderboard says *A scores 0.7 better than B*. This says *A is high-performing
because one component carries it, while B is high-performing because six
components each contribute a little* — and those are different objects even when
the scores match.

**Two things this is not.**

*Not mechanistic interpretability.* The goal is not to explain a trained model
neuron by neuron or circuit by circuit. It is **architectural explainability**:
understanding how observable design choices relate to model behaviour under
controlled experimental conditions. A model is explained by its comparative
position in a corpus and by its response to controlled changes.

*Not neural architecture search.* NAS asks *which architecture should we
choose?* This asks *what does the architecture space look like, which design
choices matter, under what conditions do they matter, and which experiment would
most efficiently reduce our uncertainty?* Different question, different object,
different output.

---

## 1. Motivation

Architecture research is reported as a sequence of isolated claims: RMSNorm
improves stability; RoPE improves long context; SwiGLU improves capacity; wider
FFNs improve accuracy; local attention cuts compute acceptably. Each is
typically evaluated within one family, on a few datasets, under one recipe, with
limited ablation.

Four problems follow.

**Incomparable conditions.** Papers differ in parameter budget, token budget,
optimiser, schedule, dataset, hardware, implementation, regularisation, and —
most corrosively — tuning effort.

**Scalar collapse.** Leaderboards say which model scored highest, but not why,
which components were essential, which were redundant, whether the gain
generalises across tasks, whether the design is fragile, or whether it depends
on an interaction between components.

**Selection bias.** Published architectures are designs researchers chose to
build, tune and report. They are not a sample of the design space. Analyse only
them and you learn *what researchers publish*, not *how architectural features
behave*.

**Structural identity confused with behavioural identity.** Two models can look
alike and behave differently; two that look different can occupy the same
behavioural region.

A cumulative science of controlled architecture experiments addresses all
four — the corpus is merely its lab notebook.

---

## 2. The core research object

Every architecture in the corpus carries three primary representations.

### 2.1 Structural representation — what the model *is*

`x_i ∈ R^d`: depth; width; parameter count; attention-head count; head
dimension; MLP expansion ratio; normalisation type and placement; residual
topology; activation; positional encoding; parameter sharing; recurrence;
convolutional components; sparse routing; MoE structure; local vs global
attention; memory complexity; estimated FLOPs; inference latency; activation
memory; graph-derived features from the computational graph.

Entries are continuous, categorical, binary, or graph-structured.

*Engineering note.* microtorch has most of this for free: a run **is** a spec
file, and a spec is very nearly `x_i` already — `arch.preset`,
`arch.custom.attention`, `d`, `layers`, `heads`, `T` are factor coordinates.
Missing: parameter count (one pass over `named_parameters()`), analytic FLOPs
(a short walk of the module tree), and a graph encoding of residual topology.

### 2.2 Behavioural representation — how the model *behaves*

`y_i ∈ R^m`: validation loss; test accuracy; convergence speed; sample
efficiency; compute efficiency; memory efficiency; inference latency;
throughput; sensitivity to learning rate; sensitivity to optimiser; gradient
instability; variance across seeds; calibration; robustness to distribution
shift; robustness to noise; long-context degradation; catastrophic-failure
frequency; task-family performance; scaling behaviour.

*Engineering note.* `events.jsonl` already carries the raw material — per-step
loss, `grad_norm`, per-module gradient norms, eval points, early-stop triggers.
Convergence speed, gradient instability and curve shape are post-hoc reductions
of a file currently discarded after each run. `live_variables()` (added with
checkpointing) supplies the memory axis. Missing: held-out task families and a
distribution-shift eval set.

### 2.3 Ablation representation — what the model *depends on*

For architecture `A_i` and component `j`:

```
Δ_ij = M(A_i) − M(A_i^−j)
a_i  = (Δ_i1, Δ_i2, …, Δ_ik)
```

where `A_i^−j` has component `j` removed, replaced, disabled or altered. `a_i`
is the **ablation signature** — not how well the architecture performs, but
which choices support that performance.

One architecture may depend moderately on many components; another almost
entirely on a single innovation; a third may contain components individually
unnecessary but jointly essential. Those differences are invisible on a
leaderboard, and they are the thing a practitioner most needs before borrowing
an idea.

*Engineering note — Δ is a substitution effect, not a removal effect.* You
cannot delete RoPE and still have a model; you replace it with NoPE or learned
embeddings. Every reported Δ must name its baseline substitute, which is what
the component taxonomy (§12) exists to standardise.

---

## 3. Architecture spaces

Architectures should be analysed in several spaces rather than forced into one
representation. Candidate methods throughout: k-means, Gaussian mixtures,
hierarchical, HDBSCAN, spectral, or a k-NN graph with community detection.

**Structural space** (`x_i`) recovers design families: deep-and-narrow,
shallow-and-wide, attention-heavy, MLP-heavy, recurrent hybrids, sparse-routing,
local-attention, parameter-sharing, conv-attention hybrids. The most obvious
analysis and probably the least surprising.

**Behavioural space** (`y_i`): fast learners that plateau early; slow learners
with high ceilings; strong small-data models; unstable high-capacity models;
efficient long-context models; robust but lower-accuracy models; models highly
sensitive to hyperparameters. This can reveal that conventional labels conceal
more meaningful empirical groupings.

**Ablation space** (`a_i`) is the scientifically distinctive one: architectures
dependent on positional encoding; architectures where MLP design matters more
than attention design; architectures robust to normalisation substitution;
architectures with strong component redundancy; architectures dominated by one
critical innovation; architectures whose performance lives in high-order
interactions.

**Efficiency space**: training FLOPs, inference FLOPs, latency, throughput,
memory, power, scaling efficiency.

**Robustness space**: distribution shift, adversarial perturbation,
context-length change, data scarcity, seed variation, optimiser change,
training instability.

### 3.1 Visual exploration

UMAP (or PCA) gives an interactive map, one point per architecture or variant.
Colour by performance, parameter count, compute budget, latency, family,
stability, long-context ability, ablation sensitivity, robustness, dataset, task
family, or confidence. The interface should let the user **switch embedding
space** — structural, behavioural, ablation, efficiency, integrated fingerprint
— since the same corpus rearranges completely between them, and that
rearrangement is itself the finding.

Clicking a point shows: the architecture graph; source paper; training config;
learning curves; uncertainty estimates; nearest neighbours of each kind;
strongest positive and negative components; redundant components; interaction
effects; likely failure modes; recommended follow-up experiments.

Treat the visualisation as **exploratory, not confirmatory**. Clusters visible
in two dimensions are not proof.

*Engineering note — cluster stability is mandatory.* With `d` features and `N`
architectures, clustering will find "families" in noise whenever `N` is not
comfortably larger than `d`. Two cheap, non-negotiable defences:

1. **Bootstrap the clustering.** Resample architectures, re-cluster, report
   pairwise co-assignment frequency. A family surviving 90% of resamples is a
   finding; 55% is a picture.
2. **Pre-register the protocol** — metric, method, `k`-selection rule — before
   looking at the embedding. Otherwise cluster count is a researcher degree of
   freedom, and the SRD replication failure (docs/SPARSE_ATTENTION.md, 2026-07-31) is
   the standing reminder of what that costs.

---

## 4. Neighbourhoods

For a selected architecture the Atlas returns five distinct notions of
proximity:

- **Structural neighbours** — built most similarly.
- **Behavioural neighbours** — most similar empirical profile.
- **Ablation neighbours** — depend on their components in similar ways.
- **Efficiency neighbours** — similar compute/latency/memory trade-offs.
- **Counterfactual neighbours** — differ by one or two design choices yet behave
  substantially differently.

The last class is the scientifically valuable one. It supports queries like:

- What is the nearest architecture that performs materially better?
- Which minimally different architecture is substantially more stable?
- What is the nearest lower-compute architecture with similar accuracy?
- Which structurally similar architecture has the opposite ablation profile?
- Which single component change most efficiently moves this model toward a
  desired behavioural region?

That turns nearest-neighbour lookup into automated comparative experimentation.

---

## 5. Interaction effects

Single-component ablation misleads, because architectural innovations interact.
For components R (RMSNorm) and P (RoPE):

```
I_R,P = M(R,P) − M(R) − M(P) + M(∅)
```

— whether the pair contributes more or less than the sum of its parts.

### 5.1 The interaction taxonomy

- **synergy** — two components work unusually well together;
- **redundancy** — either alone is sufficient;
- **suppression** — one reduces the value of the other;
- **conditional dependence** — one matters only when another is present;
- **family dependence** — the effect appears only in a specific family;
- **scale dependence** — the effect emerges only above or below a size;
- **task dependence** — the effect appears only for a task family.

The output is architectural knowledge rather than benchmark reporting. Not
"SwiGLU improves performance by 0.7 points" but "SwiGLU tends to improve wide
decoder models, contributes little to narrow models, and is especially
beneficial with pre-normalisation under compute-matched conditions."

### 5.2 Higher-order structure

Pairwise analysis still misses three-way interactions, threshold effects,
nonlinear scaling, and compensatory redundancy. A complete factorial is
unaffordable, so intervention selection matters: fractional factorial designs,
orthogonal arrays, Bayesian optimisation over intervention sets, active
learning, sequential design, sparse interaction models, hierarchical screening,
information-gain maximisation. Begin with cheap main-effect screening, then
spend on interactions that look plausible or scientifically important.

### 5.3 Engineering note — this is a design-of-experiments problem, and that changes the plan

Enumerating ablations per architecture explodes: `k` components costs `O(k)`
runs for main effects, `O(k²)` for pairs, `2^k` for the full decomposition. At
`k = 8` that is 8, 28 and 256 configurations **per architecture, per seed**.
Times a corpus of 100, it is dead on arrival.

The fix is to stop thinking "corpus of architectures, each ablated" and start
thinking **factorial design over the architectural factor space**:

- **Full factorial**, `k = 8` binary factors: 256 runs, every interaction to 8th
  order — far more than anyone needs.
- **Resolution-V fractional factorial** (2^(8−2) = **64 runs**): all 8 main
  effects and all 28 two-way interactions, unconfounded with each other. The
  entire content of §5 for a quarter of the cost.
- **Plackett–Burman screen** (12 runs, up to 11 factors): main effects only —
  the right first pass to find which factors deserve the resolution-V budget.

Recommended shape: screen with Plackett–Burman, then spend the real budget on a
resolution-V design restricted to the surviving factors. Standard industrial DoE,
and it maps onto the spec system exactly — **a design-matrix row is a spec
file**.

This also reframes `a_i`: the ablation signature for an *individual*
architecture is then estimated from the fitted model (§9) rather than measured
directly for every architecture — cheaper and less noisy.

---

## 6. Architectural fingerprints

Each architecture gets an integrated fingerprint: structural features, training
dynamics, task strengths, efficiency, robustness, ablation sensitivity,
interaction effects, uncertainty estimates.

> **Architecture profile**
> - Structurally nearest to decoder-only transformers.
> - Behaviourally nearest to recurrent-attention hybrids.
> - Strongest relative advantage: long-context retention.
> - Primary dependency: rotary positional encoding.
> - Most redundant component: gated feed-forward block.
> - Main failure mode: high variance under low-data training.
> - Distinguishing feature: depth contributes more than width relative to its
>   nearest matched peers.

Substantially more informative than a scalar score — and every line is traceable
to specific runs. The fingerprint must **preserve multiple views of similarity**
rather than collapsing to a single embedding.

---

## 7. The corpus

Seven sources, deliberately mixed:

1. **Canonical published architectures** — reconstructed from papers
   (`papers/fetch.py` is already this). Connects the corpus to published claims.
2. **Controlled variants** — one design choice changed at a time: LayerNorm vs
   RMSNorm, learned vs RoPE, GELU vs SwiGLU, pre- vs post-norm, global vs local
   attention, shared vs independent parameters.
3. **Ablated architectures** — components removed, disabled, replaced,
   simplified.
4. **Interpolated architectures** — built between known families: gradually
   increasing recurrence or attention locality, varying MLP-to-attention
   capacity, depth-to-width ratio, degree of parameter sharing.
5. **Random valid architectures** — sampled from a constrained grammar.
6. **User-contributed architectures** — via the Studio, entering the public
   corpus only after validation, metadata completion and reproducibility checks.
7. **Failed and negative-result architectures** — preserved deliberately. A
   corpus containing only successes will exaggerate positive associations and
   conceal failure regions. This is the single clearest advantage a
   purpose-built corpus has over the literature.

*Engineering note.* Source 5 needs a **constrained grammar** emitting only
valid, trainable configurations (`d` divisible by heads, legal norm placements,
compatible position encodings). Worth building early — it is also what makes the
Studio spec builder impossible to misconfigure.

---

## 8. Confounding, and the protocol

The central methodological problem. Performance is affected by parameter count,
dataset size and quality, token count, training duration, optimiser, schedule,
batch size, weight decay, dropout, regularisation, augmentation, initialisation,
precision, compiler behaviour, kernel efficiency, hardware, seed, implementation
quality, tuning effort, and stopping criteria.

A naive corpus could easily learn that architectures from larger labs perform
better, or that recent models perform better, when the true cause is compute or
tuning. The system must distinguish correlation from evidence obtained under
controlled comparison.

### 8.1 Matched comparison regimes

| Regime | Held constant |
|---|---|
| Parameter-matched | approximate parameter count |
| FLOP-matched | training compute |
| Token-matched | training examples seen |
| Wall-clock-matched | elapsed time on defined hardware |
| Latency-matched | inference-time constraint |
| Memory-matched | memory budget |
| Energy-matched | power/energy budget, where measurable |
| Frontier | nothing — compare on Pareto fronts |

The frontier regime matters because architectural strength is rarely
one-dimensional: an architecture may be preferable for lying on a better
trade-off frontier even without maximising raw accuracy.

### 8.2 Protocol families and versioning

Define reproducible protocol *families*, not one universal protocol. Each
specifies dataset, preprocessing, parameter budget, token budget, compute
budget, optimiser, scheduler, batch size, regularisation, precision, hardware
class, seed policy, checkpoint policy, stopping criteria, evaluation procedure.
**Every result records its complete protocol version.**

### 8.3 Hyperparameter fairness

Unequal tuning effort is the nastiest confound because it is unobservable after
the fact. Options: identical search spaces; identical tuning budgets; fixed
default recipes; nested optimisation budgets; reporting tuned *and* untuned
results; explicitly modelling tuning effort as a variable. The corpus must never
silently compare a heavily tuned architecture against a minimally tuned
baseline.

*Engineering note.* The practical defence is to make it protocol: every
architecture gets the **same LR search budget** (e.g. a fixed 5-point grid over
the same range), recorded in the result row. An architecture that would have won
with more tuning simply loses under the stated protocol — and the protocol is
published. "We tuned until satisfied" is neither honest nor reproducible.

### 8.4 Seeds and uncertainty

Every comparison reports multiple seeds, confidence intervals, effect sizes and
variance decomposition. Small apparent gains must not be presented as meaningful
when they fall inside run-to-run variance.

*Engineering note.* On 2026-07-31 a single seed in the SRD needle experiment
produced what looked like a clean phase-change result; three replication seeds
returned 0/3 on both pre-registered criteria, and the falsifier lane outperformed
the mechanism in two of them. Every cell needs ≥3 seeds, and **seed variance is
itself a behavioural feature**. The Atlas would have caught that automatically,
which is a decent argument for building it.

### 8.5 Implementation confounding

Two implementations of one architecture can behave differently. Record and
distinguish: architecture specification; implementation; execution backend;
kernel set; hardware. Matched comparisons should run through the same microtorch
execution path.

### 8.6 Selection bias

Mark every architecture's source — published, generated, user-submitted,
ablated, failed, reconstructed — and stratify analyses by it when necessary.

### 8.7 Task and dataset confounding

An innovation may help one task family and harm another. Never collapse all task
results into one average without preserving task structure.

### 8.8 Temporal confounding

Newer architectures arrive with better software, data and recipes. Publication
year and implementation generation are candidate confounders and must be
recorded as such.

---

## 9. Statistical analysis

Beyond visualisation and descriptive clustering.

**Regression.**
```
Y = β0 + β1(RMSNorm) + β2(RoPE) + β3(depth) + β4(width)
       + β5(RMSNorm × depth) + ε
```

**Mixed-effects**, because results are nested — runs within configurations,
configurations within architectures, architectures within families, tasks within
task families, datasets within domains:
```
performance_{i,d,s} = α + u_i + v_d + β′x_i + γ′(x_i × d) + ε_{i,d,s}
```
This separates architecture-level effects, dataset-level effects, task-specific
interactions and run-level noise, and answers directly: *which architectural
properties have general effects, and which are task-dependent?*

**Bayesian hierarchical models** pool information across related families,
quantify uncertainty explicitly, support small samples, and update as
experiments arrive.

**Tree ensembles** capture nonlinear interactions and give exploratory feature
importance — not causal explanations.

**Gaussian processes** suit lower-dimensional design spaces, especially for
active experiment selection and uncertainty-aware interpolation.

**Graph neural networks** can embed computational graphs directly, capturing
structure hand-engineered features miss. They should supplement, not replace,
interpretable descriptors.

### 9.1 Causal discipline

Strongest evidence comes from controlled intervention. Observational corpus
analysis generates hypotheses; causal language is reserved for randomised
architecture interventions, matched controlled experiments, or justified
quasi-experimental designs. The interface must visibly distinguish:

**observational association → matched comparison → controlled ablation →
replicated intervention.**

*Engineering note.* The mixed-effects model is also what makes the fractional
factorial pay off: the design gives clean unconfounded estimates for `β` and the
interaction terms, while random effects `u_i, v_d` absorb architecture- and
dataset-level noise that would otherwise read as signal. Design and model belong
together; choosing one without the other wastes the compute.

---

## 10. Evidence grades and reproducibility

Preserve microtorch's verification philosophy. Every result traces to an
architecture specification, an experiment manifest, a code commit, an
environment, a dataset version, a protocol version, and one or more raw runs.

The system flags incomplete metadata, failed reproducibility checks, non-matched
comparisons, missing seeds, suspicious variance, and implementation drift.

Results carry an **evidence grade**:

```
exploratory → single-run → replicated → matched → controlled intervention
            → cross-dataset replicated → independently reproduced
```

The grade travels with the claim everywhere it is displayed. A fingerprint line
backed by "single-run" and one backed by "cross-dataset replicated" must never
look alike in the UI.

---

## 11. The data model

Each experiment record contains at least:

**Architecture metadata** — architecture ID; family; source; source paper; code
version; graph representation; parameter count; component taxonomy entries;
parent architecture where applicable.

**Training metadata** — dataset; split; preprocessing; optimiser; scheduler;
batch size; token budget; step budget; compute estimate; hardware; precision;
seed; protocol version.

**Evaluation metadata** — metrics; evaluation code version; checkpoint policy;
task family; distribution-shift tests; robustness tests.

**Intervention metadata** — component changed; intervention type; replacement
component; parent run; expected effect; intervention rationale.

**Reproducibility metadata** — environment lockfile; dependency versions; commit
hash; hardware details; deterministic settings; validation status.

*Engineering note.* This is a superset of what a spec already holds plus what
`events.jsonl` already emits. The gap is a **result row** joining them, plus the
intervention and reproducibility blocks. JSONL is sufficient until the corpus is
large; a database is not required to start.

---

## 12. Component taxonomy

Essential, and the piece most likely to be underestimated. The same concept
appears under different names across papers; without canonical categories,
corpus-level comparison is unreliable.

Canonical categories are needed for: normalisation; activation; positional
encoding; attention topology; residual connection; feed-forward design;
recurrence; convolution; routing; sparsity; parameter sharing; memory mechanism;
output head.

Each component supports **aliases**, **implementation variants**,
**parent–child relationships**, **compatibility constraints**, and
**versioning**.

*Engineering note.* The taxonomy subsumes the substitution lattice of §2.3 —
the legal alternatives for a slot are just the taxonomy's siblings under a
parent category:

```
normalisation : {RMSNorm, LayerNorm, none}
position      : {RoPE, learned, sinusoidal, NoPE}
activation    : {SwiGLU, GELU-MLP, ReLU-MLP}
attention     : {exact, kimi-linear, SRD, sliding-window}
head tying    : {tied, untied}
residual      : {pre-norm, post-norm}
```

Compatibility constraints are what the constrained grammar (§7) enforces, so the
taxonomy is a shared dependency of corpus generation, ablation definition and
the Studio builder. Build it once, early.

---

## 13. Experiment recommendation

The paper pipeline becomes the entry point. microtorch parses a paper and
extracts the baseline architecture, claimed innovations, reported metrics,
stated mechanisms, comparison models and acknowledged limitations, then queries
the corpus:

> **Paper claim:** RMSNorm improves training stability.
> **Corpus evidence:** RMSNorm is associated with lower gradient variance in most
> matched transformer comparisons, but the effect is concentrated in deeper
> models.
> **Recommended experiment:** compare LayerNorm and RMSNorm at depths 12, 24 and
> 48, holding parameter count, token budget, optimiser and compute constant.
> **Reason:** existing evidence suggests a depth-dependent interaction.

Not a generic ablation grid — experiments chosen where evidence is uncertain,
contradictory or underpowered.

**High-information experiments** are prioritised by expected information gain,
uncertainty reduction, estimated novelty, disagreement between predictive
models, sparse coverage of architecture space, high variance in prior results,
or proximity to a Pareto frontier.

**Replication recommendations** fire when a claim rests on one paper, when
effects are small relative to seed variance, when results depend heavily on one
recipe, or when published and corpus results disagree.

**Counterfactual recommendations** propose the smallest change likely to improve
stability, the cheapest change preserving accuracy, the most informative
ablation, the nearest unexplored variant, or the strongest test of a paper's
claimed mechanism.

*Engineering note.* "High-information" can be made precise rather than
rhetorical: rank candidate experiments by expected reduction in posterior
variance of the coefficient in question (Bayesian optimal experimental design).
The corpus supplies the prior; the design supplies expected information gain.
This is the feature that makes the tool worth using rather than merely
interesting.

---

## 14. Worked example

**Workflow.** Paste an arXiv URL → parse → extract architecture and training
setup → identify claims → reconstruct the baseline → map each claimed innovation
onto the component taxonomy → query the corpus for prior evidence → report which
claims are well supported, weakly supported, contradictory or untested → propose
a controlled experiment set → estimate compute cost → run → add results to the
corpus → update the fingerprint → produce a report ranking main effects,
interactions, uncertainty, robustness and reproducibility.

**Report.**

> **Detected claims.** RMSNorm improves stability. RoPE improves long context.
> SwiGLU improves capacity.
>
> **Corpus context.** RMSNorm: broadly positive in deep decoders, uncertain in
> shallow encoders. RoPE: strong long-context benefit, mixed short-context.
> SwiGLU: positive on average, strongly dependent on width and compute budget.
>
> **Proposed experiments.** Full architecture; remove each innovation
> individually; replace each with its canonical baseline; test pairwise
> combinations; repeat at two scales; ≥5 seeds for high-variance conditions.
>
> **Confounding controls.** Matched parameter count; matched training FLOPs;
> fixed token budget; identical optimiser search budget; identical
> preprocessing; identical evaluation pipeline; common backend.
>
> **Findings.** RoPE produced the largest long-context effect. RMSNorm reduced
> gradient variance only at greater depth. SwiGLU improved accuracy but
> increased memory. RoPE and RMSNorm interacted positively. SwiGLU was partly
> redundant at the smaller scale.
>
> **Recommendation.** Retain RoPE across scales. Retain RMSNorm for deep
> variants. Evaluate a cheaper MLP alternative for constrained deployment.

---

## 15. Research questions

- Do structurally similar architectures behave similarly?
- Are there multiple architectural routes to the same performance profile?
- Which components are robustly beneficial across families?
- Which effects depend on scale? On task family?
- Which innovations work only through interaction with other components?
- Which architectures are over-engineered? Which components are redundant?
- Which designs systematically produce stable training?
- Can ablation signatures predict generalisation? Robustness?
- Can an unseen architecture's performance be estimated from its neighbours?
- Can embeddings identify underexplored regions?
- Can architectures be recommended for a desired capability profile?
- Do published families correspond to genuine behavioural families?
- **How much apparent architectural progress disappears under matched compute
  and tuning budgets?**
- Which claimed innovations survive replication across implementation backends?

The bolded one is the question most likely to produce a paper nobody else can
write, and it is answerable at small scale.

---

## 16. Product shape

**Repository one — microtorch.** Architecture definition, execution, training,
evaluation, reproducibility, paper parsing, experiment generation, Studio
interaction. *Research Mode* lives here: architecture reconstruction, protocol
execution, ablation generation, experiment scheduling, metric collection, report
creation.

**Repository two — the Atlas.** Corpus storage, metadata schemas, statistical
analysis, architecture embeddings, UMAP visualisation, clustering, similarity
search, evidence synthesis, hypothesis generation, experiment recommendation,
public dashboards. Depends on microtorch as an execution engine without being
conceptually identical to it.

**Shared infrastructure**: architecture schemas; experiment manifests; component
taxonomies; metric definitions; protocol versioning; reproducibility checks;
result formats.

The split keeps microtorch from carrying every analytical responsibility. The
two are not exclusive and Research Mode is the on-ramp: it is worth building
even if the Atlas never happens.

*On naming.* Of Architecture Observatory / Model Atlas / ArchLab — an
**observatory observes**, and this system's entire epistemic advantage is that
it **intervenes**. Controlled substitution under a fixed protocol is what buys
causal traction over scraped leaderboards. "Atlas" and "Lab" carry that better;
**Atlas** additionally matches the map-and-neighbourhood metaphor the interface
is built on.

---

## 17. Feasibility

Measured microtorch throughput after the 2026-07-31 performance work
(llama-tiny, d=128, 2 layers, T=128, batch=4, fused attention: ~1.03 s/step, so
a 3,000-step run is ~52 minutes).

| Scope | Runs | Serial CPU | 4 concurrent sessions |
|---|---|---|---|
| Plackett–Burman screen, 11 factors, 3 seeds | 36 | ~31 h | **~8 h** |
| Resolution-V design, 8 factors, 3 seeds | 192 | ~166 h | **~42 h** |
| Naive per-architecture ablation, 100 × 6 × 3 | 1,800 | ~1,560 h | ~390 h |

The screen is an overnight job; the resolution-V design is a long weekend; the
naive approach is sixteen days for strictly less information. That gap is the
whole argument for §5.3.

**Already exists**: the execution engine; the declarative spec (a design-matrix
row *is* a spec); checkpoint/resume for long sweeps; `events.jsonl` as the
measurement record; early stopping; multi-seed CLI support; and the
falsifier/pre-registration discipline the whole thing depends on.

**Needs building**, in dependency order:
1. Derived structural features (parameter count, FLOP estimate) — hours.
2. Component taxonomy + constrained grammar + design-matrix → specs — days.
3. Sweep runner and result store (JSONL) — days.
4. Behavioural extraction from `events.jsonl` — days.
5. Statistical layer (mixed-effects, interaction estimates) — days.
6. Studio map view — last, because everything above determines what it displays.

---

## 18. Risks

**Scale — the central external-validity threat.** Experiments run at d=128, 2
layers. Many architectural effects are scale-dependent and some reverse. An
Atlas built entirely at tiny scale risks being a beautiful, internally
consistent map of a regime nobody deploys in. Mitigation: make scale **an axis
rather than a constant** — measure every effect at three sizes and report the
*trend*, not the point. A component whose benefit grows with depth is a
different finding from one that shrinks, and trends extrapolate where point
estimates do not. Anything not measured across the ladder is labelled
small-scale-only.

**Proxy validity.** Related but distinct: cheap proxy tasks may not predict
behaviour on real workloads. Model scale explicitly; do not assume transfer.

**Benchmark dependence.** Results can become benchmark-specific. Needs diverse
tasks and clearly labelled domains.

**Implementation artefacts.** Differences may reflect kernels or code paths, not
architecture. Common execution infrastructure reduces but does not eliminate
this; the existing GPT-2 and Qwen parity checks are the standing evidence that
this implementation agrees with the reference, and should be cited whenever the
corpus makes a claim.

**Overinterpretation of embeddings.** UMAP and clustering are exploratory.
Visible clusters need statistical validation (§3.1).

**Causal overclaiming.** Predictive models find associations. The evidence-grade
ladder (§10) and the causal discipline of §9.1 exist to keep the interface
honest.

**Taxonomy ambiguity.** Components are not always cleanly separable; some
innovations alter several properties at once. The taxonomy needs explicit
"compound intervention" marking.

**Data leakage and duplication.** The corpus must detect repeated runs, reused
checkpoints and derivative architectures.

**Contributor bias.** User-submitted data will be uneven. Validation and
evidence grading are the defence.

**Scope.** This is a research programme, not a feature. The staging below is
arranged so each stage is independently useful if the next never happens.

---

## 19. Staged plan

**Stage 0 — instrument (days).** Derived structural features; behavioural
extraction from `events.jsonl`; persist run records instead of discarding them.
*Useful alone:* every future microtorch run becomes a data point.
**STATUS 2026-07-31: DONE.** `Module::parameter_count()`, `train.seed`, the
`model` event, `result.json` per run (mtstudio); behavioural features in
`tools/atlas_extract.py` (best_val, final_train_loss, steps_to_half_gap,
loss_auc_norm, grad_norm mean/max, grad_spike_count, loss_tail_std, gate
stats) with a fixture selftest.

**Stage 1 — taxonomy, grammar, sweep runner (days).** Component taxonomy;
constrained grammar; design matrix → specs; a runner with checkpoint/resume
writing one result row per run.
*Useful alone:* multi-seed sweeps become one command — which the SRD line needs
regardless.
**STATUS 2026-07-31: sweep runner DONE** (`tools/mtsweep.py`: grid + PB12
designs over dotted spec paths × seeds, resumable, parallel with the
OMP-oversubscription lesson baked in, per-cell seed statistics and a
signal-vs-seed-noise verdict in the aggregate). Verified end-to-end on a
4-run micro-sweep: lr=3e-3 beat lr=1e-3 with the gap 16x mean seed noise.
**STATUS 2026-08-01: taxonomy + grammar DONE** (`tools/atlas_taxonomy.py`):
slots with alternatives + aliases, compatibility constraints as predicates,
`sample()` drawing random VALID architectures (corpus source 5),
`validate()` wired into mtsweep so no compute is ever spent on an illegal
config. Stage 1 complete.
**UPDATE 2026-08-01 (same day): norm/position/activation went planned →
IMPLEMENTED** — the flex family (`tools/parity_model.hpp` FlexLM) makes
them spec-expressible (`arch.custom.norm/activation/position/d_ff`, any
depth), with a bitwise equivalence pin against ParityLM at the family
defaults. The design space §12 sketches is now three slots wider and
sample() draws from all three families (gpt2/llama/flex). Only
`residual` (post-norm) remains a planned lattice.

**Stage 2 — the screening experiment (one night).** Plackett–Burman over ~11
architectural factors, 3 seeds, one dataset, parameter-matched. Main effects
with confidence intervals.
*Useful alone:* a publishable result about which architectural factors matter at
small scale, positive or negative.
**STATUS 2026-08-01: DONE — and it was literally one night.** 7 factors ×
PB12 × 3 seeds = 36/36 runs (6.9 h serial CPU on 2 workers). Headline:
Muon is the strongest factor in the screen (loss-AUC t = −10.8, best_val
t = −6.2, throughput cost t = −2.4); lr decouples speed from quality
(half-gap t = −8.8 with **zero** best_val gain and 3× grad spikes); heads
is null on every metric. Cell *ranking* was inside seed noise while the
18-vs-18 main effects were 6–10σ — the seed-lottery lesson, demonstrated
in our own data. Full writeup with caveats and the Stage 3 prescription:
`atlas/ATLAS_STAGE2_RESULTS.md`; raw rows in `experiments/atlas_stage2/`.

**THE FINDINGS REGISTRY (added 2026-08-04, the programme's spine):**
`atlas/findings.jsonl` — every claim as a machine-readable row with
effect/SE/t, scope, status (supported/replicated/superseded/retracted/
pending) and receipts; `tools/atlas_findings.py` validates/renders/
advises (cite-or-refuse), `tools/reproduce.py` re-runs any finding's
manifest with an honest cost quote and issues a REPLICATED /
DID-NOT-REPLICATE verdict. This is what "cumulative science" compiles
to: the corpus (§6) holds runs, the registry holds *what they showed*,
and both correct themselves in public.

**Stage 3 — the resolution-V design (a weekend).** All main effects and pairwise
interactions for surviving factors; fit the mixed-effects model.
*Useful alone:* the interaction table is the scientific core of the idea.
**STATUS 2026-08-03: DONE — 48/48.** Full 2^4 factorial on the Stage-2
survivors {optimizer, lr, d, context} × 3 seeds at 3× the Stage-2 token
budget, context TOKEN-MATCHED via a linked factor. Three headline
results (`atlas/ATLAS_STAGE3_RESULTS.md`): (1) **lr × optimizer is a real
interaction** (best_val t = −3.1, spikes t = −4.0): lr=3e-3 is the best
setting under Muon and the worst under AdamW — Stage 2's "lr has no
final-loss effect" was two opposite conditional effects averaging out,
exactly what a screen cannot see and a factorial can; (2) Muon's main
effect replicates at 3× budget (t = −7.4); (3) **the token-matched
context null**: Stage 2's "T=256 better" evaporates with tokens held
equal — the alias is dead. Capacity (d=192) still unpaid at 3×. The
screen→factorial arc is now a complete worked example of the method.

**Stage 4 — the scale ladder (weeks).** Repeat 2–3 at three sizes; report
trends. *This is what converts small-scale findings into defensible claims.*

**Stage 5 — fingerprints and neighbours (weeks).** Embeddings, five neighbour
types, fingerprint reports, evidence grades.

**Stage 6 — the Atlas surface (weeks).** Map view, filtering, Pareto fronts,
paper-vs-corpus comparison, experiment recommendation.

### 19.1 The minimal viable research programme

A strong first study, and the concrete content of Stages 2–3. A constrained
transformer design space:

- LayerNorm vs RMSNorm
- learned positional embeddings vs RoPE
- GELU vs SwiGLU
- pre-normalisation vs post-normalisation
- three depths × three widths
- two task families

Under fixed parameter and compute budgets, measure validation loss, convergence
speed, seed variance, gradient statistics, memory, latency and long-context
degradation. That alone supports main-effect estimation, pairwise interaction
analysis, all three clusterings, visualisation, and early experiment
recommendation.

**A narrow but rigorous corpus is worth more than a broad but confounded one.**
That sentence should govern every scoping decision on this project.

---

## 20. Why this fits microtorch specifically

Most people cannot build this. It needs an execution engine you control
end-to-end, a declarative run format, cheap experiments, and — most rarely — a
research culture that publishes negatives and pre-registers criteria. microtorch
has all four, and the last is the scarce ingredient: an atlas assembled by
someone willing to report that their own mechanism failed to replicate is worth
considerably more than one assembled by someone who is not.

The deepest idea here is not automation. It is **cumulative architectural
evidence**: instead of every paper starting from zero, results are preserved and
integrated across architectures, tasks, interventions, scales and training
regimes — moving architecture research from isolated benchmark claims toward a
systematic empirical science.

---

## 21. The endgame: from Atlas to search (added 2026-07-31)

The Atlas is not only a map. It is the missing prerequisite for the most
ambitious version of this programme: **given a target objective, search finds
the architecture** — not just widths and depths, but compositions, and
eventually components that do not exist yet.

### 21.1 Why a decade of NAS mostly failed to deliver that

Classical NAS (including the genetic-algorithm attempt in AI_ML — the failure
there was structural, not an execution error) searches by **blind mutation
against an expensive, noisy oracle**. Every proposal costs a training run;
nothing learned from run N transfers structure to run N+1 beyond a fitness
scalar; the search space definition smuggles in most of the real knowledge; and
the proxy tasks overfit. The field's honest scorecard: **component-level search
has genuine hits** — Swish was found by search, Lion was found by symbolic
program search, EvoNorm found normalisation-activation layers, AutoML-Zero
evolved learning algorithms from primitives — while **whole-architecture novel
search has essentially none** at the level of "invented something researchers
now use as a design idea".

The 2026 wave (FunSearch, AlphaEvolve, karpathy/autoresearch) replaces random
mutation with an LLM as the proposal operator: agent edits code, short training
run scores it, keep-or-revert, loop. autoresearch is the single-GPU personal
version and its design is deliberately minimal — one metric, greedy
hill-climbing, and **no memory beyond the current best code**. Every night
starts from approximately zero accumulated understanding. That is the
time/space trade-off it embodies: all compute goes into trying things, none
into remembering *why* things worked.

### 21.2 What researchers actually have that these systems lack

A human architect carries a fitted, uncertainty-weighted mental model of the
interaction structure of design space: *RMSNorm pays off in deep nets; gating
helps selective-copy-shaped problems; this combination is redundant; that one
is untested.* New architectures are mostly **recombinations conditioned on that
model, aimed at a named problem**. The Atlas — effect estimates, interaction
terms, uncertainty, evidence grades — is precisely that mental model made
explicit and queryable. Which yields the three-layer design:

```
WORLD MODEL   the Atlas: fitted effects + interactions + uncertainty
      ↓ conditions
PROPOSER      typed-grammar generation (Chimera) / LLM proposal,
              ranked by expected improvement or information gain
      ↓ feeds
EVALUATOR     microtorch under the matched-protocol regimes,
              results flowing BACK into the world model
```

**Search proposes what the model is most uncertain-and-optimistic about**
(Bayesian optimal experimental design, §13) instead of what mutation happens to
reach. Every evaluation improves the prior for every future proposal. That
closed loop — propose, run, *integrate* — is what none of the 2026 systems
have, and it is the unclaimed position: **evidence-conditioned architecture
search**.

### 21.3 The pieces already exist in this workspace

- **The proposer's grammar**: AI_ML/chimera — MAP-Elites quality-diversity
  search over *typed compositions* of primitive blocks (attention, gating,
  recurrence, fast weights), with weight inheritance via containment init, a
  behavioural-fingerprint archive (the Atlas's `y_i` at toy scale), and a
  falsifier mode where a primitive is dropped and the search must reconstruct
  its niche. Chimera is the search layer prototyped at toy scale.
- **The world model's substrate**: this document, Stages 0–3.
- **The evaluator**: microtorch + the spec system + the matched protocols.
- **De novo components**: a lower-level symbolic grammar over scalar/tensor
  primitives (the AutoML-Zero / Swish / Lion lineage) slots in as one more
  factor family once the taxonomy (§12) exists. Component search is the part
  with historical precedent for genuine hits, and FD gradchecks make candidate
  ops cheap to admit safely here.

### 21.4 The validation gate: time-sliced rediscovery

"Can the system find novel architectures?" is unfalsifiable as stated. The
falsifiable version, and the experiment that would carry a paper:

> **Fit the world model only on evidence available before date X. Ask the
> system to propose. Does it propose what the field actually discovered after
> X?**

Concretely: build the corpus's canonical-architecture axis from papers up to
2019 (papers/fetch.py provides exactly this, with provenance); fit; ask for
proposals under the matched protocol; check whether RoPE-like position
handling, RMSNorm-like normalisation, SwiGLU-like gating emerge as
high-expected-value proposals *before the evidence that motivated them exists
in the corpus*. Chimera's drop-a-primitive falsifier is this test at toy scale;
the time-sliced version is the grand one. Rediscovery is the control experiment
that licenses any future claim of genuine novelty — a system that cannot
re-find RoPE has no business claiming its new proposal matters.

### 21.5 The wall, stated plainly

Evaluation cost is the true limit — every candidate is a training run, which is
why the world model must be sample-efficient (hierarchical pooling, §9) and why
proposals must be ranked by information gain rather than enumerated. Scale
transfer remains the standing threat (§18): the search finds what is good at
the scale it can afford to evaluate, so the scale-ladder trends, not point
estimates, are what the proposer should condition on. And the grammar is where
the intelligence hides: designing the typed space of legal compositions IS
architecture research, done once, explicitly, instead of implicitly per paper.

None of that diminishes the target. It defines the order of operations: the
Atlas stages are not a detour before the search programme — they are its
training data.
