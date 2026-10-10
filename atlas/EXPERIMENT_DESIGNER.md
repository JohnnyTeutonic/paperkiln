# ATLAS's empirical model and experiment designer

Implemented 28 September 2026. This supplies the first working contrast-model
and information-directed proposer for the loop in ARCHITECTURE_ATLAS.md
sections 9, 13 and 21. It extends ATLAS itself. It does not reanalyse the transfer
studies, replace the synthesis pipeline, or search for a leaderboard winner.

**New capability:** declare an architectural question, supply scoped observations,
and receive a costed set of legal, matched-seed experiments chosen to reduce
uncertainty about that question. After new observations arrive, fit again and
propose the next set. Predictions include conditional substitution fingerprints.

## Run the demonstration

From the microtorch directory, using an existing Python with NumPy:

```sh
python -B tools/atlas_designer_demo.py --output atlas/my_new_designer_demo
```

The checked-in [demonstration](designer_demo_v1/DEMO.md) contains the full input,
before/after plans, a linked-factor sweep, and the newly generated synthetic
observations. It starts with separate normalisation and activation measurements,
which leave their interaction unknown. The planner first requests the joint
configuration, then adds matched controls. The artificial interaction is -0.8;
after those simulated observations the posterior mean is -0.789023. The planned
variance 0.013110 equals the variance after ingesting the simulated observations.
**These numbers demonstrate functioning software under its assumptions, not an
architectural discovery or evidence of real-world design efficiency.**

For your own **prospective** protocol, copy the demonstration request, replace
the scope and complete base trainer spec, declare factors, observations, costs
and target contrasts, then run:

```sh
python -B tools/atlas_designer.py request.json --output new_plan
```

No command here invokes training, a model API, Colab, the old analyses or a
network service. Outputs are `request.json`, `plan.json`, and `sweep.json`.
Existing output directories are refused. The synthetic sweep has a deliberately
unusable corpus placeholder; it must not be submitted as a real experiment.
For real use the base spec needs the actual corpus/tokenizer and all training
settings, as with any existing mtsweep input. Real experiment authorisation
and preregistration remain separate from producing a plan.

## Input and model semantics

`space` declares a complete fixed `base_spec`, factor levels, selected pairwise
interactions, and scope including protocol ID, corpus ID, metric, evaluation
step, backend and precision. Everything in this declaration is hashed. Evidence
must carry that exact `scope_fingerprint`. The model cannot silently pool a
different width, learning rate or corpus when it is not a declared factor.
This is caller-declared scope enforcement, **not authentication of raw receipts**.
An adapter for future real-run receipts can be added later; none of the completed
research is loaded by this implementation.

Factors use the existing taxonomy's canonical slot names. Categorical features
are reference-coded: each non-reference level has one indicator, plus products
for explicitly selected interactions. Numeric levels are treated as categories,
not as a continuous scaling law. The first declared level is the reference.
Only this finite space is supported. No interpolation to an undeclared width or
unseen category is permitted. Legal configurations go through the existing
taxonomy; resolved knobs must agree with the requested values.

An observation block contains one seed and at least two distinct configurations:

```json
{
  "seed": 7,
  "runs": [
    {"settings": {"norm": "layernorm", "activation": "gelu"}, "value": 3.2},
    {"settings": {"norm": "rmsnorm", "activation": "gelu"}, "value": 3.1}
  ]
}
```

The example above illustrates a two-factor declaration. Every observation's
settings must contain exactly its declaration's factors. Merge observations
sharing a seed into **one** block; do not submit independent pair rows with the
same seed. Existing and future seed IDs must be disjoint. A new seed batch can
repeat an architecture already observed: those are genuine new replications,
accounted for in the cost and covariance, not duplicate data.

The model is

    y(seed, configuration) = seed_offset + phi(configuration)' beta + residual.

Each seed offset is arbitrary. Subtract the first run in a seed block from the
other runs, removing that offset and the intercept. If there are m differences
and each run has residual variance sigma², their covariance is

    R = sigma² (I_m + 1_m 1_m').

The shared anchor matters: treating these differences as independent would
double-count information. Both evidence fitting and future batch planning use
the correlated block. The prior is beta ~ Normal(0, tau² I). The posterior uses

    precision = I/tau² + sum_blocks X' R^-1 X
    mean = precision^-1 sum_blocks X' R^-1 delta_y.

The prior regularises incomplete designs rather than claiming they identify
every coefficient. Reports expose unidentified contrast directions and the
posterior/prior variance ratio. Results are prior-sensitive; the noise variance
and prior variance are **declared**, not fitted or calibrated here.

## Questions, fingerprints and acquisition

A target is a named weighted sum of configurations whose weights sum to zero.
Two terms encode a substitution; four encode a difference-in-differences
interaction. This permits scientific questions such as whether changing
normalisation has a different effect under GELU and ReLU. Each target has an
optional positive importance weight.

For a contrast c, the conditional mean is c' mu and variance c' Sigma c.
The report also enumerates single-slot substitutions around the requested
baseline, retaining the baseline, substitute, observation support and prior
dependence. A substitution's prediction therefore changes with its context.
Absolute validation loss is not predicted: the paired data identify contrasts.

At each selection step, compute the expected weighted target-variance reduction
from adding a candidate to the **whole proposed matched-seed block**. Divide by
incremental cost and select the highest positive score, with canonical lexical
tie-breaking. Recompute covariance before selecting another candidate. The
baseline is charged once per future seed, on the first selection. Every later
candidate costs one run per future seed. Declared costs must be positive, finite,
and cover every legal candidate. Budget limits are enforced.

The covariance update depends on the design, prior and fixed noise, not on the
observed loss values. Posterior means do depend on those values. Thus this is a
targeted information-design policy, not an outcome-adaptive improvement policy.
It is greedy and makes no claim of global budget optimality.

The exported sweep uses one linked factor called `candidate`, containing the
complete dotted-path assignment for each selected configuration and baseline.
This produces exactly configurations × seeds, never the Cartesian expansion of
individually selected factor levels. Tests exercise the existing mtsweep
expansion and materialisation in a temporary directory; no trainer is launched.

## Bounds and verification

The module sets OMP, OpenBLAS, MKL and NumExpr threads to one before importing
NumPy. Inputs are bounded at 8 factors, 128 candidate combinations, 48 features,
64 observation seed blocks (2–17 configurations each), 16 proposed candidates,
and 32 future seeds. It does not create background workers. Embedding it in an
already-running process that imported BLAS earlier may require setting those
environment variables before process startup, as the test command does.

Targeted checks only:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -B -m pytest -q tests/test_atlas_designer.py
python -m ruff check tools/atlas_designer.py tools/atlas_designer_demo.py tests/test_atlas_designer.py
```

22 tests passed. They cover a planted interaction in an unobserved synthetic
combination; conditional fingerprints; arbitrary seed offsets; shared-anchor
covariance against an independent matrix solve; covariance contraction; expected
versus realised design information; outcome-independent variance acquisition;
cost sensitivity; deterministic selection; empty evidence; illegal inputs and
scope mismatch; resource bounds; and exact linked-sweep round-trip.

No real-world predictive accuracy or uncertainty coverage has been established.
Important future checks concern omitted higher-order interactions, correlated
architecture-specific seed responses, unequal noise across architectures, and
transfer outside the declared protocol. This implementation makes those model
assumptions visible while providing the missing executable design loop.
