#!/usr/bin/env python3
"""The mathematics column: probability distributions as mechanisms.

Agreed 17 Sep 2026. The rule that keeps this from cherry-picking the
merely unused: the source is a WHOLE structured list (Wikipedia's list
of probability distributions, harvested 17 Sep 2026, about 200 names);
every entry gets the SAME schema, filled by the model; a distribution
enters a candidate only if one of its PROPERTIES is exactly the property
a DEFICIT lacks, and that property match is a sentence the generator
must write and the judge can read. "Unused" is never a reason.

    python tools/synthesis/maths.py tag        # fill the schema -> results/distributions.jsonl
    python tools/synthesis/maths.py match      # candidates -> results/math_candidates.jsonl
    python tools/synthesis/maths.py generate [--limit N]   # -> results/ideas.jsonl (tagged source=maths)
    python tools/synthesis/maths.py scope    [--limit N]   # reuses pipeline.scope_one
    python tools/synthesis/maths.py report     # -> results/math_ranked.md

Generation and scoping reuse pipeline.py (same judge, same backstops,
same controls discipline); the only new prompt is the property sentence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vocab  # noqa: E402
import pipeline as P  # noqa: E402

OUT = P.OUT

# The harvested list (Wikipedia, "List of probability distributions",
# 17 Sep 2026), by the page's own sections. Families and non-distributions
# (Dirac delta, Dirac comb, phase functions) are kept: the tagger decides.
DISTRIBUTIONS = {
    "discrete, finite support": [
        "Bernoulli", "Rademacher", "Binomial", "Beta-binomial", "Degenerate", "Discrete uniform",
        "Hypergeometric", "Negative hypergeometric", "Poisson binomial", "Fisher's noncentral hypergeometric",
        "Wallenius' noncentral hypergeometric", "Benford's law", "Soliton", "Zipf's law", "Zipf-Mandelbrot",
    ],
    "discrete, infinite support": [
        "Beta negative binomial", "Boltzmann", "Maxwell-Boltzmann", "Borel", "Discrete phase-type",
        "Extended negative binomial", "Generalized log-series", "Gauss-Kuzmin", "Geometric", "Hermite",
        "Logarithmic", "Mixed Poisson", "Negative binomial", "Compound Poisson", "Parabolic fractal", "Poisson",
        "Displaced Poisson", "Hyper-Poisson", "General Poisson binomial", "Poisson type", "Conway-Maxwell-Poisson",
        "Zero-truncated Poisson", "Polya-Eggenberger", "Skellam", "Skew elliptical", "Yule-Simon",
        "Modified half-normal", "Zeta", "Zipf", "Hardy",
    ],
    "continuous, bounded interval": [
        "Beta", "Four-parameter Beta", "Arcsine", "PERT", "Uniform", "Irwin-Hall", "Bates", "Logit-normal",
        "Dirac delta", "Kent", "Kumaraswamy", "Logit metalog", "Marchenko-Pastur", "Quantile-parameterized",
        "Raised cosine", "Reciprocal", "Triangular", "Trapezoidal", "Truncated normal", "U-quadratic",
        "Von Mises-Fisher", "Bingham", "Wigner semicircle", "Continuous Bernoulli", "Continuous binomial",
    ],
    "directional": [
        "Henyey-Greenstein phase function", "Mie phase function", "Von Mises", "Wrapped normal", "Wrapped exponential",
        "Wrapped Levy", "Wrapped Cauchy", "Wrapped Laplace", "Wrapped asymmetric Laplace", "Dirac comb",
    ],
    "continuous, semi-infinite": [
        "Beta prime", "Birnbaum-Saunders", "Chi", "Noncentral chi", "Chi-squared", "Inverse-chi-squared",
        "Noncentral chi-squared", "Scaled inverse chi-squared", "Dagum", "Exponential", "Exponential-logarithmic",
        "F", "Noncentral F", "Folded normal", "Frechet", "Gamma", "Erlang", "Inverse-gamma", "Generalized gamma",
        "Generalized Pareto", "Gamma/Gompertz", "Gompertz", "Half-normal", "Hartman-Watson", "Hotelling's T-squared",
        "Inverse Gaussian", "Levy", "Log-Cauchy", "Log-Laplace", "Log-logistic", "Log-metalog", "Log-normal", "Lomax",
        "Mittag-Leffler", "Nakagami", "Pareto", "Pearson Type III", "Phase-type", "Phased bi-exponential",
        "Phased bi-Weibull", "Semi-bounded quantile-parameterized", "Rayleigh", "Rayleigh mixture", "Rice",
        "Shifted Gompertz", "Type-2 Gumbel", "Weibull",
    ],
    "continuous, whole real line": [
        "Behrens-Fisher", "Cauchy", "Centralized inverse-Fano", "Chernoff's", "Exponentially modified Gaussian",
        "Gaussian minus exponential", "Expectile", "Fisher-Tippett", "Extreme value", "Log-Weibull", "Fisher's z",
        "Skewed generalized t", "Gamma-difference", "Generalized logistic", "Generalized normal", "Geometric stable",
        "Gumbel", "Holtsmark", "Hyperbolic", "Hyperbolic secant", "Johnson SU", "Landau", "Laplace",
        "Levy skew alpha-stable", "Stable", "Linnik", "Logistic", "Map-Airy", "Metalog", "Normal",
        "Normal-exponential-gamma", "Normal-inverse Gaussian", "Pearson Type IV", "Skew normal", "Student's t",
        "Noncentral t", "Skew t", "Champernowne", "Type-1 Gumbel", "Tracy-Widom", "Voigt", "Chen",
    ],
    "continuous, variable support": [
        "Generalized extreme value", "Tukey lambda", "Wakeby",
    ],
    "mixed discrete/continuous": [
        "Rectified Gaussian", "Compound Poisson-gamma", "Tweedie",
    ],
    "joint / multivariate": [
        "Dirichlet", "Ewens's sampling formula", "Balding-Nichols", "Multinomial", "Multivariate normal",
        "Multivariate t", "Negative multinomial", "Dirichlet negative multinomial",
        "Generalized multivariate log-gamma", "Marshall-Olkin exponential", "Continuous-categorical",
    ],
    "matrix-valued": [
        "Wishart", "Inverse-Wishart", "Lewandowski-Kurowicka-Joe", "Matrix normal", "Matrix t", "Matrix Langevin",
        "Matrix variate beta", "Uniform on a Stiefel manifold",
    ],
    "non-numeric / miscellaneous": [
        "Categorical", "Cantor", "Pearson family",
    ],
}

# The property vocabulary. A deficit LACKS some of these; a distribution
# HAS some of these. Keep it small and concrete so matches mean something.
PROPERTIES = {
    "point-mass-or-abstention": "can place mass on a single point or a null outcome (a first-class 'none')",
    "boundary-mass-on-simplex": "on the simplex, can put mass exactly on faces/vertices (exact zeros)",
    "heavy-tail": "polynomial tails; outliers are expected, not exceptional",
    "bounded-support": "support is a bounded interval or set, so estimates cannot escape it",
    "conjugate-online-update": "closed-form posterior update from a stream, no gradient step",
    "closed-form-entropy-or-moments": "entropy and moments in closed form, so uncertainty is a cheap scalar",
    "reparameterisable": "samples are a differentiable function of parameters and noise",
    "max-entropy-under-constraint": "is the maximum-entropy law given a named constraint",
    "directional-or-periodic": "lives on a circle/sphere; periodicity is built in",
    "matrix-valued": "a distribution over matrices with structure (PSD, orthogonal, low-rank)",
    "sequential-or-phase-structure": "encodes a sequence of stages or hitting times",
    "extreme-value": "the law of maxima/minima, with a tail index",
    "stable-or-self-similar": "closed under addition/scaling; same shape at every scale",
    "overdispersion-control": "variance can exceed or fall below the mean by a parameter",
    "zero-inflation-or-truncation": "explicit extra mass at zero or explicit truncation",
    "compound-or-mixture": "a random sum or scale mixture; models heterogeneity",
    "quantile-parameterised": "parameterised directly by quantiles, so tails are set explicitly",
    "discrete-count": "a law over counts",
    "memoryless": "hazard is constant; the past does not matter",
    "infinitely-divisible": "can be split into arbitrarily many iid parts",
    "log-concave": "unimodal with light tails; optimisation over it is easy",
    "multivariate-dependence": "expresses dependence between components explicitly",
    "order-statistics": "the law of ranks or of the k-th largest",
    "random-matrix-spectrum": "the law of eigenvalues of large random matrices",
    "long-memory-or-fractional": "power-law memory or fractional-order dynamics",
}

# What each deficit LACKS, in property terms. Empty means no distributional
# property fits, which is allowed: it yields no candidates rather than a
# forced one. Keyed like vocab.FIELD_TERMS, by component then a substring
# of the deficit text.
DEFICIT_PROPERTIES = {
    "kv-cache": {
        "append-only": ["conjugate-online-update"],
        "uniform precision": ["heavy-tail", "quantile-parameterised", "compound-or-mixture"],
        "no principled forgetting": ["memoryless", "sequential-or-phase-structure", "extreme-value"],
        "private to one request": [],
    },
    "attention-scores": {
        "sums to one": ["point-mass-or-abstention", "boundary-mass-on-simplex", "zero-inflation-or-truncation"],
        "heads are scored independently": ["multivariate-dependence", "random-matrix-spectrum"],
        "similarity is bilinear": ["directional-or-periodic", "matrix-valued"],
    },
    "moe-router": {
        "scored independently": ["multivariate-dependence", "order-statistics"],
        "not calibrated": ["conjugate-online-update", "closed-form-entropy-or-moments"],
        "open-loop": ["conjugate-online-update", "sequential-or-phase-structure"],
        "load balancing": ["max-entropy-under-constraint", "overdispersion-control"],
    },
    "residual-stream": {
        "additive only": ["stable-or-self-similar", "compound-or-mixture"],
        "one width": ["random-matrix-spectrum", "heavy-tail"],
    },
    "positional-encoding": {
        "fixed function of the integer index": ["directional-or-periodic", "long-memory-or-fractional"],
        "extrapolation": ["extreme-value", "stable-or-self-similar", "heavy-tail"],
    },
    "tokeniser": {
        "static": ["compound-or-mixture", "sequential-or-phase-structure"],
        "frequency-driven": ["max-entropy-under-constraint", "order-statistics"],
    },
    "output-head": {
        "calibrated": ["conjugate-online-update", "quantile-parameterised", "compound-or-mixture"],
        "abstention": ["point-mass-or-abstention", "zero-inflation-or-truncation"],
        "one-token horizon": ["sequential-or-phase-structure", "compound-or-mixture"],
    },
    "sampler": {
        "sequential": ["infinitely-divisible", "sequential-or-phase-structure"],
        "no rollback": ["order-statistics", "extreme-value"],
        "global knobs": ["quantile-parameterised", "closed-form-entropy-or-moments", "heavy-tail"],
    },
    "prefill": {
        "before any token": ["extreme-value", "order-statistics"],
        "quadratic": ["random-matrix-spectrum", "stable-or-self-similar"],
    },
    "speculative-drafting": {
        "tokens only": [],
        "draft length": ["memoryless", "sequential-or-phase-structure", "conjugate-online-update"],
    },
    "layer-schedule": {
        "same depth": ["sequential-or-phase-structure", "memoryless"],
        "no loop": ["sequential-or-phase-structure", "conjugate-online-update"],
    },
    "ffn": {
        "dense": ["boundary-mass-on-simplex", "zero-inflation-or-truncation"],
        "never edited": ["conjugate-online-update", "matrix-valued"],
    },
    "normalisation": {
        "per token": ["heavy-tail", "stable-or-self-similar", "log-concave"],
    },
    "optimiser-state": {
        "structureless": ["matrix-valued", "random-matrix-spectrum"],
        "discarded": ["matrix-valued", "conjugate-online-update"],
    },
    "lr-schedule": {
        "open-loop": ["conjugate-online-update", "extreme-value"],
    },
    "training-objective": {
        "single horizon": ["sequential-or-phase-structure", "long-memory-or-fractional"],
        "no calibration": ["closed-form-entropy-or-moments", "quantile-parameterised"],
    },
    "data-order": {
        "no replay": ["memoryless", "order-statistics", "compound-or-mixture"],
        "mixture is fixed": ["compound-or-mixture", "conjugate-online-update", "max-entropy-under-constraint"],
    },
    "weights-at-inference": {
        "static": ["conjugate-online-update", "matrix-valued"],
    },
    "attention-topology": {
        "fixed before the content": ["boundary-mass-on-simplex", "order-statistics"],
        "causal masking forbids": [],
    },
    "gradient": {
        "noisy": ["heavy-tail", "stable-or-self-similar", "compound-or-mixture", "random-matrix-spectrum"],
    },
    "checkpoints": {
        "trajectory": ["stable-or-self-similar", "conjugate-online-update"],
    },
    "context-window": {
        "uniform cost": ["heavy-tail", "extreme-value", "quantile-parameterised"],
    },
    "multi-head": {
        "always all on": ["boundary-mass-on-simplex", "multivariate-dependence"],
    },
    "serving-batch": {
        "independent": ["compound-or-mixture", "multivariate-dependence"],
    },
}


def deficit_properties(component, deficit):
    for key, props in DEFICIT_PROPERTIES.get(component, {}).items():
        if key in deficit:
            return props
    return []


# ------------------------------------------------------------------- tag
TAG_SYSTEM = """You are a mathematical statistician filling a fixed schema for probability distributions so that a search tool can match their properties against needs in transformer language models. For each named distribution return a JSON object with exactly these keys:
- name
- is_distribution: true/false (false for phase functions, the Dirac comb, whole families)
- support: short phrase
- properties: a list drawn ONLY from this fixed vocabulary, including a property only if it is a defining or well-known feature of this distribution: """ + ", ".join(PROPERTIES) + """
- signature: one sentence: the property or closed form that makes this distribution different from its nearest relative (e.g. Cauchy: no mean, stable with index 1)
- known_ml_uses: list of short phrases for where this distribution is already used in machine learning or deep learning (e.g. Gumbel: Gumbel-softmax / Gumbel-max sampling; Dirichlet: topic models, Dirichlet priors on mixtures; von Mises-Fisher: directional embeddings). Empty list if you know of none.
- used_at_components: which of these transformer components it is already used at, if any: kv-cache, attention-scores, moe-router, residual-stream, positional-encoding, tokeniser, output-head, sampler, prefill, speculative-drafting, layer-schedule, ffn, normalisation, optimiser-state, lr-schedule, training-objective, data-order, weights-at-inference, attention-topology, gradient, checkpoints, context-window, multi-head, serving-batch. Empty list if none.
Be precise and conservative; a wrong property here produces a false research lead. Return a JSON array, nothing else."""


def stage_tag(batch_size=12):
    path = os.path.join(OUT, "distributions.jsonl")
    done = {r["name"] for r in P.read_jsonl(path)}
    todo = [(sec, n) for sec, names in DISTRIBUTIONS.items() for n in names if n not in done]
    print(f"tag: {len(todo)} to do ({len(done)} cached)", flush=True)
    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        user = "DISTRIBUTIONS:\n" + "\n".join(f"- {n} (section: {sec})" for sec, n in batch)
        try:
            out = P.parse_json(P.ask(P.GEN_MODEL, TAG_SYSTEM, user, max_tokens=8000))
        except Exception as e:
            print(f"  [tag] batch failed: {e}", flush=True)
            continue
        if isinstance(out, dict):
            out = [out]
        by = {o.get("name"): o for o in out if isinstance(o, dict)}
        rows = []
        for sec, n in batch:
            o = by.get(n)
            if not o:
                continue
            o["section"] = sec
            o["properties"] = [p for p in (o.get("properties") or []) if p in PROPERTIES]
            rows.append(o)
        P.append_jsonl(path, rows)
        print(f"  batch {i // batch_size + 1}: {len(rows)}/{len(batch)} tagged", flush=True)


# ----------------------------------------------------------------- match
def stage_match():
    dists = [d for d in P.read_jsonl(os.path.join(OUT, "distributions.jsonl")) if d.get("is_distribution")]
    rows = []
    for c, cd in vocab.COMPONENTS.items():
        for deficit, ops in cd["deficits"]:
            lacks = deficit_properties(c, deficit)
            if not lacks:
                continue
            for d in dists:
                hit = sorted(set(lacks) & set(d.get("properties") or []))
                if not hit:
                    continue
                if c in (d.get("used_at_components") or []):
                    continue  # already imported at this component; the scoper would find it anyway
                cid = hashlib.sha1(f"math|{c}|{deficit}|{d['name']}".encode()).hexdigest()[:10]
                rows.append({"id": cid, "source": "maths", "component": c, "deficit": deficit,
                             "mechanism": d["name"], "ops": ops, "properties": hit,
                             "signature": d.get("signature"), "known_ml_uses": d.get("known_ml_uses")})
    path = os.path.join(OUT, "math_candidates.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    per = {}
    for r in rows:
        per[r["component"]] = per.get(r["component"], 0) + 1
    print(f"match: {len(rows)} candidates from {len(dists)} distributions; per component: "
          + ", ".join(f"{k} {v}" for k, v in sorted(per.items(), key=lambda t: -t[1])))


# -------------------------------------------------------------- generate
MATH_GEN_SYSTEM = P.GEN_SYSTEM.replace(
    "For each triple return a JSON object with exactly these keys:",
    "Here the MECHANISM is a probability distribution with a named PROPERTY that the deficit lacks. The proposal must use THAT property, and the property_sentence must say, in one sentence a statistician would accept, why this distribution's property supplies exactly what the component lacks. If the distribution's known ML uses already cover this placement, reject. If the property match is nominal (the tag applies but the property does no work at this component), reject.\n\nFor each triple return a JSON object with exactly these keys:\n- property_sentence",
)


def gen_batch(batch):
    lines = []
    for r in batch:
        comp = vocab.COMPONENTS.get(r["component"], {}).get("what", r["component"])
        lines.append(f"- id {r['id']}: COMPONENT {r['component']} ({comp}); DEFICIT: {r['deficit']}; "
                     f"DISTRIBUTION {r['mechanism']} (signature: {r.get('signature')}; matched property: "
                     f"{', '.join(r['properties'])}: {'; '.join(PROPERTIES[p] for p in r['properties'])}; "
                     f"known ML uses: {r.get('known_ml_uses')})")
    user = ("TESTBED: " + vocab.TESTBED + "\n\nINVARIANTS available: " + "; ".join(vocab.INVARIANTS)
            + "\n\nTRIPLES:\n" + "\n".join(lines))
    out = None
    for attempt in range(2):
        try:
            out = P.parse_json(P.ask(P.GEN_MODEL, MATH_GEN_SYSTEM, user, max_tokens=12000))
            break
        except ValueError:
            user += "\n\nReturn ONLY a valid JSON array. Escape quotes inside strings."
    if out is None:
        return []
    if isinstance(out, dict):
        out = [out]
    by = {o.get("id"): o for o in out if isinstance(o, dict)}
    res = []
    for r in batch:
        o = by.get(r["id"])
        if not o:
            continue
        o.update({k: r[k] for k in ("component", "deficit", "mechanism", "ops", "source", "properties")})
        res.append(o)
    return res


def stage_generate(limit, batch_size=6):
    cands = P.read_jsonl(os.path.join(OUT, "math_candidates.jsonl"))
    done = {r["id"] for r in P.read_jsonl(os.path.join(OUT, "ideas.jsonl"))}
    todo = [c for c in cands if c["id"] not in done]
    if limit:
        todo = todo[:limit]
    print(f"generate(maths): {len(todo)} to do", flush=True)
    for i in range(0, len(todo), batch_size):
        try:
            rows = gen_batch(todo[i:i + batch_size])
        except Exception as e:
            print(f"  [gen] batch failed: {e}", flush=True)
            rows = []
        if rows:
            P.append_jsonl(os.path.join(OUT, "ideas.jsonl"), rows)
            print(f"  batch {i // batch_size + 1}: {sum(1 for r in rows if not r.get('reject'))}/{len(rows)} kept", flush=True)


def stage_scope(limit):
    ideas = [r for r in P.read_jsonl(os.path.join(OUT, "ideas.jsonl"))
             if r.get("source") == "maths" and not r.get("reject")]
    done = {r["id"] for r in P.read_jsonl(os.path.join(OUT, "scoped.jsonl"))}
    todo = [r for r in ideas if r["id"] not in done]
    if limit:
        todo = todo[:limit]
    print(f"scope(maths): {len(todo)} to do", flush=True)
    for k, idea in enumerate(todo, 1):
        try:
            s = P.scope_one(idea)
        except Exception as e:
            s = {"id": idea["id"], "verdict": "ERROR", "error": str(e)[:200]}
        P.append_jsonl(os.path.join(OUT, "scoped.jsonl"), [s])
        print(f"  {k}/{len(todo)} {idea['id']} {s['verdict']:8s} {idea.get('name')}", flush=True)


def stage_report():
    ideas = {r["id"]: r for r in P.read_jsonl(os.path.join(OUT, "ideas.jsonl")) if r.get("source") == "maths"}
    scoped = {r["id"]: r for r in P.read_jsonl(os.path.join(OUT, "scoped.jsonl"))}
    rows = []
    for i, idea in ideas.items():
        if idea.get("reject"):
            continue
        s = scoped.get(i)
        v = s["verdict"] if s else "UNSCOPED"
        rows.append(({"OPEN": 3, "ADJACENT": 1}.get(v, 0) * 10 + 2 * int(idea.get("splash") or 0) + int(idea.get("feasibility") or 0), v, idea, s))
    rows.sort(key=lambda t: -t[0])
    counts = {}
    for r in rows:
        counts[r[1]] = counts.get(r[1], 0) + 1
    lines = ["# Mathematics column: ranked candidates", "", f"{len(rows)} candidates; verdicts: "
             + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())), "", "| # | verdict | splash | feas | name | distribution @ component | property sentence | closest |", "|---|---|---|---|---|---|---|---|"]
    for k, (_, v, idea, s) in enumerate(rows, 1):
        closest = "; ".join(c.get("id", "?") for c in (s or {}).get("closest", [])[:2])
        lines.append(f"| {k} | {v} | {idea.get('splash')} | {idea.get('feasibility')} | {idea.get('name')} | {idea['mechanism']} @ {idea['component']} | {str(idea.get('property_sentence')).replace('|','/')} | {closest} |")
    open(os.path.join(OUT, "math_ranked.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"report: {len(rows)} rows; {counts}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["tag", "match", "generate", "scope", "report", "all"])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.stage in ("tag", "all"):
        stage_tag()
    if a.stage in ("match", "all"):
        stage_match()
    if a.stage in ("generate", "all"):
        stage_generate(a.limit)
    if a.stage in ("scope", "all"):
        stage_scope(a.limit)
    if a.stage in ("report", "all"):
        stage_report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
