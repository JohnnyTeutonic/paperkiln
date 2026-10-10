"""ATLAS: learn conditional substitutions and design informative new experiments.

Plan-only. Never invokes a trainer, network, or historical-study analysis.
Usage: python tools/atlas_designer.py request.json --output NEW_DIRECTORY
NumPy is the sole non-standard dependency. Work is bounded and single-threaded.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import os
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np  # noqa: E402
from atlas_taxonomy import TAXONOMY, spec_assignment, violations  # noqa: E402

MAX_CANDIDATES = 128
MAX_FEATURES = 48
MAX_BLOCKS = 64
MAX_BATCH = 16


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def set_path(spec, path, value):
    parts = path.split(".")
    for part in parts[:-1]:
        spec = spec.setdefault(part, {})
    spec[parts[-1]] = value


class Space:
    """Finite, explicit categorical basis; fixed controls are hashed into scope."""

    def __init__(self, declaration):
        self.declaration = copy.deepcopy(declaration)
        self.scope = declaration["scope"]
        required = {"protocol_id", "corpus_id", "metric", "metric_step", "backend", "precision"}
        if not required.issubset(self.scope) or any(self.scope[k] in (None, "") for k in required):
            raise ValueError("scope must identify protocol, corpus, metric/step, backend and precision")
        self.base = declaration["base_spec"]
        self.factors = declaration["factors"]
        if not self.factors or len(self.factors) > 8:
            raise ValueError("declare 1..8 factors")
        for slot, levels in self.factors.items():
            if slot not in TAXONOMY or TAXONOMY[slot].get("status") != "implemented" or not TAXONOMY[slot].get("path"):
                raise ValueError(f"unsupported factor {slot}")
            if len(levels) < 2 or len(set(map(canonical, levels))) != len(levels):
                raise ValueError(f"factor {slot} needs distinct levels")
            allowed = TAXONOMY[slot].get("alternatives")
            if allowed and any(v not in allowed for v in levels):
                raise ValueError(f"unknown level of {slot}")
            if not allowed and any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in levels):
                raise ValueError(f"invalid numeric level of {slot}")
        if math.prod(len(v) for v in self.factors.values()) > MAX_CANDIDATES:
            raise ValueError("candidate product exceeds resource bound")
        self.columns = [((slot, level),) for slot, levels in self.factors.items() for level in levels[1:]]
        pairs = declaration.get("interactions", [])
        seen = set()
        for pair in pairs:
            if len(pair) != 2 or pair[0] == pair[1] or any(s not in self.factors for s in pair):
                raise ValueError("interaction requires two different declared factors")
            if frozenset(pair) in seen:
                raise ValueError("duplicate interaction")
            seen.add(frozenset(pair))
            a, b = pair
            self.columns += [((a, x), (b, y)) for x in self.factors[a][1:] for y in self.factors[b][1:]]
        if len(self.columns) > MAX_FEATURES:
            raise ValueError("feature count exceeds resource bound")
        self.names = [" * ".join(f"{s}={v}" for s, v in col) for col in self.columns]
        self.candidates = []
        self.excluded = []
        for values in itertools.product(*self.factors.values()):
            settings = dict(zip(self.factors, values))
            try:
                self.spec(settings)
            except ValueError as exc:
                self.excluded.append({"settings": settings, "reason": str(exc)})
            else:
                self.candidates.append(settings)
        if not self.candidates:
            raise ValueError("no legal candidates")
        self.scope_fingerprint = digest(declaration)

    def spec(self, settings):
        if set(settings) != set(self.factors):
            raise ValueError("settings must contain exactly the declared factors")
        spec = copy.deepcopy(self.base)
        for slot, value in settings.items():
            if value not in self.factors[slot]:
                raise ValueError(f"unknown level: {slot}={value}")
            set_path(spec, TAXONOMY[slot]["path"], value)
        assignment = spec_assignment(spec)
        for slot, value in assignment.items():
            allowed = TAXONOMY[slot].get("alternatives")
            if allowed and value not in allowed:
                raise ValueError(f"unknown fixed or varied setting: {slot}={value}")
            if slot in ("d", "heads", "layers", "T", "batch") and (type(value) is not int or value <= 0):
                raise ValueError(f"{slot} must be a positive integer")
        positive(assignment["lr"], "learning rate")
        errors = violations(assignment)
        # Verify that family resolution did not silently change requested knobs.
        if any(assignment[s] != v for s, v in settings.items()):
            errors.append("resolved architecture differs from requested settings")
        if errors:
            raise ValueError("; ".join(errors))
        return spec

    def vector(self, settings):
        self.spec(settings)
        return np.array([float(all(settings[s] == v for s, v in col)) for col in self.columns])

    def contrast(self, terms):
        weights = [t["weight"] for t in terms]
        if not weights or any(type(w) not in (int, float) or not math.isfinite(w) for w in weights):
            raise ValueError("finite contrast weights required")
        if not math.isclose(sum(weights), 0., abs_tol=1e-12):
            raise ValueError("contrast weights must sum to zero (seed/intercept cancellation)")
        vector = sum((t["weight"] * self.vector(t["settings"]) for t in terms), np.zeros(len(self.columns)))
        if np.linalg.norm(vector) < 1e-12:
            raise ValueError("zero contrast in the declared basis")
        return vector


def block_information(x, noise_variance):
    """Contrasts share an anchor: R = sigma²(I + 11'), NOT 2sigma² I."""
    if len(x) == 0:
        return np.zeros((x.shape[1], x.shape[1]))
    total = x.sum(axis=0)
    return (x.T @ x - np.outer(total, total) / (len(x) + 1)) / noise_variance


class Model:
    def __init__(self, space, observations, prior_variance=1., noise_variance=.01):
        self.space = space
        self.prior_variance = positive(prior_variance, "prior_variance")
        self.noise_variance = positive(noise_variance, "noise_variance")
        if observations["scope_fingerprint"] != space.scope_fingerprint:
            raise ValueError("observation scope mismatch")
        blocks = observations["blocks"]
        if len(blocks) > MAX_BLOCKS:
            raise ValueError("too many seed blocks")
        p = len(space.columns)
        self.information = np.zeros((p, p))
        rhs = np.zeros(p)
        self.seeds = set()
        self.observed_settings = set()
        for block in blocks:
            seed = block["seed"]
            if type(seed) is not int or not 0 <= seed < 2**32 or seed in self.seeds:
                raise ValueError("each seed must appear in exactly one block")
            self.seeds.add(seed)
            runs = block["runs"]
            if not 2 <= len(runs) <= MAX_BATCH + 1:
                raise ValueError("seed block needs 2..17 unique configurations")
            keys = [canonical(r["settings"]) for r in runs]
            if len(set(keys)) != len(keys):
                raise ValueError("duplicate configuration within seed block")
            self.observed_settings.update(keys)
            losses = [r["value"] for r in runs]
            if any(type(y) not in (int, float) or not math.isfinite(y) for y in losses):
                raise ValueError("non-finite outcome")
            vectors = [space.vector(r["settings"]) for r in runs]
            x = np.array([v - vectors[0] for v in vectors[1:]])
            y = np.array(losses[1:]) - losses[0]
            self.information += block_information(x, self.noise_variance)
            rhs += x.T @ (y - y.sum() / (len(y) + 1)) / self.noise_variance
        self.precision = np.eye(p) / self.prior_variance + self.information
        self.covariance = np.linalg.solve(self.precision, np.eye(p))
        self.mean = self.covariance @ rhs
        eigenvalues, eigenvectors = np.linalg.eigh(self.information)
        cutoff = max(1., float(eigenvalues.max(initial=0))) * 1e-10
        self.null_basis = eigenvectors[:, eigenvalues < cutoff]

    def describe(self, vector):
        variance = max(0., float(vector @ self.covariance @ vector))
        prior = self.prior_variance * float(vector @ vector)
        return {"mean": float(vector @ self.mean), "conditional_sd": math.sqrt(variance),
                "variance": variance, "prior_variance": prior,
                "posterior_to_prior_variance": variance / prior,
                "has_unidentified_component": bool(np.linalg.norm(self.null_basis.T @ vector) > 1e-8)}

    def fingerprint(self, baseline):
        base_vector = self.space.vector(baseline)
        rows = []
        for slot, levels in self.space.factors.items():
            for value in levels:
                if value == baseline[slot]:
                    continue
                target = dict(baseline, **{slot: value})
                try:
                    vector = self.space.vector(target) - base_vector
                except ValueError:
                    continue
                rows.append({"slot": slot, "baseline": baseline, "substitute": target,
                             "target_observed": canonical(target) in self.observed_settings,
                             **self.describe(vector)})
        return rows


def design(model, request):
    space = model.space
    baseline = request["baseline"]
    base_vector = space.vector(baseline)
    seeds = request["seeds"]
    if not seeds or len(seeds) > 32 or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("declare 1..32 distinct integer future seeds")
    if set(seeds) & model.seeds:
        raise ValueError("planned seed blocks must be new, not reuse observed seeds")
    budget = positive(request["budget"], "budget")
    limit = request.get("max_candidates", 4)
    if type(limit) is not int or not 1 <= limit <= MAX_BATCH:
        raise ValueError("max_candidates must be 1..16")
    targets = request["targets"]
    if not 1 <= len(targets) <= MAX_FEATURES or len({t["name"] for t in targets}) != len(targets):
        raise ValueError("declare unique named targets")
    contrasts = np.array([space.contrast(t["terms"]) for t in targets])
    weights = np.array([positive(t.get("importance", 1), "target importance") for t in targets])
    costs = request["costs"]
    # Each cost is per architecture/seed training run; anchor paid once per seed.
    cost_by_key = {}
    for entry in costs:
        space.spec(entry["settings"])
        key = canonical(entry["settings"])
        if key in cost_by_key:
            raise ValueError("duplicate candidate cost")
        cost_by_key[key] = positive(entry["per_seed"], "per-seed cost")
    keys = {canonical(s) for s in space.candidates}
    if keys != set(cost_by_key):
        raise ValueError("provide cost for every legal candidate, and no others")
    base_key = canonical(baseline)
    n = len(seeds)

    def covariance(settings):
        if not settings:
            return model.covariance
        x = np.array([space.vector(s) - base_vector for s in settings])
        precision = model.precision + n * block_information(x, model.noise_variance)
        return np.linalg.solve(precision, np.eye(len(model.mean)))

    def variances(cov):
        return np.einsum("ij,jk,ik->i", contrasts, cov, contrasts)

    selected, audit = [], []
    spent = 0.
    cov = model.covariance
    initial = variances(cov)
    while len(selected) < limit:
        before = variances(cov)
        scored = []
        for settings in space.candidates:
            key = canonical(settings)
            if key == base_key or settings in selected:
                continue
            incremental = n * (cost_by_key[key] + (cost_by_key[base_key] if not selected else 0))
            if spent + incremental > budget + 1e-12:
                continue
            after_cov = covariance(selected + [settings])
            after = variances(after_cov)
            gain = float(weights @ (before - after))
            if gain < -1e-9:
                raise ArithmeticError("posterior target variance increased")
            scored.append((max(gain, 0.) / incremental, key, settings, incremental, after_cov, after))
        if not scored:
            break
        best = sorted(scored, key=lambda item: (-item[0], item[1]))[0]
        score, _, settings, incremental, cov_next, after = best
        if score <= 1e-14:
            break
        selected.append(settings)
        spent += incremental
        audit.append({"settings": settings, "incremental_cost": incremental,
                      "cumulative_cost": spent, "gain_per_cost": score,
                      "target_variance_before": before.tolist(), "target_variance_after": after.tolist(),
                      "feasible_alternatives": len(scored)})
        cov = cov_next
    configurations = [baseline] + selected if selected else []
    levels = [{TAXONOMY[s]["path"]: v for s, v in config.items()} for config in configurations]
    sweep = {"base": copy.deepcopy(space.base), "design": "grid",
             "factors": {"candidate": levels}, "seeds": seeds,
             "out_root": request.get("out_root", "atlas_designed_runs")}
    return {"status": "PLAN ONLY; model-conditional information gain, no efficacy claim",
            "scope_fingerprint": space.scope_fingerprint,
            "selected": selected, "baseline": baseline, "seeds": seeds,
            "budget": budget, "spent": spent, "cost_unit": request["cost_unit"],
            "planned_runs": len(configurations) * n, "acquisition": audit,
            "targets": [{"name": t["name"], "terms": t["terms"], **model.describe(c),
                         "planned_variance": float(v)}
                        for t, c, v in zip(targets, contrasts, variances(cov))],
            "initial_weighted_variance": float(weights @ initial),
            "planned_weighted_variance": float(weights @ variances(cov)),
            "conditional_fingerprint": model.fingerprint(baseline), "sweep": sweep,
            "features": space.names, "excluded_illegal_candidates": space.excluded,
            "assumptions": {"prior": "zero-mean independent Gaussian coefficients",
                            "prior_variance": model.prior_variance, "per_run_noise_variance": model.noise_variance,
                            "seed": "arbitrary additive seed offset cancelled by within-seed contrasts",
                            "residuals": "independent homoskedastic Gaussian per-run residuals; shared-anchor covariance retained",
                            "scope": "caller-declared scope identity, not independent receipt authentication",
                            "acquisition": "fixed-basis/fixed-noise variance reduction is outcome-independent",
                            "search": "greedy information per incremental cost; not globally optimal",
                            "uncertainty": "conditional on basis/prior/noise; not empirically calibrated"}}


def run(payload):
    space = Space(payload["space"])
    model = Model(space, payload["observations"], **payload.get("model", {}))
    report = design(model, payload["design"])
    report["input_sha256"] = digest(payload)
    report["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["taxonomy_sha256"] = hashlib.sha256(Path(__file__).with_name("atlas_taxonomy.py").read_bytes()).hexdigest()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("output exists; choose a new directory")
    payload = json.loads(args.request.read_text(encoding="utf-8"))
    report = run(payload)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, value in (("request.json", payload), ("plan.json", report), ("sweep.json", report["sweep"])):
        (args.output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Planned {report['planned_runs']} future runs; cost {report['spent']:g}/{report['budget']:g}. No runs launched.")


if __name__ == "__main__":
    main()
