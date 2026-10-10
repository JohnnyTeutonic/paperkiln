"""Tiny synthetic demonstration of ATLAS's new design loop. No training.

python tools/atlas_designer_demo.py --output NEW_DIRECTORY
Writes initial/new evidence, before/after plans, sweep and readable report.
"""
import argparse
import copy
import json
from pathlib import Path

from atlas_designer import Space, run


def example():
    declaration = {
        "scope": {"protocol_id": "synthetic-flex-v1", "corpus_id": "SYNTHETIC-NO-CORPUS",
                  "metric": "synthetic_loss", "metric_step": 200, "backend": "synthetic",
                  "precision": "float64"},
        "base_spec": {"arch": {"preset": "gpt2-nano", "custom": {
            "d": 64, "heads": 4, "layers": 2, "attention": "exact", "position": "learned",
            "norm": "layernorm", "activation": "gelu", "residual": "residual"}},
            "data": {"T": 64, "path": "REPLACE_WITH_PROSPECTIVE_CORPUS"},
            "train": {"optimizer": "adamw", "lr": .001, "batch": 1, "steps": 200},
            "serve": {"on_finish": False}},
        "factors": {"norm": ["layernorm", "rmsnorm"], "activation": ["gelu", "relu"],
                    "residual": ["residual", "highway"]},
        "interactions": [["norm", "activation"], ["norm", "residual"], ["activation", "residual"]],
    }
    space = Space(declaration)
    baseline = {s: levels[0] for s, levels in declaration["factors"].items()}
    a = dict(baseline, norm="rmsnorm")
    b = dict(baseline, activation="relu")
    ab = dict(a, activation="relu")
    evidence = {"scope_fingerprint": space.scope_fingerprint,
                "blocks": [synthetic_block(seed, [baseline, a, b]) for seed in (1, 2)]}
    return {"space": declaration, "observations": evidence,
            "model": {"prior_variance": 1., "noise_variance": .01},
            "design": {"baseline": baseline, "seeds": [101, 102], "budget": 8.,
                       "max_candidates": 3, "cost_unit": "synthetic run units",
                       "costs": [{"settings": s, "per_seed": 1.} for s in space.candidates],
                       "targets": [{"name": "normalisation x activation difference-in-differences",
                                    "terms": [{"settings": s, "weight": w}
                                              for s, w in [(ab, 1), (a, -1), (b, -1), (baseline, 1)]]}],
                       "out_root": "prospective_atlas_runs"}}


def synthetic_value(settings, seed):
    a = float(settings["norm"] == "rmsnorm")
    b = float(settings["activation"] == "relu")
    c = float(settings["residual"] == "highway")
    # Arbitrarily large additive seed offset, exactly removable by blocking.
    return 3. + seed * 7. + .2*a - .1*b + .05*c - .8*a*b + .25*a*c + .1*b*c


def synthetic_block(seed, settings):
    return {"seed": seed, "runs": [{"settings": s, "value": synthetic_value(s, seed)} for s in settings]}


def demonstrate(output):
    if output.exists():
        raise ValueError("output exists; choose a new directory")
    payload = example()
    before = run(payload)
    configurations = [before["baseline"]] + before["selected"]
    updated = copy.deepcopy(payload)
    updated["observations"]["blocks"] += [synthetic_block(seed, configurations) for seed in before["seeds"]]
    updated["design"]["seeds"] = [201, 202]
    after = run(updated)
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("request.json", payload), ("plan.json", before), ("sweep.json", before["sweep"]),
                        ("updated_request.json", updated), ("updated_plan.json", after)):
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    b, a = before["targets"][0], after["targets"][0]
    lines = ["# ATLAS synthetic design loop", "", "Synthetic software demonstration; no empirical architectural finding.", "",
             "The initial observations vary normalisation and activation separately.",
             "The requested target is their difference-in-differences interaction.", "",
             f"Planted synthetic interaction: -0.8. Initial posterior mean: {b['mean']:.6f}; SD: {b['conditional_sd']:.6f}.",
             f"Planner selects {len(before['selected'])} configurations plus a shared baseline over two new seeds:", ""]
    lines += [f"- `{s}`" for s in before["selected"]]
    lines += ["", f"Cost: {before['spent']:.0f}/{before['budget']:.0f} declared synthetic run units.",
              f"Expected interaction variance: {b['variance']:.6f} -> {b['planned_variance']:.6f}.",
              "After generating only the selected *synthetic* observations and ingesting them:",
              f"posterior mean {a['mean']:.6f}, SD {a['conditional_sd']:.6f}, variance {a['variance']:.6f}.", "",
              "The second plan uses fresh seed IDs. Shared-baseline covariance is retained.",
              "All uncertainty is conditional on the declared Gaussian model/prior/noise.",
              "No real datasets, checkpoints, completed studies, training processes or external services were used.", "",
              "sweep.json demonstrates the mtsweep interface. Its corpus path is deliberately a placeholder;",
              "a real prospective protocol and data/tokenizer configuration are required before execution.", ""]
    (output / "DEMO.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Synthetic design loop written to {output}; no training launched.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    demonstrate(parser.parse_args().output)
