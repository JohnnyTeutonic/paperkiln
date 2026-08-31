#!/usr/bin/env python3
"""Atlas stage 1: the component taxonomy and the constrained grammar
(ARCHITECTURE_ATLAS.md sections 7 and 12).

The taxonomy declares, for each component SLOT, the legal alternatives
(with aliases), and the compatibility CONSTRAINTS between slots. It is
one shared object with three consumers:
  - corpus generation: sample() draws random VALID architectures
    (corpus source 5 — the correction for publication selection bias)
  - ablation definition: alternatives(slot) is the substitution lattice
    a Delta is measured against
  - validation: validate(spec) refuses illegal configurations before
    compute is spent on them (mtsweep calls this on every materialized
    spec)

Scoped HONESTLY to what the spec system can express today. Slots the
Atlas doc names that are not yet spec-switchable (norm, position,
activation, residual) are declared PLANNED with their lattice recorded,
so the taxonomy is ready the day the spec grows the knob — but sample()
and validate() only touch implemented slots.

    python tools/atlas_taxonomy.py --selftest
    python tools/atlas_taxonomy.py --sample 8 [--seed 3]
    python tools/atlas_taxonomy.py --validate spec.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys

# --------------------------------------------------------------------------
# The taxonomy. status: "implemented" slots are spec-expressible today;
# "planned" slots record the lattice for when the spec grows the knob.
TAXONOMY = {
    "family": {
        "status": "implemented",
        "path": None,  # chosen via arch.preset; see PRESET_OF_FAMILY
        # flex = the paper-faithful decoder (tools/parity_model.hpp
        # FlexLM): norm/activation/position/d_ff/depth all
        # constructor-real. Selected server-side by the presence of
        # flavor knobs, so its preset is just a dims donor.
        "alternatives": ["gpt2", "llama", "flex"],
        "aliases": {"gpt-2": "gpt2", "decoder": "gpt2", "llama2": "llama",
                    "paper": "flex"},
    },
    "attention": {
        "status": "implemented",
        "path": "arch.custom.attention",
        "alternatives": ["exact", "kimi", "srd", "attnres", "swa"],
        "aliases": {"softmax": "exact", "full": "exact",
                    "linear": "kimi", "kimi-linear": "kimi",
                    "surprise": "srd", "surprise-routed": "srd",
                    "attention-residuals": "attnres",
                    "sliding-window": "swa", "sliding_window": "swa"},
    },
    "optimizer": {
        "status": "implemented",
        "path": "train.optimizer",
        "alternatives": ["adamw", "muon"],
        "aliases": {"adam": "adamw"},
    },
    "d": {"status": "implemented", "path": "arch.custom.d",
          "range": [64, 96, 128, 192, 256]},
    "layers": {"status": "implemented", "path": "arch.custom.layers",
               "range": [2, 3, 4, 6]},
    "heads": {"status": "implemented", "path": "arch.custom.heads",
              "range": [2, 4, 8]},
    "T": {"status": "implemented", "path": "data.T",
          "range": [64, 128, 256]},
    "lr": {"status": "implemented", "path": "train.lr",
           "range": [1e-3, 2e-3, 3e-3]},
    "batch": {"status": "implemented", "path": "train.batch",
              "range": [1, 2, 4]},
    # ---- flavor slots: IMPLEMENTED 2026-08-01 (the flex family) ----
    # Formerly planned; norm "none" and position "nope" wait for a knob
    # and stay out of the implemented lattices on purpose.
    "norm": {"status": "implemented", "path": "arch.custom.norm",
             "alternatives": ["layernorm", "rmsnorm"],
             "aliases": {"rms": "rmsnorm", "ln": "layernorm"}},
    "residual": {"status": "implemented", "path": "arch.custom.residual",
                 "alternatives": ["residual", "highway", "plain"],
                 "aliases": {"skip": "residual", "none": "plain",
                              "gated": "highway"}},
    "position": {"status": "implemented", "path": "arch.custom.position",
                 "alternatives": ["learned", "sinusoidal", "rope"],
                 "aliases": {"rotary": "rope", "absolute": "learned"}},
    "activation": {"status": "implemented", "path": "arch.custom.activation",
                   "alternatives": ["gelu", "relu", "swiglu"],
                   "aliases": {"swish-glu": "swiglu", "gelu-mlp": "gelu",
                               "relu-mlp": "relu"}},
    # ---- planned lattices (ARCHITECTURE_ATLAS section 12) ----
    # Renamed from "residual" (12 Aug 2026): that name now belongs to the
    # IMPLEMENTED skip-combine slot above (residual|highway|plain, registry
    # #0001). This planned slot is the norm-placement / stream-wiring axis,
    # a different lattice.
    "stream": {"status": "planned",
               "alternatives": ["pre-norm", "post-norm", "attnres"],
               "aliases": {"preln": "pre-norm", "postln": "post-norm"}},
}

# flex rides a gpt2-dims preset; the flavor knobs in the spec are what
# select the family server-side (mtstudio parse_spec family resolution).
PRESET_OF_FAMILY = {"gpt2": "gpt2-nano", "llama": "llama-tiny",
                    "flex": "gpt2-nano"}
FLAVOR_SLOTS = ("norm", "activation", "position", "residual")

# Compatibility constraints, each a (predicate, message) over a flat
# {slot: value} assignment. The grammar's whole job is that sample()
# can never emit an assignment violating one of these. They mirror
# mtstudio parse_spec exactly.
CONSTRAINTS = [
    (lambda a: a["d"] % a["heads"] == 0,
     "d must divide by heads"),
    (lambda a: a["d"] // a["heads"] >= 8,
     "head_dim below 8 is degenerate at these scales"),
    (lambda a: a["family"] != "llama" or
     (a["attention"] == "exact" and a["position"] == "rope" and
      a["norm"] == "rmsnorm" and a["activation"] == "swiglu"),
     "llama family is the fixed rope/rmsnorm/swiglu block (exact attention)"),
    (lambda a: a["family"] != "gpt2" or
     (a["layers"] == 2 and a["norm"] == "layernorm" and
      a["activation"] == "gelu" and a["position"] == "learned"),
     "gpt2 family is the 2-block parity model with fixed flavors "
     "(vary them via the flex family)"),
    (lambda a: a["family"] != "flex" or
     (a["attention"] in ("exact", "swa") and a["position"] != "rope"),
     "flex family: exact or swa attention (deep SWA, ROADMAP 1a), "
     "position learned|sinusoidal (rope means the llama block)"),
    (lambda a: a["position"] != "rope" or a["family"] == "llama",
     "rope lives inside the llama block"),
    (lambda a: a.get("residual", "residual") == "residual" or
     a["family"] == "flex",
     "highway/plain residual variants live in the flex family "
     "(registry #0001)"),
]


def canonical(slot, value):
    """Resolve an alias to its canonical alternative name."""
    t = TAXONOMY[slot]
    if isinstance(value, str):
        value = t.get("aliases", {}).get(value, value)
    return value


def alternatives(slot):
    """The substitution lattice for a slot (what a Delta is measured
    against)."""
    t = TAXONOMY[slot]
    return t.get("alternatives") or t.get("range")


def violations(assignment):
    out = []
    for pred, msg in CONSTRAINTS:
        try:
            if not pred(assignment):
                out.append(msg)
        except KeyError:
            pass  # slot not present in this assignment; nothing to check
    return out


def sample(n, seed=0):
    """Draw n random VALID assignments over the implemented slots —
    corpus source 5 (random valid architectures from the grammar)."""
    rng = random.Random(seed)
    impl = {s: t for s, t in TAXONOMY.items()
            if t["status"] == "implemented"}
    out = []
    while len(out) < n:
        a = {s: rng.choice(alternatives(s)) for s in impl}
        if not violations(a):
            out.append(a)
    return out


def assignment_to_spec(a, base=None):
    """Turn a sampled assignment into an mtstudio spec fragment."""
    spec = json.loads(json.dumps(base)) if base else {}
    spec.setdefault("arch", {})["preset"] = PRESET_OF_FAMILY[a["family"]]
    for slot, val in a.items():
        path = TAXONOMY[slot].get("path")
        if not path:
            continue
        # Flavor knobs in the spec are what SELECT the flex family
        # server-side, so gpt2/llama assignments (whose flavors the
        # constraints pin to the family defaults) must omit them.
        if slot in FLAVOR_SLOTS and a["family"] != "flex":
            continue
        d = spec
        keys = path.split(".")
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = val
    return spec


def spec_assignment(spec):
    """Extract the implemented-slot assignment from a spec dict (for
    validate)."""
    def get(path, default=None):
        d = spec
        for k in path.split("."):
            if not isinstance(d, dict) or k not in d:
                return default
            d = d[k]
        return d
    preset = get("arch.preset", "")
    # Family resolution mirrors mtstudio parse_spec: rope forces llama,
    # any other flavor knob means flex, else the preset's family.
    pos = canonical("position", get("arch.custom.position"))
    flavored = any(get("arch.custom." + s) for s in FLAVOR_SLOTS)
    if pos == "rope" or (str(preset).startswith("llama") and not flavored):
        family = "llama"
    elif flavored:
        family = "flex"
    else:
        family = "gpt2"
    a = {"family": family}
    defaults = {"d": 128, "layers": 2, "heads": 4, "T": 128,
                "lr": 3e-3, "batch": 1, "attention": "exact",
                "optimizer": "adamw", "residual": "residual"}
    if family == "llama":
        defaults.update(norm="rmsnorm", activation="swiglu",
                        position="rope")
    else:
        defaults.update(norm="layernorm", activation="gelu",
                        position="learned")
    if str(preset).startswith("kimi"):
        defaults["attention"] = "kimi"
    if str(preset).startswith("srd"):
        defaults["attention"] = "srd"
    for slot, dv in defaults.items():
        path = TAXONOMY[slot]["path"]
        a[slot] = canonical(slot, get(path, dv)) if path else dv
    return a


def validate_spec_file(path):
    with open(path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    return violations(spec_assignment(spec))


ROUND_TRIP_SLOTS = ("family", "attention", "optimizer", "d", "layers",
                    "heads", "T", "lr", "batch", "norm", "activation",
                    "position", "residual")


def selftest():
    # canonicalisation
    assert canonical("attention", "softmax") == "exact"
    assert canonical("norm", "rms") == "rmsnorm"
    assert canonical("activation", "gelu-mlp") == "gelu"
    # violations caught
    flav = {"norm": "rmsnorm", "activation": "swiglu", "position": "rope"}
    bad = dict({"family": "llama", "attention": "kimi", "d": 100, "heads": 8,
                "layers": 2, "T": 128, "lr": 3e-3, "batch": 1,
                "optimizer": "adamw"}, **flav)
    v = violations(bad)
    assert any("divide" in m for m in v), v
    assert any("llama" in m for m in v), v
    bad2 = dict(bad, family="gpt2", attention="kimi", d=128, layers=4)
    assert any("2-block" in m for m in violations(bad2))
    # rope outside llama refused; flex sinusoidal accepted
    bad3 = dict(bad2, family="flex", attention="exact", layers=4,
                norm="layernorm", activation="relu", position="rope")
    assert any("rope" in m for m in violations(bad3))
    ok = dict(bad3, position="sinusoidal")
    assert not violations(ok), violations(ok)
    # sampler: everything valid, decent diversity, all families reachable
    # 200 draws: the grammar grew a 3-valued residual slot (registry
    # #0001), so valid gpt2/llama draws are ~3x rarer under rejection
    # sampling and 60 no longer covers all families at this seed.
    s = sample(200, seed=3)
    assert len(s) == 200
    assert all(not violations(a) for a in s)
    assert len({json.dumps(a, sort_keys=True) for a in s}) > 35
    assert {a["family"] for a in s} == {"gpt2", "llama", "flex"}
    # round-trip: sampled assignment -> spec -> assignment, flavors
    # included (gpt2/llama omit flavor paths; their constrained
    # defaults must round-trip identically)
    for fam in ("gpt2", "llama", "flex"):
        a = next(x for x in s if x["family"] == fam)
        spec = assignment_to_spec(a, base={"data": {"T": 999}})
        back = spec_assignment(spec)
        for slot in ROUND_TRIP_SLOTS:
            assert back[slot] == a[slot], (fam, slot, back[slot], a[slot])
    # planned slots still carry their lattices ("stream" is the renamed
    # norm-placement lattice; "residual" is now the implemented
    # skip-combine slot, registry #0001)
    assert "post-norm" in alternatives("stream")
    assert "highway" in alternatives("residual")
    # residual round-trips through a spec like the other flavors
    hw = dict(next(x for x in s if x["family"] == "flex"), residual="highway")
    assert not violations(hw), violations(hw)
    assert spec_assignment(assignment_to_spec(hw))["residual"] == "highway"
    print("SELFTEST OK: aliases, constraints (flex+rope), sampler "
          "validity+diversity+family coverage, 3-family spec round-trip, "
          "planned lattices")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--sample", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--validate")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    if args.sample:
        for a in sample(args.sample, args.seed):
            print(json.dumps(a, sort_keys=True))
        return
    if args.validate:
        v = validate_spec_file(args.validate)
        if v:
            for m in v:
                print(f"VIOLATION: {m}")
            sys.exit(1)
        print("spec valid")
        return
    ap.error("pick one of --selftest / --sample N / --validate spec.json")


if __name__ == "__main__":
    main()
