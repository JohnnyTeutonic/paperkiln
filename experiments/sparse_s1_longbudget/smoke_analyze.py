# -*- coding: utf-8 -*-
"""Smoke-test sparse_s1_longbudget/analyze.py on SYNTHETIC runs.

An analysis committed before any data must survive first contact with
the shape of the data it will meet. This fabricates the mtsweep layout
for 10 seeds x 2 lanes x 12000 steps and runs the real script.

Three regimes, so the decision rules are exercised rather than merely
executed:

    monotone   both lanes still improving at 12000 (the theorem's
               assumption (iii) intact)  -> expect no second crossing
    overfit    exact turns up before swa (the escape hatch)
               -> expect H-OVERFIT-ORDER supported AND a second crossing
    flat       nothing separates the lanes -> everything inside the zone

It proves plumbing and rule-firing, NOT science: the trajectories are
constructed, so the verdicts are meaningless as evidence about
attention. What they are is a check that each branch of the
pre-registered outcome table can actually be reached by the code.
"""
import json
import os
import random
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(os.environ.get("TEMP", "/tmp"), "longbudget_smoke")
STEPS = 12000
EVERY = 200
SEEDS = list(range(41, 51))


def curve(regime, lane, seed, step, rng):
    """Val loss at `step` for a lane, per regime."""
    t = step / STEPS
    base = 8.4 - 3.2 * (t ** 0.55)
    if regime == "flat":
        return base + rng.uniform(-0.004, 0.004)
    if regime == "monotone":
        # swa ahead early, exact overtakes and stays ahead: one crossing
        adj = 0.10 * (0.35 - t) if lane == "swa" else 0.0
        return base - adj + rng.uniform(-0.004, 0.004)
    # overfit: exact bottoms early then RISES (the bump must outrun the
    # base decay or the regime is not what it claims — the first version
    # of this fixture used 0.55 and produced no interior minimum at all,
    # which Threat 3 correctly reported as 0/10)
    # Shaped like the REAL data so both crossings exist: swa ahead
    # early (S1c), exact overtakes (B*), then exact overfits and swa
    # retakes the lead — the two-crossing signature the prereg's
    # H-SECOND-CROSS is actually about.
    if lane == "exact":
        bump = 6.0 * max(0.0, t - 0.42) ** 2
        return base + bump + 0.10 * (0.30 - t) + rng.uniform(-0.004, 0.004)
    return base + rng.uniform(-0.004, 0.004)


def build(regime):
    base = os.path.join(ROOT, regime)
    shutil.rmtree(base, ignore_errors=True)
    for si, seed in enumerate(SEEDS):
        for li, lane in enumerate(("exact", "swa")):
            name = f"run_{si * 2 + li:03d}_c{li:02d}_s{seed}"
            d = os.path.join(base, "runs", name)
            os.makedirs(d, exist_ok=True)
            rng = random.Random(seed * 100 + li)
            model = {"event": "model", "family": "gpt2", "d": 256,
                     "layers": 2, "heads": 8, "seed": seed,
                     "attention": lane, "lr": 0.001, "batch": 4, "T": 256}
            if lane == "swa":
                model["window"] = 64
                model["sinks"] = 1
            lines = [{"event": "start", "name": name, "steps": STEPS}, model]
            best = 9e9
            for step in range(EVERY, STEPS + 1, EVERY):
                v = curve(regime, lane, seed, step, rng)
                best = min(best, v)
                lines.append({"event": "eval", "step": step, "val_loss": v})
            lines.append({"event": "done", "final_step": STEPS,
                          "best_val": best, "early_stopped": False,
                          "wall_seconds": 1.0})
            with open(os.path.join(d, "events.jsonl"), "w",
                      encoding="utf-8") as f:
                for ln in lines:
                    f.write(json.dumps(ln) + "\n")
            with open(os.path.join(d, "result.json"), "w",
                      encoding="utf-8") as f:
                json.dump({"best_val": best, "final_step": STEPS}, f)
    return base


rc_all = 0
for regime in ("monotone", "overfit", "flat"):
    root = build(regime)
    print("\n" + "#" * 66)
    print(f"# REGIME: {regime}")
    print("#" * 66, flush=True)
    rc = subprocess.run([sys.executable,
                         os.path.join(HERE, "analyze.py"),
                         "--root", root], check=False).returncode
    rc_all |= rc
    print(f"[{regime}] analyze.py exit {rc}")
print(f"\nsmoke exit: {rc_all}")
