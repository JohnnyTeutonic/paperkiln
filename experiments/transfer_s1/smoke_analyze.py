# -*- coding: utf-8 -*-
"""Smoke-test transfer_s1/analyze.py on SYNTHETIC arms.

An analysis committed before any data must survive first contact with
the data's shape — and, just as importantly, each of its decision rules
must be REACHABLE. A primary hypothesis that can never fire is worse
than a wrong one: the study would report "not adopted" whatever the
world did.

Three regimes, each fabricating the mtsweep layout for S (d=256) and
M (d=512), 12 seeds x 6 lanes, plus a 3-seed L arm:

    transfers   M reproduces S's lane ordering (+ noise)
                -> F1 concordance high; "STRUCTURE TRANSFERS" must be
                   adoptable
    scrambled   M reverses S's lane ordering
                -> F1 must NOT adopt
    noise       no lane structure at all
                -> F1 must NOT adopt; nothing significant anywhere

It proves plumbing and rule-firing, NOT science: the trajectories are
constructed, so no verdict here is evidence about attention.
"""
import json
import os
import random
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(os.environ.get("TEMP", "/tmp"), "transfer_smoke")
STEPS, EVERY = 3600, 100
SEEDS = list(range(21, 33))
LANES = [("exact", None, None), ("swa", 16, 1), ("swa", 32, 1),
         ("swa", 64, 1), ("swa", 128, 1), ("swa", 64, 0)]
# a fixed "quality" ordering: index -> offset added to val loss
BASE_ORDER = [0.00, 0.05, 0.03, 0.01, 0.02, 0.04]


def offsets(regime, arm):
    if regime == "noise":
        return [0.0] * len(LANES)
    if regime == "scrambled" and arm == "M":
        return list(reversed(BASE_ORDER))
    return list(BASE_ORDER)


def build(regime, arm, d, seeds, lanes):
    base = os.path.join(ROOT, regime, arm)
    shutil.rmtree(base, ignore_errors=True)
    offs = offsets(regime, arm)
    for si, seed in enumerate(seeds):
        for li, (att, win, sink) in enumerate(lanes):
            name = f"run_{si * len(lanes) + li:03d}_c{li:02d}_s{seed}"
            p = os.path.join(base, "runs", name)
            os.makedirs(p, exist_ok=True)
            rng = random.Random(seed * 1000 + li + hash(arm) % 97)
            model = {"event": "model", "family": "gpt2", "d": d,
                     "layers": 2, "heads": d // 32, "seed": seed,
                     "attention": att, "lr": 0.001, "batch": 4, "T": 256}
            if att == "swa":
                model["window"] = win
                model["sinks"] = sink
            lines = [{"event": "start", "name": name, "steps": STEPS}, model]
            best = 9e9
            for step in range(EVERY, STEPS + 1, EVERY):
                t = step / STEPS
                v = (8.4 - 3.0 * (t ** 0.55) + offs[li]
                     + rng.uniform(-0.010, 0.010))
                best = min(best, v)
                lines.append({"event": "eval", "step": step, "val_loss": v})
            lines.append({"event": "done", "final_step": STEPS,
                          "best_val": best, "early_stopped": False,
                          "wall_seconds": 1.0})
            with open(os.path.join(p, "events.jsonl"), "w",
                      encoding="utf-8") as f:
                for ln in lines:
                    f.write(json.dumps(ln) + "\n")
            with open(os.path.join(p, "result.json"), "w",
                      encoding="utf-8") as f:
                json.dump({"best_val": best, "final_step": STEPS}, f)
    return base


rc_all = 0
for regime in ("transfers", "scrambled", "noise"):
    s = build(regime, "S", 256, SEEDS, LANES)
    m = build(regime, "M", 512, SEEDS, LANES)
    lg = build(regime, "L", 1024, SEEDS[:3], [LANES[0], LANES[3]])
    print("\n" + "#" * 66)
    print(f"# REGIME: {regime}")
    print("#" * 66, flush=True)
    rc = subprocess.run([sys.executable, os.path.join(HERE, "analyze.py"),
                         "--arms", f"S={s}", f"M={m}", f"L={lg}"],
                        check=False).returncode
    rc_all |= rc
    print(f"[{regime}] analyze.py exit {rc}")
print(f"\nsmoke exit: {rc_all}")
