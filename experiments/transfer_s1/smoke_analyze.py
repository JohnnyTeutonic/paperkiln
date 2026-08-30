# -*- coding: utf-8 -*-
"""Smoke-test transfer_s1/analyze.py on SYNTHETIC runs.

Purpose: an analysis script committed before any data must at least
survive first contact with the data shape it will meet. This fabricates
the directory layout mtsweep produces, with plausible trajectories, and
runs the real script. It proves plumbing, NOT science — the numbers it
prints are meaningless by construction.
"""
import json
import os
import random
import shutil
import subprocess
import sys

ROOT = os.path.join(os.environ.get("TEMP", "/tmp"), "transfer_smoke")
LANES = [("exact", None, None), ("swa", 16, 1), ("swa", 32, 1),
         ("swa", 64, 1), ("swa", 128, 1), ("swa", 64, 0)]


def write_run(base, name, d, seed, lane, rng, offset):
    p = os.path.join(base, "runs", name)
    os.makedirs(p, exist_ok=True)
    att, win, sink = lane
    model = {"event": "model", "family": "gpt2", "d": d, "layers": 2,
             "heads": d // 32, "seed": seed, "attention": att,
             "lr": 0.001, "batch": 4, "T": 256, "vocab": 4096}
    if att == "swa":
        model["window"] = win
        model["sinks"] = sink
    lines = [{"event": "start", "name": name, "steps": 3600}, model]
    val = 8.4
    for step in range(100, 3700, 100):
        val -= 0.09 + rng.uniform(-0.01, 0.01)
        lines.append({"event": "step", "step": step, "loss": val + 0.3})
        lines.append({"event": "eval", "step": step,
                      "val_loss": val + offset * (step / 3600.0)})
    lines.append({"event": "done", "final_step": 3600, "best_val": val,
                  "early_stopped": False, "wall_seconds": 1.0})
    with open(os.path.join(p, "events.jsonl"), "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(json.dumps(ln) + "\n")
    with open(os.path.join(p, "result.json"), "w", encoding="utf-8") as f:
        json.dump({"best_val": val, "final_step": 3600}, f)


def build(base, d, seeds, lanes):
    shutil.rmtree(base, ignore_errors=True)
    rng = random.Random(hash(base) & 0xFFFF)
    for si, seed in enumerate(seeds):
        for li, lane in enumerate(lanes):
            # a lane-dependent offset gives non-degenerate edge signs
            off = (li - 2) * 0.01 + rng.uniform(-0.004, 0.004)
            write_run(base, f"run_{si:02d}{li}_c{li:02d}_s{seed}", d, seed,
                      lane, rng, off)


seeds12 = list(range(21, 33))
build(os.path.join(ROOT, "S"), 256, seeds12, LANES)
build(os.path.join(ROOT, "M"), 512, seeds12, LANES)
build(os.path.join(ROOT, "L"), 1024, [21, 22, 23],
      [LANES[0], LANES[3]])
build(os.path.join(ROOT, "bridge"), 256, [1, 2, 3, 4, 5],
      [LANES[0], LANES[3]])
build(os.path.join(ROOT, "banked"), 256, [1, 2, 3, 4, 5],
      [LANES[0], LANES[3]])
print(f"synthetic arms built under {ROOT}")

here = os.path.dirname(os.path.abspath(__file__))
script = (sys.argv[1] if len(sys.argv) > 1
          else os.path.join(here, "analyze.py"))
rc = subprocess.run(
    [sys.executable, script,
     "--bridge", os.path.join(ROOT, "bridge"),
     "--banked", os.path.join(ROOT, "banked"),
     "--arms", f"S={os.path.join(ROOT, 'S')}",
     f"M={os.path.join(ROOT, 'M')}", f"L={os.path.join(ROOT, 'L')}"],
    check=False)
print(f"\nanalyze.py exit code: {rc.returncode}")
