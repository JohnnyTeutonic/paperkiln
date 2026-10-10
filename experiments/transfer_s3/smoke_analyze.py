#!/usr/bin/env python3
"""Smoke test for analyze.py, run BEFORE any transfer_s3 data exists.

Fabricates the six cells in the receipt layout (model event + eval events)
for each scenario, runs analyze.main on them, and checks that every
decision branch is reachable and lands where the scenario was built to
land:

  position   M cells reverse the swa ordering once the exact loss falls
             below 3.30 (between targets 3 and 4); rate has no effect
             -> H-POS supported twice, H-RATE supported twice, POSITION
  rate       the 5e-4 cells permute the swa ordering at every target;
             width has no effect
             -> H-POS not supported twice, H-RATE not supported twice, RATE
  both       position reversal in M cells AND a rate permutation
             -> H-POS supported, H-RATE not supported, BOTH
  neither    no width or rate effect at all
             -> H-POS not supported, H-RATE supported, NEITHER
  unreached  M25 plateaus above target 4 -> its H-POS is untested, UNTESTED
  guard      one run records the wrong learning rate -> refuse-to-run

Usage: python3 smoke_analyze.py
"""
import contextlib
import io
import json
import math
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as S3  # noqa: E402

LANES = S3.A.LANES
LANE_FIELDS = {"exact": ("exact", 64, 1), "swa16s1": ("swa", 16, 1), "swa32s1": ("swa", 32, 1),
               "swa64s1": ("swa", 64, 1), "swa128s1": ("swa", 128, 1), "swa64s0": ("swa", 64, 0)}
BASE = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]
PERM = [0.0, 0.10, 0.08, 0.06, 0.04, 0.02]


def exact_curve(step, plateau=False):
    if plateau:
        return 3.30 + 1.40 * math.exp(-step / 1000)
    return 3.10 + 1.60 * math.exp(-step / 1200)


def offsets(scenario, cell, loss):
    d, rate = S3.CELL_SPEC[cell]
    off = PERM if (scenario in ("rate", "both") and rate == 5e-4) else BASE
    if scenario in ("position", "both") and d == 512 and loss < 3.30:
        off = [-o for o in off]
    return off


def write_cells(root, scenario, bad_lr=False):
    paths = {}
    for cell, (d, rate) in S3.CELL_SPEC.items():
        cdir = os.path.join(root, cell)
        paths[cell] = cdir
        plateau = scenario == "unreached" and cell == "M25"
        k = 0
        for seed in range(21, 33):
            for li, lane in enumerate(LANES):
                rd = os.path.join(cdir, f"run_{k:03d}_c{li:02d}_s{seed}")
                k += 1
                os.makedirs(rd)
                att, w, s = LANE_FIELDS[lane]
                lr = rate * (2 if (bad_lr and cell == "S5" and k == 1) else 1)
                model = {"event": "model", "attention": att, "window": w, "sinks": s,
                         "seed": seed, "d": d, "layers": 2, "lr": lr}
                with open(os.path.join(rd, "events.jsonl"), "w", encoding="utf-8") as f:
                    f.write(json.dumps(model) + "\n")
                    for step in range(100, 3601, 100):
                        base = exact_curve(step, plateau)
                        noise = 0.002 * (((seed * 7 + li * 3 + step // 100) % 5) - 2) / 2
                        v = base + offsets(scenario, cell, base)[li] + (noise if li else 0.0)
                        f.write(json.dumps({"event": "eval", "step": step, "val_loss": v}) + "\n")
    return paths


def run(scenario, bad_lr=False):
    root = tempfile.mkdtemp(prefix=f"s3smoke_{scenario}_")
    try:
        paths = write_cells(root, scenario, bad_lr)
        sys.argv = ["analyze.py", "--cells"] + [f"{n}={p}" for n, p in paths.items()]
        if scenario != "position":
            sys.argv.append("--no-band")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = S3.main()
        return out, buf.getvalue()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main():
    expect = {
        "position": (["supported"] * 2, ["supported"] * 2, "POSITION"),
        "rate": (["not supported"] * 2, ["not supported"] * 2, "RATE"),
        "both": (["supported"] * 2, ["not supported"] * 2, "BOTH"),
        "neither": (["not supported"] * 2, ["supported"] * 2, "NEITHER"),
    }
    ok = True
    for sc, (pos_e, rate_e, reading) in expect.items():
        (pos, rate), text = run(sc)
        got = text.split("READING: ")[1].split(":")[0].strip()
        good = pos == pos_e and rate == rate_e and got == reading
        ok &= good
        print(f"{sc:9s} H-POS {pos} H-RATE {rate} READING {got}  {'OK' if good else 'FAIL'}")
        if sc == "position":
            print("          (band computed in this scenario):",
                  [l for l in text.splitlines() if "band [" in l][0].strip())
    (pos, rate), text = run("unreached")
    good = pos[0].startswith("untested") and "UNTESTED" in text
    ok &= good
    print(f"unreached H-POS {pos}  {'OK' if good else 'FAIL'}")
    try:
        run("neither", bad_lr=True)
        print("guard     no refusal  FAIL")
        ok = False
    except SystemExit as e:
        print(f"guard     refused: {str(e)[:70]}...  OK")
    print("SMOKE", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
