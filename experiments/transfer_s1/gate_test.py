# -*- coding: utf-8 -*-
"""End-to-end test of the transfer_s1 numerics-bridge gate.

Builds a synthetic BRIDGE arm from the banked boundary receipts by
injecting the window/sinks fields a real CUDA run will carry, then runs
the real gate against the real banked cohort. Two cases:

  IDENTICAL  bridge == banked exactly -> must PASS (5/5 signs agree,
             pooled means equal).
  PERTURBED  every swa run shifted by +0.15 val loss -> must FAIL
             (signs flip, mean far outside 2 SE).

A gate that cannot fail is not a gate.
"""
import glob
import json
import os
import shutil
import subprocess
import sys

SRC = os.path.expanduser("~/boundary_out")
ROOT = "/tmp/gate_test"
REPO = ("/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete"
        "/microtorch")


def build(dst, shift=0.0):
    shutil.rmtree(dst, ignore_errors=True)
    for d in sorted(glob.glob(os.path.join(SRC, "runs", "*"))):
        out = os.path.join(dst, "runs", os.path.basename(d))
        os.makedirs(out, exist_ok=True)
        is_swa = False
        lines = []
        for line in open(os.path.join(d, "events.jsonl"), encoding="utf-8"):
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("event") == "model":
                if ev.get("attention") == "swa":
                    is_swa = True
                    ev["window"] = 64      # what the new binary emits
                    ev["sinks"] = 1
            elif ev.get("event") == "eval" and is_swa and shift:
                ev["val_loss"] = float(ev["val_loss"]) + shift
            lines.append(json.dumps(ev))
        with open(os.path.join(out, "events.jsonl"), "w",
                  encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        rj = os.path.join(d, "result.json")
        if os.path.exists(rj):
            shutil.copy(rj, os.path.join(out, "result.json"))


def run(label, dst):
    print("\n" + "=" * 60)
    print(label)
    print("=" * 60)
    subprocess.run([sys.executable,
                    os.path.join(REPO, "experiments", "transfer_s1",
                                 "analyze.py"),
                    "--bridge", dst, "--banked", SRC], check=False)


build(f"{ROOT}/same", shift=0.0)
run("CASE 1 — identical data: gate MUST PASS", f"{ROOT}/same")
build(f"{ROOT}/shifted", shift=0.15)
run("CASE 2 — swa lane shifted +0.15: gate MUST FAIL", f"{ROOT}/shifted")
