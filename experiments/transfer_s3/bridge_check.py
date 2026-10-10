#!/usr/bin/env python3
"""transfer_s3 threat 2, the binary bridge.

Compares the bridge run (Study 2 arm S, seed 21, exact lane, 800 steps on
the binary the new cells use) with the banked Study 2 receipt for the
same run. PASS iff every evaluation at steps 100..800 matches to a relative
1e-5. The panel launches only on PASS.

Usage: python3 bridge_check.py <bridge run dir> [<banked run dir>]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BANKED = os.path.join(HERE, "..", "transfer_s2", "receipts", "s2_S", "run_000_c00_s21")
TOL = 1e-5


def evals(run_dir):
    out, model = {}, None
    with open(os.path.join(run_dir, "events.jsonl"), encoding="utf-8") as f:
        for line in f:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("event") == "model":
                model = ev
            elif ev.get("event") == "eval":
                out[int(ev["step"])] = float(ev["val_loss"])
    return model, out


def main():
    bridge = sys.argv[1]
    banked = sys.argv[2] if len(sys.argv) > 2 else BANKED
    mb, eb = evals(bridge)
    mk, ek = evals(banked)
    for key in ("attention", "d", "heads", "layers", "seed"):
        if mb.get(key) != mk.get(key):
            raise SystemExit(f"FAIL: model events differ on {key}: {mb.get(key)} vs {mk.get(key)}")
    if abs(mb["lr"] - mk["lr"]) > 1e-12:
        raise SystemExit(f"FAIL: learning rates differ: {mb['lr']} vs {mk['lr']}")
    steps = [s for s in range(100, 801, 100)]
    worst = 0.0
    for s in steps:
        if s not in eb or s not in ek:
            raise SystemExit(f"FAIL: evaluation at step {s} missing")
        rel = abs(eb[s] - ek[s]) / abs(ek[s])
        worst = max(worst, rel)
        print(f"  step {s:>4}: bridge {eb[s]:.7f}  banked {ek[s]:.7f}  rel {rel:.2e}")
    verdict = "PASS" if worst <= TOL else "FAIL"
    print(f"worst relative difference {worst:.2e} (tolerance {TOL:.0e}): {verdict}")
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
