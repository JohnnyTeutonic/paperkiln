#!/usr/bin/env python3
"""D3 smoke test for PT2: every branch of every rule fires on fabricated
results, and the arm construction keeps its promises."""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as A  # noqa: E402
import run_pt2 as P  # noqa: E402

MODELS = ("base", "E", "J10", "R10", "J25", "R25", "X")


def fabricate(gap_j_minus_r, truth_e_minus_j=0.05, noise=0.01, seed=0):
    rng = random.Random(seed)
    res = []
    for s in range(101, 111):
        m = {}
        for name in MODELS:
            m[name] = {"truth_acc": 0.4, "hack_frac": 0.2, "distractor_frac": 0.05}
            for q in A.QS:
                m[name][f"judge_truth_gap_{q}"] = 0.1
        for q in A.QS:
            m[f"J{q}"][f"judge_truth_gap_{q}"] += gap_j_minus_r + rng.gauss(0, noise)
            m["E"]["truth_acc"] += truth_e_minus_j + rng.gauss(0, noise)
        res.append({"seed": s, "K": 100, "confusion": {}, "models": m})
    return res


def check(label, res, h, q, want):
    got = A.decide(res)[h][q]["verdict"]
    assert got == want, (label, h, q, got)
    print(f"  {label:34s} {h} q=0.{q}: {got}")


def main():
    check("judge gap above noise", fabricate(+0.05), "H1", "10", "SUPPORTED")
    check("judge gap below noise", fabricate(-0.05), "H1", "25", "CONTRADICTED")
    check("no difference", fabricate(0.0, noise=0.02), "H1", "10", "NOT_RESOLVED")
    check("labels help", fabricate(0.0, truth_e_minus_j=0.05), "H2", "10", "SUPPORTED")
    check("labels hurt", fabricate(0.0, truth_e_minus_j=-0.05), "H2", "25", "CONTRADICTED")
    check("H3 two-sided, nothing there", fabricate(0.0), "H3", "10", "NOT_RESOLVED")
    for bad in (fabricate(0.0)[:9], [dict(r, models={k: v for k, v in r["models"].items() if k != "R25"})
                                       for r in fabricate(0.0)]):
        try:
            A.decide(bad)
        except SystemExit as e:
            print(f"  refused: {e}")
        else:
            raise AssertionError("decide accepted an incomplete design")

    # Arm construction: equal sizes, deterministic, and the noise-matched arm
    # reproduces the judge's confusion rates in expectation.
    rng = random.Random(5)
    scored = []
    for _ in range(4000):
        e = int(rng.random() < 0.3)
        scored.append({"exact": e, "J10": e if e else int(rng.random() < 0.10),
                       "J25": e if e else int(rng.random() < 0.25)})
    arms, K, conf, raw = P.build_arms(scored, 101)
    assert all(len(v) == K for v in arms.values()) and K > 0
    assert arms == P.build_arms(scored, 101)[0]
    assert abs(conf["J25"]["FA"] - 0.25) < 0.03 and conf["J25"]["FR"] == 0.0
    assert abs(raw["R25"] - raw["J25"]) < 0.08 * raw["J25"], (raw["R25"], raw["J25"])
    assert raw["E"] < raw["J10"] < raw["J25"]
    print(f"  arms: K={K}, raw sizes {raw}, J25 FA={conf['J25']['FA']:.3f}")
    print("smoke_analyze: every branch fires; arm construction holds")


if __name__ == "__main__":
    main()
