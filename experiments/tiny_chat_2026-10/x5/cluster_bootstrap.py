#!/usr/bin/env python3
"""Descriptive robustness of HX5-main (amendment 5): the item-level paired differences resampled by cluster.

    python experiments/tiny_chat_2026-10/x5/cluster_bootstrap.py --art C:/ml_artifacts/tiny_chat --out x5/cluster_bootstrap.json

The registered test (analyze_x5.py) treats the 2,664 item-seed pairs as independent. Carry-5's items share
templates, so here every pair d(i, s) = [self - twin](X5) - [self - twin](L16 re-run) goes with its cluster and
whole clusters are resampled with replacement (B resamples, seed 0): by item (the three seeds of an item
together), by answer frame (76 over Y's items), by slot (13) and by family x slot. Percentile 90% and 95%
intervals of the mean, and Stage L0's verdict rule applied to them (labelled DESCRIPTIVE), over Y's four
families, over the three planted-fact families (recall, correction, topic return) and over knowledge alone.
"""
import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.dirname(HERE))
import analyze_x5 as ax  # noqa: E402

SETS = {"Y (recall, correction, topic, knowledge)": ("recall", "correction", "topic", "knowledge"),
        "planted facts (recall, correction, topic)": ("recall", "correction", "topic"),
        "knowledge": ("knowledge",)}
CLUSTERS = ("item", "frame", "slot", "family_slot")


def verdict(m, lo90, hi90, lo95, hi95, sesoi=ax.SESOI_X5):
    return ax.verdict({"mean": m, "ci90": [lo90, hi90], "ci95": [lo95, hi95]}, sesoi)


def pct(xs, q):
    xs = sorted(xs)
    k = q * (len(xs) - 1)
    f = int(k)
    return xs[f] + (xs[min(f + 1, len(xs) - 1)] - xs[f]) * (k - f)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--art", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--b", type=int, default=10000)
    a = ap.parse_args(argv)
    items = {it["id"]: it for it in (json.loads(x) for x in open(os.path.join(ROOT, "corpus", "eval", "carry5_v1.jsonl"),
                                                                  encoding="utf-8"))}
    pairs = []  # (id, seed, d)
    for s in ax.SEEDS:
        xr = ax.items_of(ax.post_dir(a.art, ax.X5, ax.CELL, s))
        cr = ax.items_of(ax.post_dir(a.art, ax.CTL, ax.CONTROL, s))
        yx, yc = ax.y_items(xr), ax.y_items(cr)
        assert set(yx) == set(yc)
        pairs += [(i, s, yx[i] - yc[i]) for i in sorted(yx)]
    key = {"item": lambda i: i, "frame": lambda i: items[i]["frame"], "slot": lambda i: items[i]["slot"],
           "family_slot": lambda i: (items[i]["family"], items[i]["slot"])}
    out = {"label": "DESCRIPTIVE (amendment 5 robustness; the registered verdict is analyze_x5.py's)", "B": a.b,
           "seed": 0, "results": {}}
    for name, fams in SETS.items():
        ps = [p for p in pairs if items[p[0]]["family"] in fams]
        m = sum(d for _, _, d in ps) / len(ps)
        rows = {"pairs": len(ps), "mean": m}
        for c in CLUSTERS:
            groups = {}
            for i, s, d in ps:
                groups.setdefault(key[c](i), []).append(d)
            g = list(groups.values())
            sums = [(sum(x), len(x)) for x in g]
            rng = random.Random(0)
            boots = []
            for _ in range(a.b):
                t = n = 0
                for _ in range(len(g)):
                    x, k = sums[rng.randrange(len(g))]
                    t += x
                    n += k
                boots.append(t / n)
            lo90, hi90, lo95, hi95 = pct(boots, 0.05), pct(boots, 0.95), pct(boots, 0.025), pct(boots, 0.975)
            rows[c] = {"clusters": len(g), "ci90": [lo90, hi90], "ci95": [lo95, hi95],
                       "verdict_rule_applied": verdict(m, lo90, hi90, lo95, hi95)}
        out["results"][name] = rows
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(out, indent=1) + "\n")
    for name, rows in out["results"].items():
        print(name, rows["pairs"], round(rows["mean"], 4))
        for c in CLUSTERS:
            r = rows[c]
            print(f"  {c:12s} {r['clusters']:4d} ci90 [{r['ci90'][0]:+.4f}, {r['ci90'][1]:+.4f}] "
                  f"ci95 [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] {r['verdict_rule_applied']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
