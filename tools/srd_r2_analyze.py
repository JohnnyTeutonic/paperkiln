#!/usr/bin/env python3
"""SRD rung 2 analysis — the PRE-REGISTERED tests, and only those.

    python tools/srd_r2_analyze.py /tmp/srd_r2 [--md out.md]
    python tools/srd_r2_analyze.py --selftest

Implements experiments/SRD_PREREG_R2.md exactly: RSI (retrieval-selectivity index),
DCI (decoy-chasing index), the concentration check, and the five
predictions P1-P5 with their committed decision rules. Written BEFORE
the runs; the analysis is not allowed to grow new tests after seeing
data — anything else discovered goes in a clearly-labelled EXPLORATORY
section of the results document.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import statistics
import sys

CELLS = [(n, d) for n in ("distinct", "indist") for d in (0, 2)]
SEEDS = (1, 2, 3)


def final_row(path, lane="srd"):
    """Last probe row for a lane."""
    best = None
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["lane"] == lane:
                best = r
    return best


def rsi(row):
    t, n = float(row["target_gate"]), float(row["nontarget_gate"])
    return (t - n) / (t + n) if (t + n) > 0 else 0.0


def dci(row):
    t, d = float(row["target_gate"]), float(row["decoy_gate"])
    return d / t if t > 0 else float("nan")


def conc(row):
    return float(row["tail_gate"]) - float(row["fill_gate"])


def mean_se(xs):
    xs = [x for x in xs if x == x]  # drop nan
    if len(xs) < 2:
        return (xs[0] if xs else float("nan")), float("nan")
    return statistics.fmean(xs), statistics.stdev(xs) / (len(xs) ** 0.5)


def analyse(out_dir):
    lines = ["# SRD rung 2 — pre-registered analysis", ""]
    table = {}
    exact_ok = {}
    for needle, dec in CELLS:
        rows, ex_acc = [], []
        for s in SEEDS:
            p = os.path.join(out_dir, f"{needle}_d{dec}_s{s}_probe.csv")
            if not os.path.exists(p):
                continue
            r = final_row(p, "srd")
            e = final_row(p, "exact")
            if r:
                rows.append(r)
            if e:
                ex_acc.append(float(e["answer_acc"]))
        if not rows:
            continue
        table[(needle, dec)] = rows
        exact_ok[(needle, dec)] = mean_se(ex_acc)

    lines += ["| cell | n | exact acc (control) | RSI | DCI | concentration | srd acc |",
              "|---|---|---|---|---|---|---|"]
    stats = {}
    for key, rows in table.items():
        needle, dec = key
        r_m, r_se = mean_se([rsi(r) for r in rows])
        d_m, _ = mean_se([dci(r) for r in rows]) if dec else (float("nan"), 0)
        c_m, c_se = mean_se([conc(r) for r in rows])
        a_m, _ = mean_se([float(r["answer_acc"]) for r in rows])
        e_m, _ = exact_ok[key]
        stats[key] = {"rsi": (r_m, r_se), "dci": d_m, "conc": (c_m, c_se),
                      "acc": a_m, "exact": e_m}
        lines.append(
            f"| {needle} decoys={dec} | {len(rows)} | {e_m:.3f} | "
            f"{r_m:+.4f} ± {r_se:.4f} | {'—' if dec == 0 else f'{d_m:.3f}'} | "
            f"{c_m:+.4f} ± {c_se:.4f} | {a_m:.3f} |")

    lines += ["", "## Pre-registered predictions", ""]

    def verdict(cond, txt):
        return f"- **{'HOLDS' if cond else 'FAILS'}** — {txt}"

    k_dn = ("distinct", 0)
    if k_dn in stats:
        m, se = stats[k_dn]["rsi"]
        lines.append(verdict(se == se and m > 2 * se,
                             f"P1 RSI>0 in distinct/no-decoy: {m:+.4f} ± {se:.4f}"))
    k_in = ("indist", 0)
    if k_in in stats:
        m, se = stats[k_in]["rsi"]
        lines.append(verdict(se == se and m > 2 * se,
                             f"P2 RSI survives in-distribution: {m:+.4f} ± {se:.4f}"))
        cm, cse = stats[k_in]["conc"]
        lines.append(verdict(cm > 2 * cse if cse == cse else False,
                             f"P3 concentration survives in-distribution: "
                             f"{cm:+.4f} ± {cse:.4f}"))
    for key in (("distinct", 2), ("indist", 2)):
        if key in stats:
            d = stats[key]["dci"]
            lines.append(verdict(d < 0.5,
                                 f"P4 decoys ignored ({key[0]}): DCI {d:.3f} "
                                 f"(<0.5 = retrieval router; ~1 = novelty detector)"))
    # ---- P5: matched-density control lane (run_r2_density.sh) ----
    dens = {}
    for p in glob.glob(os.path.join(out_dir, "*_density.csv")):
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                key = (r["policy"], float(r["rho"]))
                dens.setdefault(key, []).append(float(r["answer_acc"]))
    if dens:
        lines += ["", "## P5 — matched-density quality (accuracy, mean ± SE "
                      "over all cells/seeds)", "",
                  "| policy | ρ=0.10 | ρ=0.25 |", "|---|---|---|"]
        for pol in ("srd_top", "random", "positional"):
            cells_ = []
            for rho in (0.10, 0.25):
                m, se = mean_se(dens.get((pol, rho), []))
                cells_.append(f"{m:.3f} ± {se:.3f}" if m == m else "–")
            lines.append(f"| {pol} | {cells_[0]} | {cells_[1]} |")
        for name, rho in (("exact_ref", 1.0), ("linear_ref", 0.0)):
            m, _ = mean_se(dens.get((name, rho), []))
            if m == m:
                lines.append(f"| {name} (bound) | {m:.3f} | |")
        ok = []
        for rho in (0.10, 0.25):
            s, _ = mean_se(dens.get(("srd_top", rho), []))
            r, _ = mean_se(dens.get(("random", rho), []))
            p, _ = mean_se(dens.get(("positional", rho), []))
            ok.append(s == s and r == r and p == p and s > r and s > p)
        lines += ["", verdict(all(ok),
                  "P5 SRD beats BOTH baselines at BOTH densities "
                  "(beating random but losing to positional is a negative "
                  "result and gets published as one)")]

    lines += ["", "## Control-first check", ""]
    for key, st in stats.items():
        if st["exact"] < 0.3:
            lines.append(f"- ⚠ {key[0]} decoys={key[1]}: exact lane at "
                         f"{st['exact']:.3f} accuracy — cell is UNINFORMATIVE "
                         f"about SRD (pre-registered threat to validity).")
    return "\n".join(lines)


def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        hdr = ("step,lane,answer_ce,answer_acc,tail_gate,fill_gate,"
               "target_gate,nontarget_gate,decoy_gate\n")
        # Synthetic retrieval-router cell: target >> nontarget, decoys low.
        for s in SEEDS:
            with open(os.path.join(td, f"distinct_d2_s{s}_probe.csv"), "w",
                      encoding="utf-8") as f:
                f.write(hdr)
                f.write(f"100,exact,1.0,0.90,0,0,0,0,0\n")
                f.write(f"100,srd,1.0,0.85,0.9,0.4,0.80,0.40,0.30\n")
        # Synthetic P5 lane: srd_top above both baselines at both rhos.
        for s in SEEDS:
            with open(os.path.join(td, f"distinct_d2_s{s}_density.csv"), "w",
                      encoding="utf-8") as f:
                f.write("policy,rho,answer_ce,answer_acc\n")
                for rho in (0.1, 0.25):
                    f.write(f"srd_top,{rho},1.0,{0.70 + 0.02 * s}\n")
                    f.write(f"random,{rho},1.2,{0.40 + 0.02 * s}\n")
                    f.write(f"positional,{rho},1.1,{0.50 + 0.02 * s}\n")
                f.write("exact_ref,1.0,0.9,0.90\nlinear_ref,0.0,1.4,0.20\n")
        txt = analyse(td)
        assert "P1" in txt or "P4" in txt
        assert "HOLDS" in txt, txt
        # RSI = (.8-.4)/1.2 = .333 ; DCI = .3/.8 = .375 (<0.5 -> router)
        assert "0.375" in txt, txt
        assert "P5 SRD beats BOTH" in txt and txt.count("HOLDS") >= 2, txt
    print("SELFTEST OK: RSI/DCI computed from a synthetic router cell "
          "(RSI +0.333, DCI 0.375 -> P4 holds)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", nargs="?")
    ap.add_argument("--md")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    if not args.out_dir:
        ap.error("give the sweep out_dir (or --selftest)")
    text = analyse(args.out_dir)
    print(text)
    if args.md:
        with open(args.md, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"\nwrote {args.md}")


if __name__ == "__main__":
    main()
