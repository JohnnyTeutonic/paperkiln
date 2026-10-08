#!/usr/bin/env python3
"""Figures for the transfer paper, drawn from receipts only.

Reads the run directories through analyze.py's own readers, so every number on
a figure is the number in the receipts. Writes PDF + PNG into --out.

Lane and arm names match the manuscript: d=256 and d=512 (never S/M), and
exact, w16+s, w32+s, w64+s, w128+s, w64. The seed-subset figure repeats the
post-hoc receipt's computation exactly (posthoc.py: seed 20260914, 2000 draws,
k = 3, 6, 12, one generator per study), so it shows the numbers in Table B5.
The comparison-graph figure marks each cell A (agree), R (reversal: opposite
non-zero majorities) or T (tie-involved), so it does not rely on colour.

Usage:
  python3 make_figures.py --s1S <root> --s1M <root> --s2S <root> --s2M <root> [--out .]
A root holds run directories, directly or under runs/.
"""
import argparse
import glob
import os
import random
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "posthoc"))
import analyze as A       # noqa: E402
import posthoc as P       # noqa: E402

LANE_LABEL = {"exact": "exact", "swa16s1": "w16+s", "swa32s1": "w32+s",
              "swa64s1": "w64+s", "swa128s1": "w128+s", "swa64s0": "w64"}
ADMITTED = {800, 1600, 2400, 3200}   # Study 2's registered reachability band
plt.rcParams.update({"font.size": 8})


def load(root):
    dirs = sorted(glob.glob(os.path.join(root, "runs", "*"))) or sorted(glob.glob(os.path.join(root, "*")))
    arm = {}
    for d in dirs:
        if os.path.exists(os.path.join(d, "events.jsonl")):
            r = A.read_run(d)
            arm.setdefault(r["seed"], {})[r["lane"]] = r
    return arm


def fig1(studies, out):
    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    for name, (armS, armM), marker in zip(("Study 1 (shared rate)", "Study 2 (rate per width)"),
                                          studies, ("s", "o")):
        ms = A.milestones_from_S(armS)
        per, total, n = P.f1_by_milestone(armS, armM, ms)
        pts = [(sl, f) for sl, f, c in per if f is not None]
        line, = ax.plot([x for x, _ in pts], [y for _, y in pts], marker=marker, label=name)
        if name.startswith("Study 2"):
            last = [(x, y) for x, y in pts if x not in ADMITTED]
            ax.plot([x for x, _ in last], [y for _, y in last], marker="o", ls="none",
                    mfc="white", mec=line.get_color(), ms=7,
                    label="Study 2, fifth milestone (8 of 12 d=256 seeds)")
    ax.axhline(0.75, ls="--", lw=0.8, color="grey", label="adoption threshold 0.75")
    ax.axhline(0.5, ls=":", lw=0.8, color="grey", label="0.5 reference")
    ax.set_xlabel("milestone, named by its d=256 reference step\n(the loss level is set within each study)")
    ax.set_ylabel("sign concordance $F_1$ (15 edges)")
    ax.set_ylim(0.2, 1.05)
    ax.set_xticks(A.MILESTONE_SLICES)
    ax.legend(fontsize=6.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig1_concordance.pdf"))
    fig.savefig(os.path.join(out, "fig1_concordance.png"), dpi=200)
    plt.close(fig)


def median_curve(arm, lane):
    steps = sorted({s for r in (l.get(lane) for l in arm.values()) if r for s, _ in r["evals"]})
    xs, ys = [], []
    for s in steps:
        vals = [A.val_at(l[lane]["evals"], s) for l in arm.values() if lane in l]
        vals = [v for v in vals if v is not None]
        if vals:
            xs.append(s)
            ys.append(st.median(vals))
    return xs, ys


def fig2(studies, out):
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), sharey=True)
    titles = ("Study 1: rate $10^{-3}$ at both widths",
              "Study 2: rate $2.5\\times10^{-4}$ (d=256), $5\\times10^{-4}$ (d=512)")
    for ax, title, (armS, armM) in zip(axes, titles, studies):
        ms = A.milestones_from_S(armS)
        for arm, lab, ls in ((armS, "d=256", "-"), (armM, "d=512", "--")):
            xs, ys = median_curve(arm, "exact")
            ax.plot(xs, ys, ls=ls, label=lab)
        for i, (sl, v) in enumerate(ms):
            ax.axhline(v, lw=0.5, ls=":", color="grey",
                       label="this study's five milestones" if i == 0 else None)
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel("step")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("validation loss (exact lane, seed median)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2_curves.pdf"))
    fig.savefig(os.path.join(out, "fig2_curves.png"), dpi=200)
    plt.close(fig)


def effects(armS, armM):
    """The H-SCALAR quantity exactly as the frozen script builds it: per edge, the
    seed-mean Delta (second lane minus first lane) at step 3600 in each arm."""
    final = A.MILESTONE_SLICES[-1]
    xs, ys = [], []
    for (a, b) in A.edges():
        vS = [v for v in (A.delta(armS, s, a, b, final) for s in sorted(armS)) if v is not None]
        vM = [v for v in (A.delta(armM, s, a, b, final) for s in sorted(armM)) if v is not None]
        if vS and vM:
            xs.append(st.mean(vS))
            ys.append(st.mean(vM))
    return xs, ys


def fig3(studies, out):
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0))
    for ax, name, (armS, armM) in zip(axes, ("Study 1", "Study 2"), studies):
        xs, ys = effects(armS, armM)
        rho = A.spearman(xs, ys)
        ax.scatter(xs, ys, s=22)
        ax.axhline(0, lw=0.5, color="grey")
        ax.axvline(0, lw=0.5, color="grey")
        ax.set_title(f"{name}: Spearman $\\rho$ = {rho:.2f} over {len(xs)} edges", fontsize=8.5)
        ax.set_xlabel("d=256 edge effect (step 3600)")
    axes[0].set_ylabel("d=512 edge effect (step 3600)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig3_effects.pdf"))
    fig.savefig(os.path.join(out, "fig3_effects.png"), dpi=200)
    plt.close(fig)


def fig4(studies, out, draws=2000, seed=20260914, ks=(3, 6, 12)):
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    for name, (armS, armM), marker in zip(("Study 1", "Study 2"), studies, ("s", "o")):
        rng = random.Random(seed)          # one generator per study, as posthoc.py
        meds, lo, hi = [], [], []
        for k in ks:
            vals = sorted(P.seeds_matter(armS, armM, k, draws, rng))
            meds.append(st.median(vals))
            lo.append(vals[int(0.025 * len(vals))])
            hi.append(vals[int(0.975 * len(vals)) - 1])
        ax.errorbar(ks, meds, yerr=[[m - l for m, l in zip(meds, lo)], [h - m for m, h in zip(meds, hi)]],
                    marker=marker, capsize=3, label=name)
        print(f"fig4 {name}: " + ", ".join(f"k={k} median {m:.3f} [{l:.3f}, {h:.3f}]"
                                           for k, m, l, h in zip(ks, meds, lo, hi)))
    ax.axhline(0.75, ls="--", lw=0.8, color="grey", label="0.75 point threshold")
    ax.set_xlabel("seeds per arm (random subsets of the twelve)")
    ax.set_ylabel("$F_1$, five-milestone reading\n(median, 2.5th to 97.5th percentile)")
    ax.set_xticks(ks)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig4_seeds.pdf"))
    fig.savefig(os.path.join(out, "fig4_seeds.png"), dpi=200)
    plt.close(fig)


def fig5(armS, armM, out):
    ms = A.milestones_from_S(armS)
    stepper = P.milestone_stepper(ms)
    mS, _ = A.sign_matrix(armS, ms, stepper)
    mM, _ = A.sign_matrix(armM, ms, stepper)
    E = A.edges()
    grid, text = [], []
    for (a, b) in E:
        row, trow = [], []
        for sl, _ in ms:
            k = ((a, b), sl)
            s, m = mS.get(k), mM.get(k)
            if s == m:
                row.append(0); trow.append("A")
            elif s * m == -1:
                row.append(2); trow.append("R")
            else:
                row.append(1); trow.append("T")
        grid.append(row)
        text.append(trow)
    cmap = ListedColormap(["#e8eef7", "#f6c27a", "#b2182b"])   # agree, tie-involved, reversal
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=2, aspect="auto")
    for i, trow in enumerate(text):
        for j, t in enumerate(trow):
            ax.text(j, i, t, ha="center", va="center", fontsize=7,
                    color="white" if t == "R" else "black")
    ax.set_yticks(range(len(E)))
    ax.set_yticklabels([f"{LANE_LABEL[a]} vs {LANE_LABEL[b]}" for a, b in E], fontsize=7)
    ax.set_xticks(range(len(ms)))
    ax.set_xticklabels([f"{sl}" + ("" if sl in ADMITTED else "*") for sl, _ in ms])
    ax.set_xlabel("milestone (d=256 reference step; * fifth, partially reached)")
    ax.set_title("Study 2: d=256 against d=512 orientation per edge\n"
                 "A agree, R reversal, T tie at one width", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig5_graph.pdf"))
    fig.savefig(os.path.join(out, "fig5_graph.png"), dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    for k in ("s1S", "s1M", "s2S", "s2M"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--out", default=HERE)
    args = ap.parse_args()
    s1 = (load(args.s1S), load(args.s1M))
    s2 = (load(args.s2S), load(args.s2M))
    os.makedirs(args.out, exist_ok=True)
    fig1((s1, s2), args.out)
    fig2((s1, s2), args.out)
    fig3((s1, s2), args.out)
    fig4((s1, s2), args.out)
    fig5(s2[0], s2[1], args.out)
    print("figures written to", args.out)


if __name__ == "__main__":
    main()
