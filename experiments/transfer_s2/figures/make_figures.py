#!/usr/bin/env python3
"""Figures for the transfer paper, drawn from receipts only.

Reads the same run directories the frozen analysis reads (root/runs/*/
events.jsonl) through analyze.py's own readers, so every number on a
figure is the number in the receipts. Writes PDF + PNG into this
directory. Study 1 roots may be the receipts directories (via a symlinked
`runs/`), study 2 roots the banked artefact roots.

Usage:
  python3 make_figures.py --s1S <root> --s1M <root> --s2S <root> --s2M <root> [--out .]
Figures:
  fig1_concordance.pdf   per-milestone F1, study 1 (one reached position) vs study 2 (five)
  fig2_curves.pdf        exact-lane validation loss per arm with the S milestones marked, both studies
  fig3_effects.pdf       per-edge effect sizes, S vs M, both studies (the rho scatter)
  fig4_seeds.pdf         F1 on random k-seed subsets, both studies
  fig5_graph.pdf         the comparison graph at each milestone, study 2: edge sign S vs M
"""
import argparse
import os
import random
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "posthoc"))
import analyze as A       # noqa: E402
import posthoc as P       # noqa: E402

LANE_LABEL = {"exact": "exact", "swa16s1": "swa16+s", "swa32s1": "swa32+s",
              "swa64s1": "swa64+s", "swa128s1": "swa128+s", "swa64s0": "swa64"}


def load(root):
    return A.read_arm(root)


def per_milestone(armS, armM):
    ms = A.milestones_from_S(armS)
    per, total, n = P.f1_by_milestone(armS, armM, ms)
    return ms, per, total, n


def fig1(studies, out):
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for name, (armS, armM), marker in zip(("Study 1 (shared rate)", "Study 2 (rate per width)"),
                                          studies, ("s", "o")):
        ms, per, total, n = per_milestone(armS, armM)
        xs = [sl for sl, f, c in per if f is not None]
        ys = [f for sl, f, c in per if f is not None]
        ax.plot(xs, ys, marker=marker, label=f"{name}: aggregate {total:.3f} over {n} cells")
    ax.axhline(0.75, ls="--", lw=0.8, color="grey")
    ax.axhline(0.5, ls=":", lw=0.8, color="grey")
    ax.set_xlabel("matched validation-loss milestone (S-arm slice)")
    ax.set_ylabel("sign concordance F1 (15 edges)")
    ax.set_ylim(0.2, 1.05)
    ax.set_xticks(A.MILESTONE_SLICES)
    ax.legend(fontsize=7, loc="lower left")
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
            xs.append(s); ys.append(st.median(vals))
    return xs, ys


def fig2(studies, out):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    for ax, name, (armS, armM) in zip(axes, ("Study 1: lr 1e-3 at both widths",
                                             "Study 2: lr 2.5e-4 (S), 5e-4 (M)"), studies):
        ms = A.milestones_from_S(armS)
        for arm, lab in ((armS, "S, d=256"), (armM, "M, d=512")):
            xs, ys = median_curve(arm, "exact")
            ax.plot(xs, ys, label=lab)
        for sl, v in ms:
            ax.axhline(v, lw=0.5, ls=":", color="grey")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("step")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("validation loss (exact lane, seed median)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2_curves.pdf"))
    fig.savefig(os.path.join(out, "fig2_curves.png"), dpi=200)
    plt.close(fig)


def effects(armS, armM):
    """The H-SCALAR quantity exactly as the frozen script builds it:
    per edge, the seed-mean Delta at the FINAL slice (step 3600) in each
    arm. (Not the matched milestones: the committed foil was defined on
    the final step, and the figure must show the number in the receipt.)"""
    final = A.MILESTONE_SLICES[-1]
    xs, ys, labels = [], [], []
    for (a, b) in A.edges():
        vS = [A.delta(armS, s, a, b, final) for s in sorted(armS)]
        vM = [A.delta(armM, s, a, b, final) for s in sorted(armM)]
        vS = [v for v in vS if v is not None]
        vM = [v for v in vM if v is not None]
        if vS and vM:
            xs.append(st.mean(vS)); ys.append(st.mean(vM))
            labels.append(f"{LANE_LABEL[a]} v {LANE_LABEL[b]}")
    return xs, ys, labels


def fig3(studies, out):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    for ax, name, (armS, armM) in zip(axes, ("Study 1", "Study 2"), studies):
        xs, ys, labels = effects(armS, armM)
        rho = A.spearman(xs, ys)
        ax.scatter(xs, ys, s=18)
        for x, y, l in zip(xs, ys, labels):
            ax.annotate(l, (x, y), fontsize=5, xytext=(2, 2), textcoords="offset points")
        ax.axhline(0, lw=0.5, color="grey"); ax.axvline(0, lw=0.5, color="grey")
        ax.set_title(f"{name}: Spearman rho = {rho:.2f} over {len(xs)} edges", fontsize=9)
        ax.set_xlabel("effect size at S (seed-mean Delta, step 3600)")
    axes[0].set_ylabel("effect size at M (step 3600)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig3_effects.pdf"))
    fig.savefig(os.path.join(out, "fig3_effects.png"), dpi=200)
    plt.close(fig)


def fig4(studies, out, draws=1000):
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    rng = random.Random(20260914)
    ks = [3, 4, 6, 8, 12]
    for name, (armS, armM), marker in zip(("Study 1", "Study 2"), studies, ("s", "o")):
        meds, lo, hi = [], [], []
        for k in ks:
            vals = sorted(P.seeds_matter(armS, armM, k, draws if k < 12 else 1, rng))
            meds.append(st.median(vals)); lo.append(vals[int(0.025 * len(vals))]); hi.append(vals[max(0, int(0.975 * len(vals)) - 1)])
        ax.errorbar(ks, meds, yerr=[[m - l for m, l in zip(meds, lo)], [h - m for m, h in zip(meds, hi)]],
                    marker=marker, capsize=3, label=name)
    ax.axhline(0.75, ls="--", lw=0.8, color="grey")
    ax.set_xlabel("seeds per arm (random subsets of the twelve)")
    ax.set_ylabel("F1 (median, 2.5 to 97.5 pct)")
    ax.set_xticks(ks)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig4_seeds.pdf"))
    fig.savefig(os.path.join(out, "fig4_seeds.png"), dpi=200)
    plt.close(fig)


def fig5(armS, armM, out):
    ms = A.milestones_from_S(armS)
    stepper = P.milestone_stepper(ms)
    positions = [(sl, v) for sl, v in ms]
    mS, _ = A.sign_matrix(armS, positions, stepper)
    mM, _ = A.sign_matrix(armM, positions, stepper)
    E = A.edges()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    grid = []
    for (a, b) in E:
        row = []
        for sl, _ in ms:
            k = ((a, b), sl)
            if k in mS and k in mM:
                row.append(1 if mS[k] == mM[k] else -1)
            else:
                row.append(0)
        grid.append(row)
    im = ax.imshow(grid, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax.set_yticks(range(len(E)))
    ax.set_yticklabels([f"{LANE_LABEL[a]} v {LANE_LABEL[b]}" for a, b in E], fontsize=6)
    ax.set_xticks(range(len(ms)))
    ax.set_xticklabels([str(sl) for sl, _ in ms])
    ax.set_xlabel("matched milestone")
    ax.set_title("Study 2: edge orientation agrees (green) or differs (red), S vs M", fontsize=9)
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
