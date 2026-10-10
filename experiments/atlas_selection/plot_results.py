"""Static research figure from the verified report; requires matplotlib."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import selection as S


def main():
    root = Path(__file__).resolve().parent
    report = json.loads((root / "results_v2.json").read_text())
    p = report["protocol"]
    source, sr = S.load_panel(p, "source")
    target, tr = S.load_panel(p, "target")
    if sr + tr != report["receipts"]:
        raise ValueError("receipt hashes differ from verified report")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = list(range(len(p["lanes"])))
    for panel, shift, color, label in [(source, -.18, "#3979ad", "Source width 256"),
                                       (target, .18, "#d07632", "Target width 512")]:
        values = S.means(panel, p["seeds"], p["lanes"], 3600)
        axes[0].bar([i+shift for i in x], [v-values[0] for v in values],
                    width=.36, color=color, label=label)
    axes[0].set_xticks(x, ["Exact", "W16+s", "W32+s", "W64+s", "W128+s", "W64"], rotation=25)
    axes[0].set_title("Small-width favourites reverse at target width")
    axes[0].set_ylabel("Mean validation loss minus exact attention")
    axes[0].legend(frameon=False)
    cells = sorted([c for c in report["analysis"]["cells"]
                    if c["source_step"] == c["target_step"] == 3600], key=lambda c: c["k"])
    axes[1].bar(range(len(cells)),
                [c["paired_contrasts"]["loss_vs_random_shortlist"]["mean"] for c in cells],
                color="#755b96", width=.6)
    axes[1].set_xticks(range(len(cells)), [str(c["k"]) for c in cells])
    axes[1].set_xlabel("Candidates retained from the source ranking")
    axes[1].set_ylabel("Held-out loss minus expected random shortlist")
    axes[1].set_title("Top-two screening discards useful choices")
    for ax in axes:
        ax.axhline(0, color="#333333", linewidth=.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle("ATLAS decision audit · transfer_s2 · step 3600", fontsize=14)
    fig.text(.5, .015, "Post-hoc, one corpus and width pair. Left: 12-seed means. "
             "Right: mean over 924 overlapping 6/6 splits; no confidence intervals.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, .94))
    for suffix in ("png", "pdf"):
        with (root / f"selection_audit.{suffix}").open("xb") as stream:
            fig.savefig(stream, format=suffix, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
