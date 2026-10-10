"""Render a verified audit as readable Markdown; never overwrite an artifact."""
import argparse
import hashlib
import json
from pathlib import Path


def render(report, digest):
    p, a = report["protocol"], report["analysis"]
    ref = p["reference_cell"]
    rows = ["# ATLAS selection audit: archival results", "",
            "**Post-hoc; twelve seeds, one corpus, one width pair.**",
            "No new training was run. No frozen transfer-study analysis was changed.", "",
            f"Report SHA-256: `{digest}`.", "",
            f"Input: {len(report['receipts'])} complete runs. "
            f"All {a['n_overlapping_splits']} six/six seed partitions; "
            f"{len(a['cells'])} source-position / target-position / shortlist cells.", "",
            "Every selection uses calibration seeds; scoring uses different seed IDs "
            "at both widths. All losses are the logged validation-loss units.", "",
            "## End-of-training decisions", "",
            f"Source step {ref['source_step']}, target step {ref['target_step']}. "
            "Negative contrasts favour the informed shortlist. Random is the exact "
            "expectation over uniform size-k subsets, using the same target calibration rule.", "",
            "| k | Excess vs exact | Excess vs random | Excess vs direct target | Exclusion gap | Calibration gap |",
            "|---:|---:|---:|---:|---:|---:|"]
    cells = a["cells"]
    for c in cells:
        if c["source_step"] == ref["source_step"] and c["target_step"] == ref["target_step"]:
            pol, comp = c["policy"], c["paired_contrasts"]
            values = [pol["excess_vs_exact"]["mean"], comp["loss_vs_random_shortlist"]["mean"],
                      comp["loss_vs_direct_target"]["mean"], pol["exclusion_gap"]["mean"],
                      pol["calibration_gap"]["mean"]]
            rows.append(f"| {c['k']} | " + " | ".join(f"{v:+.6f}" for v in values) + " |")
    rows += ["", "## Source training position", "",
             "Mean held-out loss difference versus random shortlisting, target step 3600. "
             "All source positions are shown; choosing the best row now is post-hoc tuning.", "",
             "| Source step | k=1 | k=2 | k=3 | k=6 |", "|---:|---:|---:|---:|---:|"]
    for step in p["source_steps"]:
        by_k = {c["k"]: c for c in cells if c["source_step"] == step and c["target_step"] == 3600}
        vals = [by_k[k]["paired_contrasts"]["loss_vs_random_shortlist"]["mean"]
                for k in p["shortlist_sizes"]]
        rows.append(f"| {step} | " + " | ".join(f"{v:+.6f}" for v in vals) + " |")
    cell = next(c for c in cells if all(c[k] == v for k, v in ref.items()))
    rows += ["", "## Reference cell details", "", f"`{ref}`", "",
             "Split percentiles below describe overlapping partitions. **They are not confidence intervals.**", "",
             "| Paired contrast | Mean | Split p05 | Split p95 |", "|---|---:|---:|---:|"]
    for key, value in cell["paired_contrasts"].items():
        rows.append(f"| {key} | {value['mean']:+.6f} | {value['split_p05']:+.6f} | {value['split_p95']:+.6f} |")
    rows += ["", "Selected-lane counts across the 924 overlapping splits:", ""]
    rows += [f"- {name}: {count}" for name, count in cell["selected_counts"].items()]
    rows += ["", "Acquisition/scoring accounting (run-steps, separated by width):", ""]
    rows += [f"- {name}: {value:,}" for name, value in cell["acquisition_budget"].items()]
    rows += ["", "## What these results support", "",
             "These are retrospective decisions in the existing six-lane transfer_s2 panel. "
             "Excess against exact and the paired policy contrasts quantify selection consequences. "
             "Exclusion plus calibration gap equals the gap to the held-out sample's best lane; "
             "that optimistic benchmark is not a population oracle.", "",
             "The informed shortlist pays additional source cost. A favourable loss contrast alone "
             "does not establish a compute saving. There are no new data/corpus holdouts, "
             "fresh seeds, population confidence intervals, or confirmatory claims here. "
             "See README.md for the design, literature anchors and prospective next experiment.", ""]
    return "\n".join(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    content = args.report.read_bytes()
    report = json.loads(content)
    with args.output.open("x", encoding="utf-8") as f:
        f.write(render(report, hashlib.sha256(content).hexdigest()))
