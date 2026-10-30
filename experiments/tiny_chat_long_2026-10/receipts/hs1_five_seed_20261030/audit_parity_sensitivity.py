"""Replay the six-known-item posthoc sensitivity bound; never repair or waive parity."""

import argparse
import hashlib
import json
import math
import pathlib
import statistics

PINNED_RECEIPTS = {
    "history_argmax_audit.json": "da3e3880fa4eb4d3ad1ed88b908c2a4b637780860eb9438f94fc93a6a92ded9e",
    "analysis_hs1.json": "6868c6374d6ca637b91ec106d762c79c193b9cd0de9c784fb43311b9588da04a",
    "raw_credit_audit.json": "bc8dd6d40b723fe5f1a0e891abd660e0c0be3964f1c9e3b61e4a3adcfa10e52b",
    "completion.json": "5f744dd5363606d8e53ff3f055b5cdd7379f6136436832108ceb6412e91733c1",
}


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def check(path, expected):
    if digest(path) != expected:
        raise ValueError("input hash differs: " + str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--art-root", required=True, type=pathlib.Path)
    parser.add_argument(
        "--output", type=pathlib.Path, help="New receipt path; otherwise print JSON"
    )
    args = parser.parse_args()
    art = args.art_root.resolve()
    here = pathlib.Path(__file__).resolve().parent
    used = {}
    for name, expected in PINNED_RECEIPTS.items():
        check(here / name, expected)
        used[name] = expected
    audit = json.loads((here / "history_argmax_audit.json").read_text())
    raw = json.loads((here / "raw_credit_audit.json").read_text())
    completion = json.loads((here / "completion.json").read_text())
    expected_items = {
        r["path"].split("/tiny_chat/", 1)[1]: r["sha256"] for r in raw["runs"]
    }
    expected_summaries = {
        k.split("/tiny_chat/", 1)[1]: v
        for k, v in completion["input_manifest"]["sha256"].items()
        if k.endswith("/summary.json")
    }
    bad = {}
    for r in audit["nonidentical"]:
        bad.setdefault((r["cell"], r["seed"]), set()).add(r["id"])
    primary = {"recall", "correction", "topic", "load", "knowledge"}
    ranges = {}
    denominators = {}
    for cell in ["B", "Sgist"]:
        for seed in range(1, 6):
            root = art / f"c12_{cell}_r{seed}" / f"post_c12_{cell}_r{seed}"
            item_path = root / "carry12/m/items.jsonl"
            check(item_path, expected_items[str(item_path.relative_to(art))])
            used[str(item_path.relative_to(art))] = digest(item_path)
            rows = [json.loads(x) for x in item_path.read_text().splitlines()]
            xs = [r for r in rows if r["family"] in primary]
            n = len(xs)
            assert n == 1380
            affected = bad.get((cell, seed), set())
            assert affected <= set(r["id"] for r in xs)
            unaffected = sum(
                r["fc"]["self"]["acc"] for r in xs if r["id"] not in affected
            )
            twin = round(sum(r["fc"]["twin"]["acc"] for r in xs) / n, 4)
            lo = round(unaffected / n, 4) - twin
            hi = round((unaffected + len(affected)) / n, 4) - twin
            summary_path = root / "summary.json"
            check(summary_path, expected_summaries[str(summary_path.relative_to(art))])
            used[str(summary_path.relative_to(art))] = digest(summary_path)
            original = json.loads(summary_path.read_text())["Y12"]
            assert lo - 1e-12 <= original <= hi + 1e-12
            ranges[cell, seed] = (lo, hi)
            denominators[cell + str(seed)] = {
                "primary_items": n,
                "affected_distinct_items": len(affected),
                "min_Y12": lo,
                "max_Y12": hi,
                "original_Y12": original,
            }
    report = json.loads((here / "analysis_hs1.json").read_text())
    d = report["HS1_Sgist_minus_B"]["per_seed"]
    lo = [ranges["Sgist", s][0] - ranges["B", s][1] for s in range(1, 6)]
    hi = [ranges["Sgist", s][1] - ranges["B", s][0] for s in range(1, 6)]
    b = [max(abs(lower - x), abs(upper - x)) for lower, upper, x in zip(lo, hi, d)]
    sd_upper = statistics.stdev(d) + math.sqrt(sum(x * x for x in b)) / math.sqrt(4)
    se_upper = sd_upper / math.sqrt(5)
    mean_low, mean_high = statistics.mean(lo), statistics.mean(hi)
    t_abs_lower = -mean_high / se_upper
    assert mean_high < -0.03 and t_abs_lower > 2.776
    result = {
        "status": "POSTHOC_SENSITIVITY_ONLY_NOT_PARITY_REPAIR",
        "assumption": "Only the six identified per-item self-history credits may vary arbitrarily in[0,1]; all other credits, twins, item weighting and registered four-decimal pool rounding stay fixed. This does not bound unrecorded parity defects or validate generation.",
        "denominator": 1380,
        "affected_items": 6,
        "cell_bounds": denominators,
        "paired_per_seed_bounds": [[lower, upper] for lower, upper in zip(lo, hi)],
        "original_mean": statistics.mean(d),
        "mean_interval": [mean_low, mean_high],
        "largest_abs_change_in_mean": max(
            abs(mean_low - statistics.mean(d)), abs(mean_high - statistics.mean(d))
        ),
        "sd_upper_bound": sd_upper,
        "abs_t_lower_bound": t_abs_lower,
        "registered_crit": 2.776,
        "ci90_envelope": [mean_low - 2.132 * se_upper, mean_high + 2.132 * se_upper],
        "all_bounded_variants_meet_numerical_supported_rule": True,
        "proof": "For componentwise delta bounds b, sampleSD(d+delta)<=sampleSD(d)+norm(b)/sqrt(n-1), since centering is an orthogonal projection. Mean extrema use exact rounded per-cell endpoint pools; all five paired differences remain negative.",
        "historical_parity_violation_remains": True,
        "canonical_data_and_analysis_modified": False,
    }

    result["replay"] = {
        "tool_sha256": digest(__file__),
        "input_sha256": used,
        "canonical_statistics_changed": False,
    }
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as target:
            target.write(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
