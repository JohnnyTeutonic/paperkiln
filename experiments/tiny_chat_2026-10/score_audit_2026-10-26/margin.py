"""From the stored sums alone: per run, the self/twin forced choices (four primary families) whose top-two gap is
below a threshold, i.e. the only ones a scoring error of that size could flip; the bound on Y's movement."""
import json
import sys
import glob
import os
sys.path.insert(0, r"C:/Users/jonat/OneDrive/Documents/research_portfolio_complete/microtorch/tools")
import carry5_eval  # noqa: E402
TC = r"C:/ml_artifacts/tiny_chat"
posts = sorted(glob.glob(f"{TC}/e1_x5_ctl/post_*") + glob.glob(f"{TC}/e1_x5_r*/post_*") + glob.glob(f"{TC}/e1/post_e1_L16_r*"))
gold = {it["id"]: it["gold_index"] for it in carry5_eval.read_jsonl(carry5_eval.SET)}
res = {}
for p in posts:
    f = f"{p}/carry5/m/items.jsonl"
    if not os.path.exists(f):
        continue
    recs = [r for r in carry5_eval.read_jsonl(f) if r["family"] in ("recall", "correction", "topic", "knowledge")]
    acc = {"self": [], "twin": []}
    gaps = {1e-4: [0, 0], 1e-3: [0, 0], 1e-2: [0, 0]}
    ties = 0
    for r in recs:
        for i, mode in enumerate(("self", "twin")):
            fc = r["fc"][mode]
            acc[mode].append(fc["acc"])
            if fc["over_ctx"]:
                continue
            s = sorted(fc["raw"], reverse=True)
            g = s[0] - s[1]
            ties += g == 0
            for t in gaps:
                gaps[t][i] += g < t
    y = sum(acc["self"]) / len(acc["self"]) - sum(acc["twin"]) / len(acc["twin"])
    summ = json.load(open(f"{p}/summary.json"))
    n = len(recs)
    res[os.path.basename(p)] = {"items": n, "Y_items": y, "Y_summary": summ["primary"], "exact_ties": ties,
                                "gap_below": {str(t): {"self": v[0], "twin": v[1], "Y_bound": (v[0] + v[1]) / n} for t, v in gaps.items()}}
    print(os.path.basename(p), json.dumps(res[os.path.basename(p)]))
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "receipts", "margin.json"), "w"), indent=1)
