"""Stored VM (CUDA) Carry-5 scores vs the home CPU rescoring: per request and the registered Y."""
import json
import os
import sys
import glob
sys.path.insert(0, r"C:/Users/jonat/OneDrive/Documents/research_portfolio_complete/microtorch/tools")
import carry5_eval  # noqa: E402
SA = os.environ.get("SCORE_AUDIT_WORK", "C:/ml_artifacts/score_audit/work")
TC = r"C:/ml_artifacts/tiny_chat"
POST = {"ctl_r1": f"{TC}/e1_x5_ctl/post_e1_x5_ctl_L16_r1", "x5_r1": f"{TC}/e1_x5_r1/post_e1_x5_X5_r1",
        "x5_r2": f"{TC}/e1_x5_r2/post_e1_x5_X5_r2", "x5_r3": f"{TC}/e1_x5_r3/post_e1_x5_X5_r3"}
PRIMARY = ("recall", "correction", "topic", "knowledge")
items = {it["id"]: it for it in carry5_eval.read_jsonl(carry5_eval.SET)}
def _mean(v):
    return sum(v) / len(v)


out = {}
for run, post in POST.items():
    cpu = {}
    files = sorted(glob.glob(f"{SA}/sa_{run}/cpu/chunk*.out.jsonl"))
    if not files:
        continue
    for f in files:
        for r in carry5_eval.read_jsonl(f):
            cpu[r["key"]] = r
    recs = carry5_eval.read_jsonl(f"{post}/carry5/m/items.jsonl")
    summ = json.load(open(f"{post}/summary.json"))
    worst, n_req, over = 0.0, 0, 0
    acc = {"self": {"cuda": [], "cpu": []}, "twin": {"cuda": [], "cpu": []}}
    flips = []
    for r in recs:
        if r["family"] not in PRIMARY:
            continue
        it = items[r["id"]]
        for mode in ("self", "twin"):
            fc = r["fc"][mode]
            rows = [cpu[f"{r['id']}|{mode}|{c}"] for c in range(4)]
            if fc["over_ctx"] or any(x["over_ctx"] for x in rows):
                over += 1
                assert fc["over_ctx"] and all(x["over_ctx"] for x in rows)
                acc[mode]["cuda"].append(0.0)
                acc[mode]["cpu"].append(0.0)
                continue
            sums = [x["sum"] for x in rows]
            for a, b in zip(sums, fc["raw"]):
                worst = max(worst, abs(a - b))
                n_req += 1
            a_cpu, ch_cpu = carry5_eval.choose(sums, it["gold_index"])
            a_cuda, ch_cuda = carry5_eval.choose(fc["raw"], it["gold_index"])
            assert a_cuda == fc["acc"]
            acc[mode]["cuda"].append(a_cuda)
            acc[mode]["cpu"].append(a_cpu)
            if ch_cpu != ch_cuda or a_cpu != a_cuda:
                flips.append((r["id"], mode, a_cuda, a_cpu))
    y_cuda = _mean(acc["self"]["cuda"]) - _mean(acc["twin"]["cuda"])
    y_cpu = _mean(acc["self"]["cpu"]) - _mean(acc["twin"]["cpu"])
    out[run] = {"chunks": len(files), "requests_compared": n_req, "max_abs_sum_diff": worst, "over_ctx_rows": over,
                "items": len(acc["self"]["cpu"]), "choice_flips": flips,
                "self_acc": {"cuda": _mean(acc["self"]["cuda"]), "cpu": _mean(acc["self"]["cpu"])},
                "twin_acc": {"cuda": _mean(acc["twin"]["cuda"]), "cpu": _mean(acc["twin"]["cpu"])},
                "Y_stored_summary": summ["primary"], "Y_from_stored_items": y_cuda, "Y_cpu": y_cpu,
                "Y_cpu_minus_stored": y_cpu - summ["primary"]}
    print(run, json.dumps(out[run]))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "receipts", "cpu_rescore.json"), "w"), indent=1)
