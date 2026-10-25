"""Spot check (one thread, < 1 min): numpy IR on the GGUF vs the stored VM (CUDA) Carry-5 sums, self and twin
modes, every K-th primary-family item. Writes sa_spot_RUN.json."""
import json
import sys
import os
import time
sys.path.insert(0, r"C:/Users/jonat/OneDrive/Documents/research_portfolio_complete/microtorch/tools")
import carry5_eval  # noqa: E402
SA = os.environ.get("SCORE_AUDIT_WORK", "C:/ml_artifacts/score_audit/work")
TC = r"C:/ml_artifacts/tiny_chat"
RUNS = {"ctl_r1": (f"{TC}/e1_x5_ctl/post_e1_x5_ctl_L16_r1", f"{TC}/e1/pick_e1_L16/model.gguf"),
        "x5_r1": (f"{TC}/e1_x5_r1/post_e1_x5_X5_r1", f"{TC}/e1_x5_r1/post_e1_x5_X5_r1/sft/tc_sft_X5_r1.gguf"),
        "x5_r2": (f"{TC}/e1_x5_r2/post_e1_x5_X5_r2", f"{TC}/e1_x5_r2/post_e1_x5_X5_r2/sft/tc_sft_X5_r2.gguf"),
        "x5_r3": (f"{TC}/e1_x5_r3/post_e1_x5_X5_r3", f"{TC}/e1_x5_r3/post_e1_x5_X5_r3/sft/tc_sft_X5_r3.gguf")}
run, K, off = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
post, gguf = RUNS[run]
t0 = time.time()
ir = carry5_eval.IrBackend(gguf)
rows = {}
for f in sorted(os.listdir(f"{SA}/sa_{run}")):
    if f.endswith(".in.jsonl"):
        for r in carry5_eval.read_jsonl(f"{SA}/sa_{run}/{f}"):
            rows[r["key"]] = r
recs = [r for r in carry5_eval.read_jsonl(f"{post}/carry5/m/items.jsonl")
        if r["family"] in ("recall", "correction", "topic", "knowledge")]
worst, n, flips, out = 0.0, 0, 0, []
for r in recs[off::K]:
    for mode in ("self", "twin"):
        sums = []
        for c in range(4):
            q = rows[f"{r['id']}|{mode}|{c}"]
            s = ir.score(q["prompt_ids"], q["continuation_ids"])
            sums.append(s["sum"])
        d = max(abs(a - b) for a, b in zip(sums, r["fc"][mode]["raw"]))
        worst = max(worst, d)
        n += 4
        same = max(range(4), key=lambda c: sums[c]) == max(range(4), key=lambda c: r["fc"][mode]["raw"][c])
        flips += not same
        out.append({"id": r["id"], "mode": mode, "max_abs_sum_diff": d, "same_choice": same})
rep = {"run": run, "engine": "numpy IR (spec/tools/ir_numpy.py) on the GGUF", "every": K, "offset": off,
       "items": len(recs[off::K]), "requests": n, "max_abs_sum_diff": worst, "choice_differs": flips,
       "seconds": round(time.time() - t0, 1), "rows": out}
json.dump(rep, open(f"{os.path.dirname(os.path.abspath(__file__))}/receipts/spot_{run}_{off}.json", "w"), indent=1)
print({k: v for k, v in rep.items() if k != "rows"})
