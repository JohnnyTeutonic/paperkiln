"""numpy IR on every self/twin choice whose stored top-two gap is under 0.01 (the only ones an error near
1e-2 could flip)."""
import json
import sys
import os
sys.path.insert(0, r"C:/Users/jonat/OneDrive/Documents/research_portfolio_complete/microtorch/tools")
import carry5_eval  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ir_spot_runs import RUNS  # noqa: E402
SA = os.environ.get("SCORE_AUDIT_WORK", "C:/ml_artifacts/score_audit/work")
run = sys.argv[1]
post, gguf = RUNS[run]
ir = carry5_eval.IrBackend(gguf)
rows = {}
for f in sorted(os.listdir(f"{SA}/sa_{run}")):
    if f.endswith(".in.jsonl"):
        for r in carry5_eval.read_jsonl(f"{SA}/sa_{run}/{f}"):
            rows[r["key"]] = r
out = []
for r in carry5_eval.read_jsonl(f"{post}/carry5/m/items.jsonl"):
    if r["family"] not in ("recall", "correction", "topic", "knowledge"):
        continue
    for mode in ("self", "twin"):
        raw = r["fc"][mode]["raw"]
        s = sorted(raw, reverse=True)
        if s[0] - s[1] >= 0.01:
            continue
        sums = [ir.score(rows[f"{r['id']}|{mode}|{c}"]["prompt_ids"], rows[f"{r['id']}|{mode}|{c}"]["continuation_ids"])["sum"] for c in range(4)]
        out.append({"id": r["id"], "mode": mode, "gap_stored": s[0] - s[1], "max_abs_sum_diff": max(abs(a - b) for a, b in zip(sums, raw)),
                    "choice_stored": raw.index(s[0]), "choice_ir": sums.index(max(sums))})
json.dump(out, open(f"{os.path.dirname(os.path.abspath(__file__))}/receipts/close_{run}.json", "w"), indent=1)
print(run, len(out), "close choices; flips", sum(o["choice_stored"] != o["choice_ir"] for o in out),
      "; max diff", max([o["max_abs_sum_diff"] for o in out] or [0]), "; min gap", min([o["gap_stored"] for o in out] or [0]))
