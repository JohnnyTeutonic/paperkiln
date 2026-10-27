#!/usr/bin/env bash
# Claude-judge pilot (PLAN.md, preference step): the first N prompts of the
# candidate file (tools/posttrain/sample_candidates.py output), each ranked
# twice by claude-opus-5-5 through the logged-in Claude Code CLI (orders
# reversed), then best-vs-worst pairs where both rankings agree. Measures
# what the full ~3,000-prompt run will cost and keep, before it is spent.
#   bash experiments/chat_general/judge_pilot.sh CANDIDATES.jsonl [N=200] [NAME=pilot]
# Outputs are judge/NAME_*.json(l) (pilot: the 128-token pilot; pilot256: the
# 256-token one). The verdict cache is experiments/chat_general/judge/judge_cache.jsonl,
# the same one the full run uses, so verdicts are never paid for twice; --max-usd
# (MAX_USD, default 100: Jonathan, 19 Oct 2026; $200 from 28 Oct 2026) caps the
# cache's TOTAL recorded spend. The report adds the share of sampled runs that
# ended on end-of-text and the pairs whose two replies were both cut by the budget.
# Run from the microtorch root. Light work: one CLI call at a time, or WORKERS in flight.
set -euo pipefail
CAND=${1:?candidates jsonl}
N=${2:-200}
NAME=${3:-pilot}
MAX_USD=${MAX_USD:-100}
D=experiments/chat_general/judge
mkdir -p "$D"
REQ="$D/${NAME}_requests.jsonl"
head -n "$N" "$CAND" > "$REQ"
NREQ=$(wc -l < "$REQ")  # the prompts actually judged (the file may hold fewer than N)
CLAUDE_BIN=${CLAUDE_BIN:-$(command -v claude)}
ARGS=(--requests "$REQ" --cache "$D/judge_cache.jsonl" --rubric tools/posttrain/rubrics/chat_v1.txt
      --judge-id claude-opus-5-5)
J=tools/posttrain/chat_judge.py
python3 "$J" plan "${ARGS[@]}" | tee "$D/${NAME}_plan.json"
python3 "$J" run "${ARGS[@]}" --claude-bin "$CLAUDE_BIN" --max-calls $((2 * NREQ)) --max-usd "$MAX_USD" --workers "${WORKERS:-1}" \
  | tee "$D/${NAME}_run.json"
python3 "$J" pairs "${ARGS[@]}" --out "$D/${NAME}_pairs.jsonl" | tee "$D/${NAME}_pairs_summary.json"
python3 - "$D" "$NREQ" "$NAME" "$REQ" <<'PY'
import json, sys
d, n, name, req = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
run = json.load(open(f"{d}/{name}_run.json"))
pr = json.load(open(f"{d}/{name}_pairs_summary.json"))
reqs = {r["uri"]: r for r in map(json.loads, open(req, encoding="utf-8"))}
runs = [x for r in reqs.values() for x in r.get("sampler", {}).get("runs", [])]


def stop(r, text):
    s = r.get("sampler", {})
    if "runs" not in s:
        return None
    return s["runs"][s["candidate_runs"][r["candidates"].index(text)][0]]["stop"]


both_cut = 0
for p in map(json.loads, open(f"{d}/{name}_pairs.jsonl", encoding="utf-8")):
    r = reqs[p["uri"]]
    both_cut += stop(r, p["chosen"]) == "length" and stop(r, p["rejected"]) == "length"
calls = max(run["calls"], 1)
report = {"prompts": n, "calls": run["calls"], "failures": run["failures"],
          "usd": run["cost_usd"], "usd_per_call": round(run["cost_usd"] / calls, 5),
          "total_spent_usd": run["total_spent_usd"], "pairs": pr["pairs"],
          "pairs_per_prompt": round(pr["pairs"] / n, 3),
          "position_disagreement_rate": round(pr["position_disagreements"] / n, 3),
          "length_mismatched": pr["length_mismatched"], "unranked": pr["unranked"],
          "runs": len(runs), "runs_eos_share": round(sum(x["stop"] == "eos" for x in runs) / max(len(runs), 1), 3),
          "pairs_both_cut": both_cut,
          "projected_usd_3000_prompts": round(run["cost_usd"] / n * 3000, 2)}
json.dump(report, open(f"{d}/{name}_report.json", "w"), indent=1)
print(json.dumps(report, indent=1))
PY
