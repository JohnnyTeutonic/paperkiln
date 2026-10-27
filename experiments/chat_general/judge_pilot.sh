#!/usr/bin/env bash
# Claude-judge pilot (PLAN.md, preference step): the first N prompts of the
# candidate file (tools/posttrain/sample_candidates.py output), each ranked
# twice by claude-opus-5-5 through the logged-in Claude Code CLI (orders
# reversed), then best-vs-worst pairs where both rankings agree. Measures
# what the full ~3,000-prompt run will cost and keep, before it is spent.
#   bash experiments/chat_general/judge_pilot.sh CANDIDATES.jsonl [N=200]
# The verdict cache is experiments/chat_general/judge/judge_cache.jsonl, the
# same one the full run uses, so pilot verdicts are never paid for twice;
# --max-usd 100 caps the cache's TOTAL recorded spend (Jonathan, 19 Oct 2026).
# Run from the microtorch root. Light work (one CLI call at a time).
set -euo pipefail
CAND=${1:?candidates jsonl}
N=${2:-200}
D=experiments/chat_general/judge
mkdir -p "$D"
REQ="$D/pilot_requests.jsonl"
head -n "$N" "$CAND" > "$REQ"
NREQ=$(wc -l < "$REQ")  # the prompts actually judged (the file may hold fewer than N)
CLAUDE_BIN=${CLAUDE_BIN:-$(command -v claude)}
ARGS=(--requests "$REQ" --cache "$D/judge_cache.jsonl" --rubric tools/posttrain/rubrics/chat_v1.txt
      --judge-id claude-opus-5-5)
J=tools/posttrain/chat_judge.py
python3 "$J" plan "${ARGS[@]}" | tee "$D/pilot_plan.json"
python3 "$J" run "${ARGS[@]}" --claude-bin "$CLAUDE_BIN" --max-calls $((2 * N)) --max-usd 100 | tee "$D/pilot_run.json"
python3 "$J" pairs "${ARGS[@]}" --out "$D/pilot_pairs.jsonl" | tee "$D/pilot_pairs_summary.json"
python3 - "$D" "$NREQ" <<'PY'
import json, sys
d, n = sys.argv[1], int(sys.argv[2])
run = json.load(open(f"{d}/pilot_run.json"))
pr = json.load(open(f"{d}/pilot_pairs_summary.json"))
calls = max(run["calls"], 1)
report = {"prompts": n, "calls": run["calls"], "failures": run["failures"],
          "usd": run["cost_usd"], "usd_per_call": round(run["cost_usd"] / calls, 5),
          "total_spent_usd": run["total_spent_usd"], "pairs": pr["pairs"],
          "pairs_per_prompt": round(pr["pairs"] / n, 3),
          "position_disagreement_rate": round(pr["position_disagreements"] / n, 3),
          "length_mismatched": pr["length_mismatched"], "unranked": pr["unranked"],
          "projected_usd_3000_prompts": round(run["cost_usd"] / n * 3000, 2)}
json.dump(report, open(f"{d}/pilot_report.json", "w"), indent=1)
print(json.dumps(report, indent=1))
PY
