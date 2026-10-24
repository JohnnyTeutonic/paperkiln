#!/usr/bin/env python3
"""Run the relevance gate over gate_bench_v1.json and score it against the frozen labels.

    python experiments/arxiv_chain_2026-10-23/gate_bench.py --env-file .env --budget-usd 2 \\
        --out experiments/arxiv_chain_2026-10-23/gate_bench_v1_result.json

One gate call per paper (cached verdicts are reused at no cost). A refused or malformed reply counts as
a refusal. False refusal: an in_scope label that does not pass. False pass: an out_of_scope or spurious
label that passes. Ambiguous papers are reported, not scored."""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "papers"))
import llm_extract as L  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=os.path.join(HERE, "gate_bench_v1.json"))
    ap.add_argument("--env-file", default=None)
    ap.add_argument("--budget-usd", type=float, required=True)
    ap.add_argument("--tag", default="relevance_gate_bench_v1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists: results are immutable")
    b = json.load(open(a.bench, encoding="utf-8"))
    if a.env_file:
        L.load_env_key(a.env_file)
    cases = ([(i, "in_scope", "bench40", None) for i in b["in_scope_bench40"]]
             + [(i, "in_scope", "component", None) for i in b["in_scope_components"]]
             + [(i, "out_of_scope", "near_miss", None) for i in b["out_of_scope_near_misses"]]
             + [(i, "ambiguous", "ambiguous", None) for i in b["ambiguous"]]
             + [(i, b["expected_for_spurious"][i], "synthetic", m) for i, m in b["spurious_synthetic"].items()])
    rows, spent = [], 0.0
    for aid, label, group, synth in cases:
        meta = None
        if synth:
            meta = {"arxiv_id": aid, "title": synth["title"], "abstract": synth["abstract"]}
        else:
            cached = (L.CACHE / f"{aid}.meta.json").exists()
            meta = L.arxiv_metadata(aid)
            if not cached:
                time.sleep(3)
        try:
            g = L.run_gate(aid, None, a.tag, a.budget_usd, meta=meta)
            verdict, passes, ev = g["verdict"], g["passes"], g["evidence_found"]
            cost = g["gate"]["cost_usd"] or 0.0
            reason = g["reason"]
        except SystemExit as e:
            if str(e).startswith("budget:"):
                print(f"budget reached at {aid}: {e}")
                break
            verdict, passes, ev, cost, reason = "error", False, False, 0.0, str(e)[:300]
        spent += cost
        ok = None if label == "ambiguous" else (passes if label == "in_scope" else not passes)
        rows.append({"id": aid, "group": group, "label": label, "verdict": verdict, "passes": passes,
                     "evidence_found": ev, "correct": ok, "reason": reason, "cost_usd": cost})
        print(f"{aid:22s} {group:9s} {label:12s} -> {verdict:12s} {'PASS' if passes else 'REFUSE':6s} "
              f"{'' if ok is None else ('ok' if ok else 'WRONG')}")
    ins = [r for r in rows if r["label"] == "in_scope"]
    outs = [r for r in rows if r["label"] in ("out_of_scope", "spurious")]
    summary = {
        "in_scope": len(ins), "false_refusals": sum(not r["passes"] for r in ins),
        "false_refusal_ids": [r["id"] for r in ins if not r["passes"]],
        "out_or_spurious": len(outs), "false_passes": sum(r["passes"] for r in outs),
        "false_pass_ids": [r["id"] for r in outs if r["passes"]],
        "spurious_verdict_on_synthetic": {r["id"]: r["verdict"] for r in rows if r["group"] == "synthetic"},
        "ambiguous": {r["id"]: r["verdict"] for r in rows if r["label"] == "ambiguous"},
        "spent_usd": round(spent, 6), "tag_spent_usd": round(L.ledger_spent(a.tag), 6)}
    out = {"date": time.strftime("%Y-%m-%d"), "bench": os.path.basename(a.bench), "tag": a.tag,
           "model": L.MODEL, "effort": L.GATE_EFFORT, "prompt_sha256": L.gate_prompt_sha256(),
           "summary": summary, "rows": rows}
    L.write_json(a.out, out)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
