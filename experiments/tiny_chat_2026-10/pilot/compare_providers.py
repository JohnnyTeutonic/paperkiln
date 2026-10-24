#!/usr/bin/env python3
"""The floor pilot on two L4 providers (PREREGISTRATION.md, implementation note of 24 Oct 2026): the Colab run
(verdict.json) and its RunPod repeat, the same jobs file (tools/colab_chat/tiny_chat_pilot.json: 20m_L8, 35%
dialogue, seed 99, lr 1e-3, its rungs, post job and Carry-5). Operational only: the pilot stays excluded from
every analysis; this file says whether a RunPod L4 reproduces a Colab L4 run to sampling noise.

    python experiments/tiny_chat_2026-10/pilot/compare_providers.py --colab C:/ml_artifacts/tiny_chat/pilot \\
        --runpod C:/ml_artifacts/tiny_chat/pilot_runpod --runpod-state STATE.json --out pilot/runpod_comparison.json

Per quantity: both values, the gap, and for Carry-5 accuracies the binomial standard error of the gap
(sqrt(2 p (1-p) / n), the two runs' items scored independently) and the gap in those units. Gate verdicts are
computed with analyze.py's own gate (FLOOR, FLOOR_QALONE, get) on each run. Wall-clock and throughput are host
properties (the note: S9/S10 do not carry over) and are reported, not tested. Spend: the RunPod runner's ledger.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import analyze  # noqa: E402

PT, POST = "pt_pilot_s35_r99", "post_pilot_s35_r99"
RUNG_STEPS = (500, 1000, 2500, 5000)


def jl(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def evals(d):
    out = {}
    with open(os.path.join(d, "events.jsonl"), encoding="utf-8") as f:
        for ln in f:
            if '"eval"' in ln:
                e = json.loads(ln)
                out[e["step"]] = e["val_loss"]  # a step logged twice (a resume) counts once, its last record
    return out


def verdict(s):
    q = {k: analyze.get(s, k) for k in ("twin_self", "chance", "twin_ctx", "noctx_ctx", "twin_minus_qalone_ctx")}
    i = q["twin_self"] >= q["chance"] + analyze.FLOOR
    ii = q["twin_minus_qalone_ctx"] >= analyze.FLOOR_QALONE
    return "PASS" if i and ii else "FAIL"


def side(art):
    s = jl(os.path.join(art, POST, "summary.json"))
    res = jl(os.path.join(art, PT, "result.json"))
    ev = evals(os.path.join(art, PT))
    rungs = {}
    for st in RUNG_STEPS:
        p = os.path.join(art, f"rung_pilot_s35_r99_s{st}", "rung.json")
        if os.path.exists(p):
            r = jl(p)
            rungs[st] = {k: r.get(k) for k in ("val_loss", "acc_gold", "Y_base")} | {
                f"panel_{k}": v for k, v in (r.get("panels") or {}).items()}
    c5 = s["carry5"]
    n = {k: c5["pools"]["acc"][k]["gold"]["n"] for k in ("primary", "context")}
    return {"summary": s, "result": res, "evals": ev, "rungs": rungs, "n": n, "verdict": verdict(s),
            "provenance": res.get("provenance") or s.get("provenance")}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def step_records(d):
    with open(os.path.join(d, "events.jsonl"), encoding="utf-8") as f:
        evs = [json.loads(ln) for ln in f if '"step"' in ln or '"eval"' in ln]
    out = {}
    for e in evs:  # a step logged twice (a resume replays from the checkpoint) counts once, its last record
        if e.get("event") in ("step", "eval"):
            out[(e["event"], e["step"])] = {k: v for k, v in e.items() if k != "event"}
    return out


def identity(ca, ra, c, r):
    """Bitwise checks: the pretrained weights, every step and eval record of pretraining, every rung's record,
    the SFT weights (by the post job's own sha256)."""
    w = os.path.join(PT, "tc_pilot_s35_r99.safetensors")
    a, b = step_records(os.path.join(ca, PT)), step_records(os.path.join(ra, PT))
    common = set(a) & set(b)
    rung_same = {}
    for st in RUNG_STEPS:
        x = jl(os.path.join(ca, f"rung_pilot_s35_r99_s{st}", "rung.json"))
        y = jl(os.path.join(ra, f"rung_pilot_s35_r99_s{st}", "rung.json"))
        rung_same[str(st)] = all(x.get(k) == y.get(k) for k in x if k not in ("provenance", "gpu"))
    return {"pt_weights_sha256_equal": sha(os.path.join(ca, w)) == sha(os.path.join(ra, w)),
            "pt_records_compared": len(common), "pt_records_differing": sum(a[k] != b[k] for k in common),
            "rung_records_identical": rung_same,
            "sft_data_sha256_equal": c["summary"].get("sft_data_sha256") == r["summary"].get("sft_data_sha256"),
            "sft_weights_sha256_equal": c["summary"].get("sft_weights_sha256") == r["summary"].get("sft_weights_sha256")}


def gap(a, b, n=None, p=None):
    if a is None or b is None:
        return {"colab": a, "runpod": b, "gap": None}
    g = {"colab": a, "runpod": b, "gap": b - a}
    if n:
        pp = p if p is not None else (a + b) / 2
        se = math.sqrt(2 * pp * (1 - pp) / n)
        g.update(se_gap=se, gap_in_se=(b - a) / se if se else None)
    return g


def diff_gap(a, b, key, parts, n):
    """The gap in a difference of two accuracies (e.g. Y = multi_self - twin_self), its SE taking the two parts
    as independent within a run (conservative when they are positively correlated, as on shared items)."""
    g = gap(a.get(key), b.get(key))
    ps = [(a[x] + b[x]) / 2 for x in parts if a.get(x) is not None and b.get(x) is not None]
    if g["gap"] is not None and len(ps) == 2:
        g["se_gap"] = math.sqrt(2 * sum(p * (1 - p) for p in ps) / n)
        g["gap_in_se"] = g["gap"] / g["se_gap"]
    return g


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--colab", required=True)
    ap.add_argument("--runpod", required=True)
    ap.add_argument("--runpod-state", required=True, help="the RunPod runner's state.json (its spend ledger)")
    ap.add_argument("--balance-before", type=float, default=None)
    ap.add_argument("--balance-after", type=float, default=None)
    ap.add_argument("--colab-commit", default=None, help="the Colab run's paperkiln commit (its files predate provenance)")
    ap.add_argument("--colab-wall-hours", type=float, default=None, help="first VM up to all jobs home (runner.log)")
    ap.add_argument("--runpod-wall-hours", type=float, default=None)
    ap.add_argument("--note", action="append", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    c, r = side(a.colab), side(a.runpod)
    gc = {k: analyze.get(c["summary"], k) for k in analyze.CELL_KEYS}
    gr = {k: analyze.get(r["summary"], k) for k in analyze.CELL_KEYS}
    np_, nc = c["n"]["primary"], c["n"]["context"]
    acc = {}
    for k, n in (("multi_self", np_), ("twin_self", np_), ("noctx", np_), ("twin_ctx", nc), ("noctx_ctx", nc)):
        acc[k] = gap(gc[k], gr[k], n)
    gc["gold"], gr["gold"] = c["summary"]["carry5"]["pool"]["gold"], r["summary"]["carry5"]["pool"]["gold"]
    diffs = {"Y": diff_gap(gc, gr, "Y", ("multi_self", "twin_self"), np_),
             "Y_all": gap(gc["Y_all"], gr["Y_all"]),
             "Y_gold": diff_gap(gc, gr, "Y_gold", ("gold", "twin_self"), np_),
             "twin_minus_qalone_ctx": diff_gap(gc, gr, "twin_minus_qalone_ctx", ("twin_ctx", "noctx_ctx"), nc)}
    losses = {k: gap(gc[k], gr[k]) for k in ("panel_fineweb", "panel_dialogue", "sft_val")}
    steps = sorted(set(c["evals"]) & set(r["evals"]))
    dv = [r["evals"][s] - c["evals"][s] for s in steps]
    pt = {"best_val": gap(c["result"].get("best_val"), r["result"].get("best_val")),
          "final_val": gap(c["evals"].get(10000), r["evals"].get(10000)),
          "val_curve": {"steps_compared": len(steps), "max_abs_gap": max(map(abs, dv)) if dv else None,
                        "mean_gap": sum(dv) / len(dv) if dv else None,
                        "gap_at": {str(s): round(d, 5) for s, d in zip(steps, dv) if s % 2500 == 0 or s == 500}},
          "tokens_per_second": gap(c["result"].get("tokens_per_second"), r["result"].get("tokens_per_second")),
          "pt_wall_seconds": gap(c["result"].get("wall_seconds"), r["result"].get("wall_seconds"))}
    rungs = {str(st): {k: gap(c["rungs"].get(st, {}).get(k), r["rungs"].get(st, {}).get(k))
                       for k in ("val_loss", "acc_gold", "Y_base", "panel_fineweb", "panel_dialogue")}
             for st in RUNG_STEPS}
    post = {"sft_tok_per_s_steady": gap(c["summary"].get("sft_tok_per_s_steady"), r["summary"].get("sft_tok_per_s_steady")),
            "sft_seconds": gap(c["summary"].get("sft_seconds"), r["summary"].get("sft_seconds"))}
    st = jl(a.runpod_state)
    led = (st.get("runpod") or {}).get("ledger", {})
    sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "tools", "colab_chat"))
    import runpod_api
    usd = runpod_api.ledger_usd(led)
    pods = [{k: p.get(k) for k in ("id", "dc", "usd_per_hr", "start", "stop", "stopped_by")} |
            {"hours": round(((p.get("stop") or 0) - p["start"]) / 3600, 3) if p.get("stop") else None} for p in led.get("pods", [])]
    out = {"comparison": "floor pilot, Colab L4 vs RunPod L4 (PREREGISTRATION.md implementation note, 24 Oct 2026)",
           "excluded_from_analysis": True,
           "verdict_colab": c["verdict"], "verdict_runpod": r["verdict"],
           "verdict_colab_file": jl(os.path.join(HERE, "verdict.json")).get("verdict"),
           "carry5_accuracy": acc, "carry5_differences": diffs, "losses": losses, "pretraining": pt,
           "rungs": rungs, "post": post,
           "provenance": {"colab": c["provenance"], "runpod": r["provenance"]},
           "runpod_spend": {"ledger_usd": round(usd, 4), "pods": pods, "volumes": led.get("volumes", []),
                            "balance_before": a.balance_before, "balance_after": a.balance_after},
           "identity": identity(a.colab, a.runpod, c, r),
           "campaign_wall_hours": {"colab": a.colab_wall_hours, "runpod": a.runpod_wall_hours},
           "commits": {"colab": (c["summary"].get("provenance") or {}).get("paperkiln_commit") or a.colab_commit,
                       "runpod": (r["summary"].get("provenance") or {}).get("paperkiln_commit")},
           "notes": a.note}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"verdict": [c["verdict"], r["verdict"]], "best_val": pt["best_val"],
                      "Y": diffs["Y"], "twin": acc["twin_self"], "twin_minus_qalone": diffs["twin_minus_qalone_ctx"],
                      "tok_s": pt["tokens_per_second"], "usd": out["runpod_spend"]["ledger_usd"]}, indent=1))


if __name__ == "__main__":
    main()
