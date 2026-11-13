#!/usr/bin/env python3
"""A31's selection rule (lane PROTOCOL.md section 50), fixed before any dev result. Claude Opus 5.5, 13 Nov 2026; revised after the
launch reviews of 13 Nov 2026 (a failed sweep run is ineligible, never fatal; no promotion without an eligible candidate; validated
scores through the lane's one measurement code).

Reads the dev sweep at seed 99. A run is **complete** when its STATUS is FINISHED, its events hold a finite held-out SFT loss at
step 1,024 (the trainer's eval), no early readout of it records a health stop or an unread health check, and its dev scores are
whole and valid (a31_readout.validated_sets on the dev request file). D0 must be complete, or no candidate is eligible.
  eligible(L)  the run is complete and its held-out SFT loss at step 1,024 is at most 1.02 times D0's
  arm M (margin):            among eligible lambdas, the highest dev both-right per token; ties to the smaller lambda
  arm N (margin + nuisance): among eligible lambdas (= mu), the highest dev selectivity (decisive less nuisance shift, per token);
                              ties to the smaller lambda
  An arm with no eligible lambda is not run (every one of its decisions false).
Writes decisions.json {"decisions": {"a31": {"M_<L>": bool, "N_<L>": bool}}, "arms": ..., "runs": ...}, which jobs_vm.py's promoted_by
reads. This job runs after every sweep job has ended, FINISHED or FAILED (after_terminal).

    python3 tools/colab_chat/a31_select.py --chat /content/chat --out DIR
"""
import argparse
import json
import math
import os
import sys

GRID = (0.1, 0.3, 1.0)
TOL = 1.02
FINAL_STEP = 1024
EARLY = (128, 256, 512)


def lname(lam):
    return str(lam).replace(".", "p")


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def run_record(chat, run, R, man, trips):
    """{complete, why, val_loss, both_tok, sel} for one sweep run."""
    rec = {"complete": False}
    st = os.path.join(chat, f"STATUS_{run}")
    status = open(st).read().strip() if os.path.exists(st) else "NONE"
    if status != "FINISHED":
        rec["why"] = f"status {status}"
        return rec
    for s in EARLY:
        p = os.path.join(chat, f"early_{run}_{s}", "early.json")
        if not os.path.exists(p):   # a health check that did not run is not a pass (round-2 reviews, 13 Nov 2026)
            rec["why"] = f"no early readout at step {s}"
            return rec
        e = json.load(open(p))
        if e.get("health_stop") or e.get("health_unread"):
            rec["why"] = f"health at step {s}: {e.get('health_stop') or e.get('health_unread')}"
            return rec
    ev = [json.loads(x) for x in open(os.path.join(chat, run, "events.jsonl")) if x.strip()]
    vals = [e.get("val_loss") for e in ev if e.get("event") == "eval" and e.get("step") == FINAL_STEP]
    if not vals or not finite(vals[-1]):
        rec["why"] = f"no finite held-out SFT loss at step {FINAL_STEP}"
        return rec
    rec["val_loss"] = vals[-1]
    try:
        rows = R.validated_sets(man, trips, R.read_jsonl(os.path.join(chat, f"score_{run}", "scores.jsonl")))
    except (OSError, ValueError) as e:
        rec["why"] = f"dev scores: {e}"
        return rec
    dev = list(rows["dev"].values())
    if len(dev) < 0.99 * len(trips["dev"]):
        rec["why"] = f"dev scores cover {len(dev)} of {len(trips['dev'])} triplets"
        return rec
    rec.update({"complete": True, "n": len(dev), "both_tok": sum(r["both_tok"] for r in dev) / len(dev),
                "sel": sum(r["sel"] for r in dev) / len(dev)})
    return rec


def safe_record(chat, run, R, man, trips):
    """run_record, with any failure to read a run's files (missing or malformed logs, early readouts or scores) making that run
    ineligible rather than ending the selection."""
    try:
        return run_record(chat, run, R, man, trips)
    except Exception as e:  # noqa: BLE001
        return {"complete": False, "why": f"unreadable: {type(e).__name__}: {e}"}


def choose(table, key):
    """The lambda the rule picks for one arm from {lambda: record with "eligible"}, or None when none is eligible."""
    ok = [x for x in GRID if table[x]["eligible"]]
    if not ok:
        return None
    best = max(table[x][key] for x in ok)
    return min(x for x in ok if table[x][key] == best)


def decide(runs):
    """runs: {run name: run_record}. Returns the decisions document."""
    base = runs["D0"]
    out = {"tolerance": TOL, "D0": base, "arms": {}}
    dec = {}
    for arm, prefix, key in (("M", "Dm", "both_tok"), ("N", "Dn", "sel")):
        table = {}
        for x in GRID:
            r = dict(runs[f"{prefix}_{lname(x)}"])
            r["eligible"] = bool(base["complete"] and r["complete"] and r["val_loss"] <= TOL * base["val_loss"])
            table[x] = r
        pick = choose(table, key)
        out["arms"][arm] = {"table": {str(x): t for x, t in table.items()}, "chosen": pick,
                            "why": f"highest dev {key} among eligible" if pick is not None else "no eligible lambda: the arm is not run"}
        for x in GRID:
            dec[f"{arm}_{lname(x)}"] = pick is not None and x == pick
    out["decisions"] = {"a31": dec}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chat", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    up = os.path.join(a.chat, "upload")
    sys.path.insert(0, up)
    import a31_readout as R   # the lane's readout.py, uploaded and hash-pinned
    man = json.load(open(os.path.join(up, "a31_early_req.manifest.json")))
    trips = {n: R.read_jsonl(os.path.join(up, f"a31_{n}_triplets.jsonl")) for n in man["sets"]}
    names = ["D0"] + [f"{p}_{lname(x)}" for p in ("Dm", "Dn") for x in GRID]
    out = decide({r: safe_record(a.chat, r, R, man, trips) for r in names})
    os.makedirs(a.out, exist_ok=True)
    tmp = os.path.join(a.out, "decisions.json.tmp")
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, os.path.join(a.out, "decisions.json"))
    print(json.dumps(out["decisions"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
