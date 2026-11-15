#!/usr/bin/env python3
"""A34's rule for the floor weight (the lane's cue_selective/a34/REGISTRATION.md), fixed before any A34 result. Claude Opus 5.5,
15 Nov 2026. Runs on the pod after every sweep job has ended (after_terminal), FINISHED or FAILED.

Reads the floor sweep at seed 99 against D0r (plain SFT at seed 99 on this build, the sweep's own reference run). A run is
**complete** when its STATUS is FINISHED, no early readout of it records a health stop or an unread check (and all three exist),
its events hold a finite held-out SFT loss at step 1,024, its dev scores are whole and valid (a31_readout.validated_sets on the
early request file) and cover 99% of the dev triplets, and (for a floor run) its step-1 log records the floor with |ref - own| at
most PARITY (the scorer's references and the trainer's own terms agree at the start weights). D0r must be complete.
  eligible(f)   complete, held-out SFT loss at step 1,024 at most 1.02 times D0r's
  hold(f)       dev own-context change against D0r, mean over dev triplets of ((AA/n_A + BB/n_B)/2)_f - (same)_D0r, >= HOLD
  retain(f)     dev decisive-shift gain against D0r, mean dec_f - dec_D0r, >= RETAIN (half A31's Dm_0p3 - D0 dev gain, 1.024)
  chosen        the smallest eligible f with hold and retain; else the smallest eligible f with hold (flag retain_not_met);
                else the largest eligible f (flag hold_not_met); else none, and neither F nor NF is run
Writes decisions.json {"decisions": {"a34": {"F_<f>": bool, "NF_<f>": bool}}, "chosen", "flags", "table", "D0r"}; NF takes
F's floor weight.

    python3 tools/colab_chat/a34_select.py --chat /content/chat --out DIR
"""
import argparse
import json
import math
import os
import sys

FLOOR_GRID = (0.3, 1.0, 3.0)
TOL = 1.02
HOLD = -0.05
RETAIN = 0.512
PARITY = 5e-3
FINAL_STEP = 1024
EARLY = (128, 256, 512)


def lname(x):
    return str(x).replace(".", "p")


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def dev_terms(R, man, trips, scores):
    """{id: (own, dec)} on the dev set, after the lane's validation of the whole file. own = (AA/n_A + BB/n_B) / 2."""
    rows = R.validated_sets(man, trips, scores)
    info = man["sets"]["dev"]
    out = {}
    for i, t in enumerate(trips["dev"]):
        if t["id"] not in rows["dev"]:
            continue   # over the context
        aa, ba, bb, ab = scores[info["first_request"] + 5 * i:info["first_request"] + 5 * i + 4]
        na, nb = len(aa["logprobs"]), len(bb["logprobs"])
        out[t["id"]] = ((aa["sum"] / na + bb["sum"] / nb) / 2, rows["dev"][t["id"]]["dec"])
    return out


def run_record(chat, run, R, man, trips, floor):
    rec = {"complete": False}
    st = os.path.join(chat, f"STATUS_{run}")
    status = open(st).read().strip() if os.path.exists(st) else "NONE"
    if status != "FINISHED":
        rec["why"] = f"status {status}"
        return rec
    for s in EARLY:
        p = os.path.join(chat, f"early_{run}_{s}", "early.json")
        if not os.path.exists(p):
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
    if floor:
        s1 = [e for e in ev if e.get("event") == "step" and e.get("step") == 1]
        fl = ((s1[-1].get("twin") or {}).get("floor") or {}) if s1 else {}
        if not finite(fl.get("gap_absmax")) or fl["gap_absmax"] > PARITY:
            rec["why"] = f"step 1 floor parity {fl.get('gap_absmax')!r} not within {PARITY}"
            return rec
        rec["parity_step1"] = fl["gap_absmax"]
    terms = dev_terms(R, man, trips, R.read_jsonl(os.path.join(chat, f"score_{run}", "scores.jsonl")))
    if len(terms) < 0.99 * len(trips["dev"]):
        rec["why"] = f"dev scores cover {len(terms)} of {len(trips['dev'])} triplets"
        return rec
    rec.update({"complete": True, "terms": terms})
    return rec


def safe_record(chat, run, R, man, trips, floor):
    try:
        return run_record(chat, run, R, man, trips, floor)
    except Exception as e:  # noqa: BLE001  (an unreadable run is ineligible, never fatal)
        return {"complete": False, "why": f"unreadable: {type(e).__name__}: {e}"}


def decide(runs):
    base = runs["D0r"]
    table = {}
    for f in FLOOR_GRID:
        r = {k: v for k, v in runs[f"Fd_{lname(f)}"].items() if k != "terms"}
        r["eligible"] = bool(base["complete"] and runs[f"Fd_{lname(f)}"]["complete"] and r["val_loss"] <= TOL * base["val_loss"])
        if r["eligible"]:
            t, b = runs[f"Fd_{lname(f)}"]["terms"], base["terms"]
            ids = sorted(set(t) & set(b))
            r["n"] = len(ids)
            r["d_own"] = sum(t[i][0] - b[i][0] for i in ids) / len(ids)
            r["d_dec"] = sum(t[i][1] - b[i][1] for i in ids) / len(ids)
            r["hold"] = r["d_own"] >= HOLD
            r["retain"] = r["d_dec"] >= RETAIN
        table[f] = r
    ok = [f for f in FLOOR_GRID if table[f]["eligible"]]
    both = [f for f in ok if table[f]["hold"] and table[f]["retain"]]
    hold = [f for f in ok if table[f]["hold"]]
    flags = []
    if both:
        pick = min(both)
    elif hold:
        pick, flags = min(hold), ["retain_not_met"]
    elif ok:
        pick, flags = max(ok), ["hold_not_met"]
    else:
        pick, flags = None, ["no_eligible_floor"]
    dec = {}
    for f in FLOOR_GRID:
        dec[f"F_{lname(f)}"] = dec[f"NF_{lname(f)}"] = pick is not None and f == pick
    return {"rule": {"tolerance": TOL, "hold": HOLD, "retain": RETAIN, "parity": PARITY},
            "D0r": {k: v for k, v in base.items() if k != "terms"}, "table": {str(f): t for f, t in table.items()},
            "chosen": pick, "flags": flags, "decisions": {"a34": dec}}


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
    runs = {"D0r": safe_record(a.chat, "D0r", R, man, trips, False)}
    for f in FLOOR_GRID:
        runs[f"Fd_{lname(f)}"] = safe_record(a.chat, f"Fd_{lname(f)}", R, man, trips, True)
    out = decide(runs)
    os.makedirs(a.out, exist_ok=True)
    tmp = os.path.join(a.out, "decisions.json.tmp")
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=1)
    os.replace(tmp, os.path.join(a.out, "decisions.json"))
    print(json.dumps({"chosen": out["chosen"], "flags": out["flags"], "decisions": out["decisions"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
