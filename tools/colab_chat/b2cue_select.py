#!/usr/bin/env python3
"""A32's lambda selection (b2cue; REGISTRATION_DRAFT.md section 4.2, frozen at eef513ad), on the pod after every sweep job has ended;
replayed at home by the lane's score_b2cue.py from the same bound files. Claude Opus 5.5, 15 Nov 2026.

The sweep at seed 37 from N250's SFT1500: D0 (P's recipe) and Dn at lambda = mu in (0.1, 0.3, 1.0).
  complete     STATUS FINISHED; a finite held-out SFT loss at step 1,024; no early readout with a health stop or an unread check; the
               selection scores whole and valid (A31's readout.validated_sets). D0 must be complete.
  ineligible   a documented failure by the run's own record: STATUS FAILED, a health stop or an unread health check, no finite held-out
               loss at 1,024
  void         evidence needed to settle a run's status that is missing or cannot be read (no STATUS, an absent early.json, a score file
               that cannot be read or validated): outcome `void_selection`, no margin arm runs
  eligible     complete, held-out SFT loss at most 1.02 x D0's
  pool         held-out-family triplets of S_hf_train, S_hf_held, S_hf_held_nq less every `speaker` qualifier triplet; families weighted
               equally, triplets equally within a family (b2cue_measures.py)
  rule         among eligible lambdas whose pooled nuisance shift is at most D0's + 0.005, the highest pooled dec^D (the decisive shift on
               answer-deciding tokens); a lambda within 0.005 of the best and smaller wins the tie; if none meets the nuisance condition,
               the smallest eligible lambda; if none is eligible, `no_arm`.
Writes decisions.json {"outcome", "chosen", "D0", "runs", ...}.

    python3 tools/colab_chat/b2cue_select.py --chat /content/chat --out DIR
"""
import argparse
import json
import math
import os
import sys

GRID = (0.1, 0.3, 1.0)
TOL = 1.02
NUI_SLACK = 0.005
TIE = 0.005
FINAL_STEP = 1024
EARLY = (128, 256, 512)
SETS = ("S_hf_train", "S_hf_held", "S_hf_held_nq")


def lname(lam):
    return str(lam).replace(".", "p")


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


class Void(Exception):
    pass


def run_record(chat, run, M, man, trips, by_id, vocab, AT, R):
    st = os.path.join(chat, f"STATUS_{run}")
    if not os.path.exists(st):
        raise Void(f"{run}: no STATUS")
    status = open(st).read().strip()
    if status != "FINISHED":
        return {"complete": False, "why": f"status {status}"}
    for s in EARLY:
        p = os.path.join(chat, f"early_{run}_{s}", "early.json")
        if not os.path.exists(p):
            raise Void(f"{run}: no early readout at step {s}")
        e = json.load(open(p))
        if e.get("health_stop") or e.get("health_unread"):
            return {"complete": False, "why": f"health at step {s}: {e.get('health_stop') or e.get('health_unread')}"}
    ev = [json.loads(x) for x in open(os.path.join(chat, run, "events.jsonl")) if x.strip()]
    vals = [e.get("val_loss") for e in ev if e.get("event") == "eval" and e.get("step") == FINAL_STEP]
    if not vals or not finite(vals[-1]):
        return {"complete": False, "why": f"no finite held-out SFT loss at step {FINAL_STEP}"}
    p = os.path.join(chat, f"score_{run}", "scores.jsonl")
    if not os.path.exists(p):
        raise Void(f"{run}: no selection scores")
    try:
        scores = R.read_jsonl(p)
        R.validated_sets(man, trips, scores)
        rows = {}
        for name, info in man["sets"].items():
            if name not in SETS:
                continue
            a = info["first_request"]
            rows.update(M.per_triplet(trips[name], scores[a:a + 5 * info["triplets"]], vocab, AT, R))
    except (OSError, ValueError, KeyError) as e:
        raise Void(f"{run}: selection scores: {e}")
    dec = M.pooled(rows, by_id, M.in_selection_pool, "dec_d")
    nui = M.pooled(rows, by_id, M.in_selection_pool, "nui")
    if dec is None or nui is None:
        raise Void(f"{run}: a held-out family is empty in the pool")
    n = sum(1 for i in rows if M.in_selection_pool(by_id[i]))
    return {"complete": True, "val_loss": vals[-1], "dec_d": dec, "nui": nui, "n_pool": n}


def decide(runs):
    """runs: {name: record}. The registered rule; no void handling here (a Void is raised before)."""
    base = runs["D0"]
    out = {"tolerance": TOL, "nuisance_slack": NUI_SLACK, "tie": TIE, "D0": base, "runs": {}}
    if not base.get("complete"):
        out.update({"outcome": "void_selection", "chosen": None, "why": f"D0 is not complete: {base.get('why')}"})
        return out
    elig = []
    for x in GRID:
        r = dict(runs[f"Dn_{lname(x)}"])
        r["eligible"] = bool(r.get("complete") and r["val_loss"] <= TOL * base["val_loss"])
        r["nuisance_ok"] = bool(r["eligible"] and r["nui"] <= base["nui"] + NUI_SLACK)
        out["runs"][str(x)] = r
        if r["eligible"]:
            elig.append(x)
    if not elig:
        out.update({"outcome": "no_arm", "chosen": None, "why": "no eligible lambda"})
        return out
    ok = [x for x in elig if out["runs"][str(x)]["nuisance_ok"]]
    if not ok:
        out.update({"outcome": "selected", "chosen": min(elig), "why": "no eligible lambda meets the nuisance condition: the smallest eligible"})
        return out
    best = max(out["runs"][str(x)]["dec_d"] for x in ok)
    pick = min(x for x in ok if out["runs"][str(x)]["dec_d"] >= best - TIE)
    out.update({"outcome": "selected", "chosen": pick, "why": "the highest pooled dec^D among eligible lambdas meeting the nuisance condition"})
    return out


def select(chat, up):
    sys.path.insert(0, up)
    import b2cue_readout as R
    import b2cue_measures as M
    import answer_tokens as AT
    man = json.load(open(os.path.join(up, "b2cue_selection_req.manifest.json")))
    trips = {n: R.read_jsonl(os.path.join(up, f"b2cue_{n}.jsonl")) for n in man["sets"]}
    by_id = {t["id"]: t for n in trips for t in trips[n]}
    vocab = AT.Vocab(os.path.join(up, "vocab.gguf"), os.path.join(up, "llama3_bpe.py"))
    names = ["D0"] + [f"Dn_{lname(x)}" for x in GRID]
    try:
        runs = {r: run_record(chat, r, M, man, trips, by_id, vocab, AT, R) for r in names}
    except Void as e:
        return {"outcome": "void_selection", "chosen": None, "why": str(e)}
    return decide(runs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chat", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    try:
        out = select(a.chat, os.path.join(a.chat, "upload"))
    except Exception as e:  # noqa: BLE001  (an error reading the evidence is an execution failure)
        out = {"outcome": "void_selection", "chosen": None, "why": f"{type(e).__name__}: {e}"}
    out["producer"] = "b2cue_select.py"
    os.makedirs(a.out, exist_ok=True)
    tmp = os.path.join(a.out, "decisions.json.tmp")
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, os.path.join(a.out, "decisions.json"))
    print(json.dumps({k: out[k] for k in ("outcome", "chosen")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
