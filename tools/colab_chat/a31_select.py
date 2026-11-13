#!/usr/bin/env python3
"""A31's selection rule (lane PROTOCOL.md section 50), fixed before any dev result. Claude Opus 5.5, 13 Nov 2026.

Reads the dev sweep at seed 99: each run's final held-out SFT loss (its events.jsonl, the eval at the last step) and its scores on the
dev twins (the first `dev` requests of the readout file, five per triplet, in readout.py's order).
  eligible(L)  the run's final held-out SFT loss is at most 1.02 times D0's
  arm 2 (margin):            among eligible lambdas, the highest dev both-right per token; ties to the smaller lambda;
                              none eligible: the smallest lambda
  arm 3 (margin + nuisance): among eligible lambdas (= mu), the highest dev selectivity (decisive less nuisance shift, per token);
                              ties to the smaller lambda; none eligible: the smallest lambda
Writes decisions.json {"decisions": {"a31": {"M_<L>": bool, "N_<L>": bool}}, "table": ...}, which jobs_vm.py's promoted_by reads.

    python3 tools/colab_chat/a31_select.py --chat /content/chat --out DIR --manifest MANIFEST.json
"""
import argparse
import json
import os
import sys

GRID = (0.1, 0.3, 1.0)
TOL = 1.02


def lname(lam):
    return str(lam).replace(".", "p")


def final_val(path):
    ev = [json.loads(x) for x in open(path) if x.strip()]
    evals = [e for e in ev if e.get("event") == "eval"]
    steps = [e["step"] for e in ev if e.get("event") == "step"]
    if not evals or not steps or evals[-1]["step"] != max(steps):
        raise SystemExit(f"{path}: no eval at the final step")
    return evals[-1]["val_loss"]


def dev_measures(path, n_dev):
    """Mean both-right per token and mean selectivity per token over the dev triplets."""
    recs = []
    with open(path) as f:
        for line in f:
            if len(recs) == 5 * n_dev:
                break
            if line.strip():
                recs.append(json.loads(line))
    if len(recs) != 5 * n_dev:
        raise SystemExit(f"{path}: fewer than {5 * n_dev} scores")
    both, sel, n = 0.0, 0.0, 0
    for i in range(n_dev):
        s = recs[5 * i:5 * i + 5]
        if any(r.get("over_ctx") for r in s):
            continue
        aa, ba, bb, ab, na = (r["sum"] for r in s)
        na_, nb_ = len(s[0]["logprobs"]), len(s[2]["logprobs"])
        both += float(aa / na_ > ab / nb_ and bb / nb_ > ba / na_)
        dec = ((aa - ba) / na_ + (bb - ab) / nb_) / 2
        sel += dec - abs(aa - na) / na_
        n += 1
    return {"both_tok": both / n, "sel": sel / n, "n": n}


def choose(table, key):
    """The lambda the rule picks for one arm from {lambda: {"eligible", key}}."""
    ok = [x for x in GRID if table[x]["eligible"]]
    if not ok:
        return min(GRID), "none eligible: the smallest"
    best = max(table[x][key] for x in ok)
    return min(x for x in ok if table[x][key] == best), "highest " + key


def decide(val, dev):
    """val: {run: final held-out SFT loss}; dev: {run: dev measures}. Returns the decisions document."""
    base = val["D0"]
    out = {"D0_val_loss": base, "tolerance": TOL, "arms": {}}
    dec = {}
    for arm, prefix, key in (("M", "Dm", "both_tok"), ("N", "Dn", "sel")):
        table = {x: {"val_loss": val[f"{prefix}_{lname(x)}"], "eligible": val[f"{prefix}_{lname(x)}"] <= TOL * base,
                     **dev[f"{prefix}_{lname(x)}"]} for x in GRID}
        pick, why = choose(table, key)
        out["arms"][arm] = {"table": {str(x): t for x, t in table.items()}, "chosen": pick, "why": why}
        for x in GRID:
            dec[f"{arm}_{lname(x)}"] = x == pick
    out["decisions"] = {"a31": dec}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chat", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", required=True)
    a = ap.parse_args()
    n_dev = json.load(open(a.manifest))["sets"]["dev"]["triplets"]
    runs = ["D0"] + [f"{p}_{lname(x)}" for p in ("Dm", "Dn") for x in GRID]
    val = {r: final_val(os.path.join(a.chat, r, "events.jsonl")) for r in runs}
    dev = {r: dev_measures(os.path.join(a.chat, f"score_{r}", "scores.jsonl"), n_dev) for r in runs}
    out = decide(val, dev)
    out["dev_D0"] = dev["D0"]
    os.makedirs(a.out, exist_ok=True)
    tmp = os.path.join(a.out, "decisions.json.tmp")
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, os.path.join(a.out, "decisions.json"))
    print(json.dumps(out["decisions"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
