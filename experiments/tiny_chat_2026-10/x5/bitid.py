#!/usr/bin/env python3
"""The plain-path receipt of amendments 5 and 6: a registered run trained again at the arm's commit for its
first steps, compared bit for bit with that run's own log.

    python experiments/tiny_chat_2026-10/x5/bitid.py --replay ART/s0_x5/replay_e1_L16_r1 \
        --e1 ART/e1/pt_e1_L16_r1 --commit SHA --e1-commit dc29b5e6... --out experiments/tiny_chat_2026-10/x5/plain_bitid.json

Amendment 5's replay is tiny_chat_s0_x5.json's training job replay_e1_L16_r1 (E1's pt_e1_L16_r1); amendment 6's
is tiny_chat_s0_150m_L4.json's replay_e1_60m_L32_r2 (E1 at 60M's pt_e1_60m_L32_r2). Each is the registered spec
with train.steps 500 (the warmup length, so the learning-rate schedule up to there is the full run's). The
receipt is ok when the two specs agree except train.steps, name and out_dir; the model events agree; and every
step record of the replay (loss, gradient norm, learning rate) and every validation record up to its last step
equal the registered run's exactly, as logged (mtstudio writes each float with its shortest round-trip digits,
so equal text is equal bits).

The model events agree (Jonathan's ruling, 27 Oct 2026; PREREGISTRATION.md A6.7) when every field both logs
carry is equal, no field is in the registered run's event alone, and every field in the replay's event alone is
one of LOGGED_LATER (fields a later mtstudio commit added to the logged event) at its fixed value.
"""
from __future__ import annotations

import argparse
import json
import os

# Fields paperkiln c6365b88 added to mtstudio's logged model event, with the values every non-flex family logs
# (mtstudio.cpp: norm_placement "pre" and final_norm true unless the family is flex). A replay at a later
# commit carries them; the registered run's log, at an earlier commit, does not.
LOGGED_LATER = {"final_norm": True, "norm_placement": "pre"}

RECEIPTS = {   # the receipt's label, by the replay's job name
    "replay_e1_L16_r1": "amendment 5: plain 20m_L16 at the arm's commit against E1's pt_e1_L16_r1 log",
    "replay_e1_60m_L32_r2": "amendment 6: plain 60m_L32 at the arm's commit against E1 at 60M's pt_e1_60m_L32_r2 log",
}


def label(replay, e1):
    name = os.path.basename(os.path.normpath(replay))
    return RECEIPTS.get(name, f"plain replay {name} at the arm's commit against "
                              f"{os.path.basename(os.path.normpath(e1))}'s log")


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def events(d):
    with open(os.path.join(d, "events.jsonl"), encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def spec_of(d):
    name = os.path.basename(os.path.normpath(d))
    return jload(os.path.join(d, f"{name}_spec.json"))


def bare(spec):
    s = json.loads(json.dumps(spec))
    s.pop("name", None)
    s.pop("out_dir", None)
    s.get("train", {}).pop("steps", None)
    return s


def compare(replay, e1):
    rev, eev = events(replay), events(e1)
    steps_r = {e["step"]: e for e in rev if e.get("event") == "step"}
    steps_e = {e["step"]: e for e in eev if e.get("event") == "step"}
    evals_r = {e["step"]: e for e in rev if e.get("event") == "eval"}
    evals_e = {e["step"]: e for e in eev if e.get("event") == "eval"}
    last = max(steps_r) if steps_r else 0
    diff = []
    for k in sorted(steps_r):
        a, b = steps_r[k], steps_e.get(k)
        for f in ("loss", "grad_norm", "lr"):
            if b is None or a.get(f) != b.get(f):
                diff.append({"step": k, "field": f, "replay": a.get(f), "e1": None if b is None else b.get(f)})
    for k in sorted(evals_r):
        a, b = evals_r[k], evals_e.get(k)
        if b is None or a.get("val_loss") != b.get("val_loss"):
            diff.append({"step": k, "field": "val_loss", "replay": a.get("val_loss"),
                         "e1": None if b is None else b.get("val_loss")})
    model_r = next((e for e in rev if e.get("event") == "model"), None)
    model_e = next((e for e in eev if e.get("event") == "model"), None)
    model = model_events(model_r, model_e)
    spec_same = bare(spec_of(replay)) == bare(spec_of(e1))
    starts = sum(1 for e in eev[:] if e.get("event") == "start")
    res = {"steps_compared": len(steps_r), "last_step": last, "evals_compared": sorted(evals_r),
           "differences": diff[:20], "n_differences": len(diff), "spec_equal_but_steps": spec_same,
           "model_event_equal": model["equal"], "e1_starts": starts,
           "first_loss": steps_r.get(1, {}).get("loss"), "last_loss": steps_r.get(last, {}).get("loss"),
           "val_loss": {k: v["val_loss"] for k, v in evals_r.items()}}
    if model["replay_only"] or model["e1_only"] or model["shared_differences"]:
        res["model_event"] = model
    res["ok"] = (bool(steps_r) and not diff and spec_same and res["model_event_equal"]
                 and sorted(steps_r) == list(range(1, last + 1)))
    return res


def model_events(r, e):
    """The model events agree on every field both carry; a field in E1's event alone fails; a field in the
    replay's event alone passes only when it is in LOGGED_LATER at its fixed value."""
    if r is None or e is None:
        return {"equal": False, "replay_only": {}, "e1_only": {}, "shared_differences": {}, "missing": True}
    r_only = {k: r[k] for k in r if k not in e}
    e_only = {k: e[k] for k in e if k not in r}
    shared = {k: {"replay": r[k], "e1": e[k]} for k in r if k in e and r[k] != e[k]}
    later = {k: {"value": v, "fixed": LOGGED_LATER.get(k), "ok": k in LOGGED_LATER and v == LOGGED_LATER[k]}
             for k, v in r_only.items()}
    ok = not e_only and not shared and all(x["ok"] for x in later.values())
    return {"equal": ok, "rule": " ".join(model_events.__doc__.split()), "replay_only": later, "e1_only": e_only,
            "shared_differences": shared}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--replay", required=True)
    ap.add_argument("--e1", required=True)
    ap.add_argument("--commit", required=True, help="the arm's commit the replay ran at")
    ap.add_argument("--e1-commit", required=True, help="E1's pinned commit")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = {"receipt": label(a.replay, a.e1),
           "replay_commit": a.commit, "e1_commit": a.e1_commit, **compare(a.replay, a.e1)}
    for k, d in (("replay_result", a.replay), ("e1_result", a.e1)):
        p = os.path.join(d, "result.json")
        res[k] = jload(p) if os.path.exists(p) else None
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("ok", "steps_compared", "n_differences", "spec_equal_but_steps",
                                          "model_event_equal", "val_loss")}))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
