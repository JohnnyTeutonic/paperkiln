#!/usr/bin/env python3
"""Amendment 5's plain-path receipt: E1's 20m_L16 run trained again at the arm's commit for its first steps,
compared bit for bit with E1's own log.

    python experiments/tiny_chat_2026-10/x5/bitid.py --replay ART/s0_x5/replay_e1_L16_r1 \
        --e1 ART/e1/pt_e1_L16_r1 --commit SHA --e1-commit dc29b5e6... --out experiments/tiny_chat_2026-10/x5/plain_bitid.json

The replay is tiny_chat_s0_x5.json's training job replay_e1_L16_r1: E1's pt_e1_L16_r1 spec with train.steps 500
(the warmup length, so the learning-rate schedule up to there is the full run's). The receipt is ok when the
two specs agree except train.steps, name and out_dir; the model events agree; and every step record of the
replay (loss, gradient norm, learning rate) and every validation record up to its last step equal E1's exactly,
as logged (mtstudio writes each float with its shortest round-trip digits, so equal text is equal bits).
"""
from __future__ import annotations

import argparse
import json
import os


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
    spec_same = bare(spec_of(replay)) == bare(spec_of(e1))
    starts = sum(1 for e in eev[:] if e.get("event") == "start")
    res = {"steps_compared": len(steps_r), "last_step": last, "evals_compared": sorted(evals_r),
           "differences": diff[:20], "n_differences": len(diff), "spec_equal_but_steps": spec_same,
           "model_event_equal": model_r == model_e and model_r is not None, "e1_starts": starts,
           "first_loss": steps_r.get(1, {}).get("loss"), "last_loss": steps_r.get(last, {}).get("loss"),
           "val_loss": {k: v["val_loss"] for k, v in evals_r.items()}}
    res["ok"] = (bool(steps_r) and not diff and spec_same and res["model_event_equal"]
                 and sorted(steps_r) == list(range(1, last + 1)))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--replay", required=True)
    ap.add_argument("--e1", required=True)
    ap.add_argument("--commit", required=True, help="the arm's commit the replay ran at")
    ap.add_argument("--e1-commit", required=True, help="E1's pinned commit")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = {"receipt": "amendment 5: plain 20m_L16 at the arm's commit against E1's pt_e1_L16_r1 log",
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
