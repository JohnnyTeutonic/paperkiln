#!/usr/bin/env python3
"""A stand-in for mtstudio in A34's scheduler simulation (tests/test_a34_schedule.py): A31's stand-in (fake_mtstudio_a31.py, imported)
with the floor. Claude Opus 5.5, 15 Nov 2026.

  run SPEC   A31's stand-in; a spec with data.twin_floor_ref must find that file (else exit 1, as the trainer refuses), logs a
             twin_floor event, and every twin record carries a floor record: on the gate spec (a reference path holding
             "gate_ref") inactive with gap 0.5 at step 1 and active from step 2, as a34_gate.py expects; otherwise a step-1
             gap of 1e-4 (the scorer and trainer agreeing)
  score DIR --in REQ --out OUT   per-token own-context rows (row k of 5 in 0, 2, 4, or every row of a two-row own request)
             at -1.0 and mismatched rows at -1.5, two tokens a record; the sweep's floor runs move them: Fd_0p3 lowers own rows
             to -1.1 (fails the hold), Fd_1p0 and Fd_3p0 lower mismatched rows to -2.1 (dec +0.6 over D0r: retained)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fake_mtstudio_a31 as A31  # noqa: E402


def run(spec_path):
    s = json.load(open(spec_path))
    ref = (s.get("data") or {}).get("twin_floor_ref")
    if ref and not os.path.exists(ref):
        print(f"cannot open data.twin_floor_ref {ref}", file=sys.stderr)
        return 1
    rc = A31.run(spec_path)
    if rc or not ref:
        return rc
    out = s["out_dir"]
    ev_p = os.path.join(out, "events.jsonl")
    ev = [json.loads(x) for x in open(ev_p) if x.strip()]
    gate = "gate_ref" in ref
    if gate:
        groups = 16
    else:   # the training twins: the own-row request file in the chat dir's upload folder holds two rows a group
        up = os.path.join(os.path.dirname(os.path.dirname(ref)), "upload", "a34_own_req.jsonl")
        groups = sum(1 for x in open(up) if x.strip()) // 2
    bad = s.get("name", "") in os.environ.get("FAKE_BAD_PARITY", "").split(",")   # a sweep run whose preflight read badly
    with open(ev_p + ".tmp", "w") as f:
        f.write(json.dumps({"event": "twin_floor", "floor": s["train"]["twin_loss"]["floor"], "ref": ref}) + "\n")
        f.write(json.dumps({"event": "twin_floor_parity", "groups": groups, "max_abs": 0.02 if bad else 2e-7,
                            "mean_abs": 1e-7, "worst_group": 0, "tolerance": 5e-3}) + "\n")
        for e in ev:
            if e.get("event") == "step" and "twin" in e:
                if gate:
                    e["twin"]["floor"] = ({"loss": 0.0, "active": 0.0, "gap": -0.5, "gap_absmax": 0.5} if e["step"] == 1
                                          else {"loss": 0.5, "active": 1.0, "gap": 0.5, "gap_absmax": 0.5})
                else:
                    e["twin"]["floor"] = ({"loss": 0.0, "active": 0.0, "gap": 0.0, "gap_absmax": 1e-7} if e["step"] == 1 else
                                          {"loss": 0.01, "active": 0.2, "gap": -0.01, "gap_absmax": 0.1})
            f.write(json.dumps(e) + "\n")
    os.replace(ev_p + ".tmp", ev_p)
    return 0


def score(args):
    d, req = args[0], args[args.index("--in") + 1]
    out = args[args.index("--out") + 1]
    try:
        name = json.load(open(os.path.join(d, "spec.json"))).get("name", "")
    except (OSError, ValueError):
        name = ""
    rows = [json.loads(x) for x in open(req) if x.strip()]
    own_v, cross_v = -1.0, -1.5
    if name.startswith("Fd_0p3"):
        own_v = -1.1
    elif name.startswith(("Fd_1p0", "Fd_3p0")):
        cross_v = -2.1
    with open(out, "w") as f:
        for i, r in enumerate(rows):
            own = r.get("row") in ("AA", "BB") if "row" in r else (i % 5) in (0, 2, 4)
            v = own_v if own else cross_v
            f.write(json.dumps({"i": i, "sum": 2 * v, "logprobs": [v, v], "over_ctx": False, "continuation_ids": [1, 2]}) + "\n")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1]
    sys.exit(run(sys.argv[2]) if cmd == "run" else score(sys.argv[2:]) if cmd == "score" else 2)
