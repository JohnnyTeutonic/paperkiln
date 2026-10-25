#!/usr/bin/env python3
"""`mtstudio score` is stateless across requests: a request scored inside a sequence of requests equals the
same request scored alone (SCORE_AUDIT_2026-10-26.md, experiments/tiny_chat_2026-10/).

    python tools/score_sequence_check.py --mtstudio BIN --spec SPEC --in REQ.jsonl --out DIR
        [--alone N] [--tol 0] [--ref REF.out.jsonl --ref-tol 1e-3]

Scores REQ.jsonl (mtstudio score's input: one {"key", "prompt_ids", "continuation_ids"} per line, keys
distinct) three ways, each in its own process: in file order (DIR/seq.out.jsonl), in reverse order
(DIR/rev.out.jsonl: the order of the lengths reversed), and the first N requests one process each
(DIR/alone/; default all). Every request's sum, per-token log-probabilities and argmax must agree between
the sequence and the other two within --tol (0: bit for bit, the CPU build's contract; a device build's
cuBLAS may pick another algorithm in another process, so its gate passes a small --tol and records the
largest difference). --ref compares the sequence's log-probabilities with another engine's output for
the requests it holds (mtstudio score's or carry5_eval's IR backend's records: key, over_ctx, logprobs,
argmax, sum) within --ref-tol; argmax agreement is counted, not gated. Writes DIR/report.json and exits 1 when a gate fails.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def score(binary, spec, rows, path):
    inp = path + ".in.jsonl"
    with open(inp, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    p = subprocess.run([binary, "score", spec, "--in", inp, "--out", path], capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"mtstudio score failed on {inp} (rc {p.returncode}): {p.stderr.strip()[-800:]}")
    os.remove(inp)
    return {r["key"]: r for r in read_jsonl(path)}


def diff(a, b):
    """Largest |difference| over sum and per-token log-probs, plus argmax and over_ctx agreement."""
    if bool(a.get("over_ctx")) != bool(b.get("over_ctx")):
        return {"over_ctx_differs": True, "max_abs": float("inf"), "argmax_equal": False}
    if a.get("over_ctx"):
        return {"max_abs": 0.0, "argmax_equal": True}
    if len(a["logprobs"]) != len(b["logprobs"]):
        return {"length_differs": True, "max_abs": float("inf"), "argmax_equal": False}
    d = max([abs(a["sum"] - b["sum"])] + [abs(x - y) for x, y in zip(a["logprobs"], b["logprobs"])])
    return {"max_abs": d, "argmax_equal": list(a["argmax"]) == list(b["argmax"])}


def gate(name, A, B, keys, tol, argmax=True):
    """argmax=False (another engine's reference): the log-probs gate, argmax agreement counted only, since
    engines may ban different specials from the argmax."""
    worst, bad, same = 0.0, [], 0
    for k in keys:
        d = diff(A[k], B[k])
        worst = max(worst, d["max_abs"])
        same += d["argmax_equal"]
        if d["max_abs"] > tol or (argmax and not d["argmax_equal"]):
            bad.append({"key": k, **d})
    return {"compared": len(keys), "max_abs_diff": worst, "argmax_equal": same, "tol": tol, "failures": len(bad),
            "first_failures": bad[:10], "pass": not bad, "name": name}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mtstudio", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alone", type=int, default=0, help="score the first N requests alone (0: all)")
    ap.add_argument("--tol", type=float, default=0.0)
    ap.add_argument("--ref", default="")
    ap.add_argument("--ref-tol", type=float, default=1e-3)
    a = ap.parse_args(argv)
    rows = read_jsonl(a.inp)
    keys = [r["key"] for r in rows]
    if len(set(keys)) != len(keys):
        sys.exit(f"{a.inp}: keys repeat, so a key would not name one request")
    os.makedirs(os.path.join(a.out, "alone"), exist_ok=True)
    seq = score(a.mtstudio, a.spec, rows, os.path.join(a.out, "seq.out.jsonl"))
    rev = score(a.mtstudio, a.spec, rows[::-1], os.path.join(a.out, "rev.out.jsonl"))
    n = a.alone or len(rows)
    alone = {}
    for i, r in enumerate(rows[:n]):
        alone.update(score(a.mtstudio, a.spec, [r], os.path.join(a.out, "alone", f"r{i:05d}.out.jsonl")))
    lengths = [len(r["prompt_ids"]) + len(r["continuation_ids"]) for r in rows]
    rep = {"in": os.path.abspath(a.inp), "spec": os.path.abspath(a.spec), "mtstudio": os.path.abspath(a.mtstudio),
           "env": {k: v for k, v in os.environ.items() if k.startswith("MICROTORCH_")},
           "requests": len(rows), "lengths": {"min": min(lengths), "max": max(lengths)},
           "seq_vs_alone": gate("seq_vs_alone", seq, alone, keys[:n], a.tol),
           "seq_vs_reverse": gate("seq_vs_reverse", seq, rev, keys, a.tol)}
    if a.ref:
        ref = {r["key"]: r for r in read_jsonl(a.ref)}
        common = [k for k in keys if k in ref]
        rep["seq_vs_ref"] = {**gate("seq_vs_ref", seq, ref, common, a.ref_tol, argmax=False), "ref": os.path.abspath(a.ref)}
    ok = all(rep[g]["pass"] for g in ("seq_vs_alone", "seq_vs_reverse", "seq_vs_ref") if g in rep)
    rep["pass"] = ok
    with open(os.path.join(a.out, "report.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps({g: {x: rep[g][x] for x in ("compared", "max_abs_diff", "failures")}
                      for g in ("seq_vs_alone", "seq_vs_reverse", "seq_vs_ref") if g in rep} | {"pass": ok}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
