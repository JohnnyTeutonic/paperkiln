#!/usr/bin/env python3
"""A34's floor references (the lane's cue_selective/a34/REGISTRATION.md), jobs on the pod. Claude Opus 5.5, 15 Nov 2026.

The floor holds each training twin's own-context per-token log-likelihood, lp(yA|A)/n_A and lp(yB|B)/n_B, at or above its value
under plain SFT at the same step, same seed and same SFT batches: the reference run R (D0r, P_1r, P_50r) is that plain run, trained
on this build and snapshotted every 128 steps. Its weights after 0 (the start, A's SFT1500), 128, ..., 896 and 1,024 updates are
scored with `mtstudio score` on a34_own_req.jsonl (two rows per training twin in data.twin_examples' order: A's context with A's
completion, then B's with B's, each completion followed by <|endoftext|>, as the trainer reads them), and the per-token values become
the knots of floor_ref.jsonl, which the trainer interpolates linearly at the weights each step reads.

  start  --chat C --bin B --out DIR     the start weights (upload tc_sft_b1_r1.safetensors): DIR/start_own.jsonl, receipt.json
  build  --job R --chat C --bin B --out DIR    R's rung_<k> snapshots (k in REF_STEPS) and its final weights, with ref_start's
                                               knot 0: DIR/floor_ref.jsonl and receipt.json

Each scoring is validated (one record per request, in order, finite, whole, not over the context, A's and B's token counts the
twin file's), written to a temporary file and renamed, so a restarted job reuses finished knots. Exits non-zero on any failure:
the floor runs that depend on it then never start.
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys

REF_STEPS = (128, 256, 384, 512, 640, 768, 896)
FINAL = 1024
SCORE_TIMEOUT_S = 1200


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def own_values(req, recs):
    """[(a, b, n_a, n_b)] per group from the own-row requests and their score records; raises ValueError on any departure."""
    if len(recs) != len(req) or len(req) % 2:
        raise ValueError(f"{len(recs)} records for {len(req)} requests")
    out = []
    for g in range(len(req) // 2):
        qa, qb, ra, rb = req[2 * g], req[2 * g + 1], recs[2 * g], recs[2 * g + 1]
        if (qa.get("id"), qa.get("row"), qb.get("id"), qb.get("row")) != (f"g{g}", "AA", f"g{g}", "BB"):
            raise ValueError(f"request {2 * g}: not group {g}'s AA and BB rows")
        vals = []
        for i, r in ((2 * g, ra), (2 * g + 1, rb)):
            if r.get("i") != i or r.get("over_ctx") is not False:
                raise ValueError(f"record {i}: out of order or over the context")
            lp = r.get("logprobs")
            if not finite(r.get("sum")) or not isinstance(lp, list) or not lp or not all(finite(x) for x in lp):
                raise ValueError(f"record {i}: not a whole, finite score")
            if abs(r["sum"] - sum(lp)) > 1e-4 * (1.0 + abs(r["sum"])):
                raise ValueError(f"record {i}: its sum is not its log-probabilities' sum")
            vals.append((r["sum"] / len(lp), len(lp)))
        out.append((vals[0][0], vals[1][0], vals[0][1], vals[1][1]))
    return out


def scoring_spec(spec_src, name, m):
    spec = json.load(open(spec_src))
    spec["name"] = name
    spec["out_dir"] = m
    for k in ("twin_examples", "twin_floor_ref"):
        spec.get("data", {}).pop(k, None)
    spec.get("train", {}).pop("twin_loss", None)
    return spec


def score(binary, weights, spec_src, name, req, out_scores, work):
    """Score `weights` (a model with the spec at spec_src) on the own-row requests into out_scores (renamed into place). A scores
    file is reused only when its binding (scores_<k>.binding.json: the sha256 of the weights, requests, binary and effective
    scoring spec it was made from, and of itself) matches the current inputs; otherwise it is scored again (A34 launch review,
    15 Nov 2026: a cached knot must never be bound to weights it was not made from)."""
    m = os.path.join(work, f"m_{name}")
    os.makedirs(m, exist_ok=True)
    spec = scoring_spec(spec_src, name, m)
    want = {"weights_sha256": sha(weights), "requests_sha256": sha(req), "binary_sha256": sha(binary),
            "spec_sha256": hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()}
    bind = out_scores[:-len(".jsonl")] + ".binding.json"
    if os.path.exists(out_scores) and os.path.exists(bind):
        have = json.load(open(bind))
        if have == {**want, "scores_sha256": sha(out_scores)}:
            return want["weights_sha256"]
    for p in (out_scores, bind):
        if os.path.exists(p):
            os.remove(p)
    json.dump(spec, open(os.path.join(m, "spec.json"), "w"), indent=1)
    link = os.path.join(m, f"{name}.safetensors")
    if os.path.lexists(link):
        os.remove(link)
    os.symlink(weights, link)
    tmp = out_scores + ".tmp"
    r = subprocess.run([binary, "score", m, "--in", req, "--out", tmp, "--batch-tokens", "8192"], timeout=SCORE_TIMEOUT_S)
    if r.returncode != 0:
        raise RuntimeError(f"score of {name} exited {r.returncode}")
    os.replace(tmp, out_scores)
    write_json(bind, {**want, "scores_sha256": sha(out_scores)})
    return want["weights_sha256"]


def write_json(p, obj):
    with open(p + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(p + ".tmp", p)


def start(a):
    up = os.path.join(a.chat, "upload")
    req_p = os.path.join(up, "a34_own_req.jsonl")
    req = read_jsonl(req_p)
    w = os.path.join(up, "tc_sft_b1_r1.safetensors")
    os.makedirs(a.out, exist_ok=True)
    sc = os.path.join(a.out, "scores_0.jsonl")
    score(a.bin, w, os.path.join(up, "a34_gate_spec.json"), "ref_start", req_p, sc, a.out)
    vals = own_values(req, read_jsonl(sc))
    with open(os.path.join(a.out, "start_own.jsonl.tmp"), "w") as f:
        for g, (va, vb, na, nb) in enumerate(vals):
            f.write(json.dumps({"g": g, "a": va, "b": vb, "n_a": na, "n_b": nb}) + "\n")
    os.replace(os.path.join(a.out, "start_own.jsonl.tmp"), os.path.join(a.out, "start_own.jsonl"))
    write_json(os.path.join(a.out, "receipt.json"),
               {"producer": "a34_ref.py start", "binding": json.load(open(os.path.join(a.out, "scores_0.binding.json"))),
                "groups": len(vals), "start_own_sha256": sha(os.path.join(a.out, "start_own.jsonl"))})
    print(f"start: {len(vals)} groups, mean own {sum(v[0] + v[1] for v in vals) / (2 * len(vals)):.4f}")
    return 0


def build(a):
    up = os.path.join(a.chat, "upload")
    req_p = os.path.join(up, "a34_own_req.jsonl")
    req = read_jsonl(req_p)
    os.makedirs(a.out, exist_ok=True)
    st = os.path.join(a.chat, "ref_start")
    # knot 0 is ref_start's: its binding must be the current start weights, requests and binary, and its scores the bound ones.
    # It was scored under the gate spec's architecture, vocabulary and context (the runs' own, A31's S71 spec); its spec hash is
    # not compared with the run's, because the trainer's preflight compares knot 0's "start" values with its own forward under
    # each run's spec, for every group, before step 1.
    sb = json.load(open(os.path.join(st, "scores_0.binding.json")))
    w0 = os.path.join(up, "tc_sft_b1_r1.safetensors")
    if (sb["weights_sha256"], sb["requests_sha256"], sb["binary_sha256"], sb["scores_sha256"]) != \
            (sha(w0), sha(req_p), sha(a.bin), sha(os.path.join(st, "scores_0.jsonl"))):
        raise SystemExit("ref_start's scores are not bound to the current start weights, requests and binary")
    knots = {0: own_values(req, read_jsonl(os.path.join(st, "scores_0.jsonl")))}
    weights = {0: sb["weights_sha256"]}
    spec_src = os.path.join(a.chat, f"{a.job}_spec.json")
    seed = json.load(open(spec_src))["train"]["seed"]
    for k in REF_STEPS + (FINAL,):
        w = (os.path.join(a.chat, a.job, f"rung_{k}", "model.safetensors") if k != FINAL
             else os.path.join(a.chat, a.job, f"{a.job}.safetensors"))
        if k != FINAL:   # the snapshot is the weights after k updates: its state.txt must say so
            if int(open(os.path.join(a.chat, a.job, f"rung_{k}", "state.txt")).readline().strip()) != k:
                raise SystemExit(f"rung_{k} of {a.job} is not at step {k}")
        sc = os.path.join(a.out, f"scores_{k}.jsonl")
        weights[k] = score(a.bin, w, spec_src, f"{a.job}_k{k}", req_p, sc, a.out)
        knots[k] = own_values(req, read_jsonl(sc))
    steps = sorted(knots)
    G = len(req) // 2
    for k in steps:
        if [v[2:] for v in knots[k]] != [v[2:] for v in knots[0]]:
            raise SystemExit(f"knot {k}: reply token counts differ from the start's")
    ref = os.path.join(a.out, "floor_ref.jsonl")
    with open(ref + ".tmp", "w") as f:
        for g in range(G):
            na, nb = knots[0][g][2], knots[0][g][3]
            f.write(json.dumps({"g": g, "seed": seed, "n_a": na, "n_b": nb, "start": [knots[0][g][0], knots[0][g][1]], "steps": steps,
                                "a": [knots[k][g][0] for k in steps], "b": [knots[k][g][1] for k in steps]}) + "\n")
    os.replace(ref + ".tmp", ref)
    means = {str(k): sum(v[0] + v[1] for v in knots[k]) / (2 * G) for k in steps}
    write_json(os.path.join(a.out, "receipt.json"),
               {"producer": "a34_ref.py build", "job": a.job, "seed": seed, "steps": steps, "weights_sha256": {str(k): weights[k] for k in steps},
                "requests_sha256": sha(req_p), "binary_sha256": sha(a.bin), "spec_sha256": sha(spec_src), "groups": G,
                "scores_sha256": {str(k): sha(os.path.join(a.out if k else st, f"scores_{k}.jsonl")) for k in steps},
                "floor_ref_sha256": sha(ref), "mean_own_per_token": means})
    print(f"{a.job}: floor_ref.jsonl, {G} groups, knots {steps}; mean own " + " ".join(f"{k}:{means[str(k)]:.3f}" for k in steps))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("start", "build"))
    ap.add_argument("--job")
    ap.add_argument("--chat", default="/content/chat")
    ap.add_argument("--bin", default="/content/pk/build_cuda/mtstudio")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "build" and not a.job:
        ap.error("build needs --job")
    return start(a) if a.cmd == "start" else build(a)


if __name__ == "__main__":
    sys.exit(main())
