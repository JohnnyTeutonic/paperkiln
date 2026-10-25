#!/usr/bin/env python3
"""Depth-attention weights of a llama_attnres chat model on Carry-5 (experiments/tiny_chat_2026-10/PREREGISTRATION.md
amendment 5, A5.7; descriptive): where on the depth axis each sublayer reads at the positions that predict the
answer, against the other assistant positions, and how close to uniform the routing is (the collapse check).

    python tools/carry5_attnres_trace.py run --gguf SFT.gguf --out DIR
        [--set corpus/eval/carry5_v1.jsonl] [--families recall,correction,topic] [--limit N] [--every N]

The model runs in keramos's numpy interpreter (spec/tools/ir_numpy.py) over its F32 GGUF and embedded IR, one
node at a time, so every depth_attend node's weights are read where the interpreter computes them: for each row
t, a = softmax_i(sum_j rmsnorm(s_i)[t, j] w[j]) over the node's S sources. The tool recomputes the node's
mixture from those weights and requires it to equal the interpreter's output (relative 1e-5), so the weights are
the model's. Mixtures: L{i}.attn and L{i}.ffn (the inputs of layer i's sublayers; S = 2i + 1 and 2i + 2) and
head (S = 2L + 1). Source 0 is the token embedding, source 2i + 1 layer i's attention output, 2i + 2 its FFN
output.

The rendering and the positions are tools/carry5_depth.py's (carry5_eval's scored sequence for the gold
candidate, gold history, teacher-forced): answer positions predict a token of the gold value in the target
reply; other assistant positions predict any other token of an assistant turn. Statistics, pooled over
positions and items:
  per mixture: the mean weight vector at answer positions and at other assistant positions, their difference,
               and the normalised entropy H(a) / log S at answer, other assistant and all positions;
  headline:    the embedding's weight (source 0) and the early sources' weight (sources 0-2: the embedding
               and layer 0's two outputs) at answer minus other assistant positions, averaged over the
               mixtures that can see them; the mean normalised entropy over mixtures and the number of mixtures
               above 0.95 (near-uniform: the plain model, section 2 of CROSSLAYER_LITCHECK.md);
  by distance: the embedding's and the early sources' answer-position weight per fact distance (1-4).
Progress is checkpointed atomically (DIR/state.json every CHUNK items), so a rerun continues.
Writes DIR/attnres_trace.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "spec", "tools"))  # keramos
import carry5_depth as cd  # noqa: E402
import carry5_eval as ce  # noqa: E402
import ir_check  # noqa: E402
import ir_numpy  # noqa: E402

CHUNK = 50
EARLY = 3          # sources 0-2: the embedding and layer 0's attention and FFN outputs
NEAR_UNIFORM = 0.95


def load_model(gguf):
    meta, tensors, data_start, raw, _ = ir_check.read_gguf(gguf)
    W = {}
    for name, dims, typ, off in tensors:
        if typ != 0:
            raise SystemExit(f"{gguf}: {name} is not F32 (the trace reads the F32 export)")
        n = int(np.prod(dims))
        W[name] = np.frombuffer(raw, dtype=np.float32, count=n, offset=data_start + off).reshape(list(reversed(dims)))
    doc = json.loads(meta["paperkiln.ir"][1])
    if doc.get("family_hint") != "llama_attnres":
        raise SystemExit(f"{gguf}: family {doc.get('family_hint')!r}, not llama_attnres")
    return doc, W


def weights_of(nd, env, W, i, doc):
    """(sources, a [T, S]) of a depth_attend node, by the interpreter's own arithmetic (ir_numpy.run_nodes)."""
    ins = nd["in"] if isinstance(nd["in"], list) else [nd["in"]]
    src = list(env[ins[0]]) + [env[n] for n in ins[1:]]
    if len(src) == 1:
        return src, np.ones((src[0].shape[0], 1), dtype=np.float32)
    name = nd["weight"]
    if i is not None:  # "{i}": the stored layer (weight_index), or the application under tensor_index "application"
        idx = i if nd.get("tensor_index") == "application" else ir_numpy.weight_index(doc, i)
        name = name.replace("{i}", str(idx))
    wq, eps = W[name], np.float32(nd["eps"])
    logits = []
    for s in src:
        r = np.float32(1) / np.sqrt((s * s).sum(axis=1, keepdims=True) / np.float32(s.shape[1]) + eps)
        logits.append(((s * r) * wq).sum(axis=1))
    lg = np.stack(logits, axis=1)
    e = np.exp(lg - lg.max(axis=1, keepdims=True))
    return src, e / e.sum(axis=1, keepdims=True)


def trace(doc, W, ids):
    """{mixture: a [T, S]} for one sequence, every depth_attend node of the blocks and the head."""
    env = {"tokens": np.array(ids), "pos": np.arange(len(ids))}
    ir_numpy.run_nodes(doc["embed"]["nodes"], env, W)
    out = {}

    def run(nodes, i, tag):
        k = 0
        for nd in nodes:
            if nd["op"] == "depth_attend":
                src, a = weights_of(nd, env, W, i, doc)
                ir_numpy.run_nodes([nd], env, W, i, wi=None if i is None else ir_numpy.weight_index(doc, i))
                y = env[nd["out"]]
                mix = sum(s * a[:, j:j + 1] for j, s in enumerate(src))
                if np.abs(mix - y).max() > 1e-5 * max(1.0, float(np.abs(y).max())):
                    raise SystemExit(f"{tag}: the recomputed mixture differs from the interpreter's output")
                out[f"{tag}.{('attn', 'ffn')[k]}" if tag != "head" else "head"] = a
                k += 1
                if tag == "head":
                    return
            else:
                ir_numpy.run_nodes([nd], env, W, i, wi=None if i is None else ir_numpy.weight_index(doc, i))
    for i in range(doc["block"]["repeat"]):
        run(doc["block"]["nodes"], i, f"L{i}")
    run(doc["head"]["nodes"], None, "head")
    return out


def entropy_norm(a):
    """H(a) / log S per row (S >= 2)."""
    S = a.shape[1]
    p = np.clip(a.astype(np.float64), 1e-300, 1.0)
    return -(a * np.log(p)).sum(axis=1) / math.log(S)


class Acc:
    """Pooled sums per mixture (JSON-serialisable, for the checkpoint)."""

    def __init__(self, d=None):
        self.d = d or {"mix": {}, "dist": {}, "done": [], "positions": {"answer": 0, "other": 0, "all": 0}}

    def add(self, it, tr, rend):
        ans, oth = rend["answer_pos"], rend["other_pos"]
        for m, a in tr.items():
            S = a.shape[1]
            r = self.d["mix"].setdefault(m, {"S": S, "ans_sum": [0.0] * S, "ans_n": 0, "oth_sum": [0.0] * S, "oth_n": 0,
                                             "ent_ans": 0.0, "ent_oth": 0.0, "ent_all": 0.0, "all_n": 0})
            if S != r["S"]:
                raise SystemExit(f"{m}: {S} sources, {r['S']} before")
            r["ans_sum"] = (np.array(r["ans_sum"]) + a[ans].sum(axis=0)).tolist()
            r["oth_sum"] = (np.array(r["oth_sum"]) + a[oth].sum(axis=0)).tolist()
            r["ans_n"] += len(ans)
            r["oth_n"] += len(oth)
            r["all_n"] += a.shape[0]
            if S >= 2:
                e = entropy_norm(a)
                r["ent_ans"] += float(e[ans].sum())
                r["ent_oth"] += float(e[oth].sum())
                r["ent_all"] += float(e.sum())
            dk = str(it["distance"])
            dd = self.d["dist"].setdefault(dk, {}).setdefault(m, {"emb": 0.0, "early": 0.0, "n": 0})
            dd["emb"] += float(a[ans, 0].sum())
            dd["early"] += float(a[ans, :EARLY].sum())
            dd["n"] += len(ans)
        self.d["positions"]["answer"] += len(ans)
        self.d["positions"]["other"] += len(oth)
        self.d["positions"]["all"] += len(rend["prompt_ids"]) + len(rend["continuation_ids"]) - 1
        self.d["done"].append(it["id"])


def summarise(d):
    mix, heads = {}, []
    for m, r in d["mix"].items():
        S = r["S"]
        am = [x / r["ans_n"] for x in r["ans_sum"]] if r["ans_n"] else None
        om = [x / r["oth_n"] for x in r["oth_sum"]] if r["oth_n"] else None
        row = {"S": S, "uniform": 1 / S, "answer_mean": am, "other_mean": om,
               "answer_minus_other": [a - b for a, b in zip(am, om)] if am and om else None,
               "entropy_norm": {"answer": r["ent_ans"] / r["ans_n"] if S >= 2 and r["ans_n"] else None,
                                "other": r["ent_oth"] / r["oth_n"] if S >= 2 and r["oth_n"] else None,
                                "all": r["ent_all"] / r["all_n"] if S >= 2 and r["all_n"] else None}}
        mix[m] = row
        heads.append((m, row))
    seen = [(m, r) for m, r in heads if r["S"] >= 2 and r["answer_minus_other"]]
    early = [(m, r) for m, r in seen if r["S"] > EARLY]
    ents = [r["entropy_norm"]["all"] for _, r in seen if r["entropy_norm"]["all"] is not None]
    head = {
        "embedding_answer_minus_other": sum(r["answer_minus_other"][0] for _, r in seen) / len(seen) if seen else None,
        "early_answer_minus_other": (sum(sum(r["answer_minus_other"][:EARLY]) for _, r in early) / len(early)
                                     if early else None),
        "embedding_answer_mean": sum(r["answer_mean"][0] for _, r in seen) / len(seen) if seen else None,
        "embedding_uniform_mean": sum(1 / r["S"] for _, r in seen) / len(seen) if seen else None,
        "entropy_norm_mean": sum(ents) / len(ents) if ents else None,
        "mixtures_near_uniform": sum(1 for e in ents if e > NEAR_UNIFORM), "mixtures": len(ents),
        "head_embedding_answer_minus_other": (mix.get("head") or {}).get("answer_minus_other", [None])[0]}
    dist = {}
    for k, ms in sorted(d["dist"].items()):
        rows = [v for m, v in ms.items() if v["n"] and d["mix"][m]["S"] >= 2]
        n = sum(v["n"] for v in rows)
        dist[k] = {"answer_positions": n // max(1, len(rows)),
                   "embedding_mean_over_mixtures": sum(v["emb"] / v["n"] for v in rows) / len(rows) if rows else None,
                   "early_mean_over_mixtures": sum(v["early"] / v["n"] for v in rows) / len(rows) if rows else None}
    return {"headline": head, "mixtures": mix, "by_distance": dist, "positions": d["positions"], "items": len(d["done"])}


def write_atomic(p, obj):
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f)
    os.replace(tmp, p)


def cmd_run(a):
    os.makedirs(a.out, exist_ok=True)
    doc, W = load_model(a.gguf)
    tok = ce.Tok(a.gguf)
    items = ce.load_items(a)
    st_p = os.path.join(a.out, "state.json")
    acc = Acc(json.load(open(st_p, encoding="utf-8")) if os.path.exists(st_p) else None)
    done = set(acc.d["done"])
    todo = [it for it in items if it["id"] not in done]
    for k, it in enumerate(todo, 1):
        rend = cd.render(tok, it, "gold")
        ids = (rend["prompt_ids"] + rend["continuation_ids"])[:-1]
        if len(ids) > doc["model"].get("ctx", 10 ** 9):
            raise SystemExit(f"{it['id']}: {len(ids)} positions exceed the context")
        acc.add(it, trace(doc, W, ids), rend)
        if k % CHUNK == 0:
            write_atomic(st_p, acc.d)
            print(f"traced {len(acc.d['done'])}/{len(items)}", flush=True)
    write_atomic(st_p, acc.d)
    with open(a.set, "rb") as f:
        set_sha = hashlib.sha256(f.read()).hexdigest()
    with open(a.gguf, "rb") as f:
        g_sha = hashlib.sha256(f.read()).hexdigest()
    out = {"tool": "tools/carry5_attnres_trace.py", "label": "DESCRIPTIVE (PREREGISTRATION.md amendment 5, A5.7)",
           "gguf": a.gguf, "gguf_sha256": g_sha, "set": os.path.relpath(a.set, ROOT).replace(os.sep, "/"),
           "set_sha256": set_sha, "families": a.families.split(","), "limit": a.limit, "every": a.every,
           "mode": "gold history, teacher-forced", "summary": summarise(acc.d)}
    write_atomic(os.path.join(a.out, "attnres_trace.json"), out)
    os.remove(st_p)  # the final output replaces the checkpoint
    print(json.dumps(out["summary"]["headline"]))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--gguf", required=True, help="the SFT model's F32 GGUF (IR and vocabulary embedded)")
    p.add_argument("--out", required=True)
    p.add_argument("--set", default=ce.SET)
    p.add_argument("--families", default=",".join(cd.CONTEXT_FAMILIES))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--every", type=int, default=1)
    a = ap.parse_args(argv)
    return cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())
