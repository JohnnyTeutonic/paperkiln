#!/usr/bin/env python3
"""Score the arXiv extractor on sizes and block structure, not only on the flavour verdicts.

    python papers/bench_score.py                       # every cached extraction of the 40-paper benchmark
    python papers/bench_score.py --out papers/llm_extract_receipts/2026-10-26_bench_scores.json

No API call: each paper's cached reply (papers/llm_extract_receipts/<id>.extraction.json, its raw
llm.output) is verified again by today's verifier against the cached LaTeX, then scored against
papers/bench_truth_structure.json (labels read off the LaTeX, never from an extraction) and
flavor_bench.TRUTH:

  flavours   norm, activation, positional: llm_extract.compare (correct / abstain / WRONG), as the
             2026-10-22 bench receipts score them.
  sizes      every size field of every truth config: correct, wrong, or abstain (the extractor
             asserts nothing there), each extracted config matched to the truth config it agrees with
             most (d_model, n_layers, n_heads; name as tie-break). An asserted size with no truth
             label (the paper states it where the labels have null, or a config the labels lack) is
             counted apart as unlabelled, not scored. The rule-based extractor (papers/fetch.py) is
             scored the same way on its d_model / n_heads / n_layers variants.
  graph      the block_ops graph, by data flow (graph_structure): each sublayer's norm placement
             (pre, post, sandwich, branch_post, none), parallel attention and FFN, a gated FFN, a rope
             node in the block / an absolute position table in the embed. Scored against the structure
             labels and flavor_bench.TRUTH's positional; abstain when the reply has no block graph or
             the sublayer cannot be found.

LOCAL_LLM_STACK.md section 11 item 4.2.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fetch  # noqa: E402
import flavor_bench  # noqa: E402
import llm_extract as L  # noqa: E402

TRUTH_FILE = HERE / "bench_truth_structure.json"
SIZE_FIELDS = ["d_model", "n_layers", "n_heads", "n_kv_heads", "d_ff", "vocab_size", "context_length"]
NORM_OPS = {"rmsnorm", "layernorm"}
ACT_OPS = {"gelu", "silu", "relu", "sigmoid"}
ATTN_OPS = {"attention", "linear_attention"}
ASSERTED = ("verified", "quote_only")


# --------------------------------------------------------------------------
# the block graph, by data flow

def graph_structure(block_ops: list, norm_ops: set = frozenset(), act_ops: set = frozenset()) -> dict | None:
    """The block's structure from its ops (inputs and output names, in execution order): per sublayer
    (the first attention node; the FFN's activation: an activation op, an expr standing for the
    activation, or a moe_ffn) the norm placement, plus parallel, gated, rope and absolute positions.
    norm_ops / act_ops: the names of new ops (expr) that stand for the norm / the activation. A norm
    reading a stream value (the block input, a residual add, a post norm) is post when its output is
    the skip input of a residual add or the block's last stream value, else pre; a norm reading
    anything else (a sublayer's output) is an output norm. None when the graph has no block."""
    nodes = [n for n in block_ops if n.get("subgraph") == "block"]
    embed = [n for n in block_ops if n.get("subgraph") == "embed"]
    if not nodes:
        return None
    writer, prod = {}, []
    for i, n in enumerate(nodes):
        prod.append([writer.get(v) for v in n.get("inputs") or []])
        for name in str(n.get("output") or "").split(","):  # split and rope write "q,k,v" / "q_r,k_r"
            writer[name.strip()] = i
    cons = {i: [] for i in range(len(nodes))}
    for j, ps in enumerate(prod):
        for p in ps:
            if p is not None:
                cons[p].append(j)

    def kind(i):
        n = nodes[i]
        op = n.get("op")
        if op in NORM_OPS or (op == "expr" and n.get("new_op") in norm_ops):
            return "norm"
        role = (n.get("role") or "").lower()
        if op in ATTN_OPS or (op == "unsupported" and "attention" in role and "cross" not in role):
            return "attn"
        if op in ACT_OPS or op == "moe_ffn" or (op == "expr" and n.get("new_op") in act_ops):
            return "act"
        return op

    stream = set()  # node indices whose output is a stream value (None, the block input, is one)

    def is_stream(p):
        return p is None or p in stream

    # One pass in execution order: a residual add reads a stream value; a norm on a stream value is
    # post when the stream goes on only through it (its output goes straight into an add, the next
    # residual's skip, or is the block's output, or nothing else reads its input), else pre; a norm
    # on anything else normalises a sublayer's output.
    residual, norm_role = set(), {}
    for i in range(len(nodes)):
        k = kind(i)
        if k == "add" and len(prod[i]) == 2 and any(is_stream(p) for p in prod[i]):
            residual.add(i)
            stream.add(i)
        elif k == "norm" and prod[i]:
            if is_stream(prod[i][0]):
                src_ = prod[i][0]
                post = (any(kind(j) == "add" for j in cons[i]) or (i == len(nodes) - 1 and not cons[i])
                        or (src_ is not None and cons[src_] == [i]))  # the stream goes on only through it
                norm_role[i] = "post" if post else "pre"
                if post:
                    stream.add(i)
            else:
                norm_role[i] = "out"

    def back(i, seen=None):
        """(a pre norm feeds the sublayer, the stream value it starts from)."""
        seen = seen or set()
        pre, roots = False, set()
        todo = list(prod[i])
        while todo:
            p = todo.pop()
            if p in seen:
                continue
            seen.add(p)
            if p is None or p in residual or norm_role.get(p) == "post":
                roots.add(p)
                continue
            if norm_role.get(p) == "pre":
                pre = True
                roots |= {q for q in prod[p][:1]}
                continue
            todo += prod[p]
        return pre, roots

    def forward(i):
        """(an output norm on the way, the residual add reached)."""
        out, todo, seen = False, list(cons[i]), set()
        while todo:
            j = todo.pop(0)
            if j in seen:
                continue
            seen.add(j)
            if j in residual:
                return out, j
            if norm_role.get(j) == "out":
                out = True
            todo += cons[j]
        return out, None

    def sublayer(i):
        if i is None:
            return None
        pre, roots = back(i)
        out, add = forward(i)
        post = add is not None and any(norm_role.get(j) == "post" for j in cons[add])
        place = ("sandwich" if pre and out else "pre" if pre else "branch_post" if out else
                 "post" if post else "none")
        return {"placement": place, "roots": roots, "add": add}

    a = next((i for i in range(len(nodes)) if kind(i) == "attn"), None)
    f = next((i for i in range(len(nodes)) if kind(i) == "act"), None)
    sa, sf = sublayer(a), sublayer(f)
    parallel = None
    if sa and sf:
        parallel = bool(sa["roots"] & sf["roots"]) and sa["add"] not in sf["roots"]
    gated = None
    if f is not None:
        n = nodes[f]
        if n.get("op") == "moe_ffn":
            acts = {(x.get("value") or "").lower() for x in n.get("attributes") or [] if x.get("name") == "activation"}
            gated = bool(acts & {"swiglu", "silu", "geglu"}) if acts else None
        elif n.get("op") == "expr" and len(n.get("inputs") or []) >= 2:
            gated = True
        else:
            gated = False
            todo, seen = list(cons[f]), set()
            while todo:
                j = todo.pop(0)
                if j in seen or j in residual:
                    continue
                seen.add(j)
                if nodes[j].get("op") == "mul":
                    gated = True
                    break
                todo += cons[j]
    return {"attn": sa["placement"] if sa else None, "ffn": sf["placement"] if sf else None,
            "parallel": parallel, "gated": gated,
            "rope": any(n.get("op") == "rope" for n in nodes),
            "abs_pos": any(n.get("op") == "pos_embedding" or "posit" in (n.get("role") or "").lower()
                           for n in embed)}


def truth_placement(v):
    if v is None:
        return None
    if isinstance(v, dict):
        return v
    if v in ("pre", "post", "sandwich"):
        return {"attn": v, "ffn": v}
    return None


def score_graph(g: dict | None, struct: dict, positional) -> list[dict]:
    rows = []

    def row(field, truth, got):
        if truth is None:
            return
        verdict = "abstain" if got is None else "correct" if got == truth else "WRONG"
        rows.append({"field": field, "truth": truth, "got": got, "score": verdict})

    g = g or {}
    place = truth_placement((struct.get("norm_placement") or {}).get("value"))
    if place:
        row("placement.attn", place["attn"], g.get("attn"))
        row("placement.ffn", place["ffn"], g.get("ffn"))
    row("parallel_block", (struct.get("parallel_block") or {}).get("value"), g.get("parallel"))
    row("ffn_gated", (struct.get("ffn_gated") or {}).get("value"), g.get("gated"))
    if positional == "rope":
        row("rope_node", True, g.get("rope") if g else None)
    elif positional in ("learned", "sinusoidal"):
        row("abs_position", True, g.get("abs_pos") if g else None)
    return rows


# --------------------------------------------------------------------------
# sizes

def _num(v):
    return None if v is None else float(v)


def match_configs(truth: list[dict], got: list[dict]) -> list[tuple[dict, dict | None]]:
    """Each truth config with the extracted config that agrees with it on most of d_model, n_layers,
    n_heads (at least one), the name breaking ties; each extracted config used once."""
    pairs, used = [], set()
    for t in truth:
        best, key = None, (0, 0)
        for k, c in enumerate(got):
            if k in used:
                continue
            agree = sum(1 for f in ("d_model", "n_layers", "n_heads")
                        if (t.get(f) or {}).get("value") is not None and c.get(f) is not None
                        and _num(t[f]["value"]) == c[f])
            same = int(str(t.get("name", "")).lower().replace(" ", "") == str(c.get("_name", "")).lower().replace(" ", ""))
            if agree and (agree, same) > key:
                best, key = k, (agree, same)
        if best is not None:
            used.add(best)
        pairs.append((t, got[best] if best is not None else None))
    return pairs, [c for k, c in enumerate(got) if k not in used]


def score_sizes(truth_cfgs: list[dict], got: list[dict]) -> dict:
    """got: extracted configs as {field: number asserted, "_name": name}."""
    tally = {"correct": 0, "WRONG": 0, "abstain": 0, "unlabelled": 0}
    rows = []
    pairs, spare = match_configs(truth_cfgs, got)
    for t, c in pairs:
        for f in SIZE_FIELDS:
            tv = (t.get(f) or {}).get("value")
            gv = (c or {}).get(f)
            if tv is None:
                if gv is not None:
                    tally["unlabelled"] += 1
                continue
            accept = {_num(v) for v in (t.get(f) or {}).get("accept") or [tv]}
            s = "abstain" if gv is None else "correct" if gv in accept else "WRONG"
            tally[s] += 1
            if s != "correct":
                rows.append({"config": t.get("name"), "field": f, "truth": tv, "got": gv, "score": s,
                             "matched": (c or {}).get("_name")})
    tally["unlabelled"] += sum(1 for c in spare for f in SIZE_FIELDS if c.get(f) is not None)
    return {"tally": tally, "misses": rows, "unmatched_extracted": [c.get("_name") for c in spare]}


def llm_configs(rep: dict) -> list[dict]:
    out = []
    for c in rep.get("configs") or []:
        d = {"_name": c.get("name")}
        for f in SIZE_FIELDS:
            r = c.get(f) or {}
            if r.get("check") in ASSERTED:
                v = L.parse_number(r.get("value") or "")
                if v is not None:
                    d[f] = float(v)
        out.append(d)
    return out


def rule_configs(arch_json: dict) -> list[dict]:
    out = [dict({k: float(v) for k, v in var.items() if k in SIZE_FIELDS}, _name=f"variant {i}")
           for i, var in enumerate(arch_json.get("variants") or [])]
    if not out:
        f = arch_json.get("fields") or {}
        d = {k: float(f[k]["value"]) for k in SIZE_FIELDS if k in f and f[k].get("value") is not None}
        if d:
            out.append(dict(d, _name="fields"))
    return out


# --------------------------------------------------------------------------
# driver

def score_paper(pid: str, truth: dict, ext: dict, tex: str) -> dict:
    sent = L.prepare_source(tex)
    output = ext["llm"]["output"]
    rep = L.verify(output, sent)
    fl = rep["flavours"]
    norm_ops = {fl["norm"]["value"][7:]} if (fl["norm"]["value"] or "").startswith("new_op:") else set()
    act_ops = {fl["activation"]["value"][7:]} if (fl["activation"]["value"] or "").startswith("new_op:") else set()
    g = graph_structure(output.get("block_ops") or [], norm_ops, act_ops)
    positional = flavor_bench.TRUTH.get(pid, {}).get("positional")
    rule = fetch.to_json(fetch.extract(pid, tex))
    return {"arxiv_id": pid,
            "flavours": L.compare(pid, tex, rep)["rows"],
            "sizes_llm": score_sizes(truth["configs"], llm_configs(rep)),
            "sizes_rule": score_sizes(truth["configs"], rule_configs(rule)),
            "graph": {"derived": g, "rows": score_graph(g, truth.get("structure") or {}, positional)}}


def tally_rows(rows, key="score"):
    out = {}
    for r in rows:
        out[r[key]] = out.get(r[key], 0) + 1
    return out


def summarise(papers: list[dict]) -> dict:
    flav = {"llm": {}, "rule": {}}
    for p in papers:
        for r in p["flavours"]:
            for who in ("llm", "rule"):
                flav[who][r[f"{who}_score"]] = flav[who].get(r[f"{who}_score"], 0) + 1
    sizes = {}
    for who in ("llm", "rule"):
        t = {"correct": 0, "WRONG": 0, "abstain": 0, "unlabelled": 0}
        for p in papers:
            for k, v in p[f"sizes_{who}"]["tally"].items():
                t[k] += v
        sizes[who] = t
    graph = {}
    for p in papers:
        for r in p["graph"]["rows"]:
            graph.setdefault(r["field"], {}).setdefault(r["score"], 0)
            graph[r["field"]][r["score"]] += 1
    graph_all = {}
    for v in graph.values():
        for k, n in v.items():
            graph_all[k] = graph_all.get(k, 0) + n
    return {"flavours": flav, "sizes": sizes, "graph": graph, "graph_all": graph_all}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--receipts", default=str(L.RECEIPTS))
    ap.add_argument("--out")
    ap.add_argument("ids", nargs="*")
    a = ap.parse_args(argv)
    truth = json.loads(TRUTH_FILE.read_text(encoding="utf-8"))["papers"]
    ids = a.ids or sorted(truth)
    papers, missing = [], []
    for pid in ids:
        p = pathlib.Path(a.receipts) / f"{pid}.extraction.json"
        if not p.exists():
            missing.append(pid)
            continue
        ext = json.loads(p.read_text(encoding="utf-8"))
        papers.append(score_paper(pid, truth[pid], ext, L.get_tex(pid, offline=True)))
    s = summarise(papers)
    print(f"{len(papers)} papers scored ({len(missing)} without a cached extraction: {missing})")
    print(f"flavours  llm {s['flavours']['llm']}  rule {s['flavours']['rule']}")
    print(f"sizes     llm {s['sizes']['llm']}  rule {s['sizes']['rule']}")
    print(f"graph     {s['graph_all']}")
    for f, t in sorted(s["graph"].items()):
        print(f"  {f:16} {t}")
    for p in papers:
        bad = [r for r in p["graph"]["rows"] if r["score"] != "correct"]
        wrong = [m for m in p["sizes_llm"]["misses"] if m["score"] == "WRONG"]
        if bad or wrong:
            print(f"  {p['arxiv_id']}: graph {[(r['field'], r['truth'], r['got']) for r in bad]} "
                  f"sizes wrong {[(m['config'], m['field'], m['truth'], m['got']) for m in wrong]}")
    if a.out:
        L.write_json(a.out, {"truth": str(TRUTH_FILE.relative_to(HERE.parent)), "summary": s, "papers": papers,
                             "missing": missing, "scorer": "papers/bench_score.py"})
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
