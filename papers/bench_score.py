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
             most (d_model, n_layers, n_heads; name as tie-break, or the name alone when the paper
             states none of the three, as Mamba's headline configs). An asserted size with no truth
             label (the paper states it where the labels have null, or a config the labels lack) is
             counted apart as unlabelled, not scored. The rule-based extractor (papers/fetch.py) is
             scored the same way on its d_model / n_heads / n_layers variants.
  graph      the block_ops graph, by data flow (graph_structure): each sublayer's norm placement
             (pre, post, sandwich, branch_post, none), parallel attention and FFN, a gated FFN, a rope
             node in the block / an absolute position table in the embed. Scored against the structure
             labels and flavor_bench.TRUTH's positional; abstain when the reply has no block graph or
             the sublayer cannot be found.
  new fields the 26 Oct prompt's fields, against each paper's new_fields labels (27 Oct; a reply made
             before them reads them as not_stated):
               placement_field.attn / .ffn  the flavours' per-sublayer placement (norm_placement_attn
                   / _ffn, else norm_placement), against the structure label;
               qk_norm, ssm  against per_head / full / unspecified / mamba1 / s4, or "absent" (the paper
                   never mentions one): then nothing asserted (or none) is correct, any assertion WRONG
                   (for ssm also an asserted d_state, d_conv, expand or dt_rank);
               positional_ext, rel_buckets, rel_max_distance  the 11 papers flavor_bench.TRUTH leaves
                   unlabelled (relative, t5_bias, learned; T5's bucket count and distance);
               new_op_inputs  each scored label op (an elementwise or row op outside the standard set),
                   matched to the reply's new op of the same category (activation, norm, embed,
                   final_logits, attention_logits, bottleneck_scale: by its name, what it replaces and
                   its block node's role): abstain when none is found, WRONG when the match is not an
                   expr or its inputs' roles differ from the label's (one input; a learned scalar or a
                   gate the label lacks; learned vectors only as a norm's gain and bias). An input times
                   constants may instead be the IR's affine node (embed or head); an op on the attention
                   logits is a needs_new_primitive (the attention node has no slot for it);
               learned_inputs  per paper, the learned or gate inputs the reply's new ops assert beyond
                   the labels' (with any verified init): correct when there are none.
  --baseline OLD.json prints every field's tallies against an earlier --out file.

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
    n_heads (at least one, or else the same name), the name breaking ties; each extracted config used once."""
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
            if (agree or same) and (agree, same) > key:  # a config stating no size matches by its name (Mamba's)
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
# the 26 Oct prompt's fields

PLACEMENTS = {"pre", "post", "sandwich", "branch_post", "none"}


def _asserted(r: dict | None):
    """A flavour's value when the verifier lets it be asserted, else None."""
    return (r or {}).get("value") if (r or {}).get("check") in ASSERTED else None


def op_category(text: str) -> str | None:
    t = text.lower()
    if "posit" in t:
        return None
    if ("attn" in t or "attention" in t) and any(w in t for w in ("logit", "cap", "score")):
        return "attention_logits"
    if "logit" in t:
        return "final_logits"
    if "bottleneck" in t or "latent" in t:
        return "bottleneck_scale"
    if "embed" in t:
        return "embed"
    if "norm" in t:
        return "norm"
    if "activation" in t or "relu" in t or "ffn" in t or "feed" in t:
        return "activation"
    return None


def _roles_ok(op: dict, category: str, truth_roles: list[str]) -> tuple[bool, list]:
    got = [i.get("role") for i in op.get("inputs") or []]
    want_learned = sorted(r for r in truth_roles if r in ("gate", "learned_scalar", "learned_vector"))
    got_learned = sorted(r for r in got if r in ("gate", "learned_scalar", "learned_vector")
                         and not (category == "norm" and r == "learned_vector" and "learned_vector" not in want_learned))
    return got.count("input") == truth_roles.count("input") and got_learned == want_learned, got


def score_new_fields(pid: str, rep: dict, output: dict, truth: dict) -> list[dict]:
    rows = []
    nf = truth.get("new_fields") or {}
    fl = rep["flavours"]

    def row(field, t, got, verdict=None, **extra):
        v = verdict or ("abstain" if got is None else "correct" if got == t else "WRONG")
        rows.append(dict({"field": field, "truth": t, "got": got, "score": v}, **extra))

    # per-sublayer placement from the flavours
    place = truth_placement(((truth.get("structure") or {}).get("norm_placement") or {}).get("value"))
    if place:
        whole = _asserted(fl.get("norm_placement"))
        for sub in ("attn", "ffn"):
            got = _asserted(fl.get(f"norm_placement_{sub}"))
            if got not in PLACEMENTS:
                got = whole if whole in PLACEMENTS else None
            row(f"placement_field.{sub}", place[sub], got)

    # qk_norm and ssm: a label, or "absent" (nothing may be asserted)
    for field, sizes in (("qk_norm", ()), ("ssm", ("d_state", "d_conv", "expand", "dt_rank"))):
        t = (nf.get(field) or {}).get("value")
        if t is None:
            continue
        got = _asserted(fl.get(field))
        asserted_sizes = {s: _asserted(fl.get(s)) for s in sizes if _asserted(fl.get(s)) is not None}
        if t == "absent":
            bad = (got not in (None, "none")) or bool(asserted_sizes)
            row(field, t, got if not asserted_sizes else {"value": got, **asserted_sizes},
                "WRONG" if bad else "correct")
        else:
            accept = nf[field].get("accept") or [t]  # a label that reads two ways (OLMo 2's q/k norm)
            row(field, t, got, None if got is None else "correct" if got in accept else "WRONG")
            for s in sizes:
                ts = (nf[field].get(s) or {}).get("value")
                if ts is not None:
                    gs = asserted_sizes.get(s)
                    row(f"ssm.{s}", ts, gs, None if gs is None else "correct" if str(gs) == str(ts) or
                        L.parse_number(str(gs)) == float(ts) else "WRONG")

    # positional for the papers the flavour truth leaves unlabelled
    pos = nf.get("positional")
    if pos and pos.get("value") in ("relative", "t5_bias", "learned", "sinusoidal", "rope", "alibi"):
        got = _asserted(fl.get("positional"))
        if got is not None and str(got).startswith("other:"):
            got = None  # outside the lattice: never built, an abstention (as llm_extract.compare)
        row("positional_ext", pos["value"], got)
        for f in ("rel_buckets", "rel_max_distance"):
            tv = (pos.get(f) or {}).get("value")
            if tv is not None:
                gv = _asserted(fl.get(f))
                gv = None if gv is None else L.parse_number(str(gv))
                row(f, float(tv), gv)

    # new ops: input roles, matched by category
    ops = rep.get("new_ops") or []
    role_of = {}
    for n in output.get("block_ops") or []:
        if n.get("new_op"):
            role_of.setdefault(n["new_op"], " ".join(filter(None, [n.get("role"), n.get("subgraph")])))
    cat = {o["name"]: op_category(" ".join([o.get("name", ""), o.get("replaces", ""), role_of.get(o["name"], "")]))
           for o in ops}
    used = set()
    for t in nf.get("new_ops") or []:
        if not t.get("scored"):
            continue
        cands = [o for o in ops if cat[o["name"]] == t["category"] and o["name"] not in used]
        cands.sort(key=lambda o: (o.get("kind") != "expr", o.get("check") != "verified"))
        truth_roles = [i["role"] for i in t["inputs"] if i["role"] != "other"]
        if not cands:
            # An input times constants is the IR's own affine node, not a new op: in the embed for an
            # embedding scale, in the head for a logit scale.
            sub = {"embed": "embed", "final_logits": "head"}.get(t["category"])
            aff = [n for n in output.get("block_ops") or [] if n.get("op") == "affine" and n.get("subgraph") == sub]
            if sub and truth_roles == ["input"] and aff:
                row("new_op_inputs", truth_roles, ["input"], "correct", op=t["name"], matched="affine node",
                    attributes=[(x.get("name"), x.get("value")) for x in aff[0].get("attributes") or []])
            else:
                row("new_op_inputs", truth_roles, None, op=t["name"])
            continue
        o = cands[0]
        used.add(o["name"])
        ok, got = _roles_ok(o, t["category"], truth_roles)
        good = o.get("kind") == "expr" and o.get("check") == "verified" and ok
        if t["category"] == "attention_logits" and o.get("kind") == "needs_new_primitive":
            # The attention node has no slot for an op on its [heads, T, T] logits, so an op there is a
            # new primitive by the prompt's rule, not an expr.
            good, got = o.get("check") == "verified", ["input"]
        # A reply made before roles existed (22 Oct) gives none: the field reads as not stated.
        unstated = o.get("kind") == "expr" and not any(got)
        row("new_op_inputs", truth_roles, got, "abstain" if unstated else "correct" if good else "WRONG",
            op=t["name"], matched=o["name"],
            kind=o.get("kind"), check=o.get("check"), expr=(o.get("expr_check") or {}).get("canonical"))

    # learned or gate inputs no label has
    want = {i["role"] for t in nf.get("new_ops") or [] for i in t["inputs"]}
    extra = []
    for o in ops:
        for i in o.get("inputs") or []:
            r = i.get("role")
            if r in ("gate", "learned_scalar", "learned_vector") and r not in want \
                    and not (cat[o["name"]] == "norm" and r == "learned_vector"):
                extra.append({"op": o["name"], "input": i.get("name"), "role": r,
                              "init": (i.get("init") or {}).get("value") if i.get("init_check") == "verified" else None})
    row("learned_inputs", [], extra, "WRONG" if extra else "correct")
    return rows


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
            "graph": {"derived": g, "rows": score_graph(g, truth.get("structure") or {}, positional)},
            "new_fields": score_new_fields(pid, rep, output, truth)}


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
    new = {}
    for p in papers:
        for r in p.get("new_fields") or []:
            new.setdefault(r["field"], {}).setdefault(r["score"], 0)
            new[r["field"]][r["score"]] += 1
    return {"flavours": flav, "sizes": sizes, "graph": graph, "graph_all": graph_all, "new_fields": new}


def print_baseline(old: dict, new: dict) -> None:
    """Each field's tallies, the baseline's beside this run's."""
    def fmt(t):
        return " / ".join(f"{t.get(k, 0)}" for k in ("correct", "WRONG", "abstain")) + (
            f" (+{t['unlabelled']} unlabelled)" if t.get("unlabelled") else "")
    print()
    print("against the baseline (correct / WRONG / abstain)")
    print(f"  {'flavours (llm)':24} {fmt(old['flavours']['llm']):24} -> {fmt(new['flavours']['llm'])}")
    print(f"  {'sizes (llm)':24} {fmt(old['sizes']['llm']):24} -> {fmt(new['sizes']['llm'])}")
    print(f"  {'graph (all)':24} {fmt(old['graph_all']):24} -> {fmt(new['graph_all'])}")
    for sec in ("graph", "new_fields"):
        for f in sorted(set(old.get(sec, {})) | set(new.get(sec, {}))):
            print(f"  {sec + '.' + f:24} {fmt(old.get(sec, {}).get(f, {})):24} -> {fmt(new.get(sec, {}).get(f, {}))}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--receipts", default=str(L.RECEIPTS))
    ap.add_argument("--out")
    ap.add_argument("--baseline", help="an earlier --out file: print each field's tallies beside it")
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
    print("new fields")
    for f, t in sorted(s["new_fields"].items()):
        print(f"  {f:22} {t}")
    for p in papers:
        bad = [r for r in p["new_fields"] if r["score"] != "correct"]
        if bad:
            print(f"  {p['arxiv_id']}: {[(r['field'], r.get('op'), r['truth'], r['got'], r['score']) for r in bad]}")
    if a.baseline:
        print_baseline(json.loads(pathlib.Path(a.baseline).read_text(encoding="utf-8"))["summary"], s)
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
