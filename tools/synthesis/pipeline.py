#!/usr/bin/env python3
"""Synthesis search: enumerate (component, deficit, mechanism) candidates,
write each up as a concrete architectural synthesis, scope it against the
literature, and rank what survives.

    python tools/synthesis/pipeline.py enumerate                 # -> results/candidates.jsonl
    python tools/synthesis/pipeline.py generate [--limit N]      # -> results/ideas.jsonl
    python tools/synthesis/pipeline.py scope    [--limit N]      # -> results/scoped.jsonl
    python tools/synthesis/pipeline.py score                     # -> results/ranked.md
    python tools/synthesis/pipeline.py all --limit 16 --control  # smoke run

Every stage caches by candidate id, so re-running only does new work.
`--control` injects known cases whose verdict we already know (the KV
diffusion idea, taken by arXiv 2505.16950); a run whose scoper does not
mark the control TAKEN is not to be trusted on the rest.

Keys: ANTHROPIC_API_KEY from the environment, else from
AI_ML/agora/.env two levels above the paperkiln root. Search: the arXiv
API (no key; polite 3 s spacing). Models: generation and judging on
claude-sonnet-5, query writing on claude-haiku-4-5-20251001.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vocab  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "results")
GEN_MODEL = "claude-sonnet-5"
JUDGE_MODEL = "claude-sonnet-5"
QUERY_MODEL = "claude-haiku-4-5-20251001"
ARXIV_SLEEP = 3.0


# ----------------------------------------------------------------- utils
def load_key():
    k = os.environ.get("ANTHROPIC_API_KEY")
    if k:
        return k
    p = os.path.join(REPO, "..", "AI_ML", "agora", ".env")
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            m = re.match(r"\s*ANTHROPIC_API_KEY\s*=\s*(.*)", line)
            if m:
                return m.group(1).strip().strip('"').strip("'")
    raise SystemExit("no ANTHROPIC_API_KEY")


_client = None


def client():
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic(api_key=load_key())
    return _client


def ask(model, system, user, max_tokens=4000, retries=4):
    for i in range(retries):
        try:
            # thinking is off: by default it consumes the output budget and
            # the text block comes back empty on long structured answers
            r = client().messages.create(model=model, max_tokens=max_tokens, system=system,
                                         messages=[{"role": "user", "content": user}],
                                         thinking={"type": "disabled"})
            text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
            if r.stop_reason == "max_tokens" and max_tokens < 32000:
                print(f"  [api] truncated at {max_tokens}; retrying with double", flush=True)
                max_tokens *= 2
                continue
            return text
        except Exception as e:  # rate limits, transient
            wait = 5 * (i + 1)
            print(f"  [api] {type(e).__name__}: {str(e)[:120]}; retry in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError("api failed")


def parse_json(text):
    """First JSON array or object in the text, tolerant of code fences."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M)
    # take whichever container opens first, so an object holding an array
    # is not mistaken for the array
    starts = [(text.find(o), o, c) for o, c in (("[", "]"), ("{", "}")) if text.find(o) >= 0]
    for i, opener, closer in sorted(starts):
        j = text.rfind(closer)
        if j > i:
            try:
                return json.loads(text[i:j + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON in response: " + text[:200])


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def append_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def cid(component, deficit, mechanism):
    return hashlib.sha1(f"{component}|{deficit}|{mechanism}".encode()).hexdigest()[:10]


# ------------------------------------------------------------- enumerate
CONTROLS = [
    {
        "id": "ctl-kvdiff", "control": True, "expect": "TAKEN", "expect_ref": "2505.16950",
        "component": "kv-cache",
        "deficit": "append-only: an entry written at step t is never revised by later context",
        "mechanism": "masked-diffusion-denoising",
        "ops": ["rewrite", "denoise"],
    },
    {
        "id": "ctl-pid", "control": True, "expect": "TAKEN", "expect_ref": "2408.15664 (Auxiliary-Loss-Free Load Balancing: per-expert bias driven by load error)",
        "component": "moe-router",
        "deficit": "load balancing is imposed by an auxiliary loss that fights the quality objective",
        "mechanism": "feedback-control",
        "ops": ["feedback", "schedule"],
    },
    {
        "id": "ctl-hebbffn", "control": True, "expect": "TAKEN", "expect_any": ["TAKEN", "ADJACENT"], "expect_ref": "2212.02475 (Meta-Learning Fast Weight Language Models) and 2601.00671 (Fast-weight Product Key Memory)",
        "component": "ffn",
        "deficit": "its memory is fixed at training time and never edited by the stream",
        "mechanism": "hebbian-fast-weights",
        "ops": ["adapt", "rewrite"],
    },
    {
        "id": "ctl-choquet", "control": True, "expect": "ADJACENT", "expect_ref": "2607.26164 (Choquet MoE aggregation in IR spectroscopy; different purpose)",
        "component": "moe-router",
        "deficit": "experts are scored independently, so top-k cannot express that two experts are redundant or complementary",
        "mechanism": "choquet-integral",
        "ops": ["aggregate-interactions", "select-set"],
    },
]


def enumerate_candidates(control=False):
    rows = []
    for c, cd in vocab.COMPONENTS.items():
        for deficit, ops in cd["deficits"]:
            for m, (desc, mops) in vocab.MECHANISMS.items():
                inter = sorted(set(ops) & set(mops))
                if inter:
                    rows.append({"id": cid(c, deficit, m), "component": c, "deficit": deficit,
                                 "mechanism": m, "ops": inter})
    if control:
        rows = CONTROLS + rows
    return rows


# -------------------------------------------------------------- generate
GEN_SYSTEM = """You are a senior ML architecture researcher writing candidate research syntheses for a two-person team with a from-scratch C++ trainer and a few Colab GPUs. Each candidate pairs a COMPONENT of a transformer language-model system, a DEFICIT that component has by construction, and a MECHANISM from elsewhere whose native operation negates the deficit. Your job is to turn each triple into a concrete, testable architectural proposal, or to reject it if the pairing is incoherent or trivial.

Be concrete and specific. The good examples of this move: "run one masked-denoising step over the last window of the KV cache every k tokens so later context can rewrite earlier entries; outputs stay causal because edits only affect future reads"; "route MoE experts by maximising a 2-additive Choquet capacity over the set instead of summing independent scores". The bad examples: vague "use X to improve Y", or re-descriptions of existing methods (speculative decoding, RoPE, early exit) under new names.

For each triple return a JSON object with exactly these keys:
- id
- reject: true/false. Reject if incoherent, trivial, or obviously an existing standard method.
- reject_reason: short, or null
- name: a 3-6 word name
- claim: one sentence, the mechanism placed at the component, in the form "<mechanism> applied to <component> so that <new capability>"
- new_capability: one sentence: what the system can now do that the vanilla system cannot by construction
- placement: 2-4 sentences: exactly what is computed, where, how often, what is trained
- invariant: which property keeps the system valid (choose from: outputs stay causal; exactness recoverable by verification; matched FLOPs/params; base model untouched; streaming) and one sentence on why it holds
- falsifier: object with keys hypothesis, metric, baseline, protocol (nano-scale, seeds, matched budget), gpu_hours (integer estimate)
- feasibility: 1-5 (5 = a week of work on the stated testbed)
- splash: 1-5 (5 = changes what the system can do at an interface, objective or invariant; 1 = a tuning trick)
- nearest_known: your best guess of the closest existing method or paper, or null. Be honest; this is checked.
Return a JSON array, one object per triple, nothing else."""


def gen_batch(batch):
    lines = []
    for r in batch:
        comp = vocab.COMPONENTS.get(r["component"], {}).get("what", r["component"])
        mech = vocab.MECHANISMS.get(r["mechanism"], (r["mechanism"], []))[0]
        lines.append(f"- id {r['id']}: COMPONENT {r['component']} ({comp}); DEFICIT: {r['deficit']}; "
                     f"MECHANISM {r['mechanism']} ({mech}); negating ops {r['ops']}")
    user = ("TESTBED: " + vocab.TESTBED + "\n\nINVARIANTS available: " + "; ".join(vocab.INVARIANTS)
            + "\n\nTRIPLES:\n" + "\n".join(lines))
    out = None
    for attempt in range(2):
        try:
            out = parse_json(ask(GEN_MODEL, GEN_SYSTEM, user, max_tokens=12000))
            break
        except ValueError as e:
            print(f"  [gen] unparseable batch ({str(e)[:60]}); retry {attempt + 1}", flush=True)
            user = user + "\n\nReturn ONLY a valid JSON array. Escape quotes inside strings. No prose."
    if out is None:
        return []  # leave the batch uncached so a later run retries it
    if isinstance(out, dict):
        out = [out]
    by = {o.get("id"): o for o in out if isinstance(o, dict)}
    res = []
    for r in batch:
        o = by.get(r["id"])
        if not o:
            o = {"id": r["id"], "reject": True, "reject_reason": "generator returned nothing"}
        o.update({k: r[k] for k in ("component", "deficit", "mechanism", "ops")})
        for k in ("control", "expect", "expect_any", "expect_ref"):
            if k in r:
                o[k] = r[k]
        res.append(o)
    return res


def stage_generate(limit, control, batch_size=6, ids=None):
    cands = enumerate_candidates(control)
    if ids:
        cands = [c for c in cands if c["id"] in ids]
    done = {r["id"] for r in read_jsonl(os.path.join(OUT, "ideas.jsonl"))}
    todo = [c for c in cands if c["id"] not in done]
    if limit:
        todo = todo[:limit]
    print(f"generate: {len(todo)} to do ({len(done)} cached)", flush=True)
    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        try:
            rows = gen_batch(batch)
        except Exception as e:  # one bad batch must not end the run
            print(f"  [gen] batch failed: {type(e).__name__}: {str(e)[:100]}", flush=True)
            rows = []
        if not rows:
            continue
        append_jsonl(os.path.join(OUT, "ideas.jsonl"), rows)
        kept = sum(1 for r in rows if not r.get("reject"))
        print(f"  batch {i // batch_size + 1}: {kept}/{len(rows)} kept", flush=True)


# ------------------------------------------------------------------ scope
QUERY_SYSTEM = """You write literature-search queries for the arXiv API. A paper that already does the proposed thing will usually NOT use the proposer's vocabulary: it may name the mechanism differently, or describe only the operation and the purpose. So write FIVE fielded arXiv queries at three levels, using the API syntax (all:, ti:, abs:, AND, OR, quoted phrases; no plain-keyword queries):
1. mechanism level: the mechanism's name and its two most common synonyms, with the component;
2. operation level (TWO queries): what is done to which component, in plain words, WITHOUT naming the mechanism (e.g. all:"KV cache" AND (all:rewrite OR all:revise OR all:consolidation));
3. purpose level (TWO queries): the capability gained, in the words a paper would put in its title, without the mechanism name.
Return only a JSON array of 5 strings."""


def backstop_queries(idea):
    """Deterministic fielded queries from the component's aliases and the
    negating ops' synonyms: no mechanism word at all."""
    aliases = vocab.ALIASES.get(idea["component"], [idea["component"]])[:2]
    ops = (idea.get("ops") or [])[:2]
    out = []
    for a in aliases:
        for op in ops:
            syn = vocab.OP_SYNONYMS.get(op, [op])[:4]
            out.append(f'all:"{a}" AND (' + " OR ".join(f'all:"{x}"' if " " in x else f"all:{x}" for x in syn) + ")")
    # the field's own names for the deficit, with no component alias at all:
    # this is what finds "auxiliary-loss-free load balancing" for a PID router
    terms = vocab.field_terms(idea["component"], idea.get("deficit", ""))
    if terms:
        out.append("(" + " OR ".join(f'all:"{t}"' for t in terms[:4]) + ")")
        out.append(f'all:"{aliases[0]}" AND (' + " OR ".join(f'all:"{t}"' for t in terms[:4]) + ")")
    return out[:6]


def arxiv_search(q, n=8):
    url = ("https://export.arxiv.org/api/query?search_query=" + urllib.parse.quote(q)
           + f"&max_results={n}&sortBy=relevance")
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "paperkiln-synthesis-scoper/0.1 (mailto:jonathanreich100@gmail.com)",
                "Accept": "application/atom+xml"})
            xml = urllib.request.urlopen(req, timeout=40).read().decode("utf-8", "replace")
            break
        except urllib.error.HTTPError as e:
            if e.code == 400:
                # malformed fielded syntax from the query writer: fall back
                # to the bare words once, then give up on this query
                if attempt == 0:
                    bare = re.sub(r"(all|ti|abs|au|cat):", "", q).replace("(", " ").replace(")", " ")
                    bare = re.sub(r"(AND|OR|ANDNOT)", " ", bare).replace('"', " ")
                    url = ("https://export.arxiv.org/api/query?search_query=" + urllib.parse.quote(" ".join(bare.split()))
                           + f"&max_results={n}&sortBy=relevance")
                    time.sleep(3)
                    continue
                return []
            print(f"  [arxiv] {e}; retry", flush=True)
            time.sleep(10)
        except Exception as e:
            print(f"  [arxiv] {e}; retry", flush=True)
            time.sleep(10)
    else:
        return []
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = []
    for e in ET.fromstring(xml).findall("a:entry", ns):
        aid = (e.findtext("a:id", "", ns) or "").rsplit("/", 1)[-1]
        out.append({"id": re.sub(r"v\d+$", "", aid),
                    "title": " ".join((e.findtext("a:title", "", ns) or "").split()),
                    "published": (e.findtext("a:published", "", ns) or "")[:10],
                    "abstract": " ".join((e.findtext("a:summary", "", ns) or "").split())[:1500]})
    return out


JUDGE_SYSTEM = """You are a sceptical reviewer deciding whether a proposed research synthesis is already in the literature. You are given the proposal and candidate papers (id, date, title, abstract). Decide:
- TAKEN: a paper places an equivalent mechanism at the SAME component for the SAME purpose (the same new capability), whatever it calls the mechanism. Differences of scale, training recipe, domain of application or framing do not save the proposal. Equivalent mechanism means the same operation on the same object (a non-causal in-place rewrite of cache entries is equivalent to a denoising rewrite of cache entries; a per-expert bias updated from the load error is equivalent to a proportional controller on expert capacity, and adding integral or derivative terms to it is a tuning of the same mechanism, not a new one).
- ADJACENT: clear overlap but not the same purpose: the same mechanism used at that component for a different capability (e.g. aggregating expert outputs versus selecting the expert set), or the same capability obtained by a different mechanism, or the same mechanism at a neighbouring component. The proposal would then be a follow-up unless its stated twist is genuinely different.
- OPEN: nothing in the set does this; say what the nearest paper does instead.
Be strict and grounded: the team refuses to write follow-ups, so a false OPEN costs weeks, and a false TAKEN kills a live idea. For each closest paper quote the phrase from its abstract that establishes the overlap. Return a JSON object: {"verdict": "TAKEN|ADJACENT|OPEN", "closest": [{"id": ..., "title": ..., "evidence": quoted phrase, "why": one sentence}] (up to 3, most relevant first), "survives": one sentence on what twist, if any, would still be new given the closest paper (or null), "confidence": 0-1}. Return only the JSON."""


def scope_one(idea):
    desc = (f"NAME: {idea.get('name')}\nCLAIM: {idea.get('claim')}\nNEW CAPABILITY: {idea.get('new_capability')}\n"
            f"PLACEMENT: {idea.get('placement')}\nCOMPONENT: {idea['component']}; MECHANISM: {idea['mechanism']}\n"
            f"GENERATOR'S OWN NEAREST-KNOWN GUESS: {idea.get('nearest_known')}")
    queries = parse_json(ask(QUERY_MODEL, QUERY_SYSTEM, desc, max_tokens=800))
    if not isinstance(queries, list):
        queries = [str(queries)]
    # backstops FIRST: the mechanism-free queries are the ones that find a
    # paper written in the field's own words, and the paper cap must not
    # truncate them behind the query writer's mechanism-named hits
    queries = backstop_queries(idea) + [str(q) for q in queries[:5]]
    seen = {}
    for q in queries:
        for p in arxiv_search(q, n=10):
            if p["id"] and p["id"] not in seen:
                seen[p["id"]] = p
        time.sleep(ARXIV_SLEEP)
    papers = list(seen.values())[:60]
    if not papers:
        # empty retrieval is NOT evidence of openness (17 Sep 2026: arXiv
        # returned 406 for hours and 71 ideas were marked OPEN on nothing)
        return {"id": idea["id"], "queries": queries, "papers": [], "n_papers": 0,
                "verdict": "ERROR", "error": "no retrieval", "closest": [], "survives": None, "confidence": 0.0}
    plist = "\n\n".join(f"[{p['id']}] ({p['published']}) {p['title']}\n{p['abstract']}" for p in papers)
    j = parse_json(ask(JUDGE_MODEL, JUDGE_SYSTEM, "PROPOSAL:\n" + desc + "\n\nPAPERS:\n" + plist, max_tokens=1200))
    return {"id": idea["id"], "queries": queries,
            "papers": [{k: p[k] for k in ("id", "title", "published")} for p in papers],
            "n_papers": len(papers),
            "verdict": j.get("verdict", "OPEN"), "closest": j.get("closest", []),
            "survives": j.get("survives"), "confidence": j.get("confidence")}


def stage_scope(limit, ids=None):
    ideas = [r for r in read_jsonl(os.path.join(OUT, "ideas.jsonl")) if not r.get("reject")]
    if ids:
        ideas = [r for r in ideas if r["id"] in ids]
    done = {r["id"] for r in read_jsonl(os.path.join(OUT, "scoped.jsonl"))}
    todo = [r for r in ideas if r["id"] not in done]
    if limit:
        todo = todo[:limit]
    print(f"scope: {len(todo)} to do ({len(done)} cached)", flush=True)
    for k, idea in enumerate(todo, 1):
        try:
            s = scope_one(idea)
        except Exception as e:
            s = {"id": idea["id"], "verdict": "ERROR", "error": f"{type(e).__name__}: {str(e)[:200]}"}
        append_jsonl(os.path.join(OUT, "scoped.jsonl"), [s])
        flag = ""
        if idea.get("control"):
            ok = s["verdict"] in (idea.get("expect_any") or [idea["expect"]])
            flag = f"  [CONTROL expected {idea['expect']} -> {'PASS' if ok else 'FAIL'}]"
        print(f"  {k}/{len(todo)} {idea['id']} {s['verdict']:8s} {idea.get('name')}{flag}", flush=True)


# ------------------------------------------------------------------ score
def stage_score():
    ideas = {r["id"]: r for r in read_jsonl(os.path.join(OUT, "ideas.jsonl"))}
    scoped = {r["id"]: r for r in read_jsonl(os.path.join(OUT, "scoped.jsonl"))}
    rows = []
    for i, idea in ideas.items():
        if idea.get("reject"):
            continue
        s = scoped.get(i)
        v = s["verdict"] if s else "UNSCOPED"
        nov = {"OPEN": 3, "ADJACENT": 1, "TAKEN": 0}.get(v, 0)
        feas = int(idea.get("feasibility") or 0)
        spl = int(idea.get("splash") or 0)
        rows.append((nov * 10 + spl * 2 + feas, v, spl, feas, idea, s))
    rows.sort(key=lambda t: -t[0])
    n = len(rows)
    counts = {}
    for r in rows:
        counts[r[1]] = counts.get(r[1], 0) + 1
    lines = ["# Synthesis search: ranked candidates", "",
             f"{n} candidates scored; verdicts: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())), "",
             "Rank = novelty (OPEN 30 / ADJACENT 10 / TAKEN 0) + 2*splash + feasibility. "
             "Novelty is the scoper's verdict against arXiv; read the closest paper before believing any OPEN.", ""]
    controls = [r for r in rows if r[4].get("control")]
    if controls:
        lines.append("## Controls")
        for _, v, _, _, idea, s in controls:
            ok = "PASS" if v in (idea.get("expect_any") or [idea.get("expect")]) else "FAIL"
            lines.append(f"- {idea['id']} expected {idea.get('expect')} got {v}: **{ok}**; closest "
                         + "; ".join(c.get("id", "?") for c in (s or {}).get("closest", [])))
        lines.append("")
    lines.append("## Ranked")
    lines.append("")
    lines.append("| # | verdict | splash | feas | name | claim | closest | survives |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for k, (_, v, spl, feas, idea, s) in enumerate(rows, 1):
        closest = "; ".join(f"{c.get('id','?')}" for c in (s or {}).get("closest", [])[:2])
        surv = ((s or {}).get("survives") or "").replace("|", "/")
        lines.append(f"| {k} | {v} | {spl} | {feas} | {idea.get('name')} | {str(idea.get('claim')).replace('|','/')} | {closest} | {surv} |")
    lines.append("")
    lines.append("## Top OPEN, in full")
    lines.append("")
    for _, v, spl, feas, idea, s in [r for r in rows if r[1] == "OPEN"][:25]:
        f = idea.get("falsifier") or {}
        lines += [f"### {idea.get('name')}  (splash {spl}, feasibility {feas}, id {idea['id']})", "",
                  f"**Claim.** {idea.get('claim')}", "", f"**New capability.** {idea.get('new_capability')}", "",
                  f"**Placement.** {idea.get('placement')}", "", f"**Invariant.** {idea.get('invariant')}", "",
                  f"**Falsifier.** H: {f.get('hypothesis')} Metric: {f.get('metric')} Baseline: {f.get('baseline')} "
                  f"Protocol: {f.get('protocol')} GPU-hours: {f.get('gpu_hours')}", "",
                  f"**Nearest (generator's guess).** {idea.get('nearest_known')}", "",
                  "**Scoper.** closest: " + "; ".join(f"[{c.get('id')}] {c.get('title')}: {c.get('why')}" for c in (s or {}).get("closest", []))
                  + f" | survives: {(s or {}).get('survives')} | confidence {(s or {}).get('confidence')}", ""]
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "ranked.md"), "w", encoding="utf-8").write("\n".join(lines))
    print(f"score: {n} rows -> {os.path.join(OUT, 'ranked.md')}; verdicts {counts}")


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["enumerate", "generate", "scope", "score", "all"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--ids", nargs="*")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.stage in ("enumerate", "all"):
        rows = enumerate_candidates(args.control)
        with open(os.path.join(OUT, "candidates.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"enumerate: {len(rows)} candidates")
    if args.stage in ("generate", "all"):
        stage_generate(args.limit, args.control, ids=args.ids)
    if args.stage in ("scope", "all"):
        stage_scope(args.limit, ids=args.ids)
    if args.stage in ("score", "all"):
        stage_score()
    return 0


if __name__ == "__main__":
    sys.exit(main())
