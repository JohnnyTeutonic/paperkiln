#!/usr/bin/env python3
"""Score models on the standard small-model benchmarks (tools/bench_build.py's item file) by log-prob.

    python tools/bench_eval.py run --items ITEMS --out DIR --scorer mtstudio --mtstudio BIN --model tag=spec.json
    python tools/bench_eval.py run --items ITEMS --out DIR --scorer hf --hf tag=REPO@REVISION [--device cuda]
    python tools/bench_eval.py run --items ITEMS --out DIR --scorer ember --ember-url URL (--model tag=spec.json |
                                   --hf tag=REPO@REVISION, for the tokeniser)
    python tools/bench_eval.py run ... --scorer ir --model tag=spec.json        (spec/tools/ir_numpy.py on the GGUF)
        [--sets hellaswag,arc_easy,...] [--every N] [--blimp-per N] [--chunk 2048]
    python tools/bench_eval.py report DIR [DIR ...] [--md OUT.md]
    python tools/bench_eval.py compare DIR_A/TAG DIR_B/TAG     # the same model under two engines

Every item becomes one or more (prompt ids, continuation ids) requests, scored as the summed log-probability of
the continuation (temperature 1, log-softmax in double) after the prompt, with the argmax at each continuation
position. Context and continuation are tokenised separately and concatenated; every prompt starts with the
model's end-of-text id (the document boundary all these models were trained with), so BLiMP's empty context is
that one id. A sequence longer than the model's context + 1 loses context tokens from the left (after the
end-of-text id); the count per set is in meta.json.

Metrics (lm-evaluation-harness's zero-shot definitions):
  mc        acc (highest summed log-prob; a tie at the top earns 1/tied) and acc_norm (log-prob divided by the
            choice's length in characters, without the delimiter space);
  partial   (WinoGrande) acc: the continuation's log-prob after each filled-in context;
  lastword  (LAMBADA) acc: every target token is the argmax; ppl = exp(-mean target log-prob per item).
95% intervals are Wilson intervals on the item count; chance is the mean of 1/choices over the set's items.
Engines: `mtstudio score` (paperkiln's own forward pass, the spec's safetensors), HF transformers (third-party
references, float32), ember's POST /score, the numpy IR (slow; cross-checks). Scores are written per chunk,
atomically, so a rerun resumes; meta.json records the items' sha256, the tokeniser, the engine and the timings.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "spec", "tools"))  # keramos (the ir scorer)
sys.path.insert(0, HERE)

SETS = ("hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande", "lambada", "blimp")
HEADLINE = {"hellaswag": "acc_norm", "arc_easy": "acc", "arc_challenge": "acc_norm", "piqa": "acc",
            "winogrande": "acc", "lambada": "acc", "blimp": "acc"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(p):
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def write_atomic(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(max(p * (1 - p), 0.0) / n + z * z / (4 * n * n)) / d
    return [max(0.0, round(c - h, 4)), min(1.0, round(c + h, 4))]


def unique_ids(items):
    """Every item its own id, so every request key names one request. bench_v1.jsonl gives 433 ids to two
    HellaSwag items each (the validation split's `ind` is not unique): the second item's results replaced the
    first's under the shared key, so the first was credited with the second's scores, and a cross-check that
    sampled the first compared it with the second (HellaSwag 17234, 26 Oct 2026; experiments/tiny_chat_2026-10/
    SCORE_AUDIT_2026-10-26.md). The n-th item (n >= 2) under an id becomes id#n; first ones keep their id."""
    seen, out = {}, []
    for it in items:
        n = seen[it["id"]] = seen.get(it["id"], 0) + 1
        out.append(it if n == 1 else {**it, "id": f"{it['id']}#{n}"})
    return out


def select(items, sets, every, blimp_per):
    out, seen = [], {}
    for it in items:
        if it["set"] not in sets:
            continue
        if it["set"] == "blimp" and blimp_per is not None:
            seen[it["paradigm"]] = seen.get(it["paradigm"], 0) + 1
            if seen[it["paradigm"]] > blimp_per:
                continue
        out.append(it)
    return out[::every] if every > 1 else out


# ---------------------------------------------------------------- tokenisers
class PkTok:
    """paperkiln's tokenisers (tools/carry5_eval.py Tok: bpe.py, or the 16k Llama 3-scheme vocabulary)."""

    def __init__(self, spec_path):
        import carry5_eval
        spec, gguf, vocab = carry5_eval.model_paths(spec_path)
        t = carry5_eval.Tok(vocab)
        self.encode, self.eot = t.encode, t.eot
        self.ctx = int(spec["data"]["T"])
        self.desc = {"kind": "paperkiln", "vocab": os.path.basename(vocab), "vocab_sha256": sha256(vocab)}
        self.spec, self.gguf = spec, gguf


class HfTok:
    def __init__(self, repo_rev, ctx_cap=2048):
        from transformers import AutoConfig, AutoTokenizer
        repo, rev = repo_rev.split("@", 1)
        self.t = AutoTokenizer.from_pretrained(repo, revision=rev)
        cfg = AutoConfig.from_pretrained(repo, revision=rev)
        eot = self.t.convert_tokens_to_ids("<|endoftext|>")
        self.eot = eot if isinstance(eot, int) and eot != self.t.unk_token_id else self.t.eos_token_id
        self.ctx = min(int(getattr(cfg, "max_position_embeddings", ctx_cap)), ctx_cap)
        self.desc = {"kind": "hf", "repo": repo, "revision": rev, "eot_id": self.eot}
        self.repo, self.rev = repo, rev

    def encode(self, s):
        return self.t(s, add_special_tokens=False)["input_ids"]


# ---------------------------------------------------------------- requests
UNFIT = set()  # request keys whose continuation alone does not fit the context (filled by requests())


def requests(items, tok):
    """[(key, prompt_ids, continuation_ids)] and per-set truncation counts. A continuation longer than the
    context is cut to fit and its key goes in UNFIT: the item then earns chance credit (item_records)."""
    out, trunc = [], {}
    limit = tok.ctx + 1

    def add(key, ctx, cont, s):
        p = [tok.eot] + (tok.encode(ctx) if ctx else [])
        c = tok.encode(cont)
        if len(p) + len(c) > limit:
            keep = limit - len(c) - 1
            if keep < 0:
                UNFIT.add(key)
                p, c = [tok.eot], c[:limit - 1]
                keep = 0
            p = [tok.eot] + (p[len(p) - keep:] if keep else [])
            trunc[s] = trunc.get(s, 0) + 1
        out.append((key, p, c))

    for it in items:
        s = it["set"]
        if it["kind"] == "mc":
            for k, ch in enumerate(it["choices"]):
                add(f"{it['id']}|{k}", it["context"], ch, s)
        elif it["kind"] == "partial":
            for k, cx in enumerate(it["contexts"]):
                add(f"{it['id']}|{k}", cx, it["continuation"], s)
        elif it["kind"] == "lastword":
            add(f"{it['id']}|0", it["context"], it["target"], s)
        else:
            sys.exit(f"{it['id']}: unknown kind {it['kind']}")
    seen = {}
    for k, _, _ in out:
        seen[k] = seen.get(k, 0) + 1
    dup = sorted(k for k, n in seen.items() if n > 1)
    if dup:
        sys.exit(f"{len(dup)} request keys name more than one request (e.g. {dup[:3]}): pass the items "
                 "through unique_ids")
    return out, trunc


# ---------------------------------------------------------------- engines
class MtstudioEngine:
    name = "mtstudio"

    def __init__(self, a, tok):
        self.bin, self.spec_path = a.mtstudio, a.spec_path

    def score(self, part, work, n):
        inp, raw = os.path.join(work, f"chunk{n:05d}.in.jsonl"), os.path.join(work, f"chunk{n:05d}.out.jsonl")
        with open(inp, "w", encoding="utf-8", newline="\n") as f:
            for k, p, c in part:
                f.write(json.dumps({"key": k, "prompt_ids": p, "continuation_ids": c}) + "\n")
        for attempt in (1, 2):
            r = subprocess.run([self.bin, "score", self.spec_path, "--in", inp, "--out", raw],
                               capture_output=True, text=True)
            if r.returncode == 0:
                break
            print(f"mtstudio score failed (attempt {attempt}, rc {r.returncode}): {r.stderr.strip()[-600:]}",
                  file=sys.stderr, flush=True)
        else:
            sys.exit("mtstudio score failed twice")
        got = {x["key"]: {f: x[f] for f in ("over_ctx", "logprobs", "argmax", "sum") if f in x} for x in read_jsonl(raw)}
        os.remove(inp)
        os.remove(raw)
        return got


class IrEngine:
    name = "ir"

    def __init__(self, a, tok):
        import carry5_eval
        self.b = carry5_eval.IrBackend(tok.gguf)

    def score(self, part, work, n):
        return {k: self.b.score(p, c) for k, p, c in part}


class EmberEngine:
    """ember's POST /score (the contract agreed with ember on 26 Oct 2026: mtstudio score's fields)."""
    name = "ember"

    def __init__(self, a, tok):
        self.url = a.ember_url.rstrip("/")
        self.health = json.loads(urllib.request.urlopen(self.url + "/health", timeout=30).read())

    def score(self, part, work, n):
        got = {}
        for i in range(0, len(part), 128):
            body = {"requests": [{"key": k, "prompt_ids": p, "continuation_ids": c} for k, p, c in part[i:i + 128]]}
            req = urllib.request.Request(self.url + "/score", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=3600) as r:
                for x in json.loads(r.read())["results"]:
                    got[x["key"]] = {f: x[f] for f in ("over_ctx", "logprobs", "argmax", "sum") if f in x}
        return got


class HfEngine:
    """HF transformers in float32: one padded batch per group of similar lengths; the log-softmax in double."""
    name = "hf"

    def __init__(self, a, tok):
        import torch
        from transformers import AutoModelForCausalLM
        self.torch = torch
        self.dev = a.device
        self.m = self.load(AutoModelForCausalLM, tok, torch)
        self.m.to(self.dev).eval()
        self.ban = self.banned(tok)
        self.desc = {"torch": torch.__version__, "transformers": __import__("transformers").__version__,
                     "dtype": "float32", "banned_argmax": self.ban,
                     "tf32": bool(torch.backends.cuda.matmul.allow_tf32), "engine": self.name}

    def load(self, cls, tok, torch):
        return cls.from_pretrained(tok.repo, revision=tok.rev, torch_dtype=torch.float32)

    def banned(self, tok):
        return sorted({x for x in (tok.t.unk_token_id, tok.t.pad_token_id) if x is not None and x != tok.eot})

    def score(self, part, work, n, max_tokens=16384):
        torch = self.torch
        order = sorted(range(len(part)), key=lambda i: len(part[i][1]) + len(part[i][2]))
        got, i = {}, 0
        while i < len(order):
            L = len(part[order[i]][1]) + len(part[order[i]][2]) - 1
            j = i
            while j < len(order) and (j - i + 1) * (len(part[order[j]][1]) + len(part[order[j]][2]) - 1) <= max_tokens:
                j += 1
            j = max(j, i + 1)
            batch = [part[order[q]] for q in range(i, j)]
            L = max(len(p) + len(c) - 1 for _, p, c in batch)
            ids = torch.zeros((len(batch), L), dtype=torch.long)
            att = torch.zeros((len(batch), L), dtype=torch.long)
            for r, (_, p, c) in enumerate(batch):
                s = p + c[:-1]
                ids[r, :len(s)] = torch.tensor(s)
                att[r, :len(s)] = 1
            with torch.no_grad():
                lg = self.m(input_ids=ids.to(self.dev), attention_mask=att.to(self.dev)).logits
            for r, (k, p, c) in enumerate(batch):
                rows = lg[r, len(p) - 1:len(p) - 1 + len(c)].double()
                lp = torch.log_softmax(rows, dim=-1)
                tgt = torch.tensor(c, device=rows.device)
                vals = lp.gather(1, tgt[:, None])[:, 0].cpu().tolist()
                rb = rows.clone()
                if self.ban:
                    rb[:, self.ban] = -math.inf
                got[k] = {"over_ctx": False, "logprobs": vals, "argmax": rb.argmax(dim=1).cpu().tolist(),
                          "sum": float(sum(vals))}
            i = j
        return got


class HfGgufEngine(HfEngine):
    """A paperkiln llama GGUF through transformers' GGUF loader (LlamaForCausalLM, float32), batched like the hf
    engine. paperkiln's llama exports are llama.cpp-convention files (LOCAL_LLM_STACK.md section 3, point 2:
    llama.cpp matches mtstudio greedy 40/40 given ids), which the loader maps back to HF's layout. The ids come
    from paperkiln's own tokeniser (the GGUF's is not one transformers can read); the argmax leaves out the
    specials mtstudio score leaves out. Accepted only against mtstudio score on the same requests
    (`compare`)."""
    name = "hfgguf"

    def load(self, cls, tok, torch):
        return cls.from_pretrained(os.path.dirname(os.path.abspath(tok.gguf)), gguf_file=os.path.basename(tok.gguf),
                                   torch_dtype=torch.float32)

    def banned(self, tok):
        import llama3_bpe
        toks = llama3_bpe.read_gguf_kv(tok.gguf, ["tokenizer.ggml.tokens"])["tokenizer.ggml.tokens"]
        names = {"<unk>", "<s>", "</s>", "<pad>"}
        return sorted(i for i, t in enumerate(toks) if (t.decode("latin-1") if isinstance(t, bytes) else t) in names)


ENGINES = {"mtstudio": MtstudioEngine, "ir": IrEngine, "ember": EmberEngine, "hf": HfEngine, "hfgguf": HfGgufEngine}


# ---------------------------------------------------------------- run
def cmd_run(a):
    items_all = unique_ids(read_jsonl(a.items))
    items_sha = sha256(a.items)
    sets = a.sets.split(",") if a.sets else list(SETS)
    items = select(items_all, sets, a.every, a.blimp_per)
    models = [("pk", m) for m in a.model or []] + [("hf", m) for m in a.hf or []]
    if not models:
        sys.exit("pass --model tag=spec.json or --hf tag=REPO@REV")
    for kind, m in models:
        tag, src = m.split("=", 1)
        work = os.path.join(a.out, tag)
        os.makedirs(os.path.join(work, "score"), exist_ok=True)
        tok = PkTok(src) if kind == "pk" else HfTok(src)
        a.spec_path = src if kind == "pk" else None
        UNFIT.clear()
        reqs, trunc = requests(items, tok)
        eng = None
        t0, timing = time.time(), []
        res = {}
        for n, off in enumerate(range(0, len(reqs), a.chunk)):
            part = reqs[off:off + a.chunk]
            sig = hashlib.sha256(json.dumps([(k, p, c) for k, p, c in part]).encode()).hexdigest()
            outp = os.path.join(work, "score", f"chunk{n:05d}.json")
            if os.path.exists(outp):
                got = json.load(open(outp, encoding="utf-8"))
                if got["sig"] == sig:
                    res.update(got["results"])
                    continue
            eng = eng or ENGINES[a.scorer](a, tok)
            t1 = time.time()
            got = eng.score(part, os.path.join(work, "score"), n)
            if set(got) != {k for k, _, _ in part}:
                sys.exit(f"{tag}: the engine returned {len(got)} of {len(part)} rows for chunk {n}")
            write_atomic(outp, json.dumps({"sig": sig, "results": got}))
            res.update(got)
            timing.append([n, len(part), round(time.time() - t1, 2)])
            print(f"{tag}: scored {min(off + a.chunk, len(reqs))}/{len(reqs)} ({time.time() - t1:.1f} s)", flush=True)
        meta = {"tag": tag, "source": src, "scorer": a.scorer, "items": os.path.basename(a.items),
                "items_sha256": items_sha, "sets": sets, "every": a.every, "blimp_per": a.blimp_per,
                "n_items": len(items), "n_requests": len(reqs), "tokeniser": tok.desc, "ctx": tok.ctx,
                "prefix": "end-of-text id", "truncated": trunc, "unfit_items": sorted({k.rsplit("|", 1)[0] for k in UNFIT}), "seconds_this_run": round(time.time() - t0, 1),
                "chunk_timing_this_run": timing}
        if eng is not None and hasattr(eng, "desc"):
            meta["engine"] = eng.desc
        if eng is not None and hasattr(eng, "health"):
            meta["engine"] = eng.health
        if kind == "pk":
            meta["gguf"] = os.path.basename(tok.gguf)
            st = os.path.join(tok.spec["out_dir"], tok.spec["name"] + ".safetensors")
            if os.path.exists(st):
                meta["safetensors_sha256"] = sha256(st)
        prev = os.path.join(work, "meta.json")
        if os.path.exists(prev):
            old = json.load(open(prev, encoding="utf-8"))
            meta["seconds_total"] = round(old.get("seconds_total", old.get("seconds_this_run", 0)) +
                                          meta["seconds_this_run"], 1)
        write_atomic(prev, json.dumps(meta, indent=1))
        summarise(work, items, res, meta, {k: c for k, _, c in reqs})


def choose(scores, gold):
    top = max(scores)
    tied = [k for k, s in enumerate(scores) if s == top]
    return (1.0 / len(tied) if gold in tied else 0.0), tied[0]


def item_records(items, res, conts):
    recs = []
    for it in items:
        r = {"id": it["id"], "set": it["set"]}
        if it["set"] == "blimp":
            r["paradigm"] = it["paradigm"]
        n_req = len(it["choices"]) if it["kind"] == "mc" else len(it["contexts"]) if it["kind"] == "partial" else 1
        if any(f"{it['id']}|{k}" in UNFIT for k in range(n_req)):
            # a choice longer than the model's context cannot be scored: chance credit, counted in meta.json
            ch = 1.0 / n_req if it["kind"] != "lastword" else 0.0
            r.update(acc=ch, chance=ch, unfit=True)
            if it["kind"] == "mc" and it["set"] != "blimp":
                r["acc_norm"] = ch
            if it["kind"] == "lastword":
                r["sum"] = res[f"{it['id']}|0"]["sum"]
            recs.append(r)
            continue
        if it["kind"] == "mc":
            sums = [res[f"{it['id']}|{k}"]["sum"] for k in range(len(it["choices"]))]
            lens = [max(len(ch[1:] if ch.startswith(" ") else ch), 1) for ch in it["choices"]]
            r["acc"], r["pick"] = choose(sums, it["gold"])
            if it["set"] != "blimp":  # BLiMP compares whole sentences: no length normalisation
                r["acc_norm"], r["pick_norm"] = choose([s / n for s, n in zip(sums, lens)], it["gold"])
            r["chance"] = 1.0 / len(sums)
            r["sums"] = sums
        elif it["kind"] == "partial":
            sums = [res[f"{it['id']}|{k}"]["sum"] for k in range(len(it["contexts"]))]
            r["acc"], r["pick"] = choose(sums, it["gold"])
            r["chance"] = 1.0 / len(sums)
            r["sums"] = sums
        else:
            key = f"{it['id']}|0"
            x = res[key]
            r["acc"] = 1.0 if list(x["argmax"]) == list(conts[key]) else 0.0
            r["sum"] = x["sum"]
            r["chance"] = 0.0
        recs.append(r)
    return recs


def summarise(work, items, res, meta, conts):
    """summary.json (per set: n, chance, acc / acc_norm with Wilson 95% intervals, LAMBADA ppl, BLiMP per
    paradigm) and items.jsonl.gz (per item: the summed log-probs, the pick, the credit)."""
    recs = item_records(items, res, conts)
    out = {"meta": {k: meta[k] for k in ("tag", "scorer", "items_sha256", "n_items", "ctx", "truncated")}, "sets": {}}
    for s in SETS:
        rs = [r for r in recs if r["set"] == s]
        if not rs:
            continue
        n = len(rs)
        d = {"n": n, "chance": round(sum(r["chance"] for r in rs) / n, 4),
             "unfit": sum(1 for r in rs if r.get("unfit"))}
        for m in ("acc", "acc_norm"):
            if m in rs[0]:
                k = sum(r[m] for r in rs)
                d[m] = round(k / n, 4)
                d[m + "_ci95"] = wilson(k, n)
        if s == "lambada":
            d["ppl"] = round(math.exp(-sum(r["sum"] for r in rs) / n), 3)
        if s == "blimp":
            par = {}
            for r in rs:
                par.setdefault(r["paradigm"], []).append(r["acc"])
            d["per_paradigm"] = {p: round(sum(v) / len(v), 4) for p, v in sorted(par.items())}
        d["headline"] = HEADLINE[s]
        out["sets"][s] = d
    write_atomic(os.path.join(work, "summary.json"), json.dumps(out, indent=1))
    with gzip.open(os.path.join(work, "items.jsonl.gz"), "wt", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(json.dumps({s: (v[v["headline"]], v[v["headline"] + "_ci95"]) for s, v in out["sets"].items()}), flush=True)


def fmt_cell(d, m):
    if d is None or m not in d:
        return "-"
    lo, hi = d[m + "_ci95"]
    return f"{d[m]:.3f} [{lo:.3f}, {hi:.3f}]"


def cmd_report(a):
    rows = []
    for d in a.dirs:
        for tag in sorted(os.listdir(d)):
            p = os.path.join(d, tag, "summary.json")
            if os.path.exists(p):
                rows.append((tag, json.load(open(p, encoding="utf-8"))))
    sets = [s for s in SETS if any(s in r["sets"] for _, r in rows)]
    lines = ["| model | engine | " + " | ".join(f"{s} ({HEADLINE[s]})" for s in sets) + " |",
             "|---|---|" + "---|" * len(sets)]
    for tag, r in rows:
        lines.append(f"| {tag} | {r['meta']['scorer']} | " +
                     " | ".join(fmt_cell(r["sets"].get(s), HEADLINE[s]) for s in sets) + " |")
    chance = {s: next(r["sets"][s]["chance"] for _, r in rows if s in r["sets"]) for s in sets}
    lines.append("| chance | | " + " | ".join(f"{chance[s]:.3f}" for s in sets) + " |")
    lines += ["", "| model | " + " | ".join(f"{s} {m}" for s in sets for m in ("acc", "acc_norm")
                                           if any(m in r["sets"].get(s, {}) for _, r in rows)) + " | lambada ppl |"]
    cols = [(s, m) for s in sets for m in ("acc", "acc_norm") if any(m in r["sets"].get(s, {}) for _, r in rows)]
    lines.append("|---|" + "---|" * (len(cols) + 1))
    for tag, r in rows:
        lines.append(f"| {tag} | " + " | ".join(
            f"{r['sets'][s][m]:.3f}" if s in r["sets"] and m in r["sets"][s] else "-" for s, m in cols) +
            f" | {r['sets'].get('lambada', {}).get('ppl', '-')} |")
    text = "\n".join(lines) + "\n"
    if a.md:
        write_atomic(a.md, text)
    print(text)


def load_scores(work):
    res = {}
    for f in sorted(os.listdir(os.path.join(work, "score"))):
        if f.endswith(".json"):
            res.update(json.load(open(os.path.join(work, "score", f), encoding="utf-8"))["results"])
    return res


def cmd_compare(a):
    """Two engines on the same model and requests: per-token log-prob differences, argmax and choice agreement."""
    A, B = load_scores(a.a), load_scores(a.b)
    keys = sorted(set(A) & set(B))
    if not keys:
        sys.exit("no common requests")
    # The same key must be the same request: a continuation of another length is another request (a
    # zip over the two would compare the shorter's positions with the other's first ones, as the 17234
    # cross-check did). Continuations differ in length only when the items differ.
    other = [k for k in keys if not (A[k].get("over_ctx") or B[k].get("over_ctx"))
             and len(A[k]["logprobs"]) != len(B[k]["logprobs"])]
    if other:
        sys.exit(f"{len(other)} keys name requests of different lengths in the two runs, so not the same requests "
                 f"(e.g. {other[:4]}); rescore both from one items file")
    worst = 0.0
    tok_n = arg_eq = arg_n = 0
    sums = []
    for k in keys:
        x, y = A[k], B[k]
        if x.get("over_ctx") or y.get("over_ctx"):
            continue
        d = max(abs(p - q) for p, q in zip(x["logprobs"], y["logprobs"]))
        worst = max(worst, d)
        tok_n += len(x["logprobs"])
        arg_n += len(x["argmax"])
        arg_eq += sum(p == q for p, q in zip(x["argmax"], y["argmax"]))
        sums.append(abs(x["sum"] - y["sum"]))
    # choice agreement per item (items whose every request is in both)
    by = {}
    for k in keys:
        iid, c = k.rsplit("|", 1)
        by.setdefault(iid, {})[int(c)] = (A[k]["sum"], B[k]["sum"])
    same = tot = 0
    for iid, d in by.items():
        if len(d) < 2:
            continue
        ca = max(d, key=lambda c: d[c][0])
        cb = max(d, key=lambda c: d[c][1])
        tot += 1
        same += ca == cb
    sums.sort()
    out = {"a": a.a, "b": a.b, "requests": len(keys), "tokens": tok_n, "max_abs_logprob_diff": worst,
           "median_abs_sum_diff": sums[len(sums) // 2] if sums else None, "max_abs_sum_diff": sums[-1] if sums else None,
           "argmax_agree": f"{arg_eq}/{arg_n}", "choice_agree": f"{same}/{tot}"}
    print(json.dumps(out, indent=1))
    if a.json:
        write_atomic(a.json, json.dumps(out, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--items", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--scorer", choices=sorted(ENGINES), required=True)
    r.add_argument("--model", action="append", help="tag=spec.json (paperkiln)")
    r.add_argument("--hf", action="append", help="tag=REPO@REVISION (HF transformers model and tokeniser)")
    r.add_argument("--mtstudio")
    r.add_argument("--ember-url")
    r.add_argument("--device", default="cuda")
    r.add_argument("--sets")
    r.add_argument("--every", type=int, default=1, help="every Nth selected item (a sample for engine checks)")
    r.add_argument("--blimp-per", type=int, default=None, help="BLiMP pairs per paradigm (the first N)")
    r.add_argument("--chunk", type=int, default=2048)
    p = sub.add_parser("report")
    p.add_argument("dirs", nargs="+")
    p.add_argument("--md")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--json")
    a = ap.parse_args(argv)
    {"run": cmd_run, "report": cmd_report, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
