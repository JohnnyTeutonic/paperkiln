#!/usr/bin/env python3
"""Base 2's corpus arithmetic and document checks, for tools/colab_chat/base2_data.sh (BASE2_PLAN.md sections 2.3 and 2.4, in
the author's research tree). Claude Fable 5.2, 11 Nov 2026.

    base2_corpus.py recipe-n --avail fineweb=B,simplewiki=B,human=B,soda=B,ultrachat=B,everyday=B --out recipe_N.json
    base2_corpus.py recipe-s --recipe-n recipe_N.json --n-ledger N.ledger.json --n-ids ids_N.json \
                             --scn-bytes B --scn-tokens T --scn-docs D --shard-docs K --max-drop F --out recipe_S.json
    base2_corpus.py get FILE KEY[.KEY...]                 # one value of a JSON file, for the shell
    base2_corpus.py docfilter --repo . --heldout VAL.txt SFT.jsonl [--exclude PANEL.txt] --in X.txt --out Y.txt --ledger L.json
    base2_corpus.py share --mix S.main.txt --scenario scenario.txt --vocab V.gguf [--ids-record ids_S.json]
                                                          # scenario tokens over the part's tokens; non-zero outside 14-16%
    base2_corpus.py subset N.ledger.json S.ledger.json    # S's natural documents are among N's
    base2_corpus.py head-docs IN OUT BYTES                # whole documents from the start of IN, up to BYTES
    base2_corpus.py store-has CONFIG BRANCH PREFIX        # exit 0 when the store holds that set's manifest, 1 when not, 2 unread
    base2_corpus.py scn-ledger / stage-ledger ...         # the scenario stream's and a stage's records
    base2_corpus.py loadprobe --spec SPEC.json --out DIR  # a short run on the real ids: start-up time and peak memory

The natural corpus (arm N). Section 2.3's table is the shape; the bytes come from what Stage 1 measured. Each dialogue source
is in the file COPIES times and is used up to USE of that; the dialogue total then sets the corpus size, because the main part
is 24.2% dialogue (FineWeb-Edu and Simple English Wikipedia are 75.8% together), the tail is 47% dialogue, and the tail is
2,600 of 16,600 bytes. Simple English Wikipedia goes whole into the main part and FineWeb-Edu takes the rest of the 75.8%.

The scenario corpus (arms S and C). Each part keeps N's token count: 85% of N's natural bytes, source by source, plus scenario
text for 15% of N's tokens. The scenario bytes for that follow from the scenario text's measured bytes a token, so S's file
is not N's size in bytes when the two kinds of text tokenise differently; its token count is N's. With the same seed and the
same source files, mix_pretrain.py then takes a prefix of the documents it took for N from every natural source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SEP = b"<|endoftext|>\n"

COPIES = {"fineweb": 1, "simplewiki": 1, "human": 3, "soda": 2, "ultrachat": 2, "everyday": 2}
DIALOGUE = ("human", "soda", "ultrachat", "everyday")
NATURAL = ("fineweb", "simplewiki", *DIALOGUE)
USE = 0.99                      # of a source's bytes: a part's budget is met by whole documents, so 100% runs a source dry
TAIL_FRACTION = 2600 / 16600    # section 2.3: 14.0 GB main, 2.6 GB tail
MAIN_NONDIALOGUE = 0.758        # FineWeb-Edu 0.74 and Simple English Wikipedia 0.018 in the plan's table
TAIL_FINEWEB = 0.53
SIMPLEWIKI_MAX = 0.03           # of the main part; the plan expects 0.018
SCENARIO_SHARE = 0.15           # of tokens, both parts
SCENARIO_SPARE = 1.03           # scenario text generated over what the mix takes


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def jsave(p, obj):
    with open(p + ".tmp", "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")
    os.replace(p + ".tmp", p)


def ratio_arg(r):
    return ",".join(f"{k}={v:.8f}" for k, v in sorted(r.items()))


def recipe_n(avail):
    """avail: bytes of one copy of each source after filtering. Returns the mix's arguments and the working."""
    missing = [s for s in NATURAL if avail.get(s, 0) <= 0]
    if missing:
        raise SystemExit(f"recipe-n: no bytes for {missing}")
    supply = {s: COPIES[s] * avail[s] * USE for s in NATURAL}
    d = sum(supply[s] for s in DIALOGUE)
    f = TAIL_FRACTION
    total = int(d / ((1 - MAIN_NONDIALOGUE) * (1 - f) + (1 - TAIL_FINEWEB) * f))
    main_bytes, tail_bytes = total * (1 - f), total * f
    sw = min(supply["simplewiki"] / main_bytes, SIMPLEWIKI_MAX)
    main = {"fineweb": MAIN_NONDIALOGUE - sw, "simplewiki": sw}
    tail = {"fineweb": TAIL_FINEWEB, "simplewiki": 0.0}
    for s in DIALOGUE:
        main[s] = (1 - MAIN_NONDIALOGUE) * supply[s] / d
        tail[s] = (1 - TAIL_FINEWEB) * supply[s] / d
    need_fw = main["fineweb"] * main_bytes + tail["fineweb"] * tail_bytes
    if need_fw > supply["fineweb"]:
        raise SystemExit(f"recipe-n: FineWeb-Edu is short: {need_fw:.0f} bytes wanted, {supply['fineweb']:.0f} usable")
    for r in (main, tail):
        if abs(sum(r.values()) - 1) > 1e-9:
            raise SystemExit("recipe-n: ratios do not sum to 1")
    return {"arm": "N", "unit": "bytes of document text", "available_bytes": {s: int(avail[s]) for s in NATURAL},
            "copies": COPIES, "use": USE, "tail_fraction": f, "total_bytes": total,
            "main": main, "tail": tail, "main_arg": ratio_arg(main), "tail_arg": ratio_arg(tail),
            "sources": list(NATURAL), "simplewiki_capped": sw >= SIMPLEWIKI_MAX,
            "demand_bytes": {s: main[s] * main_bytes + tail[s] * tail_bytes for s in NATURAL},
            "dialogue_share_of_bytes": d / total}


def recipe_s(rn, n_ledger, n_tokens, scn_bytes, scn_tokens, scn_docs, shard_docs, max_drop):
    """rn: recipe-n's output; n_ledger: N's mix ledger; n_tokens: {"main": tokens, "tail": tokens} of N's ids; the scenario
    sample's bytes, tokens and documents (after its filter). Each part: 85% of N's natural bytes and 15% of N's tokens."""
    if n_ledger["sources"].keys() != set(NATURAL):
        raise SystemExit("recipe-s: N's ledger does not hold the natural sources alone")
    b_s = scn_bytes / scn_tokens
    parts, out_ratios = {}, {}
    for p in ("main", "tail"):
        n_bytes, toks = n_ledger["parts"][p]["bytes"], n_tokens[p]
        natural = (1 - SCENARIO_SHARE) * n_bytes
        scenario = SCENARIO_SHARE * toks * b_s
        part = natural + scenario
        r = {s: rn[p][s] * natural / part for s in NATURAL}
        r["scenario"] = scenario / part
        if abs(sum(r.values()) - 1) > 1e-9:
            raise SystemExit("recipe-s: ratios do not sum to 1")
        out_ratios[p] = r
        parts[p] = {"n_bytes": n_bytes, "n_tokens": toks, "n_bytes_per_token": n_bytes / toks, "natural_bytes": natural,
                    "scenario_bytes": scenario, "scenario_tokens_target": SCENARIO_SHARE * toks, "bytes": part,
                    "scenario_byte_ratio": r["scenario"]}
    total = parts["main"]["bytes"] + parts["tail"]["bytes"]
    want = (parts["main"]["scenario_bytes"] + parts["tail"]["scenario_bytes"]) * SCENARIO_SPARE / (1 - max_drop)
    docs = math.ceil(want / (scn_bytes / scn_docs))
    shards = max(1, math.ceil(docs / shard_docs))
    return {"arm": "S", "unit": "bytes of document text", "scenario_share_of_tokens": SCENARIO_SHARE,
            "scenario_bytes_per_token": b_s, "scenario_bytes_per_document": scn_bytes / scn_docs,
            "scenario_sample": {"bytes": scn_bytes, "tokens": scn_tokens, "documents": scn_docs},
            "parts": parts, "total_bytes": int(total), "tail_fraction": parts["tail"]["bytes"] / total,
            "main": out_ratios["main"], "tail": out_ratios["tail"],
            "main_arg": ratio_arg(out_ratios["main"]), "tail_arg": ratio_arg(out_ratios["tail"]),
            "sources": [*NATURAL, "scenario"], "scenario_bytes_wanted": want, "scenario_documents": shards * shard_docs,
            "scenario_shards": shards, "shard_documents": shard_docs, "max_drop": max_drop}


# ---------------------------------------------------------------- documents
def docs(path):
    """Each document of a pretraining text file, as bytes without its separator."""
    buf = b""
    with open(path, "rb") as f:
        while True:
            block = f.read(1 << 24)
            if not block:
                if buf:
                    raise SystemExit(f"{path}: {len(buf)} bytes after the last document separator")
                return
            parts = (buf + block).split(SEP)
            buf = parts.pop()
            for d in parts:
                yield d


def key(d):
    return hashlib.blake2b(d, digest_size=12).digest()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


TURN = re.compile(r"<\|user\|>(.*?)<\|assistant\|>", re.S)
MARKERS = ("<|user|>", "<|assistant|>", "<|endoftext|>")


def doc_views(text):
    """(user turns, the document's words as training text) for the decontamination rules: a chat document's user turns
    are its prompts; any other document has none. The markers are taken out, as tc_b1_filter.py takes them out."""
    users = [u.strip() for u in TURN.findall(text)]
    for m in MARKERS:
        text = text.replace(m, "\n")
    return users, text


def docfilter(repo, heldout, exclude, src, out, ledger_path):
    """Plain or rendered documents through tc_b1_filter.py's two rules (corpus/decontam.py against the frozen eval sets,
    Carry-5 and Carry-12; a word 13-gram shared with a held-out dialogue) and, with --exclude, never a document equal to
    one of that file. Rows in file order, nothing random."""
    sys.path.insert(0, os.path.join(repo, "corpus", "fetchers"))
    sys.path.insert(0, os.path.join(repo, "corpus"))
    sys.path.insert(0, HERE)
    import human_dialogue
    import tc_b1_filter as F
    dc, carry5_sha = human_dialogue.decontaminator(carry5=True, carry12=True)
    if not carry5_sha:
        raise SystemExit("Carry-5 is missing")
    held = F.heldout_grams(*heldout)
    banned = {key(d) for d in docs(exclude)} if exclude else set()
    n = kept = d_eval = d_held = d_equal = 0
    h = hashlib.sha256()
    with open(out + ".tmp", "wb") as g:
        for d in docs(src):
            if not d:
                continue
            n += 1
            if key(d) in banned:
                d_equal += 1
                continue
            users, text = doc_views(d.decode("utf-8"))
            hit, _ = dc.check(users, text)
            ws = F.words(text)
            hh = len(ws) >= F.N and not held.isdisjoint(F.grams(ws))
            d_eval += bool(hit)
            d_held += bool(hh)
            if hit or hh:
                continue
            kept += 1
            g.write(d + SEP)
            h.update(d + SEP)
    os.replace(out + ".tmp", out)
    led = {"recipe": "tools/colab_chat/base2_corpus.py docfilter", "input": os.path.basename(src), "input_sha256": sha256(src),
           "carry5_v1_sha256": carry5_sha, "carry12_v1_sha256": human_dialogue.carry12_sha(), "heldout_grams": len(held),
           "heldout": {os.path.basename(p): sha256(p) for p in heldout},
           "exclude": {os.path.basename(exclude): sha256(exclude)} if exclude else None,
           "documents_in": n, "documents_kept": kept, "dropped_eval_rules": d_eval, "dropped_heldout_13gram": d_held,
           "dropped_equal_excluded": d_equal, "bytes": os.path.getsize(out), "sha256": h.hexdigest()}
    jsave(ledger_path, led)
    return led


def ids_tokens(record, mix):
    """The token count corpus_ids.py printed for a mixed part's ids (ids_<arm>.json, one JSON line a part)."""
    stem = os.path.basename(mix)[:-len(".txt")]
    for ln in open(record, encoding="utf-8"):
        if ln.strip() and os.path.basename(json.loads(ln)["out"]) == f"{stem}.ids":
            return json.loads(ln)["tokens"]
    raise SystemExit(f"{record}: no line for {stem}.ids")


def share(mix, scenario, vocab, total=None):
    """Tokens of the scenario documents in a mixed part over the part's tokens, by the real tokeniser. The scenario
    documents are found by their bytes and encoded again, each on its own with its closing newline, which the whole-file
    encoding gives to the next document: the count can differ from the file's by a token a document. total: the part's
    token count from its ids; without it every other document is encoded too and the two counts are added."""
    sys.path.insert(0, os.path.dirname(HERE))
    import corpus_ids
    _, enc = corpus_ids.encoder(vocab)
    scn = {key(d) for d in docs(scenario)}
    tok = {True: 0, False: 0}
    n = {True: 0, False: 0}
    batch = []

    def flush():
        texts = [(t + SEP).decode("utf-8") for _, t in batch]
        for (is_s, _), ids in zip(batch, enc(texts)):
            tok[is_s] += len(ids)
        batch.clear()
    for d in docs(mix):
        is_s = key(d) in scn
        n[is_s] += 1
        if is_s or total is None:
            batch.append((is_s, d))
        if len(batch) >= 4096:
            flush()
    flush()
    by_doc = total is None
    total = tok[True] + tok[False] if by_doc else total
    return {"mix": os.path.basename(mix), "scenario_documents": n[True], "other_documents": n[False],
            "scenario_tokens": tok[True], "tokens": total, "tokens_from": "documents" if by_doc else "ids",
            "scenario_share_of_tokens": tok[True] / total}


def subset(n_ledger, s_ledger):
    """mix_pretrain.py takes each source's documents in one seeded order with one cursor, so with the same file and seed a
    smaller count is a prefix of a larger one."""
    bad = []
    if n_ledger["seed"] != s_ledger["seed"]:
        bad.append("seeds differ")
    for s, n in n_ledger["sources"].items():
        t = s_ledger["sources"].get(s)
        if t is None:
            bad.append(f"{s}: not in S")
        elif t["sha256"] != n["sha256"]:
            bad.append(f"{s}: the source files differ")
        elif t["documents_used"] > n["documents_used"]:
            bad.append(f"{s}: S used {t['documents_used']} documents, N {n['documents_used']}")
    return bad


def head_docs(src, dst, limit):
    n = 0
    with open(dst + ".tmp", "wb") as g:
        for d in docs(src):
            if n and n + len(d) + len(SEP) > limit:
                break
            g.write(d + SEP)
            n += len(d) + len(SEP)
    os.replace(dst + ".tmp", dst)
    return n


def store_has(config, branch, prefix):
    """True when the store holds the set's manifest. A set that is not there reads as absent; any other failure is tried
    again by hf_store.retry and then stands, so a network fault never reads as an unfinished stage for long."""
    sys.path.insert(0, HERE)
    import hf_store
    store = hf_store.open_store(config)

    def once():
        try:
            store.read_manifest(branch, prefix)
            return True
        except hf_store.VerifyError:
            raise
        except Exception as e:  # noqa: BLE001 the library's not-found types vary by version; by name, as hf_store does
            if isinstance(e, hf_store.StoreError) or "NotFound" in type(e).__name__:
                return False
            raise
    return hf_store.retry(once, tries=3, delay=float(os.environ.get("BASE2_STORE_RETRY_S", "20")),
                          what=f"manifest of {prefix}", token=store.b.token)


def loadprobe(spec_path, out, binary):
    """mtstudio on the real ids for a few steps: seconds to the first step and the peak resident memory of the run. The
    least MemAvailable seen is beside it, read every two seconds."""
    import resource
    import subprocess
    import threading
    import time
    os.makedirs(out, exist_ok=True)
    spec = jload(spec_path)
    spec["out_dir"] = os.path.join(out, "run")
    sp = os.path.join(out, "spec.json")
    jsave(sp, spec)
    low = [None]
    stop = threading.Event()

    def watch():
        while not stop.wait(2):
            try:
                for line in open("/proc/meminfo"):
                    if line.startswith("MemAvailable:"):
                        kb = int(line.split()[1])
                        low[0] = kb if low[0] is None else min(low[0], kb)
            except OSError:
                return
    threading.Thread(target=watch, daemon=True).start()
    t0 = time.time()
    with open(os.path.join(out, "run.log"), "w") as log:
        p = subprocess.Popen([binary, "run", sp], stdout=log, stderr=subprocess.STDOUT)
        first = None
        ev = os.path.join(spec["out_dir"], "events.jsonl")
        while p.poll() is None:
            if first is None and os.path.exists(ev):
                with open(ev) as f:
                    if any('"event": "step"' in ln or '"event":"step"' in ln for ln in f):
                        first = time.time() - t0
            time.sleep(1)
    stop.set()
    res = {"spec": os.path.basename(spec_path), "returncode": p.returncode, "seconds": round(time.time() - t0, 1),
           "seconds_to_first_step": None if first is None else round(first, 1),
           "peak_rss_mib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss // 1024,
           "least_mem_available_mib": None if low[0] is None else low[0] // 1024,
           "corpus_bytes": {k: os.path.getsize(v) for k, v in (("main", spec["data"]["corpus"]),
                                                                ("tail", spec["data"].get("anneal", {}).get("corpus")))
                            if v and os.path.exists(v)}}
    jsave(os.path.join(out, "loadprobe.json"), res)
    print(json.dumps(res))
    return 0 if p.returncode == 0 else 1


def scn_ledger(d, shards, text, recipe, seed, max_drop):
    """The scenario stream's record from its shards' filter ledgers. Refuses a stream that lost more than max_drop of its
    documents to the filter (the engine's author was blind to the eval sets, so a large loss is a finding to read, not
    to train through) or that is shorter than the mix will take."""
    tot = {k: 0 for k in ("documents_in", "documents_kept", "dropped_eval_rules", "dropped_heldout_13gram",
                          "dropped_equal_excluded")}
    per = []
    for k in range(shards):
        led = jload(os.path.join(d, f"shard_{k}.ledger.json"))
        for key_ in tot:
            tot[key_] += led[key_]
        per.append({"shard": k, "start_pair": k * recipe["shard_documents"] // 2, "sha256": led["sha256"],
                    "documents_kept": led["documents_kept"]})
    first = jload(os.path.join(d, "shard_0.ledger.json"))
    need = recipe["parts"]["main"]["scenario_bytes"] + recipe["parts"]["tail"]["scenario_bytes"]
    size = os.path.getsize(text)
    out = {"recipe": "generate.py --seed SEED --docs SHARD_DOCUMENTS --start START_PAIR per shard, then "
                     "base2_corpus.py docfilter; shards joined in order",
           "seed": seed, "shard_documents": recipe["shard_documents"], "shards": per, **tot,
           "dropped_share": 1 - tot["documents_kept"] / tot["documents_in"], "max_drop": max_drop,
           "carry5_v1_sha256": first["carry5_v1_sha256"], "carry12_v1_sha256": first["carry12_v1_sha256"],
           "heldout": first["heldout"], "exclude": first["exclude"],
           "bytes": size, "bytes_the_mix_takes": need, "sha256": sha256(text)}
    out["ok"] = out["dropped_share"] <= max_drop and size >= need * 1.005
    return out


def stage_ledger(files, jsons, jsonls, kvs):
    out = dict(k.split("=", 1) for k in kvs)
    out["bytes"] = {}
    for item in files:
        name, p = item.split("=", 1)
        out["bytes"][name] = os.path.getsize(p)
    for item in jsons:
        name, p = item.split("=", 1)
        out[name] = jload(p) if os.path.exists(p) else None
    for item in jsonls:
        name, p = item.split("=", 1)
        out[name] = [json.loads(ln) for ln in open(p, encoding="utf-8") if ln.strip()]
    return out


def kv(text):
    return {k: float(v) for k, v in (x.split("=", 1) for x in text.split(","))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("recipe-n")
    p.add_argument("--avail", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("recipe-s")
    for a in ("--recipe-n", "--n-ledger", "--n-ids", "--out"):
        p.add_argument(a, required=True)
    for a in ("--scn-bytes", "--scn-tokens", "--scn-docs", "--shard-docs"):
        p.add_argument(a, required=True, type=int)
    p.add_argument("--max-drop", type=float, default=0.02)
    p = sub.add_parser("get")
    p.add_argument("file")
    p.add_argument("key")
    p = sub.add_parser("docfilter")
    p.add_argument("--repo", required=True)
    p.add_argument("--heldout", nargs=2, required=True)
    p.add_argument("--exclude")
    p.add_argument("--in", dest="src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ledger", required=True)
    p = sub.add_parser("share")
    for a in ("--mix", "--scenario", "--vocab"):
        p.add_argument(a, required=True)
    p.add_argument("--ids-record", help="ids_<arm>.json: the part's token count comes from its ids")
    p.add_argument("--min", type=float, default=0.14)
    p.add_argument("--max", type=float, default=0.16)
    p.add_argument("--out")
    p = sub.add_parser("subset")
    p.add_argument("n")
    p.add_argument("s")
    p = sub.add_parser("head-docs")
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("bytes", type=int)
    p = sub.add_parser("store-has")
    p.add_argument("config")
    p.add_argument("branch")
    p.add_argument("prefix")
    p = sub.add_parser("scn-ledger")
    for a in ("--dir", "--text", "--recipe", "--out"):
        p.add_argument(a, required=True)
    p.add_argument("--shards", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--max-drop", type=float, required=True)
    p = sub.add_parser("stage-ledger")
    p.add_argument("--out", required=True)
    for a in ("--file", "--json", "--jsonl", "--kv"):
        p.add_argument(a, action="append", default=[])
    p = sub.add_parser("loadprobe")
    p.add_argument("--spec", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--bin", default=os.environ.get("JOBS_MTSTUDIO", "/content/pk/build_cuda/mtstudio"))
    a = ap.parse_args(argv)
    if a.cmd == "recipe-n":
        r = recipe_n(kv(a.avail))
        jsave(a.out, r)
        print(json.dumps({k: r[k] for k in ("total_bytes", "tail_fraction", "main_arg", "tail_arg")}))
    elif a.cmd == "recipe-s":
        ids = [json.loads(ln) for ln in open(a.n_ids) if ln.strip()]
        toks = {p: next(x["tokens"] for x in ids if f".{p}.ids" in x["out"]) for p in ("main", "tail")}
        r = recipe_s(jload(a.recipe_n), jload(a.n_ledger), toks, a.scn_bytes, a.scn_tokens, a.scn_docs, a.shard_docs,
                     a.max_drop)
        jsave(a.out, r)
        print(json.dumps({k: r[k] for k in ("total_bytes", "tail_fraction", "main_arg", "tail_arg", "scenario_shards",
                                            "scenario_bytes_per_token")}))
    elif a.cmd == "get":
        v = jload(a.file)
        for k in a.key.split("."):
            v = v[k]
        print(v if not isinstance(v, (dict, list)) else json.dumps(v))
    elif a.cmd == "docfilter":
        print(json.dumps(docfilter(a.repo, a.heldout, a.exclude, a.src, a.out, a.ledger)))
    elif a.cmd == "share":
        r = share(a.mix, a.scenario, a.vocab, ids_tokens(a.ids_record, a.mix) if a.ids_record else None)
        r["bounds"] = [a.min, a.max]
        r["ok"] = a.min <= r["scenario_share_of_tokens"] <= a.max
        if a.out:
            jsave(a.out, r)
        print(json.dumps(r))
        return 0 if r["ok"] else 1
    elif a.cmd == "subset":
        bad = subset(jload(a.n), jload(a.s))
        print(json.dumps({"subset": not bad, "differences": bad}))
        return 1 if bad else 0
    elif a.cmd == "head-docs":
        print(json.dumps({"bytes": head_docs(a.src, a.dst, a.bytes)}))
    elif a.cmd == "store-has":
        try:
            return 0 if store_has(a.config, a.branch, a.prefix) else 1
        except Exception as e:  # noqa: BLE001 the type alone: an error text could carry a request header
            print(f"store-has failed: {type(e).__name__}", file=sys.stderr)
            return 2
    elif a.cmd == "scn-ledger":
        r = scn_ledger(a.dir, a.shards, a.text, jload(a.recipe), a.seed, a.max_drop)
        jsave(a.out, r)
        print(json.dumps({k: r[k] for k in ("documents_in", "documents_kept", "dropped_share", "bytes", "ok")}))
        if not r["ok"]:
            os.replace(a.out, a.out + ".refused")
            return 1
    elif a.cmd == "stage-ledger":
        jsave(a.out, stage_ledger(a.file, a.json, a.jsonl, a.kv))
    elif a.cmd == "loadprobe":
        return loadprobe(a.spec, a.out, a.bin)
    return 0


if __name__ == "__main__":
    sys.exit(main())
