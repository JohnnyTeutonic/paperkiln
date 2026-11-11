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
    base2_corpus.py store-verify CONFIG BRANCH PREFIX FILE...   # the store's manifest lists these local files' bytes
    base2_corpus.py bind OUT --input F... [--key K=V...]  # record what OUT was built from (OUT.inputs.json)
    base2_corpus.py bound OUT --input F... [--key K=V...] # exit 0 when OUT was built from exactly these; else OUT goes
    base2_corpus.py doc-keys OUT FILE... / subset-docs --n-keys K --scenario F --out J S...   # S within N, by document
    base2_corpus.py token-report --n-ids A --s-ids B --out J   # N's and S's token counts by part
    base2_corpus.py split-docs IN A B BYTES               # whole documents: the first BYTES to A, the rest to B
    base2_corpus.py drill-watch --chat DIR --job J --out DIR   # the rehearsal: training step against pushed step, on the VM only
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
    """Tokens of the scenario documents in a mixed part over the part's tokens, by the real tokeniser, with no
    boundary error. corpus_ids.py cuts the file right after each special token and encodes the pieces one by one, so
    the newline that closes a document is encoded with the start of the next one. Each scenario document is therefore
    encoded here as the file encodes it: the newline before it (none for the file's first document), its text, and its
    end-of-text marker. Summed over every document, plus the file's last newline, these counts are the ids file's count
    exactly. The one convention is that a document owns the newline before it and not the one after: where a scenario
    document follows another kind, or the reverse, that is one newline's tokens either way, and
    `convention_bound` is the largest share that could move (one token a scenario document).
    total: the part's token count from its ids. Without it every document is encoded and the sum is checked against
    nothing; `tokens` is then the sum, which a caller can compare with the ids' count."""
    sys.path.insert(0, os.path.dirname(HERE))
    import corpus_ids
    _, enc = corpus_ids.encoder(vocab)
    scn = {key(d) for d in docs(scenario)}
    tok = {True: 0, False: 0}
    n = {True: 0, False: 0}
    batch = []
    eot = SEP[:-1]

    def flush():
        for (is_s, _), ids in zip(batch, enc([t for _, t in batch])):
            tok[is_s] += len(ids)
        batch.clear()
    first = True
    for d in docs(mix):
        is_s = key(d) in scn
        n[is_s] += 1
        if is_s or total is None:
            batch.append((is_s, ((b"" if first else b"\n") + d + eot).decode("utf-8")))
        first = False
        if len(batch) >= 4096:
            flush()
    flush()
    by_doc = total is None
    if by_doc:
        last = len(enc(["\n"])[0]) if not first else 0     # the newline after the file's last document
        total = tok[True] + tok[False] + last
    return {"mix": os.path.basename(mix), "scenario_documents": n[True], "other_documents": n[False],
            "scenario_tokens": tok[True], "tokens": total, "tokens_from": "documents" if by_doc else "ids",
            "scenario_share_of_tokens": tok[True] / total, "boundary_error_tokens": 0,
            "convention": "a document owns the newline before it", "convention_bound": n[True] / total}


def token_report(n_ids, s_ids, max_rel):
    """N's and S's token counts by part, from their ids' own records, and how far S is from N's count."""
    def read(p):
        return {os.path.basename(x["out"]).split(".")[1]: x["tokens"] for x in
                (json.loads(ln) for ln in open(p, encoding="utf-8") if ln.strip())}
    n, s = read(n_ids), read(s_ids)
    out = {"N": n, "S": s, "N_total": sum(n.values()), "S_total": sum(s.values()), "max_relative_difference": max_rel,
           "relative_difference": {p: (s[p] - n[p]) / n[p] for p in ("main", "tail")}}
    out["ok"] = all(abs(v) <= max_rel for v in out["relative_difference"].values())
    return out


# ---------------------------------------------------------------- cached outputs and their inputs
def file_sha(path):
    """sha256 of a file, remembered in BASE2_HASH_CACHE by path, size and modification time, so the build's many
    checks of one 14 GB file read it once."""
    cache_path = os.environ.get("BASE2_HASH_CACHE")
    st = os.stat(path)
    k = f"{os.path.abspath(path)}|{st.st_size}|{st.st_mtime_ns}"
    cache = {}
    if cache_path and os.path.exists(cache_path):
        try:
            cache = jload(cache_path)
        except ValueError:
            cache = {}
    if k not in cache:
        cache[k] = sha256(path)
        if cache_path:
            try:
                fresh = jload(cache_path) if os.path.exists(cache_path) else {}
            except ValueError:
                fresh = {}
            fresh[k] = cache[k]
            tmp = f"{cache_path}.{os.getpid()}.tmp"
            with open(tmp, "w") as f:
                json.dump(fresh, f)
            os.replace(tmp, cache_path)
    return cache[k]


def binding(inputs, keys):
    out = {"inputs": {}, "keys": dict(k.split("=", 1) for k in keys)}
    for p in inputs:
        out["inputs"][os.path.basename(p)] = file_sha(p) if os.path.isfile(p) else None
    return out


def bind(out, inputs, keys):
    """Record beside OUT the sha256 of every input it was built from (OUT.inputs.json)."""
    b = binding(inputs, keys)
    missing = [k for k, v in b["inputs"].items() if v is None]
    if missing:
        raise SystemExit(f"bind {out}: inputs missing: {missing}")
    jsave(out + ".inputs.json", b)


def bound(out, inputs, keys):
    """True when OUT is there and was built from exactly these inputs. Otherwise OUT and its record go, so a stale
    recipe, ledger or share file is never read again."""
    side = out + ".inputs.json"
    ok = os.path.isfile(out) and os.path.getsize(out) > 0 and os.path.isfile(side)
    if ok:
        try:
            ok = jload(side) == binding(inputs, keys)
        except ValueError:
            ok = False
    if not ok:
        for p in (out, side):
            if os.path.exists(p):
                os.remove(p)
    return ok


def split_docs(src, a, b, limit):
    """Whole documents of SRC: the first LIMIT bytes' worth to A, the rest to B."""
    n = m = 0
    with open(a + ".tmp", "wb") as fa, open(b + ".tmp", "wb") as fb:
        for d in docs(src):
            if m == 0 and (n == 0 or n + len(d) + len(SEP) <= limit):
                fa.write(d + SEP)
                n += len(d) + len(SEP)
            else:
                fb.write(d + SEP)
                m += len(d) + len(SEP)
    if not n or not m:
        raise SystemExit(f"split-docs: {src} gives an empty part at {limit} bytes")
    os.replace(a + ".tmp", a)
    os.replace(b + ".tmp", b)
    return n, m


def doc_keys(paths, out):
    """How many times each document occurs in the files, by a 12-byte digest of its bytes: sorted records of the digest
    and a 4-byte count. N's own documents, kept so that S's can be compared with them after N's text is gone."""
    import collections
    import struct
    c = collections.Counter()
    for p in paths:
        for d in docs(p):
            c[key(d)] += 1
    with open(out + ".tmp", "wb") as f:
        for k in sorted(c):
            f.write(k + struct.pack("<I", c[k]))
    os.replace(out + ".tmp", out)
    return {"documents": sum(c.values()), "distinct": len(c)}


def subset_docs(n_keys, scenario, s_paths):
    """Document by document: every document of S that is not a scenario document occurs in N at least as often."""
    import collections
    import struct
    n = {}
    with open(n_keys, "rb") as f:
        while True:
            rec = f.read(16)
            if not rec:
                break
            n[rec[:12]] = struct.unpack("<I", rec[12:])[0]
    scn = {key(d) for d in docs(scenario)}
    s = collections.Counter()
    n_scn = 0
    for p in s_paths:
        for d in docs(p):
            k = key(d)
            if k in scn and k not in n:
                n_scn += 1
            else:
                s[k] += 1
    over = sum(1 for k, v in s.items() if v > n.get(k, 0))
    return {"natural_documents_in_S": sum(s.values()), "scenario_documents_in_S": n_scn, "documents_in_N": sum(n.values()),
            "documents_of_S_not_in_N": over, "subset": over == 0}


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


ABSENT = {"EntryNotFoundError", "RevisionNotFoundError"}    # the file, or the branch, is not in a repo that answered
NOT_ABSENT = {"RepositoryNotFoundError", "LocalEntryNotFoundError", "GatedRepoError"}


def reads_as_absent(e, missing):
    """Only a missing file or a missing branch means "not in the store". `missing` is hf_store.Missing: what both
    backends raise for exactly that, the Hugging Face one after the hub's own entry-not-found or revision-not-found
    (HfBackend.fetch), the local one for a path that is not on disk. The library's two errors are read by name as
    well, for a caller that hands them over untranslated. "Repository not found" is what a wrong or expired token
    gets, and a file missing from the local cache is what an outage gets: both are an unreadable store, and a
    multi-hour stage must not run again because of either. Along a chain of causes the first decisive error wins."""
    seen = set()
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        names = {c.__name__ for c in type(e).__mro__}
        if names & NOT_ABSENT:
            return False
        if isinstance(e, missing) or names & ABSENT:
            return True
        e = e.__cause__ or (None if e.__suppress_context__ else e.__context__)
    return False


def store_limit():
    """Seconds one small store call of the data job may take (BASE2_STORE_LIMIT_S); the call runs in a child that
    is ended at the limit, so a request that hangs is a failed try, not a job that never ends."""
    return float(os.environ.get("BASE2_STORE_LIMIT_S", "300"))


def store_has(config, branch, prefix):
    """True when the store holds the set's manifest, False when the store answered and the set (or its branch) is not
    there. Any other failure is tried again by hf_store.retry and then raised. Each try has a wall-time limit."""
    sys.path.insert(0, HERE)
    import hf_store
    store = hf_store.open_store(config, tries=1)  # hf_store.retry below does the trying again

    def read():
        try:
            store.read_manifest(branch, prefix)
            return True
        except hf_store.VerifyError:
            raise
        except Exception as e:  # noqa: BLE001 the library's types vary by version; by name, as hf_store does
            if reads_as_absent(e, hf_store.Missing):
                return False
            raise

    def once():
        return hf_store.bounded(read, store_limit(), store.b.token)
    return hf_store.retry(once, tries=3, delay=float(os.environ.get("BASE2_STORE_RETRY_S", "20")),
                          what=f"manifest of {prefix}", token=store.b.token)


def store_verify(config, branch, prefix, files):
    """A push is published when the store's manifest for the set lists every one of these local files at its size and
    sha256. Read back from the store with a fresh handle, after the push. Each try has a wall-time limit."""
    sys.path.insert(0, HERE)
    import hf_store
    store = hf_store.open_store(config, tries=1)
    man = hf_store.retry(lambda: hf_store.bounded(lambda: store.read_manifest(branch, prefix), store_limit(), store.b.token),
                         tries=3, delay=float(os.environ.get("BASE2_STORE_RETRY_S", "20")), what=f"manifest of {prefix}",
                         token=store.b.token)
    bad = []
    for p in files:
        want = man["files"].get(os.path.basename(p))
        if want is None:
            bad.append(f"{os.path.basename(p)}: not in the store's manifest")
        elif want["bytes"] != os.path.getsize(p) or want["sha256"] != file_sha(p):
            bad.append(f"{os.path.basename(p)}: the store's copy differs")
    return bad, {f: man["files"][f] for f in sorted(man["files"])}


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

        def stepped():
            if not os.path.exists(ev):
                return False
            with open(ev) as f:
                return any('"event": "step"' in ln or '"event":"step"' in ln for ln in f)
        while p.poll() is None:
            if first is None and stepped():
                first = time.time() - t0
            time.sleep(1)
        if first is None and stepped():   # a run that ended inside one second of polling
            first = time.time() - t0
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


def training_step(events, tail=1 << 18):
    """The trainer's current training step: the step of the last `step` event in its own event stream (events.jsonl),
    read from the file's last `tail` bytes. With it, the file's modification time. (None, None) when there is none.
    This is where the trainer is, not what it has saved: a saved generation is `live`, a pushed one the courier's."""
    try:
        size = os.path.getsize(events)
        with open(events, "rb") as f:
            f.seek(max(0, size - tail))
            lines = f.read().split(b"\n")
        at = os.path.getmtime(events)
    except OSError:
        return None, None
    for ln in reversed(lines):
        if b'"step"' not in ln:
            continue
        try:
            e = json.loads(ln)
        except ValueError:
            continue
        if isinstance(e, dict) and e.get("event") == "step" and isinstance(e.get("step"), int):
            return e["step"], at
    return None, None


def watch_sample(chat, job):
    """The trainer's training step (its last `step` event), the step of the generation it last saved (`live`), the
    flat state.txt's step, and the courier's pushed step."""
    import time
    d = os.path.join(chat, job)
    try:
        live = int(os.path.basename(os.readlink(os.path.join(d, "ckpt_gen", "live")))[1:])
    except (OSError, ValueError):
        live = None
    try:
        state = int(open(os.path.join(d, "state.txt")).readline().strip() or 0)
    except (OSError, ValueError):
        state = None
    try:
        cs = jload(os.path.join(chat, "courier_status.json"))
        courier = (cs.get("jobs") or {}).get(job)
    except (OSError, ValueError):
        courier = None
    pushed = courier.get("pushed_step") if isinstance(courier, dict) else None
    train, train_at = training_step(os.path.join(d, "events.jsonl"))
    return {"job": job, "at": time.time(), "training_step": train, "training_step_at": train_at,
            "live_step": live, "state_step": state, "pushed_step": pushed,
            "training_minus_pushed": None if train is None or pushed is None else train - pushed,
            "live_minus_pushed": None if live is None or pushed is None else live - pushed, "courier": courier}


def drill_watch(chat, job, out, interval, limit_s):
    """The rehearsal's record of the trainer against the courier: every `interval` seconds the trainer's training
    step, its saved step and the courier's pushed step, appended to OUT/watch.jsonl on the VM. Nothing goes to the
    store: the hub limits commits per repository, and a watcher that committed every 30 seconds used the whole
    limit by itself (the rehearsal of 12 Nov 2026). The runner brings watch.jsonl home with the job's outputs when
    the job ends; a stopped VM takes its copy along, and what a forced stop costs is read by the runner from the VM
    just before the stop (runner_multi.py store_drill_evidence). Ends when the job ends or after `limit_s`."""
    import time
    os.makedirs(out, exist_ok=True)
    boot = int(time.time())
    t0, n = time.time(), 0
    while True:
        try:
            status = open(os.path.join(chat, f"STATUS_{job}")).read().strip()
        except OSError:
            status = "NONE"
        rec = {**watch_sample(chat, job), "status": status, "vm_started": boot, "sample": n}
        with open(os.path.join(out, "watch.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        n += 1
        if status in ("FINISHED", "FAILED") or time.time() - t0 > limit_s:
            break
        time.sleep(interval)
    jsave(os.path.join(out, "watch_summary.json"), {"job": job, "vm_started": boot, "samples": n, "store_commits": 0,
                                                    "last": rec})
    return 0


def scn_ledger(d, shards, text, recipe, seed, max_drop, validate=False):
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
    # The engine's own validator, on every shard before its filter (scn_shard in base2_data.sh).
    if validate:
        verdicts = []
        for k in range(shards):
            p = os.path.join(d, f"validate_{k}.json")
            verdicts.append(jload(p).get("verdict") if os.path.exists(p) else None)
        out["validator"] = {"shards": shards, "verdicts": verdicts, "passed": all(v == "PASS" for v in verdicts)}
    else:
        out["validator"] = {"shards": 0, "verdicts": [], "passed": None, "note": "switched off (BASE2_SCN_VALIDATE=0)"}
    out["ok"] = (out["dropped_share"] <= max_drop and size >= need * 1.005
                 and out["validator"]["passed"] is not False)
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
    p.add_argument("--validated", action="store_true", help="every shard must hold the validator's PASS")
    p = sub.add_parser("store-verify")
    p.add_argument("config")
    p.add_argument("branch")
    p.add_argument("prefix")
    p.add_argument("files", nargs="+")
    p.add_argument("--out")
    for name in ("bind", "bound"):
        p = sub.add_parser(name)
        p.add_argument("out")
        p.add_argument("--input", action="append", default=[])
        p.add_argument("--key", action="append", default=[])
    p = sub.add_parser("split-docs")
    p.add_argument("src")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("bytes", type=int)
    p = sub.add_parser("doc-keys")
    p.add_argument("out")
    p.add_argument("files", nargs="+")
    p = sub.add_parser("subset-docs")
    p.add_argument("--n-keys", required=True)
    p.add_argument("--scenario", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("s", nargs="+")
    p = sub.add_parser("drill-watch")
    for a in ("--chat", "--job", "--out"):
        p.add_argument(a, required=True)
    p.add_argument("--interval", type=float, default=30.0)
    p.add_argument("--limit-s", type=float, default=4 * 3600.0)
    # A jobs file written before 12 Nov 2026 still passes these two; the watcher no longer opens the store.
    p.add_argument("--config", default=None, help="ignored: the watcher writes on the VM only")
    p.add_argument("--push-limit-s", type=float, default=None, help="ignored: nothing is pushed")
    p = sub.add_parser("token-report")
    for a in ("--n-ids", "--s-ids", "--out"):
        p.add_argument(a, required=True)
    p.add_argument("--max-rel", type=float, default=0.01)
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
        r = scn_ledger(a.dir, a.shards, a.text, jload(a.recipe), a.seed, a.max_drop, a.validated)
        jsave(a.out, r)
        print(json.dumps({k: r[k] for k in ("documents_in", "documents_kept", "dropped_share", "bytes", "ok")}))
        if not r["ok"]:
            os.replace(a.out, a.out + ".refused")
            return 1
    elif a.cmd == "store-verify":
        try:
            bad, files = store_verify(a.config, a.branch, a.prefix, a.files)
        except Exception as e:  # noqa: BLE001 the type alone, as store-has
            print(f"store-verify failed: {type(e).__name__}", file=sys.stderr)
            return 2
        if a.out:
            jsave(a.out, {"branch": a.branch, "prefix": a.prefix, "verified": not bad, "differences": bad, "files": files})
        if bad:
            print("\n".join(bad), file=sys.stderr)
        return 1 if bad else 0
    elif a.cmd == "bind":
        bind(a.out, a.input, a.key)
    elif a.cmd == "bound":
        return 0 if bound(a.out, a.input, a.key) else 1
    elif a.cmd == "split-docs":
        print(json.dumps({"bytes": split_docs(a.src, a.a, a.b, a.bytes)}))
    elif a.cmd == "doc-keys":
        print(json.dumps(doc_keys(a.files, a.out)))
    elif a.cmd == "subset-docs":
        r = subset_docs(a.n_keys, a.scenario, a.s)
        jsave(a.out, r)
        print(json.dumps(r))
        return 0 if r["subset"] else 1
    elif a.cmd == "drill-watch":
        return drill_watch(a.chat, a.job, a.out, a.interval, a.limit_s)
    elif a.cmd == "token-report":
        r = token_report(a.n_ids, a.s_ids, a.max_rel)
        jsave(a.out, r)
        print(json.dumps(r))
        return 0 if r["ok"] else 1
    elif a.cmd == "stage-ledger":
        jsave(a.out, stage_ledger(a.file, a.json, a.jsonl, a.kv))
    elif a.cmd == "loadprobe":
        return loadprobe(a.spec, a.out, a.bin)
    return 0


if __name__ == "__main__":
    sys.exit(main())
