#!/usr/bin/env python3
"""ember.cpp's corpus draft table for a paperkiln chat model, written beside its GGUF (LOCAL_LLM_STACK.md
section 7, speculative decoding; opt-in, mtstudio's export.draft_table).

ember's EMBER_DRAFT=table drafter (ember bace837, draft_table.h; EMBER_SPEC=k with EMBER_DRAFT_TABLE=<file>)
drafts, where prompt lookup has none, from an n-gram table of the most frequent next token built from the
model's own chat training text: on an L4, chat decoding 1.91x (pick_e1_L16) and 2.23x (pk_base_v1) plain
decode, the replies plain decode's whatever the table holds (verification decides every token). Each model
needs its own table (another model's gives 10-19% less on chat), so the export can write it.

    python tools/draft_table.py build --gguf MODEL.gguf --out MODEL.draft_table.bin SOURCE... [--max-bytes N]
                                      [--orders 4] [--min-count 8] [--min-conf 0.10] [--prune-redundant 1]
    python tools/draft_table.py check TABLE.bin --gguf MODEL.gguf
    python tools/draft_table.py info TABLE.bin

build: the table from the SOURCEs in order, through the GGUF's own vocabulary (tools/llama3_bpe.py; a
  paperkiln "gpt2"/"llama-bpe" vocabulary, at most 65535 tokens), and a receipt <out>.json (settings,
  each source's sha256, bytes read and tokens, the token stream's sha256, the table's sha256 and entries).
  A SOURCE is chat-format text (.txt, as the corpus fetchers' text.txt: <|user|>...<|assistant|>...
  <|endoftext|>, one dialogue per document, a newline after each) or JSONL rows (.jsonl, the forms mtstudio
  trains on: {"messages"} rendered as one document by the GGUF's own chat template, chat_template.py
  render_document; the rendered {"segments"} rows, with <|endoftext|> after each trained segment as mtstudio
  appends it; {"prompt", "completion"} rows, the completion closed by <|endoftext|>; {"text"} documents),
  each rendered row a document ending in a newline. Text is read as Python reads a text file (universal
  newlines), whole lines, at most --max-bytes of each source (0: all), each line encoded on its own, the
  specials parsed out first, as ember encodes a prompt. The defaults are the settings ember measured
  (orders 1-4, contexts seen at least 8 times, top share at least 0.10, redundant back-off entries pruned:
  ember's h_c8_p10 and s_c8_p10). The table's vocabulary size is the model's, as ember reads it
  (tokenizer.ggml.vocab_size, else <arch>.vocab_size, else the IR's model.vocab, else the token count);
  values that disagree refuse.
check: the table as ember's loader reads it (draft_table.h DraftTable::load: magic, orders 1-4, the
  vocabulary size equal to the model's, sizes, ids below the vocabulary, keys strictly ascending, no
  trailing bytes); exit 1 with the reason otherwise.

The builder below is ember's experiments/draft_2026-10-27/dr_table.py (ember.cpp bace837, same author),
vendored: its tokenisation and table logic unchanged, so the bytes equal ember's builder on the same text
(tests/test_draft_table.py runs both on one corpus; ember's h_c8_p10.bin is reproduced byte for byte from
the 13 human dialogue sources, sha256 5a58236a...). The format is ember's loader's contract:

Keys (draft_table.h): t0 is the most recent token; for o <= 3 the key is t0 | t1 << 16 | t2 << 32; for o = 4
  it is mix64(t0 | t1 << 16 | t2 << 32 | t3 << 48) >> 16 (48 bits, splitmix64's finaliser; a collision
  only costs a wrong draft, which verification rejects).
File (little-endian): b"EMBERDT1", u32 vocab_size, u32 orders, u64 entries[orders]; then per order,
  ascending keys u64[n], next u16[n], conf u8[n] (round(255 p)), each array zero-padded to 8 bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

MAGIC = b"EMBERDT1"
DEFAULTS = {"orders": 4, "min_count": 8, "min_conf": 0.10, "prune_redundant": 1}
BUILDER = "ember.cpp experiments/draft_2026-10-27/dr_table.py (bace837), vendored in paperkiln tools/draft_table.py"
BATCH = 2000  # lines per encode_batch call (dr_table.py's; lines are encoded independently either way)


class Refused(Exception):
    pass


# --------------------------------------------------------------------------
# the model's vocabulary

def model_vocab(gguf):
    """(vocab size as ember reads it, tokenizer, chat template record or None, metadata)."""
    import chat_template
    import llama3_bpe

    meta = llama3_bpe.read_gguf_kv(gguf)
    tokens = meta.get("tokenizer.ggml.tokens") or []
    sizes = {}
    if "tokenizer.ggml.vocab_size" in meta:
        sizes["tokenizer.ggml.vocab_size"] = int(meta["tokenizer.ggml.vocab_size"])
    arch = meta.get("general.architecture", "")
    if f"{arch}.vocab_size" in meta:
        sizes[f"{arch}.vocab_size"] = int(meta[f"{arch}.vocab_size"])
    if "paperkiln.ir" in meta:
        v = (json.loads(meta["paperkiln.ir"]).get("model") or {}).get("vocab")
        if v is not None:
            sizes["paperkiln.ir model.vocab"] = int(v)
    if tokens:
        sizes["tokenizer.ggml.tokens"] = len(tokens)
    if not sizes:
        raise Refused(f"{gguf}: no vocabulary size (no tokenizer, no vocab_size key, no IR)")
    if len(set(sizes.values())) != 1:
        raise Refused(f"{gguf}: the vocabulary sizes disagree ({sizes}); a table needs the one ember loads")
    vocab = next(iter(sizes.values()))
    if vocab > 65535:
        raise Refused(f"{gguf}: vocabulary {vocab} does not fit the table's u16 ids")
    try:
        tok = llama3_bpe.from_gguf(gguf)
    except (ValueError, KeyError) as e:
        raise Refused(f"{gguf}: no paperkiln BPE tokenizer to encode the text with ({e})")
    tmpl, _ = chat_template.resolve(meta)
    return vocab, tok, tmpl, meta


# --------------------------------------------------------------------------
# sources -> lines (as dr_table.py reads text.txt) -> uint16 ids

def render_row(row, tmpl):
    """One JSONL training row as the text the model saw, a document ending in a newline."""
    import chat_template as CT

    if "messages" in row:
        if tmpl is None:
            raise Refused("a {\"messages\"} row, but the GGUF resolves to no chat template")
        return CT.render_document(row["messages"], tmpl)
    if "segments" in row:
        text = "".join(s["text"] + ("<|endoftext|>" if s.get("loss") else "") for s in row["segments"])
    elif "prompt" in row and "completion" in row:
        text = row["prompt"] + row["completion"] + "<|endoftext|>"
    elif "text" in row:
        text = row["text"]
    else:
        raise Refused(f"a row with none of messages, segments, prompt + completion, text: {sorted(row)}")
    return text if text.endswith("\n") else text + "\n"


def source_lines(path, tmpl):
    """The source's lines, as `for line in open(path, encoding="utf-8")` yields them (universal newlines)."""
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            for n, raw in enumerate(f, 1):
                if not raw.strip():
                    continue
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError as e:
                    raise Refused(f"{path}:{n}: not JSON ({e})")
                yield from io.StringIO(render_row(row, tmpl), newline=None)
    else:
        with open(path, encoding="utf-8") as f:
            yield from f


def tokenise(sources, tok, tmpl, max_bytes=0, log=print):
    """(ids uint16, per-source records, sha256 of the stream as little-endian u16): dr_table.py tokenize."""
    parts, recs = [], []
    h = hashlib.sha256()
    total = 0
    for path in sources:
        got, n_tok, batch, chunks = 0, 0, [], []
        for line in source_lines(path, tmpl):
            if max_bytes and got >= max_bytes:
                break
            got += len(line.encode("utf-8"))
            batch.append(line)
            if len(batch) >= BATCH:
                chunks.append(encode(tok, batch))
                batch = []
        if batch:
            chunks.append(encode(tok, batch))
        arr = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.uint16)
        h.update(arr.astype("<u2").tobytes())
        n_tok = int(arr.size)
        total += n_tok
        parts.append(arr)
        recs.append({"path": os.path.abspath(path), "form": "jsonl" if path.endswith(".jsonl") else "text",
                     "sha256": sha256_file(path), "bytes_read": got, "tokens": n_tok})
        log(json.dumps({"file": path, "bytes": got, "tokens_so_far": total}))
    ids = np.concatenate(parts) if parts else np.zeros(0, dtype=np.uint16)
    return ids, recs, h.hexdigest()


def encode(tok, lines):
    out = [np.array(e.ids, dtype=np.uint16) for e in tok.encode_batch(lines, add_special_tokens=False)]
    return np.concatenate(out) if out else np.zeros(0, dtype=np.uint16)


# --------------------------------------------------------------------------
# the table (dr_table.py build, unchanged)

M48 = np.uint64((1 << 48) - 1)


def mix64(x):
    x = x.copy()
    with np.errstate(over="ignore"):
        x ^= x >> np.uint64(30)
        x *= np.uint64(0xBF58476D1CE4E5B9)
        x ^= x >> np.uint64(27)
        x *= np.uint64(0x94D049BB133111EB)
        x ^= x >> np.uint64(31)
    return x


def context_keys(ids, o):
    """Key of the context ending at each position i >= o - 1 (ids[i] the most recent), for positions
    o-1 .. n-2 (each has a next token)."""
    n = len(ids)
    ids = ids.astype(np.uint64)
    end = n - 1  # contexts whose next token exists
    k = np.zeros(end - (o - 1), dtype=np.uint64)
    for j in range(o):
        k |= ids[o - 1 - j:end - j] << np.uint64(16 * j)
    if o == 4:
        k = mix64(k) >> np.uint64(16)
    return k


def key_of(ctx):
    """The key of one context (a list, most recent last), as draft_table.h computes it."""
    o = len(ctx)
    k = 0
    for j in range(o):
        k |= int(ctx[-1 - j]) << (16 * j)
    if o == 4:
        k = int(mix64(np.array([k], dtype=np.uint64))[0] >> np.uint64(16))
    return k


def build_tables(ids, vocab, orders=4, min_count=8, min_conf=0.10, prune_redundant=1, log=print):
    """For every order o = 1..orders and every context seen at least min_count times, the most frequent next
    token (ties: the smaller id) and its share p, kept when p >= min_conf and, with prune_redundant, when the
    next-lower order's lookup would not give the same token anyway."""
    if ids.size and int(ids.max()) >= vocab:
        raise Refused(f"token id {int(ids.max())} outside the vocabulary of {vocab}")
    if ids.size <= orders:
        raise Refused(f"{ids.size} tokens: too few for a table of order {orders}")
    tables = []
    for o in range(1, orders + 1):
        ck = context_keys(ids, o)
        nxt = ids[o:].astype(np.uint64)
        comb = (ck << np.uint64(16)) | nxt
        del ck, nxt
        comb.sort()
        vals, counts = np.unique(comb, return_counts=True)
        del comb
        ctx = vals >> np.uint64(16)
        tokn = (vals & np.uint64(0xFFFF)).astype(np.uint16)
        del vals
        # per context: the largest count (ties: the smaller next id, first in this ascending order)
        starts = np.flatnonzero(np.concatenate(([True], ctx[1:] != ctx[:-1])))
        tot = np.add.reduceat(counts, starts)
        order = np.lexsort((-counts.astype(np.int64), ctx))  # ctx ascending, count descending, stable in next
        first = order[starts]  # the same group boundaries: ctx ascending in both
        best_t = tokn[first]
        best_c = counts[first]
        keys = ctx[starts]
        del ctx, tokn, counts, order
        p = best_c / tot
        keep = (tot >= min_count) & (p >= min_conf)
        tables.append({"keys": keys[keep], "next": best_t[keep], "conf": np.round(p[keep] * 255).astype(np.uint8),
                       "contexts_seen": int(keys.size), "kept_before_prune": int(keep.sum())})
        log(json.dumps({"order": o, "contexts": int(keys.size), "kept": int(keep.sum())}))
    if prune_redundant:
        # an order-o entry whose context's lower-order lookup (with back-off) gives the same token is dropped
        for o in range(orders, 1, -1):
            t = tables[o - 1]
            pred = lower_lookup(tables, o - 1, t["keys"], o)
            keep = pred != t["next"].astype(np.int64)
            for f in ("keys", "next", "conf"):
                t[f] = t[f][keep]
            log(json.dumps({"order": o, "after_prune": int(keep.sum())}))
    return tables


def lower_lookup(tables, upto, keys_hi, o_hi):
    """For order-o_hi context keys, the back-off prediction from orders upto..1 (-1 when none). Exact for
    o_hi <= 3 (the lower keys are a prefix of the packing); an order-4 key is hashed and has lost its tokens,
    so order-4 entries are kept."""
    res = np.full(keys_hi.size, -1, dtype=np.int64)
    if o_hi == 4:
        return res
    for o in range(upto, 0, -1):
        mask = np.uint64((1 << (16 * o)) - 1)
        lk = keys_hi & mask
        t = tables[o - 1]
        idx = np.searchsorted(t["keys"], lk)
        idx = np.minimum(idx, max(t["keys"].size - 1, 0))
        hit = (t["keys"].size > 0) & (t["keys"][idx] == lk) & (res < 0)
        res[hit] = t["next"][idx[hit]].astype(np.int64)
    return res


def table_bytes(vocab, tables):
    out = bytearray(MAGIC)
    out += struct.pack("<II", vocab, len(tables))
    for t in tables:
        out += struct.pack("<Q", t["keys"].size)
    for t in tables:
        for arr, dt in ((t["keys"], "<u8"), (t["next"], "<u2"), (t["conf"], "u1")):
            b = arr.astype(dt).tobytes()
            out += b
            out += b"\0" * ((-len(b)) % 8)
    return bytes(out)


def read(path):
    """(vocab, tables) of a well-formed file (check() says why one is not)."""
    buf = open(path, "rb").read()
    why = load_problem(buf, None)
    if why:
        raise Refused(f"{path}: {why}")
    vocab, orders = struct.unpack_from("<II", buf, 8)
    ns = struct.unpack_from("<" + "Q" * orders, buf, 16)
    p = 16 + 8 * orders
    tables = []
    for n in ns:
        t = {}
        for name, dt, sz in (("keys", "<u8", 8), ("next", "<u2", 2), ("conf", "u1", 1)):
            t[name] = np.frombuffer(buf, dtype=dt, count=n, offset=p)
            p += n * sz + ((-(n * sz)) % 8)
        tables.append(t)
    return vocab, tables


def load_problem(buf, vocab_size):
    """Why ember's DraftTable::load (draft_table.h) refuses these bytes, or "" (vocab_size None: not compared)."""
    if len(buf) < 16 or buf[:8] != MAGIC:
        return "not a draft table"
    vocab, orders = struct.unpack_from("<II", buf, 8)
    if orders < 1 or orders > 4:
        return "orders must be 1..4"
    if vocab_size is not None and vocab != vocab_size:
        return f"built for vocabulary size {vocab}, the model has {vocab_size}"
    p = 16 + 8 * orders
    if len(buf) < p:
        return "truncated header"
    ns = struct.unpack_from("<" + "Q" * orders, buf, 16)
    pad8 = lambda b: (b + 7) // 8 * 8  # noqa: E731
    for n in ns:
        need = pad8(8 * n) + pad8(2 * n) + pad8(n)
        if len(buf) - p < need:
            return "truncated tables"
        keys = np.frombuffer(buf, dtype="<u8", count=n, offset=p)
        nxt = np.frombuffer(buf, dtype="<u2", count=n, offset=p + pad8(8 * n))
        p += need
        if n and (int(nxt.max()) >= vocab or (n > 1 and not bool(np.all(keys[1:] > keys[:-1])))):
            return "entries out of range or not ascending"
    if p != len(buf):
        return "trailing bytes"
    return ""


def draft(tables, history, k, theta=0.02):
    """draft_table.h DraftTable::draft: up to k tokens continuing history (the longest order whose context
    is in the table gives each; stop when the product of the shares falls below theta)."""
    no = len(tables)
    ctx = list(history[-no:])
    d, p = [], 1.0
    while len(d) < k:
        hit = None
        for o in range(min(no, len(ctx)), 0, -1):
            t = tables[o - 1]
            if not t["keys"].size:
                continue
            key = key_of(ctx[-o:])
            i = int(np.searchsorted(t["keys"], np.uint64(key)))
            if i < t["keys"].size and int(t["keys"][i]) == key:
                hit = (int(t["next"][i]), int(t["conf"][i]) / 255.0)
                break
        if hit is None:
            break
        p *= hit[1]
        if p < theta:
            break
        d.append(hit[0])
        ctx = (ctx + [hit[0]])[-no:]
    return d


# --------------------------------------------------------------------------
# commands

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_atomic(path, data):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def build(gguf, out, sources, max_bytes=0, orders=4, min_count=8, min_conf=0.10, prune_redundant=1, log=print):
    """Writes the table and its receipt <out>.json; returns the receipt."""
    if not sources:
        raise Refused("no source text: name the SFT data (or the dialogue text) the model was trained on")
    for s in sources:
        if not os.path.isfile(s):
            raise Refused(f"source {s} does not exist")
    if not 1 <= orders <= 4:
        raise Refused("--orders must be 1..4 (ember's loader)")
    vocab, tok, tmpl, _ = model_vocab(gguf)
    ids, recs, stream_sha = tokenise(sources, tok, tmpl, max_bytes, log)
    tables = build_tables(ids, vocab, orders, min_count, min_conf, prune_redundant, log)
    data = table_bytes(vocab, tables)
    why = load_problem(data, vocab)
    if why:
        raise Refused(f"the table built does not load in ember: {why}")
    import chat_template as CT
    receipt = {"table": os.path.basename(out), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
               "vocab": vocab, "entries": [int(t["keys"].size) for t in tables],
               "settings": {"orders": orders, "min_count": min_count, "min_conf": min_conf,
                            "prune_redundant": int(prune_redundant), "max_bytes": max_bytes},
               "gguf": os.path.abspath(gguf), "template": CT.rid(tmpl) if tmpl else None,
               "sources": recs, "corpus_tokens": int(ids.size), "corpus_sha256": stream_sha,
               "builder": BUILDER,
               "serve": "EMBER_SPEC=8 EMBER_DRAFT=table EMBER_DRAFT_TABLE=<this file> (ember bace837 or later)"}
    write_atomic(out, data)
    write_atomic(out + ".json", (json.dumps(receipt, indent=1) + "\n").encode("utf-8"))
    log(json.dumps({k: receipt[k] for k in ("table", "bytes", "vocab", "entries", "sha256", "corpus_tokens")}))
    return receipt


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("sources", nargs="+")
    b.add_argument("--gguf", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--max-bytes", type=int, default=0)
    b.add_argument("--orders", type=int, default=DEFAULTS["orders"])
    b.add_argument("--min-count", type=int, default=DEFAULTS["min_count"])
    b.add_argument("--min-conf", type=float, default=DEFAULTS["min_conf"])
    b.add_argument("--prune-redundant", type=int, default=DEFAULTS["prune_redundant"])
    c = sub.add_parser("check")
    c.add_argument("table")
    c.add_argument("--gguf", required=True)
    i = sub.add_parser("info")
    i.add_argument("table")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "build":
            build(a.gguf, a.out, a.sources, a.max_bytes, a.orders, a.min_count, a.min_conf, a.prune_redundant,
                  log=lambda s: print(s, flush=True))
        elif a.cmd == "check":
            vocab = model_vocab(a.gguf)[0]
            why = load_problem(open(a.table, "rb").read(), vocab)
            if why:
                print(f"REFUSED {a.table}: {why}")
                return 1
            print(f"ok {a.table}: loads in ember for {a.gguf} (vocabulary {vocab})")
        else:
            vocab, tables = read(a.table)
            print(json.dumps({"table": a.table, "bytes": os.path.getsize(a.table), "vocab": vocab,
                              "entries": [int(t["keys"].size) for t in tables], "sha256": sha256_file(a.table)}))
    except Refused as e:
        print(f"draft_table: REFUSED: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
