#!/usr/bin/env python3
"""Conversation schema v2's vocabulary and weights (LOCAL_LLM_STACK.md section 3, point 3g).

    python tools/chat_v2_vocab.py vocab IN.gguf OUT.gguf
    python tools/chat_v2_vocab.py extend IN.safetensors OUT.safetensors --rows 16384

`vocab` appends paperkiln/2's eight control tokens (tools/chat_template.py V2_CONTROL: <|system|>,
<|tool_call|>, <|tool|>, <|spare_3|> .. <|spare_7|>) after a llama.cpp-form vocabulary's last id, with
tokenizer.ggml.token_type 3 (control): merges, every other key and every existing id are unchanged, so
every .ids corpus stays valid. The 16k Llama 3-scheme vocabulary (16,384 entries) becomes 16,392, the new
ids 16384-16391. The input must be a vocabulary-only GGUF (no tensors) of tokenizer.ggml.model "gpt2".

`extend` appends rows to every tensor whose first dimension is the old vocabulary size (`--rows`, the
embedding; paperkiln's llama ties the head to it), each new row the mean of the old rows, so the new
tokens start at the average embedding and logit. Other tensors and the metadata are copied byte for byte.
Both write a new file (refusing to overwrite) and print the sha256 of input and output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import chat_template as CT  # noqa: E402

_SCAL = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_kv(buf):
    """[(key, type, raw value bytes)] and the tensor count of a GGUF; values are kept as their bytes
    so that untouched keys are rewritten exactly."""
    p = 0

    def rd(fmt):
        nonlocal p
        v = struct.unpack_from("<" + fmt, buf, p)[0]
        p += struct.calcsize("<" + fmt)
        return v

    def skip(t):
        nonlocal p
        if t in _SCAL:
            p += struct.calcsize("<" + _SCAL[t])
        elif t == 8:
            p += 8 + struct.unpack_from("<Q", buf, p)[0]
        elif t == 9:
            et, n = struct.unpack_from("<IQ", buf, p)
            p += 12
            for _ in range(n):
                skip(et)
        else:
            raise ValueError(f"GGUF type {t}")
    if rd("I") != 0x46554747:
        raise ValueError("not GGUF")
    ver, n_t, n_kv = rd("I"), rd("Q"), rd("Q")
    kv = []
    for _ in range(n_kv):
        n = rd("Q")
        key = buf[p:p + n].decode("utf-8")
        p += n
        t = rd("I")
        a = p
        skip(t)
        kv.append((key, t, buf[a:p]))
    return ver, n_t, kv, p


def _str(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def cmd_vocab(a):
    if os.path.exists(a.out):
        sys.exit(f"{a.out} exists")
    buf = open(a.inp, "rb").read()
    ver, n_t, kv, end = _read_kv(buf)
    if n_t:
        sys.exit("the input holds tensors: give a vocabulary-only GGUF")
    keys = {k: (t, raw) for k, t, raw in kv}
    model = keys.get("tokenizer.ggml.model")
    if not model or model[1][8:].decode() != "gpt2":
        sys.exit("vocab: a llama.cpp-form (tokenizer.ggml.model gpt2) vocabulary is needed: its control tokens "
                 "are parsed by token type; paperkiln's 8k 'bpe' vocabularies stay v1")
    t, raw = keys["tokenizer.ggml.tokens"]
    et, n = struct.unpack_from("<IQ", raw, 0)
    tt, traw = keys["tokenizer.ggml.token_type"]
    tet, tn = struct.unpack_from("<IQ", traw, 0)
    if et != 8 or tet not in (4, 5) or tn != n:
        sys.exit("unexpected tokens / token_type layout")
    # the existing strings, to refuse a vocabulary that already holds a control string
    q, toks = 12, []
    for _ in range(n):
        m = struct.unpack_from("<Q", raw, q)[0]
        toks.append(raw[q + 8:q + 8 + m].decode("utf-8", "replace"))
        q += 8 + m
    clash = [c for c in CT.V2_CONTROL if c in toks]
    if clash:
        sys.exit(f"the vocabulary already holds {clash}")
    new_tokens = struct.pack("<IQ", 8, n + len(CT.V2_CONTROL)) + raw[12:] + b"".join(_str(c) for c in CT.V2_CONTROL)
    new_types = struct.pack("<IQ", tet, n + len(CT.V2_CONTROL)) + traw[12:] + \
        struct.pack("<" + ("i" if tet == 5 else "I") * len(CT.V2_CONTROL), *([3] * len(CT.V2_CONTROL)))
    out = [struct.pack("<IIQQ", 0x46554747, ver, 0, len(kv))]
    for k, t, raw in kv:
        if k == "tokenizer.ggml.tokens":
            raw = new_tokens
        elif k == "tokenizer.ggml.token_type":
            raw = new_types
        out.append(_str(k) + struct.pack("<I", t) + raw)
    body = b"".join(out)
    # a vocabulary-only GGUF ends after its metadata and the zero padding to the 32-byte alignment
    if len(buf) - end >= 32 or buf[end:].strip(b"\0"):
        sys.exit(f"{len(buf) - end} bytes after the metadata: not a vocabulary-only GGUF")
    body += b"\0" * ((32 - len(body) % 32) % 32)
    with open(a.out + ".tmp", "wb") as f:
        f.write(body)
    os.replace(a.out + ".tmp", a.out)
    print(json.dumps({"in": a.inp, "in_sha256": sha256(a.inp), "in_tokens": n, "out": a.out,
                      "out_sha256": sha256(a.out), "out_tokens": n + len(CT.V2_CONTROL),
                      "control_ids": {c: n + i for i, c in enumerate(CT.V2_CONTROL)}}, indent=1))


def cmd_extend(a):
    import numpy as np
    if os.path.exists(a.out):
        sys.exit(f"{a.out} exists")
    with open(a.inp, "rb") as f:
        hn = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hn))
        data = f.read()
    k_new = len(CT.V2_CONTROL)
    items = sorted(((k, v) for k, v in header.items() if k != "__metadata__"), key=lambda kv: kv[1]["data_offsets"][0])
    out_header = {"__metadata__": header["__metadata__"]} if "__metadata__" in header else {}
    chunks, off, grown = [], 0, []
    for k, v in items:
        b0, b1 = v["data_offsets"]
        raw = data[b0:b1]
        shape = list(v["shape"])
        if shape and shape[0] == a.rows:
            if v["dtype"] != "F32":
                sys.exit(f"{k}: dtype {v['dtype']} (F32 expected)")
            w = np.frombuffer(raw, dtype="<f4").reshape(shape)
            mean = w.astype(np.float64).mean(axis=0).astype("<f4")
            raw = raw + np.tile(mean, (k_new,) + (1,) * (len(shape) - 1)).tobytes()
            shape[0] += k_new
            grown.append(k)
        out_header[k] = {"dtype": v["dtype"], "shape": shape, "data_offsets": [off, off + len(raw)]}
        chunks.append(raw)
        off += len(raw)
    if not grown:
        sys.exit(f"no tensor has first dimension {a.rows}")
    hj = json.dumps(out_header, separators=(",", ":")).encode()
    hj += b" " * ((8 - len(hj) % 8) % 8)
    with open(a.out + ".tmp", "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for c in chunks:
            f.write(c)
    os.replace(a.out + ".tmp", a.out)
    print(json.dumps({"in": a.inp, "in_sha256": sha256(a.inp), "out": a.out, "out_sha256": sha256(a.out),
                      "grown": grown, "rows": [a.rows, a.rows + k_new], "init": "mean of the old rows"}, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("vocab")
    v.add_argument("inp")
    v.add_argument("out")
    e = sub.add_parser("extend")
    e.add_argument("inp")
    e.add_argument("out")
    e.add_argument("--rows", type=int, required=True, help="the old vocabulary size (16384 for the 16k vocabulary)")
    a = ap.parse_args(argv)
    {"vocab": cmd_vocab, "extend": cmd_extend}[a.cmd](a)


if __name__ == "__main__":
    main()
