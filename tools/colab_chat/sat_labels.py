#!/usr/bin/env python3
"""SATL v1, the label file of SAT ("symbols as targets"; the registration is
AI_ML/chat_screens_claude/base2/token_ladder/symbols_as_targets/REGISTRATION.md in the author's research tree).
Claude Opus 5.5, 26 Nov 2026. The trainer's reader (tools/mtstudio.cpp, read_sat_labels) reads the same bytes.

Layout, every integer little-endian:
    8-byte magic b"SATL\\x01\\x00\\x00\\x00"; u32 header length; a UTF-8 JSON header
        {"corpus_sha256", "corpus_tokens", "roles", "spans", "pointer_targets"};
    then the spans, sorted by raw offset and non-overlapping, each:
        u64 raw_offset, u32 length, 32 bytes (sha256 of the span's raw token ids as little-endian u32),
        `length` code bytes, u32 n_ptr, n_ptr pairs (u32 pos_in_span, u32 back).
    The gold entity's raw position is raw_offset + pos_in_span - back, 1 <= back <= pos_in_span, and it must be a mention.
    A code byte: bits 0-2 the role id, bit 3 labelled (the role loss applies), bit 4 a mention (a pointer candidate),
    bits 5-7 zero.
"Raw" positions index the corpus as the trainer's read_corpus returns it, before the state-token insertion.

A span is a dict {"raw_offset": int, "codes": bytes, "pointers": [(pos_in_span, back), ...]}, and on reading also
"sha256": bytes. write() computes each span's sha256 from the corpus ids it is given.
"""
import hashlib
import json
import struct

MAGIC = b"SATL\x01\x00\x00\x00"
ROLES = ["O", "SUBJ", "VERB", "OBJ", "IOBJ", "OBL", "VALUE", "OWNER"]
LABELLED, MENTION = 8, 16


def code(role, labelled=True, mention=False):
    """One code byte: a role id or name, the labelled bit and the mention bit."""
    r = ROLES.index(role) if isinstance(role, str) else int(role)
    if not 0 <= r < 8:
        raise ValueError(f"role {role!r} is not an SATL v1 role")
    return r | (LABELLED if labelled else 0) | (MENTION if mention else 0)


def ids_sha256(ids):
    """sha256 of token ids as little-endian u32, the hash a span carries."""
    ids = [int(i) for i in ids]
    return hashlib.sha256(struct.pack(f"<{len(ids)}I", *ids)).digest()


def _check(spans):
    prev_end = None
    n_ptr = 0
    for s in spans:
        off, codes, ptrs = int(s["raw_offset"]), bytes(s["codes"]), list(s.get("pointers", []))
        if not codes:
            raise ValueError(f"an empty span at raw offset {off}")
        if prev_end is not None and off < prev_end:
            raise ValueError(f"spans unsorted or overlapping at raw offset {off}")
        prev_end = off + len(codes)
        for c in codes:
            if c & 0xE0:
                raise ValueError(f"a code byte with bits 5-7 set in the span at {off}")
        for pos, back in ptrs:
            if not (0 <= pos < len(codes) and 1 <= back <= pos):
                raise ValueError(f"a pointer outside its span at raw offset {off + pos}")
            if not codes[pos - back] & MENTION:
                raise ValueError(f"a pointer whose gold is not a mention at raw offset {off + pos}")
        n_ptr += len(ptrs)
    return n_ptr


def write(path, spans, corpus_ids, corpus_sha256):
    """Write spans keyed to corpus_ids (the raw ids; each span's hash is taken from them)."""
    spans = sorted(spans, key=lambda s: int(s["raw_offset"]))
    n_ptr = _check(spans)
    header = {"corpus_sha256": corpus_sha256, "corpus_tokens": len(corpus_ids), "roles": ROLES,
              "spans": len(spans), "pointer_targets": n_ptr}
    h = json.dumps(header, sort_keys=True).encode("utf-8")
    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<I", len(h)) + h)
        for s in spans:
            off, codes, ptrs = int(s["raw_offset"]), bytes(s["codes"]), list(s.get("pointers", []))
            if off + len(codes) > len(corpus_ids):
                raise ValueError(f"the span at {off} runs past the corpus")
            sha = s.get("sha256") if s.get("force_sha256") else ids_sha256(corpus_ids[off:off + len(codes)])
            f.write(struct.pack("<QI", off, len(codes)) + sha + codes + struct.pack("<I", len(ptrs)))
            for pos, back in ptrs:
                f.write(struct.pack("<II", pos, back))
    return header


def read(path):
    """(header, spans) of an SATL v1 file, checked as the trainer checks it."""
    with open(path, "rb") as f:
        b = f.read()
    if b[:8] != MAGIC:
        raise ValueError(f"{path}: not an SATL v1 file")
    (hl,) = struct.unpack_from("<I", b, 8)
    header = json.loads(b[12:12 + hl].decode("utf-8"))
    for k in ("corpus_sha256", "corpus_tokens", "roles", "spans", "pointer_targets"):
        if k not in header:
            raise ValueError(f"{path}: the header has no {k}")
    if header["roles"] != ROLES:
        raise ValueError(f"{path}: the header's roles are not SATL v1's")
    p = 12 + hl
    spans = []
    for _ in range(header["spans"]):
        off, n = struct.unpack_from("<QI", b, p)
        p += 12
        sha = b[p:p + 32]
        p += 32
        codes = b[p:p + n]
        p += n
        (np_,) = struct.unpack_from("<I", b, p)
        p += 4
        ptrs = [struct.unpack_from("<II", b, p + 8 * i) for i in range(np_)]
        p += 8 * np_
        if len(codes) != n or len(sha) != 32:
            raise ValueError(f"{path}: truncated")
        spans.append({"raw_offset": off, "codes": codes, "pointers": [tuple(x) for x in ptrs], "sha256": sha})
    if p != len(b):
        raise ValueError(f"{path}: trailing bytes after the last span")
    if _check(spans) != header["pointer_targets"]:
        raise ValueError(f"{path}: the header's pointer_targets differs from the spans'")
    return header, spans


def verify(spans, corpus_ids):
    """The spans whose sha256 differs from corpus_ids at their offset (the trainer refuses the run on any)."""
    return [s["raw_offset"] for s in spans
            if ids_sha256(corpus_ids[s["raw_offset"]:s["raw_offset"] + len(s["codes"])]) != s["sha256"]]
