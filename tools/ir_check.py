"""Checks for the paperkiln IR embedded in a GGUF (microtorch/ir.hpp).

    python tools/ir_check.py model.gguf                    names + sidecar checks
    python tools/ir_check.py model.gguf --like ir/chat_sftfinal.ir.json
    python tools/ir_check.py model.gguf --tensors-as base.gguf

Without options: the GGUF carries paperkiln.ir (string) and
paperkiln.ir_version (uint32) agreeing with the document, every weight the
IR names ("{i}" expanded over the blocks) is a tensor of the file (a
missing linear weight passes only through its "tied_to"), and a sidecar
<stem>.ir.json, when present, is the same document. --like: the embedded IR
is semantically equal to a reference (key order and "_"-prefixed prose keys
ignored). --tensors-as: tensor infos and the tensor-data region are byte
for byte those of another GGUF (an export without the IR, say).
Standard library only. Exit status 1 on any failure.
"""
import json
from pathlib import Path
import struct
import sys

_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def read_gguf(path):
    """(metadata {key: (type, value)}, tensors [(name, dims, type, offset)], data_start, raw bytes)."""
    b = Path(path).read_bytes()
    pos = 0

    def take(fmt):
        nonlocal pos
        v = struct.unpack_from(fmt, b, pos)
        pos += struct.calcsize(fmt)
        return v[0]

    def string():
        nonlocal pos
        n = take("<Q")
        s = b[pos:pos + n].decode("utf-8", errors="surrogateescape")
        pos += n
        return s

    def value(t):
        if t == 8:
            return string()
        if t == 9:
            et, n = take("<I"), take("<Q")
            return [value(et) for _ in range(n)]
        return take(_SCALAR[t])

    if b[:4] != b"GGUF":
        raise ValueError(f"{path}: not a GGUF file")
    pos = 4
    take("<I")  # version
    n_t, n_kv = take("<Q"), take("<Q")
    meta = {}
    for _ in range(n_kv):
        k = string()
        t = take("<I")
        meta[k] = (t, value(t))
    infos_start = pos
    tensors = []
    for _ in range(n_t):
        name = string()
        nd = take("<I")
        dims = [take("<Q") for _ in range(nd)]
        tensors.append((name, dims, take("<I"), take("<Q")))
    align = meta.get("general.alignment", (4, 32))[1]
    data_start = (pos + align - 1) // align * align
    return meta, tensors, data_start, b, (infos_start, pos)


def strip(x):
    """The document without "_"-prefixed keys, for semantic comparison."""
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if not k.startswith("_")}
    if isinstance(x, list):
        return [strip(v) for v in x]
    return x


def weight_refs(doc):
    """(where, name, fallback) for every tensor name the IR refers to, blocks expanded."""
    out = []

    def names(node):
        keys = ("weight", "bias", "router") + (("a", "b", "c", "d") if node.get("op") == "ssm_scan" else ())
        for key in keys:
            if isinstance(node.get(key), str):
                yield key, node[key], node.get("tied_to") if key == "weight" else None
        for key, v in (node.get("experts") or {}).items():
            yield "experts." + key, v, None

    for sect in ("embed", "head"):
        for j, node in enumerate(doc.get(sect, {}).get("nodes", [])):
            for key, n, fb in names(node):
                out.append((f"{sect}[{j}].{key}", n, fb))
    blk = doc.get("block", {})
    for i in range(blk.get("repeat", 0)):
        for j, node in enumerate(blk.get("nodes", [])):
            for key, n, fb in names(node):
                out.append((f"block {i}[{j}].{key}", n.replace("{i}", str(i)), fb))
    return out


STATE_OPS = {"attention": "kv", "linear_attention": "recurrent", "ssm_scan": "recurrent"}


def state_errors(doc):
    """What an engine loading from the IR alone needs: the model fields, and
    every node's state slot declared once in block.state with its kind's
    fields (recurrent slots: named tensors with shapes and initial values)."""
    errs = []
    model, blk = doc.get("model", {}), doc.get("block", {})
    for k in ("d", "vocab", "ctx", "n_layers"):
        if not isinstance(model.get(k), int) or model[k] <= 0:
            errs.append(f"model.{k} missing or not a positive integer")
    if blk.get("repeat") != model.get("n_layers"):
        errs.append(f"block.repeat {blk.get('repeat')} != model.n_layers {model.get('n_layers')}")
    slots = {}
    for s in blk.get("state", []):
        if s.get("name") in slots:
            errs.append(f"state slot {s.get('name')} declared twice")
        slots[s.get("name")] = s
        if s.get("kind") == "kv":
            for k in ("heads", "kv_heads", "head_dim", "max_len", "window", "sinks"):
                if not isinstance(s.get(k), int):
                    errs.append(f"kv state {s.get('name')}: {k} missing")
        elif s.get("kind") == "recurrent":
            ts = s.get("tensors")
            if s.get("op") not in ("linear_attention", "ssm_scan") or s.get("dtype") != "f32" or not ts:
                errs.append(f"recurrent state {s.get('name')}: op, dtype f32 and tensors required")
            for t in ts or []:
                if not t.get("name") or not t.get("shape") or not all(isinstance(x, int) and x > 0 for x in t["shape"]) \
                        or t.get("init") != 0:
                    errs.append(f"recurrent state {s.get('name')}: tensor {t} needs name, shape and init 0")
        else:
            errs.append(f"state slot {s.get('name')}: unknown kind {s.get('kind')}")
    used = set()
    for sect in ("embed", "block", "head"):
        for node in doc.get(sect, {}).get("nodes", []):
            want = STATE_OPS.get(node.get("op"))
            if want is None:
                continue
            slot = slots.get(node.get("state")) if sect == "block" else None
            if slot is None or slot.get("kind") != want or (want == "recurrent" and slot.get("op") != node["op"]):
                errs.append(f"{sect}: {node['op']} reads state {node.get('state')!r}, not a declared {want} slot")
            used.add(node.get("state"))
    for name in slots:
        if name not in used:
            errs.append(f"state slot {name} is read by no node")
    return errs


def check(path, like=None, tensors_as=None):
    errs = []
    meta, tensors, data_start, raw, (i0, i1) = read_gguf(path)
    if "paperkiln.ir" not in meta or meta["paperkiln.ir"][0] != 8:
        return [f"{path}: no string key paperkiln.ir"]
    doc = json.loads(meta["paperkiln.ir"][1])
    ver = meta.get("paperkiln.ir_version")
    if ver is None or ver[0] != 4 or ver[1] != doc.get("ir_version"):
        errs.append(f"paperkiln.ir_version {ver} does not match the document's ir_version {doc.get('ir_version')}")
    have = {t[0] for t in tensors}
    refs = weight_refs(doc)
    if not refs:
        errs.append("the IR names no tensors")
    for where, n, fb in refs:
        if n not in have and not (fb and fb in have):
            errs.append(f"{where}: {n} is not a tensor of {path}")
    errs += state_errors(doc)
    side = Path(path).with_suffix(".ir.json")
    if side.is_file() and json.loads(side.read_text(encoding="utf-8")) != doc:
        errs.append(f"sidecar {side} differs from the embedded IR")
    if like is not None:
        ref = json.loads(Path(like).read_text(encoding="utf-8"))
        if strip(ref) != strip(doc):
            a, b = strip(ref), strip(doc)
            for k in sorted(set(a) | set(b)):
                if a.get(k) != b.get(k):
                    errs.append(f"--like: section {k!r} differs:\n  ref {json.dumps(a.get(k))}\n  got {json.dumps(b.get(k))}")
    if tensors_as is not None:
        m2, t2, d2, raw2, (j0, j1) = read_gguf(tensors_as)
        if raw[i0:i1] != raw2[j0:j1]:
            errs.append(f"tensor infos differ from {tensors_as}")
        if raw[data_start:] != raw2[d2:]:
            errs.append(f"tensor data differ from {tensors_as}")
        missing = [k for k in m2 if k not in meta or meta[k] != m2[k]]
        if missing:
            errs.append(f"metadata keys lost or changed against {tensors_as}: {missing[:5]}")
        if list(meta)[:len(m2)] != list(m2):
            errs.append("the IR keys are not after every other key")
    return errs


def main(argv):
    if not argv or argv[0].startswith("-"):
        print(__doc__)
        return 2
    path, like, tensors_as = argv[0], None, None
    rest = argv[1:]
    while rest:
        opt, val, rest = rest[0], rest[1], rest[2:]
        if opt == "--like":
            like = val
        elif opt == "--tensors-as":
            tensors_as = val
        else:
            raise SystemExit(f"unknown option {opt}")
    errs = check(path, like, tensors_as)
    for e in errs:
        print("FAIL", e)
    if not errs:
        print(f"OK {path}")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
