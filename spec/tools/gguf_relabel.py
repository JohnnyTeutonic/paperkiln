#!/usr/bin/env python3
"""Relabel a GGUF with an embedded IR document as general.architecture "paperkiln" (IR_SPEC.md section
3.1, rule 38): for a file written before revision 1.3 under a llama.cpp architecture whose graph its
document is not (paperkiln's kimi, gpt2_swa, recurrence and plain flex exports carried "gpt2").

    python tools/gguf_relabel.py IN.gguf OUT.gguf [--force]

Every metadata key "<arch>.<rest>" becomes "paperkiln.<rest>" and general.architecture becomes
"paperkiln"; every other key, the key order, the tensor infos, the tensor data and the embedded
document (and so its digest and paperkiln.ir_sha256) are unchanged. Only the padding before the data
section moves (tensor offsets count from that section's start, section 3.1). The result is what an IR
engine runs, unchanged; a program that builds its graph from general.architecture (llama.cpp) refuses
it instead of computing another model. Without --force a file that rule 38 accepts as it is, or that
already carries "paperkiln", is not rewritten. Standard library only.
"""
from __future__ import annotations

from pathlib import Path
import struct
import sys

_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def _spans(b):
    """(version, n_tensors, [(key, type, value_start, value_end, value)], infos_start, infos_end, data_start)."""
    pos = 0

    def take(fmt):
        nonlocal pos
        v = struct.unpack_from(fmt, b, pos)[0]
        pos += struct.calcsize(fmt)
        return v

    def string():
        nonlocal pos
        n = take("<Q")
        s = b[pos:pos + n]
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
        raise ValueError("not a GGUF file")
    pos = 4
    version = take("<I")
    if version not in (2, 3):
        raise ValueError(f"GGUF version {version}")
    n_t, n_kv = take("<Q"), take("<Q")
    kvs = []
    for _ in range(n_kv):
        k = string().decode("utf-8", errors="surrogateescape")
        t = take("<I")
        v0 = pos
        v = value(t)
        kvs.append((k, t, v0, pos, v))
    i0 = pos
    for _ in range(n_t):
        string()
        nd = take("<I")
        pos += 8 * nd + 4 + 8
    i1 = pos
    align = next((v for k, t, _, _, v in kvs if k == "general.alignment"), 32)
    data = (i1 + align - 1) // align * align
    return version, n_t, kvs, i0, i1, data, align


def relabel(raw):
    """The relabelled file's bytes, and the old architecture."""
    version, n_t, kvs, i0, i1, data, align = _spans(raw)
    arch = next((v for k, t, _, _, v in kvs if k == "general.architecture"), None)
    if arch is None:
        raise ValueError("no general.architecture")
    arch = arch.decode("utf-8")
    if not any(k == "paperkiln.ir" for k, *_ in kvs):
        raise ValueError("no embedded document (paperkiln.ir): only a file whose document is the model can be "
                         "relabelled")
    out = bytearray(b"GGUF" + struct.pack("<IQQ", version, n_t, len(kvs)))
    pre = arch + "."
    for k, t, v0, v1, v in kvs:
        if k == "general.architecture":
            body = struct.pack("<Q", len(b"paperkiln")) + b"paperkiln"
        else:
            body = raw[v0:v1]
        if k.startswith(pre):
            k = "paperkiln." + k[len(pre):]
        kb = k.encode("utf-8", errors="surrogateescape")
        out += struct.pack("<Q", len(kb)) + kb + struct.pack("<I", t) + body
    out += raw[i0:i1]
    while len(out) % align:
        out += b"\0"
    out += raw[data:]
    return bytes(out), arch


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    force = "--force" in argv
    paths = [a for a in argv if not a.startswith("--")]
    if len(paths) != 2:
        print(__doc__)
        return 2
    src, dst = Path(paths[0]), Path(paths[1])
    raw = src.read_bytes()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import ir_validate  # noqa: PLC0415
    meta, tensors, *_ = ir_validate.read_gguf(src)
    arch = meta.get("general.architecture", (None, None))[1]
    if arch == "paperkiln":
        print(f"{src}: already paperkiln; nothing to do")
        return 0
    if not force and not ir_validate.architecture_errors(meta, tensors):
        print(f"{src}: rule 38 accepts it as {arch!r} (its document is that architecture's graph); not relabelled "
              f"(--force to relabel anyway)")
        return 0
    out, old = relabel(raw)
    dst.write_bytes(out)
    m2, t2, *_ = ir_validate.read_gguf(dst)
    errs = ir_validate.file_errors(dst)
    print(f"{dst}: {old} -> paperkiln, {len(t2)} tensors; validator: {'OK' if not errs else errs[:3]}")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
