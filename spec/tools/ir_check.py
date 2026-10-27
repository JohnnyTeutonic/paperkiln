"""Checks for the keramos IR document embedded in a GGUF (IR_SPEC.md).

    python tools/ir_check.py model.gguf                    container + specification checks
    python tools/ir_check.py model.gguf --like examples/chat_sftfinal.ir.json
    python tools/ir_check.py model.gguf --tensors-as base.gguf

Without options: the GGUF carries paperkiln.ir (string) and paperkiln.ir_version (uint32) agreeing
with the document, and the document passes tools/ir_validate.py (the schema and every refusal rule
of IR_SPEC.md section 10.1, the container rules included: every weight the IR names, "{i}" expanded
over the blocks, is a tensor of the file, a missing linear weight passing only through its
"tied_to"); a sidecar <stem>.ir.json, when present, is the same document. --like: the embedded IR
is semantically equal to a reference (key order and "_"-prefixed prose keys ignored). --tensors-as:
tensor infos and the tensor-data region are byte for byte those of another GGUF (an export without
the IR, say). Standard library only. Exit status 1 on any failure.
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ir_validate  # noqa: E402
from ir_validate import read_gguf, weight_refs  # noqa: E402,F401  (public here too)


def strip(x):
    """The document without "_"-prefixed keys, for semantic comparison."""
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if not k.startswith("_")}
    if isinstance(x, list):
        return [strip(v) for v in x]
    return x


def state_errors(doc):
    """The model fields and the state slots (IR_SPEC.md sections 3.3 and 6; ir_validate)."""
    return ir_validate.state_errors(doc)


def depth_route_errors(doc):
    """depth_route and kv_rows (IR_SPEC.md section 7.4 to 7.7; ir_validate)."""
    return ir_validate.depth_route_errors(doc)


def state_token_errors(doc, meta, tensors):
    """Per-turn state tokens (IR_SPEC.md sections 6.8 and 8.2; ir_validate): the mask, eviction and
    the state_weight in the document; paperkiln.state_tokens.k, the tensor's shape, the architecture,
    the specials and the template in the file."""
    return ir_validate.state_token_doc_errors(doc) + ir_validate.state_token_gguf_errors(doc, meta, tensors)


def check(path, like=None, tensors_as=None):
    errs = []
    meta, tensors, data_start, raw, (i0, i1) = read_gguf(path)
    if "paperkiln.ir" not in meta or meta["paperkiln.ir"][0] != 8:
        return [f"{path}: no string key paperkiln.ir"]
    doc = json.loads(meta["paperkiln.ir"][1])
    errs += ir_validate.doc_errors(doc)
    errs += ir_validate.gguf_errors(doc, meta, tensors)
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
