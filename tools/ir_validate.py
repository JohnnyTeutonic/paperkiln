"""The shared validator for the paperkiln IR (spec/IR_SPEC.md section 10): the JSON Schema
(spec/ir.schema.json) and every refusal rule of section 10.1, each message naming its section.

    python tools/ir_validate.py doc.ir.json [model.gguf ...]   # documents and GGUFs; exit 1 on a refusal
    python tools/ir_validate.py --corpus [--quiet]              # every stored document and gate GGUF

    doc_errors(doc)                     -> [message]   schema + document rules (10.1 rules 1-23)
    gguf_errors(doc, meta, tensors)     -> [message]   container rules (24-28)
    file_errors(path)                   -> [message]   a .ir.json document or a GGUF with paperkiln.ir
    schema_errors(doc)                  -> [message]   the schema alone (this module's evaluator)

Standard library only (the expr grammar is checked with tools/ir_expr.py when numpy is importable).
The schema is evaluated by this module's own evaluator of the keywords spec/ir.schema.json uses;
tests/test_ir_validate.py checks it agrees with the `jsonschema` package where that is installed.
Messages read "IR_SPEC <section>: <what>" (a section of spec/IR_SPEC.md); tools/ir_check.py calls this module for its semantic checks.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path
import re
import struct
import sys

HERE = Path(__file__).resolve().parent
SCHEMA_PATH = HERE.parent / "spec" / "ir.schema.json"
VERSIONS = (0, 1)  # section 2.1: version 0 documents are version 1 documents
STATEFUL = {"attention": "kv", "linear_attention": "recurrent", "ssm_scan": "recurrent",
            "conv1d_causal": "recurrent", "selective_scan": "recurrent"}
RECURRENT_OPS = ("linear_attention", "ssm_scan", "conv1d_causal", "selective_scan")
# Rule 7: (exact input count or None, minimum, maximum) and the number of outputs.
ARITY = {"embedding": 1, "pos_embedding": 2, "linear": 1, "scale": 1, "split": 1, "rmsnorm": 1, "layernorm": 1,
         "gelu": 1, "silu": 1, "sigmoid": 1, "relu": 1, "add": 2, "mul": 2, "sub": 2, "affine": 1, "mul_col": 2,
         "rms_row": 1, "rope": 3, "attention": 3, "moe_ffn": 1, "append": 2, "linear_attention": 3,
         "ssm_scan": 1, "conv1d_causal": 1, "selective_scan": 4, "depth_route": 1}
AT_LEAST_ONE = {"list", "depth_attend", "expr"}
TENSOR_KEYS = ("weight", "bias", "router", "state_weight", "tied_to")
# Section 3.8: the structural keys, never per-layer (nodes, then slots).
NOT_PER_LAYER = ("op", "in", "out", "state", "tensor_index", "sizes", "expr", "weight", "bias", "router", "experts",
                 "state_weight", "tied_to", "a", "b", "c", "d")
SLOT_NOT_PER_LAYER = ("name", "kind", "op", "dtype", "tensors", "evict", "kv_rows", "kv_from")
MARKERS = {"eot": "<|endoftext|>", "user": "<|user|>", "assistant": "<|assistant|>"}  # paperkiln/1


def E(section, msg):
    return f"IR_SPEC {section}: {msg}"


# ---------------------------------------------------------------- GGUF reading

_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def read_gguf(path):
    """(metadata {key: (type, value)}, tensors [(name, dims, type, offset)], data_start, raw bytes,
    (tensor infos start, end))."""
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


# ---------------------------------------------------------------- the schema

_SCHEMA = None


def schema():
    global _SCHEMA
    if _SCHEMA is None:
        _SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return _SCHEMA


def _type_ok(v, t):
    if isinstance(t, list):
        return any(_type_ok(v, x) for x in t)
    return {"object": isinstance(v, dict), "array": isinstance(v, list), "string": isinstance(v, str),
            "boolean": isinstance(v, bool), "null": v is None,
            "integer": isinstance(v, int) and not isinstance(v, bool),
            "number": isinstance(v, (int, float)) and not isinstance(v, bool)}[t]


def _json_eq(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    return a == b


def _eval(s, v, root, path, out):
    """Append (path, message) for every failure of instance v against schema s (the keywords
    spec/ir.schema.json uses: $ref, type, enum, const, required, properties, patternProperties,
    additionalProperties, items, minItems, minimum, minLength, allOf, anyOf, if/then)."""
    if s is True or s == {}:
        return
    if s is False:
        out.append((path, "is not allowed"))
        return
    if "$ref" in s:
        ref = s["$ref"]
        assert ref.startswith("#/$defs/"), ref
        _eval(root["$defs"][ref[len("#/$defs/"):]], v, root, path, out)
    if "type" in s and not _type_ok(v, s["type"]):
        out.append((path, f"must be of type {s['type']}, not {json.dumps(v)[:60]}"))
        return
    if "enum" in s and not any(_json_eq(v, e) for e in s["enum"]):
        out.append((path, f"{json.dumps(v)[:60]} is not one of {s['enum']}"))
    if "const" in s and not _json_eq(v, s["const"]):
        out.append((path, f"must be {json.dumps(s['const'])}"))
    if isinstance(v, (int, float)) and not isinstance(v, bool) and "minimum" in s and v < s["minimum"]:
        out.append((path, f"{v} is below the minimum {s['minimum']}"))
    if isinstance(v, str) and "minLength" in s and len(v) < s["minLength"]:
        out.append((path, "must not be empty"))
    if isinstance(v, list):
        if "minItems" in s and len(v) < s["minItems"]:
            out.append((path, f"needs at least {s['minItems']} item(s)"))
        if "items" in s:
            for i, x in enumerate(v):
                _eval(s["items"], x, root, f"{path}[{i}]", out)
    if isinstance(v, dict):
        for k in s.get("required", []):
            if k not in v:
                out.append((path, f"lacks the required key {k!r}"))
        props, pats = s.get("properties", {}), s.get("patternProperties", {})
        for k, x in v.items():
            matched = False
            if k in props:
                matched = True
                _eval(props[k], x, root, f"{path}.{k}", out)
            for p, ps in pats.items():
                if re.search(p, k):
                    matched = True
                    _eval(ps, x, root, f"{path}.{k}", out)
            if not matched and "additionalProperties" in s:
                ap = s["additionalProperties"]
                if ap is False:
                    out.append((path, f"has the key {k!r}, which its specification does not define"))
                else:
                    _eval(ap, x, root, f"{path}.{k}", out)
    for sub in s.get("allOf", []):
        _eval(sub, v, root, path, out)
    if "anyOf" in s:
        if not any(not _collect(sub, v, root, path) for sub in s["anyOf"]):
            # Report the closest alternative's failures (the first one that is not a type mismatch).
            best = None
            for sub in s["anyOf"]:
                errs = _collect(sub, v, root, path)
                if best is None or (len(errs) < len(best)):
                    best = errs
            out.extend(best or [(path, "matches no alternative")])
    if "if" in s:
        if not _collect(s["if"], v, root, path):
            if "then" in s:
                _eval(s["then"], v, root, path, out)
        elif "else" in s:
            _eval(s["else"], v, root, path, out)


def _collect(s, v, root, path):
    out = []
    _eval(s, v, root, path, out)
    return out


def schema_errors(doc):
    """The schema's failures as messages (section 10.1 rule 1)."""
    out = _collect(schema(), doc, schema(), "doc")
    seen, msgs = set(), []
    for p, m in out:
        if (p, m) not in seen:
            seen.add((p, m))
            msgs.append(E("10.1 rule 1", f"{p} {m}"))
    return msgs


# ---------------------------------------------------------------- document rules

def _layers(v, L):
    """The value per application: the per_layer list or the scalar repeated."""
    if isinstance(v, dict) and "per_layer" in v:
        return v["per_layer"] if isinstance(v["per_layer"], list) and len(v["per_layer"]) == L else None
    return [v] * L


def _sections(doc):
    for sect in ("embed", "block", "head"):
        nodes = (doc.get(sect) or {}).get("nodes") if isinstance(doc.get(sect), dict) else None
        for j, n in enumerate(nodes if isinstance(nodes, list) else []):
            if isinstance(n, dict):
                yield sect, j, n


def _tensor_names(node):
    for key in TENSOR_KEYS:
        if isinstance(node.get(key), str):
            yield key, node[key]
    for key in ("a", "b", "c", "d"):
        if node.get("op") in ("ssm_scan", "selective_scan") and isinstance(node.get(key), str):
            yield key, node[key]
    if isinstance(node.get("experts"), dict):
        for key, v in node["experts"].items():
            if isinstance(v, str):
                yield "experts." + key, v


def _expr_error(text, n_inputs):
    try:
        sys.path.insert(0, str(HERE))
        import ir_expr  # noqa: PLC0415 (numpy; optional)
    except ImportError:
        return None
    finally:
        if sys.path and sys.path[0] == str(HERE):
            sys.path.pop(0)
    try:
        ir_expr.parse(text, n_inputs)
    except ValueError as e:
        return str(e)
    return None


def node_errors(doc):
    """Rules 4-10 and 14 over every node (the schema covers the closed key sets; these add the
    rules a schema cannot state)."""
    errs = []
    blk = doc.get("block") if isinstance(doc.get("block"), dict) else {}
    L = blk.get("repeat") if isinstance(blk.get("repeat"), int) else 0
    for sect, j, n in _sections(doc):
        op, where = n.get("op"), f"{sect}[{j}] {n.get('op')}"
        ins, outs = n.get("in"), n.get("out")
        ins = ins if isinstance(ins, list) else []
        # Rule 6: per-layer values.
        for k, v in n.items():
            if isinstance(v, dict) and "per_layer" in v:
                if sect != "block":
                    errs.append(E("3.8", f"{where}: per-layer {k!r} outside the block"))
                elif k in NOT_PER_LAYER:
                    errs.append(E("3.8", f"{where}: {k!r} is structural and cannot be per-layer"))
                elif not isinstance(v["per_layer"], list) or len(v["per_layer"]) != L:
                    got = len(v["per_layer"]) if isinstance(v["per_layer"], list) else "no list"
                    errs.append(E("3.8", f"{where}: per-layer {k!r} has {got} entries for block.repeat {L}"))
        # Rule 7: arity.
        want = ARITY.get(op)
        if want is not None and len(ins) != want:
            errs.append(E("10.1 rule 7", f"{where} reads {len(ins)} value(s), not {want}"))
        if op in AT_LEAST_ONE and not ins:
            errs.append(E("10.1 rule 7", f"{where} reads no value"))
        if op == "expr" and len(ins) > 23:
            errs.append(E("5.19", f"{where} reads {len(ins)} values (at most 23)"))
        if op in ("split", "rope"):
            if not isinstance(outs, list):
                errs.append(E("10.1 rule 7", f"{where} writes a list of names"))
            elif op == "rope" and len(outs) != 2:
                errs.append(E("5.14", f"{where} writes {len(outs)} values, not [q, k]"))
        elif op is not None and not isinstance(outs, str):
            errs.append(E("10.1 rule 7", f"{where} writes one value"))
        # Rule 9: expr parses.
        if op == "expr" and isinstance(n.get("expr"), str):
            msg = _expr_error(n["expr"], max(1, len(ins)))
            if msg:
                errs.append(E("5.19", f"{where}: {msg}"))
        # Rule 10 and 14: values.
        if op == "split" and isinstance(n.get("sizes"), list) and isinstance(outs, list) and len(outs) != len(n["sizes"]):
            errs.append(E("5.5", f"{where}: {len(n['sizes'])} sizes for {len(outs)} outputs"))
        if op == "rope" and L:
            hd, nr, fs = _layers(n.get("head_dim"), L), _layers(n.get("n_rot"), L), _layers(n.get("freq_scale"), L)
            for i in range(L):
                if nr and hd and nr[i] is not None and isinstance(hd[i], int) and isinstance(nr[i], int) \
                        and not (0 < nr[i] <= hd[i] and nr[i] % 2 == 0):
                    errs.append(E("5.14", f"{where}: n_rot {nr[i]} is not even and in 1 .. head_dim {hd[i]}"))
                    break
                if fs and fs[i] is not None and isinstance(fs[i], (int, float)) and not fs[i] > 0:
                    errs.append(E("5.14", f"{where}: freq_scale {fs[i]} is not positive"))
                    break
        if op == "attention" and L:
            h, kv, w, s, c = (_layers(n.get(k), L) for k in ("heads", "kv_heads", "window", "sinks", "causal"))
            for i in range(L):
                if h and kv and isinstance(h[i], int) and isinstance(kv[i], int) and kv[i] > 0 and h[i] % kv[i]:
                    errs.append(E("5.15", f"{where}: heads {h[i]} is not a multiple of kv_heads {kv[i]}"))
                    break
            for i in range(L):
                if (w and isinstance(w[i], int) and w[i] < 0) or (s and isinstance(s[i], int) and s[i] < 0):
                    errs.append(E("5.15", f"{where}: window and sinks must be >= 0"))
                    break
            if c and any(x is not True for x in c):
                errs.append(E("5.15", f"{where}: causal must be true (Q6)"))
        if op == "linear_attention" and n.get("causal") is not True:
            errs.append(E("6.4", f"{where}: causal must be true"))
        if op == "moe_ffn" and isinstance(n.get("top_k"), int) and isinstance(n.get("n_experts"), int) \
                and not 1 <= n["top_k"] <= n["n_experts"]:
            errs.append(E("5.16", f"{where}: top_k {n['top_k']} is not in 1 .. n_experts {n['n_experts']}"))
        if op == "conv1d_causal" and ((isinstance(n.get("channels"), int) and n["channels"] < 1)
                                      or (isinstance(n.get("kernel"), int) and n["kernel"] < 2)):
            errs.append(E("6.6", f"{where}: channels must be >= 1 and kernel >= 2"))
        # Names: {i} outside the block or twice (3.7); tensor_index (rule 19).
        for key, name in _tensor_names(n):
            if "{i}" in name and sect != "block":
                errs.append(E("3.7", f"{where}: {key} {name!r} uses {{i}} outside the block"))
            if name.count("{i}") > 1:
                errs.append(E("3.7", f"{where}: {key} {name!r} holds {{i}} more than once"))
        # The graph inputs (3.5): tokens for embeddings, pos as rope's third and pos_embedding's second input.
        for k, nm in enumerate(ins):
            allowed = {"tokens": op == "embedding" and k == 0,
                       "pos": (op == "rope" and k == 2) or (op == "pos_embedding" and k == 1)}
            if nm in allowed and not allowed[nm]:
                errs.append(E("3.5", f"{where} reads the graph input {nm!r} as its input {k}"))
        if (op == "rope" and len(ins) == 3 and ins[2] != "pos") or (op == "pos_embedding" and len(ins) == 2
                                                                     and ins[1] != "pos"):
            errs.append(E("3.5", f"{where}: the position input must be the graph input 'pos'"))
        if op == "embedding" and ins and ins[0] != "tokens":
            errs.append(E("3.5", f"{where}: an embedding reads the graph input 'tokens'"))
        for o in (outs if isinstance(outs, list) else [outs]):
            if o in ("tokens", "pos"):
                errs.append(E("3.5", f"{where} writes the graph input {o!r}"))
        if "tensor_index" in n:
            if sect != "block" or "weight_index" not in blk:
                errs.append(E("7.2", f"{where}: tensor_index needs a block with weight_index"))
    return errs


def state_errors(doc):
    """Rules 3, 11-13 and 18: the model fields, the slots, the stateful nodes, weight_index."""
    errs = []
    model = doc.get("model") if isinstance(doc.get("model"), dict) else {}
    blk = doc.get("block") if isinstance(doc.get("block"), dict) else {}
    for k in ("d", "vocab", "ctx", "n_layers"):
        if not isinstance(model.get(k), int) or isinstance(model.get(k), bool) or model[k] <= 0:
            errs.append(E("3.3", f"model.{k} missing or not a positive integer"))
    if blk.get("repeat") != model.get("n_layers"):
        errs.append(E("3.3", f"block.repeat {blk.get('repeat')} != model.n_layers {model.get('n_layers')}"))
    L = blk.get("repeat") if isinstance(blk.get("repeat"), int) else 0
    if "weight_index" in blk:
        m = blk["weight_index"].get("per_layer") if isinstance(blk["weight_index"], dict) else None
        if not isinstance(m, list) or len(m) != L or not all(isinstance(x, int) and not isinstance(x, bool) and x >= 0
                                                              for x in m):
            errs.append(E("7.1", "block.weight_index must be {per_layer: [block.repeat integers >= 0]}"))
        elif sorted(set(m)) != list(range(max(m) + 1)):
            errs.append(E("7.1", f"block.weight_index uses stored layers {sorted(set(m))}, not every one of 0..{max(m)}"))
    slots = {}
    for s in blk.get("state", []) if isinstance(blk.get("state"), list) else []:
        if not isinstance(s, dict):
            continue
        name = s.get("name")
        if name in slots:
            errs.append(E("6.1", f"state slot {name} declared twice"))
        slots[name] = s
        for k, v in s.items():
            if isinstance(v, dict) and "per_layer" in v:
                if k in SLOT_NOT_PER_LAYER:
                    errs.append(E("3.8", f"state slot {name}: {k!r} is structural and cannot be per-layer"))
                elif not isinstance(v["per_layer"], list) or len(v["per_layer"]) != L:
                    errs.append(E("3.8", f"state slot {name}: per-layer {k!r} needs block.repeat {L} entries"))
        if s.get("kind") == "kv":
            for k in ("heads", "kv_heads", "head_dim", "max_len", "window", "sinks"):
                v = s.get(k)
                if not (isinstance(v, int) or (isinstance(v, dict) and "per_layer" in v)):
                    errs.append(E("6.2", f"kv state {name}: {k} missing"))
            if "kv_from" in s:
                errs.append(E("7.6", f"kv state {name}: kv_from is reserved and not implemented"))
        elif s.get("kind") == "recurrent":
            ts = s.get("tensors")
            if s.get("op") not in RECURRENT_OPS or s.get("dtype") != "f32" or not ts:
                errs.append(E("6.3", f"recurrent state {name}: op (a recurrent op), dtype f32 and tensors required"))
            for t in ts if isinstance(ts, list) else []:
                if not isinstance(t, dict) or not t.get("name") or not t.get("shape") or not isinstance(t["shape"], list) \
                        or not all(isinstance(x, int) and x > 0 for x in t["shape"]) or t.get("init") != 0:
                    errs.append(E("6.3", f"recurrent state {name}: tensor {t} needs name, shape and init 0"))
        else:
            errs.append(E("6.1", f"state slot {name}: unknown kind {s.get('kind')}"))
    used, readers = set(), {}
    for sect, j, node in _sections(doc):
        want = STATEFUL.get(node.get("op"))
        if want is None:
            if "state" in node:
                errs.append(E("6.1", f"{sect}[{j}] {node.get('op')} names state {node['state']!r} but has no state"))
            continue
        if sect != "block":
            errs.append(E("6.1", f"{sect}: {node['op']} outside the block has no state"))
            continue
        slot = slots.get(node.get("state"))
        if slot is None or slot.get("kind") != want or (want == "recurrent" and slot.get("op") != node["op"]):
            errs.append(E("6.1", f"{sect}: {node['op']} reads state {node.get('state')!r}, not a declared {want} slot"
                                 + (f" of op {node['op']}" if want == "recurrent" else "")))
            continue
        used.add(node.get("state"))
        if node["op"] == "attention":
            readers[node.get("state")] = readers.get(node.get("state"), 0) + 1
        errs += _slot_fit(node, slot, L)
    for name, n in readers.items():
        if n > 1:
            errs.append(E("6.1", f"kv state {name} is read by {n} attention nodes (exactly one is defined)"))
    for name in slots:
        if name not in used:
            errs.append(E("6.1", f"state slot {name} is read by no node"))
    return errs


def _slot_fit(node, slot, L):
    """Rule 12 (attention against its kv slot) and rule 13 (a recurrent slot's tensors fit its op)."""
    op, errs = node["op"], []
    if op == "attention":
        for key in ("window", "sinks", "kv_heads", "head_dim"):
            a, b = _layers(node.get(key), L), _layers(slot.get(key), L)
            if a is None or b is None:
                continue
            for i in range(L):
                if a[i] != b[i]:
                    errs.append(E("5.15", f"attention {key} {a[i]} (application {i}) differs from its kv state "
                                          f"{slot.get('name')}'s {b[i]}" + (" (Q7)" if key in ("kv_heads", "head_dim") else "")))
                    break
        return errs
    shapes = {t.get("name"): t.get("shape") for t in slot.get("tensors", []) if isinstance(t, dict)}
    if op == "linear_attention":
        want = {"num": [node.get("heads"), node.get("head_dim")], "den": [node.get("heads"), node.get("head_dim")]}
    elif op == "ssm_scan":
        want = {"s": [node.get("n")]}
    elif op == "conv1d_causal":
        k = node.get("kernel")
        want = {"x": [k - 1 if isinstance(k, int) else k, node.get("channels")]}
    else:
        want = {"s": [node.get("channels"), node.get("d_state")]}
    if shapes != want:
        errs.append(E("6.3", f"recurrent state {slot.get('name')}: tensors {shapes} do not fit {op} (want {want})"))
    return errs


def depth_route_errors(doc):
    """Rules 20-21 and 23: depth_route first and once, its steps, the router name, kv_rows active."""
    errs = []
    blk = doc.get("block") if isinstance(doc.get("block"), dict) else {}
    nodes = blk.get("nodes") if isinstance(blk.get("nodes"), list) else []
    for sect, j, n in _sections(doc):
        if n.get("op") == "depth_route" and sect != "block":
            errs.append(E("7.4", f"depth_route outside the block ({sect})"))
    at = [j for j, n in enumerate(nodes) if isinstance(n, dict) and n.get("op") == "depth_route"]
    slots = [s for s in blk.get("state", []) if isinstance(s, dict) and s.get("kind") == "kv"]
    for s in slots:
        if "kv_rows" in s and s["kv_rows"] != "active":
            errs.append(E("7.5", f"kv state {s.get('name')}: kv_rows {s['kv_rows']!r} (only \"active\" is defined)"))
    if not at:
        return errs
    if at != [0]:
        return errs + [E("7.4", "depth_route must be the block's first node, once")]
    r = nodes[0]
    R = r.get("n_rec")
    if not isinstance(R, int) or isinstance(R, bool) or R < 1:
        return errs + [E("7.4", "depth_route: n_rec must be an integer >= 1")]
    if r.get("route") != "token_choice":
        errs.append(E("7.7", f"depth_route: route {r.get('route')!r} (only token_choice, causal, is defined)"))
    if r.get("gate") not in ("score", "none"):
        errs.append(E("7.4", f"depth_route: gate {r.get('gate')!r} is not score or none"))
    w = r.get("weight")
    if not isinstance(w, str) or "{" in w:
        errs.append(E("7.7", "depth_route: weight must be the router's full tensor name (constraint 4: no substitution)"))
    if not isinstance(r.get("out"), str):
        errs.append(E("7.4", "depth_route: out must name the depth trace"))
    st = r.get("step").get("per_layer") if isinstance(r.get("step"), dict) else None
    if not isinstance(st, list) or len(st) != blk.get("repeat") or not all(isinstance(x, int) for x in st):
        return errs + [E("7.4", "depth_route: step must be {per_layer: [block.repeat integers]}")]
    runs = []
    for i, x in enumerate(st):
        if x == -1:
            continue
        if i == 0 or st[i - 1] != x:
            runs.append([x, 0])
        runs[-1][1] += 1
    inside = [i for i, x in enumerate(st) if x != -1]
    contiguous = bool(inside) and inside == list(range(inside[0], inside[-1] + 1))
    if any(x < -1 or x >= R for x in st) or [k for k, _ in runs] != list(range(R)) or not contiguous:
        errs.append(E("7.4", f"depth_route: steps {st} are not -1 or the consecutive steps 0 .. {R - 1} in order"))
    elif len({n for _, n in runs}) != 1:
        errs.append(E("7.4", f"depth_route: steps {st} differ in length"))
    if not slots or any(s.get("kv_rows") != "active" for s in slots):
        errs.append(E("7.5", "depth_route needs kv_rows \"active\" on every kv slot"))
    if any(n.get("op") == "attention" and n.get("mask") == "turn_state" for n in nodes if isinstance(n, dict)):
        errs.append(E("9.2", "depth_route with mask turn_state is not defined (llama_state)"))
    if any(n.get("op") in ("list", "append", "depth_attend") for _, _, n in _sections(doc)):
        errs.append(E("7.4", "depth_route with list values (list, append, depth_attend) is not defined"))
    out = r.get("out")
    for sect, j, n in _sections(doc):
        if isinstance(out, str) and out in (n.get("in") or []):
            errs.append(E("7.4", f"{sect}[{j}] {n.get('op')} reads depth_route's out {out!r} (the depth report)"))
    return errs


def state_token_doc_errors(doc):
    """Rules 15-17: the turn_state mask, kv evict and the state_weight's placement."""
    errs = []
    blk = doc.get("block") if isinstance(doc.get("block"), dict) else {}
    L = blk.get("repeat") if isinstance(blk.get("repeat"), int) else 0
    sws = []
    for sect, j, n in _sections(doc):
        if n.get("op") == "embedding" and "state_weight" in n:
            sws.append(n["state_weight"])
            if sect != "embed":
                errs.append(E("6.8.1", f"{sect}[{j}]: an embedding state_weight outside the embed section (Q8)"))
            if isinstance(n["state_weight"], str) and "{" in n["state_weight"]:
                errs.append(E("6.8.1", "state_weight names its tensor in full (Q8)"))
    if len(sws) > 1:
        errs.append(E("6.8.1", "more than one embedding with a state_weight (Q8)"))
    slots = {s.get("name"): s for s in blk.get("state", []) if isinstance(s, dict)}
    masks = {}
    for n in blk.get("nodes", []) if isinstance(blk.get("nodes"), list) else []:
        if not isinstance(n, dict) or n.get("op") != "attention":
            continue
        m = n.get("mask", "causal")
        if m not in ("causal", "turn_state"):
            errs.append(E("6.8.2", f"attention mask {m!r} (causal or turn_state)"))
        if m == "turn_state":
            w, s = _layers(n.get("window", 0), L) or [], _layers(n.get("sinks", 0), L) or []
            slot = slots.get(n.get("state")) or {}
            sw, ss = _layers(slot.get("window", 0), L) or [], _layers(slot.get("sinks", 0), L) or []
            if any(w) or any(s) or any(sw) or any(ss) or not sws:
                errs.append(E("6.8.2", "attention mask turn_state needs window 0, sinks 0 (every application, and its "
                                       "kv slot) and an embedding state_weight"))
        masks.setdefault(n.get("state"), set()).add(m)
    for st in blk.get("state", []) if isinstance(blk.get("state"), list) else []:
        if not isinstance(st, dict) or "evict" not in st:
            continue
        if st["evict"] != "turn_state":
            errs.append(E("6.8.3", f"kv state {st.get('name')}: evict {st['evict']!r} (only turn_state is defined)"))
        elif masks.get(st.get("name")) != {"turn_state"}:
            errs.append(E("6.8.3", f"kv state {st.get('name')}: evict turn_state needs every attention node reading it "
                                   f"to mask turn_state"))
    return errs


def doc_errors(doc, versions=VERSIONS):
    """Every document rule (10.1 rules 1-23). The schema runs first; the semantic rules run on what the
    schema leaves usable, so one fault does not cascade into many messages where it can be avoided."""
    if not isinstance(doc, dict):
        return [E("10.1 rule 1", "the document is not a JSON object")]
    errs = schema_errors(doc)
    v = doc.get("ir_version")
    if isinstance(v, int) and not isinstance(v, bool) and v not in versions:
        errs.append(E("2.3", f"ir_version {v} is not supported (this validator reads {', '.join(map(str, versions))})"))
    errs += node_errors(doc)
    errs += state_errors(doc)
    errs += depth_route_errors(doc)
    errs += state_token_doc_errors(doc)
    out, seen = [], set()
    for e in errs:
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out


# ---------------------------------------------------------------- container rules

def weight_refs(doc):
    """(where, name, fallback) for every tensor name the document refers to, blocks expanded
    (weight_index's m_i; tensor_index "application": i)."""
    out = []
    blk = doc.get("block", {})
    wmap = (blk.get("weight_index") or {}).get("per_layer") if isinstance(blk.get("weight_index"), dict) else None

    def names(node):
        for key, n in _tensor_names(node):
            if key == "tied_to":
                continue
            yield key, n, node.get("tied_to") if key == "weight" else None

    for sect in ("embed", "head"):
        for j, node in enumerate(doc.get(sect, {}).get("nodes", [])):
            for key, n, fb in names(node):
                out.append((f"{sect}[{j}].{key}", n, fb))
    for i in range(blk.get("repeat", 0) if isinstance(blk.get("repeat"), int) else 0):
        m = wmap[i] if isinstance(wmap, list) and i < len(wmap) else i
        for j, node in enumerate(blk.get("nodes", [])):
            for key, n, fb in names(node):
                t = i if node.get("tensor_index") == "application" else m
                out.append((f"block {i}[{j}].{key}", n.replace("{i}", str(t)), fb))
    return out


def gguf_errors(doc, meta, tensors):
    """Container rules 24-28 (section 10.1) for a document embedded in a GGUF."""
    errs = []
    ver = meta.get("paperkiln.ir_version")
    if ver is None or ver[0] != 4 or ver[1] != doc.get("ir_version"):
        errs.append(E("3.1", f"paperkiln.ir_version {ver} does not match the document's ir_version "
                             f"{doc.get('ir_version')}"))
    have = {t[0] for t in tensors}
    dims = {t[0]: t[1] for t in tensors}
    refs = weight_refs(doc)
    if not refs:
        errs.append(E("3.7", "the document names no tensors"))
    for where, n, fb in refs:
        if n not in have and not (fb and fb in have):
            errs.append(E("3.7", f"{where}: {n} is not a tensor of the file"))
    arch = meta.get("general.architecture", (None, None))[1]
    for node in doc.get("block", {}).get("nodes", [])[:1]:
        if node.get("op") == "depth_route":
            got = dims.get(node.get("weight"))
            want = [doc.get("model", {}).get("d"), node.get("n_rec")]
            if got is not None and got != want:
                errs.append(E("7.4", f"depth_route: {node['weight']} has ne {got}, not [d, n_rec] = {want}"))
    wmap = (doc.get("block", {}).get("weight_index") or {}).get("per_layer")
    if isinstance(wmap, list) and wmap and all(isinstance(x, int) for x in wmap):
        if arch != "paperkiln":
            errs.append(E("7.1", f"block.weight_index needs architecture paperkiln, not {arch}"))
        bc = meta.get(f"{arch}.block_count", (None, None))[1]
        if bc != max(wmap) + 1:
            errs.append(E("7.1", f"{arch}.block_count {bc} != the {max(wmap) + 1} stored layers weight_index uses"))
    errs += state_token_gguf_errors(doc, meta, tensors)
    return errs


def state_token_gguf_errors(doc, meta, tensors):
    """Rule 28: paperkiln.state_tokens.k, the state_weight's shape, the architecture, the specials and
    the template."""
    errs = []
    dims = {t[0]: t[1] for t in tensors}
    k = meta.get("paperkiln.state_tokens.k", (None, None))[1]
    sws = [n.get("state_weight") for n in doc.get("embed", {}).get("nodes", [])
           if n.get("op") == "embedding" and "state_weight" in n]
    if (k is not None and k > 0) != bool(sws):
        errs.append(E("6.8.1", f"paperkiln.state_tokens.k {k} without an embedding state_weight, or the reverse"))
    for sw in sws:
        want = [doc.get("model", {}).get("d"), k]
        if sw not in dims:
            errs.append(E("6.8.1", f"embedding state_weight {sw} is not a tensor of the file"))
        elif dims[sw] != want:
            errs.append(E("6.8.1", f"embedding state_weight {sw} has ne {dims[sw]}, not [d, paperkiln.state_tokens.k] "
                                   f"= {want}"))
    if sws:
        arch = meta.get("general.architecture", (None, None))[1]
        if arch != "paperkiln":
            errs.append(E("8.2", f"state tokens need architecture paperkiln, not {arch}"))
        sp = (doc.get("tokenizer") or {}).get("specials") or {}
        toks = meta.get("tokenizer.ggml.tokens", (None, None))[1]
        for role in ("eot", "user", "assistant"):
            if not isinstance(sp.get(role), int):
                errs.append(E("8.2", f"state tokens: the tokenizer names no {role} id (the template needs it)"))
            elif isinstance(toks, list) and not (0 <= sp[role] < len(toks) and toks[sp[role]] == MARKERS[role]):
                got = toks[sp[role]] if 0 <= sp[role] < len(toks) else None
                errs.append(E("8.2", f"state tokens: specials.{role} = {sp[role]} is the token {got!r}, not "
                                     f"{MARKERS[role]!r}"))
        tmpl = meta.get("paperkiln.chat_template", (None, None))[1]
        if tmpl is not None:
            try:
                rec = json.loads(tmpl)
                name = f"{rec.get('name')}/{rec.get('version')}" if "/" not in str(rec.get("name")) else rec.get("name")
            except (ValueError, AttributeError):
                name = None
            if name not in ("paperkiln/1",):
                errs.append(E("8.2", f"state tokens are defined over paperkiln/1 only, not {name}"))
    return errs


def file_errors(path):
    """A .json document, or a GGUF with paperkiln.ir (its sidecar too)."""
    path = Path(path)
    if path.suffix == ".gguf":
        meta, tensors, *_ = read_gguf(path)
        if "paperkiln.ir" not in meta or meta["paperkiln.ir"][0] != 8:
            return [E("3.1", f"{path}: no string key paperkiln.ir")]
        doc = json.loads(meta["paperkiln.ir"][1])
        errs = doc_errors(doc) + gguf_errors(doc, meta, tensors)
        side = path.with_suffix(".ir.json")
        if side.is_file() and json.loads(side.read_text(encoding="utf-8")) != doc:
            errs.append(E("3.1", f"sidecar {side} differs from the embedded document"))
        return errs
    return doc_errors(json.loads(path.read_text(encoding="utf-8")))


# ---------------------------------------------------------------- the corpus

WORKSPACE = HERE.parents[1]


def corpus():
    """Every stored IR document and every gate GGUF: the worked documents, the experiment receipts of
    both repositories, and C:/ml_artifacts (documents and the *_gate GGUFs)."""
    pats = [str(HERE.parent / "ir" / "*.ir.json"), str(HERE.parent / "experiments" / "**" / "*.ir.json"),
            str(WORKSPACE / "ember.cpp" / "experiments" / "**" / "*.ir.json")]
    art = os.environ.get("ML_ARTIFACTS", "C:/ml_artifacts" if os.name == "nt" else "/mnt/c/ml_artifacts")
    pats += [art + "/**/*.ir.json", art + "/*gate*/**/*.gguf"]
    return sorted({f for p in pats for f in glob.glob(p, recursive=True) if Path(f).name != "v.gguf"})


def main(argv):
    quiet = "--quiet" in argv
    paths = [a for a in argv if not a.startswith("--")]
    if "--corpus" in argv:
        paths = corpus() + paths
    if not paths:
        print(__doc__)
        return 2
    bad = 0
    for p in paths:
        try:
            errs = file_errors(p)
        except (OSError, ValueError) as e:
            errs = [f"cannot read: {e}"]
        bad += bool(errs)
        if errs:
            print(f"REFUSED {p}")
            for e in errs:
                print("   ", e)
        elif not quiet:
            print(f"OK {p}")
    print(f"{len(paths) - bad} accepted, {bad} refused, of {len(paths)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
