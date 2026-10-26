"""The shared validator for the paperkiln IR (IR_SPEC.md section 10): the JSON Schema
(ir.schema.json) and every refusal rule of section 10.1, each message naming its section.

    python tools/ir_validate.py doc.ir.json [model.gguf ...]   # documents and GGUFs; exit 1 on a refusal
    python tools/ir_validate.py --corpus [--corpus-glob PAT ...] [--quiet]   # stored documents (corpus())
    python tools/ir_validate.py digest doc.ir.json|model.gguf [--canonical]   # the IR digest (section 3.10)

    text_errors(text)                   -> [message]   a document's JSON text: rule 35 (duplicate member names), then doc_errors
    load_document(text)                 -> doc         the parsed document; DocumentError (rule 1 or 35) otherwise
    doc_errors(doc)                     -> [message]   schema + document rules (10.1 rules 1-23, 31-33, 36)
    gguf_errors(doc, meta, tensors)     -> [message]   container rules (24-28, 30, 34)
    file_errors(path)                   -> [message]   a .ir.json document or a GGUF with paperkiln.ir
    schema_errors(doc)                  -> [message]   the schema alone (this module's evaluator), each failure
                                                       under the rule of section 10.1 that names it
    canonical(text_or_value) -> str; digest(text_or_value) -> 64 hex digits   (section 3.10; DigestError)

Standard library only (the expr grammar is checked with tools/ir_expr.py when numpy is importable).
The schema is evaluated by this module's own evaluator of the keywords ir.schema.json uses;
tests/test_ir_validate.py checks it agrees with the `jsonschema` package where that is installed.
Every refusal reads "IR_SPEC <section> (rule <n>): <what>": a section of IR_SPEC.md and the rule of its
section 10.1 that the document breaks. tools/ir_check.py calls this module for its semantic checks.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path
import re
import struct
import sys

HERE = Path(__file__).resolve().parent
SCHEMA_PATH = HERE.parent / "ir.schema.json"
VERSIONS = (0, 1)  # section 2.1: version 0 documents are version 1 documents
STATEFUL = {"attention": "kv", "linear_attention": "recurrent", "ssm_scan": "recurrent",
            "conv1d_causal": "recurrent", "selective_scan": "recurrent", "recurrence": "recurrent"}
RECURRENT_OPS = ("linear_attention", "ssm_scan", "conv1d_causal", "selective_scan", "recurrence")
# Rule 7: (exact input count or None, minimum, maximum) and the number of outputs.
ARITY = {"embedding": 1, "pos_embedding": 2, "linear": 1, "scale": 1, "split": 1, "rmsnorm": 1, "layernorm": 1,
         "gelu": 1, "silu": 1, "sigmoid": 1, "relu": 1, "add": 2, "mul": 2, "sub": 2, "affine": 1, "mul_col": 2,
         "rms_row": 1, "rope": 3, "attention": 3, "moe_ffn": 1, "append": 2, "linear_attention": 3,
         "ssm_scan": 1, "conv1d_causal": 1, "selective_scan": 4, "depth_route": 1}
AT_LEAST_ONE = {"list", "depth_attend", "expr", "recurrence"}
TENSOR_KEYS = ("weight", "bias", "router", "state_weight", "tied_to")
# Section 3.8: the structural keys, never per-layer (nodes, then slots).
NOT_PER_LAYER = ("op", "in", "out", "state", "tensor_index", "sizes", "expr", "weight", "bias", "router", "experts",
                 "state_weight", "tied_to", "a", "b", "c", "d", "index", "inputs", "weights", "init", "let", "update",
                 "readout", "position_bias")
SLOT_NOT_PER_LAYER = ("name", "kind", "op", "dtype", "tensors", "evict", "kv_rows")
MARKERS = {"eot": "<|endoftext|>", "user": "<|user|>", "assistant": "<|assistant|>"}  # paperkiln/1


def E(section, msg, rule=None):
    """A message: "IR_SPEC <section> (rule <n>): <what>" for a refusal, "IR_SPEC <section>: <what>" otherwise."""
    return f"IR_SPEC {section} (rule {rule}): {msg}" if rule is not None else f"IR_SPEC {section}: {msg}"


class DocumentError(ValueError):
    """A text that is not a document: not JSON (rule 1), or a duplicate member name (rule 35)."""


def _strict_pairs(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise DocumentError(E("3.2", f"the member name {k!r} occurs twice in one object (a duplicate is refused at "
                                         f"any depth, '_' members included)", 35))
        out[k] = v
    return out


def load_document(text):
    """The document a JSON text holds (section 3.2): DocumentError for a text that is not JSON (rule 1) or that
    holds a duplicate member name in any object, '_' members included (rule 35)."""
    if isinstance(text, bytes):
        text = text.decode("utf-8")

    def bad_constant(name):
        raise DocumentError(E("3.2", f"{name} is not JSON", 1))
    try:
        return json.loads(text, object_pairs_hook=_strict_pairs, parse_constant=bad_constant)
    except DocumentError:
        raise
    except ValueError as e:
        raise DocumentError(E("3.2", f"not a JSON text ({e})", 1)) from None


def text_errors(text):
    """Every refusal of a document given as its JSON text: rule 35 and the text's parse (rule 1) first, then
    doc_errors."""
    try:
        doc = load_document(text)
    except DocumentError as e:
        return [str(e)]
    return doc_errors(doc)


# ---------------------------------------------------------------- the digest (section 3.10)

class DigestError(ValueError):
    """The value has no digest (section 3.10): not I-JSON, a duplicate key, or a negative zero."""


def _no_duplicates(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise DigestError(f"duplicate key {k!r}")
        out[k] = v
    return out


def _es_number(x):
    """ECMAScript Number::toString of a finite double (RFC 8785 section 3.2.2.3)."""
    if x != x or x in (float("inf"), float("-inf")):
        raise DigestError("a number is not finite")
    if x == 0:
        if str(x).startswith("-"):
            raise DigestError("a negative zero (RFC 8785 writes it as 0, the digest would equal its +0 twin's)")
        return "0"
    if x < 0:
        return "-" + _es_number(-x)
    r = repr(x)  # the shortest round-tripping digits, the closest of them (David Gay's mode 0)
    mant, _, exp = r.partition("e")
    e = int(exp) if exp else 0
    if "." in mant:
        ip, fp = mant.split(".")
    else:
        ip, fp = mant, ""
    digits = (ip + fp).lstrip("0")
    lead = len(ip + fp) - len((ip + fp).lstrip("0"))
    n = len(ip) + e - lead  # x = 0.digits * 10^n
    digits = digits.rstrip("0")
    k = len(digits)
    if k <= n <= 21:
        return digits + "0" * (n - k)
    if 0 < n <= 21:
        return digits[:n] + "." + digits[n:]
    if -6 < n <= 0:
        return "0." + "0" * (-n) + digits
    sign = "+" if n - 1 >= 0 else "-"
    return (digits if k == 1 else digits[0] + "." + digits[1:]) + "e" + sign + str(abs(n - 1))


_ESC = {'"': '\\"', "\\": "\\\\", "\b": "\\b", "\t": "\\t", "\n": "\\n", "\f": "\\f", "\r": "\\r"}


def _es_string(s):
    """JSON.stringify of a string (RFC 8785 section 3.2.2.2): '"', '\\' and the controls escaped, the
    rest as is; a lone surrogate is not well-formed Unicode and has no digest."""
    out = ['"']
    for ch in s:
        if 0xD800 <= ord(ch) <= 0xDFFF:
            raise DigestError("a string holds a lone surrogate")
        if ch in _ESC:
            out.append(_ESC[ch])
        elif ord(ch) < 0x20:
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def _canon(v):
    if v is None:
        return "null"
    if v is True:
        return "true"
    if v is False:
        return "false"
    if isinstance(v, int):
        if abs(v) > 2 ** 53:
            raise DigestError(f"the integer {v} is outside I-JSON's exact range (2^53)")
        return _es_number(float(v))
    if isinstance(v, float):
        return _es_number(v)
    if isinstance(v, str):
        return _es_string(v)
    if isinstance(v, list):
        return "[" + ",".join(_canon(x) for x in v) + "]"
    if isinstance(v, dict):
        keys = sorted(v, key=lambda k: k.encode("utf-16-be", errors="surrogatepass"))  # UTF-16 code units
        return "{" + ",".join(_es_string(k) + ":" + _canon(v[k]) for k in keys) + "}"
    raise DigestError(f"not a JSON value: {type(v).__name__}")


def _strip_prose(v):
    """Every member whose key starts with '_' removed, at any depth (section 3.2: prose)."""
    if isinstance(v, dict):
        return {k: _strip_prose(x) for k, x in v.items() if not k.startswith("_")}
    if isinstance(v, list):
        return [_strip_prose(x) for x in v]
    return v


def _parse_ijson(text):
    def bad_constant(name):
        raise DigestError(f"{name} is not JSON")
    return json.loads(text, object_pairs_hook=_no_duplicates, parse_constant=bad_constant)


def canonical(doc, prose=False):
    """The RFC 8785 serialisation of a document (a JSON text, or a parsed value) with its '_' keys removed
    (kept with prose=True, for testing the serialiser on arbitrary values). Raises DigestError."""
    v = _parse_ijson(doc) if isinstance(doc, (str, bytes)) else doc
    if prose:
        return _canon(v)
    _canon(v)  # step 1 on the whole document, its '_' members included: no digest if any check fails
    return _canon(_strip_prose(v))


def digest(doc, prose=False):
    """The IR digest (section 3.10): sha256 of canonical(doc) in UTF-8, as 64 lowercase hex digits."""
    import hashlib
    return hashlib.sha256(canonical(doc, prose).encode("utf-8")).hexdigest()


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
    version = take("<I")  # section 3.1: GGUF version 3 (version 2 has the same little-endian layout)
    if version not in (2, 3):
        raise ValueError(f"{path}: GGUF version {version} (this reader reads versions 2 and 3)")
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
    ir.schema.json uses: $ref, type, enum, const, required, properties, patternProperties,
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


# Rule 5: the closed sets of attribute values, and the section that defines each.
CLOSED = {"approx": "5.8", "pairing": "5.14", "rope_heads": "5.14", "mask": "6.8.2", "gate_norm": "5.16",
          "activation": "5.16", "feature_map": "6.4", "route": "7.7", "gate": "7.4", "tensor_index": "7.2"}
SLOT_CLOSED = {"evict": "6.8.3", "kv_rows": "7.5"}
_NODE = re.compile(r"^doc\.(embed|block|head)\.nodes\[(\d+)\]")
_SLOT = re.compile(r"^doc\.block\.state\[(\d+)\]")


def _schema_rule(doc, path, msg):
    """(rule, section) of a schema failure: the rule of section 10.1 that names it (rule 1 covers only what no
    other rule names: the types and closed key sets outside nodes, the required sections)."""
    m = _NODE.match(path)
    if m:
        rest = path[m.end():]
        node = doc[m.group(1)]["nodes"][int(m.group(2))]
        op = node.get("op") if isinstance(node, dict) else None
        if rest == "":
            if msg.startswith("has the key"):
                return 4, "3.5"
            if msg.startswith("lacks the required key 'op'"):
                return 4, "3.5"
            if msg.startswith(("lacks the required key 'in'", "lacks the required key 'out'")):
                return 7, "3.5"
            if msg.startswith("lacks the required key"):
                return 8, "3.5"
            return 1, "3.2"
        attr = rest[1:].split(".")[0].split("[")[0]
        if attr == "op":
            return 4, "3.5"
        if op == "recurrence" and attr in ("index", "inputs", "weights", "init", "let", "update", "readout"):
            return 31, "6.10"
        if op == "attention" and attr == "position_bias":
            return 39, "5.15"
        if attr in CLOSED and "is not one of" in msg:
            return 5, CLOSED[attr]
        if attr in ("in", "out"):
            return 7, "3.5"
        return 1, "3.2"
    m = _SLOT.match(path)
    if m:
        slot = doc["block"]["state"][int(m.group(1))]
        kind = slot.get("kind") if isinstance(slot, dict) else None
        attr = path[m.end():][1:].split(".")[0].split("[")[0]
        if attr == "kind":
            return 11, "6.1"
        if kind == "recurrent" and attr in ("", "op", "dtype", "tensors"):
            return 13, "6.3"
        if kind == "kv" and attr in SLOT_CLOSED:
            return 5, SLOT_CLOSED[attr]
        if kind == "kv" and attr == "dtype":
            return 5, "6.2"
        if kind == "kv" and (attr == "" and msg.startswith("lacks the required key")
                             or attr in ("heads", "kv_heads", "head_dim", "max_len", "window", "sinks")):
            return 12, "6.2"
        return 1, "3.2"
    if path.startswith("doc.model") and any(path == f"doc.model.{k}" or msg == f"lacks the required key {k!r}"
                                            for k in ("vocab", "d", "n_layers", "ctx")):
        return 3, "3.3"
    if path == "doc.block.repeat":
        return 3, "3.3"
    if path == "doc.ir_version":
        return 2, "2.3"
    return 1, "3.2"


def schema_errors(doc):
    """The schema's failures as messages, each under the rule of section 10.1 that names it."""
    out = _collect(schema(), doc, schema(), "doc")
    seen, msgs = set(), []
    for p, m in out:
        if (p, m) not in seen:
            seen.add((p, m))
            try:
                rule, sect = _schema_rule(doc, p, m)
            except (KeyError, IndexError, TypeError):
                rule, sect = 1, "3.2"
            msgs.append(E(sect, f"{p} {m}", rule))
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
    if node.get("op") == "recurrence" and isinstance(node.get("weights"), dict):
        for key, v in node["weights"].items():
            if isinstance(v, str):
                yield "weights." + key, v
    pb = node.get("position_bias")
    if node.get("op") == "attention" and isinstance(pb, dict) and isinstance(pb.get("weight"), str):
        yield "position_bias.weight", pb["weight"]


def _recurrence():
    sys.path.insert(0, str(HERE))
    try:
        import ir_recurrence  # noqa: PLC0415 (standard library only for parsing and checking)
    finally:
        if sys.path and sys.path[0] == str(HERE):
            sys.path.pop(0)
    return ir_recurrence


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
                    errs.append(E("3.8", f"{where}: per-layer {k!r} outside the block", 6))
                elif k in NOT_PER_LAYER:
                    errs.append(E("3.8", f"{where}: {k!r} is structural and cannot be per-layer", 6))
                elif not isinstance(v["per_layer"], list) or len(v["per_layer"]) != L:
                    got = len(v["per_layer"]) if isinstance(v["per_layer"], list) else "no list"
                    errs.append(E("3.8", f"{where}: per-layer {k!r} has {got} entries for block.repeat {L}", 6))
        # Rule 7: arity.
        want = ARITY.get(op)
        if want is not None and len(ins) != want:
            errs.append(E("3.5", f"{where} reads {len(ins)} value(s), not {want}", 7))
        if op in AT_LEAST_ONE and not ins:
            errs.append(E("3.5", f"{where} reads no value", 7))
        if op == "expr" and len(ins) > 23:
            errs.append(E("5.19", f"{where} reads {len(ins)} values (at most 23)", 7))
        if op in ("split", "rope"):
            if not isinstance(outs, list):
                errs.append(E("3.5", f"{where} writes a list of names", 7))
            elif op == "rope" and len(outs) != 2:
                errs.append(E("5.14", f"{where} writes {len(outs)} values, not [q, k]", 7))
        elif op is not None and not isinstance(outs, str):
            errs.append(E("3.5", f"{where} writes one value", 7))
        # Rules 31-33: a recurrence's declarations and bodies (6.10).
        if op == "recurrence":
            errs += [re.sub(r"^(IR_SPEC 6\.10 \(rule \d+\)): ", lambda g: f"{g.group(1)}: {where}: ", m)
                     for m in _recurrence().errors(n)]
        # Rule 9: expr parses.
        if op == "expr" and isinstance(n.get("expr"), str):
            msg = _expr_error(n["expr"], max(1, len(ins)))
            if msg:
                errs.append(E("5.19", f"{where}: {msg}", 9))
        # Rule 10 and 14: values.
        if op == "split" and isinstance(n.get("sizes"), list) and isinstance(outs, list) and len(outs) != len(n["sizes"]):
            errs.append(E("5.5", f"{where}: {len(n['sizes'])} sizes for {len(outs)} outputs", 10))
        if op == "rope" and L:
            hd, nr, fs = _layers(n.get("head_dim"), L), _layers(n.get("n_rot"), L), _layers(n.get("freq_scale"), L)
            for i in range(L):
                if nr and hd and nr[i] is not None and isinstance(hd[i], int) and isinstance(nr[i], int) \
                        and not (0 < nr[i] <= hd[i] and nr[i] % 2 == 0):
                    errs.append(E("5.14", f"{where}: n_rot {nr[i]} is not even and in 1 .. head_dim {hd[i]}", 10))
                    break
                if fs and fs[i] is not None and isinstance(fs[i], (int, float)) and not fs[i] > 0:
                    errs.append(E("5.14", f"{where}: freq_scale {fs[i]} is not positive", 10))
                    break
        if op == "attention" and L:
            h, kv, w, s, c = (_layers(n.get(k), L) for k in ("heads", "kv_heads", "window", "sinks", "causal"))
            for i in range(L):
                if h and kv and isinstance(h[i], int) and isinstance(kv[i], int) and kv[i] > 0 and h[i] % kv[i]:
                    errs.append(E("5.15", f"{where}: heads {h[i]} is not a multiple of kv_heads {kv[i]}", 10))
                    break
            for i in range(L):
                if (w and isinstance(w[i], int) and w[i] < 0) or (s and isinstance(s[i], int) and s[i] < 0):
                    errs.append(E("5.15", f"{where}: window and sinks must be >= 0", 10))
                    break
            if c and any(x is not True for x in c):
                errs.append(E("5.15", f"{where}: causal must be true (Q6)", 14))
        if op == "linear_attention" and n.get("causal") is not True:
            errs.append(E("6.4", f"{where}: causal must be true", 14))
        if op == "moe_ffn" and isinstance(n.get("top_k"), int) and isinstance(n.get("n_experts"), int) \
                and not 1 <= n["top_k"] <= n["n_experts"]:
            errs.append(E("5.16", f"{where}: top_k {n['top_k']} is not in 1 .. n_experts {n['n_experts']}", 10))
        if op == "conv1d_causal" and ((isinstance(n.get("channels"), int) and n["channels"] < 1)
                                      or (isinstance(n.get("kernel"), int) and n["kernel"] < 2)):
            errs.append(E("6.6", f"{where}: channels must be >= 1 and kernel >= 2", 10))
        # Names: {i} outside the block or twice (3.7); tensor_index (rule 19).
        for key, name in _tensor_names(n):
            if "{i}" in name and sect != "block":
                errs.append(E("3.7", f"{where}: {key} {name!r} uses {{i}} outside the block", "17a"))
            if name.count("{i}") > 1:
                errs.append(E("3.7", f"{where}: {key} {name!r} holds {{i}} more than once", "17a"))
        # The graph inputs (3.5): tokens for embeddings, pos as rope's third and pos_embedding's second input.
        for k, nm in enumerate(ins):
            allowed = {"tokens": op == "embedding" and k == 0,
                       "pos": (op == "rope" and k == 2) or (op == "pos_embedding" and k == 1)}
            if nm in allowed and not allowed[nm]:
                errs.append(E("3.5", f"{where} reads the graph input {nm!r} as its input {k}", "17a"))
        if (op == "rope" and len(ins) == 3 and ins[2] != "pos") or (op == "pos_embedding" and len(ins) == 2
                                                                     and ins[1] != "pos"):
            errs.append(E("3.5", f"{where}: the position input must be the graph input 'pos'", "17a"))
        if op == "embedding" and ins and ins[0] != "tokens":
            errs.append(E("3.5", f"{where}: an embedding reads the graph input 'tokens'", "17a"))
        for o in (outs if isinstance(outs, list) else [outs]):
            if o in ("tokens", "pos"):
                errs.append(E("3.5", f"{where} writes the graph input {o!r}", "17a"))
        if "tensor_index" in n:
            if sect != "block" or "weight_index" not in blk:
                errs.append(E("7.2", f"{where}: tensor_index needs a block with weight_index", 19))
    return errs


def state_errors(doc):
    """Rules 3, 11-13 and 18: the model fields, the slots, the stateful nodes, weight_index."""
    errs = []
    model = doc.get("model") if isinstance(doc.get("model"), dict) else {}
    blk = doc.get("block") if isinstance(doc.get("block"), dict) else {}
    for k in ("d", "vocab", "ctx", "n_layers"):
        if not isinstance(model.get(k), int) or isinstance(model.get(k), bool) or model[k] <= 0:
            errs.append(E("3.3", f"model.{k} missing or not a positive integer", 3))
    if blk.get("repeat") != model.get("n_layers"):
        errs.append(E("3.3", f"block.repeat {blk.get('repeat')} != model.n_layers {model.get('n_layers')}", 3))
    L = blk.get("repeat") if isinstance(blk.get("repeat"), int) else 0
    if "weight_index" in blk:
        m = blk["weight_index"].get("per_layer") if isinstance(blk["weight_index"], dict) else None
        if not isinstance(m, list) or len(m) != L or not all(isinstance(x, int) and not isinstance(x, bool) and x >= 0
                                                              for x in m):
            errs.append(E("7.1", "block.weight_index must be {per_layer: [block.repeat integers >= 0]}", 18))
        elif sorted(set(m)) != list(range(max(m) + 1)):
            errs.append(E("7.1", f"block.weight_index uses stored layers {sorted(set(m))}, not every one of 0..{max(m)}", 18))
    slots = {}
    for s in blk.get("state", []) if isinstance(blk.get("state"), list) else []:
        if not isinstance(s, dict):
            continue
        name = s.get("name")
        if name in slots:
            errs.append(E("6.1", f"state slot {name} declared twice", 11))
        slots[name] = s
        for k, v in s.items():
            if isinstance(v, dict) and "per_layer" in v:
                if k in SLOT_NOT_PER_LAYER:
                    errs.append(E("3.8", f"state slot {name}: {k!r} is structural and cannot be per-layer", 6))
                elif not isinstance(v["per_layer"], list) or len(v["per_layer"]) != L:
                    errs.append(E("3.8", f"state slot {name}: per-layer {k!r} needs block.repeat {L} entries", 6))
        if s.get("kind") == "kv":
            for k in ("heads", "kv_heads", "head_dim", "max_len", "window", "sinks"):
                v = s.get(k)
                if not (isinstance(v, int) or (isinstance(v, dict) and "per_layer" in v)):
                    errs.append(E("6.2", f"kv state {name}: {k} missing", 12))
            if "kv_from" in s:
                errs.append(E("7.6", f"kv state {name}: kv_from is reserved and not implemented", 22))
        elif s.get("kind") == "recurrent":
            ts = s.get("tensors")
            if s.get("op") not in RECURRENT_OPS or s.get("dtype") != "f32" or not ts:
                errs.append(E("6.3", f"recurrent state {name}: op (a recurrent op), dtype f32 and tensors required", 13))
            for t in ts if isinstance(ts, list) else []:
                init_ok = isinstance(t, dict) and (t.get("init") == 0 or (t.get("init") == "body"
                                                                          and s.get("op") == "recurrence"))
                if not isinstance(t, dict) or not t.get("name") or not t.get("shape") or not isinstance(t["shape"], list) \
                        or not all(isinstance(x, int) and x > 0 for x in t["shape"]) or not init_ok:
                    errs.append(E("6.3", f"recurrent state {name}: tensor {t} needs name, shape and init 0 "
                                         f"(or 'body', for a recurrence)", 13))
        else:
            errs.append(E("6.1", f"state slot {name}: unknown kind {s.get('kind')}", 11))
    used, readers = set(), {}
    for sect, j, node in _sections(doc):
        want = STATEFUL.get(node.get("op"))
        if want is None:
            if "state" in node:
                errs.append(E("3.5", f"{sect}[{j}] {node.get('op')} names state {node['state']!r} but has no state", 4))
            continue
        if sect != "block":
            errs.append(E("6.1", f"{sect}: {node['op']} outside the block has no state", 11))
            continue
        slot = slots.get(node.get("state"))
        if slot is None or slot.get("kind") != want:
            errs.append(E("6.1", f"{sect}: {node['op']} reads state {node.get('state')!r}, not a declared {want} slot",
                          11))
            continue
        if want == "recurrent" and slot.get("op") != node["op"]:
            errs.append(E("6.3", f"{sect}: {node['op']} reads the recurrent state {node.get('state')!r} of op "
                                 f"{slot.get('op')!r}", 13))
            continue
        used.add(node.get("state"))
        if node["op"] in ("attention", "recurrence"):
            readers[node.get("state")] = readers.get(node.get("state"), 0) + 1
        errs += _slot_fit(node, slot, L)
    for name, n in readers.items():
        if n > 1 and slots.get(name, {}).get("kind") == "kv":
            errs.append(E("6.1", f"kv state {name} is read by {n} attention nodes (exactly one is defined)", "17a"))
        elif n > 1:
            errs.append(E("6.10", f"recurrence state {name} is read by {n} nodes (exactly one is defined)", 33))
    for name in slots:
        if name not in used:
            errs.append(E("6.1", f"state slot {name} is read by no node", 11))
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
                                          f"{slot.get('name')}'s {b[i]}" + (" (Q7)" if key in ("kv_heads", "head_dim") else ""), 12))
                    break
        return errs
    if op == "recurrence":
        return [m for m in _recurrence().errors(node, slot) if "rule 33" in m]
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
        errs.append(E("6.3", f"recurrent state {slot.get('name')}: tensors {shapes} do not fit {op} (want {want})", 13))
    return errs


def depth_route_errors(doc):
    """Rules 20-21 and 23: depth_route first and once, its steps, the router name, kv_rows active."""
    errs = []
    blk = doc.get("block") if isinstance(doc.get("block"), dict) else {}
    nodes = blk.get("nodes") if isinstance(blk.get("nodes"), list) else []
    for sect, j, n in _sections(doc):
        if n.get("op") == "depth_route" and sect != "block":
            errs.append(E("7.4", f"depth_route outside the block ({sect})", 20))
    at = [j for j, n in enumerate(nodes) if isinstance(n, dict) and n.get("op") == "depth_route"]
    slots = [s for s in blk.get("state", []) if isinstance(s, dict) and s.get("kind") == "kv"]
    for s in slots:
        if "kv_rows" in s and s["kv_rows"] != "active":
            errs.append(E("7.5", f"kv state {s.get('name')}: kv_rows {s['kv_rows']!r} (only \"active\" is defined)", 5))
    if not at:
        return errs
    if at != [0]:
        return errs + [E("7.4", "depth_route must be the block's first node, once", 20)]
    r = nodes[0]
    R = r.get("n_rec")
    if not isinstance(R, int) or isinstance(R, bool) or R < 1:
        return errs + [E("7.4", "depth_route: n_rec must be an integer >= 1", 20)]
    if r.get("route") != "token_choice":
        errs.append(E("7.7", f"depth_route: route {r.get('route')!r} (only token_choice, causal, is defined)", 5))
    if r.get("gate") not in ("score", "none"):
        errs.append(E("7.4", f"depth_route: gate {r.get('gate')!r} is not score or none", 5))
    w = r.get("weight")
    if not isinstance(w, str) or "{" in w:
        errs.append(E("7.7", "depth_route: weight must be the router's full tensor name (constraint 4: no substitution)", 20))
    if not isinstance(r.get("out"), str):
        errs.append(E("7.4", "depth_route: out must name the depth trace", 20))
    st = r.get("step").get("per_layer") if isinstance(r.get("step"), dict) else None
    if not isinstance(st, list) or not all(isinstance(x, int) and not isinstance(x, bool) for x in st):
        return errs + [E("7.4", "depth_route: step must be {per_layer: [block.repeat integers]}", 20)]
    if len(st) != blk.get("repeat"):
        return errs  # rule 6 (a per_layer list of another length)
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
        errs.append(E("7.4", f"depth_route: steps {st} are not -1 or the consecutive steps 0 .. {R - 1} in order", 20))
    elif len({n for _, n in runs}) != 1:
        errs.append(E("7.4", f"depth_route: steps {st} differ in length", 20))
    if not slots or any(s.get("kv_rows") != "active" for s in slots):
        errs.append(E("7.5", "depth_route needs kv_rows \"active\" on every kv slot", 21))
    if any(n.get("op") == "attention" and n.get("mask") == "turn_state" for n in nodes if isinstance(n, dict)):
        errs.append(E("9.2", "depth_route with mask turn_state is not defined (llama_state)", 23))
    if any(n.get("op") in ("list", "append", "depth_attend") for _, _, n in _sections(doc)):
        errs.append(E("7.4", "depth_route with list values (list, append, depth_attend) is not defined", 23))
    out = r.get("out")
    for sect, j, n in _sections(doc):
        if isinstance(out, str) and out in (n.get("in") or []):
            errs.append(E("7.4", f"{sect}[{j}] {n.get('op')} reads depth_route's out {out!r} (the depth report)", "17a"))
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
                errs.append(E("6.8.1", f"{sect}[{j}]: an embedding state_weight outside the embed section (Q8)", 17))
            if isinstance(n["state_weight"], str) and "{" in n["state_weight"]:
                errs.append(E("6.8.1", "state_weight names its tensor in full (Q8)", 17))
    if len(sws) > 1:
        errs.append(E("6.8.1", "more than one embedding with a state_weight (Q8)", 17))
    slots = {s.get("name"): s for s in blk.get("state", []) if isinstance(s, dict)}
    masks = {}
    for n in blk.get("nodes", []) if isinstance(blk.get("nodes"), list) else []:
        if not isinstance(n, dict) or n.get("op") != "attention":
            continue
        m = n.get("mask", "causal")
        if m not in ("causal", "turn_state"):
            errs.append(E("6.8.2", f"attention mask {m!r} (causal or turn_state)", 5))
        if m == "turn_state":
            w, s = _layers(n.get("window", 0), L) or [], _layers(n.get("sinks", 0), L) or []
            slot = slots.get(n.get("state")) or {}
            sw, ss = _layers(slot.get("window", 0), L) or [], _layers(slot.get("sinks", 0), L) or []
            if any(w) or any(s) or any(sw) or any(ss) or not sws:
                errs.append(E("6.8.2", "attention mask turn_state needs window 0, sinks 0 (every application, and its "
                                       "kv slot) and an embedding state_weight", 15))
        masks.setdefault(n.get("state"), set()).add(m)
    for st in blk.get("state", []) if isinstance(blk.get("state"), list) else []:
        if not isinstance(st, dict) or "evict" not in st:
            continue
        if st["evict"] != "turn_state":
            errs.append(E("6.8.3", f"kv state {st.get('name')}: evict {st['evict']!r} (only turn_state is defined)", 5))
        elif masks.get(st.get("name")) != {"turn_state"}:
            errs.append(E("6.8.3", f"kv state {st.get('name')}: evict turn_state needs every attention node reading it "
                                   f"to mask turn_state", 16))
    return errs


def _outs(node):
    o = node.get("out")
    return [x for x in (o if isinstance(o, list) else [o]) if isinstance(x, str)]


def read_errors(doc):
    """Rule 36 (sections 3.5 and 7.4): every value a node reads is written before it. The sections run in order
    (embed, block, head); a value stays readable in later nodes, applications and sections until it is written
    again; the block is checked as application 0, which reads nothing an application before it wrote. The embed
    section writes x and the head writes logits. Under depth_route the block's other nodes read only x, pos
    and the values written before them in the same application (a recursion step's applications start from
    the active rows of x and pos alone), and the head reads x and the embed section's values, not the
    block's other values."""
    errs = []
    sections = {k: (doc.get(k) or {}).get("nodes") if isinstance(doc.get(k), dict) else None
                for k in ("embed", "block", "head")}
    if not all(isinstance(v, list) for v in sections.values()):
        return errs  # rule 1
    blk_nodes = [n for n in sections["block"] if isinstance(n, dict)]
    routed = bool(blk_nodes) and blk_nodes[0].get("op") == "depth_route"

    def walk(sect, nodes, have):
        for j, n in enumerate(nodes):
            if not isinstance(n, dict):
                continue
            for k, nm in enumerate(n.get("in") if isinstance(n.get("in"), list) else []):
                if isinstance(nm, str) and nm not in have:
                    why = (" (under depth_route, the block's nodes read x, pos and the values written before them in "
                           "the same application)" if routed and sect == "block" else
                           " (under depth_route, the head reads x and the embed section's values)" if routed and
                           sect == "head" else "")
                    errs.append(E("7.4" if why else "3.5", f"{sect}[{j}] {n.get('op')} reads {nm!r} (its input {k}), "
                                  f"which no earlier node writes{why}", 36))
            have |= set(_outs(n))
        return have

    embed = walk("embed", sections["embed"], {"tokens", "pos"})
    if "x" not in embed:
        errs.append(E("3.5", "the embed section writes no x", 36))
    if routed:
        walk("block", blk_nodes[:1], set(embed))
        walk("block", blk_nodes[1:], {"x", "pos"})
        walk("head", sections["head"], set(embed) | {"x"})
    else:
        block = walk("block", sections["block"], set(embed))
        walk("head", sections["head"], block)
    if not any("logits" in _outs(n) for n in sections["head"] if isinstance(n, dict)):
        errs.append(E("3.4", "the head writes no logits", 36))
    return errs


def doc_errors(doc, versions=VERSIONS):
    """Every document rule (10.1 rules 1-23, 31-33, 36). The schema runs first; the semantic rules run on what
    the schema leaves usable, so one fault does not cascade into many messages where it can be avoided."""
    if not isinstance(doc, dict):
        return [E("3.2", "the document is not a JSON object", 1)]
    errs = schema_errors(doc)
    v = doc.get("ir_version")
    if isinstance(v, int) and not isinstance(v, bool) and v not in versions:
        errs.append(E("2.3", f"ir_version {v} is not supported (this validator reads {', '.join(map(str, versions))})", 2))
    errs += node_errors(doc)
    errs += state_errors(doc)
    errs += depth_route_errors(doc)
    errs += state_token_doc_errors(doc)
    errs += read_errors(doc)
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


def depth_attend_sources(doc):
    """{tensor name: set of S} for every depth_attend pseudo-query the document reads, S its source count
    at each use (IR_SPEC 5.18: S is structural, fixed by the list and append nodes before it)."""
    out, lens = {}, {}
    blk = doc.get("block", {})
    wmap = (blk.get("weight_index") or {}).get("per_layer") if isinstance(blk.get("weight_index"), dict) else None

    def walk(nodes, i=None, m=None):
        for n in nodes:
            ins = n.get("in") or []
            if n.get("op") == "list" and isinstance(n.get("out"), str):
                lens[n["out"]] = len(ins)
            elif n.get("op") == "append" and isinstance(n.get("out"), str) and ins and ins[0] in lens:
                lens[n["out"]] = lens[ins[0]] + 1
            elif n.get("op") == "depth_attend" and isinstance(n.get("weight"), str) and ins and ins[0] in lens:
                t = i if n.get("tensor_index") == "application" else m
                name = n["weight"].replace("{i}", str(t)) if t is not None else n["weight"]
                out.setdefault(name, set()).add(lens[ins[0]] + len(ins) - 1)

    walk(doc.get("embed", {}).get("nodes", []))
    for i in range(blk.get("repeat", 0) if isinstance(blk.get("repeat"), int) else 0):
        walk(blk.get("nodes", []), i, wmap[i] if isinstance(wmap, list) and i < len(wmap) else i)
    walk(doc.get("head", {}).get("nodes", []))
    return out


def gguf_report(doc, meta, tensors):
    """(errors, warnings) of the container rules 24-28 (section 10.1) for a document read with a GGUF.
    Rule 24 compares the version key with the file's own embedded document when it has one (a document read
    in place of it, as ember's EMBER_IR, is governed by its own ir_version: rule 2). A missing pseudo-query
    that only S = 1 depth_attend nodes name is a warning, the exporter rule of 5.18, not a refusal."""
    errs, warns = [], []
    ver = meta.get("paperkiln.ir_version")
    embedded = meta.get("paperkiln.ir")
    try:
        emb_ver = json.loads(embedded[1]).get("ir_version") if embedded and embedded[0] == 8 else doc.get("ir_version")
    except ValueError:
        emb_ver = doc.get("ir_version")
    if ver is None or ver[0] != 4 or ver[1] != emb_ver:
        errs.append(E("3.1", f"paperkiln.ir_version {ver} does not match the embedded document's ir_version "
                             f"{emb_ver}", 24))
    sha = meta.get("paperkiln.ir_sha256")
    if sha is not None:  # rule 30: the digest key names the embedded document (section 3.10)
        try:
            want = digest(embedded[1]) if embedded and embedded[0] == 8 else None
        except DigestError as e:
            want = f"none ({e})"
        if sha[0] != 8 or sha[1] != want:
            errs.append(E("3.10", f"paperkiln.ir_sha256 {sha[1]!r} is not the digest of the embedded document "
                                  f"({want})", 30))
    have = {t[0] for t in tensors}
    dims = {t[0]: t[1] for t in tensors}
    refs = weight_refs(doc)
    if not refs:
        errs.append(E("3.7", "the document names no tensors", 25))
    single = {n for n, ss in depth_attend_sources(doc).items() if ss == {1}}
    for where, n, fb in refs:
        if n not in have and not (fb and fb in have):
            if n in single:
                warns.append(E("5.18", f"exporter rule: {where}: the S = 1 pseudo-query {n} is not a tensor of the "
                                       f"file (exporters write it; the document stays valid)"))
            else:
                errs.append(E("3.7", f"{where}: {n} is not a tensor of the file", 25))
    arch = meta.get("general.architecture", (None, None))[1]
    for node in doc.get("block", {}).get("nodes", [])[:1]:
        if node.get("op") == "depth_route":
            got = dims.get(node.get("weight"))
            want = [doc.get("model", {}).get("d"), node.get("n_rec")]
            if got is not None and got != want:
                errs.append(E("7.4", f"depth_route: {node['weight']} has ne {got}, not [d, n_rec] = {want}", 26))
    wmap = (doc.get("block", {}).get("weight_index") or {}).get("per_layer")
    if isinstance(wmap, list) and wmap and all(isinstance(x, int) for x in wmap):
        if arch != "paperkiln":
            errs.append(E("7.1", f"block.weight_index needs architecture paperkiln, not {arch}", 27))
        bc = meta.get(f"{arch}.block_count", (None, None))[1]
        if bc != max(wmap) + 1:
            errs.append(E("7.1", f"{arch}.block_count {bc} != the {max(wmap) + 1} stored layers weight_index uses", 27))
    errs += recurrence_gguf_errors(doc, dims)
    errs += state_token_gguf_errors(doc, meta, tensors)
    errs += position_bias_gguf_errors(doc, dims)
    errs += architecture_errors(meta, tensors)
    return errs, sorted(set(warns))


def position_bias_gguf_errors(doc, dims):
    """Rule 39 (container): an attention node's position_bias tensor is ne {heads, B}, B above every bucket
    (section 5.15), in every application that reads it."""
    errs = []
    blk = doc.get("block", {})
    L = blk.get("repeat") if isinstance(blk.get("repeat"), int) else 0
    wmap = (blk.get("weight_index") or {}).get("per_layer") if isinstance(blk.get("weight_index"), dict) else None
    for sect, j, n in _sections(doc):
        pb = n.get("position_bias")
        if n.get("op") != "attention" or not isinstance(pb, dict) or not isinstance(pb.get("weight"), str):
            continue
        bk = pb.get("buckets")
        if not (isinstance(bk, list) and bk and all(isinstance(b, int) and not isinstance(b, bool) for b in bk)):
            continue  # the schema's part of rule 39
        heads = _layers(n.get("heads"), L or 1) or []
        apps = range(L) if sect == "block" else [None]
        for i in apps:
            h = heads[i] if i is not None and i < len(heads) else (heads[0] if heads else None)
            m = (wmap[i] if isinstance(wmap, list) and i < len(wmap) else i) if i is not None else None
            name = pb["weight"].replace("{i}", str(m)) if m is not None else pb["weight"]
            got = dims.get(name)
            if got is None:
                continue  # rule 25
            if len(got) != 2 or got[0] != h or got[1] <= max(bk):
                where = f"{sect}[{j}] attention" + (f" (application {i})" if i is not None else "")
                errs.append(E("5.15", f"{where}: position_bias {name} has ne {got}, not [heads {h}, B] with B > "
                                      f"the largest bucket {max(bk)}", 39))
    return sorted(set(errs))


# ---------------------------------------------------------------- rule 38: the architecture and the document

INFORMATIVE_ATTRS = {"attention": ("softmax",), "moe_ffn": ("routing",)}
# Section 3.5's optional attributes at their defaults (rmsnorm.group's default, the row width, is not filled:
# it needs widths).
NODE_DEFAULTS = {"linear": {"bias": None, "tied_to": None}, "scale": {"bias": None},
                 "rope": {"freq_scale": 1.0}, "attention": {"mask": "causal"},
                 "rmsnorm": {"weight_offset": 0.0}, "depth_attend": {"eps": 1e-05},
                 "conv1d_causal": {"bias": None}, "selective_scan": {"d": None, "dt_softplus": True}}
COMMUTATIVE = ("add", "mul")


def _f32_value(x):
    """A JSON number as the f32 an attribute holds (section 3.5): the binary64 value rounded once."""
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def _norm_value(v):
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return v
    if isinstance(v, (int, float)):
        return _f32_value(v)
    if isinstance(v, list):
        return [_norm_value(x) for x in v]
    if isinstance(v, dict):
        return {k: _norm_value(x) for k, x in v.items() if not k.startswith("_")}
    return v


def structure(doc, have=None):
    """The document's graph up to what rule 38 ignores: '_' members, the informative keys (family_hint,
    tokenizer, inputs, outputs, modes, ir_version, the informative attributes), optional attributes written at
    their defaults, the names of values and slots, the operand order of add and mul, the spelling of numbers
    (each the f32 it rounds to) and, given the file's tensor names (have), how a linear names the tensor it
    reads (its weight, or tied_to when the file lacks the weight: section 5.3)."""
    slots = {}
    out = {"model": _norm_value(doc.get("model")), "block": {}}
    blk = doc.get("block") if isinstance(doc.get("block"), dict) else {}
    state = []
    for k, sl in enumerate(blk.get("state") or []):
        if not isinstance(sl, dict):
            state.append(sl)
            continue
        slots[sl.get("name")] = f"slot{k}"
        sl = dict(sl, name=f"slot{k}")
        if sl.get("kind") == "kv":
            sl.setdefault("dtype", "f32")
        state.append(_norm_value(sl))
    version = {"tokens": "tokens", "pos": "pos"}
    count = [0]

    def read(name):
        return version.get(name, "unwritten:" + str(name))

    def write(name):
        count[0] += 1
        version[name] = f"v{count[0]}"
        return version[name]

    def nodes(sect):
        res = []
        for n in ((doc.get(sect) or {}).get("nodes") or []):
            if not isinstance(n, dict):
                res.append(n)
                continue
            op = n.get("op")
            n = {k: v for k, v in n.items() if not k.startswith("_") and k not in INFORMATIVE_ATTRS.get(op, ())}
            for k, v in NODE_DEFAULTS.get(op, {}).items():
                n.setdefault(k, v)
            if op == "linear" and have is not None and n.get("tied_to") is not None:
                if n.get("weight") not in have:
                    n["weight"] = n["tied_to"]
                n["tied_to"] = None
            if op == "rope":
                n.setdefault("kv_heads", n.get("heads"))
                n.setdefault("n_rot", n.get("head_dim"))
            ins = [read(x) for x in n.get("in") or []]
            if op in COMMUTATIVE:
                ins = sorted(ins)
            n["in"] = ins
            outs = n.get("out")
            n["out"] = [write(o) for o in outs] if isinstance(outs, list) else write(outs)
            if "state" in n:
                n["state"] = slots.get(n["state"], n["state"])
            res.append(_norm_value(n))
        return res

    out["embed"] = nodes("embed")
    out["block"] = {k: _norm_value(v) for k, v in blk.items() if k not in ("nodes", "state") and not k.startswith("_")}
    out["block"]["state"] = state
    out["block"]["nodes"] = nodes("block")
    out["head"] = nodes("head")
    return out


def _first_difference(a, b, path="document"):
    if type(a) is not type(b):
        return f"{path}: {json.dumps(a)[:120]} against {json.dumps(b)[:120]}"
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b), key=lambda k: (k != "op", str(k))):
            if k not in a or k not in b:
                return f"{path}.{k}: {'absent' if k not in a else json.dumps(a[k])[:80]} against " \
                       f"{'absent' if k not in b else json.dumps(b[k])[:80]}"
            d = _first_difference(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: {len(a)} entries against {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = _first_difference(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    return None if a == b else f"{path}: {json.dumps(a)[:80]} against {json.dumps(b)[:80]}"


def _builder():
    sys.path.insert(0, str(HERE))
    try:
        import gguf_ir  # noqa: PLC0415 (standard library only when building)
    finally:
        if sys.path and sys.path[0] == str(HERE):
            sys.path.pop(0)
    return gguf_ir


def architecture_errors(meta, tensors):
    """Rule 38 (section 3.1): a file whose general.architecture is not "paperkiln" holds, as its embedded
    document, the graph that architecture's builder (section 9.3) builds from the file's own metadata and
    tensors, up to what structure() ignores. Checked on the embedded document only: a document read in place
    of it is not in the file a program without the IR reads."""
    arch = meta.get("general.architecture", (None, None))[1]
    emb = meta.get("paperkiln.ir")
    if arch == "paperkiln" or not emb or emb[0] != 8:
        return []
    try:
        doc = json.loads(emb[1])
    except ValueError:
        return []  # rule 1 or 35, reported with the document
    gi = _builder()
    tail = ("; a file whose model is not that architecture's graph has general.architecture \"paperkiln\", which "
            "no llama.cpp release builds a graph for")
    if arch not in gi.BUILDERS:
        return [E("3.1", f"general.architecture {arch!r} is neither \"paperkiln\" nor an architecture of section "
                         f"9.3, so nothing states what it computes{tail}", 38)]
    values = {k: v for k, (_, v) in meta.items()}
    toks = values.get("tokenizer.ggml.tokens")
    if isinstance(toks, list):
        values["_n_tokens"] = len(toks)
    shapes = {t[0]: list(reversed(t[1])) for t in tensors}
    try:
        built = gi.build(values, shapes)
    except (SystemExit, KeyError, TypeError, ValueError, ZeroDivisionError) as e:
        return [E("3.1", f"general.architecture {arch!r}: its metadata builds no graph ({e}){tail}", 38)]
    have = set(shapes)
    diff = _first_difference(structure(doc, have), structure(built, have))
    if diff:
        return [E("3.1", f"general.architecture {arch!r}: the embedded document is not the {arch} graph its "
                         f"metadata describes (section 9.3; first difference, document against graph: {diff}){tail}",
                  38)]
    return []


def recurrence_gguf_errors(doc, dims):
    """Rule 34 (container): each recurrence weight's tensor holds as many elements as its declaration."""
    errs = []
    blk = doc.get("block", {})
    wmap = (blk.get("weight_index") or {}).get("per_layer") if isinstance(blk.get("weight_index"), dict) else None
    for j, node in enumerate(blk.get("nodes", [])):
        if not isinstance(node, dict) or node.get("op") != "recurrence":
            continue
        try:
            prog = _recurrence().compile_node(node)
        except ValueError:
            continue  # reported by the document rules
        for i in range(blk.get("repeat", 0) if isinstance(blk.get("repeat"), int) else 0):
            m = wmap[i] if isinstance(wmap, list) and i < len(wmap) else i
            t = i if node.get("tensor_index") == "application" else m
            for name, _, tensor in prog.weights:
                full = tensor.replace("{i}", str(t))
                if full in dims:
                    n = 1
                    for x in dims[full]:
                        n *= x
                    if n != prog.numel(name):
                        errs.append(E("6.10", f"block {i}[{j}] recurrence: weight {name} is {full}, "
                                              f"{n} elements, its declaration {prog.numel(name)}", 34))
    return sorted(set(errs))


def gguf_errors(doc, meta, tensors):
    """The refusals of gguf_report (its warnings dropped)."""
    return gguf_report(doc, meta, tensors)[0]


def state_token_gguf_errors(doc, meta, tensors):
    """Rule 28: paperkiln.state_tokens.k, the state_weight's shape, the architecture, the specials and
    the template."""
    errs = []
    dims = {t[0]: t[1] for t in tensors}
    k = meta.get("paperkiln.state_tokens.k", (None, None))[1]
    sws = [n.get("state_weight") for n in doc.get("embed", {}).get("nodes", [])
           if n.get("op") == "embedding" and "state_weight" in n]
    if (k is not None and k > 0) != bool(sws):
        errs.append(E("6.8.1", f"paperkiln.state_tokens.k {k} without an embedding state_weight, or the reverse", 28))
    for sw in sws:
        want = [doc.get("model", {}).get("d"), k]
        if sw not in dims:
            continue  # rule 25 (every tensor the document names)
        if dims[sw] != want:
            errs.append(E("6.8.1", f"embedding state_weight {sw} has ne {dims[sw]}, not [d, paperkiln.state_tokens.k] "
                                   f"= {want}", 28))
    if sws:
        arch = meta.get("general.architecture", (None, None))[1]
        if arch != "paperkiln":
            errs.append(E("8.2", f"state tokens need architecture paperkiln, not {arch}", 28))
        sp = (doc.get("tokenizer") or {}).get("specials") or {}
        toks = meta.get("tokenizer.ggml.tokens", (None, None))[1]
        for role in ("eot", "user", "assistant"):
            if not isinstance(sp.get(role), int):
                errs.append(E("8.2", f"state tokens: the tokenizer names no {role} id (the template needs it)", 28))
            elif isinstance(toks, list) and not (0 <= sp[role] < len(toks) and toks[sp[role]] == MARKERS[role]):
                got = toks[sp[role]] if 0 <= sp[role] < len(toks) else None
                errs.append(E("8.2", f"state tokens: specials.{role} = {sp[role]} is the token {got!r}, not "
                                     f"{MARKERS[role]!r}", 28))
        tmpl = meta.get("paperkiln.chat_template", (None, None))[1]
        if tmpl is not None:
            try:
                rec = json.loads(tmpl)
                name = f"{rec.get('name')}/{rec.get('version')}" if "/" not in str(rec.get("name")) else rec.get("name")
            except (ValueError, AttributeError):
                name = None
            if name not in ("paperkiln/1",):
                errs.append(E("8.2", f"state tokens are defined over paperkiln/1 only, not {name}", 28))
    return errs


def file_report(path):
    """(errors, warnings) of a .json document, or of a GGUF with paperkiln.ir (its sidecar too)."""
    path = Path(path)
    if path.suffix == ".gguf":
        meta, tensors, *_ = read_gguf(path)
        if "paperkiln.ir" not in meta or meta["paperkiln.ir"][0] != 8:
            return [E("3.1", f"{path}: no string key paperkiln.ir")], []
        try:
            doc = load_document(meta["paperkiln.ir"][1])
        except DocumentError as e:
            return [str(e)], []
        gerrs, warns = gguf_report(doc, meta, tensors)
        errs = doc_errors(doc) + gerrs
        side = path.with_suffix(".ir.json")
        if side.is_file() and json.loads(side.read_text(encoding="utf-8")) != doc:
            errs.append(E("3.1", f"sidecar {side} differs from the embedded document", 29))
        return errs, warns
    return text_errors(path.read_text(encoding="utf-8")), []


def file_errors(path):
    """The refusals of file_report."""
    return file_report(path)[0]


def check_errors(gguf=None, ir=None):
    """The refusals of a check (IR_SPEC 11.3): a GGUF with its own document, a document read in place of it
    (ir: a path), or a document alone (gguf None: the document rules, no container rules)."""
    if ir is None:
        return file_errors(gguf)
    try:
        doc = load_document(Path(ir).read_bytes())
    except DocumentError as e:
        return [str(e)]
    errs = doc_errors(doc)
    if gguf is not None:
        meta, tensors, *_ = read_gguf(gguf)
        errs += gguf_errors(doc, meta, tensors)
    return errs


# ---------------------------------------------------------------- the corpus

REPO = HERE.parent


def corpus(patterns=()):
    """Every stored document of this repository that must be accepted (the worked documents, the suite's
    models and its rewritten documents; refuse/ holds the ones that must not), and every file the glob
    patterns name (recursive "**" allowed): an engine passes its own stores (documents, and GGUFs with an
    embedded document) with --corpus-glob."""
    pats = [str(REPO / "examples" / "*.ir.json"), str(REPO / "conformance" / "models" / "*" / "*.gguf"),
            str(REPO / "conformance" / "recurrence" / "*.ir.json")] + list(patterns)
    return sorted({f for p in pats for f in glob.glob(p, recursive=True) if Path(f).name != "v.gguf"})


def document_text(path):
    """The document a file holds, as its JSON text: a .json file's text, or a GGUF's paperkiln.ir."""
    path = Path(path)
    if path.suffix == ".gguf":
        meta = read_gguf(path)[0]
        if "paperkiln.ir" not in meta or meta["paperkiln.ir"][0] != 8:
            raise ValueError(f"{path}: no string key paperkiln.ir")
        return meta["paperkiln.ir"][1]
    return path.read_text(encoding="utf-8")


def digest_main(argv):
    """`digest FILE... [--canonical]`: each file's IR digest (section 3.10), or why it has none."""
    paths = [a for a in argv if not a.startswith("--")]
    if not paths:
        print(digest_main.__doc__)
        return 2
    bad = 0
    for p in paths:
        try:
            text = document_text(p)
            print(canonical(text) if "--canonical" in argv else f"{digest(text)}  {p}")
        except (OSError, ValueError) as e:  # DigestError is a ValueError
            bad += 1
            print(f"NO DIGEST {p}: {e}")
    return 1 if bad else 0


def main(argv):
    if argv[:1] == ["digest"]:
        return digest_main(argv[1:])
    quiet = "--quiet" in argv
    globs, rest = [], []
    it = iter(argv)
    for a in it:
        if a == "--corpus-glob":
            globs.append(next(it, ""))
        else:
            rest.append(a)
    paths = [a for a in rest if not a.startswith("--")]
    if "--corpus" in rest:
        paths = corpus(globs) + paths
    if not paths:
        print(__doc__)
        return 2
    bad = 0
    for p in paths:
        try:
            errs, warns = file_report(p)
        except (OSError, ValueError) as e:
            errs, warns = [f"cannot read: {e}"], []
        bad += bool(errs)
        for w in warns:
            print(f"WARN {p}: {w}")
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
