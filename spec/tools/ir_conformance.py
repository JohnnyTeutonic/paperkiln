"""The IR conformance suite's runner and generator (IR_SPEC.md section 11; suite in conformance/).

    python tools/ir_conformance.py --adapter ir_numpy               # run the suite against the reference
    python tools/ir_conformance.py --adapter cmd:"engine --conformance"   # any engine, JSON Lines (11.3)
    python tools/ir_conformance.py --regenerate                     # expected values from tools/ir_numpy.py

Options: --only SUBSTRING (cases whose id contains it), --json RESULTS.json (per-case results), --list.

Kinds of case: op (one node on given inputs), model (prompts, greedy ids, log-probabilities, depth traces),
trace (adapter sessions: prefill, incremental decode, truncate, snapshot and restore, with kv rows held,
depths and recurrent tensors checked), refuse (documents and GGUFs every consumer must refuse).
The adapter contract is section 11.3: NumpyAdapter and CmdAdapter below implement it. An in-process
adapter (a subclass of Adapter) runs through main(argv, adapters={name: factory}), as paperkiln's
mtstudio adapter does. A case an adapter cannot run is SKIP, never PASS. Exit status 1 on any FAIL.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SUITE = ROOT / "conformance"
sys.path.insert(0, str(HERE))
import ir_numpy  # noqa: E402
import ir_validate  # noqa: E402

TOL = {"cpu": 1e-5, "cuda": 5e-5}


# ---------------------------------------------------------------- tensors (section 11.2)

def enc(a):
    a = np.ascontiguousarray(np.asarray(a, dtype=np.float32))
    return {"shape": list(a.shape), "f32": a.view(np.uint32).astype(">u4").tobytes().hex()}


def dec(t):
    if isinstance(t, list):
        return [dec(x) for x in t]
    a = np.frombuffer(bytes.fromhex(t["f32"]), dtype=">u4").astype(np.uint32).view(np.float32)
    return a.reshape(t["shape"]).copy()


def close(got, ref, cls, tol):
    """(ok, worst): EXACT is bit equality, ROUNDING |a - b| <= tol * max(1, max|ref|)."""
    got, ref = np.asarray(got, dtype=np.float32), np.asarray(ref, dtype=np.float32)
    if got.shape != ref.shape:
        return False, f"shape {list(got.shape)} != {list(ref.shape)}"
    if cls == "EXACT":  # bits, the sign of zero included; any NaN matches any NaN (IR_SPEC 4.2)
        diff = (got.view(np.uint32) != ref.view(np.uint32)) & ~(np.isnan(got) & np.isnan(ref))
        bad = int(np.sum(diff))
        return bad == 0, f"{bad} values differ in their bits"
    both_nan = np.isnan(got) & np.isnan(ref)
    d = np.where(both_nan, 0.0, np.abs(got.astype(np.float64) - ref.astype(np.float64)))
    scale = max(1.0, float(np.nanmax(np.abs(ref))) if ref.size else 1.0)
    worst = float(np.nanmax(d)) / scale if d.size else 0.0
    return bool(worst <= tol and not np.isnan(worst)), f"worst {worst:.2e} of max(1, |ref|)"


# ---------------------------------------------------------------- GGUF (F32 tensors)

def load_model(model_dir):
    """(doc, {name: f32 array, numpy order}, meta, gguf path) of the one GGUF in a model directory."""
    path = gguf_of(model_dir)
    meta, tensors, data_start, raw, _ = ir_validate.read_gguf(path)
    W = {}
    for name, dims, typ, off in tensors:
        if typ != 0:
            raise SystemExit(f"{path}: {name} is not F32 (the suite stores F32 weights only)")
        n = int(np.prod(dims))
        W[name] = np.frombuffer(raw, dtype="<f4", count=n, offset=data_start + off).reshape(list(reversed(dims))).copy()
    return json.loads(meta["paperkiln.ir"][1]), W, meta, path


def gguf_of(model_dir):
    d = Path(model_dir)
    name = d.name + ".gguf"
    if not (d / name).is_file():
        raise SystemExit(f"{d}: no {name}")
    return d / name


def _gguf_value(v):
    """(type, packed bytes) of a metadata value (read_gguf's (type, value)); arrays infer their element type."""
    def s(x):
        b = x.encode("utf-8", errors="surrogateescape")
        return struct.pack("<Q", len(b)) + b
    t, x = v
    if t == 8:
        return s(x)
    if t == 9:
        if all(isinstance(e, str) for e in x):
            et, body = 8, b"".join(s(e) for e in x)
        elif all(isinstance(e, int) and not isinstance(e, bool) for e in x):
            et, body = 5, b"".join(struct.pack("<i", e) for e in x)
        else:
            et, body = 6, b"".join(struct.pack("<f", e) for e in x)
        return struct.pack("<IQ", et, len(x)) + body
    return struct.pack(ir_validate._SCALAR[t], x)


def write_gguf(path, meta, W, ir_doc=None):
    """A GGUF v3 with the metadata meta {key: (type, value)} (paperkiln.ir and its version written last
    when ir_doc is given) and the F32 tensors W {name: numpy-order array}."""
    meta = {k: v for k, v in meta.items() if k not in ("paperkiln.ir", "paperkiln.ir_version")}
    if ir_doc is not None:
        meta["paperkiln.ir"] = (8, json.dumps(ir_doc, separators=(",", ":")))
        meta["paperkiln.ir_version"] = (4, int(ir_doc["ir_version"]))
    align = 32
    out = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(W), len(meta)))
    for k, v in meta.items():
        kb = k.encode()
        out += struct.pack("<Q", len(kb)) + kb + struct.pack("<I", v[0]) + _gguf_value(v)
    offs, blob = [], bytearray()
    for name, a in W.items():
        a = np.ascontiguousarray(a, dtype="<f4")
        while len(blob) % align:
            blob += b"\0"
        offs.append(len(blob))
        blob += a.tobytes()
    for (name, a), off in zip(W.items(), offs):
        nb = name.encode()
        dims = list(reversed(a.shape)) or [1]
        out += struct.pack("<Q", len(nb)) + nb + struct.pack("<I", len(dims)) + b"".join(
            struct.pack("<Q", x) for x in dims) + struct.pack("<IQ", 0, off)
    while len(out) % align:
        out += b"\0"
    Path(path).write_bytes(bytes(out + blob))


# ---------------------------------------------------------------- the reference rules for kv rows held

def segment_scan(ids, vocab, k):
    """(codes, block_end_positions) for one sequence (ir_numpy.turn_state_codes and where blocks complete)."""
    codes = ir_numpy.turn_state_codes(list(ids), vocab, k)
    ends, prev = [], 0
    for p in range(len(ids)):
        nxt = ir_numpy.turn_state_codes(list(ids[:p + 1]) + [0], vocab, k)[-1] >> 1 if k else 0
        if nxt > prev:
            ends.append(p)
        prev = nxt
    return codes, ends


def evict_allowed(ids, vocab, k):
    """The legal position sets a turn_state-evicting slot may hold after a call ending at len(ids)
    (IR_SPEC.md 6.8.3): eager, and lazy while no text position follows the last completed block."""
    L = len(ids)
    codes, ends = segment_scan(ids, vocab, k)
    c, s = [x & 1 for x in codes], [x >> 1 for x in codes]
    m = len(ends)
    eager = [p for p in range(L) if c[p] == 1 or s[p] == m]
    allowed = [eager]
    if m >= 1:
        q = ends[-1]
        if not any(c[p] == 0 for p in range(q + 1, L)):
            lazy = sorted(set(eager) | {p for p in range(L) if c[p] == 0 and s[p] == m - 1})
            if lazy != eager:
                allowed.append(lazy)
    return allowed


def active_positions(steps, depth, L):
    """IR_SPEC.md 7.5: application i holds the rows of tokens t < L with s_i = -1 or n_t > s_i."""
    return [[t for t in range(L) if s == -1 or depth[t] > s] for s in steps]


# ---------------------------------------------------------------- adapters (section 11.3)

class Adapter:
    """The contract's methods; anything not implemented raises NotImplementedError (reported SKIP)."""
    name, backend, exact = "?", "cpu", True
    profiles = set()
    score_only = False  # True: model cases only through score (no open/forward fallback)

    def hello(self):
        return {"engine": self.name, "backend": self.backend, "exact": self.exact, "profiles": sorted(self.profiles)}

    def check(self, gguf, ir):
        raise NotImplementedError

    def open(self, gguf, ir=None, evict=None):
        raise NotImplementedError

    def forward(self, session, ids, report):
        raise NotImplementedError

    def truncate(self, session, n, report):
        raise NotImplementedError

    def snapshot(self, session, tag):
        raise NotImplementedError

    def restore(self, session, tag, report):
        raise NotImplementedError

    def close(self, session):
        pass

    def run_node(self, node, inputs, weights, state):
        raise NotImplementedError

    def digest(self, text, prose=False):
        """{"sha256", "canonical"?} of a JSON text (IR_SPEC 3.10), or {"refused": message} when it has none."""
        raise NotImplementedError

    def dequantize(self, type_name, n, raw):
        """The f32 values [n] of n elements of a GGUF storage type stored as the bytes raw (IR_SPEC 4.7)."""
        raise NotImplementedError

    def run_node_grad(self, node, inputs, weights, out_grad):
        """{value or tensor name: gradient} of sum(out * out_grad) for one node (IR_SPEC 6.10.5)."""
        raise NotImplementedError

    def score(self, model_dir, prompt, cont, kv_cache=False, evict=False, kv_step=0):
        """Optional (an engine that scores, such as mtstudio): {"logprobs", "argmax", "depth"?, "kv_rows"?}; the runner otherwise uses
        open/forward."""
        raise NotImplementedError


class NumpyAdapter(Adapter):
    """The reference: tools/ir_numpy.py with no cache (a forward recomputes the whole prefix), and
    tools/ir_validate.py as its loader. It reports the kv rows the rules of IR_SPEC.md 6.8.3 and 7.5
    define (the eager count under eviction)."""
    name, profiles = "ir_numpy", {"core", "state", "recursion", "chat", "thirdparty", "recurrence"}

    def __init__(self):
        self.sessions, self.n = {}, 0

    def check(self, gguf, ir):
        if ir is not None:
            doc = json.loads(Path(ir).read_text(encoding="utf-8")) if not isinstance(ir, dict) else ir
            errs = ir_validate.doc_errors(doc)
            if gguf is not None:
                meta, tensors, *_ = ir_validate.read_gguf(gguf)
                errs += ir_validate.gguf_errors(doc, meta, tensors)
        else:
            errs = ir_validate.file_errors(gguf)
        return {"accepted": not errs, "message": errs[0] if errs else "", "errors": errs}

    def open(self, gguf, ir=None, evict=None):
        doc, W, meta, _ = load_model(Path(gguf).parent)
        if ir is not None:
            doc = json.loads(Path(ir).read_text(encoding="utf-8"))
        k = 0
        for n in doc["embed"]["nodes"]:
            if n["op"] == "embedding" and n.get("state_weight"):
                k = W[n["state_weight"]].shape[0]
        self.n += 1
        sid = f"s{self.n}"
        self.sessions[sid] = {"doc": doc, "W": W, "ids": [], "snap": {}, "k": k, "evict": evict is not False}
        return {"session": sid, "vocab": doc["model"]["vocab"], "applications": doc["block"]["repeat"], "state_k": k}

    def _report(self, s, report, n_new):
        doc, W, ids = s["doc"], s["W"], s["ids"]
        out = {"len": len(ids)}
        if not ids:
            return out
        trace, states = [], {}
        lg = ir_numpy.forward(doc, W, ids, trace=trace, states=states)
        if "logits" in report and n_new:
            out["logits"] = lg[-n_new:]
        if "depth" in report and trace:
            out["depth"] = trace[-n_new:] if n_new else []
        if "kv_rows" in report or "kv_positions" in report:
            kv = _reference_kv(doc, ids, trace, s["evict"], s["k"])
            if "kv_rows" in report:
                out["kv_rows"] = {sl: [len(a[0]) for a in apps] for sl, apps in kv.items()}
            if "kv_positions" in report:
                out["kv_positions"] = {sl: [a[0] for a in apps] for sl, apps in kv.items()}
        if "recurrent" in report and states:
            L = doc["block"]["repeat"]
            out["recurrent"] = {sl: [per[i] for i in range(L)] for sl, per in states.items()}
        return out

    def forward(self, session, ids, report):
        s = self.sessions[session]
        s["ids"] = s["ids"] + list(ids)
        return self._report(s, report, len(ids))

    def truncate(self, session, n, report):
        s = self.sessions[session]
        s["ids"] = s["ids"][:n]
        return self._report(s, [r for r in report if r != "logits"], 0)

    def snapshot(self, session, tag):
        s = self.sessions[session]
        s["snap"][tag] = list(s["ids"])
        return {"len": len(s["ids"])}

    def restore(self, session, tag, report):
        s = self.sessions[session]
        s["ids"] = list(s["snap"][tag])
        return self._report(s, [r for r in report if r != "logits"], 0)

    def close(self, session):
        self.sessions.pop(session, None)

    def digest(self, text, prose=False):
        try:
            return {"sha256": ir_validate.digest(text, prose), "canonical": ir_validate.canonical(text, prose)}
        except ir_validate.DigestError as e:
            return {"refused": str(e)}

    def dequantize(self, type_name, n, raw):
        import ir_quant
        return ir_quant.dequantize(type_name, raw, n)

    def run_node_grad(self, node, inputs, weights, out_grad):
        raise NotImplementedError("the reference has no backward: the gradient cases are its finite differences")

    def run_node(self, node, inputs, weights, state):
        env = {}
        for k, v in inputs.items():
            env[k] = np.array(v) if k in ("tokens", "pos") else v
        states = {}
        ir_numpy.run_nodes([node], env, weights, i=0, states=states)
        outs = node["out"] if isinstance(node["out"], list) else [node["out"]]
        return {"outputs": {o: env[o] for o in outs}, "state": {k: v[0] for k, v in states.items()}}


def _reference_kv(doc, ids, trace, evict, k):
    blk, L = doc["block"], len(ids)
    route = blk["nodes"][0] if blk["nodes"] and blk["nodes"][0]["op"] == "depth_route" else None
    out = {}
    for slot in blk.get("state", []):
        if slot["kind"] != "kv":
            continue
        windows = slot["window"]["per_layer"] if isinstance(slot["window"], dict) else [slot["window"]] * blk["repeat"]
        if any(windows):
            continue  # a windowed slot's rows held are not checked (11.3)
        if route is not None:
            out[slot["name"]] = [[p] for p in active_positions(route["step"]["per_layer"], trace, L)]
        elif evict and slot.get("evict") == "turn_state" and k:
            out[slot["name"]] = [evict_allowed(ids, doc["model"]["vocab"], k)] * blk["repeat"]
        else:
            out[slot["name"]] = [[list(range(L))]] * blk["repeat"]
    return out


class CmdAdapter(Adapter):
    """Any engine through a subprocess speaking section 11.3's JSON Lines."""

    def __init__(self, command, keep_env=()):
        import shlex
        env = {k: v for k, v in os.environ.items()
               if k in keep_env or not k.startswith(("EMBER_", "MICROTORCH_", "MTSTUDIO_"))}
        self.p = subprocess.Popen(shlex.split(command), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                                  bufsize=1, env=env)
        self.n = 0
        h = self._call("hello", {})
        self.name, self.backend, self.exact = h.get("engine", "cmd"), h.get("backend", "cpu"), bool(h.get("exact"))
        self.profiles, self.methods = set(h.get("profiles", [])), set(h.get("methods", []))

    def _call(self, method, params):
        self.n += 1
        self.p.stdin.write(json.dumps({"id": self.n, "method": method, "params": params}) + "\n")
        self.p.stdin.flush()
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError(f"the engine closed its output during {method}")
        r = json.loads(line)
        if "skip" in r:
            raise NotImplementedError(r["skip"])
        if "error" in r:
            raise RuntimeError(r["error"])
        return _decode_tensors(r["result"])

    def check(self, gguf, ir):
        return self._call("check", {"gguf": _p(gguf), "ir": _p(ir)})

    def open(self, gguf, ir=None, evict=None):
        return self._call("open", {"gguf": _p(gguf), "ir": _p(ir), "evict": evict})

    def forward(self, session, ids, report):
        return self._call("forward", {"session": session, "ids": list(ids), "report": report})

    def truncate(self, session, n, report):
        return self._call("truncate", {"session": session, "n": n, "report": report})

    def snapshot(self, session, tag):
        return self._call("snapshot", {"session": session, "tag": tag})

    def restore(self, session, tag, report):
        return self._call("restore", {"session": session, "tag": tag, "report": report})

    def close(self, session):
        try:
            self._call("close", {"session": session})
        except (NotImplementedError, RuntimeError):
            pass

    def digest(self, text, prose=False):
        return self._call("digest", {"text": text, "prose": prose})

    def dequantize(self, type_name, n, raw):
        return self._call("dequantize", {"type": type_name, "n": n, "bytes": raw.hex()})["values"]

    def run_node_grad(self, node, inputs, weights, out_grad):
        return self._call("run_node_grad", {"node": node, "inputs": _encode_tensors(inputs),
                                            "weights": _encode_tensors(weights), "out_grad": enc(out_grad)})["grads"]

    def run_node(self, node, inputs, weights, state):
        enc_in = {k: (v if k in ("tokens", "pos") else _encode_tensors(v)) for k, v in inputs.items()}
        return self._call("run_node", {"node": node, "inputs": enc_in, "weights": _encode_tensors(weights),
                                       "state": _encode_tensors(state)})


def _p(x):
    return None if x is None else str(Path(x).resolve())


def _encode_tensors(x):
    if isinstance(x, np.ndarray):
        return enc(x)
    if isinstance(x, dict):
        return {k: _encode_tensors(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_encode_tensors(v) for v in x]
    return x


def _decode_tensors(x):
    if isinstance(x, dict):
        if set(x) == {"shape", "f32"}:
            return dec(x)
        return {k: _decode_tensors(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_decode_tensors(v) for v in x]
    return x


def make_adapter(spec, keep_env=(), adapters=None):
    """ir_numpy, cmd:<program and arguments>, or a name in adapters {name: factory()}."""
    adapters = dict(adapters or {})
    if spec in adapters:
        return adapters[spec]()
    if spec == "ir_numpy":
        return NumpyAdapter()
    if spec.startswith("cmd:"):
        return CmdAdapter(spec[4:], keep_env)
    raise SystemExit(f"unknown adapter {spec} ({', '.join(['ir_numpy', *adapters, 'cmd:<program>'])})")


# ---------------------------------------------------------------- running the suite

def logprobs_of(lg, cont, banned):
    """Per-token log-probabilities of cont from logits rows lg (row j predicts cont[j]) and the argmax
    at each row with the banned ids excluded (mtstudio score's rule: <unk> is never the argmax)."""
    lg = np.asarray(lg, dtype=np.float64)
    lp, am = [], []
    for j, t in enumerate(cont):
        row = lg[j]
        lse = row.max() + np.log(np.exp(row - row.max()).sum())
        lp.append(float(row[t] - lse))
        b = row.copy()
        b[list(banned)] = -np.inf
        am.append(int(np.argmax(b)))
    return lp, am


def run_op_case(adapter, case, tol):
    weights = {k: dec(v) for k, v in case.get("weights", {}).items()}
    inputs = {k: (v if k in ("tokens", "pos") else dec(v)) for k, v in case["inputs"].items()}
    res = adapter.run_node(case["node"], inputs, weights, None)
    msgs, ok = [], True
    cls = case["class"] if adapter.exact else "ROUNDING"
    for name, want in case["expect"].items():
        got = res["outputs"].get(name)
        if got is None:
            return False, f"no output {name}"
        if isinstance(want, list):
            if not isinstance(got, list) or len(got) != len(want):
                return False, f"{name}: a list of {len(want)} tensors expected"
            pairs = list(zip(got, want))
        else:
            pairs = [(got, want)]
        for g, w in pairs:
            o, m = close(g, dec(w), cls, tol)
            ok &= o
            msgs.append(f"{name} {m}")
    for slot, want in case.get("expect_state", {}).items():
        got = (res.get("state") or {}).get(slot)
        if got is None:
            msgs.append(f"state {slot} not reported")
            continue
        for t, w in want.items():
            o, m = close(got[t], dec(w), case["class"] if adapter.exact else "ROUNDING", tol)
            ok &= o
            msgs.append(f"state {slot}.{t} {m}")
    return ok, "; ".join(msgs)


def run_digest_case(adapter, v):
    """IR_SPEC 3.10: the digest of v["text"] (the serialisation too, when the engine reports it), or a refusal
    for a text that has none."""
    res = adapter.digest(v["text"], v.get("prose", False))
    if "refused" in v:
        return "refused" in res, ("refused" if "refused" in res else f"digested to {res.get('sha256')}, but "
                                  f"the text has no digest ({v['refused']})")
    if "refused" in res:
        return False, f"refused: {res['refused']}"
    ok = res.get("sha256") == v["sha256"]
    msg = "sha256 " + ("equal" if ok else f"{res.get('sha256')} != {v['sha256']}")
    if res.get("canonical") is not None:
        same = res["canonical"] == v["canonical"]
        ok &= same
        msg += "; canonical " + ("equal" if same else "DIFFERS")
    return ok, msg


def run_grad_case(adapter, c):
    """IR_SPEC 6.10.5: the engine's gradients against central differences of the reference (float64)."""
    inputs = {k: dec(v) for k, v in c["inputs"].items()}
    weights = {k: dec(v) for k, v in c["weights"].items()}
    got = adapter.run_node_grad(c["node"], inputs, weights, dec(c["out_grad"]))
    ok, msgs = True, []
    for name, want in c["expect"].items():
        if name not in got:
            return False, f"no gradient for {name}"
        o, m = close(np.asarray(got[name], np.float32).reshape(dec(want).shape), dec(want), "ROUNDING", c["tol"])
        ok &= o
        msgs.append(f"d{name} {m}")
    return ok, "; ".join(msgs)


def run_dequant_case(adapter, c):
    """IR_SPEC 4.7: the f32 values of stored blocks, bit for bit on every backend."""
    got = np.asarray(adapter.dequantize(c["type"], c["rows"] * c["n"], bytes.fromhex(c["bytes"])), np.float32)
    return close(got.reshape(c["rows"], c["n"]), dec(c["expect"]), "EXACT", 0.0)


def run_model_case(adapter, case, model_dir, tol):
    banned = case.get("banned", [0])
    gguf = gguf_of(model_dir)
    worst, agree, depth_same, n = 0.0, 0, 0, 0
    for pr in case["prompts"]:
        ids, cont = pr["ids"], pr["continuation"]
        try:
            res = adapter.score(model_dir, ids, cont)
            lp, am, dep = res["logprobs"], res["argmax"], res.get("depth")
        except NotImplementedError:
            if adapter.score_only:
                raise
            s = adapter.open(gguf)["session"]
            r = adapter.forward(s, ids + cont[:-1], ["logits"] + (["depth"] if "depth" in pr else []))
            adapter.close(s)
            lp, am = logprobs_of(r["logits"][len(ids) - 1:], cont, banned)
            dep = r.get("depth")
        ref = np.array(pr["logprobs"])
        worst = max(worst, float(np.max(np.abs(np.array(lp) - ref))) / max(1.0, float(np.max(np.abs(ref)))))
        agree += am == pr["argmax"]
        if "depth" in pr:
            depth_same += dep == pr["depth"]
        n += 1
    ok = worst <= tol and agree == n and (depth_same == n or not any("depth" in p for p in case["prompts"]))
    msg = f"log-probs worst {worst:.2e}; greedy {agree}/{n}"
    if any("depth" in p for p in case["prompts"]):
        msg += f"; depth traces {depth_same}/{n}"
    return ok, msg


def _kv_ok(got, allowed):
    """got {slot: [per application]}, allowed {slot: [[allowed values] per application]}."""
    for slot, apps in allowed.items():
        g = got.get(slot)
        if g is None or len(g) != len(apps):
            return False, f"kv {slot}: {None if g is None else len(g)} applications reported for {len(apps)}"
        for i, (x, al) in enumerate(zip(g, apps)):
            if x not in al:
                return False, f"kv {slot} application {i}: {x} not among the legal {al}"
    return True, ""


def run_trace(adapter, case, model_dir, tol):
    try:
        return _run_trace_sessions(adapter, case, model_dir, tol)
    except NotImplementedError:
        if "score_form" not in case or case.get("ir"):
            raise
    sf = case["score_form"]
    res = adapter.score(model_dir, sf["prompt"], sf["continuation"], kv_cache=True, evict=sf["evict"], kv_step=1)
    ref = np.array(sf["logprobs"])
    worst = float(np.max(np.abs(np.array(res["logprobs"]) - ref))) / max(1.0, float(np.max(np.abs(ref))))
    ok, msgs = worst <= tol and res["argmax"] == sf["argmax"], [f"decode by kv cache, token by token: log-probs "
                                                                   f"worst {worst:.2e}, greedy "
                                                                   f"{'same' if res['argmax'] == sf['argmax'] else 'DIFFERS'}"]
    if "depth" in sf and res.get("depth") is not None:
        # Scored through the cache, mtstudio reports the depths of its last call only (the last context
        # token, kv_step 1): compare them with the reference trace's tail.
        got = res["depth"]
        same = got == sf["depth"][-len(got):] if got else False
        ok &= same
        msgs.append(f"depths of the last {len(got)} decoded token(s) {'same' if same else 'DIFFER'}")
    if "kv_rows" in sf and "kv_rows" in res:
        good = res["kv_rows"] in sf["kv_rows"]
        ok &= good
        msgs.append(f"kv rows {res['kv_rows']} {'legal' if good else 'not among ' + str(sf['kv_rows'])}")
    return ok, "; ".join(msgs) + " (via the adapter's score)"


def _run_trace_sessions(adapter, case, model_dir, tol):
    gguf = gguf_of(model_dir)
    s = adapter.open(gguf, SUITE / case["ir"] if case.get("ir") else None, case.get("evict"))["session"]
    ok, msgs = True, []
    try:
        for j, (step, want) in enumerate(zip(case["steps"], case["expect"])):
            report = [r for r in ("logits", "kv_rows", "kv_positions", "depth", "recurrent") if r in want]
            m = step["method"]
            if m == "forward":
                got = adapter.forward(s, step["ids"], report)
            elif m == "truncate":
                got = adapter.truncate(s, step["n"], report)
            elif m == "snapshot":
                got = adapter.snapshot(s, step["tag"])
            elif m == "restore":
                got = adapter.restore(s, step["tag"], report)
            else:
                raise SystemExit(f"trace step {m}")
            if "len" in want and got.get("len") != want["len"]:
                ok = False
                msgs.append(f"step {j}: len {got.get('len')} != {want['len']}")
            if "logits" in want:
                o, mm = close(got["logits"], dec(want["logits"]), "ROUNDING", tol)
                ok &= o
                if not o:
                    msgs.append(f"step {j} logits {mm}")
            if "depth" in want and got.get("depth") != want["depth"]:
                ok = False
                msgs.append(f"step {j}: depths {got.get('depth')} != {want['depth']}")
            if "kv_rows" in want:
                o, mm = _kv_ok(got.get("kv_rows", {}), want["kv_rows"])
                ok &= o
                if not o:
                    msgs.append(f"step {j}: {mm}")
            if "kv_positions" in want:
                o, mm = _kv_ok(got.get("kv_positions", {}), want["kv_positions"])
                ok &= o
                if not o:
                    msgs.append(f"step {j} positions: {mm}")
            if "recurrent" in want:
                for slot, apps in want["recurrent"].items():
                    for i, tens in enumerate(apps):
                        for t, w in tens.items():
                            o, mm = close(got["recurrent"][slot][i][t], dec(w), "ROUNDING", tol)
                            ok &= o
                            if not o:
                                msgs.append(f"step {j} recurrent {slot}[{i}].{t} {mm}")
    finally:
        adapter.close(s)
    return ok, (f"{len(case['steps'])} steps" + ("; " + "; ".join(msgs) if msgs else ""))


def run_refuse_case(adapter, case, path):
    gguf = SUITE / case["gguf"] if case.get("gguf") else None
    with tempfile.TemporaryDirectory(prefix="irconf_") as tmp:
        ir = None
        if "doc" in case:
            ir = Path(tmp) / "doc.ir.json"
            ir.write_text(json.dumps(case["doc"]), encoding="utf-8")
        r = adapter.check(gguf, ir)
    if r["accepted"]:
        return False, "accepted"
    if isinstance(adapter, NumpyAdapter):
        tag = f"IR_SPEC {case['section']}"
        hit = any(e.startswith(tag) for e in r["errors"])
        return hit, (f"refused ({tag})" if hit else f"refused, but not under {tag}: {r['errors'][:2]}")
    return True, "refused: " + r.get("message", "")[:120]


def run_suite(adapter, only=None, out_json=None):
    man = json.loads((SUITE / "manifest.json").read_text(encoding="utf-8"))
    tol = TOL.get(adapter.backend, 5e-5)
    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0, "XFAIL": 0, "XPASS": 0}
    results = []
    xf_path = SUITE / "expected_fail.json"
    xf = {}
    if xf_path.is_file():
        for eng, cases in json.loads(xf_path.read_text(encoding="utf-8")).items():
            if not eng.startswith("_") and adapter.name.startswith(eng):
                xf.update(cases)
    print(f"adapter {adapter.name} (backend {adapter.backend}, exact {adapter.exact}); {len(man['cases'])} cases")
    for c in man["cases"]:
        if only and only not in c["id"]:
            continue
        case = json.loads((SUITE / c["file"]).read_text(encoding="utf-8"))
        if c["kind"] == "op":
            sub = [(f"{c['id']}/{x['name']}", x) for x in case["cases"]]
        elif c["kind"] == "digest":
            sub = [(f"{c['id']}/{x['name']}", x) for x in case["vectors"]]
        elif c["kind"] in ("dequant", "grad"):
            sub = [(f"{c['id']}/{x['name']}", x) for x in case["cases"]]
        else:
            sub = [(c["id"], case)]
        for cid, x in sub:
            try:
                if c["kind"] == "op":
                    ok, msg = run_op_case(adapter, x, x.get("tol", tol) if adapter.backend == "cpu" else tol)
                elif c["kind"] == "model":
                    ok, msg = run_model_case(adapter, x, SUITE / c["model"], tol)
                elif c["kind"] == "trace":
                    ok, msg = run_trace(adapter, x, SUITE / x["model"], tol)
                elif c["kind"] == "digest":
                    ok, msg = run_digest_case(adapter, x)
                elif c["kind"] == "dequant":
                    ok, msg = run_dequant_case(adapter, x)
                elif c["kind"] == "grad":
                    ok, msg = run_grad_case(adapter, x)
                else:
                    ok, msg = run_refuse_case(adapter, x, c["file"])
                verdict = "PASS" if ok else "FAIL"
                if cid in xf or cid.split("/", 2)[0] + "/" + cid.split("/")[1] in xf:
                    verdict = {"PASS": "XPASS", "FAIL": "XFAIL"}[verdict]
                    msg += f" [expected to fail: {xf.get(cid, '')[:80]}]"
            except NotImplementedError as e:
                verdict, msg = "SKIP", str(e) or "not implemented"
            except (RuntimeError, ValueError, KeyError, SystemExit) as e:
                verdict, msg = "FAIL", f"{type(e).__name__}: {e}"
            counts[verdict] += 1
            results.append({"id": cid, "kind": c["kind"], "verdict": verdict, "detail": msg})
            print(f"{verdict} {cid}: {msg}")
    by_kind = {}
    for r in results:
        by_kind.setdefault(r["kind"], {"PASS": 0, "FAIL": 0, "SKIP": 0, "XFAIL": 0, "XPASS": 0})[r["verdict"]] += 1
    print(f"summary {adapter.name}: {counts['PASS']} pass, {counts['FAIL']} fail, {counts['SKIP']} skip, "
          f"{counts['XFAIL']} xfail, {counts['XPASS']} xpass; by kind "
          + ", ".join(f"{k} {v['PASS']}/{v['FAIL']}/{v['SKIP']}" for k, v in by_kind.items()) + " (pass/fail/skip)")
    if out_json:
        Path(out_json).write_text(json.dumps({"adapter": adapter.hello(), "counts": counts, "by_kind": by_kind,
                                              "results": results}, indent=1) + "\n", encoding="utf-8")
    return counts


# ---------------------------------------------------------------- generating the suite

def main(argv=None, adapters=None):
    """The command line; adapters {name: factory()} adds in-process adapters to --adapter's choices."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter")
    ap.add_argument("--only")
    ap.add_argument("--json")
    ap.add_argument("--regenerate", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--keep-env", action="append", default=[], help="an engine variable to pass through (repeatable)")
    a = ap.parse_args(argv)
    if a.regenerate:
        import ir_conformance_gen
        ir_conformance_gen.regenerate()
        return 0
    if a.list:
        for c in json.loads((SUITE / "manifest.json").read_text(encoding="utf-8"))["cases"]:
            print(c["kind"], c["id"], c.get("profile", ""))
        return 0
    if not a.adapter:
        ap.print_help()
        return 2
    counts = run_suite(make_adapter(a.adapter, a.keep_env, adapters), a.only, a.json)
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
