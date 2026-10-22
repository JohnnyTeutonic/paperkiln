"""Compare the diag-linear export of the expr-norm gate model with its scale re-export.

usage: python3 scale_compare.py OLD.gguf NEW.gguf parity.jsonl TOOLS_DIR OUT.json

1. Same weights: every tensor of NEW except the *_norm_gain.weight vectors is byte-identical to
   OLD's tensor of that name, and OLD's *_norm_gain.weight is diag(NEW's) with +0 off the diagonal.
2. IR: OLD has the gain as linear nodes, NEW as scale nodes, and nothing else differs.
3. numpy IR logits (tools/ir_numpy.py) of every prompt (chat-templated as tools/ir_run.py does)
   followed by mtstudio's recorded reply, cut to ctx: bit-identical between OLD and NEW.
"""
import hashlib
import json
import sys

import numpy as np

old_p, new_p, parity_p, tools, out_p = sys.argv[1:6]
sys.path.insert(0, tools)
import bpe  # noqa: E402
import ir_check  # noqa: E402
import ir_numpy  # noqa: E402


def load(path):
    meta, tensors, data_start, raw, _ = ir_check.read_gguf(path)
    doc = json.loads(meta["paperkiln.ir"][1])
    W, raw_bytes = {}, {}
    for name, dims, typ, off in tensors:
        assert typ == 0, name
        n = int(np.prod(dims))
        W[name] = np.frombuffer(raw, dtype=np.float32, count=n, offset=data_start + off).reshape(list(reversed(dims)))
        raw_bytes[name] = raw[data_start + off:data_start + off + 4 * n]
    return doc, W, raw_bytes, {t[0]: t[1] for t in tensors}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


old_doc, Wo, Bo, Do = load(old_p)
new_doc, Wn, Bn, Dn = load(new_p)
rep = {"old": old_p, "new": new_p, "old_sha256": sha(old_p), "new_sha256": sha(new_p)}

# 1. Same weights.
assert set(Wo) == set(Wn), sorted(set(Wo) ^ set(Wn))
gains = sorted(k for k in Wn if k.endswith("_norm_gain.weight"))
same = [k for k in Wn if k not in gains and Bo[k] == Bn[k] and Do[k] == Dn[k]]
assert len(same) == len(Wn) - len(gains), sorted(set(Wn) - set(gains) - set(same))
for k in gains:
    g = Wn[k]
    d = g.shape[0]
    assert g.ndim == 1 and Do[k] == [d, d], (k, Do[k], Dn[k])
    want = np.zeros((d, d), np.float32)
    want[np.arange(d), np.arange(d)] = g
    assert Wo[k].tobytes() == want.tobytes(), k
rep["tensors"] = len(Wn)
rep["tensors_byte_identical"] = len(same)
rep["gain_tensors"] = {k: {"old_ne": Do[k], "new_ne": Dn[k]} for k in gains}
rep["old_gain_is_diag_of_new_with_plus_zero_off_diagonal"] = True
rep["bytes_saved"] = sum(4 * (Do[k][0] * Do[k][1] - Dn[k][0]) for k in gains)

# 2. IR: only the gain nodes differ (linear -> scale).
conv = json.loads(json.dumps(new_doc))
n_scale = 0
for sect in ("embed", "block", "head"):
    for n in conv[sect]["nodes"]:
        if n["op"] == "scale":
            n_scale += 1
            n["op"] = "linear"
assert ir_check.strip(conv) == ir_check.strip(old_doc), "IR differs beyond linear -> scale"
rep["ir_scale_nodes"] = n_scale
rep["block_ops"] = [n["op"] for n in new_doc["block"]["nodes"]]
rep["head_ops"] = [n["op"] for n in new_doc["head"]["nodes"]]
rep["ir_check_new"] = ir_check.check(new_p) or "OK"

# 3. numpy IR logits, bit for bit.
tok = bpe.read_gguf(new_p)
ctx = new_doc["model"]["ctx"]
rows = [json.loads(line) for line in open(parity_p, encoding="utf-8")]
n_ident, positions, worst = 0, 0, 0.0
for r in rows:
    ids = tok.encode("<|user|>" + r["prompt"] + "<|assistant|>")
    ids = (ids + tok.encode(r["mtstudio"]))[:ctx]
    a = ir_numpy.forward(new_doc, Wn, ids)
    b = ir_numpy.forward(old_doc, Wo, ids)
    assert a.dtype == b.dtype == np.float32 and a.shape == b.shape
    ok = bool(np.array_equal(a.view(np.uint32), b.view(np.uint32)))
    n_ident += ok
    positions += a.shape[0]
    worst = max(worst, float(np.abs(a - b).max()))
rep["numpy_logits"] = {"prompts": len(rows), "bit_identical": n_ident, "positions": positions,
                       "max_abs_diff": worst,
                       "input": "chat-templated prompt + mtstudio's recorded reply tokens, cut to ctx"}
json.dump(rep, open(out_p, "w", encoding="utf-8"), indent=1)
print(json.dumps(rep["numpy_logits"]), rep["tensors_byte_identical"], "/", rep["tensors"], "tensors byte-identical")
