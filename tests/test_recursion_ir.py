"""Tied-block recursion through mtstudio, the GGUF and the IR (arch.custom.recursion; ir.hpp weight_index).

Tiny recursed llama models (1 + [2] x 3 + 1, i.e. 8 applications of 4 stored layers, without and with
inject "add"; and [2] x 4) train for 40 steps on CPU and export F32 and Q8_0 GGUFs. Checks: the files
pass spec/tools/ir_check.py; architecture "paperkiln" with block_count = the stored layers and tensors
blk.0..3 only (each shared layer stored once); the IR's block is the llama block (plus the injection
affine and add) with repeat = n_layers = 8 applications and weight_index [0, 1, 2, 1, 2, 1, 2, 3];
spec/tools/ir_numpy.py's forward over the GGUF equals mtstudio's own forward (`mtstudio score`) on 40
prompts: log-probabilities of a greedy continuation within ROUNDING (1e-5 of max(1, max |ref|)) and
mtstudio's argmax at every scored position equal to numpy's greedy token (40/40 greedy). Without the
weight_index the IR names tensors the file does not have (ir_check fails), so the map is load-bearing.
A one-repeat recursion exports as the plain llama; mtstudio refuses inconsistent recursion specs.
Needs MTSTUDIO_BIN (or ./build/mtstudio); skipped otherwise.
"""
import copy
import json
from pathlib import Path
import random
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "spec" / "tools"))  # keramos (the IR reference)
from test_mtstudio_chat import BIN, write_vocab_gguf  # noqa: E402
from test_mtstudio_moe import WORDS, corpus, events, run  # noqa: E402
import ir_check  # noqa: E402

needs_bin = pytest.mark.skipif(BIN is None, reason="set MTSTUDIO_BIN to a built mtstudio")
V0 = json.loads((ROOT / "spec" / "examples" / "chat_sftfinal.ir.json").read_text(encoding="utf-8"))
REC = {"prelude": 1, "block": 2, "repeats": 3, "coda": 1}


def spec_of(tmp, name, recursion, layers=None, steps=40, formats=("safetensors", "gguf", "gguf_q8_0"), extra=None):
    tmp.mkdir(parents=True, exist_ok=True)
    write_vocab_gguf(tmp / "v.gguf", ["<unk>"] + WORDS)
    (tmp / "c.txt").write_text(corpus(), encoding="utf-8")
    custom = {"d": 32, "heads": 2, "recursion": recursion}
    if layers is not None:
        custom["layers"] = layers
    arch = {"preset": "llama-tiny", "custom": custom}
    arch.update(extra or {})
    spec = {"name": name, "arch": arch,
            "data": {"vocab": str(tmp / "v.gguf"), "corpus": str(tmp / "c.txt"), "T": 16, "max_tokens": 0},
            "train": {"steps": steps, "lr": 0.003, "eval_every": 20, "checkpoint_every": 20, "batch": 2, "seed": 3,
                      "gradmap_every": 0},
            "export": {"formats": list(formats)}, "out_dir": str(tmp / name)}
    p = tmp / f"{name}.json"
    p.write_text(json.dumps(spec), encoding="utf-8")
    return p


def gguf_exports(spec):
    return [e for e in events(spec) if e["event"] == "export" and e["format"].startswith("gguf")]


def weights(path):
    import numpy as np
    meta, tensors, data_start, raw, _ = ir_check.read_gguf(path)
    W = {}
    for name, dims, typ, off in tensors:
        assert typ == 0, name
        n = int(np.prod(dims))
        W[name] = np.frombuffer(raw, dtype=np.float32, count=n, offset=data_start + off).reshape(list(reversed(dims)))
    return meta, W


def prompts(n=40, seed=11):
    rng = random.Random(seed)
    ids = list(range(1, len(WORDS) + 1))  # 0 is <unk>
    return [[rng.choice(ids) for _ in range(rng.randint(2, 6))] for _ in range(n)]


def parity(spec, tmp, n_new=8):
    """numpy greedy over the F32 GGUF vs mtstudio score on the same ids: (logprob worst, greedy agree, n)."""
    import numpy as np
    import ir_numpy
    f32 = next(e for e in gguf_exports(spec) if e["format"] == "gguf")["path"]
    meta, W = weights(f32)
    doc = json.loads(meta["paperkiln.ir"][1])
    rows, cont = [], []
    for p in prompts():
        ids, out, lps = list(p), [], []
        for _ in range(n_new):
            lg = ir_numpy.forward(doc, W, ids)[-1].astype(np.float64)
            ban = lg.copy()
            ban[0] = -np.inf  # mtstudio score's argmax skips <unk>
            t = int(np.argmax(ban))
            lse = lg.max() + np.log(np.exp(lg - lg.max()).sum())
            lps.append(lg[t] - lse)
            out.append(t)
            ids.append(t)
        rows.append({"prompt_ids": p, "continuation_ids": out})
        cont.append((out, lps))
    req, res = tmp / "score_in.jsonl", tmp / "score_out.jsonl"
    req.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    r = subprocess.run([BIN, "score", str(spec), "--in", str(req), "--out", str(res)], capture_output=True, text=True,
                       timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    got = [json.loads(x) for x in res.read_text().splitlines() if x.strip()]
    worst, agree = 0.0, 0
    for (out, lps), g in zip(cont, got):
        assert not g["over_ctx"]
        ref = np.array(g["logprobs"])
        worst = max(worst, float(np.max(np.abs(np.array(lps) - ref)) / max(1.0, float(np.max(np.abs(ref))))))
        agree += g["argmax"] == out
    return worst, agree, len(got), doc


@needs_bin
@pytest.mark.parametrize("rec,inject", [(REC, "none"), (REC, "add"), ({"block": 2, "repeats": 4}, "none")])
def test_recursed_export_and_numpy_parity(tmp_path, rec, inject):
    rec = dict(rec, inject=inject)
    spec = spec_of(tmp_path, "r", rec)
    run(spec)
    ev = events(spec)
    model = next(e for e in ev if e["event"] == "model")
    stored = rec.get("prelude", 0) + rec["block"] + rec.get("coda", 0)
    apps = rec.get("prelude", 0) + rec["block"] * rec["repeats"] + rec.get("coda", 0)
    assert model["layers"] == stored and model["recursion"]["effective_layers"] == apps
    steps = [e["loss"] for e in ev if e["event"] == "step"]
    assert steps[-1] < steps[0] - 0.3, steps  # it trains
    ex = gguf_exports(spec)
    assert {e["format"] for e in ex} == {"gguf", "gguf_q8_0"}
    for e in ex:
        assert ir_check.check(e["path"]) == [], ir_check.check(e["path"])
        meta, tensors, *_ = ir_check.read_gguf(e["path"])
        assert meta["general.architecture"][1] == "paperkiln"
        assert meta["paperkiln.block_count"][1] == stored
        blk = {t[0].split(".")[1] for t in tensors if t[0].startswith("blk.")}
        assert blk == {str(i) for i in range(stored)}  # each shared layer once
    doc = json.loads(ir_check.read_gguf(ex[0]["path"])[0]["paperkiln.ir"][1])
    p, b, r, c = rec.get("prelude", 0), rec["block"], rec["repeats"], rec.get("coda", 0)
    sched = list(range(p)) + [p + j for _ in range(r) for j in range(b)] + [p + b + i for i in range(c)]
    assert doc["family_hint"] == "llama_recursive"
    assert doc["model"]["n_layers"] == doc["block"]["repeat"] == apps
    assert doc["block"]["weight_index"] == {"per_layer": sched}
    nodes = doc["block"]["nodes"]
    plain = [n["op"] for n in V0["block"]["nodes"]]
    if inject == "add":
        assert [n["op"] for n in doc["embed"]["nodes"]] == ["embedding", "embedding"]
        assert doc["embed"]["nodes"][1]["out"] == "e"
        inj = [1.0 if (i >= p and i < p + b * r and (i - p) % b == 0 and i > p) else 0.0 for i in range(apps)]
        assert nodes[0] == {"op": "affine", "in": ["e"], "scale": {"per_layer": inj}, "shift": 0.0, "out": "inj"}
        assert nodes[1] == {"op": "add", "in": ["x", "inj"], "out": "x"}
        assert [n["op"] for n in nodes[2:]] == plain
    else:
        assert [n["op"] for n in nodes] == plain
    # Without the map the IR reads blk.{i} for i up to apps - 1: tensors the file does not have.
    bad = copy.deepcopy(doc)
    del bad["block"]["weight_index"]
    have = {t[0] for t in ir_check.read_gguf(ex[0]["path"])[1]}
    assert any(n not in have for _, n, _ in ir_check.weight_refs(bad))
    worst, agree, n, _ = parity(spec, tmp_path)
    print(f"recursion {rec}: numpy IR vs mtstudio log-probs worst {worst:.2e} of max(1, |ref|); "
          f"greedy {agree}/{n}")
    assert n == 40 and agree == 40
    assert worst <= 1e-5


@needs_bin
def test_one_repeat_exports_the_plain_llama(tmp_path):
    spec = spec_of(tmp_path, "r1", {"block": 2, "repeats": 1}, steps=2, formats=("safetensors", "gguf"))
    run(spec)
    path = gguf_exports(spec)[0]["path"]
    assert ir_check.check(path) == []
    meta, *_ = ir_check.read_gguf(path)
    assert meta["general.architecture"][1] == "llama"
    doc = json.loads(meta["paperkiln.ir"][1])
    assert doc["family_hint"] == "llama" and "weight_index" not in doc["block"]


@needs_bin
@pytest.mark.parametrize("rec,layers,extra,formats,msg", [
    (REC, 3, None, ("safetensors",), "counts stored layers"),
    ({"block": 0}, None, None, ("safetensors",), "block must be >= 1"),
    ({"block": 2, "repeats": 0}, None, None, ("safetensors",), "repeats must be >= 1"),
    ({"block": 2, "schedule": "router"}, None, None, ("safetensors",), "not implemented"),
    ({"block": 2, "inject": "concat"}, None, None, ("safetensors",), "inject must be none or add"),
    ({"block": 2, "depth": 3}, None, None, ("safetensors",), "unknown key depth"),
    ({"block": 2, "repeats": 2}, None, {"moe": {"experts": 4, "top_k": 2}}, ("safetensors",), "not wired yet"),
    ({"block": 2, "repeats": 2}, None, None, ("safetensors", "gguf_q4_0"), "no reference for a recursed"),
])
def test_bad_recursion_specs_are_refused(tmp_path, rec, layers, extra, formats, msg):
    spec = spec_of(tmp_path, "bad", rec, layers=layers, formats=formats, extra=extra)
    r = subprocess.run([BIN, "plan", str(spec)], capture_output=True, text=True, timeout=60)
    assert r.returncode != 0 and msg in r.stderr, r.stderr


@needs_bin
def test_recursion_needs_the_llama_family(tmp_path):
    spec = spec_of(tmp_path, "flex", {"block": 2, "repeats": 2})
    j = json.loads(spec.read_text())
    j["arch"]["custom"]["norm"] = "layernorm"
    spec.write_text(json.dumps(j))
    r = subprocess.run([BIN, "plan", str(spec)], capture_output=True, text=True, timeout=60)
    assert r.returncode != 0 and "needs the llama family" in r.stderr, r.stderr
