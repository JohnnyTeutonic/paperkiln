"""Mixture-of-Recursions (causal token choice) through mtstudio, the GGUF and the IR (arch.custom.recursion
schedule "mor"; ir.hpp depth_route and kv_rows "active"; tiny-chat amendment 4).

Tiny routed llama models train for 40 steps on CPU and export F32 and Q8_0 GGUFs:
  mor3   1 + [2] x {1, 2, 3} + 1, gate score (the registered topology)
  mor2p0 0 + [3] x {1, 2} + 1, gate score (no prelude: the router reads the embedding; two steps of three)
  mor3n  1 + [2] x {1, 2, 3} + 1, gate none
Checks: the run logs the router (depth fractions, mean depth, entropy, balance and z-loss on every step;
depth fractions at every validation); the files pass spec/tools/ir_check.py; architecture "paperkiln",
block_count the stored layers, rec.0.router.weight [d, R] stored F32 in both variants; the IR's block is
depth_route then the llama block, family llama_mor, kv_rows active, weight_index over every application;
spec/tools/ir_numpy.py's forward over the F32 GGUF against `mtstudio score` on 40 prompts: log-probabilities of
a greedy continuation within ROUNDING (1e-5 of max(1, max |ref|)), greedy 40/40, and the depth trace
token for token (the smallest routing margin is reported). Malformed mor specs are refused.

MOR_GATE_DIR=DIR also writes the gate files for ember (F32 GGUF, IR, score_in/score_out, the numpy depth
traces and a manifest with sha256) for mor3 and mor2p0 into DIR/<name>/.
Needs MTSTUDIO_BIN (or ./build/mtstudio); skipped otherwise.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "spec" / "tools"))  # keramos (the IR reference)
from test_mtstudio_chat import BIN  # noqa: E402
from test_mtstudio_moe import events, run  # noqa: E402
from test_recursion_ir import V0, gguf_exports, prompts, spec_of, weights  # noqa: E402
import ir_check  # noqa: E402

needs_bin = pytest.mark.skipif(BIN is None, reason="set MTSTUDIO_BIN to a built mtstudio")

CASES = {
    "mor3": {"prelude": 1, "block": 2, "repeats": 3, "coda": 1, "schedule": "mor"},
    "mor2p0": {"prelude": 0, "block": 3, "repeats": 2, "coda": 1, "schedule": "mor"},
    "mor3n": {"prelude": 1, "block": 2, "repeats": 3, "coda": 1, "schedule": "mor", "mor": {"gate": "none"}},
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parity(spec, tmp, n_new=8):
    """numpy greedy over the F32 GGUF vs mtstudio score on the same ids: log-prob worst, greedy agreement,
    depth-trace agreement, the smallest routing margin, the depth histogram, and the files written."""
    import numpy as np
    import ir_numpy
    f32 = next(e for e in gguf_exports(spec) if e["format"] == "gguf")["path"]
    meta, W = weights(f32)
    doc = json.loads(meta["paperkiln.ir"][1])
    R = doc["block"]["nodes"][0]["n_rec"]
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
        trace = []
        ir_numpy.forward(doc, W, p + out[:-1], trace=trace)  # mtstudio scores the context ids[:-1]
        rows.append({"prompt_ids": p, "continuation_ids": out})
        cont.append((out, lps, trace))
    req, res = tmp / "score_in.jsonl", tmp / "score_out.jsonl"
    req.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    r = subprocess.run([BIN, "score", str(spec), "--in", str(req), "--out", str(res)], capture_output=True, text=True,
                       timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    got = [json.loads(x) for x in res.read_text().splitlines() if x.strip()]
    worst, agree, same_depth, margin, hist = 0.0, 0, 0, float("inf"), [0] * R
    for (out, lps, trace), g in zip(cont, got):
        assert not g["over_ctx"]
        ref = np.array(g["logprobs"])
        worst = max(worst, float(np.max(np.abs(np.array(lps) - ref)) / max(1.0, float(np.max(np.abs(ref))))))
        agree += g["argmax"] == out
        assert len(g["depth"]) == len(trace)
        same_depth += g["depth"] == trace
        margin = min(margin, g.get("depth_margin_min", float("inf")))
        for d in g["depth"]:
            hist[d - 1] += 1
    traces = tmp / "depth_trace_numpy.jsonl"
    traces.write_text("".join(json.dumps({"i": i, "context_ids": rw["prompt_ids"] + c[0][:-1], "depth": c[2]}) + "\n"
                              for i, (rw, c) in enumerate(zip(rows, cont))), encoding="utf-8")
    return {"worst": worst, "agree": agree, "same_depth": same_depth, "n": len(got), "margin": margin, "hist": hist,
            "doc": doc, "f32": f32, "files": [req, res, traces]}


def write_gate(name, spec, res, rec):
    out = Path(os.environ["MOR_GATE_DIR"]) / name
    out.mkdir(parents=True, exist_ok=True)
    f32 = Path(res["f32"])
    q8 = next(Path(e["path"]) for e in gguf_exports(spec) if e["format"] == "gguf_q8_0")
    files = [f32, f32.with_suffix(".ir.json"), q8, Path(spec).parent / "v.gguf", *res["files"]]
    for f in files:
        shutil.copyfile(f, out / f.name)
    shutil.copyfile(spec, out / "spec.json")
    files.append(out / "spec.json")
    manifest = {"model": name, "recursion": rec, "spec": json.loads(Path(spec).read_text()),
                "numpy_vs_mtstudio": {k: res[k] for k in ("worst", "agree", "same_depth", "n", "margin", "hist")},
                "files": {f.name: sha256(out / f.name) for f in files},
                "_note": "score_in.jsonl: prompt_ids and the numpy greedy continuation; score_out.jsonl: mtstudio "
                         "score (logprobs, argmax, depth per context position, depth_margin_min); "
                         "depth_trace_numpy.jsonl: spec/tools/ir_numpy.py's depths over the same contexts."}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")


@needs_bin
@pytest.mark.parametrize("name", list(CASES))
def test_mor_export_and_numpy_parity(tmp_path, name):
    rec = CASES[name]
    spec = spec_of(tmp_path, name, dict(rec, inject="none"))
    run(spec)
    ev = events(spec)
    model = next(e for e in ev if e["event"] == "model")
    P, B, R, C = rec["prelude"], rec["block"], rec["repeats"], rec["coda"]
    stored, apps = P + B + C, P + B * R + C
    assert model["layers"] == stored and model["recursion"]["effective_layers"] == apps
    assert model["recursion"]["mor"]["balance"] == pytest.approx(0.1)
    assert model["recursion"]["mor"]["z_loss"] == pytest.approx(0.001)
    steps = [e for e in ev if e["event"] == "step"]
    assert all(len(e["mor_frac"]) == R and abs(sum(e["mor_frac"]) - 1) < 1e-4 for e in steps)
    assert all(1 <= e["mor_depth"] <= R and e["mor_balance"] > 0 for e in steps)
    evals = [e for e in ev if e["event"] == "eval"]
    assert evals and all(len(e["mor_frac"]) == R for e in evals)
    assert steps[-1]["loss"] < steps[0]["loss"] - 0.3, [e["loss"] for e in steps]  # it trains
    ex = gguf_exports(spec)
    assert {e["format"] for e in ex} == {"gguf", "gguf_q8_0"}
    for e in ex:
        assert ir_check.check(e["path"]) == [], ir_check.check(e["path"])
        meta, tensors, *_ = ir_check.read_gguf(e["path"])
        assert meta["general.architecture"][1] == "paperkiln"
        assert meta["paperkiln.block_count"][1] == stored
        router = [t for t in tensors if t[0] == "rec.0.router.weight"]
        assert len(router) == 1 and router[0][1] == [32, R] and router[0][2] == 0  # ne {d, R}, F32
    doc = json.loads(ir_check.read_gguf(ex[0]["path"])[0]["paperkiln.ir"][1])
    assert doc["family_hint"] == "llama_mor"
    assert doc["model"]["n_layers"] == doc["block"]["repeat"] == apps
    sched = list(range(P)) + [P + j for _ in range(R) for j in range(B)] + [P + B + i for i in range(C)]
    assert doc["block"]["weight_index"] == {"per_layer": sched}
    nodes = doc["block"]["nodes"]
    step = [-1] * P + [k for k in range(R) for _ in range(B)] + [-1] * C
    gate = rec.get("mor", {}).get("gate", "score")
    assert nodes[0] == {"op": "depth_route", "in": ["x"], "weight": "rec.0.router.weight", "route": "token_choice",
                        "n_rec": R, "gate": gate, "step": {"per_layer": step}, "out": "rec_depth"}
    assert [n["op"] for n in nodes[1:]] == [n["op"] for n in V0["block"]["nodes"]]
    assert all(s["kv_rows"] == "active" for s in doc["block"]["state"])
    res = parity(spec, tmp_path)
    print(f"{name} {rec}: numpy IR vs mtstudio log-probs worst {res['worst']:.2e} of max(1, |ref|); greedy "
          f"{res['agree']}/{res['n']}; depth traces {res['same_depth']}/{res['n']} identical; depths {res['hist']}; "
          f"smallest routing margin {res['margin']:.3e}")
    assert res["n"] == 40 and res["agree"] == 40 and res["same_depth"] == 40
    assert res["worst"] <= 1e-5
    assert sum(1 for c in res["hist"] if c > 0) >= 2, res["hist"]  # the router routes
    if os.environ.get("MOR_GATE_DIR") and name in ("mor3", "mor2p0"):
        write_gate(name, spec, res, rec)


def _bad(tmp_path, rec, extra=None, formats=("safetensors",)):
    spec = spec_of(tmp_path, "bad", rec, formats=formats, extra=extra)
    return subprocess.run([BIN, "plan", str(spec)], capture_output=True, text=True, timeout=60)


MOR = {"block": 2, "repeats": 3, "prelude": 1, "coda": 1, "schedule": "mor"}


@needs_bin
@pytest.mark.parametrize("rec,extra,formats,msg", [
    (dict(MOR, mor={"route": "expert_choice"}), None, ("safetensors",), "must be token_choice"),
    (dict(MOR, mor={"gate": "sigmoid"}), None, ("safetensors",), "gate must be score or none"),
    (dict(MOR, mor={"kv_rows": "shared"}), None, ("safetensors",), "kv_rows must be active"),
    (dict(MOR, mor={"capacity": 0.5}), None, ("safetensors",), "unknown key capacity"),
    (dict(MOR, mor={"balance": -1}), None, ("safetensors",), "must be >= 0"),
    (dict(MOR, schedule="fixed", mor={}), None, ("safetensors",), "needs schedule"),
    (dict(MOR, inject="add"), None, ("safetensors",), "inject add is not wired with schedule mor"),
    (MOR, {"moe": {"experts": 4, "top_k": 2}}, ("safetensors",), "not wired yet"),
    (MOR, None, ("safetensors", "gguf_q4_0"), "no reference for a recursed"),
])
def test_bad_mor_specs_are_refused(tmp_path, rec, extra, formats, msg):
    r = _bad(tmp_path, rec, extra, formats)
    assert r.returncode != 0 and msg in r.stderr, r.stderr


@needs_bin
def test_plan_prints_the_router(tmp_path):
    r = _bad(tmp_path, MOR)
    assert r.returncode == 0, r.stderr
    assert "mor: causal token choice over depths 1..3" in r.stdout, r.stdout
