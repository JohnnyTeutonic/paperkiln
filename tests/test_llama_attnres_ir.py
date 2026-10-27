"""Attention Residuals on the llama block through mtstudio, the GGUF and the IR (arch.custom.attn_res; family
llama_attnres; include/microtorch/llama_attnres.hpp; experiments/tiny_chat_long_2026-10/IR_SPEC_DRAFT.md
sections 2-3, cell X).

Tiny llama_attnres models train for 40 steps on CPU and export F32 and Q8_0 GGUFs:
  ar3     3 layers, scale "sources" (the arm's form)
  ar3n    3 layers, scale "none" (Kimi's convex mixture)
  ar_rec  1 + [2] x 2 + 1 fixed recursion (weight_index [0, 1, 2, 1, 2, 3]), one pair of queries per
          application ("tensor_index": "application", IR_SPEC 7.2); also checked against the document
          without its tags, which reads the stored layer's queries and must move the log-probabilities
  ar_x5   the arm's shape (tiny-chat amendment 5, cell X5): 20m_L16, d 320, 16 layers, 5 heads, d_ff 864,
          rope on every head, scale "sources" (33 depth sources), on the toy vocabulary
Checks: the run logs the attn_res model event; the files pass spec/tools/ir_check.py; architecture "paperkiln";
the pseudo-queries are [d] tensors stored F32 in both variants (blk.{i}.attnres_attn_q / attnres_ffn_q,
attnres_out_q; rec_app.{i}.* under recursion), application 0's attention query written as zeros and the
trained ones nonzero; the IR is embed + list, a block of depth_attend (+ affine S under "sources"),
rmsnorm, the llama attention, append, depth_attend (+ affine), rmsnorm, SwiGLU, append, and a head of
depth_attend (+ affine), output_norm, output; spec/tools/ir_numpy.py's forward over the F32 GGUF against
`mtstudio score` on 40 prompts: log-probabilities of a greedy continuation within ROUNDING (1e-5 of
max(1, max |ref|)) and greedy 40/40. At step 1 an attn_res run's loss equals the plain llama's of the same
seed (zero queries; scale "sources"). Malformed specs are refused.

ATTNRES_GATE_DIR=DIR also writes the gate files for ember (F32 GGUF, IR, Q8_0 GGUF, vocabulary,
score_in/score_out, spec and a manifest with sha256) for ar3, ar_rec and ar_x5 into DIR/<name>/, the layout of
tests/test_mor_ir.py's MOR_GATE_DIR.
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
from test_mtstudio_chat import BIN, write_vocab_gguf  # noqa: E402
from test_mtstudio_moe import WORDS, corpus, events, run  # noqa: E402
from test_recursion_ir import gguf_exports, parity, prompts, weights  # noqa: E402
import ir_check  # noqa: E402

needs_bin = pytest.mark.skipif(BIN is None, reason="set MTSTUDIO_BIN to a built mtstudio")

CASES = {
    "ar3": {"layers": 3, "attn_res": {"form": "full"}},
    "ar3n": {"layers": 3, "attn_res": {"form": "full", "scale": "none"}},
    "ar_rec": {"attn_res": {"form": "full"},
               "recursion": {"prelude": 1, "block": 2, "repeats": 2, "coda": 1, "inject": "none"}},
    "ar_x5": {"layers": 16, "d": 320, "heads": 5, "d_ff": 864, "attn_res": {"form": "full"}},
}
EXTRA = {"ar_x5": {"rope_heads": "all"}}  # the arm's arch (make_jobs.pt_spec)
D = 32
GATES = ("ar3", "ar_rec", "ar_x5")


def spec_of(tmp, name, custom, steps=40, formats=("safetensors", "gguf", "gguf_q8_0"), extra=None, train=None):
    tmp.mkdir(parents=True, exist_ok=True)
    write_vocab_gguf(tmp / "v.gguf", ["<unk>"] + WORDS)
    (tmp / "c.txt").write_text(corpus(), encoding="utf-8")
    arch = {"preset": "llama-tiny", "custom": dict({"d": D, "heads": 2}, **custom)}
    arch.update(extra or {})
    tr = {"steps": steps, "lr": 0.003, "eval_every": 20, "checkpoint_every": 20, "batch": 2, "seed": 3,
          "gradmap_every": 0}
    tr.update(train or {})
    spec = {"name": name, "arch": arch,
            "data": {"vocab": str(tmp / "v.gguf"), "corpus": str(tmp / "c.txt"), "T": 16, "max_tokens": 0},
            "train": tr, "export": {"formats": list(formats)}, "out_dir": str(tmp / name)}
    p = tmp / f"{name}.json"
    p.write_text(json.dumps(spec), encoding="utf-8")
    return p


def applications(custom):
    r = custom.get("recursion")
    if not r:
        return custom["layers"]
    return r.get("prelude", 0) + r["block"] * r["repeats"] + r.get("coda", 0)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_values(path, name):
    import numpy as np
    meta, tensors, data_start, raw, _ = ir_check.read_gguf(path)
    t = next(x for x in tensors if x[0] == name)
    assert t[2] == 0, (name, t[2])  # F32
    return np.frombuffer(raw, dtype=np.float32, count=int(np.prod(t[1])), offset=data_start + t[3])


def untagged_gap(doc, f32, n=8):
    """The largest |logit| difference, relative to max(1, max |ref|), between the document and the same
    document without its tensor_index tags (application i reading rec_app.{m_i}), over n prompts."""
    import numpy as np
    import ir_numpy
    alt = json.loads(json.dumps(doc))
    for nd in alt["block"]["nodes"]:
        nd.pop("tensor_index", None)
    W = weights(f32)[1]
    gap = 0.0
    for p in prompts()[:n]:
        a, b = ir_numpy.forward(doc, W, list(p)), ir_numpy.forward(alt, W, list(p))
        gap = max(gap, float(np.abs(a - b).max() / max(1.0, float(np.abs(a).max()))))
    return gap


def write_gate(name, spec, res, custom, extra=None):
    out = Path(os.environ["ATTNRES_GATE_DIR"]) / name
    out.mkdir(parents=True, exist_ok=True)
    ex = gguf_exports(spec)
    f32 = Path(next(e["path"] for e in ex if e["format"] == "gguf"))
    q8 = Path(next(e["path"] for e in ex if e["format"] == "gguf_q8_0"))
    tmp = Path(spec).parent
    files = [f32, f32.with_suffix(".ir.json"), q8, tmp / "v.gguf", tmp / "score_in.jsonl", tmp / "score_out.jsonl"]
    for f in files:
        shutil.copyfile(f, out / f.name)
    shutil.copyfile(spec, out / "spec.json")
    files.append(out / "spec.json")
    worst, agree, n = res
    manifest = {"model": name, "family_hint": "llama_attnres", "arch_custom": custom,
                "spec": json.loads(Path(spec).read_text()),
                "numpy_vs_mtstudio": {"worst": worst, "agree": agree, "n": n},
                **(extra or {}),
                "files": {f.name: sha256(out / f.name) for f in files},
                "_note": "score_in.jsonl: prompt_ids and spec/tools/ir_numpy.py's greedy continuation over the F32 GGUF; "
                         "score_out.jsonl: mtstudio score over the same ids (logprobs, argmax). ember gate: "
                         "greedy 40/40 and log-probabilities within ROUNDING on CPU, CUDA and WASM "
                         "(IR_SPEC_DRAFT section 5, gate 5)."}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")


@needs_bin
@pytest.mark.parametrize("name", list(CASES))
def test_attnres_export_and_numpy_parity(tmp_path, name):
    custom = CASES[name]
    d = custom.get("d", D)
    spec = spec_of(tmp_path, name, custom, extra=EXTRA.get(name))
    run(spec)
    ev = events(spec)
    model = next(e for e in ev if e["event"] == "model")
    A = applications(custom)
    scale = custom["attn_res"].get("scale", "sources")
    assert model["attn_res"] == {"form": "full", "scale": scale, "sources": 2 * A + 1, "query_params": 2 * A * d}
    steps = [e["loss"] for e in ev if e["event"] == "step"]
    assert steps[-1] < steps[0] - 0.3, steps  # it trains
    ex = gguf_exports(spec)
    assert {e["format"] for e in ex} == {"gguf", "gguf_q8_0"}
    rec = "recursion" in custom
    qpre = "rec_app." if rec else "blk."
    qnames = {f"{qpre}{a}.attnres_{s}_q.weight" for a in range(A) for s in ("attn", "ffn")} | {"attnres_out_q.weight"}
    for e in ex:
        assert ir_check.check(e["path"]) == [], ir_check.check(e["path"])
        meta, tensors, *_ = ir_check.read_gguf(e["path"])
        assert meta["general.architecture"][1] == "paperkiln"
        q = {t[0]: t for t in tensors if "attnres_" in t[0]}
        assert set(q) == qnames
        assert all(t[1] == [d] and t[2] == 0 for t in q.values())  # [d], F32 in every variant
    f32 = next(e["path"] for e in ex if e["format"] == "gguf")
    assert set(tensor_values(f32, f"{qpre}0.attnres_attn_q.weight")) == {0.0}
    for nm in sorted(qnames - {f"{qpre}0.attnres_attn_q.weight"}):
        assert abs(tensor_values(f32, nm)).max() > 0, nm  # every read query trained
    doc = json.loads(ir_check.read_gguf(f32)[0]["paperkiln.ir"][1])
    assert doc["family_hint"] == "llama_attnres"
    assert doc["model"]["n_layers"] == doc["block"]["repeat"] == A
    assert [n["op"] for n in doc["embed"]["nodes"]] == ["embedding", "list"]
    mix = ["depth_attend"] + (["affine"] if scale == "sources" else [])
    assert [n["op"] for n in doc["block"]["nodes"]] == (
        mix + ["rmsnorm", "linear", "linear", "linear", "rope", "attention", "linear", "append"] +
        mix + ["rmsnorm", "linear", "linear", "silu", "mul", "linear", "append"])
    assert [n["op"] for n in doc["head"]["nodes"]] == mix + ["rmsnorm", "linear"]
    das = [n for n in doc["block"]["nodes"] if n["op"] == "depth_attend"]
    assert all(("tensor_index" in n) == rec for n in das)
    if scale == "sources":
        aff = [n for n in doc["block"]["nodes"] if n["op"] == "affine"]
        assert aff[0]["scale"] == {"per_layer": [2 * a + 1 for a in range(A)]}
        assert aff[1]["scale"] == {"per_layer": [2 * a + 2 for a in range(A)]}
        assert doc["head"]["nodes"][1]["scale"] == 2 * A + 1
    if rec:
        assert "weight_index" in doc["block"]
    worst, agree, n, _ = parity(spec, tmp_path)
    print(f"{name} {custom}: numpy IR vs mtstudio log-probs worst {worst:.2e} of max(1, |ref|); greedy {agree}/{n}")
    assert n == 40 and agree == 40
    assert worst <= 1e-5
    extra = None
    if rec:
        # The tags matter (IR_SPEC 7.2): the stored layer's queries give another function.
        gap = untagged_gap(doc, f32)
        print(f"{name}: the untagged document's logits differ by {gap:.2e} of max(1, |ref|) on 8 prompts")
        assert gap > 1e-3, gap
        extra = {"weight_index": doc["block"]["weight_index"]["per_layer"], "untagged_gap": gap,
                 "_untagged_gap": "max |logit difference| / max(1, max |ref|) over the first 8 prompts between the "
                                  "document and the same document without its tensor_index tags (IR_SPEC 7.2)"}
    if os.environ.get("ATTNRES_GATE_DIR") and name in GATES:
        write_gate(name, spec, (worst, agree, n), custom, extra)


@needs_bin
@pytest.mark.parametrize("layers", [3, 16])
def test_step1_loss_equals_the_plain_llama(tmp_path, layers):
    """Zero queries, scale sources: the first step's loss is the plain model's (same seed, same batch)."""
    losses = {}
    for name, extra in (("plain", {}), ("ar", {"attn_res": {"form": "full"}}), ("arn", {"attn_res": {"scale": "none"}})):
        spec = spec_of(tmp_path / name, name, dict({"layers": layers}, **extra), steps=1, formats=("safetensors",),
                       train={"eval_every": 1, "checkpoint_every": 1})
        run(spec)
        losses[name] = next(e["loss"] for e in events(spec) if e["event"] == "step")
    print(f"{layers} layers, step-1 loss: plain {losses['plain']:.7f}, attn_res sources {losses['ar']:.7f}, "
          f"none {losses['arn']:.7f}")
    assert abs(losses["ar"] - losses["plain"]) <= 1e-5 * abs(losses["plain"])


def _bad(tmp_path, custom, extra=None, formats=("safetensors",)):
    spec = spec_of(tmp_path, "bad", custom, formats=formats, extra=extra)
    return subprocess.run([BIN, "plan", str(spec)], capture_output=True, text=True, timeout=60)


AR = {"layers": 2, "attn_res": {"form": "full"}}
MOR = {"block": 2, "repeats": 3, "prelude": 1, "coda": 1, "schedule": "mor"}


@needs_bin
@pytest.mark.parametrize("custom,extra,formats,msg", [
    (dict(AR, attn_res={"form": "block"}), None, ("safetensors",), "is not implemented"),
    (dict(AR, attn_res={"scale": "mean"}), None, ("safetensors",), "scale must be sources or none"),
    (dict(AR, attn_res={"queries": "input"}), None, ("safetensors",), "unknown key queries"),
    (dict(AR, attn_res="full"), None, ("safetensors",), "must be an object"),
    ({"attn_res": {}, "recursion": MOR}, None, ("safetensors",), "schedule mor is not implemented"),
    ({"attn_res": {}, "recursion": {"block": 2, "repeats": 2, "inject": "add"}}, None, ("safetensors",),
     "inject add"),
    (AR, {"moe": {"experts": 4, "top_k": 2}}, ("safetensors",), "not wired"),
    (AR, None, ("safetensors", "gguf_q4_0"), "attention-residual model"),
    (dict(AR, layers=64), None, ("safetensors",), "exceed the depth_attend kernel"),
    (dict(AR, norm="layernorm"), None, ("safetensors",), "needs the llama family"),
])
def test_bad_attnres_specs_are_refused(tmp_path, custom, extra, formats, msg):
    r = _bad(tmp_path, custom, extra, formats)
    assert r.returncode != 0 and msg in r.stderr, r.stderr


@needs_bin
def test_plan_prints_the_depth_sources(tmp_path):
    r = _bad(tmp_path, AR)
    assert r.returncode == 0, r.stderr
    assert "attn_res: full (sublayer level), scale sources, 5 depth sources, 4 zero-initialised pseudo-queries of 32" \
        in r.stdout, r.stdout
