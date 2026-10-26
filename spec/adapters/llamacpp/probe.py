#!/usr/bin/env python3
"""Probes of llama.cpp beyond the suite's verdicts, through the same adapter (FINDINGS.txt reads them).

    python3 adapters/llamacpp/probe.py --bin ~/lc_keramos/llamacpp_adapter --json OUT.json

Run from the keramos root, on the platform the adapter was built for (WSL). Each probe uses the suite's
models and the numpy reference (tools/ir_numpy.py); nothing is written except --json.

  attribution  llama.cpp's prefill logits of 16 ids against three references: the file's own IR
               document (what the suite checks); the same with GELU computed as ggml's CPU does it
               (fp16(gelu_tanh(fp16(x))) by table, for |x| < 10); and the document tools/gguf_ir.py
               builds from the GGUF's llama.cpp metadata alone, without windows (what llama.cpp's graph
               is meant to compute), with ggml's GELU. Per reference: the worst error (|a - b| over
               max(1, max|ref|), the suite's ROUNDING measure), the first position past 1e-5, and the
               positions where the greedy token differs.
  consistency  inside llama.cpp: one prefill of 16 ids against token-by-token decode, against chunks
               5 + 6 + 5, and against a second identical run: bit identity, or the worst difference.
  rewind       prefill, truncate, forward: the logits after the truncate against a fresh prefill of the
               same ids in llama.cpp and against the reference; with the cache settings named.
  snapshot     snapshot at 8, forward 4, restore, forward the same 4: bit identity.
"""
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import gguf_ir  # noqa: E402
import ir_conformance as rc  # noqa: E402
import ir_numpy  # noqa: E402
import ir_validate  # noqa: E402

SUITE = ROOT / "conformance" / "models"
TOL = 1e-5
MODELS = ["tp_qwen3", "tp_mistral", "tp_gemma3", "tp_gptneox", "tp_mamba1", "pk_llama", "pk_moe", "pk_gpt2",
          "pk_gpt2_swa", "pk_kimi", "pk_recurrence"]


# ---------------------------------------------------------------- references

def ggml_gelu(x):
    """ggml-cpu b6000 ggml_vec_gelu_f32 under GGML_GELU_FP16 (vec.h): the table of
    fp16(gelu_tanh(f)) over every fp16 f, read at fp16(x); 0 at x <= -10 and x at x >= 10."""
    x = np.asarray(x, dtype=np.float32)
    f = x.astype(np.float16).astype(np.float32)
    c, a = np.float32(0.79788456080286535587989211986876), np.float32(0.044715)
    g = (np.float32(0.5) * f * (np.float32(1) + np.tanh(c * f * (np.float32(1) + a * f * f)))).astype(np.float32)
    y = g.astype(np.float16).astype(np.float32)
    return np.where(x <= -10, np.float32(0), np.where(x >= 10, x, y)).astype(np.float32)


@contextlib.contextmanager
def ggml_gelu_in_reference():
    """ir_numpy with every gelu node (tanh or exact) computed as ggml_gelu."""
    orig = ir_numpy.run_nodes

    def run_nodes(nodes, env, W, i=None, **kw):
        seg = []
        for nd in nodes:
            if nd["op"] == "gelu":
                if seg:
                    orig(seg, env, W, i, **kw)
                    seg = []
                env[nd["out"]] = ggml_gelu(env[nd["in"][0]])
            else:
                seg.append(nd)
        if seg:
            orig(seg, env, W, i, **kw)

    ir_numpy.run_nodes = run_nodes
    try:
        yield
    finally:
        ir_numpy.run_nodes = orig


def arch_document(model_dir, alias=None):
    """tools/gguf_ir.py's document of the GGUF from its llama.cpp metadata alone (paperkiln.ir ignored), or
    None when general.architecture has no builder; alias {from: to} renames the architecture first."""
    path = rc.gguf_of(model_dir)
    meta_raw, tensors, *_ = ir_validate.read_gguf(path)
    meta = {k: v[1] for k, v in meta_raw.items() if not k.startswith("tokenizer.ggml.") or k == "tokenizer.ggml.model"}
    meta["_n_tokens"] = len(meta_raw["tokenizer.ggml.tokens"][1])
    arch = meta["general.architecture"]
    if alias and arch in alias:
        to = alias[arch]
        meta = {(to + k[len(arch):] if k.startswith(arch + ".") else k): v for k, v in meta.items()}
        meta["general.architecture"] = to
    shapes = {n: list(reversed(d)) for n, d, *_ in tensors}
    try:
        return gguf_ir.build(meta, shapes)
    except SystemExit:
        return None


def ids_for(vocab, n=16, seed=7):
    rng = np.random.default_rng(seed)
    return [int(x) for x in rng.integers(1, vocab, size=n)]


def compare(got, ref):
    """Worst error over max(1, max|ref|), first position past TOL, positions where the argmax differs."""
    got, ref = np.asarray(got, np.float64), np.asarray(ref, np.float64)
    scale = max(1.0, float(np.max(np.abs(ref))))
    per = np.max(np.abs(got - ref), axis=1) / scale
    first = next((int(p) for p, e in enumerate(per) if e > TOL), None)
    flips = [int(p) for p in range(len(got)) if int(np.argmax(got[p])) != int(np.argmax(ref[p]))]
    return {"worst": float(per.max()), "per_position": [float(f"{e:.3g}") for e in per], "first_past_tol": first,
            "greedy_differs_at": flips}


def bitdiff(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    same = bool(np.array_equal(a.view(np.uint32), b.view(np.uint32)))
    return {"bit_identical": same, "values_differing": int(np.sum(a.view(np.uint32) != b.view(np.uint32))),
            "worst_abs": float(np.max(np.abs(a.astype(np.float64) - b)))}


# ---------------------------------------------------------------- llama.cpp through the adapter

class Engine:
    def __init__(self, binary, *args):
        self.cmd = " ".join([binary, "--threads", "2", *args])
        self.a = rc.CmdAdapter(self.cmd)

    def run(self, model_dir, chunks, report=("logits",), info=None):
        """Logits of one session fed the chunks in order (a list of id lists), concatenated; info (a dict)
        receives open's result."""
        o = self.a.open(rc.gguf_of(model_dir))
        if info is not None:
            info.update(o)
        s = o["session"]
        try:
            return np.concatenate([self.a.forward(s, c, list(report))["logits"] for c in chunks])
        finally:
            self.a.close(s)


def attribution(eng, name, alias):
    d = SUITE / name
    doc, W, _, _ = rc.load_model(d)
    ids = ids_for(doc["model"]["vocab"])
    info = {}
    got = eng.run(d, [ids], info=info)
    out = {"ids": ids, "llamacpp_arch": info.get("_arch"), "llamacpp_n_swa": info.get("_n_swa"),
           "rewrites": info.get("_rewrites"), "vs_file_ir": compare(got, ir_numpy.forward(doc, W, ids))}
    with ggml_gelu_in_reference():
        out["vs_file_ir_ggml_gelu"] = compare(got, ir_numpy.forward(doc, W, ids))
        ad = arch_document(d, alias)
        if ad is not None:
            # Windows only where llama.cpp's graph has them (n_swa > 0: gemma3 here).
            key = "vs_arch_document_ggml_gelu" + ("" if info.get("_n_swa") else "_no_window")
            try:
                out[key] = compare(got, ir_numpy.forward(ad, W, ids, no_window=not info.get("_n_swa")))
            except KeyError as e:  # the builder has no graph for the file's tensors (llama with experts)
                out["arch_document_not_built"] = (f"tools/gguf_ir.py's {info.get('_arch')} graph does not fit the "
                                                  f"file's tensors (KeyError {e}: no experts in its llama graph)")
    return out


def consistency(eng, name):
    d = SUITE / name
    doc, *_ = rc.load_model(d)
    ids = ids_for(doc["model"]["vocab"])
    pre = eng.run(d, [ids])
    return {"prefill_vs_token_by_token": bitdiff(pre, eng.run(d, [[t] for t in ids])),
            "prefill_vs_chunks_5_6_5": bitdiff(pre, eng.run(d, [ids[:5], ids[5:11], ids[11:]])),
            "prefill_run_to_run": bitdiff(pre, eng.run(d, [ids]))}


def rewind(eng, name, L, n, k, label):
    """Prefill L ids, truncate to n, forward ids[n:n+k] (the same ids as before): the logits against a fresh
    prefill of ids[:n+k] (rows n .. n+k-1) in llama.cpp and against the reference."""
    d = SUITE / name
    doc, W, _, _ = rc.load_model(d)
    ids = ids_for(doc["model"]["vocab"], n=L, seed=11)
    a = eng.a
    s = a.open(rc.gguf_of(d))["session"]
    try:
        a.forward(s, ids, [])
        tr = a.truncate(s, n, [])
        got = a.forward(s, ids[n:n + k], ["logits"])["logits"]
    finally:
        a.close(s)
    fresh = eng.run(d, [ids[:n + k]])[n:]
    ref = ir_numpy.forward(doc, W, ids[:n + k])[n:]
    with ggml_gelu_in_reference():
        ref_g = ir_numpy.forward(doc, W, ids[:n + k])[n:]
    return {"settings": label, "L": L, "truncate_to": n, "then": k, "truncate_how": tr.get("_how"),
            "vs_fresh_prefill_in_llamacpp": bitdiff(got, fresh),
            "vs_reference": compare(got, ref), "vs_reference_ggml_gelu": compare(got, ref_g),
            "fresh_prefill_vs_reference_ggml_gelu": compare(fresh, ref_g)}


def snapshot(eng, name):
    d = SUITE / name
    doc, *_ = rc.load_model(d)
    ids = ids_for(doc["model"]["vocab"])
    a = eng.a
    s = a.open(rc.gguf_of(d))["session"]
    try:
        a.forward(s, ids[:8], [])
        a.snapshot(s, "p")
        first = a.forward(s, ids[8:12], ["logits"])["logits"]
        a.restore(s, "p", [])
        again = a.forward(s, ids[8:12], ["logits"])["logits"]
    finally:
        a.close(s)
    return bitdiff(first, again)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--json")
    ap.add_argument("--only")
    a = ap.parse_args(argv)
    alias = {"mistral": "llama"}
    eng = Engine(a.bin, "--alias", "mistral=llama")
    res = {"adapter": eng.a.hello(), "command": eng.cmd, "tol": TOL, "models": {}}
    for name in MODELS:
        if a.only and a.only not in name:
            continue
        r = {"attribution": attribution(eng, name, alias), "consistency": consistency(eng, name),
             "snapshot_restore": snapshot(eng, name),
             "rewind": [rewind(eng, name, 16, 8, 4, "default (swa_full 1, n_ubatch 512)")]}
        res["models"][name] = r
        at = r["attribution"]
        print(name, {k: (v["worst"], v["first_past_tol"], v["greedy_differs_at"]) for k, v in at.items() if k.startswith("vs_")},
              {k: v["bit_identical"] for k, v in r["consistency"].items()}, "snapshot", r["snapshot_restore"]["bit_identical"],
              "rewind", r["rewind"][0]["vs_fresh_prefill_in_llamacpp"]["bit_identical"], r["rewind"][0]["truncate_how"],
              flush=True)
    # Rewind into a sliding window (tp_gemma3: window 4 on layers 0-4). With swa_full 0 and n_ubatch 4 the
    # windowed cache has 32 cells (GGML_PAD(n_swa + n_ubatch, 32)), so a 48-token prefill reuses the cells of
    # masked positions; truncating to 10 then needs positions 7..9, which are gone.
    sw = []
    for args, label in [((), "swa_full 1 (llama_context_default_params), n_ubatch 512"),
                        (("--swa-full", "0", "--n-ubatch", "4"), "swa_full 0 (common/llama-server default), n_ubatch 4"),
                        (("--swa-full", "0", "--n-ubatch", "4", "--rewind-guard"),
                         "swa_full 0, n_ubatch 4, with llama-server's rewind guard")]:
        e = Engine(a.bin, *args)
        for L, n in [(48, 10), (48, 40)]:
            r = rewind(e, "tp_gemma3", L, n, 4, label)
            sw.append(r)
            print("swa rewind", label, L, n, r["truncate_how"], r["vs_fresh_prefill_in_llamacpp"]["bit_identical"],
                  r["vs_reference_ggml_gelu"]["worst"], r["vs_reference_ggml_gelu"]["greedy_differs_at"], flush=True)
    res["swa_rewind_tp_gemma3"] = sw
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
