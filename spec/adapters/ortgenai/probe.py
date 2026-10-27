#!/usr/bin/env python3
"""Probes of ONNX Runtime GenAI beyond the suite's verdicts, through the same adapter (FINDINGS.txt reads them).

    python adapters/ortgenai/probe.py --json OUT.json [--tf4-py PYTHON] [--only SECTION]

Run from the keramos root in the pinned environment. --tf4-py names a Python whose environment is this one with
onnxruntime-genai 0.15.2 and transformers 4.57.6 (requirements.txt's last lines: 0.15.2 is the newest release
whose builder imports under transformers 4); without it the gemma3_tf4 part is skipped. Nothing is written
except --json and the adapter's cache (~/ort_keramos, probe builds under ~/ort_keramos/probe).

  kernel        onnxruntime's CPU GroupQueryAttention on its own (one node): which head sizes it runs, with and
                without fused rotary (do_rotary).
  builds        per suite file of an HF architecture: what the stock builder does (declines, fails, or builds),
                the attention nodes it emits, genai_config's sliding_window, and whether the CPU kernel runs it.
  attribution   prefill logits of 16 ids (48 for the windowed models) against the file's own document, in the
                configuration where ONNX Runtime runs the model (unfused: GQA with a separate RotaryEmbedding,
                for head_size 8; mha: MultiHeadAttention, for head_size 4; the stock build for the head_dim 16
                files); windowed models also against the document with its window removed. error = |a - b| / max(1, max|ref|), tol 1e-5.
  consistency   inside ONNX Runtime GenAI (unfused, pk_llama and tp_gemma3 with the local-RoPE fix; the stock
                build, tp_llama_hd16 and tp_qwen3_hd16): one prefill
                against token-by-token append_tokens, chunks 5 + 6 + 5, and a second run: bit identity or the
                worst difference; rewind_to against a fresh prefill; snapshot (rewind) and restore.
  generate      the GenAI search (generate_next_token, greedy; id 0 masked through set_logits, as the suite
                bans it) on each model case's prompts: the continuation against the suite's, up to where
                the generator reports is_done (the file's eos id).
  gemma3        tp_gemma3: the builder under transformers 5 with the export's config and with the config as
                Google ships Gemma 3 (both KeyError 'rope_type'); then, with the transformers 4 view of the
                config, which candidate fix brings the logits to the reference (local RoPE without the linear
                factor; local_window_size - 1), and the RoPE caches the builder wrote against cos/sin of each
                theta with and without the factor. With --tf4-py: transformers 4.57.6's own logits for the
                config as Google ships it, and the stock builder of onnxruntime-genai 0.15.2 on it (its RoPE
                caches; its graph is refused by the CPU kernel at head_size 8, so no logits).
  mistral       tp_mistral: GroupQueryAttention's local_window_size in the stock and unfused graphs and
                genai_config's sliding_window layers; MistralConfig's layer_types; the MHA build's logits
                against the document with and without its window; with --tf4-py, the same export through the
                stock builder of onnxruntime-genai 0.15.2 (which keeps the window).
  mistral_hd16  tp_mistral_hd16 (head_dim 16, which the stock CPU graph runs): the same graph and config facts,
                the stock build's logits of 16 ids against the document with and without its window, and how
                many tokens the stock generator takes before it errors, with genai_config's sliding_window
                window_size as built (4), 8 and 12, and with that section removed.
  gguf          pk_llama through the builder's own GGUF loader (-i model.gguf): logits against the export
                path's, bit for bit.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(HERE))
import ir_conformance as rc  # noqa: E402
import ir_numpy  # noqa: E402
import ortgenai_adapter as oa  # noqa: E402

SUITE = ROOT / "conformance" / "models"
TOL = 1e-5
HF_MODELS = ["pk_llama", "pk_moe", "pk_gpt2", "tp_qwen3", "tp_mistral", "tp_gemma3", "tp_gptneox", "tp_mamba1",
             "tp_llama_hd16", "tp_mistral_hd16", "tp_qwen3_hd16"]
WINDOWED = {"tp_mistral", "tp_gemma3", "tp_mistral_hd16"}
UNFUSED = "unfused_rope,gemma3_config"
GEMMA_FIX = "unfused_rope,gemma3_config,local_rope_unscaled"
MHA = "no_gqa,gemma3_config"


def ids_for(vocab, n=16, seed=7):
    rng = np.random.default_rng(seed)
    return [int(x) for x in rng.integers(1, vocab, size=n)]


def compare(got, ref):
    got, ref = np.asarray(got, np.float64), np.asarray(ref, np.float64)
    scale = max(1.0, float(np.max(np.abs(ref))))
    per = np.max(np.abs(got - ref), axis=1) / scale
    first = next((int(p) for p, e in enumerate(per) if e > TOL), None)
    flips = [int(p) for p in range(len(got)) if int(np.argmax(got[p])) != int(np.argmax(ref[p]))]
    return {"worst": float(per.max()), "first_past_tol": first, "greedy_differs_at": flips}


def bitdiff(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return {"bit_identical": bool(np.array_equal(a.view(np.uint32), b.view(np.uint32))),
            "values_differing": int(np.sum(a.view(np.uint32) != b.view(np.uint32))),
            "worst_abs": float(np.max(np.abs(a.astype(np.float64) - b)))}


ENGINES = {}


def engine(patch, loader="export"):
    key = (patch, loader)
    if key not in ENGINES:
        ENGINES[key] = oa.ORTGenAIAdapter(patch=patch, loader=loader)
    return ENGINES[key]


def run(a, name, chunks):
    s = a.open(rc.gguf_of(SUITE / name))["session"]
    try:
        return np.concatenate([a.forward(s, c, ["logits"])["logits"] for c in chunks])
    finally:
        a.close(s)


def run_dir(a, out, chunks):
    s = a.session_of(a.load(out))
    return np.concatenate([a._append(s, c) for c in chunks])


def reference(name, ids, no_window=False, doc=None):
    d, W, _, _ = rc.load_model(SUITE / name)
    return ir_numpy.forward(doc or d, W, ids, no_window=no_window)


def n_ids(name, n=None):
    doc, *_ = rc.load_model(SUITE / name)
    n = n or (48 if name in WINDOWED else min(16, doc["model"]["ctx"]))
    return ids_for(doc["model"]["vocab"], n)


# ---------------------------------------------------------------- kernel

def kernel():
    import onnxruntime as ort
    from onnx import TensorProto, helper
    f = TensorProto.FLOAT

    def one(hd, rot, H=2, KV=1, S=3, P=16):
        ins = [("q", f, [1, S, H * hd]), ("k", f, [1, S, KV * hd]), ("v", f, [1, S, KV * hd]),
               ("pk", f, [1, KV, P, hd]), ("pv", f, [1, KV, P, hd]), ("sl", TensorProto.INT32, [1]),
               ("tl", TensorProto.INT32, [])] + ([("cos", f, [P, hd // 2]), ("sin", f, [P, hd // 2])] if rot else [])
        node = helper.make_node("GroupQueryAttention", [n for n, _, _ in ins], ["o", "ok", "ov"], domain="com.microsoft",
                                num_heads=H, kv_num_heads=KV, do_rotary=int(rot))
        g = helper.make_graph([node], "g", [helper.make_tensor_value_info(*x) for x in ins],
                              [helper.make_tensor_value_info(n, f, None) for n in ("o", "ok", "ov")])
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)],
                              ir_version=10)
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        so.log_severity_level = 4
        sess = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
        rng = np.random.default_rng(0)
        feed = {"q": rng.standard_normal((1, S, H * hd), np.float32), "k": rng.standard_normal((1, S, KV * hd), np.float32),
                "v": rng.standard_normal((1, S, KV * hd), np.float32), "pk": np.zeros((1, KV, P, hd), np.float32),
                "pv": np.zeros((1, KV, P, hd), np.float32), "sl": np.array([S - 1], np.int32), "tl": np.array(S, np.int32)}
        if rot:
            feed["cos"], feed["sin"] = np.ones((P, hd // 2), np.float32), np.zeros((P, hd // 2), np.float32)
        try:
            sess.run(None, feed)
            return "runs"
        except Exception as e:  # noqa: BLE001
            return "refused: " + str(e).split("Status Message: ")[-1].strip()

    return {f"head_size {hd}, do_rotary {rot}": one(hd, rot) for hd in (4, 8, 16, 24, 32, 64, 72, 80) for rot in (0, 1)}


# ---------------------------------------------------------------- builds

def builds():
    out = {}
    for name in HF_MODELS:
        row = {}
        for label, patch in (("stock", "none"), ("unfused", UNFUSED), ("mha", MHA)):
            a = engine(patch)
            try:
                m = a._model(rc.gguf_of(SUITE / name))
                n = m["notes"]
                row[label] = {"attention": n["attention_op"], "sliding_window_layers":
                              (n["sliding_window"] or {}).get("layers"), "runs": n["refused"] or "yes"}
            except NotImplementedError as e:
                row[label] = {"declined": str(e)}
            except Exception as e:  # noqa: BLE001
                row[label] = {"error": f"{type(e).__name__}: {str(e)[:240]}"}
        out[name] = row
    return out


# ---------------------------------------------------------------- numbers

def attribution():
    res = {}
    for name, patch in (("pk_llama", UNFUSED), ("tp_gemma3", UNFUSED), ("tp_gemma3", GEMMA_FIX), ("tp_qwen3", MHA),
                        ("tp_mistral", MHA), ("pk_llama", MHA), ("tp_llama_hd16", "none"), ("tp_qwen3_hd16", "none"),
                        ("tp_mistral_hd16", "none")):
        ids = n_ids(name, 16 if name == "tp_mistral_hd16" else None)  # its stock generator holds 20 tokens
        got = run(engine(patch), name, [ids])
        r = {"n": len(ids), "vs_reference": compare(got, reference(name, ids))}
        if name in WINDOWED:
            r["vs_reference_without_window"] = compare(got, reference(name, ids, no_window=True))
        res[f"{name} [{patch}]"] = r
    return res


def consistency():
    res = {}
    for name, patch in (("pk_llama", UNFUSED), ("tp_gemma3", GEMMA_FIX), ("tp_llama_hd16", "none"),
                        ("tp_qwen3_hd16", "none")):
        a = engine(patch)
        ids = n_ids(name)[:16]
        ref = reference(name, ids)
        pre = run(a, name, [ids])
        tok = run(a, name, [[t] for t in ids])
        ch = run(a, name, [ids[:5], ids[5:11], ids[11:]])
        r = {"prefill_vs_token_by_token": bitdiff(pre, tok), "prefill_vs_chunks_5_6_5": bitdiff(pre, ch),
             "prefill_run_to_run": bitdiff(pre, run(a, name, [ids])),
             "token_by_token_vs_reference": compare(tok, ref)["worst"], "chunks_vs_reference": compare(ch, ref)["worst"]}
        # rewind_to(n) then the next 4 ids, against a fresh prefill of the same n + 4 ids
        s = a.open(rc.gguf_of(SUITE / name))["session"]
        rows = []
        for n in (10, 3):
            a.forward(s, ids[len(a.sessions[s]["ids"]):] or ids[:1], [])
            tr = a.truncate(s, n, [])
            got = a.forward(s, ids[n:n + 4], ["logits"])["logits"]
            fresh = run(a, name, [ids[:n + 4]])[n:]
            rows.append({"truncate_to": n, "how": tr["_how"], "vs_fresh_prefill": bitdiff(got, fresh),
                         "vs_reference": compare(got, ref[n:n + 4])["worst"]})
        a.close(s)
        r["rewind"] = rows
        s = a.open(rc.gguf_of(SUITE / name))["session"]
        a.forward(s, ids[:8], [])
        a.snapshot(s, "p")
        first = a.forward(s, ids[8:12], ["logits"])["logits"]
        rs = a.restore(s, "p", [])
        again = a.forward(s, ids[8:12], ["logits"])["logits"]
        a.close(s)
        r["snapshot_restore"] = {"how": rs["_how"], **bitdiff(first, again)}
        res[f"{name} [{patch}]"] = r
    return res


def generate():
    res = {}
    for name, patch in (("pk_llama", UNFUSED), ("tp_gemma3", UNFUSED), ("tp_gemma3", GEMMA_FIX), ("tp_qwen3", MHA),
                        ("tp_llama_hd16", "none"), ("tp_qwen3_hd16", "none"), ("tp_mistral_hd16", "none")):
        a = engine(patch)
        m = a._model(rc.gguf_of(SUITE / name))
        case = json.loads((SUITE / name / "case.json").read_text(encoding="utf-8"))
        rows = []
        for pr in case["prompts"]:
            n = len(pr["continuation"])
            p = a.og.GeneratorParams(m["model"])
            p.set_search_options(max_length=m["ctx"], do_sample=False)
            g = a.og.Generator(m["model"], p)
            got, stop = [], None
            try:
                g.append_tokens(pr["ids"])
                for _ in range(n):
                    lg = np.array(g.get_logits(), dtype=np.float32).reshape(1, -1)
                    lg[0, 0] = -np.inf
                    g.set_logits(lg)
                    g.generate_next_token()
                    got.append(int(g.get_next_tokens()[0]))
                    if g.is_done():
                        stop = f"is_done after {len(got)} (eos {m['genai_config']['model']['eos_token_id']})"
                        break
            except Exception as e:  # noqa: BLE001
                rows.append({"error": f"{type(e).__name__}: {str(e)[:200]}", "got": got})
                continue
            want = pr["continuation"][:len(got)]
            rows.append({"same": got == want, "got": got, "suite": pr["continuation"], "stopped": stop})
        res[f"{name} [{patch}]"] = {"same": sum(bool(r.get("same")) for r in rows), "of": len(rows), "prompts": rows}
    return res


# ---------------------------------------------------------------- Gemma 3

def hub_config(export_dir, dest):
    """The export's Gemma 3 config as Google ships Gemma 3 (transformers 4 field names): rope_theta,
    rope_local_base_freq, rope_scaling {"rope_type": "linear", "factor": f} for the global layers."""
    c = json.loads((Path(export_dir) / "config.json").read_text(encoding="utf-8"))
    rp = c.pop("rope_parameters")
    full, local = rp["full_attention"], rp["sliding_attention"]
    c["rope_theta"], c["rope_local_base_freq"] = full["rope_theta"], local["rope_theta"]
    c["rope_scaling"] = {"rope_type": "linear", "factor": full["factor"]} if full.get("rope_type") == "linear" else None
    if c.get("eos_token_id") is None:
        c["eos_token_id"] = 0  # the GGUF declares none; the unknown id, as the adapter's --eos-fallback
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for f in Path(export_dir).iterdir():
        if f.name != "config.json":
            shutil.copy2(f, dest / f.name)
    (dest / "config.json").write_text(json.dumps(c, indent=1) + "\n", encoding="utf-8")
    return dest


def build_with(py, src, out, patch):
    if Path(out).exists():
        shutil.rmtree(out)
    cmd = [py, str(HERE / "build_onnx.py"), "--input", str(src), "--output", str(out),
           "--cache", str(Path.home() / "ort_keramos" / "cache"), "--patch", patch]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, (r.stderr.strip().splitlines() or [""])[-1]


def gemma3(tf4_py=None):
    name = "tp_gemma3"
    gguf = rc.gguf_of(SUITE / name)
    a0 = engine("none")
    ex, _, _ = a0.build_dir(gguf)
    if not (ex / "config.json").is_file():
        oa.g2h.export(gguf, ex)
    work = Path.home() / "ort_keramos" / "probe"
    hub = hub_config(ex, work / "tp_gemma3-hub")
    res = {"hub_config": {k: v for k, v in json.loads((hub / "config.json").read_text()).items()
                          if k.startswith("rope") or k in ("sliding_window", "layer_types", "query_pre_attn_scalar")}}
    res["builder_tf5_export_config"] = build_with(sys.executable, ex, work / "g-tf5-export", "none")
    res["builder_tf5_hub_config"] = build_with(sys.executable, hub, work / "g-tf5-hub", "none")
    ids = n_ids(name)
    ref = reference(name, ids)
    doc, *_ = rc.load_model(SUITE / name)
    wider = copy.deepcopy(doc)
    for n in wider["block"]["nodes"]:
        if n["op"] == "attention" and isinstance(n.get("window"), dict):
            n["window"]["per_layer"] = [w + 1 if w else w for w in n["window"]["per_layer"]]
    refs = {"reference": ref, "reference_without_window": reference(name, ids, no_window=True),
            "reference_window_plus_1": reference(name, ids, doc=wider)}
    iso = {}
    for extra in ("", ",local_rope_unscaled", ",window_minus_one", ",local_rope_unscaled,window_minus_one"):
        got = run(engine(UNFUSED + extra), name, [ids])
        iso[UNFUSED + extra] = {k: compare(got, r)["worst"] for k, r in refs.items()}
    res["isolation_tf4_view"] = iso
    # The caches the current builder writes under the transformers 4 view, against both hypotheses.
    res["rope_caches_tf4_view"] = rope_caches(oa.ORTGenAIAdapter(patch=UNFUSED).build(gguf), hub)
    if tf4_py:
        res["tf4"] = gemma3_tf4(tf4_py, hub, work, ids, ref)
    return res


def gemma3_tf4(tf4_py, hub, work, ids, ref):
    """The stock builder of the newest onnxruntime-genai whose builder imports under transformers 4 (the
    --tf4-py environment), on the config as Google ships Gemma 3: build, then the RoPE caches it wrote."""
    t = {}
    np.save(work / "ids.npy", np.array(ids))
    r = subprocess.run([tf4_py, str(Path(__file__)), "--worker-hf-logits", str(hub), str(work / "ids.npy"),
                        str(work / "hf_tf4.npy")], capture_output=True, text=True)
    if r.returncode == 0:
        t["transformers_vs_reference"] = compare(np.load(work / "hf_tf4.npy"), ref)["worst"]
        t["transformers"] = r.stdout.strip().splitlines()[-1]
    else:
        t["transformers_error"] = r.stderr.strip().splitlines()[-1:]
    out = work / "g-tf4-stock"
    if out.exists():
        shutil.rmtree(out)
    r = subprocess.run([tf4_py, "-m", "onnxruntime_genai.models.builder", "-i", str(hub), "-o", str(out), "-p", "fp32",
                        "-e", "cpu", "-c", str(Path.home() / "ort_keramos" / "cache")], capture_output=True, text=True)
    t["build"] = "ok" if r.returncode == 0 else (r.stderr.strip().splitlines() or [""])[-1]
    if r.returncode == 0:
        t["attention"] = oa._attention_ops(out / "model.onnx")
        t["rope_caches"] = rope_caches(out, hub)
        try:
            got = run_dir(engine("none"), out, [ids])
            t["vs_reference"] = compare(got, ref)
        except Exception as e:  # noqa: BLE001
            t["run"] = f"{type(e).__name__}: {str(e)[:200]}"
    return t


def rope_caches(out, hub):
    """Each cos/sin cache initialiser of a built Gemma 3 graph against cos/sin(p * inv_freq) for the local and the
    global theta, each with and without the global layers' linear factor: max |difference|."""
    import onnx
    from onnx import numpy_helper
    c = json.loads((Path(hub) / "config.json").read_text(encoding="utf-8"))
    hd, factor = c["head_dim"], (c.get("rope_scaling") or {}).get("factor", 1.0)
    m = onnx.load(str(Path(out) / "model.onnx"))
    res = {}
    for init in m.graph.initializer:
        if "cos_cache" not in init.name and "sin_cache" not in init.name:
            continue
        got = numpy_helper.to_array(init).astype(np.float64)
        p = np.arange(got.shape[0])[:, None]
        row = {}
        for tname, theta in (("local", c["rope_local_base_freq"]), ("global", c["rope_theta"])):
            for fname, f in (("unscaled", 1.0), (f"factor {factor}", factor)):
                inv = 1.0 / (f * theta ** (np.arange(0, hd, 2) / hd))
                want = (np.cos if "cos" in init.name else np.sin)(p * inv[None, :got.shape[1]])
                row[f"{tname} theta, {fname}"] = float(np.max(np.abs(got - want)))
        best = min(row, key=row.get)
        res[init.name] = {"matches": best, "max_abs": row}
    return res


def worker_hf_logits(model_dir, ids_npy, out_npy):
    import torch
    import transformers
    torch.set_num_threads(2)
    m = transformers.AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32).eval()
    ids = np.load(ids_npy).tolist()
    with torch.no_grad():
        lg = m(input_ids=torch.tensor([ids])).logits[0].float().numpy()
    np.save(out_npy, lg)
    from importlib.metadata import version
    print(f"transformers {transformers.__version__} ({type(m).__name__}), onnxruntime-genai "
          f"{version('onnxruntime-genai')}, onnxruntime {version('onnxruntime')}")


# ---------------------------------------------------------------- Mistral, GGUF loader

def mistral(tf4_py=None, name="tp_mistral"):
    from transformers import AutoConfig
    gguf = rc.gguf_of(SUITE / name)
    res = {}
    if tf4_py:
        # The same export through the stock builder of onnxruntime-genai 0.15.2 (transformers 4.57.6).
        ex, _, _ = engine("none").build_dir(gguf)
        engine("none").build(gguf)
        out = Path.home() / "ort_keramos" / "probe" / "m-tf4-stock"
        if out.exists():
            shutil.rmtree(out)
        r = subprocess.run([tf4_py, "-m", "onnxruntime_genai.models.builder", "-i", str(ex), "-o", str(out), "-p",
                            "fp32", "-e", "cpu", "-c", str(Path.home() / "ort_keramos" / "cache")],
                           capture_output=True, text=True)
        res["stock_0.15.2"] = ({"attention": oa._attention_ops(out / "model.onnx"),
                                "genai_config_sliding_window": json.loads((out / "genai_config.json").read_text(
                                    encoding="utf-8"))["model"]["decoder"].get("sliding_window")}
                               if r.returncode == 0 else {"build": (r.stderr.strip().splitlines() or [""])[-1]})
    for label, patch in (("stock", "none"), ("unfused", UNFUSED)):
        a = engine(patch)
        out = a.build(gguf)
        g = json.loads((out / "genai_config.json").read_text(encoding="utf-8"))
        res[label] = {"attention": oa._attention_ops(out / "model.onnx"),
                      "genai_config_sliding_window": g["model"]["decoder"].get("sliding_window")}
    ex, _, _ = engine("none").build_dir(gguf)
    c = AutoConfig.from_pretrained(str(ex))
    res["MistralConfig"] = {"sliding_window": c.sliding_window, "layer_types": getattr(c, "layer_types", None)}
    ids = n_ids(name)
    if name == "tp_mistral":  # head_size 4: only the MHA build runs on CPU
        got = run(engine(MHA), name, [ids])
        res["mha_vs_reference"] = compare(got, reference(name, ids))["worst"]
        res["mha_vs_reference_without_window"] = compare(got, reference(name, ids, no_window=True))["worst"]
    else:  # head_size 16: the stock graph runs, up to window_size + 16 tokens (capacity below)
        ids = n_ids(name, 16)
        got = run(engine("none"), name, [ids])
        res["stock_vs_reference"] = compare(got, reference(name, ids))
        res["stock_vs_reference_without_window"] = compare(got, reference(name, ids, no_window=True))
        res["capacity"] = capacity(engine("none").build(gguf))
    return res


def capacity(built):
    """Tokens a stock generator takes one at a time (max_length 64) before its first error, for genai_config's
    sliding_window.window_size as built, set to 8 and 12, and with the section removed (copies of the build)."""
    og = engine("none").og
    ids = ids_for(24, 64)
    work = Path.home() / "ort_keramos" / "probe" / "capacity"
    out = {}
    for label, w in (("as built", None), ("window_size 8", 8), ("window_size 12", 12), ("no sliding_window", 0)):
        d = work / label.replace(" ", "_")
        if d.exists():
            shutil.rmtree(d)
        shutil.copytree(built, d)
        gc = json.loads((d / "genai_config.json").read_text(encoding="utf-8"))
        if w == 0:
            gc["model"]["decoder"].pop("sliding_window", None)
        elif w:
            gc["model"]["decoder"]["sliding_window"]["window_size"] = w
        (d / "genai_config.json").write_text(json.dumps(gc, indent=1), encoding="utf-8")
        cfg = og.Config(str(d))
        cfg.clear_providers()
        m = og.Model(cfg)
        gp = og.GeneratorParams(m)
        gp.set_search_options(max_length=64)
        g = og.Generator(m, gp)
        n, err = 0, None
        try:
            for t in ids:
                g.append_tokens([t])
                n += 1
        except Exception as e:  # noqa: BLE001
            err = str(e).strip().splitlines()[-1][-160:]
        out[label] = {"tokens": n, "error": err}
        del g, gp, m
    shutil.rmtree(work)
    return out


def gguf_loader():
    name = "pk_llama"
    ids = n_ids(name)
    a, b = engine(UNFUSED), engine(UNFUSED, loader="gguf")
    x, y = run(a, name, [ids]), run(b, name, [ids])
    return {"gguf_loader_vs_export": bitdiff(y, x), "gguf_loader_vs_reference": compare(y, reference(name, ids))["worst"]}


SECTIONS = {"kernel": kernel, "builds": builds, "attribution": attribution, "consistency": consistency,
            "generate": generate, "gemma3": None, "mistral": mistral, "mistral_hd16": None, "gguf": gguf_loader}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--worker-hf-logits"]:
        worker_hf_logits(*argv[1:4])
        return 0
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json")
    ap.add_argument("--only")
    ap.add_argument("--tf4-py")
    a = ap.parse_args(argv)
    res = {"adapter": engine("none").hello(), "tol": TOL}
    for name, fn in SECTIONS.items():
        if a.only and a.only != name:
            continue
        res[name] = (gemma3(a.tf4_py) if name == "gemma3" else mistral(a.tf4_py) if name == "mistral"
                     else mistral(a.tf4_py, "tp_mistral_hd16") if name == "mistral_hd16" else fn())
        print(name, json.dumps(res[name], default=str)[:4000], flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, default=str) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
