#!/usr/bin/env python3
"""Probes of transformers beyond the suite's verdicts, through the same adapter (FINDINGS.txt reads them).

    python adapters/hf/probe.py --json OUT.json [--only NAME]

Run from the keramos root in the pinned environment. Each probe uses the suite's models and the numpy
reference (tools/ir_numpy.py); nothing is written except --json and the adapter's export cache.

  attribution   transformers' prefill logits (sdpa and eager) of 16 ids (48 for the windowed models, so the
                window bites) against the file's own document; error = |a - b| / max(1, max|ref|), the
                suite's ROUNDING measure, tol 1e-5.
  consistency   inside transformers (sdpa): one prefill against token-by-token decode, chunks 5 + 6 + 5, and
                a second run: bit identity or the worst difference; each against the reference.
  generate      model.generate (greedy, bad_words_ids [[0]] as the suite bans id 0) on each model case's
                prompts: the continuation against the suite's.
  mamba         a multi-token forward after a prefill (chunks 8 + 8, and 4 + 3 as the snapshot trace does),
                against the reference and against token-by-token decode; the same with the selective scan
                seeded from the cached state (the fix); a single-token step after activate_past_recording
                (the path assisted decoding takes).
  swa_rewind    tp_mistral and tp_gemma3: prefill 48, truncate to 10 and to 40, forward 4: Cache.crop as
                transformers ships it, with activate_past_recording, and the adapter's re-prefill; each
                against a fresh prefill and the reference.
  snapshot      snapshot at 8, forward 4, restore, forward the same 4: bit identity.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(HERE))
import ir_conformance as rc  # noqa: E402
import ir_numpy  # noqa: E402
import hf_adapter  # noqa: E402

SUITE = ROOT / "conformance" / "models"
TOL = 1e-5
MODELS = ["tp_qwen3", "tp_mistral", "tp_gemma3", "tp_gptneox", "tp_mamba1", "pk_llama", "pk_moe", "pk_gpt2"]
WINDOWED = {"tp_mistral", "tp_gemma3"}


def ids_for(vocab, n=16, seed=7):
    rng = np.random.default_rng(seed)
    return [int(x) for x in rng.integers(1, vocab, size=n)]


def compare(got, ref):
    got, ref = np.asarray(got, np.float64), np.asarray(ref, np.float64)
    scale = max(1.0, float(np.max(np.abs(ref))))
    per = np.max(np.abs(got - ref), axis=1) / scale
    first = next((int(p) for p, e in enumerate(per) if e > TOL), None)
    flips = [int(p) for p in range(len(got)) if int(np.argmax(got[p])) != int(np.argmax(ref[p]))]
    return {"worst": float(per.max()), "first_past_tol": first, "greedy_differs_at": flips,
            "per_position": [float(f"{e:.3g}") for e in per]}


def bitdiff(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return {"bit_identical": bool(np.array_equal(a.view(np.uint32), b.view(np.uint32))),
            "values_differing": int(np.sum(a.view(np.uint32) != b.view(np.uint32))),
            "worst_abs": float(np.max(np.abs(a.astype(np.float64) - b)))}


class Engine:
    def __init__(self, **kw):
        self.a = hf_adapter.HFAdapter(**kw)

    def run(self, name, chunks):
        s = self.a.open(rc.gguf_of(SUITE / name))["session"]
        try:
            return np.concatenate([self.a.forward(s, c, ["logits"])["logits"] for c in chunks])
        finally:
            self.a.close(s)


def reference(name, ids):
    doc, W, _, _ = rc.load_model(SUITE / name)
    return ir_numpy.forward(doc, W, ids)


def n_ids(name):
    doc, *_ = rc.load_model(SUITE / name)
    n = 48 if name in WINDOWED else min(16, doc["model"]["ctx"])
    return ids_for(doc["model"]["vocab"], n)


def attribution(eng, eager, name):
    ids = n_ids(name)
    ref = reference(name, ids)
    out = {"n": len(ids), "sdpa": compare(eng.run(name, [ids]), ref)}
    if not name.startswith("tp_mamba"):
        out["eager"] = compare(eager.run(name, [ids]), ref)
    return out


def consistency(eng, name):
    ids = n_ids(name)[:16]
    ref = reference(name, ids)
    pre = eng.run(name, [ids])
    tok = eng.run(name, [[t] for t in ids])
    ch = eng.run(name, [ids[:5], ids[5:11], ids[11:]])
    return {"prefill_vs_token_by_token": bitdiff(pre, tok), "prefill_vs_chunks_5_6_5": bitdiff(pre, ch),
            "prefill_run_to_run": bitdiff(pre, eng.run(name, [ids])),
            "token_by_token_vs_reference": compare(tok, ref), "chunks_vs_reference": compare(ch, ref)}


def generate(eng, name):
    import torch
    case = json.loads((SUITE / name / "case.json").read_text(encoding="utf-8"))
    m = eng.a._model(rc.gguf_of(SUITE / name))["model"]
    res = []
    for pr in case["prompts"]:
        x = torch.tensor([pr["ids"]], dtype=torch.long)
        with torch.no_grad():
            g = m.generate(x, max_new_tokens=len(pr["continuation"]), do_sample=False, bad_words_ids=[[0]],
                           eos_token_id=None, pad_token_id=0, attention_mask=torch.ones_like(x))
        got = [int(t) for t in g[0, len(pr["ids"]):]]
        res.append({"prompt": pr["ids"], "same": got == pr["continuation"], "got": got, "suite": pr["continuation"]})
    return {"same": sum(r["same"] for r in res), "of": len(res), "prompts": res}


# ---------------------------------------------------------------- Mamba: continuation by chunks

def seeded_scan_patch():
    """modeling_mamba.mamba_selective_scan seeded from the layer's cached recurrent state when the cache has
    one (the fix): a forward pre-hook on each mixer records the state; the patched scan starts from it."""
    import torch
    from transformers.models.mamba import modeling_mamba as mm
    orig = mm.mamba_selective_scan
    box = {"init": None}

    def scan(hidden_states, dt, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
             return_last_state=False, **kw):
        init = box["init"]
        if init is None:
            return orig(hidden_states, dt, A, B, C, D=D, z=z, delta_bias=delta_bias, delta_softplus=delta_softplus,
                        return_last_state=return_last_state, **kw)
        import torch.nn.functional as F
        if delta_bias is not None:
            dt = dt + delta_bias.to(dt.dtype)[..., None]
        if delta_softplus:
            dt = F.softplus(dt)
        B, C = B.transpose(1, 2), C.transpose(1, 2)
        dA = torch.exp(A[None, :, None, :] * dt[:, :, :, None])
        dBu = dt[:, :, :, None] * B[:, None, :, :].float() * hidden_states[:, :, :, None].float()
        s = init.clone()
        outs = []
        for t in range(hidden_states.shape[-1]):
            s = dA[:, :, t] * s + dBu[:, :, t]
            outs.append(torch.matmul(s.to(hidden_states.dtype), C[:, t, :].unsqueeze(-1))[:, :, 0])
        y = torch.stack(outs, dim=-1)
        if D is not None:
            y = y + hidden_states * D[None, :, None]
        if z is not None:
            y = y * F.silu(z)
        return (y, s) if return_last_state else y

    def hook(module, args, kwargs):
        cp = kwargs.get("cache_params")
        box["init"] = (cp.layers[module.layer_idx].recurrent_states[0].clone()
                       if cp is not None and cp.has_previous_state(module.layer_idx) else None)

    return mm, orig, scan, hook


def mamba(eng, name="tp_mamba1"):
    ids = n_ids(name)
    ref = reference(name, ids)
    out = {"prefill": compare(eng.run(name, [ids]), ref),
           "token_by_token": compare(eng.run(name, [[t] for t in ids]), ref),
           "chunks_8_8": compare(eng.run(name, [ids[:8], ids[8:]]), ref),
           "chunks_4_3": compare(eng.run(name, [ids[:4], ids[4:7]]), ref[:7]),
           "prefill_4_then_tokens_then_chunk_2": compare(eng.run(name, [ids[:4], [ids[4]], ids[5:7]]), ref[:7])}
    m = eng.a._model(rc.gguf_of(SUITE / name))["model"]
    mm, orig, scan, hook = seeded_scan_patch()
    handles = [layer.mixer.register_forward_pre_hook(hook, with_kwargs=True) for layer in m.backbone.layers]
    mm.mamba_selective_scan = scan
    try:
        out["chunks_8_8_seeded_scan"] = compare(eng.run(name, [ids[:8], ids[8:]]), ref)
        out["chunks_4_3_seeded_scan"] = compare(eng.run(name, [ids[:4], ids[4:7]]), ref[:7])
    finally:
        mm.mamba_selective_scan = orig
        for h in handles:
            h.remove()
    # The path assisted decoding takes: activate_past_recording, then one token at a time.
    import torch
    from transformers import DynamicCache
    cache = DynamicCache(config=m.config)
    cache.activate_past_recording()
    rows = []
    with torch.no_grad():
        o = m(input_ids=torch.tensor([ids[:8]]), cache_params=cache, use_cache=True)
        rows.append(o.logits[0].float().numpy())
        for t in ids[8:12]:
            o = m(input_ids=torch.tensor([[t]]), cache_params=o.cache_params, use_cache=True)
            rows.append(o.logits[0].float().numpy())
    out["past_recording_prefill_8_then_4_single"] = compare(np.concatenate(rows), ref[:12])
    # Does generate reach the multi-token path? Prompt-lookup (assisted) decoding verifies several candidate
    # tokens in one forward over the cache; a repetitive prompt makes the lookup propose candidates.
    prompt = (ids[:4] * 4)[:16]
    x = torch.tensor([prompt])
    plain_ref = []
    seq = list(prompt)
    doc, W, _, _ = rc.load_model(SUITE / name)
    for _ in range(12):  # the reference's greedy continuation (id 0 banned, as the suite does)
        row = ir_numpy.forward(doc, W, seq)[-1].copy()
        row[0] = -np.inf
        seq.append(int(np.argmax(row)))
    plain_ref = seq[len(prompt):]
    gen = {}
    for label, kw in (("greedy", {}), ("prompt_lookup_3", {"prompt_lookup_num_tokens": 3})):
        try:
            with torch.no_grad():
                g = m.generate(x, max_new_tokens=12, do_sample=False, bad_words_ids=[[0]], eos_token_id=None,
                               pad_token_id=0, attention_mask=torch.ones_like(x), **kw)
            got = [int(t) for t in g[0, len(prompt):]]
            gen[label] = {"same_as_reference": got == plain_ref, "got": got}
        except Exception as e:  # noqa: BLE001
            gen[label] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
    gen["reference"] = plain_ref
    # Multi-turn cache reuse (transformers' documented pattern for continuing a conversation): generate turn 1,
    # then generate turn 2 = turn 1's sequence + 3 new ids, passing turn 1's cache; against a fresh generate.
    kw = {"do_sample": False, "bad_words_ids": [[0]], "eos_token_id": None, "pad_token_id": 0}
    try:
        with torch.no_grad():
            x1 = torch.tensor([ids[:6]])
            o1 = m.generate(x1, max_new_tokens=4, return_dict_in_generate=True, attention_mask=torch.ones_like(x1), **kw)
            cache1 = getattr(o1, "past_key_values", None) or getattr(o1, "cache_params", None)
            x2 = torch.cat([o1.sequences, torch.tensor([ids[6:9]])], dim=1)
            g2 = {"max_new_tokens": 1, "attention_mask": torch.ones_like(x2), "return_dict_in_generate": True,
                  "output_logits": True, **kw}
            fresh = m.generate(x2, **g2).logits[0][0].float().numpy()
            reused = m.generate(x2, cache_params=copy.deepcopy(cache1), **g2).logits[0][0].float().numpy()
            # Only the uncached ids (the last generated one, which turn 1 never fed, and the 3 new ones).
            xn = x2[:, -4:]
            gn = {**g2, "attention_mask": torch.ones_like(xn)}
            new_only = m.generate(xn, cache_params=copy.deepcopy(cache1), **gn).logits[0][0].float().numpy()
            mm, orig, scan, hook = seeded_scan_patch()
            handles = [layer.mixer.register_forward_pre_hook(hook, with_kwargs=True) for layer in m.backbone.layers]
            mm.mamba_selective_scan = scan
            try:
                new_only_seeded = m.generate(xn, cache_params=copy.deepcopy(cache1), **gn).logits[0][0].float().numpy()
            finally:
                mm.mamba_selective_scan = orig
                for h in handles:
                    h.remove()
        seq2 = [int(t) for t in x2[0]]
        ref2 = ir_numpy.forward(doc, W, seq2)[-1:]
        out["generate_reusing_cache"] = {
            "cache_type": type(cache1).__name__, "turn2_len": len(seq2),
            "fresh_vs_reference": compare(fresh[None], ref2)["worst"],
            "full_sequence_with_cache_vs_reference": compare(reused[None], ref2)["worst"],
            "new_ids_only_with_cache_vs_reference": compare(new_only[None], ref2)["worst"],
            "new_ids_only_with_cache_seeded_scan_vs_reference": compare(new_only_seeded[None], ref2)["worst"]}
    except Exception as e:  # noqa: BLE001
        out["generate_reusing_cache"] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
    out["generate"] = gen
    return out


# ---------------------------------------------------------------- rewind into a sliding window

def swa_rewind(eng, name, L=48):
    import torch
    from transformers import DynamicCache
    doc, *_ = rc.load_model(SUITE / name)
    ids = ids_for(doc["model"]["vocab"], n=L, seed=11)
    m = eng.a._model(rc.gguf_of(SUITE / name))["model"]
    res = []
    for n in (10, 40, 2):
        k = 4
        ref = reference(name, ids[:n + k])[n:]
        fresh = eng.run(name, [ids[:n + k]])[n:]
        row = {"prefill": L, "truncate_to": n, "then": k}
        for label, record in (("crop", False), ("crop_after_activate_past_recording", True)):
            cache = DynamicCache(config=m.config)
            if record:
                cache.activate_past_recording()
            with torch.no_grad():
                o = m(input_ids=torch.tensor([ids]), past_key_values=cache, use_cache=True)
                try:
                    o.past_key_values.crop(n - L)
                except Exception as e:  # noqa: BLE001
                    row[label] = {"refused": f"{type(e).__name__}: {str(e)[:160]}"}
                    continue
                o2 = m(input_ids=torch.tensor([ids[n:n + k]]), past_key_values=o.past_key_values, use_cache=True)
            got = o2.logits[0].float().numpy()
            row[label] = {"vs_fresh_prefill": bitdiff(got, fresh), "vs_reference": compare(got, ref),
                          "seq_length_after_crop": int(o.past_key_values.get_seq_length())}
        s = eng.a.open(rc.gguf_of(SUITE / name))["session"]
        try:
            eng.a.forward(s, ids, [])
            tr = eng.a.truncate(s, n, [])
            got = eng.a.forward(s, ids[n:n + k], ["logits"])["logits"]
        finally:
            eng.a.close(s)
        row["adapter"] = {"how": tr["_how"], "vs_fresh_prefill": bitdiff(got, fresh), "vs_reference": compare(got, ref)}
        res.append(row)
    return res


# ---------------------------------------------------------------- transformers' own GGUF reader

CONFIG_KEYS = ("head_dim", "num_key_value_heads", "query_pre_attn_scalar", "rope_parameters", "sliding_window",
               "layer_types", "tie_word_embeddings", "n_positions", "n_inner", "num_local_experts",
               "intermediate_size", "rms_norm_eps", "hidden_activation", "hidden_act", "activation_function")


def _logits(model, ids):
    import torch
    with torch.no_grad():
        return model(input_ids=torch.tensor([ids]), use_cache=False).logits[0].float().numpy()


def gguf_loader(name, mistral_name="Mistral-7B-v0.1"):
    """transformers' from_pretrained(gguf_file=...) against the keramos export: load result, tensors, config
    fields that differ, logits against the reference; for gemma3, the export with the GGUF reader's config
    values substituted one at a time (what each costs)."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    gg = hf_adapter.HFAdapter(loader="gguf", gguf_name=mistral_name)
    ex = hf_adapter.HFAdapter()
    gguf = rc.gguf_of(SUITE / name)
    ids = n_ids(name)
    ref = reference(name, ids)
    out = {}
    try:
        mg = gg._model(gguf)
    except NotImplementedError as e:
        return {"loads": False, "skip": str(e)[:200]}
    except Exception as e:  # noqa: BLE001
        out = {"loads": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}
        if "does not load cleanly" not in str(e):
            return out
        # Load it anyway, as a user who ignores the warning would, to see what runs.
        src = gguf if name != "tp_mistral" else gg._as_llama(gguf, "x")
        m = AutoModelForCausalLM.from_pretrained(str(Path(src).parent), gguf_file=Path(src).name, dtype=torch.float32)
        mg = {"model": m.eval()}
    me = ex._model(gguf)["model"]
    m = mg["model"]
    out.setdefault("loads", True)
    out["class"] = type(m).__name__
    sd_g, sd_e = m.state_dict(), me.state_dict()
    tens = {}
    for k, v in sd_e.items():
        if k not in sd_g:
            tens[k] = "absent from the GGUF reader's model"
        elif tuple(sd_g[k].shape) != tuple(v.shape):
            tens[k] = f"shape {tuple(sd_g[k].shape)} != {tuple(v.shape)}"
        else:
            a, b = sd_g[k].detach().float().numpy(), v.detach().float().numpy()
            nd = int(np.sum(a.view(np.uint32) != b.view(np.uint32)))
            if nd:
                tens[k] = f"{nd} values differ, worst {float(np.max(np.abs(a - b))):.2e}"
    out["tensors_differing_from_export"] = tens
    cg, ce = m.config.to_dict(), me.config.to_dict()
    out["config_differs"] = {k: {"gguf_reader": cg.get(k), "export": ce.get(k)} for k in CONFIG_KEYS
                             if cg.get(k) != ce.get(k) and (k in cg or k in ce)}
    try:
        lg = _logits(m, ids)
        out["vs_reference"] = compare(lg, ref)
        if name in WINDOWED:
            doc, W, _, _ = rc.load_model(SUITE / name)
            out["vs_reference_no_window"] = compare(lg, ir_numpy.forward(doc, W, ids, no_window=True))
    except Exception as e:  # noqa: BLE001
        out["forward_error"] = f"{type(e).__name__}: {str(e)[:200]}"
        return out
    if name == "tp_gemma3":
        exdir = Path(me.config._name_or_path)
        subs = {"query_pre_attn_scalar": cg["query_pre_attn_scalar"],
                "rope_parameters": cg["rope_parameters"]}
        out["export_with_reader_values"] = {}
        for label, keys in (("query_pre_attn_scalar", ["query_pre_attn_scalar"]),
                            ("rope_parameters", ["rope_parameters"]),
                            ("both", ["query_pre_attn_scalar", "rope_parameters"])):
            cfg = AutoConfig.from_pretrained(str(exdir))
            for k in keys:
                setattr(cfg, k, copy.deepcopy(subs[k]))
            mx = AutoModelForCausalLM.from_pretrained(str(exdir), config=cfg, dtype=torch.float32).eval()
            lx = _logits(mx, ids)
            out["export_with_reader_values"][label] = {"vs_reference": compare(lx, ref)["worst"],
                                                       "vs_gguf_reader": bitdiff(lx, lg)}
    return out


def snapshot(eng, name):
    ids = n_ids(name)[:16]
    a = eng.a
    s = a.open(rc.gguf_of(SUITE / name))["session"]
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
    ap.add_argument("--json")
    ap.add_argument("--only")
    a = ap.parse_args(argv)
    eng, eager = Engine(), Engine(attn="eager")
    res = {"adapter": eng.a.hello(), "tol": TOL, "models": {}}
    for name in MODELS:
        if a.only and a.only not in name:
            continue
        r = {"attribution": attribution(eng, eager, name), "consistency": consistency(eng, name),
             "snapshot_restore": snapshot(eng, name), "generate": generate(eng, name),
             "load_notes": eng.a.load_notes.get(name)}
        res["models"][name] = r
        at = r["attribution"]
        print(name, {k: (v["worst"], v["first_past_tol"], v["greedy_differs_at"]) for k, v in at.items() if k != "n"},
              {k: (v["bit_identical"] if "bit_identical" in v else v["worst"]) for k, v in r["consistency"].items()},
              "snapshot", r["snapshot_restore"]["bit_identical"], "generate", f"{r['generate']['same']}/{r['generate']['of']}",
              flush=True)
    if not a.only or "mamba" in a.only:
        res["mamba"] = mamba(eng)
        print("mamba", {k: (v["worst"], v["first_past_tol"]) for k, v in res["mamba"].items() if "worst" in v},
              res["mamba"]["generate"], res["mamba"]["generate_reusing_cache"], flush=True)
    res["swa_rewind"] = {}
    for name in sorted(WINDOWED):
        if a.only and a.only not in name:
            continue
        res["swa_rewind"][name] = rows = swa_rewind(eng, name)
        for row in rows:
            print("swa rewind", name, row["truncate_to"],
                  {k: (v.get("refused") or (v["vs_fresh_prefill"]["bit_identical"], v["vs_reference"]["worst"]))
                   for k, v in row.items() if isinstance(v, dict) and k != "adapter"},
                  "adapter", row["adapter"]["how"][:40], row["adapter"]["vs_reference"]["worst"], flush=True)
    res["gguf_reader"] = {}
    for name in MODELS:
        if a.only and a.only not in name:
            continue
        g = res["gguf_reader"][name] = gguf_loader(name)
        print("gguf reader", name, {k: g[k] for k in ("loads", "class", "skip", "error", "forward_error") if k in g},
              "tensors differing", len(g.get("tensors_differing_from_export", {})),
              "config", sorted(g.get("config_differs", {})),
              {k: (g[k]["worst"], g[k]["first_past_tol"]) for k in ("vs_reference", "vs_reference_no_window") if k in g},
              g.get("export_with_reader_values", ""), flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
