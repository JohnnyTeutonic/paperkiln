#!/usr/bin/env python3
"""The third-party architecture gate, reference leg (LOCAL_LLM_STACK.md section 7): a llama.cpp GGUF run
through the IR that spec/tools/gguf_ir.py builds from its metadata, on spec/tools/ir_numpy.py, against HF
transformers loading the same GGUF (gguf_file=..., dequantised to f32 by its own converter, an
independent reading of the tensor layout). Both get the same token ids (HF's tokenizer and chat
template), so this tests the graph, not tokenisation.

    python3 tools/thirdparty_parity.py --gguf M.gguf --hf-dir DIR --out RECEIPT.jsonl \
        [--prompts corpus/eval/chat_prompts_v1.jsonl] [--n 40] [--max-new 32] [--raw] \
        [--hf-weights gguf|native] [--low-mem]

Per prompt: the first-token logits (max |dlogit| over HF's logit range, top-1 agreement) and greedy
continuations of up to --max-new tokens (argmax of HF's forward with its KV cache, never hf.generate,
whose generation_config can add a repetition penalty or top-k even with do_sample=False, vs the IR
recomputed from position 0 each step; both stop after the tokenizer's or generation_config's eos); the token index where they first differ, if they do. The receipt
is append-only JSONL, one row per finished prompt, and a rerun resumes after the last row. The
summary goes to RECEIPT.summary.json.

--raw        base models without a chat template: the prompt text encoded as is (the tokenizer's
             own special tokens, e.g. a BOS, included as HF adds them).
--hf-weights native: HF loads the model's own safetensors in DIR (f32) instead of the GGUF, for
             families transformers cannot read from GGUF (gptneox); the GGUF must then be an f32
             conversion of those same weights (llama.cpp's convert_hf_to_gguf.py --outtype f32).
--low-mem    HF's side first for every pending prompt (ids, first-token logits, greedy reply, kept in
             RECEIPT.hf/), then HF is freed and the IR side runs: one model in memory at a time.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "spec", "tools"))  # keramos
import gguf_ir  # noqa: E402
import ir_numpy  # noqa: E402


def load_hf(a):
    import torch
    from transformers import AutoModelForCausalLM
    if a.hf_weights == "native":
        hf = AutoModelForCausalLM.from_pretrained(a.hf_dir, torch_dtype=torch.float32)
    else:
        hf = AutoModelForCausalLM.from_pretrained(a.hf_dir, gguf_file=os.path.basename(a.gguf),
                                                  torch_dtype=torch.float32)
    return hf.eval()


def eos_ids(tok, hf):
    eos = set([tok.eos_token_id] if tok.eos_token_id is not None else [])
    gen_eos = getattr(hf.generation_config, "eos_token_id", None)
    eos |= set(gen_eos if isinstance(gen_eos, list) else [gen_eos] if gen_eos is not None else [])
    return eos


def hf_side(a, hf, tok, eos, p):
    """(prompt ids, HF first-token logits, HF greedy reply ids)."""
    if a.raw:
        ids = tok(p)["input_ids"]
    else:
        ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True)
    ref, hf_new = hf_greedy(hf, ids, a.max_new, eos)
    return list(ids), ref, hf_new


def hf_greedy(hf, ids, max_new, eos):
    """(first-token logits, greedy reply): plain argmax of HF's own forward, step by step, stopping
    after an eos. Not hf.generate, which applies the checkpoint's generation_config (repetition
    penalty, top-k ...) even with do_sample=False. With a KV cache when the model returns one
    (past_key_values), else the whole sequence recomputed each step (Mamba)."""
    import torch
    out = hf(torch.tensor([ids]), use_cache=True)
    past = getattr(out, "past_key_values", None)
    logits = out.logits[0, -1]
    first = logits.numpy().copy()
    seq, new = list(ids), []
    for _ in range(max_new):
        t = int(torch.argmax(logits))  # the first maximal index, as numpy's argmax
        new.append(t)
        seq.append(t)
        if t in eos or len(new) == max_new:
            break
        if past is not None:
            out = hf(torch.tensor([[t]]), past_key_values=past, use_cache=True)
            past = out.past_key_values
        else:
            out = hf(torch.tensor([seq]))
        logits = out.logits[0, -1]
    return first, new


def ir_side(doc, W, ids, eos, max_new):
    mine = ir_numpy.forward(doc, W, ids, last_only=True)[-1]
    seq, ir_new = list(ids), []
    for _ in range(max_new):
        t = int(ir_numpy.forward(doc, W, seq, last_only=True)[-1].argmax())
        ir_new.append(t)
        seq.append(t)
        if t in eos:
            break
    return mine, ir_new


def row_of(i, ids, ref, hf_new, mine, ir_new, tok, t0):
    rng = float(ref.max() - ref.min())
    d = float(np.abs(ref - mine).max())
    first_diff = next((k for k, (x, y) in enumerate(zip(hf_new, ir_new)) if x != y),
                      None if len(hf_new) == len(ir_new) else min(len(hf_new), len(ir_new)))
    return {"i": i, "prompt_tokens": len(ids), "first_max_dlogit": d, "logit_range": rng,
            "first_rel": d / rng, "top1_same": int(ref.argmax()) == int(mine.argmax()),
            "greedy_same": hf_new == ir_new, "first_diff": first_diff, "hf_tokens": len(hf_new),
            "hf_reply": tok.decode(hf_new, skip_special_tokens=True), "seconds": round(time.time() - t0, 1)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--hf-dir", required=True, help="the model's HF config and tokenizer files (and the GGUF)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                      "corpus", "eval", "chat_prompts_v1.jsonl"))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--raw", action="store_true", help="encode the prompt text without a chat template")
    ap.add_argument("--hf-weights", choices=("gguf", "native"), default="gguf")
    ap.add_argument("--low-mem", action="store_true")
    a = ap.parse_args(argv)
    import torch
    from transformers import AutoTokenizer
    torch.set_grad_enabled(False)

    meta, shapes = gguf_ir.read_meta(a.gguf)
    doc = gguf_ir.build(meta, shapes)
    json.dump(doc, open(a.out + ".ir.json", "w", encoding="utf-8"), indent=1)
    tok = AutoTokenizer.from_pretrained(a.hf_dir)
    prompts = [json.loads(x)["prompt"] for x in open(a.prompts, encoding="utf-8")][:a.n]
    done = sum(1 for _ in open(a.out, encoding="utf-8")) if os.path.exists(a.out) else 0
    pending = [i for i in range(len(prompts)) if i >= done]
    W = None
    if a.low_mem:
        cache = a.out + ".hf"
        os.makedirs(cache, exist_ok=True)
        todo = [i for i in pending if not os.path.exists(os.path.join(cache, f"{i}.json"))]
        if todo:
            hf = load_hf(a)
            eos = eos_ids(tok, hf)
            for i in todo:
                t0 = time.time()
                ids, ref, hf_new = hf_side(a, hf, tok, eos, prompts[i])
                np.save(os.path.join(cache, f"{i}.npy"), ref)
                tmp = os.path.join(cache, f"{i}.json.tmp")
                json.dump({"ids": ids, "hf_new": hf_new, "eos": sorted(eos), "seconds": time.time() - t0},
                          open(tmp, "w"))
                os.replace(tmp, os.path.join(cache, f"{i}.json"))
                print(f"hf {i}: {len(hf_new)} tok", flush=True)
            del hf
            gc.collect()
    else:
        hf = load_hf(a)
        eos = eos_ids(tok, hf)
    W = ir_numpy.load_weights(a.gguf)
    print(f"{doc['family_hint']}: {len(prompts)} prompts, {done} done earlier", flush=True)
    with open(a.out, "a", encoding="utf-8") as f:
        for i in pending:
            t0 = time.time()
            if a.low_mem:
                rec = json.load(open(os.path.join(a.out + ".hf", f"{i}.json")))
                ids, hf_new, eos = rec["ids"], rec["hf_new"], set(rec["eos"])
                ref = np.load(os.path.join(a.out + ".hf", f"{i}.npy"))
                t0 -= rec["seconds"]
            else:
                ids, ref, hf_new = hf_side(a, hf, tok, eos, prompts[i])
            mine, ir_new = ir_side(doc, W, ids, eos, a.max_new)
            row = row_of(i, ids, ref, hf_new, mine, ir_new, tok, t0)
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"{i}: rel {row['first_rel']:.2e} top1 {row['top1_same']} greedy {row['greedy_same']} "
                  f"({len(hf_new)} tok, {row['seconds']} s)", flush=True)
    rows = [json.loads(x) for x in open(a.out, encoding="utf-8")]
    summ = {"family": doc["family_hint"], "gguf": os.path.basename(a.gguf), "prompts": len(rows),
            "top1_same": sum(r["top1_same"] for r in rows), "greedy_same": sum(r["greedy_same"] for r in rows),
            "worst_first_rel": max(r["first_rel"] for r in rows), "max_new": a.max_new,
            "raw": a.raw, "hf_weights": a.hf_weights, "hf_greedy": "argmax of HF's forward with its KV cache (no generate)"}
    json.dump(summ, open(a.out + ".summary.json", "w"), indent=1)
    if a.low_mem and len(rows) == len(prompts):
        shutil.rmtree(a.out + ".hf", ignore_errors=True)
    print(json.dumps(summ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
