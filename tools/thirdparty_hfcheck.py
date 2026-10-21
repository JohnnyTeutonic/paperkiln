#!/usr/bin/env python3
"""Re-derive the HF side of a thirdparty_parity.py receipt with plain argmax greedy (hf_greedy) and
compare it with the recorded HF replies (text and token count). For receipts written before
hf_greedy replaced hf.generate (which applies the checkpoint's generation_config): if every reply
is the same, the recorded greedy_same results stand.

    python3 tools/thirdparty_hfcheck.py --receipt R.jsonl --gguf M.gguf --hf-dir DIR [--raw]
        [--hf-weights gguf|native] [--max-new 32]

Writes R.jsonl.hf_argmax_check.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thirdparty_parity as tp  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--receipt", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--prompts", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                      "corpus", "eval", "chat_prompts_v1.jsonl"))
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--hf-weights", choices=("gguf", "native"), default="gguf")
    ap.add_argument("--max-new", type=int, default=32)
    a = ap.parse_args(argv)
    import torch
    from transformers import AutoTokenizer
    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(a.hf_dir)
    hf = tp.load_hf(a)
    eos = tp.eos_ids(tok, hf)
    prompts = [json.loads(x)["prompt"] for x in open(a.prompts, encoding="utf-8")]
    rows = [json.loads(x) for x in open(a.receipt, encoding="utf-8")]
    same, diffs = 0, []
    for r in rows:
        _, _, hf_new = tp.hf_side(a, hf, tok, eos, prompts[r["i"]])
        ok = tok.decode(hf_new, skip_special_tokens=True) == r["hf_reply"] and len(hf_new) == r["hf_tokens"]
        same += ok
        if not ok:
            diffs.append(r["i"])
        print(r["i"], ok, flush=True)
    out = {"receipt": os.path.basename(a.receipt), "prompts": len(rows), "hf_argmax_equals_recorded": same,
           "differ": diffs, "generation_config": hf.generation_config.to_diff_dict()}
    json.dump(out, open(a.receipt + ".hf_argmax_check.json", "w"), indent=1, default=str)
    print(json.dumps(out, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
