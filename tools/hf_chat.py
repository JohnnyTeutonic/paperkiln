#!/usr/bin/env python3
"""Chat with a paperkiln model from Python. No C++ required.

    python hf_chat.py HF_DIR                      # interactive
    python hf_chat.py HF_DIR --prompt "once upon a time"
    options: --tokens 60  --temp 0.8  --topk 40  --seed 0

HF_DIR is the folder written by `tools/hf_export.py` (config.json,
model.safetensors, tokenizer.json). It opens with transformers'
AutoModelForCausalLM like any Llama-family checkpoint; this script is a
plain sampling loop around it. Requires transformers, tokenizers and
torch (`pip install transformers torch`).

The loop is written by hand rather than through model.generate(): the
word-level vocabulary has no real end-of-sequence token (id 0 is <unk>,
which the config also lists as eos), so generate() would stop as soon as
it sampled it. Here the special tokens are simply banned at every step,
the same rule paperkiln's own sampler applies, and greedy decoding
(--temp 0) reproduces `mtstudio sample --topk 1` token for token
(tools/hf_export_verify.py is that receipt).

These are tiny models trained for minutes on a slice of TinyStories with
a 4096-word vocabulary: expect children's-story English in lower case,
a short attention span, and the occasional wandering pronoun.
"""
from __future__ import annotations

import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hf_dir")
    ap.add_argument("--prompt", help="one-shot prompt instead of the loop")
    ap.add_argument("--tokens", type=int, default=60)
    ap.add_argument("--temp", type=float, default=0.8, help="0 = greedy")
    ap.add_argument("--topk", type=int, default=40)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    import torch
    from tokenizers import Tokenizer
    from transformers import AutoModelForCausalLM

    tok = Tokenizer.from_file(os.path.join(args.hf_dir, "tokenizer.json"))
    vocab = tok.get_vocab()
    inv = {i: t for t, i in vocab.items()}
    banned = [vocab[t] for t in ("<unk>", "<s>", "</s>", "<pad>") if t in vocab]
    model = AutoModelForCausalLM.from_pretrained(args.hf_dir, torch_dtype=torch.float32)
    model.eval()
    gen = torch.Generator()
    if args.seed is not None:
        gen.manual_seed(args.seed)
    max_ctx = int(getattr(model.config, "max_position_embeddings", 256))

    def reply(prompt):
        # Word-level tokenizer over lower-cased, whitespace-split text:
        # lower-case the prompt the way the training corpus was.
        ids = tok.encode(prompt.lower()).ids
        words = []
        with torch.no_grad():
            for _ in range(args.tokens):
                ctx = ids[-max_ctx:]
                logits = model(torch.tensor([ctx])).logits[0, -1]
                for b in banned:
                    logits[b] = -1e30
                if args.temp <= 0:
                    nxt = int(torch.argmax(logits))
                else:
                    probs = torch.softmax(logits / args.temp, dim=-1)
                    if args.topk > 0:
                        top_p, top_i = torch.topk(probs, min(args.topk, probs.numel()))
                        nxt = int(top_i[torch.multinomial(top_p / top_p.sum(), 1, generator=gen)])
                    else:
                        nxt = int(torch.multinomial(probs, 1, generator=gen))
                ids.append(nxt)
                words.append(inv.get(nxt, "?"))
        return " ".join(words)

    if args.prompt:
        print(args.prompt.lower() + " " + reply(args.prompt))
        return 0
    print("paperkiln chat (empty line to quit)")
    while True:
        try:
            p = input("you> ").strip()
        except EOFError:
            break
        if not p:
            break
        print("model>", reply(p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
