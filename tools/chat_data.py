#!/usr/bin/env python3
"""Dialogues -> SFT examples in the chat template that `mtstudio chat` serves.

    python tools/chat_data.py sft DIALOGUES.jsonl --vocab V.gguf --T 256 --out SFT.jsonl [--max N]

The template (tools/mtstudio.cpp chat_prompt_bpe): every past turn is
<|user|>U<|assistant|>A<|endoftext|>, and a reply is prompted by
<|user|>U<|assistant|>. Each assistant turn k becomes one example:
  prompt     = turns 1..k-1 in full, then <|user|>U_k<|assistant|>
  completion = A_k   (mtstudio appends <|endoftext|> and masks the prompt)
An example whose BPE length (prompt + completion + end-of-text) exceeds
T + 1 is dropped and counted, never truncated: a cut reply would teach the
model to stop mid-sentence.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpe  # noqa: E402


def turn(u, a):
    return f"<|user|>{u}<|assistant|>{a}<|endoftext|>"


def examples(messages):
    hist = ""
    for k in range(0, len(messages) - 1, 2):
        u, a = messages[k]["content"], messages[k + 1]["content"]
        yield {"prompt": hist + f"<|user|>{u}<|assistant|>", "completion": a}
        hist += turn(u, a)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sft")
    s.add_argument("dialogues")
    s.add_argument("--vocab", required=True)
    s.add_argument("--T", type=int, default=256)
    s.add_argument("--out", required=True)
    s.add_argument("--max", type=int, default=0, help="stop after this many examples (0 = all)")
    a = ap.parse_args(argv)
    tok = bpe.read_gguf(a.vocab)
    kept = dropped = 0
    with open(a.dialogues, encoding="utf-8") as f, open(a.out, "w", encoding="utf-8", newline="\n") as g:
        for line in f:
            d = json.loads(line)
            for ex in examples(d["messages"]):
                n = len(tok.encode(ex["prompt"])) + len(tok.encode(ex["completion"])) + 1
                if n > a.T + 1:
                    dropped += 1
                    continue
                ex["uri"] = d.get("uri", "")
                g.write(json.dumps(ex, ensure_ascii=False) + "\n")
                kept += 1
                if a.max and kept >= a.max:
                    break
            if a.max and kept >= a.max:
                break
    print(json.dumps({"kept": kept, "dropped_over_T": dropped, "out": a.out}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
