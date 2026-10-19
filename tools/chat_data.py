#!/usr/bin/env python3
"""Dialogues -> SFT examples and preference pairs in the chat template that
`mtstudio chat` serves.

    python tools/chat_data.py sft DIALOGUES.jsonl --vocab V.gguf --T 256 --out SFT.jsonl [--max N] [--skip N]
    python tools/chat_data.py prompts SFT.jsonl --out PROMPTS.jsonl
    python tools/chat_data.py pairs SFT.jsonl SAMPLES.jsonl --vocab V.gguf --T 256 --out PAIRS.jsonl
    python tools/chat_data.py mix A.jsonl B.jsonl [...] --seed 1 --out MIX.jsonl

The template (tools/mtstudio.cpp chat_prompt_bpe): every past turn is
<|user|>U<|assistant|>A<|endoftext|>, and a reply is prompted by
<|user|>U<|assistant|>. Each assistant turn k becomes one example:
  prompt     = turns 1..k-1 in full, then <|user|>U_k<|assistant|>
  completion = A_k   (mtstudio appends <|endoftext|> and masks the prompt)
An example whose BPE length (prompt + completion + end-of-text) exceeds
T + 1 is dropped and counted, never truncated: a cut reply would teach the
model to stop mid-sentence. A dialogue whose text still carries a source
role marker ([INST], [/INST] or a variant) is dropped whole and counted:
about 0.3% of TinyChat conversations separate turns with bare [/INST], so
one "assistant turn" holds several (ledger 0003), and training on them
would teach the model to print markers. --skip N starts after the first N
kept examples, so a later stage can draw examples the SFT run never saw.

`prompts` writes the examples' prompts for `mtstudio sample-set`; `pairs`
joins its samples back to the examples as SPIN-style preference pairs: the
dataset's reply is chosen, the model's own sample is rejected. A sample
identical to the reference is not a pair and is counted. Each side carries
a receipt naming where it came from.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpe  # noqa: E402

MARKER = re.compile(r"\[\s*/?\W?\s*INST|INST\s*\]")


def turn(u, a):
    return f"<|user|>{u}<|assistant|>{a}<|endoftext|>"


def examples(messages):
    hist = ""
    for k in range(0, len(messages) - 1, 2):
        u, a = messages[k]["content"], messages[k + 1]["content"]
        yield {"prompt": hist + f"<|user|>{u}<|assistant|>", "completion": a}
        hist += turn(u, a)


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def write_jsonl(p, rows):
    with open(p, "w", encoding="utf-8", newline="\n") as g:
        for r in rows:
            g.write(json.dumps(r, ensure_ascii=False) + "\n")


def cmd_sft(a):
    tok = bpe.read_gguf(a.vocab)
    kept = dropped = marked = skipped = 0
    with open(a.dialogues, encoding="utf-8") as f, open(a.out, "w", encoding="utf-8", newline="\n") as g:
        for line in f:
            d = json.loads(line)
            if any(MARKER.search(m["content"]) for m in d["messages"]):
                marked += 1
                continue
            for ex in examples(d["messages"]):
                n = len(tok.encode(ex["prompt"])) + len(tok.encode(ex["completion"])) + 1
                if n > a.T + 1:
                    dropped += 1
                    continue
                if skipped < a.skip:
                    skipped += 1
                    continue
                ex["uri"] = d.get("uri", "")
                g.write(json.dumps(ex, ensure_ascii=False) + "\n")
                kept += 1
                if a.max and kept >= a.max:
                    break
            if a.max and kept >= a.max:
                break
    return {"kept": kept, "skipped": skipped, "dropped_over_T": dropped, "dropped_dialogues_with_markers": marked,
            "out": a.out}


def cmd_prompts(a):
    ex = read_jsonl(a.examples)
    write_jsonl(a.out, ({"prompt": e["prompt"], "uri": e.get("uri", "")} for e in ex))
    return {"prompts": len(ex), "out": a.out}


def cmd_pairs(a):
    tok = bpe.read_gguf(a.vocab)
    ex = read_jsonl(a.examples)
    pairs, same, over = [], 0, 0
    for s in read_jsonl(a.samples):
        e = ex[s["prompt_index"]]
        ref, mine = e["completion"], s["completion"]
        if mine.strip() == ref.strip():
            same += 1
            continue
        p = len(tok.encode(e["prompt"]))
        if p + max(len(tok.encode(ref)), len(tok.encode(mine))) + 1 > a.T + 1:
            over += 1
            continue
        pairs.append({"prompt": e["prompt"], "chosen": ref, "rejected": mine,
                      "chosen_evidence": {"evidence_kind": "reference", "evidence_uri": e.get("uri", ""),
                                          "rationale": "the dataset's own reply"},
                      "rejected_evidence": {"evidence_kind": "policy_sample", "evidence_uri": a.samples,
                                            "rationale": f"model sample, stop={s.get('stop', '')}"}})
    write_jsonl(a.out, pairs)
    return {"pairs": len(pairs), "identical_to_reference": same, "over_T": over, "out": a.out}


def cmd_mix(a):
    """Shuffle several SFT example files into one with a fixed seed. mtstudio
    holds out the file's tail as validation, so the shuffle also makes the
    held-out set a sample of every source; each example records its source file."""
    import random
    rows = []
    for path in a.inputs:
        for ex in read_jsonl(path):
            ex.setdefault("mix_source", os.path.basename(path))
            rows.append(ex)
    random.Random(a.seed).shuffle(rows)
    write_jsonl(a.out, rows)
    by = {}
    for r in rows:
        by[r["mix_source"]] = by.get(r["mix_source"], 0) + 1
    return {"examples": len(rows), "by_source": by, "seed": a.seed, "out": a.out}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    mx = sub.add_parser("mix")
    mx.add_argument("inputs", nargs="+")
    mx.add_argument("--seed", type=int, default=1)
    mx.add_argument("--out", required=True)
    s = sub.add_parser("sft")
    s.add_argument("dialogues")
    s.add_argument("--vocab", required=True)
    s.add_argument("--T", type=int, default=256)
    s.add_argument("--out", required=True)
    s.add_argument("--max", type=int, default=0, help="stop after this many examples (0 = all)")
    s.add_argument("--skip", type=int, default=0, help="skip the first N examples that would be kept")
    p = sub.add_parser("prompts")
    p.add_argument("examples")
    p.add_argument("--out", required=True)
    q = sub.add_parser("pairs")
    q.add_argument("examples")
    q.add_argument("samples")
    q.add_argument("--vocab", required=True)
    q.add_argument("--T", type=int, default=256)
    q.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    print(json.dumps({"sft": cmd_sft, "prompts": cmd_prompts, "pairs": cmd_pairs, "mix": cmd_mix}[a.cmd](a)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
