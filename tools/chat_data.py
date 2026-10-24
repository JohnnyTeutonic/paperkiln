#!/usr/bin/env python3
"""Dialogues -> SFT examples and preference pairs in the chat template that
`mtstudio chat` serves.

    python tools/chat_data.py sft DIALOGUES.jsonl --vocab V.gguf --T 256 --out SFT.jsonl [--max N] [--skip N] [--all-turns]
    python tools/chat_data.py prompts SFT.jsonl --out PROMPTS.jsonl
    python tools/chat_data.py pairs SFT.jsonl SAMPLES.jsonl --vocab V.gguf --T 256 --out PAIRS.jsonl
    python tools/chat_data.py mix A.jsonl B.jsonl [...] --seed 1 --out MIX.jsonl

The template (tools/chat_template.py, paperkiln/1; mtstudio renders the same through
include/microtorch/chat_template.hpp): every past turn is <|user|>U<|assistant|>A<|endoftext|>,
and a reply is prompted by <|user|>U<|assistant|>. Each assistant turn k becomes one example:
  prompt     = turns 1..k-1 in full, then <|user|>U_k<|assistant|>
  completion = A_k   (mtstudio appends <|endoftext|> and masks the prompt)
An example whose BPE length (prompt + completion + end-of-text) exceeds
T + 1 is dropped and counted, never truncated: a cut reply would teach the
model to stop mid-sentence. A dialogue whose text still carries a source
role marker ([INST], [/INST] or a variant) is dropped whole and counted:
about 0.3% of TinyChat conversations separate turns with bare [/INST], so
one "assistant turn" holds several (ledger 0003), and training on them
would teach the model to print markers. A dialogue whose content holds the template's own
markers (<|user|>, <|assistant|>, <|endoftext|>) is dropped and counted the same way. --skip N starts after the first N
kept examples, so a later stage can draw examples the SFT run never saw.
--all-turns writes one example per dialogue instead, in mtstudio's
"segments" form ({"segments": [{"text", "loss"}...]}: user turns with
loss false, assistant turns with loss true), the longest prefix of whole
turns that fits T + 1; the same targets as the per-turn form, one row.

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
import chat_template as CT  # noqa: E402

MARKER = re.compile(r"\[\s*/?\W?\s*INST|INST\s*\]")


def _pairs(messages):
    """(user, assistant) contents by position, as the template's exchanges."""
    return [(messages[k]["content"], messages[k + 1]["content"]) for k in range(0, len(messages) - 1, 2)]


def turn(u, a):
    return CT.render(CT.from_pairs([(u, a)]))


def examples(messages):
    pairs = _pairs(messages)
    for k, (u, a) in enumerate(pairs):
        yield {"prompt": CT.prompt(pairs[:k], u), "completion": a}


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def write_jsonl(p, rows):
    with open(p, "w", encoding="utf-8", newline="\n") as g:
        for r in rows:
            g.write(json.dumps(r, ensure_ascii=False) + "\n")


_TOK = None


class _Llama3:
    """tools/llama3_bpe's tokenizer behind bpe.BPE's encode(text) -> ids."""

    def __init__(self, vocab):
        import llama3_bpe
        self.tok = llama3_bpe.from_gguf(vocab)

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False).ids


def _init_tok(vocab):
    """The vocabulary's own encoder: paperkiln "bpe" (tools/bpe.py) or the
    llama.cpp form "gpt2" / "llama-bpe" (tools/llama3_bpe.py). Both split at
    the special tokens first, which _sized relies on."""
    global _TOK
    import llama3_bpe
    model = llama3_bpe.read_gguf_kv(vocab, {"tokenizer.ggml.model"}).get("tokenizer.ggml.model")
    _TOK = _Llama3(vocab) if model == "gpt2" else bpe.read_gguf(vocab)


def _sized(line):
    """One dialogue's examples with their token counts, or None if it carries
    an [INST] marker. Each message is encoded once: the special tokens are
    pre-tokeniser chunk boundaries, so a prompt's length is the sum of its
    pieces' lengths, and re-encoding the growing history for every turn (the
    old cost, quadratic in turns) gives the same numbers."""
    d = json.loads(line)
    if any(MARKER.search(m["content"]) for m in d["messages"]):
        return d, None
    ms = d["messages"]
    out, hist = [], 0
    try:
        exs = list(examples(ms))
    except CT.TemplateError:  # content holding the template's own markers: dropped with the marked ones
        return d, None
    for k, ex in enumerate(exs):
        nu = len(_TOK.encode(ms[2 * k]["content"]))
        na = len(_TOK.encode(ms[2 * k + 1]["content"]))
        out.append((ex, hist + 2 + nu + na + 1))  # <|user|> u <|assistant|> + a + <|endoftext|>
        hist += 3 + nu + na  # turn(u, a)
    return d, out


def all_turns(messages, sized, T):
    """One whole-dialogue example (mtstudio's "segments" form: every assistant
    turn trained in one row) from the longest prefix of whole turns that fits
    T + 1 tokens, with how many turns it keeps; None when even the first turn
    does not fit. A turn is never cut, for the reason above."""
    k = 0
    while k < len(sized) and sized[k][1] <= T + 1:
        k += 1
    if k == 0:
        return None, 0
    return CT.to_segments(CT.from_pairs(_pairs(messages)[:k])), k


def cmd_sft(a):
    if a.all_turns:
        return cmd_sft_all_turns(a)
    kept = dropped = marked = skipped = 0
    with open(a.dialogues, encoding="utf-8") as f, open(a.out, "w", encoding="utf-8", newline="\n") as g:
        if a.jobs > 1:
            import multiprocessing as mp
            pool = mp.Pool(a.jobs, initializer=_init_tok, initargs=(a.vocab,))
            sized = pool.imap(_sized, f, chunksize=256)  # imap keeps the file's order
        else:
            _init_tok(a.vocab)
            pool, sized = None, map(_sized, f)
        for d, exs in sized:
            if exs is None:
                marked += 1
                continue
            for ex, n in exs:
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
        if pool is not None:
            pool.terminate()
    return {"kept": kept, "skipped": skipped, "dropped_over_T": dropped, "dropped_dialogues_with_markers": marked,
            "out": a.out}


def cmd_sft_all_turns(a):
    """`sft --all-turns`: one example per dialogue (its turns in one row), so a
    k-turn dialogue costs one row instead of k. Same marker filter and token
    counts as the per-turn form; turns past T are dropped from the end."""
    kept = dropped = marked = skipped = turns_kept = turns_cut = 0
    _init_tok(a.vocab)
    with open(a.dialogues, encoding="utf-8") as f, open(a.out, "w", encoding="utf-8", newline="\n") as g:
        for d, exs in map(_sized, f):
            if exs is None:
                marked += 1
                continue
            ex, k = all_turns(d["messages"], exs, a.T)
            if ex is None:
                dropped += 1
                continue
            if skipped < a.skip:
                skipped += 1
                continue
            ex["uri"] = d.get("uri", "")
            g.write(json.dumps(ex, ensure_ascii=False) + "\n")
            kept += 1
            turns_kept += k
            turns_cut += len(exs) - k
            if a.max and kept >= a.max:
                break
    return {"kept": kept, "skipped": skipped, "turns_kept": turns_kept, "turns_cut_over_T": turns_cut,
            "dropped_over_T": dropped, "dropped_dialogues_with_markers": marked, "out": a.out}


def cmd_prompts(a):
    ex = read_jsonl(a.examples)
    write_jsonl(a.out, ({"prompt": e["prompt"], "uri": e.get("uri", "")} for e in ex))
    return {"prompts": len(ex), "out": a.out}


def cmd_pairs(a):
    _init_tok(a.vocab)
    tok = _TOK
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
    s.add_argument("--jobs", type=int, default=1, help="tokenising processes (output is identical for any value)")
    s.add_argument("--skip", type=int, default=0, help="skip the first N examples that would be kept")
    s.add_argument("--all-turns", action="store_true",
                   help="one example per dialogue with every assistant turn trained (mtstudio segments)")
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
