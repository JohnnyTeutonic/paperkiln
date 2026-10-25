#!/usr/bin/env python3
"""Sample replies to SFT-format examples from an ember.cpp chat server.

    python tools/chat_sample_ember.py EXAMPLES.jsonl --url http://127.0.0.1:8091 --vocab V.gguf \\
        --out SAMPLES.jsonl [--temp 0.7] [--max-new 64] [--seed 17] [--ember-commit SHA]

Why: mtstudio's sampler has no KV cache (about 10 s per reply for the chat
model on a laptop GPU); ember serves the same GGUF with greedy parity to
mtstudio (tinyllama.cpp experiments/chat_2026-10-18, 40/40) in milliseconds.

Each example's templated prompt (tools/chat_data.py) is split back into
history + user_input for ember's /chat. Sampling is pure temperature
(top_k 0, top_p 1.0: ember draws from the whole softmax), one seed per
example (seed + index). The request returns ember's own token ids
(return_ids), and the prompt ids must equal tools/bpe.py's encoding of the
prompt, or the run stops: a mismatch would mean ember saw a different
prompt from the one mtstudio trains on.

Output records match `mtstudio sample-set`'s fields used downstream
(prompt_index, completion, completion_ids, stop) plus the sampler receipt.
ember's stop rule: end-of-text, or a generated <|user|> / <|assistant|>
(stop "turn"); the marker is not part of the completion.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpe  # noqa: E402

TURN = re.compile(r"<\|user\|>(.*?)<\|assistant\|>(.*?)<\|endoftext\|>", re.S)


def split_prompt(prompt):
    """'<|user|>U1<|assistant|>A1<|endoftext|>...<|user|>Un<|assistant|>' -> (history, Un)."""
    if not prompt.startswith("<|user|>") or not prompt.endswith("<|assistant|>"):
        raise ValueError("prompt is not in the chat template")
    head, _, last = prompt[:-len("<|assistant|>")].rpartition("<|user|>")
    hist = [{"user": u, "assistant": a} for u, a in TURN.findall(head)]
    if "".join(f"<|user|>{t['user']}<|assistant|>{t['assistant']}<|endoftext|>" for t in hist) != head:
        raise ValueError("prompt history does not parse back exactly")
    return hist, last


def post(url, body, timeout=120):
    req = urllib.request.Request(url + "/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("examples")
    ap.add_argument("--url", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--greedy", action="store_true", help="top_k 1 (argmax) instead of pure temperature")
    ap.add_argument("--ember-commit", default="")
    ap.add_argument("--ctx", type=int, default=256, help="model context; longer prompt + max-new is skipped")
    ap.add_argument("--repeat-penalty", type=float, default=1.0, help="contract repetition penalty (1.0 = off)")
    ap.add_argument("--repeat-last-n", type=int, default=64)
    a = ap.parse_args(argv)
    tok = bpe.load_vocab(a.vocab)
    stops, skipped = {}, 0
    with open(a.examples, encoding="utf-8") as f, open(a.out, "w", encoding="utf-8", newline="\n") as g:
        for i, line in enumerate(f):
            e = json.loads(line)
            # ember trims the oldest turns when prompt + max-new overflows the
            # context; such a sample would answer a different prompt.
            if len(tok.encode(e["prompt"])) + a.max_new > a.ctx:
                skipped += 1
                continue
            hist, user = split_prompt(e["prompt"])
            top_k = 1 if a.greedy else 0
            body = {"user_input": user, "history": hist, "temperature": a.temp, "top_k": top_k, "top_p": 1.0,
                    "seed": a.seed + i, "max_new_tokens": a.max_new, "return_ids": True}
            if a.repeat_penalty != 1.0:  # LOCAL_LLM_STACK.md section 3 point 4
                body.update(repeat_penalty=a.repeat_penalty, repeat_last_n=a.repeat_last_n)
            r = post(a.url, body)
            if r["prompt_ids"] != tok.encode(e["prompt"]):
                sys.exit(f"example {i}: ember's prompt ids differ from tools/bpe.py's encoding of the prompt")
            stop = {"eos": "eos", "turn": "turn", "length": "length"}[r["stop_reason"]]
            stops[stop] = stops.get(stop, 0) + 1
            g.write(json.dumps({"prompt_index": i, "completion": tok.decode(r["ids"]), "completion_ids": r["ids"],
                                "stop": stop, "stop_id": r.get("stop_id"),
                                "sampler": {"engine": "ember.cpp", "commit": a.ember_commit, "url": a.url,
                                            "temperature": a.temp, "top_k": top_k, "top_p": 1.0, "seed": a.seed + i,
                                            "max_new_tokens": a.max_new}}, ensure_ascii=False) + "\n")
    print(json.dumps({"samples": sum(stops.values()), "stops": stops, "skipped_over_ctx": skipped, "out": a.out}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
