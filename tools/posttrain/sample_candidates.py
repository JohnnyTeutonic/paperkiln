#!/usr/bin/env python3
"""Candidate replies for the judged preference stage (LOCAL_LLM_STACK.md section 7), sampled from
an ember.cpp chat server and written as the requests tools/posttrain/chat_judge.py ranks.

    python tools/posttrain/sample_candidates.py --url http://127.0.0.1:8095 --vocab MODEL.gguf \\
        --out REQUESTS.jsonl [--prompts N] [--k 4] [--temps 0.8,0.9,1.0,0.9] [--seed 31] \\
        [--max-new 64] [--repeat-penalty P] [--ember-commit SHA] [--model-sha256 HEX]

Prompts: the first user turn of each dialogue in smol-smoltalk's test split
(corpus/data/smol_smoltalk_qa_v1/dialogues_test.jsonl, the split lever 1's SFT never reads,
by smol_smoltalk.py's sha256-of-the-first-user-turn rule), in file order, minus any that
corpus/decontam.py drops against the frozen eval sets. Each is the one-turn chat template
<|user|>U<|assistant|>.

Candidates: k samples per prompt; candidate j uses temperature temps[j % len(temps)] (pure
temperature: top_k 0, top_p 1) and seed seed + k * prompt_number + j, all recorded. Identical
replies are kept once; a prompt left with fewer than two distinct replies yields no request
and is counted. ember's prompt ids must equal tools/bpe.py's encoding of the prompt, or the
run stops (chat_sample_ember.py's check).

Resumable: an existing OUT is read first, and its prompts (by uri) are not sampled again.
Output lines: {"prompt", "candidates", "uri", "sampler": {...}}; chat_judge.py reads the first
three.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)
ROOT = os.path.dirname(TOOLS)
sys.path.insert(0, TOOLS)
sys.path.insert(0, os.path.join(ROOT, "corpus"))
from chat_sample_ember import post  # noqa: E402

TEST_SPLIT = os.path.join(ROOT, "corpus", "data", "smol_smoltalk_qa_v1", "dialogues_test.jsonl")


def prompts(path=TEST_SPLIT, dc=None):
    """[(uri, user text)] in file order, decontaminated; and the number dropped."""
    import decontam
    dc = dc or decontam.Decontaminator()
    out, dropped = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            user = row["messages"][0]["content"]
            if dc.check([user], user)[0]:
                dropped += 1
                continue
            out.append((row["uri"], user))
    return out, dropped


def distinct(replies):
    seen, out = set(), []
    for r in replies:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def sample(items, send, k, temps, seed, max_new, repeat_penalty, encode, ctx, done=()):
    """Yield (request or None, reason) per prompt not in done. send(body) -> /chat response;
    encode(text) -> token ids (tools/bpe.py)."""
    for n, (uri, user) in enumerate(items):
        if uri in done:
            continue
        prompt = f"<|user|>{user}<|assistant|>"
        ids = encode(prompt)
        if len(ids) + max_new > ctx:
            yield None, "over_context"
            continue
        replies, runs = [], []
        for j in range(k):
            t, s = temps[j % len(temps)], seed + k * n + j
            body = {"user_input": user, "history": [], "temperature": t, "top_k": 0, "top_p": 1.0, "seed": s,
                    "max_new_tokens": max_new, "return_ids": True}
            if repeat_penalty != 1.0:
                body["repeat_penalty"] = repeat_penalty
            r = send(body)
            if r["prompt_ids"] != ids:
                raise SystemExit(f"{uri}: ember's prompt ids differ from tools/bpe.py's encoding of the prompt")
            replies.append(r["reply"])
            runs.append({"temperature": t, "seed": s, "stop": r["stop_reason"], "tokens": r["tokens"]})
        cands = distinct(replies)
        if len(cands) < 2:
            yield None, "fewer_than_two_distinct"
            continue
        yield {"prompt": prompt, "candidates": cands, "uri": uri, "sampler": {"runs": runs}}, "ok"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", required=True)
    ap.add_argument("--vocab", required=True, help="the served model's GGUF (its tokenizer)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", type=int, default=3000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temps", default="0.8,0.9,1.0,0.9")
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--repeat-penalty", type=float, default=1.0)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--ember-commit", default="")
    ap.add_argument("--model-sha256", default="")
    ap.add_argument("--test-split", default=TEST_SPLIT)
    a = ap.parse_args(argv)
    if not 2 <= a.k <= 8:
        raise SystemExit("--k must be 2-8 (chat_judge.py ranks 2-8 candidates)")
    import bpe
    tok = bpe.read_gguf(a.vocab)
    items, dropped = prompts(a.test_split)
    items = items[:a.prompts]
    done = set()
    if os.path.exists(a.out):
        with open(a.out, encoding="utf-8") as f:
            done = {json.loads(line)["uri"] for line in f if line.strip()}
    common = {"engine": "ember.cpp", "commit": a.ember_commit, "model_sha256": a.model_sha256, "url": a.url,
              "top_k": 0, "top_p": 1.0, "max_new_tokens": a.max_new, "repeat_penalty": a.repeat_penalty}
    counts = {"ok": 0, "over_context": 0, "fewer_than_two_distinct": 0, "already_done": len(done & {u for u, _ in items}),
              "decontaminated": dropped}
    with open(a.out, "a", encoding="utf-8", newline="\n") as g:
        for req, why in sample(items, lambda b: post(a.url, b), a.k, [float(t) for t in a.temps.split(",")], a.seed,
                               a.max_new, a.repeat_penalty, tok.encode, a.ctx, done):
            counts[why] += 1
            if req:
                req["sampler"].update(common)
                g.write(json.dumps(req, ensure_ascii=False) + "\n")
                g.flush()
    print(json.dumps({"prompts": len(items), **counts, "out": a.out}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
