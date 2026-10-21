#!/usr/bin/env python3
"""Candidate replies for the judged preference stage (LOCAL_LLM_STACK.md section 7), sampled from
an ember.cpp chat server and written as the requests tools/posttrain/chat_judge.py ranks.

    python tools/posttrain/sample_candidates.py --url http://127.0.0.1:8095 --vocab MODEL.gguf \\
        --out REQUESTS.jsonl [--prompts N] [--k 4] [--temps 0.8,0.9,1.0,0.9] [--seed 31] \\
        [--max-new 64] [--repeat-penalty P] [--ember-commit SHA] [--model-sha256 HEX]

Prompts: the first user turn of each dialogue in smol-smoltalk's test split
(corpus/data/smol_smoltalk_qa_v1/dialogues_test.jsonl, the split lever 1's SFT never reads,
by smol_smoltalk.py's sha256-of-the-first-user-turn rule), in file order, minus
- any that corpus/decontam.py drops against the frozen eval sets;
- any whose normalised text equals a train-split first turn (the split hashes the raw text,
  so "3x^2 sin(x)." and "3x^2 * sin(x)?" can land in different splits);
- any containing a chat-template marker.
Each is the one-turn chat template <|user|>U<|assistant|>.

Candidates: k samples per prompt; run j uses temperature temps[j % len(temps)] (pure
temperature: top_k 0, top_p 1) and seed seed + k * prompt_number + j, all recorded. Identical
replies are kept once, with the runs each came from; a prompt left with fewer than two distinct
replies yields no request. ember's prompt ids must equal tools/bpe.py's encoding of the prompt,
or the run stops (chat_sample_ember.py's check).

Resumable: OUT and OUT.skipped.jsonl are read first, and their prompts (by uri) are not sampled
again. A resume refuses unless the recorded configuration (model, vocabulary, ember commit, k,
temperatures, seed, max-new, penalty, context) matches, so one file never mixes two samplers. A
torn last line (a crash mid-write) is cut off. Output lines: {"prompt", "candidates", "uri",
"sampler": {...}}; chat_judge.py reads the first three.
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

SPLIT_DIR = os.path.join(ROOT, "corpus", "data", "smol_smoltalk_qa_v1")
TEST_SPLIT = os.path.join(SPLIT_DIR, "dialogues_test.jsonl")
TRAIN_SPLIT = os.path.join(SPLIT_DIR, "dialogues_train.jsonl")
MARKERS = ("<|user|>", "<|assistant|>", "<|endoftext|>")


def first_turns(path):
    """Normalised first user turns of a split (decontam.words joined)."""
    import decontam
    out = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.add(" ".join(decontam.words(json.loads(line)["messages"][0]["content"])))
    return out


def prompts(path=TEST_SPLIT, dc=None, train_turns=None):
    """[(uri, user text)] in file order after the three filters, and the counts dropped."""
    import decontam
    dc = dc or decontam.Decontaminator()
    out, dropped = [], {"decontaminated": 0, "train_overlap": 0, "markers": 0}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            user = row["messages"][0]["content"]
            if any(m in user for m in MARKERS):
                dropped["markers"] += 1
            elif train_turns is not None and " ".join(decontam.words(user)) in train_turns:
                dropped["train_overlap"] += 1
            elif dc.check([user], user)[0]:
                dropped["decontaminated"] += 1
            else:
                out.append((row["uri"], user))
    return out, dropped


def distinct(replies):
    """(distinct replies in first-seen order, [[run indices] per distinct reply])."""
    where = {}
    for j, r in enumerate(replies):
        where.setdefault(r, []).append(j)
    return list(where), list(where.values())


def sample(items, send, k, temps, seed, max_new, repeat_penalty, encode, ctx, done=()):
    """Yield (record, reason) per prompt not in done: a request ("ok") or a skip record
    ({"uri", "reason"}). send(body) -> /chat response; encode(text) -> token ids (tools/bpe.py)."""
    for n, (uri, user) in enumerate(items):
        if uri in done:
            continue
        prompt = f"<|user|>{user}<|assistant|>"
        ids = encode(prompt)
        if len(ids) + max_new > ctx:
            yield {"uri": uri, "reason": "over_context"}, "over_context"
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
        cands, sources = distinct(replies)
        if len(cands) < 2:
            yield {"uri": uri, "reason": "fewer_than_two_distinct"}, "fewer_than_two_distinct"
            continue
        yield {"prompt": prompt, "candidates": cands, "uri": uri,
               "sampler": {"runs": runs, "candidate_runs": sources}}, "ok"


def read_resumable(path):
    """Rows of a jsonl that may end in a torn line (a crash mid-write); the torn line is cut off."""
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        data = f.read()
    keep = data.rfind(b"\n") + 1
    if keep < len(data):
        with open(path, "r+b") as f:
            f.truncate(keep)
    return [json.loads(line) for line in data[:keep].decode("utf-8").splitlines() if line.strip()]


def check_config(rows, config, path):
    for r in rows:
        got = r.get("sampler", r).get("config")
        if got != config:
            raise SystemExit(f"{path} was written with a different sampler configuration ({got} vs {config}); "
                             "use a new --out")


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
    ap.add_argument("--train-split", default=TRAIN_SPLIT)
    a = ap.parse_args(argv)
    if not 2 <= a.k <= 8:
        raise SystemExit("--k must be 2-8 (chat_judge.py ranks 2-8 candidates)")
    import bpe
    import judge_sidecar
    tok = bpe.read_gguf(a.vocab)
    temps = [float(t) for t in a.temps.split(",")]
    config = {"model_sha256": a.model_sha256, "vocab_sha256": judge_sidecar.sha256_bytes(open(a.vocab, "rb").read()),
              "ember_commit": a.ember_commit, "k": a.k, "temps": temps, "seed": a.seed, "max_new_tokens": a.max_new,
              "repeat_penalty": a.repeat_penalty, "ctx": a.ctx}
    skipped_path = a.out + ".skipped.jsonl"
    rows, skips = read_resumable(a.out), read_resumable(skipped_path)
    check_config(rows, config, a.out)
    check_config(skips, config, skipped_path)
    done = {r["uri"] for r in rows} | {r["uri"] for r in skips}
    items, dropped = prompts(a.test_split, train_turns=first_turns(a.train_split))
    items = items[:a.prompts]
    common = {"engine": "ember.cpp", "url": a.url, "top_k": 0, "top_p": 1.0, "config": config}
    counts = {"ok": 0, "over_context": 0, "fewer_than_two_distinct": 0,
              "already_done": len(done & {u for u, _ in items}), **dropped}
    with open(a.out, "a", encoding="utf-8", newline="\n") as g, \
         open(skipped_path, "a", encoding="utf-8", newline="\n") as sk:
        for rec, why in sample(items, lambda b: post(a.url, b), a.k, temps, a.seed, a.max_new, a.repeat_penalty,
                               tok.encode, a.ctx, done):
            counts[why] += 1
            if why == "ok":
                rec["sampler"].update(common)
                g.write(json.dumps(rec, ensure_ascii=False) + "\n")
                g.flush()
            else:
                sk.write(json.dumps({**rec, "config": config}, ensure_ascii=False) + "\n")
                sk.flush()
    print(json.dumps({"prompts": len(items), **counts, "out": a.out}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
