#!/usr/bin/env python3
"""Judged side-by-side of two chat models (experiments/chat_general/PLAN.md, the preference
step's gate): one sampled reply from each model per held-out prompt, both served by ember.cpp,
ranked by tools/posttrain/chat_judge.py in both orders, then counted.

    python tools/posttrain/side_by_side.py sample --url-a URL --url-b URL --name-a A --name-b B \\
        --vocab MODEL.gguf --out SBS.jsonl [--prompts 300] [--temp 0.7] [--seed 71] [--max-new 256] \\
        [--ctx 512] [--split VAL.jsonl] [--ember-commit SHA]
    python tools/posttrain/side_by_side.py count --requests SBS.jsonl --cache C.jsonl --rubric R --judge-id M

Prompts: the first user turns of smol-smoltalk's val split (sample_candidates.prompts' filters:
eval overlap, train-split overlap, markers), in file order; a prompt whose prompt + max_new
exceeds ctx is skipped. Both models answer with the same body (temperature, top_k 0, top_p 1,
seed per prompt). Replies follow sample_candidates.py's U+FFFD rule; a prompt whose two replies
are equal or where one is invalid is skipped and counted. Output lines are chat_judge.py
requests: {"prompt", "candidates": [reply_a, reply_b], "uri", "sampler": {"models": [A, B],
"runs", "candidate_runs", "config"}}.

count: a prompt is won by the model both orders rank first; it is a tie when the orders disagree
(position decided it). Reports wins, ties, B's win share over decided prompts with a Wilson 95%
interval, and per model the share of replies that ended on end-of-text and the median words.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import sample_candidates as S  # noqa: E402

VAL_SPLIT = os.path.join(S.SPLIT_DIR, "dialogues_val.jsonl")


def sample(items, send_a, send_b, temp, seed, max_new, encode, ctx):
    """Yield (record, reason): a request ("ok") or a skip ({"uri", "reason"})."""
    for n, (uri, user) in enumerate(items):
        prompt = f"<|user|>{user}<|assistant|>"
        ids = encode(prompt)
        if len(ids) + max_new > ctx:
            yield {"uri": uri, "reason": "over_context"}, "over_context"
            continue
        body = {"user_input": user, "history": [], "temperature": temp, "top_k": 0, "top_p": 1.0,
                "seed": seed + n, "max_new_tokens": max_new, "return_ids": True}
        texts, runs = [], []
        for send in (send_a, send_b):
            r = send(dict(body))
            if r["prompt_ids"] != ids:
                raise SystemExit(f"{uri}: ember's prompt ids differ from tools/bpe.py's encoding of the prompt")
            text, status = S.utf8_status(r["reply"], r["stop_reason"])
            texts.append(text)
            runs.append({"temperature": temp, "seed": seed + n, "stop": r["stop_reason"], "tokens": r["tokens"],
                         "utf8": status})
        if None in texts:
            yield {"uri": uri, "reason": "invalid_utf8"}, "invalid_utf8"
        elif texts[0] == texts[1]:
            yield {"uri": uri, "reason": "identical"}, "identical"
        else:
            yield {"prompt": prompt, "candidates": texts, "uri": uri,
                   "sampler": {"runs": runs, "candidate_runs": [[0], [1]]}}, "ok"


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def count(requests, cache_path, judge_id, rubric_path):
    import chat_judge as J
    from judge_sidecar import read_jsonl, rubric_identity
    _, version, _ = rubric_identity(rubric_path)
    cache = {c["key"]: c for c in read_jsonl(cache_path)}
    names = None
    wins, ties, unranked = [0, 0], 0, 0
    eos, words = [[], []], [[], []]
    for r in requests:
        names = names or r["sampler"].get("models", ["A", "B"])
        for i in (0, 1):
            eos[i].append(r["sampler"]["runs"][i]["stop"] == "eos")
            words[i].append(len(r["candidates"][i].split()))
        got = [cache.get(J.key_of(r["prompt"], r["candidates"], o, judge_id, version)) for o in ([0, 1], [1, 0])]
        if not all(got):
            unranked += 1
        elif got[0]["ranking"][0] == got[1]["ranking"][0]:
            wins[got[0]["ranking"][0]] += 1
        else:
            ties += 1
    decided = wins[0] + wins[1]
    return {"models": names, "prompts": len(requests), "wins": dict(zip(names, wins)), "ties": ties,
            "unranked": unranked, "b_win_share": round(wins[1] / decided, 4) if decided else None,
            "b_win_share_wilson95": wilson(wins[1], decided),
            "eos_share": {names[i]: round(sum(eos[i]) / max(len(eos[i]), 1), 4) for i in (0, 1)},
            "median_words": {names[i]: statistics.median(words[i]) if words[i] else None for i in (0, 1)}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--url-a", required=True)
    s.add_argument("--url-b", required=True)
    s.add_argument("--name-a", required=True)
    s.add_argument("--name-b", required=True)
    s.add_argument("--vocab", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--prompts", type=int, default=300)
    s.add_argument("--temp", type=float, default=0.7)
    s.add_argument("--seed", type=int, default=71)
    s.add_argument("--max-new", type=int, default=256)
    s.add_argument("--ctx", type=int, default=512)
    s.add_argument("--split", default=VAL_SPLIT)
    s.add_argument("--train-split", default=S.TRAIN_SPLIT)
    s.add_argument("--ember-commit", default="")
    c = sub.add_parser("count")
    c.add_argument("--requests", required=True)
    c.add_argument("--cache", required=True)
    c.add_argument("--rubric", required=True)
    c.add_argument("--judge-id", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "count":
        from judge_sidecar import read_jsonl
        print(json.dumps(count(read_jsonl(a.requests), a.cache, a.judge_id, a.rubric)))
        return 0
    import bpe
    from chat_sample_ember import post
    tok = bpe.load_vocab(a.vocab)
    items, dropped = S.prompts(a.split, train_turns=S.first_turns(a.train_split))
    items = items[:a.prompts]
    config = {"models": [a.name_a, a.name_b], "ember_commit": a.ember_commit, "temp": a.temp, "seed": a.seed,
              "max_new_tokens": a.max_new, "ctx": a.ctx, "utf8_rule": S.UTF8_RULE}
    counts = {"ok": 0, "over_context": 0, "identical": 0, "invalid_utf8": 0, **dropped}
    with open(a.out, "w", encoding="utf-8", newline="\n") as g:
        for rec, why in sample(items, lambda b: post(a.url_a, b), lambda b: post(a.url_b, b), a.temp, a.seed,
                               a.max_new, tok.encode, a.ctx):
            counts[why] += 1
            if why == "ok":
                rec["sampler"].update(models=[a.name_a, a.name_b], config=config)
                g.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps({"prompts": len(items), **counts, "out": a.out}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
