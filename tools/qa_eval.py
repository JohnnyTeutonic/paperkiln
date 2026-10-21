#!/usr/bin/env python3
"""Score chat models on the frozen simple-QA probe (corpus/eval/simple_qa_v1.jsonl).

    python tools/qa_eval.py --ember-bin BIN --model tag=spec.json [--model ...] \\
        --qa corpus/eval/simple_qa_v1.jsonl --out DIR [--port 8091]

Each question is asked as a single user turn in the served chat template and
answered greedily (argmax) by an ember.cpp server on the model's GGUF, so a
score is deterministic. A reply counts as correct when it contains an
accepted answer starting at a word boundary, ignoring case (the probe lists
each accepted form, e.g. "eight" and "8"; a few are stems such as
"strawberr"); a number must also end at a non-digit, so "4" does not match
"40", and a word of three letters or fewer at a word boundary, so "hi" does
not match "high". Reported per kind (fact, science, arith, turn) and overall, with
every reply in qa_replies.md for reading. Containment is generous: a reply
that lists wrong answers alongside the right one counts. It measures whether
the knowledge is reachable, not whether the reply is good.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chat_eval  # noqa: E402


def correct(reply, answers):
    r = reply.lower()
    for a in answers:
        # Numbers end at a non-digit ("4" is not "40"); short words end at a
        # word boundary ("hi" is not "high").
        tail = r"(?![0-9])" if a.isdigit() else (r"(?![a-z0-9])" if len(a) <= 3 else "")
        if re.search(r"(?<![a-z0-9])" + re.escape(a.lower()) + tail, r):
            return True
    return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ember-bin", required=True)
    ap.add_argument("--model", action="append", required=True, help="tag=spec.json")
    ap.add_argument("--qa", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--repeat-penalty", type=float, default=1.0, help="contract repetition penalty (1.0 = off)")
    ap.add_argument("--repeat-last-n", type=int, default=64)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    qs = chat_eval.read_jsonl(a.qa)
    prompts = [{"prompt": q["prompt"], "reference": q["answers"][0]} for q in qs]
    summary, replies = {}, {}
    for m in a.model:
        tag, spec = m.split("=", 1)
        chat_eval.check_rope_heads(spec)
        recs = chat_eval.sample_ember(a.ember_bin, a.port, spec, prompts, 1, 1.0, a.max_new,
                                      os.path.join(a.out, f"{tag}_qa.jsonl"), greedy=True,
                                      repeat=(a.repeat_penalty, a.repeat_last_n))
        got = {r["prompt_index"]: r["completion"] for r in recs}
        by = {}
        for i, q in enumerate(qs):
            ok = correct(got.get(i, ""), q["answers"])
            by.setdefault(q["kind"], []).append(ok)
        summary[tag] = {k: sum(v) / len(v) for k, v in by.items()}
        summary[tag]["all"] = sum(sum(v) for v in by.values()) / len(qs)
        replies[tag] = got
    with open(os.path.join(a.out, "qa_summary.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=1)
    kinds = sorted({q["kind"] for q in qs})
    L = ["# Simple QA probe (greedy, contains an accepted answer)", "",
         "| model | " + " | ".join(kinds) + " | all |", "|---|" + "---:|" * (len(kinds) + 1)]
    for t, s in summary.items():
        L.append(f"| {t} | " + " | ".join(f"{s.get(k, 0):.2f}" for k in kinds) + f" | {s['all']:.2f} |")
    L.append("")
    for i, q in enumerate(qs):
        L.append(f"### {i + 1}. [{q['kind']}] {q['prompt']} (accepted: {', '.join(q['answers'])})")
        for t in summary:
            r = replies[t].get(i, "")
            L.append(f"- **{t}** {'✓' if correct(r, q['answers']) else '✗'}: {r}")
        L.append("")
    open(os.path.join(a.out, "qa_replies.md"), "w", encoding="utf-8", newline="\n").write("\n".join(L))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
