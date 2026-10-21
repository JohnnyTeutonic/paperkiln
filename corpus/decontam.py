#!/usr/bin/env python3
"""Decontamination against the frozen eval sets, for training corpora that could contain chat or
question answering (lever 2's corpus/fetchers/dialogue_pt.py; lever 1's SFT builder).

Targets: corpus/eval/simple_qa_v1 (prompts; answers flag only), tinychat_hf_prompts_v1 and
chat_prompts_v1 (prompts and reference replies). Text is normalised to words: lowercased,
every run of non-alphanumeric characters a space. A candidate is DROPPED when

- ngram:  it shares a word 13-gram with an eval text of 13 words or more;
- prompt: one of its user turns equals an eval prompt of 4 words or more (after
  normalisation; "Hi" or "Hey!" alone is everyday text, not a leak);
- span:   it contains an eval prompt of 8-12 words as a contiguous word sequence.

Short fact answers slip past n-grams ("the capital of france is paris"), so the answers of
simple-QA's knowledge items (kinds fact and science; arithmetic and greeting answers such as
"two" or "hello" are everywhere) are FLAGGED, not dropped:

- answer: an answer appears as whole words, and within 30 words of it at least two of its
  question's content words (all of them when the question has fewer than two).

    python corpus/decontam.py IN.jsonl OUT.jsonl REPORT.json [--field messages|text]

IN rows hold {"messages": [{"role", "content"}...]} (the default) or {"text": ...}; OUT keeps
the rows that pass, unchanged. REPORT lists every drop and flag with the eval item it matched.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

CORPUS_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.join(CORPUS_DIR, "eval")
SIMPLE_QA = os.path.join(EVAL_DIR, "simple_qa_v1.jsonl")
CHAT_SETS = [os.path.join(EVAL_DIR, f) for f in ("tinychat_hf_prompts_v1.jsonl", "chat_prompts_v1.jsonl")]
N = 13            # drop: shared word n-gram length
SPAN_MIN = 8      # drop: an eval prompt this long or longer, anywhere, as a word sequence
PROMPT_MIN = 4    # drop: a user turn equal to an eval prompt this long or longer
ANSWER_WINDOW = 30
ANSWER_KINDS = {"fact", "science"}
STOP = set("a an the of in on at to for from by with and or but is are was were be been being do does did "
           "what which who whom whose when where why how this that these those it its it's i you he she we they "
           "me my your our their his her them us there here as if than then so not no yes can could would should "
           "will shall may might must have has had about into over under".split())
_SPLIT = re.compile(r"[^0-9a-z]+")


def words(text):
    return [w for w in _SPLIT.split(text.lower()) if w]


def _rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class Decontaminator:
    def __init__(self, simple_qa=SIMPLE_QA, chat_sets=None):
        # Every drop pattern is 8+ words, so its first three words index it; a row's
        # three-word windows meet that index by set intersection (in C), and only the
        # rare hits are checked word by word.
        self.patterns = defaultdict(list)   # (w0, w1, w2) -> [(word tuple, rule, eval id)]
        self.prompts = {}                   # normalised prompt -> eval id
        self.answers = defaultdict(list)    # first answer word -> [(answer words, content words, eval id)]
        texts = []                          # (eval id, text, is_prompt)
        name = os.path.splitext(os.path.basename(simple_qa))[0]
        for i, r in enumerate(_rows(simple_qa)):
            eid = f"{name}:{i + 1}"
            texts.append((eid, r["prompt"], True))
            content = [w for w in words(r["prompt"]) if w not in STOP]
            for a in r.get("answers", []) if r.get("kind", "fact") in ANSWER_KINDS else []:
                aw = tuple(words(a))
                if aw and content:
                    self.answers[aw[0]].append((aw, content, eid))
        for path in CHAT_SETS if chat_sets is None else chat_sets:
            name = os.path.splitext(os.path.basename(path))[0]
            for i, r in enumerate(_rows(path)):
                eid = f"{name}:{i + 1}"
                texts.append((eid, r["prompt"], True))
                if r.get("reference"):
                    texts.append((eid + ":reference", r["reference"], False))
        for eid, text, is_prompt in texts:
            w = words(text)
            if is_prompt and len(w) >= PROMPT_MIN:
                self.prompts.setdefault(" ".join(w), eid)
            for i in range(len(w) - N + 1):
                self.patterns[tuple(w[i:i + 3])].append((tuple(w[i:i + N]), "ngram", eid))
            if is_prompt and SPAN_MIN <= len(w) < N:
                self.patterns[tuple(w[:3])].append((tuple(w), "span", eid))
        self.keys = set(self.patterns)
        self.answer_words = set(self.answers)

    def check(self, user_turns, all_text):
        """(drops, flags): lists of {rule, eval, match}. user_turns: the user messages;
        all_text: everything the row would contribute to training."""
        drops, flags = [], []
        for u in user_turns:
            eid = self.prompts.get(" ".join(words(u)))
            if eid:
                drops.append({"rule": "prompt", "eval": eid, "match": u[:200]})
        w = words(all_text)
        seen = set()
        tri = list(zip(w, w[1:], w[2:]))
        hit = self.keys.intersection(tri)
        if hit:
            for i, k in enumerate(tri):
                if k not in hit:
                    continue
                for pat, rule, eid in self.patterns[k]:
                    if (rule, eid) not in seen and tuple(w[i:i + len(pat)]) == pat:
                        seen.add((rule, eid))
                        drops.append({"rule": rule, "eval": eid, "match": " ".join(pat)})
        ans = self.answer_words.intersection(w)
        if ans:
            for i, first in enumerate(w):
                if first not in ans:
                    continue
                for aw, content, eid in self.answers[first]:
                    if ("answer", eid) in seen or tuple(w[i:i + len(aw)]) != aw:
                        continue
                    near = set(w[max(0, i - ANSWER_WINDOW):i + len(aw) + ANSWER_WINDOW])
                    if sum(c in near for c in content) >= min(2, len(content)):
                        seen.add(("answer", eid))
                        flags.append({"rule": "answer", "eval": eid,
                                      "match": " ".join(w[max(0, i - 12):i + len(aw) + 12])})
        return drops, flags


def row_text(row, field):
    if field == "messages":
        msgs = row["messages"]
        return [m["content"] for m in msgs if m["role"] == "user"], "\n".join(m["content"] for m in msgs)
    return [], row[field]


def run(inp, out, report, field="messages", dc=None):
    dc = dc or Decontaminator()
    rule_counts, eval_counts, drops, flags = Counter(), Counter(), [], []
    n_in = n_out = 0
    with open(inp, encoding="utf-8") as fi, open(out, "w", encoding="utf-8", newline="\n") as fo:
        for line_no, line in enumerate(fi, 1):
            if not line.strip():
                continue
            n_in += 1
            row = json.loads(line)
            users, text = row_text(row, field)
            d, f = dc.check(users, text)
            ref = {"line": line_no, "uri": row.get("uri")}
            if d:
                drops.append({**ref, "matches": d})
                rule_counts.update({x["rule"] for x in d})
                eval_counts.update({x["eval"].split(":reference")[0] for x in d})
                continue
            for x in f:                      # flags only on rows that are kept
                flags.append({**ref, **x})
            fo.write(line if line.endswith("\n") else line + "\n")
            n_out += 1
    summary = {"input": os.path.basename(inp), "rows_in": n_in, "rows_kept": n_out, "rows_dropped": n_in - n_out,
               "dropped_by_rule": dict(rule_counts), "evals_hit": len(eval_counts),
               "answer_flags": len(flags), "rules": {"ngram": N, "span_min_words": SPAN_MIN,
                                                     "prompt_min_words": PROMPT_MIN,
                                                     "answer_window_words": ANSWER_WINDOW,
                                                     "answer_kinds": sorted(ANSWER_KINDS)}}
    with open(report, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"summary": summary, "dropped": drops, "flagged": flags,
                   "eval_hits": dict(eval_counts.most_common())}, f, ensure_ascii=False, indent=1)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description="Drop rows that overlap the frozen eval sets; flag answer leaks.")
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("report")
    ap.add_argument("--field", default="messages", help="'messages' (chat rows) or a plain-text field name")
    a = ap.parse_args(argv)
    print(json.dumps(run(a.inp, a.out, a.report, a.field)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
