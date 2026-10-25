#!/usr/bin/env python3
"""Lever 1's SFT set as conversation rows (LOCAL_LLM_STACK.md section 3, points 3a-3b).

    python tools/lever1_conversations.py LEVER1.jsonl --out LEVER1_CONV.jsonl

lever1_sft.py writes the per-turn rendered form: one {"prompt", "completion"} row per reply,
the prompt holding the dialogue's history, rows over T + 1 tokens dropped, all rows shuffled
together (chat_data.py mix). A dialogue's rows (one "uri") are therefore the prefixes of one
conversation, and the kept ones are its first k replies, because a reply's row is longer than
every earlier reply's row. This tool keeps, per uri, the row with the most replies, parses it
back into a conversation (chat_template.parse_row, which refuses a row that does not
round-trip), checks that every other row of the uri is a prefix of it, and writes one
{"messages"} row per dialogue in the order of the dialogue's first row in the input (the
seeded mix order). Under train.sft_loss {"turns": "all"}, mtstudio keeps the longest prefix
of whole exchanges that fits T + 1 and trains every reply in it: the same k replies the
per-turn rows train, now in one row per dialogue.

Writes OUT and OUT.ledger.json (source and output sha256, counts: dialogues, replies, rows
read, rows per dialogue). Refuses an existing OUT.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chat_template as CT  # noqa: E402


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("rows", help="lever1_sft.py's per-turn {prompt, completion} file")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        sys.exit(f"{a.out} exists; this tool writes a new file")
    order, best, extra = [], {}, {}  # uri -> (replies, row) ; uri -> other fields
    rows = 0
    with open(a.rows, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows += 1
            r = json.loads(line)
            uri = r.get("uri")
            if not uri:
                sys.exit(f"row {rows}: no uri; dialogues cannot be grouped")
            k = r["prompt"].count("<|assistant|>")  # this row's reply is the k-th
            if uri not in best:
                order.append(uri)
                extra[uri] = {key: r[key] for key in ("mix_source",) if key in r}
                best[uri] = []
            best[uri].append((k, r["prompt"], r["completion"]))
    per = {}
    tmp = a.out + ".tmp"
    replies = 0
    with open(tmp, "w", encoding="utf-8", newline="\n") as g:
        for uri in order:
            turns = sorted(best[uri])
            ks = [k for k, _, _ in turns]
            if ks != list(range(1, len(ks) + 1)):
                sys.exit(f"{uri}: replies {ks} are not the first {len(ks)} of the dialogue")
            k, prompt, completion = turns[-1]
            conv, policy, form = CT.parse_row({"prompt": prompt, "completion": completion})
            if form != "prompt_completion" or policy != "final":
                sys.exit(f"{uri}: unexpected form {form}/{policy}")
            msgs = conv["messages"]
            full = prompt + completion + "<|endoftext|>"
            for j, p, c in turns[:-1]:
                if not full.startswith(p + c + "<|endoftext|>"):
                    sys.exit(f"{uri}: reply {j}'s row is not a prefix of reply {k}'s")
            if sum(m["role"] == "assistant" for m in msgs) != k:
                sys.exit(f"{uri}: parsed {sum(m['role'] == 'assistant' for m in msgs)} replies, expected {k}")
            g.write(json.dumps({"messages": msgs, "uri": uri, **extra[uri]}, ensure_ascii=False) + "\n")
            replies += k
            per[k] = per.get(k, 0) + 1
    if replies != rows:
        sys.exit(f"{replies} replies written for {rows} rows read")
    os.replace(tmp, a.out)
    ledger = {"tool": "tools/lever1_conversations.py", "schema": CT.SCHEMA, "template": CT.V1,
              "sft_loss": {"turns": "all"}, "source": os.path.abspath(a.rows), "source_sha256": sha256_file(a.rows),
              "out": os.path.abspath(a.out), "out_sha256": sha256_file(a.out), "rows_read": rows,
              "dialogues": len(order), "replies": replies,
              "dialogues_by_replies": {str(k): per[k] for k in sorted(per)}}
    with open(a.out + ".ledger.json", "w", encoding="utf-8", newline="\n") as g:
        json.dump(ledger, g, indent=1)
    print(json.dumps({k: ledger[k] for k in ("rows_read", "dialogues", "replies", "out_sha256")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
