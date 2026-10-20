#!/usr/bin/env python3
"""Dialogue text for pretraining (lever 2, LOCAL_LLM_STACK.md section 7): permissively licensed
conversation corpora, one source per corpus id, rendered in the chat template so conversation
is a native register of the next base model, not only a thin SFT layer.

    python corpus/fetchers/dialogue_pt.py SOURCE|all [--pin]

Sources (pinned revisions; licence from each dataset card):
- soda:      allenai/soda@fdc848ab, train split; CC-BY-4.0 (attribution in corpus/README.md).
             Two-speaker social dialogues; the first speaker renders as the user, the second
             as the assistant; dialogues whose speakers do not alternate are dropped.
- oasst:     OpenAssistant/oasst2@179dd21f plus the oasst1@fdf72ae0 trees absent from oasst2
             (oasst2 contains oasst1's trees; trees are deduplicated by message_tree_id);
             Apache-2.0. English trees in state ready_for_export; one dialogue per tree, its
             best-ranked branch (see best_branch).
- ultrachat: HuggingFaceH4/ultrachat_200k@8049631c, train_sft; MIT. Its text is ChatGPT output
             (ledger 0004 records it); kept as its own source so every mix states whether it
             read UltraChat.
- everyday:  HuggingFaceTB/everyday-conversations-llama3.1-2k@14f54321, train_sft; Apache-2.0.

No smol-smoltalk: that is lever 1's SFT data, and its test split is the preference-prompt pool.

Every dialogue goes through _common.normalise_dialogue (system turns with an instruction drop
the dialogue; only BOILERPLATE_SYSTEM text is stripped; template markers drop it), then a
within-source duplicate check (_common.dialogue_key), then corpus/decontam.py against the frozen
eval sets. Outputs in corpus/data/dialogue_<source>_v1/:
  dialogues.jsonl   kept dialogues {"messages", "uri"}
  text.txt          one rendered document per dialogue (_common.render_dialogue)
  report.json       counts by reason, and the decontamination drops and answer flags
`all` also writes corpus/data/dialogue_pt_v1.ledger.json, the per-source counts together.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

sys.path.insert(0, C.CORPUS_DIR)
import decontam  # noqa: E402

SODA = ("allenai/soda", "fdc848ab0183208ea7808206c91c724414d0a071", ["train.parquet"])
OASST2 = ("OpenAssistant/oasst2", "179dd21fc55192153d94adb0e0ce8f69e222bf75",
          ["data/train-00000-of-00001-88ba0162028a73fc.parquet",
           "data/validation-00000-of-00001-1deeef95c3248fe0.parquet"])
OASST1 = ("OpenAssistant/oasst1", "fdf72ae0827c1cda404aff25b6603abec9e3399b",
          ["data/train-00000-of-00001-b42a775f407cee45.parquet",
           "data/validation-00000-of-00001-134b8fd0c89408b6.parquet"])
ULTRA = ("HuggingFaceH4/ultrachat_200k", "8049631c405ae6576f93f445c6b8166f76f5505a",
         ["data/train_sft-00000-of-00003-a3ecf92756993583.parquet",
          "data/train_sft-00001-of-00003-0a1804bcb6ae68c6.parquet",
          "data/train_sft-00002-of-00003-ee46ed25cfae92c6.parquet"])
EVERYDAY = ("HuggingFaceTB/everyday-conversations-llama3.1-2k", "14f543216b9ba42b6b951dc5bd199460d193b162",
            ["data/train_sft-00000-of-00001.parquet"])
SOURCES = ("soda", "oasst", "ultrachat", "everyday")
NAMES = ["dialogues.jsonl", "text.txt"]


# ---- per-source raw dialogues: (uri, messages) or (uri, None, reason) ------
def soda_dialogue(row):
    """Messages for one SODA row, or (None, reason): the speakers must alternate between
    exactly two names, the first speaking first."""
    utts, speakers = list(row["dialogue"]), list(row["speakers"])
    if len(utts) != len(speakers) or len(set(speakers)) != 2:
        return None, "non_alternating"
    a, b = speakers[0], next(s for s in speakers if s != speakers[0])
    if any(s != (a if k % 2 == 0 else b) for k, s in enumerate(speakers)):
        return None, "non_alternating"
    return [{"role": "user" if k % 2 == 0 else "assistant", "content": u} for k, u in enumerate(utts)], None


def best_branch(root_id, children, msgs):
    """One dialogue per oasst tree: from the root prompt, at each prompt the assistant reply
    with the lowest rank (rank 0 is the best; an unranked reply only when it is the sole
    reply), at each assistant turn the earliest follow-up prompt that itself has a reply.
    Every message on the path must be English, not deleted, and not rejected in review."""
    def ok(m):
        return m["lang"] == "en" and not m["deleted"] and m["review_result"] is not False

    path, node = [], msgs[root_id]
    while node is not None and ok(node):
        path.append({"role": "user" if node["role"] == "prompter" else "assistant", "content": node["text"]})
        kids = [msgs[c] for c in children.get(node["message_id"], ()) if ok(msgs[c])]
        if node["role"] == "prompter":
            ranked = [k for k in kids if k["rank"] is not None]
            node = min(ranked, key=lambda k: (k["rank"], k["message_id"])) if ranked else (
                kids[0] if len(kids) == 1 else None)
        else:
            follow = sorted((k for k in kids if children.get(k["message_id"])),
                            key=lambda k: (k["created_date"], k["message_id"]))
            node = follow[0] if follow else None
    return path


def oasst_trees():
    cols = ["message_id", "parent_id", "text", "role", "lang", "rank", "deleted", "review_result",
            "message_tree_id", "tree_state", "created_date"]
    trees = defaultdict(dict)   # tree id -> {message id: row}
    origin = {}
    for name, (repo, rev, files) in (("oasst2", OASST2), ("oasst1", OASST1)):
        for _, row in C.parquet_rows(repo, rev, files, cols):
            t = row["message_tree_id"]
            if origin.get(t, name) != name:
                continue            # an oasst1 tree already read from oasst2
            origin[t] = name
            trees[t][row["message_id"]] = row
    for t, msgs in sorted(trees.items()):
        roots = [m for m in msgs.values() if m["parent_id"] is None]
        uri = f"{origin[t]}:{t}"
        if len(roots) != 1 or roots[0]["tree_state"] != "ready_for_export" or roots[0]["lang"] != "en":
            yield uri, None, "not_english_or_not_ready"
            continue
        children = defaultdict(list)
        for m in msgs.values():
            if m["parent_id"] is not None:
                children[m["parent_id"]].append(m["message_id"])
        yield uri, best_branch(roots[0]["message_id"], children, msgs), None


def raw(source):
    if source == "soda":
        repo, rev, files = SODA
        for uri, row in C.parquet_rows(repo, rev, files, ["dialogue", "speakers"]):
            m, reason = soda_dialogue(row)
            yield uri, m, reason
    elif source == "oasst":
        yield from oasst_trees()
    else:
        repo, rev, files = ULTRA if source == "ultrachat" else EVERYDAY
        for uri, row in C.parquet_rows(repo, rev, files, ["messages"]):
            yield uri, [dict(m) for m in row["messages"]], None


# ---- build ----------------------------------------------------------------
def _out(cid, name, out_dir):
    if out_dir is None:
        return C.out_path(cid, name)
    os.makedirs(os.path.join(out_dir, cid), exist_ok=True)
    return os.path.join(out_dir, cid, name)


def build(source, pin, dc=None, rows=None, out_dir=None):
    """Write one source's corpus; returns its report. rows: an iterable of (uri, messages,
    reason) standing in for the download (tests). out_dir: a directory in place of
    corpus/data/ (a remote run; the hashes are checked when the files come home)."""
    cid = f"dialogue_{source}_v1"
    if rows is None and out_dir is None and C.already_done(cid, NAMES):
        print(f"{cid}: present and verified")
        return json.load(open(C.out_path(cid, "report.json"), encoding="utf-8"))
    dc = dc or decontam.Decontaminator()
    counts, drops, flags, keys = Counter(), [], [], set()
    d_path, t_path = _out(cid, "dialogues.jsonl", out_dir), _out(cid, "text.txt", out_dir)
    with open(d_path, "w", encoding="utf-8", newline="\n") as fd, open(t_path, "w", encoding="utf-8", newline="\n") as ft:
        for uri, msgs, reason in (raw(source) if rows is None else rows):
            counts["rows_in"] += 1
            if msgs is None:
                counts[reason] += 1
                continue
            turns, reason = C.normalise_dialogue(msgs)
            if turns is None:
                counts[reason] += 1
                continue
            k = C.dialogue_key(turns)
            if k in keys:
                counts["duplicate"] += 1
                continue
            keys.add(k)
            d, f = dc.check([t["content"] for t in turns if t["role"] == "user"],
                            "\n".join(t["content"] for t in turns))
            if d:
                counts["decontaminated"] += 1
                drops.append({"uri": uri, "matches": d})
                continue
            flags += [{"uri": uri, **x} for x in f]   # flags only on rows that are kept
            counts[reason] += 1          # "kept" or "system_stripped"
            fd.write(json.dumps({"messages": turns, "uri": uri}, ensure_ascii=False) + "\n")
            ft.write(C.render_dialogue(turns))
    counts["written"] = counts["kept"] + counts["system_stripped"]
    report = {"id": cid, "counts": dict(sorted(counts.items())), "bytes": os.path.getsize(t_path),
              "boilerplate_system": sorted(C.BOILERPLATE_SYSTEM), "decontam_dropped": drops,
              "answer_flags": flags}
    with open(_out(cid, "report.json", out_dir), "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"{cid}: {report['counts']}")
    if rows is None and out_dir is None:
        C.finish(cid, [d_path, t_path], f"{source} dialogue text for pretraining (dialogue_pt.py)", pin=pin)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build lever 2's dialogue pretraining text from pinned sources.")
    ap.add_argument("source", choices=list(SOURCES) + ["all"])
    ap.add_argument("--pin", action="store_true")
    ap.add_argument("--out-dir", help="write here instead of corpus/data/ (a remote run; pin at home)")
    a = ap.parse_args(argv)
    try:
        dc = decontam.Decontaminator()
        out = {s: build(s, a.pin, dc, out_dir=a.out_dir) for s in (SOURCES if a.source == "all" else (a.source,))}
    except C.HashMismatch as e:
        C.die(str(e))
    if a.source == "all":
        ledger = {s: {"counts": r["counts"], "bytes": r["bytes"],
                      "decontam_dropped": len(r["decontam_dropped"]), "answer_flags": len(r["answer_flags"])}
                  for s, r in out.items()}
        with open(os.path.join(a.out_dir or C.DATA_DIR, "dialogue_pt_v1.ledger.json"), "w", encoding="utf-8",
                  newline="\n") as f:
            json.dump(ledger, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
