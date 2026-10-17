#!/usr/bin/env python3
"""Dialogue corpora for the chat model, from pinned revisions.

    python corpus/fetchers/chat_sft.py everyday_v1 [--pin]
    python corpus/fetchers/chat_sft.py ultrachat_v1 [--pin]
    python corpus/fetchers/chat_sft.py eval_v1 [--pin]      # frozen held-out chat prompts

Writes corpus/data/<id>/dialogues.jsonl, one dialogue per line:
{"messages": [{"role": "user"|"assistant", "content": ...}, ...], "uri": ...}.
System turns are dropped. A dialogue must alternate user then assistant,
starting with the user; any that does not is skipped and counted.

- everyday_v1: HuggingFaceTB/everyday-conversations-llama3.1-2k@14f54321,
  train_sft; licence apache-2.0 (dataset card). Short everyday exchanges,
  in the register a small model can reach.
- ultrachat_v1: HuggingFaceH4/ultrachat_200k@8049631c, train_sft (three
  shards); licence mit (dataset card). Open-domain multi-turn dialogue.
- eval_v1: frozen held-out prompts: the first user turn of the first 100
  test_sft dialogues of each source. Written to
  corpus/eval/chat_prompts_v1.jsonl and pinned like every evaluation set:
  never trained on, versioned on any change.

Requires huggingface_hub, pandas and pyarrow.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

EVERYDAY = ("HuggingFaceTB/everyday-conversations-llama3.1-2k", "14f543216b9ba42b6b951dc5bd199460d193b162",
            ["data/train_sft-00000-of-00001.parquet"], ["data/test_sft-00000-of-00001.parquet"])
ULTRA = ("HuggingFaceH4/ultrachat_200k", "8049631c405ae6576f93f445c6b8166f76f5505a",
         ["data/train_sft-00000-of-00003-a3ecf92756993583.parquet",
          "data/train_sft-00001-of-00003-0a1804bcb6ae68c6.parquet",
          "data/train_sft-00002-of-00003-ee46ed25cfae92c6.parquet"],
         ["data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet"])
EVAL_DIR = os.path.join(C.CORPUS_DIR, "eval")


def rows(repo, rev, files):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    for fn in files:
        df = pd.read_parquet(hf_hub_download(repo, fn, repo_type="dataset", revision=rev), columns=["messages"])
        for i, msgs in enumerate(df["messages"]):
            yield f"{repo}@{rev[:8]}/{fn}#{i}", [dict(m) for m in msgs]


def normalise(msgs):
    out = [{"role": m["role"], "content": str(m["content"]).strip()} for m in msgs if m["role"] in ("user", "assistant")]
    if len(out) < 2 or any(m["role"] != ("user" if k % 2 == 0 else "assistant") for k, m in enumerate(out)):
        return None
    if out[-1]["role"] != "assistant" or any(not m["content"] for m in out):
        return None
    return out


def write_dialogues(cid, source, pin):
    repo, rev, train, _ = source
    path = C.out_path(cid, "dialogues.jsonl")
    if C.already_done(cid, ["dialogues.jsonl"]):
        print(f"{cid}: present and verified")
        return
    kept = skipped = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for uri, msgs in rows(repo, rev, train):
            m = normalise(msgs)
            if m is None:
                skipped += 1
                continue
            f.write(json.dumps({"messages": m, "uri": uri}, ensure_ascii=False) + "\n")
            kept += 1
    print(f"{cid}: {kept} dialogues kept, {skipped} skipped")
    C.finish(cid, [path], f"{repo}@{rev} train_sft, user/assistant turns only", pin=pin)


def write_eval(pin):
    eid = "chat_prompts_v1"
    path = os.path.join(EVAL_DIR, f"{eid}.jsonl")
    out = []
    for name, (repo, rev, _, test) in (("everyday", EVERYDAY), ("ultrachat", ULTRA)):
        n = 0
        for uri, msgs in rows(repo, rev, test):
            m = normalise(msgs)
            if m is None:
                continue
            out.append({"prompt": m[0]["content"], "reference": m[1]["content"], "source": name, "uri": uri})
            n += 1
            if n == 100:
                break
    body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out).encode("utf-8")
    want = C.expected_hashes(eid).get(os.path.basename(path))
    if want is None:
        if not pin:
            C.die(f"{eid}: not pinned; run with --pin once")
        os.makedirs(EVAL_DIR, exist_ok=True)
        open(path, "wb").write(body)
        C.pin_hashes(eid, {os.path.basename(path): C.sha256_file(path)}, "FROZEN chat evaluation prompts")
        print(f"{eid}: PINNED ({len(out)} prompts)")
        return
    if C.hashlib.sha256(body).hexdigest() != want:
        C.die(f"{eid}: the sources no longer reproduce the frozen set")
    if not os.path.exists(path):
        open(path, "wb").write(body)
    print(f"{eid}: OK (frozen, reproduces identically)")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fetch pinned dialogue corpora.")
    ap.add_argument("target", choices=["everyday_v1", "ultrachat_v1", "eval_v1"])
    ap.add_argument("--pin", action="store_true")
    a = ap.parse_args(argv)
    try:
        if a.target == "eval_v1":
            write_eval(a.pin)
        else:
            write_dialogues(a.target, EVERYDAY if a.target == "everyday_v1" else ULTRA, a.pin)
    except C.HashMismatch as e:
        C.die(str(e))
    return 0


if __name__ == "__main__":
    sys.exit(main())
