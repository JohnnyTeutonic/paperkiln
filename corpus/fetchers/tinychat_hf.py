#!/usr/bin/env python3
"""TinyChat (starhopp3r/TinyChat): a million short synthetic chat
conversations in BASIC English, built on the TinyStories method. This is
the chat model's data (corpus ledger 0003).

    python corpus/fetchers/tinychat_hf.py full_v1 [--pin]     # 745 MB source; run on the training VM
    python corpus/fetchers/tinychat_hf.py sample_v1 [--pin]   # first 20 MiB only (tokenizer training, tests)
    python corpus/fetchers/tinychat_hf.py eval_v1 [--pin]     # frozen held-out prompts (needs full_v1)

Source: tinychat.txt at revision REV; licence cc-by-nc-4.0 (dataset card):
non-commercial, which any model trained on it inherits. One conversation
per line, turns written "[INST] user [/INST] assistant ...".

Each conversation is parsed into alternating user/assistant turns and
assigned to train / val / test by a hash of its text: 98 / 1 / 1,
deterministic and independent of line order. Outputs in corpus/data/<id>/:
  dialogues_{train,val,test}.jsonl  {"messages": [...], "uri"}
  pretrain_train.txt                the train conversations as plain dialogue
                                    text, one turn per line with no role markers,
                                    conversations separated by <|endoftext|>:
                                    the base model learns language and
                                    conversation, not the turn protocol (SFT's job)
eval_v1 writes corpus/eval/tinychat_hf_prompts_v1.jsonl: the first user turn
and reference reply of the first 200 test conversations. It is frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

REPO = "starhopp3r/TinyChat"
REV = "1a19729d536023d40072d491d4691e4c71461e92"
URL = f"https://huggingface.co/datasets/{REPO}/resolve/{REV}/tinychat.txt"
SAMPLE_BYTES = 20 * 1024 * 1024
EVAL_DIR = os.path.join(C.CORPUS_DIR, "eval")
TURN = re.compile(r"\[INST\](.*?)\[/INST\](.*?)(?=\[INST\]|$)", re.S)


def parse(line):
    """[INST] u [/INST] a ... -> [{"role": "user"...}, {"role": "assistant"...}, ...] or None."""
    msgs = []
    for u, a in TURN.findall(line):
        u, a = " ".join(u.split()), " ".join(a.split())
        if not u or not a:
            return None
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    return msgs or None


def split_of(line):
    h = int(hashlib.sha256(line.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "test" if h == 0 else ("val" if h == 1 else "train")


def lines(sample):
    if sample:
        req = urllib.request.Request(URL, headers={"Range": f"bytes=0-{SAMPLE_BYTES - 1}"})
        with urllib.request.urlopen(req, timeout=300) as r:
            body = r.read()
        text = body.decode("utf-8", errors="ignore")
        yield from text.split("\n")[:-1]  # the last line may be cut mid-way
        return
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(REPO, "tinychat.txt", repo_type="dataset", revision=REV)
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield line.rstrip("\n")


def build(cid, sample, pin):
    names = ["dialogues_train.jsonl", "dialogues_val.jsonl", "dialogues_test.jsonl", "pretrain_train.txt"]
    if C.already_done(cid, names):
        print(f"{cid}: present and verified")
        return
    outs = {k: open(C.out_path(cid, f"dialogues_{k}.jsonl"), "w", encoding="utf-8", newline="\n")
            for k in ("train", "val", "test")}
    pre = open(C.out_path(cid, "pretrain_train.txt"), "w", encoding="utf-8", newline="\n")
    counts = {"train": 0, "val": 0, "test": 0, "skipped": 0}
    for i, line in enumerate(lines(sample)):
        line = line.strip()
        msgs = parse(line) if line else None
        if msgs is None:
            counts["skipped"] += line != ""
            continue
        sp = split_of(line)
        outs[sp].write(json.dumps({"messages": msgs, "uri": f"{REPO}@{REV[:8]}/tinychat.txt#L{i + 1}"},
                                  ensure_ascii=False) + "\n")
        counts[sp] += 1
        if sp == "train":
            pre.write("\n".join(m["content"] for m in msgs) + "\n<|endoftext|>\n")
    for f in list(outs.values()) + [pre]:
        f.close()
    print(f"{cid}: {counts}")
    C.finish(cid, [C.out_path(cid, n) for n in names],
             f"{REPO}@{REV} tinychat.txt{' first 20 MiB' if sample else ''}, split 98/1/1 by sha256", pin=pin)


def build_eval(pin):
    src = os.path.join(C.DATA_DIR, "tinychat_hf_full_v1", "dialogues_test.jsonl")
    if not os.path.exists(src):
        C.die("eval_v1 needs full_v1 first")
    out = []
    with open(src, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out.append({"prompt": r["messages"][0]["content"], "reference": r["messages"][1]["content"],
                        "uri": r["uri"]})
            if len(out) == 200:
                break
    eid = "tinychat_hf_prompts_v1"
    path = os.path.join(EVAL_DIR, f"{eid}.jsonl")
    body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out).encode("utf-8")
    want = C.expected_hashes(eid).get(os.path.basename(path))
    if want is None:
        if not pin:
            C.die(f"{eid}: not pinned; run with --pin once")
        os.makedirs(EVAL_DIR, exist_ok=True)
        open(path, "wb").write(body)
        C.pin_hashes(eid, {os.path.basename(path): C.sha256_file(path)}, "FROZEN TinyChat held-out chat prompts")
        print(f"{eid}: PINNED ({len(out)} prompts)")
    elif C.hashlib.sha256(body).hexdigest() != want:
        C.die(f"{eid}: the source no longer reproduces the frozen set")
    else:
        print(f"{eid}: OK (frozen, reproduces identically)")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fetch TinyChat (starhopp3r) from a pinned revision.")
    ap.add_argument("target", choices=["full_v1", "sample_v1", "eval_v1"])
    ap.add_argument("--pin", action="store_true")
    a = ap.parse_args(argv)
    try:
        if a.target == "eval_v1":
            build_eval(a.pin)
        else:
            build(f"tinychat_hf_{a.target}", a.target == "sample_v1", a.pin)
    except C.HashMismatch as e:
        C.die(str(e))
    return 0


if __name__ == "__main__":
    sys.exit(main())
