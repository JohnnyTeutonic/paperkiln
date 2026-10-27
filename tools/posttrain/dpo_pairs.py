#!/usr/bin/env python3
"""The DPO arm's pair file from chat_judge.py pairs output (experiments/chat_general/PLAN.md, the DPO
arm): every pair kept in file order, except one mtstudio would refuse (prompt + the longer reply,
with its end-of-text when it stopped on one, past data.T + 1 tokens in the model's own vocabulary:
re-encoding a sampled reply can take more tokens than were sampled). Reports the training set and the
breakdown by how each reply stopped, and the subset whose chosen reply finished.

    python tools/posttrain/dpo_pairs.py PAIRS.jsonl --vocab MODEL.gguf --T 512 --out DPO_PAIRS.jsonl
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def build(pairs, encode, T):
    """(kept pairs, report)."""
    kept, over = [], 0
    for p in pairs:
        n_p = len(encode(p["prompt"]))
        lens = [len(encode(p[s])) + (p.get(f"{s}_stop", "eos") == "eos") for s in ("chosen", "rejected")]
        if n_p + max(lens) > T + 1:
            over += 1
            continue
        kept.append(p)
    stops = collections.Counter(f"{p.get('chosen_stop', 'eos')}/{p.get('rejected_stop', 'eos')}" for p in kept)
    finished = sum(p.get("chosen_stop", "eos") == "eos" for p in kept)
    val = max(1, len(kept) // 20)
    return kept, {"pairs_in": len(pairs), "over_T": over, "pairs": len(kept), "train_pairs": len(kept) - val,
                  "val_pairs": val, "stops_chosen_rejected": dict(stops), "chosen_finished": finished,
                  "chosen_finished_share": round(finished / max(len(kept), 1), 4)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("pairs")
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import bpe
    tok = bpe.load_vocab(a.vocab)
    pairs = [json.loads(x) for x in open(a.pairs, encoding="utf-8") if x.strip()]
    kept, rep = build(pairs, tok.encode, a.T)
    data = "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in kept).encode("utf-8")
    with open(a.out, "wb") as g:
        g.write(data)
    rep.update(out=a.out, sha256=hashlib.sha256(data).hexdigest())
    print(json.dumps(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
