#!/usr/bin/env python3
"""FineWeb-Edu shards 0 and 1 as one source for B's mix (stronger_base_spec.md, section 10). Claude Opus 5.5, 7 Nov 2026.

    python3 tools/colab_chat/tc_b1_fineweb.py OUT.txt --val VAL.txt SHARD_TRAIN.txt [SHARD_TRAIN.txt ...]

Stage 1c measured the two train files: 4,624 distinct documents in both, 3,075 repeats inside them, and 17 of shard 1's train
documents equal to documents of shard 0's val.txt, the FineWeb panel. Each document is kept once, first seen first, and any
document equal to one of the val file is dropped. Whole documents only; their order is the input order, so the output is
deterministic (mix_pretrain.py shuffles by seed). Writes OUT.txt and OUT.txt.ledger.json (counts, bytes, sha256).
"""
import argparse
import hashlib
import json
import os

SEP = b"<|endoftext|>\n"


def docs(path):
    buf = b""
    with open(path, "rb") as f:
        while True:
            block = f.read(1 << 24)
            parts = (buf + block).split(SEP)
            buf = parts.pop() if block else b""
            for d in parts:
                yield d
            if not block:
                return


def key(d):
    return hashlib.blake2b(d, digest_size=12).digest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out")
    ap.add_argument("--val", required=True)
    ap.add_argument("train", nargs="+")
    a = ap.parse_args(argv)
    val = {key(d) for d in docs(a.val)}
    seen = set()
    led = {"val_documents": len(val), "inputs": [], "kept": 0, "dropped_equal_val": 0, "dropped_repeat": 0}
    h = hashlib.sha256()
    with open(a.out + ".tmp", "wb") as g:
        for p in a.train:
            n = 0
            for d in docs(p):
                if not d:
                    continue
                n += 1
                k = key(d)
                if k in val:
                    led["dropped_equal_val"] += 1
                elif k in seen:
                    led["dropped_repeat"] += 1
                else:
                    seen.add(k)
                    led["kept"] += 1
                    g.write(d + SEP)
                    h.update(d + SEP)
            led["inputs"].append({"file": os.path.basename(os.path.dirname(p)), "documents": n})
    os.replace(a.out + ".tmp", a.out)
    led.update(bytes=os.path.getsize(a.out), sha256=h.hexdigest())
    with open(a.out + ".ledger.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(led, f, indent=1)
    print(json.dumps(led))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
