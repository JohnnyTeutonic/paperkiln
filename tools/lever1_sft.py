#!/usr/bin/env python3
"""Lever 1 (LOCAL_LLM_STACK.md section 7): the large dialogue SFT set, built
end to end from pinned sources so it can run as a Colab data job.

    python3 tools/lever1_sft.py --out-dir DIR [--vocab GGUF] [--T 256] [--jobs N]

1. Fetch the sources at their pinned revisions (each fetcher verifies its
   files against corpus/hashes): smol-smoltalk qa_v1 (train split only; its
   test split is the preference prompt pool), the TinyChat sample and
   everyday-conversations.
2. Decontaminate each against the frozen eval sets (corpus/decontam.py).
3. Build SFT examples per source with tools/chat_data.py sft (over-context
   examples dropped, dialogues containing turn markers dropped).
4. Shuffle them into one file with a fixed seed (chat_data.py mix).

Writes DIR/lever1_sft_v1.jsonl, the per-source decontamination reports, and
DIR/lever1_sft_v1.ledger.json: every command, its summary line and the
sha256 of every file it read or wrote. Nothing is pinned here; the ledger's
hashes are checked at home before a training run cites the set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "corpus", "data")
VOCAB = os.path.join(DATA, "vocab_fineweb_tinychat_bpe8k_v1", "fineweb_tinychat_bpe8k.gguf")
# (name, fetcher argv, dialogue file the fetcher writes)
SOURCES = [
    ("smol", ["corpus/fetchers/smol_smoltalk.py", "qa_v1"], "smol_smoltalk_qa_v1/dialogues_train.jsonl"),
    ("tinychat", ["corpus/fetchers/tinychat_hf.py", "sample_v1"], "tinychat_hf_sample_v1/dialogues_train.jsonl"),
    ("everyday", ["corpus/fetchers/chat_sft.py", "everyday_v1"], "everyday_v1/dialogues.jsonl"),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run(ledger, argv, files):
    """Run one step from the repo root; record its last stdout line (each tool
    prints a JSON summary) and the hashes of the files it touched."""
    t0 = time.time()
    p = subprocess.run([sys.executable, *argv], cwd=ROOT, capture_output=True, text=True)
    sys.stdout.write(p.stdout)
    sys.stderr.write(p.stderr)
    if p.returncode != 0:
        raise SystemExit(f"step failed ({p.returncode}): {' '.join(argv)}")
    lines = [ln for ln in p.stdout.splitlines() if ln.strip()]
    try:
        summary = json.loads(lines[-1]) if lines else None
    except json.JSONDecodeError:
        summary = lines[-1]
    ledger["steps"].append({"argv": argv, "seconds": round(time.time() - t0, 1), "summary": summary,
                            "sha256": {os.path.relpath(f, ROOT) if f.startswith(ROOT) else os.path.basename(f):
                                       sha256(f) for f in files}})
    print(f"done: {' '.join(argv[:2])} ({time.time() - t0:.0f} s)", flush=True)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--vocab", default=VOCAB)
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--jobs", type=int, default=1, help="tokenising processes for chat_data.py sft (needs its --jobs)")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    out = os.path.abspath(a.out_dir)
    os.makedirs(out, exist_ok=True)
    ledger = {"set": "lever1_sft_v1", "T": a.T, "seed": a.seed, "vocab": os.path.relpath(a.vocab, ROOT),
              "vocab_sha256": sha256(a.vocab), "steps": []}
    parts = []
    for name, fetch, rel in SOURCES:
        src = os.path.join(DATA, rel)
        run(ledger, fetch, [src])
        clean = os.path.join(out, f"{name}_clean.jsonl")
        report = os.path.join(out, f"decontam_{name}.json")
        run(ledger, ["corpus/decontam.py", src, clean, report], [clean, report])
        sft = os.path.join(out, f"sft_{name}.jsonl")
        par = ["--jobs", str(a.jobs)] if a.jobs > 1 else []
        run(ledger, ["tools/chat_data.py", "sft", clean, "--vocab", a.vocab, "--T", str(a.T), *par, "--out", sft],
            [sft])
        os.remove(clean)  # the report lists every drop; the kept rows are recoverable from the source
        parts.append(sft)
    final = os.path.join(out, "lever1_sft_v1.jsonl")
    ledger["result"] = run(ledger, ["tools/chat_data.py", "mix", *parts, "--seed", str(a.seed), "--out", final],
                           [final])
    for p in parts:
        os.remove(p)
    with open(os.path.join(out, "lever1_sft_v1.ledger.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(ledger, f, indent=1)
    print(json.dumps(ledger["result"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
