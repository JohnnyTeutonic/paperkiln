#!/usr/bin/env python3
"""Fetch a public training corpus and build the vocabulary GGUF mtstudio needs.

    python tools/get_tinystories_data.py            # -> data/tinystories_valid.txt
                                                    #    releases/tinystories_vocab.gguf
    options: --split valid|train  --max-mb 20  --vocab 4096  --out-dir .

mtstudio's spec wants two files: `data.corpus`, plain text, and
`data.vocab`, a GGUF whose `tokenizer.ggml.tokens` list supplies the
vocabulary (the first `vocab_cap` entries are used). This script makes
both from TinyStories (Eldan & Li, 2023; Hugging Face dataset
`roneneldan/TinyStories`, CDLA-Sharing-1.0), so a fresh clone can train
and chat without any file from the author's machine:

  1. downloads `TinyStories-valid.txt` (about 22 MB; `--split train` for the
     2 GB training file) with huggingface_hub, no account needed;
  2. keeps the first `--max-mb` megabytes as the corpus;
  3. tokenises it exactly as mtstudio does (lower-case; a token is a run of
     letters, digits and apostrophes, or a single punctuation character;
     the story separator `<|endoftext|>` is one token);
  4. writes the most frequent `--vocab` tokens, with `<unk>` at id 0 and,
     when the corpus has story separators, `<|endoftext|>` at id 1 (also
     recorded as the GGUF's eos_token_id, so generation stops at the end
     of a story), into a minimal GGUF that mtstudio's reader accepts.

Requires: huggingface_hub, gguf  (`pip install huggingface_hub gguf`).
"""
from __future__ import annotations

import argparse
import collections
import os
import sys


EOS = "<|endoftext|>"


def tokenize_like_mtstudio(text):
    """Mirror of wordtok::tokenize (include/microtorch/word_tokenizer.hpp,
    used by tools/mtstudio.cpp): yields tokens. EOS is always one token
    here; the C++ side makes it one token whenever the vocabulary has it,
    which a vocabulary built from these counts does."""
    cur = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        i += 1
        if ch == "<" and text.startswith(EOS, i - 1):
            if cur:
                yield "".join(cur)
                cur = []
            yield EOS
            i += len(EOS) - 1
        elif ch.isalpha() or ch.isdigit() or ch == "'":
            cur.append(ch.lower())
        else:
            if cur:
                yield "".join(cur)
                cur = []
            if not ch.isspace():
                yield ch
    if cur:
        yield "".join(cur)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["valid", "train"], default="valid")
    ap.add_argument("--max-mb", type=float, default=20.0)
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download
    import gguf

    fname = "TinyStories-valid.txt" if args.split == "valid" else "TinyStories-train.txt"
    print(f"downloading {fname} from roneneldan/TinyStories ...")
    src = hf_hub_download(repo_id="roneneldan/TinyStories", filename=fname,
                          repo_type="dataset")

    data_dir = os.path.join(args.out_dir, "data")
    rel_dir = os.path.join(args.out_dir, "releases")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(rel_dir, exist_ok=True)
    corpus = os.path.join(data_dir, f"tinystories_{args.split}.txt")
    limit = int(args.max_mb * 1024 * 1024)
    with open(src, "rb") as f, open(corpus, "wb") as g:
        g.write(f.read(limit))
    print(f"corpus: {corpus} ({os.path.getsize(corpus) / 1e6:.1f} MB)")

    counts = collections.Counter()
    with open(corpus, encoding="utf-8", errors="replace") as f:
        for line in f:
            counts.update(tokenize_like_mtstudio(line))
    # mtstudio only ever looks at ASCII-classified characters; keep the
    # vocabulary ASCII so the C++ tokenizer and this one agree.
    words = [w for w, _ in counts.most_common() if w.isascii() and w != EOS]
    # End-of-text first after <unk>, so no vocab cap can drop it.
    special = ["<unk>"] + ([EOS] if counts[EOS] else [])
    tokens = special + words[: args.vocab - len(special)]
    print(f"vocabulary: {len(tokens)} tokens from {sum(counts.values())} in corpus "
          f"(coverage {sum(counts[w] for w in tokens[1:]) / max(1, sum(counts.values())):.3f})")

    vocab_path = os.path.join(rel_dir, "tinystories_vocab.gguf")
    w = gguf.GGUFWriter(vocab_path, "llama")
    w.add_tokenizer_model("word")
    w.add_token_list(tokens)
    if EOS in tokens:
        w.add_eos_token_id(tokens.index(EOS))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"vocab gguf: {vocab_path} ({os.path.getsize(vocab_path) / 1e3:.0f} kB)")
    print("next: ./build/mtstudio run specs/tinystories-llama-3k.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
