#!/usr/bin/env python3
"""Generate the TinyChat dialogue corpus and the vocabulary GGUF mtstudio needs.

    python tools/get_tinychat_data.py                  # -> data/tinychat.txt
                                                       #    releases/tinychat_vocab.gguf
    options: --version 1|2  --dialogues 1500  --seed (7 for v1, 11 for v2)  --out-dir .

TinyChat is synthetic small talk with consistent question-to-answer
semantics and a deliberately tiny vocabulary (a few hundred words): eight
kinds of exchange (how are you, food, drinks, where did you go, weather,
hobbies, pets, invitations) between a greeting and an optional farewell.
It comes from the author's transformer_cpp experiments
(`scripts/prepare_chat_data.py`, CHAT_EXPERIMENTS.md), where a 2-layer,
128-wide model trained for many epochs on 1,500 of these dialogues gave
the right kind of reply to every test question under greedy decoding,
while the same model on real dialogue (DailyDialog) learned the register
but not the answers. The generator below is that one, unchanged, so the
same seed gives the same corpus.

`--version 2` (tools/tinychat_v2.py) keeps that world and its answers but
says everything in many more ways: several phrasings per question, a
greeting and a question in one message, the assistant asking back, its
name, and one polite fallback for questions outside its world. Version 1
stays the default here so earlier results reproduce; the chat specs and
the quickstart use version 2.

Each dialogue is one line, `user: ... assistant: ... <|endoftext|>`. The
end-of-text marker is a vocabulary entry of its own (id 1), so a chat
server can stop a reply where the dialogue ends.

Nothing is downloaded and nothing outside the standard library is needed:
the GGUF writer here emits only the header fields mtstudio reads.
"""
from __future__ import annotations

import argparse
import collections
import os
import random
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from get_tinystories_data import tokenize_like_mtstudio  # noqa: E402
from tinychat_v2 import tinychat_v2_rows  # noqa: E402

EOT = "<|endoftext|>"


def tinychat_rows(n_dialogues=150_000, seed=7):
    """Synthetic dialogue; verbatim from transformer_cpp/scripts/prepare_chat_data.py."""
    rng = random.Random(seed)
    foods = ["pizza", "pasta", "rice", "soup", "salad", "eggs", "bread",
             "cheese", "apples", "fish", "chicken", "pancakes", "noodles"]
    drinks = ["tea", "coffee", "juice", "water", "milk"]
    places = ["the park", "the beach", "the market", "the library",
              "the museum", "the garden", "the city", "the lake"]
    activities = ["reading", "cooking", "running", "painting", "swimming",
                  "gardening", "playing chess", "watching movies", "hiking"]
    weather = ["sunny", "rainy", "cold", "warm", "windy", "cloudy"]
    feelings_good = ["great", "very good", "happy", "fine", "wonderful"]
    feelings_bad = ["tired", "a little sad", "busy", "sleepy", "not so good"]
    days = ["today", "yesterday", "this morning", "last night", "this week"]
    pets = ["dog", "cat", "bird", "rabbit"]

    def greeting(rng):
        g = rng.choice(["hi !", "hello !", "good morning !", "hey there !"])
        r = rng.choice(["hi ! nice to see you .", "hello ! how are you ?",
                        "good morning ! it is nice to see you ."])
        return [("user", g), ("assistant", r)]

    def qa_pair(rng):
        kind = rng.randrange(8)
        if kind == 0:
            f = rng.choice(feelings_good) if rng.random() < 0.7 else rng.choice(feelings_bad)
            return [("user", rng.choice(["how are you ?", "how do you feel " + rng.choice(days) + " ?"])),
                    ("assistant", rng.choice(["i am " + f + " , thank you .",
                                              "i feel " + f + " " + rng.choice(days) + " ."]))]
        if kind == 1:
            food = rng.choice(foods)
            return [("user", rng.choice(["what do you like to eat ?", "what is your favorite food ?"])),
                    ("assistant", rng.choice(["i like " + food + " very much .",
                                              "my favorite food is " + food + " ."]))]
        if kind == 2:
            d = rng.choice(drinks)
            return [("user", rng.choice(["would you like some " + d + " ?", "do you want " + d + " or " + rng.choice(drinks) + " ?"])),
                    ("assistant", rng.choice(["yes please , i would love some " + d + " .",
                                              "no thank you , i just had some " + rng.choice(drinks) + " ."]))]
        if kind == 3:
            p = rng.choice(places)
            a = rng.choice(activities)
            return [("user", rng.choice(["where did you go " + rng.choice(days) + " ?", "what did you do " + rng.choice(days) + " ?"])),
                    ("assistant", rng.choice(["i went to " + p + " .", "i spent the day " + a + " .",
                                              "i went to " + p + " and enjoyed " + a + " ."]))]
        if kind == 4:
            w = rng.choice(weather)
            return [("user", "how is the weather " + rng.choice(days) + " ?"),
                    ("assistant", rng.choice(["it is very " + w + " .", "it was " + w + " all day ."]))]
        if kind == 5:
            a = rng.choice(activities)
            return [("user", rng.choice(["what is your hobby ?", "what do you like to do ?"])),
                    ("assistant", rng.choice(["i really enjoy " + a + " .", "my hobby is " + a + " ."]))]
        if kind == 6:
            pet = rng.choice(pets)
            return [("user", rng.choice(["do you have a pet ?", "do you like animals ?"])),
                    ("assistant", rng.choice(["yes , i have a little " + pet + " .",
                                              "yes , my " + pet + " is very sweet ."]))]
        p = rng.choice(places)
        return [("user", rng.choice(["do you want to go to " + p + " with me ?", "shall we visit " + p + " ?"])),
                ("assistant", rng.choice(["yes , that sounds lovely .", "sure , i would like that .",
                                          "sorry , i am too busy " + rng.choice(days) + " ."]))]

    def farewell(rng):
        return [("user", rng.choice(["i have to go now . goodbye !", "see you later !"])),
                ("assistant", rng.choice(["goodbye ! have a nice day .", "see you soon ! take care ."]))]

    for _ in range(n_dialogues):
        turns = greeting(rng)
        for _ in range(rng.randrange(1, 4)):
            turns += qa_pair(rng)
        if rng.random() < 0.5:
            turns += farewell(rng)
        yield " ".join(f"{who}: {text}" for who, text in turns)


# ---- a minimal GGUF v3 writer: header and key/values only, no tensors ----
_U32, _STRING, _ARRAY = 4, 8, 9


def _str(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def write_vocab_gguf(path, tokens):
    kv = [
        _str("general.architecture") + struct.pack("<I", _STRING) + _str("llama"),
        _str("tokenizer.ggml.model") + struct.pack("<I", _STRING) + _str("word"),
        _str("tokenizer.ggml.tokens") + struct.pack("<II", _ARRAY, _STRING)
        + struct.pack("<Q", len(tokens)) + b"".join(_str(t) for t in tokens),
        _str("tokenizer.ggml.eos_token_id") + struct.pack("<I", _U32) + struct.pack("<I", tokens.index(EOT)),
    ]
    with open(path, "wb") as f:
        f.write(b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", 0, len(kv)))
        for item in kv:
            f.write(item)
        pad = (-f.tell()) % 32  # data section alignment; there is no data
        f.write(b"\0" * pad)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dialogues", type=int, default=1500)
    ap.add_argument("--version", type=int, choices=[1, 2], default=1)
    ap.add_argument("--seed", type=int, default=None, help="default 7 for v1, 11 for v2")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    data_dir = os.path.join(args.out_dir, "data")
    rel_dir = os.path.join(args.out_dir, "releases")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(rel_dir, exist_ok=True)

    if args.version == 2:
        rows = tinychat_v2_rows(args.dialogues, 11 if args.seed is None else args.seed)
    else:
        rows = tinychat_rows(args.dialogues, 7 if args.seed is None else args.seed)
    lines = [row + " " + EOT for row in rows]
    corpus = os.path.join(data_dir, "tinychat.txt")
    with open(corpus, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")

    counts = collections.Counter()
    for line in lines:
        counts.update(t for t in tokenize_like_mtstudio(line.replace(EOT, " ")))
    tokens = ["<unk>", EOT] + [w for w, _ in counts.most_common() if w.isascii() and w != EOT]
    vocab_path = os.path.join(rel_dir, "tinychat_vocab.gguf")
    write_vocab_gguf(vocab_path, tokens)

    n_tokens = sum(counts.values()) + len(lines)
    print(f"corpus: {corpus} (TinyChat v{args.version}, {len(lines)} dialogues, {n_tokens} tokens)")
    print(f"vocab gguf: {vocab_path} ({len(tokens)} tokens, end-of-text id 1)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
