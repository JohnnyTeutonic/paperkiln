#!/usr/bin/env python3
"""Colab data job for the tiny-chat experiments (experiments/tiny_chat_2026-10/PREREGISTRATION.md):
human dialogues split three ways, pretraining mixes at the registered dialogue shares, the two
fixed text panels and the SFT set, all pre-tokenised with the llama3-scheme vocabulary.

    python3 tools/colab_chat/tiny_chat_data.py --out DIR --vocab V.gguf \
        --fineweb corpus/data/fineweb_edu_shard0_v1 --shares 0,25,50,100 \
        --mix-bytes 800000000 --max-epochs 4 --T 512 --seed 1

1. Human dialogues: --human-fetch runs the corpus fetcher (corpus/fetchers/human_dialogue.py,
   built by the corpus agent; expected to write dialogues.jsonl files of {"messages", "uri"} under
   {human}); every dialogues.jsonl under --human-dir is read. Messages must alternate user and
   assistant from a user turn; a trailing user turn is dropped, other dialogues are counted and
   dropped.
2. corpus/decontam.py over all of them (its frozen eval sets; Carry-5's once decontam.py knows it).
3. Split by sha256 of the dialogue's content key (_common.dialogue_key, so duplicates share a
   split): 5% val (the dialogue panel), 10% SFT, 85% pretraining.
4. SFT set: chat_data.py sft --all-turns at T (one row per dialogue, every assistant turn
   trained), shuffled (chat_data.py mix, seed): sft_human.jsonl. mtstudio holds out its last 5%.
5. Panels: FineWeb-Edu's val.txt and the dialogue val split, rendered, as .ids.
6. Mixes, one per share: --mix-bytes of FineWeb-Edu train text and rendered pretraining dialogues,
   share% dialogue by bytes (corpus/mix_pretrain.py, no tail). When the dialogues are short of
   share x mix-bytes, the dialogue source is k concatenated copies (k = the fewest that cover it),
   so FineWeb stays unrepeated and dialogue documents repeat; a share needing k > --max-epochs runs
   at the largest share k = max-epochs covers and the ledger says so (share_effective).
Outputs in --out: mix_s<share>.ids, panel_fineweb.ids, panel_dialogue.ids, sft_human.jsonl and
data.ledger.json (counts, bytes, tokens, copies, sha256 of every output). Idempotent: a finished
output is not rebuilt.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "corpus", "fetchers"))
import _common as C  # noqa: E402

SPLIT = {"val": 5, "sft": 15}  # sha256 bucket < 5: val; < 15: sft; else pretraining


def sh(argv, ledger, step):
    print("+", " ".join(argv), flush=True)
    p = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
    sys.stdout.write(p.stdout[-4000:])
    sys.stderr.write(p.stderr[-4000:])
    if p.returncode != 0:
        raise SystemExit(f"{step} failed ({p.returncode})")
    lines = [x for x in p.stdout.splitlines() if x.strip()]
    try:
        ledger.setdefault("steps", {})[step] = json.loads(lines[-1]) if lines else None
    except json.JSONDecodeError:
        ledger.setdefault("steps", {})[step] = lines[-1]


def ids(text, vocab, out, ledger, step):
    if not os.path.exists(out):
        sh([sys.executable, "tools/corpus_ids.py", text, vocab, out + ".tmp", "--threads", "6"], ledger, step)
        os.replace(out + ".tmp", out)
    else:
        ledger.setdefault("steps", {})[step] = "present"
    with open(out, "rb") as f:  # PKIDS001, uint32 width, uint32 vocab, uint64 count (tools/corpus_ids.py)
        head = f.read(24)
    ledger.setdefault("outputs", {})[os.path.basename(out)] = {"bytes": os.path.getsize(out),
                                                             "tokens": int.from_bytes(head[16:24], "little"),
                                                             "sha256": C.sha256_file(out)}


def alternate(msgs):
    """user/assistant alternation from a user turn, a trailing user turn dropped; None if broken."""
    msgs = [m for m in msgs if m.get("role") in ("user", "assistant")]
    if len(msgs) % 2:
        msgs = msgs[:-1]
    if len(msgs) < 2 or any(m["role"] != ("user" if i % 2 == 0 else "assistant") or not m["content"].strip()
                            for i, m in enumerate(msgs)):
        return None
    return msgs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--fineweb", required=True, help="the FineWeb-Edu corpus dir (pretrain_train.txt, val.txt)")
    ap.add_argument("--shares", required=True, help="dialogue shares in %%, e.g. 0,25,50,100")
    ap.add_argument("--mix-bytes", type=int, required=True)
    ap.add_argument("--max-epochs", type=int, default=4)
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--human-dir", default=None, help="default OUT/human")
    ap.add_argument("--human-fetch", default="python3 corpus/fetchers/human_dialogue.py default --out-dir {human}")
    a = ap.parse_args(argv)
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    human = os.path.abspath(a.human_dir or os.path.join(out, "human"))
    ledger_path = os.path.join(out, "data.ledger.json")
    ledger = {"recipe": "tools/colab_chat/tiny_chat_data.py", "args": vars(a), "vocab_sha256": C.sha256_file(a.vocab)}
    shares = [int(x) for x in a.shares.split(",")]

    # 1-3. Human dialogues: fetch, normalise, decontaminate, split.
    pt_txt, val_txt, sft_dl = (os.path.join(out, f) for f in ("human_pt.txt", "human_val.txt", "human_sft.jsonl"))
    if not all(os.path.exists(p) for p in (pt_txt, val_txt, sft_dl)):
        files = sorted(glob.glob(os.path.join(human, "**", "dialogues.jsonl"), recursive=True))
        if not files and a.human_fetch:
            sh(a.human_fetch.format(human=human).split(), ledger, "human_fetch")
            files = sorted(glob.glob(os.path.join(human, "**", "dialogues.jsonl"), recursive=True))
        if not files:
            raise SystemExit(f"no dialogues.jsonl under {human}: the human dialogue corpus is missing")
        allf = os.path.join(out, "human_all.jsonl")
        n_in = n_bad = 0
        with open(allf, "w", encoding="utf-8", newline="\n") as g:
            for p in files:
                src = os.path.relpath(p, human)
                with open(p, encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        n_in += 1
                        d = json.loads(line)
                        m = alternate(d.get("messages", []))
                        if m is None:
                            n_bad += 1
                            continue
                        g.write(json.dumps({"messages": m, "uri": d.get("uri", ""), "source": src},
                                           ensure_ascii=False) + "\n")
        ledger["human"] = {"files": {os.path.relpath(p, human): C.sha256_file(p) for p in files},
                           "dialogues_in": n_in, "dropped_not_alternating": n_bad}
        clean = os.path.join(out, "human_clean.jsonl")
        sh([sys.executable, "corpus/decontam.py", allf, clean, os.path.join(out, "human_decontam.json")], ledger,
           "decontam")
        counts = {"val": 0, "sft": 0, "pt": 0}
        with open(clean, encoding="utf-8") as f, open(pt_txt, "w", encoding="utf-8", newline="\n") as fp, \
                open(val_txt, "w", encoding="utf-8", newline="\n") as fv, \
                open(sft_dl, "w", encoding="utf-8", newline="\n") as fs:
            for line in f:
                d = json.loads(line)
                b = int(hashlib.sha256(C.dialogue_key(d["messages"]).encode()).hexdigest()[:8], 16) % 100
                part = "val" if b < SPLIT["val"] else "sft" if b < SPLIT["sft"] else "pt"
                counts[part] += 1
                if part == "sft":
                    fs.write(line)
                else:
                    (fv if part == "val" else fp).write(C.render_dialogue(d["messages"]))
        ledger["human"]["split"] = counts
        os.remove(allf)
        os.remove(clean)
    H = os.path.getsize(pt_txt)
    ledger["human_pt_bytes"] = H

    # 4. SFT set: one row per dialogue, every assistant turn trained, shuffled.
    sft = os.path.join(out, "sft_human.jsonl")
    if not os.path.exists(sft):
        raw = os.path.join(out, "sft_raw.jsonl")
        sh([sys.executable, "tools/chat_data.py", "sft", sft_dl, "--vocab", a.vocab, "--T", str(a.T), "--all-turns",
            "--out", raw], ledger, "sft")
        sh([sys.executable, "tools/chat_data.py", "mix", raw, "--seed", str(a.seed), "--out", sft + ".tmp"], ledger,
           "sft_shuffle")
        os.replace(sft + ".tmp", sft)
        os.remove(raw)
    ledger.setdefault("outputs", {})["sft_human.jsonl"] = {"bytes": os.path.getsize(sft), "sha256": C.sha256_file(sft)}

    # 5. Panels.
    ids(os.path.join(a.fineweb, "val.txt"), a.vocab, os.path.join(out, "panel_fineweb.ids"), ledger, "panel_fineweb")
    ids(val_txt, a.vocab, os.path.join(out, "panel_dialogue.ids"), ledger, "panel_dialogue")

    # 6. Mixes.
    fw = os.path.join(a.fineweb, "pretrain_train.txt")
    ledger["mixes"] = {}
    for s in shares:
        dst = os.path.join(out, f"mix_s{s}.ids")
        info = {"share_registered": s}
        need = s / 100 * a.mix_bytes
        k = max(1, math.ceil(need / (0.95 * H))) if s > 0 else 0
        share = s
        if k > a.max_epochs:
            share = int(100 * a.max_epochs * 0.95 * H / a.mix_bytes)
            k = a.max_epochs
            info["note"] = f"dialogue short: {k} copies cover {share}% of {a.mix_bytes} bytes"
        info.update({"share_effective": share, "dialogue_copies": k})
        prefix = os.path.join(out, f"mix_s{s}")
        if os.path.exists(dst):
            ids(prefix + ".main.txt", a.vocab, dst, ledger, f"ids_s{s}")  # present: records its hash
            if os.path.exists(prefix + ".ledger.json"):
                info["mix_ledger"] = json.load(open(prefix + ".ledger.json", encoding="utf-8"))
        else:
            src = [f"fineweb={fw}"]
            dial = pt_txt
            if k > 1:
                dial = os.path.join(out, "human_pt_copies.txt")
                with open(dial, "wb") as g:
                    for _ in range(k):
                        with open(pt_txt, "rb") as f:
                            while True:
                                b = f.read(1 << 24)
                                if not b:
                                    break
                                g.write(b)
            if share > 0:
                src.append(f"dialogue={dial}")
            ratio = f"fineweb={1 - share / 100:.4f}" + (f",dialogue={share / 100:.4f}" if share > 0 else "")
            argv = [sys.executable, "corpus/mix_pretrain.py", prefix, "--main", ratio, "--tail", ratio,
                    "--tail-fraction", "0", "--total-bytes", str(a.mix_bytes), "--seed", str(a.seed)]
            for x in src:
                argv += ["--source", x]
            sh(argv, ledger, f"mix_s{s}")
            info["mix_ledger"] = json.load(open(prefix + ".ledger.json", encoding="utf-8"))
            ids(prefix + ".main.txt", a.vocab, dst, ledger, f"ids_s{s}")
            for f in (prefix + ".main.txt", prefix + ".tail.txt", os.path.join(out, "human_pt_copies.txt")):
                if os.path.exists(f):
                    os.remove(f)
        ledger["mixes"][f"s{s}"] = info
    with open(ledger_path + ".tmp", "w", encoding="utf-8", newline="\n") as f:
        json.dump(ledger, f, indent=1)
    os.replace(ledger_path + ".tmp", ledger_path)
    print(json.dumps({"mixes": {k: v["share_effective"] for k, v in ledger["mixes"].items()},
                      "human_pt_bytes": H}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
