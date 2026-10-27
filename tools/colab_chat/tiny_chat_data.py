#!/usr/bin/env python3
"""Colab data job for the tiny-chat experiments (experiments/tiny_chat_2026-10/PREREGISTRATION.md):
human dialogues split three ways, pretraining mixes at the registered dialogue shares, the two
fixed text panels and the SFT set, all pre-tokenised with the llama3-scheme vocabulary.

    python3 tools/colab_chat/tiny_chat_data.py --out DIR --vocab V.gguf \
        --fineweb corpus/data/fineweb_edu_shard0_v1 --shares 0,25,50,100 \
        --mix-bytes 800000000 --max-epochs 4 --T 512 --seed 1

1. Human dialogues: --human-fetch runs the corpus fetcher (corpus/fetchers/human_dialogue.py,
   which writes human_<source>_v1/dialogues.jsonl of {"messages", "uri"} under {human}); the
   dialogues.jsonl of the --sources (all found, without it) under the --human-dir(s) are read, in
   source order. --require-pinned refuses unless each file and its text.txt match the source's
   pin (corpus/hashes/human_<source>_v1.sha256), so every VM trains on the bytes measured at home.
   Messages must alternate user and assistant from a user turn; a trailing user turn is dropped,
   other dialogues are counted and dropped.
2. corpus/decontam.py over all of them: its frozen eval sets plus Carry-5
   (corpus/eval/carry5_v1.jsonl, required: its user turns as prompts, its gold replies as
   references; PREREGISTRATION.md amendment 1).
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

The long-carry programme (experiments/tiny_chat_long_2026-10/PREREGISTRATION_DRAFT.md sections 2, 4 and 6):
--carry12 adds Carry-12 v1 (corpus/eval/carry12_v1.jsonl, which must equal its pin) to step 2's
decontamination beside Carry-5, so dialogues are dropped by the same rules and counted; --T 1024 builds the
SFT set at T 1,024 (the mixes are token streams and do not depend on T: mtstudio draws its windows); and
--long-sft MIN:COPIES also writes sft_human_long<COPIES>.jsonl for cell D: the same SFT rows, each row of
MIN or more trained turns (exchanges) present COPIES times, shuffled with the same seed.

--measure-only stops after step 3 and prints human_pt_bytes, the P of the registered level rule
(make_jobs.py --dialogue-pt-bytes): the same files and code as the data job, so P equals what the
VMs measure.
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

sys.path.insert(0, os.path.join(ROOT, "corpus"))
import decontam  # noqa: E402

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


def human_files(dirs, sources):
    """[(source, dialogues.jsonl)] in source order: each source's human_<source>_v1 under the
    first dir that has it (all sources found, when sources is None)."""
    found = {}
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "**", "dialogues.jsonl"), recursive=True)):
            cid = os.path.basename(os.path.dirname(p))
            src = cid[len("human_"):-len("_v1")] if cid.startswith("human_") and cid.endswith("_v1") else cid
            found.setdefault(src, p)
    if sources is None:
        return sorted(found.items())
    missing = [s for s in sources if s not in found]
    if missing:
        raise SystemExit(f"human dialogue sources missing under {dirs}: {missing}")
    return [(s, found[s]) for s in sorted(sources)]


def check_pins(files):
    """Every source file must equal its pin (corpus/hashes/human_<source>_v1.sha256)."""
    bad = []
    for src, p in files:
        want = C.expected_hashes(f"human_{src}_v1")
        for f in ("dialogues.jsonl", "text.txt"):
            q = os.path.join(os.path.dirname(p), f)
            if not want or not os.path.exists(q) or C.sha256_file(q) != want.get(f):
                bad.append(f"human_{src}_v1/{f}")
    if bad:
        raise SystemExit(f"not the pinned bytes (or no pin): {bad}")


def trained_turns(row):
    """Exchanges trained in an all-turns SFT row (chat_data.py sft --all-turns: {"segments"}, one trained
    segment per assistant reply)."""
    return sum(1 for seg in row.get("segments", []) if seg.get("loss"))


def long_rows(raw, dst, min_turns, copies):
    """Cell D's SFT rows: every row of raw once, and each row of min_turns or more trained turns copies
    times in all (shuffled afterwards). Returns the counts for the ledger."""
    n = n_long = 0
    with open(raw, encoding="utf-8") as f, open(dst, "w", encoding="utf-8", newline="\n") as g:
        for line in f:
            if not line.strip():
                continue
            n += 1
            row = json.loads(line)
            k = copies if trained_turns(row) >= min_turns else 1
            n_long += k > 1
            row.setdefault("mix_source", os.path.basename(raw))  # as in sft_human.jsonl (chat_data.py mix)
            line = json.dumps(row, ensure_ascii=False) + "\n"
            for _ in range(k):
                g.write(line)
    return {"min_turns": min_turns, "copies": copies, "rows": n, "rows_long": n_long,
            "rows_out": n + n_long * (copies - 1)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab")
    ap.add_argument("--fineweb", help="the FineWeb-Edu corpus dir (pretrain_train.txt, val.txt)")
    ap.add_argument("--shares", help="dialogue shares in %%, e.g. 0,25,50,100")
    ap.add_argument("--mix-bytes", type=int)
    ap.add_argument("--max-epochs", type=int, default=4)
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--human-dir", action="append", default=None,
                    help="where human_<source>_v1/ are (repeat; default OUT/human, where --human-fetch writes)")
    ap.add_argument("--human-fetch", default="python3 corpus/fetchers/human_dialogue.py default --out-dir {human}")
    ap.add_argument("--sources", default=None, help="comma-separated human dialogue sources (default: all found)")
    ap.add_argument("--require-pinned", action="store_true", help="refuse unless every source file matches its pin")
    ap.add_argument("--measure-only", action="store_true", help="stop after the split; print human_pt_bytes")
    ap.add_argument("--carry12", action="store_true", help="Carry-12 v1 joins the decontamination (pinned bytes)")
    ap.add_argument("--long-sft", default="", help="MIN:COPIES: sft_human_long<COPIES>.jsonl, rows of MIN+ turns x COPIES")
    a = ap.parse_args(argv)
    long_min = long_copies = 0
    if a.long_sft:
        try:
            long_min, long_copies = (int(x) for x in a.long_sft.split(":"))
        except ValueError:
            ap.error("--long-sft is MIN:COPIES, e.g. 10:4")
        if long_min < 1 or long_copies < 2:
            ap.error("--long-sft needs MIN >= 1 and COPIES >= 2")
    if not a.measure_only and not all((a.vocab, a.fineweb, a.shares, a.mix_bytes)):
        ap.error("--vocab, --fineweb, --shares and --mix-bytes are required unless --measure-only")
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    dirs = [os.path.abspath(d) for d in (a.human_dir or [os.path.join(out, "human")])]
    human = dirs[0]
    sources = a.sources.split(",") if a.sources else None
    ledger_path = os.path.join(out, "data.ledger.json")
    ledger = {"recipe": "tools/colab_chat/tiny_chat_data.py", "args": vars(a),
              "vocab_sha256": C.sha256_file(a.vocab) if a.vocab else None}
    shares = [int(x) for x in a.shares.split(",")] if a.shares else []

    # 1-3. Human dialogues: fetch, normalise, decontaminate, split.
    pt_txt, val_txt, sft_dl = (os.path.join(out, f) for f in ("human_pt.txt", "human_val.txt", "human_sft.jsonl"))
    if not all(os.path.exists(p) for p in (pt_txt, val_txt, sft_dl)):
        have = glob.glob(os.path.join(human, "**", "dialogues.jsonl"), recursive=True)
        if not have and a.human_fetch:
            sh(a.human_fetch.format(human=human).split(), ledger, "human_fetch")
        files = human_files(dirs, sources)
        if not files:
            raise SystemExit(f"no dialogues.jsonl under {dirs}: the human dialogue corpus is missing")
        if a.require_pinned:
            check_pins(files)
        allf = os.path.join(out, "human_all.jsonl")
        n_in = n_bad = 0
        with open(allf, "w", encoding="utf-8", newline="\n") as g:
            for name, p in files:
                src = f"human_{name}_v1/dialogues.jsonl"
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
        ledger["human"] = {"files": {f"human_{n}_v1/dialogues.jsonl": C.sha256_file(p) for n, p in files},
                           "sources": [n for n, _ in files], "pins_checked": a.require_pinned,
                           "dialogues_in": n_in, "dropped_not_alternating": n_bad}
        clean = os.path.join(out, "human_clean.jsonl")
        import human_dialogue  # the fetcher's Decontaminator with Carry-5's turns added
        dc, carry5_sha = human_dialogue.decontaminator(carry5=True, carry12=a.carry12)
        if not carry5_sha:
            raise SystemExit("corpus/eval/carry5_v1.jsonl is missing: the human dialogues cannot be decontaminated")
        if a.carry12:
            want = C.expected_hashes("carry12_v1").get("carry12_v1.jsonl")
            got = human_dialogue.carry12_sha()
            if not want or got != want:
                raise SystemExit(f"corpus/eval/carry12_v1.jsonl is not its pinned bytes ({got} vs {want})")
            ledger["human"]["carry12_v1_sha256"] = got
        ledger.setdefault("steps", {})["decontam"] = decontam.run(allf, clean, os.path.join(out, "human_decontam.json"),
                                                                  dc=dc)
        ledger["human"]["carry5_v1_sha256"] = carry5_sha
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
    if a.measure_only:
        with open(ledger_path + ".tmp", "w", encoding="utf-8", newline="\n") as f:
            json.dump(ledger, f, indent=1)
        os.replace(ledger_path + ".tmp", ledger_path)
        print(json.dumps({"human_pt_bytes": H, "split": ledger.get("human", {}).get("split")}))
        return 0

    # 4. SFT set: one row per dialogue, every assistant turn trained, shuffled.
    sft = os.path.join(out, "sft_human.jsonl")
    sft_long = os.path.join(out, f"sft_human_long{long_copies}.jsonl") if long_copies else None
    raw = os.path.join(out, "sft_raw.jsonl")
    if not os.path.exists(sft) or (sft_long and not os.path.exists(sft_long)):
        if not os.path.exists(raw):
            sh([sys.executable, "tools/chat_data.py", "sft", sft_dl, "--vocab", a.vocab, "--T", str(a.T),
                "--all-turns", "--out", raw + ".tmp"], ledger, "sft")
            os.replace(raw + ".tmp", raw)
        if not os.path.exists(sft):
            sh([sys.executable, "tools/chat_data.py", "mix", raw, "--seed", str(a.seed), "--out", sft + ".tmp"],
               ledger, "sft_shuffle")
            os.replace(sft + ".tmp", sft)
        if sft_long and not os.path.exists(sft_long):
            ledger["long_sft"] = long_rows(raw, os.path.join(out, "sft_long_raw.jsonl"), long_min, long_copies)
            sh([sys.executable, "tools/chat_data.py", "mix", os.path.join(out, "sft_long_raw.jsonl"), "--seed",
                str(a.seed), "--out", sft_long + ".tmp"], ledger, "sft_long_shuffle")
            os.replace(sft_long + ".tmp", sft_long)
            os.remove(os.path.join(out, "sft_long_raw.jsonl"))
        os.remove(raw)
    ledger.setdefault("outputs", {})["sft_human.jsonl"] = {"bytes": os.path.getsize(sft), "sha256": C.sha256_file(sft)}
    if sft_long:
        ledger["outputs"][os.path.basename(sft_long)] = {"bytes": os.path.getsize(sft_long),
                                                         "sha256": C.sha256_file(sft_long)}

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
