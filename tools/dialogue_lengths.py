#!/usr/bin/env python3
"""Dialogue lengths of the pinned human corpora, in turns and tokens (experiments/tiny_chat_long_2026-10).

    python tools/dialogue_lengths.py --vocab V.gguf --out REPORT.json [--state STATE.jsonl] CORPUS ...

CORPUS is a human_<source>_v1/dialogues.jsonl. Each dialogue is normalised as the tiny-chat data job does
(tools/colab_chat/tiny_chat_data.py `alternate`: user/assistant alternation from a user turn, a trailing
user turn dropped, a broken dialogue dropped and counted). A TURN is one exchange (a user message and the
assistant reply), as Carry-5 and Carry-12 count turns. Tokens are the 16k Llama 3-scheme vocabulary in the
chat template: an exchange costs <|user|> + U + <|assistant|> + A + <|endoftext|> = 3 + |U| + |A| tokens,
the cost tools/chat_data.py `_sized` uses, so "fits T" here is chat_data's rule (`all_turns`: the longest
prefix of whole exchanges within T + 1 tokens).

Per corpus: the distribution of turns and tokens per dialogue; how many dialogues reach 12, 14 and 20
turns; whether a whole dialogue, and its first 12 turns, fit T = 512, 1024 and 2048; how many turns the
SFT rule keeps at each T (and how many of them are turn 10 or later: the long-carry training signal); the
reply lengths; `long_gap_echo` (a word-overlap proxy: how often turns 10+ revisit content last seen in
turns 1-4); and the same for the dialogues that the data job's sha256 split puts in the SFT 10%
(`_common.dialogue_key` bucket 5-14; computed before decontamination, which drops 52 of 170,572).

Light (one pass, the HF tokeniser's own threads, capped by scripts/cpu_budget.py); finished corpora are
appended to --state (JSONL) and skipped on a rerun.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "corpus", "fetchers"))
import _common as C  # noqa: E402
import llama3_bpe  # noqa: E402

TS = (512, 1024, 2048)
MARKS = (10, 12, 14, 20)


def alternate(msgs):
    """tools/colab_chat/tiny_chat_data.py `alternate`, verbatim (copied, not imported, so this tool loads
    nothing the running jobs use)."""
    msgs = [m for m in msgs if m.get("role") in ("user", "assistant")]
    if len(msgs) % 2:
        msgs = msgs[:-1]
    if len(msgs) < 2 or any(m["role"] != ("user" if i % 2 == 0 else "assistant") or not m["content"].strip()
                            for i, m in enumerate(msgs)):
        return None
    return msgs


def split_of(msgs):
    b = int(hashlib.sha256(C.dialogue_key(msgs).encode()).hexdigest()[:8], 16) % 100
    return "val" if b < 5 else "sft" if b < 15 else "pt"


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 1)


def dist(xs):
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "mean": round(sum(xs) / len(xs), 1), "p10": pct(xs, 0.1), "p50": pct(xs, 0.5),
            "p90": pct(xs, 0.9), "p99": pct(xs, 0.99), "max": max(xs)}


def kept_turns(cum, T):
    """chat_data all_turns: whole exchanges while the prefix fits T + 1 tokens."""
    k = 0
    while k < len(cum) and cum[k] <= T + 1:
        k += 1
    return k


def summarise(dialogues):
    """dialogues: [(split, [exchange token costs], [user tokens], [reply tokens])]."""
    turns = [len(c) for _, c, _, _ in dialogues]
    toks = [sum(c) for _, c, _, _ in dialogues]
    out = {"dialogues": len(dialogues), "turns": dist(turns), "tokens": dist(toks), "total_tokens": sum(toks),
           "total_turns": sum(turns)}
    for m in MARKS:
        sel = [(c, sum(c)) for _, c, _, _ in dialogues if len(c) >= m]
        out[f"reach_{m}"] = {"dialogues": len(sel), "tokens": sum(t for _, t in sel),
                             f"tokens_to_turn_{m}": dist([sum(c[:m]) for c, _ in sel]),
                             **{f"first_{m}_fit_{T}": sum(1 for c, _ in sel if sum(c[:m]) <= T + 1) for T in TS},
                             **{f"whole_fit_{T}": sum(1 for _, t in sel if t <= T + 1) for T in TS}}
    for T in TS:
        kept, late, late_all, tok_kept = 0, 0, 0, 0
        for _, c, _, _ in dialogues:
            cum, s = [], 0
            for x in c:
                s += x
                cum.append(s)
            k = kept_turns(cum, T)
            kept += k
            late += max(0, k - 9)          # turns 10 .. k kept
            late_all += max(0, len(c) - 9)
            tok_kept += cum[k - 1] if k else 0
        out[f"sft_rule_T{T}"] = {"whole_dialogues_fit": sum(1 for t in toks if t <= T + 1),
                                 "turns_kept": kept, "turns_cut": sum(turns) - kept,
                                 "turns_10_plus_kept": late, "turns_10_plus_total": late_all,
                                 "tokens_kept": tok_kept}
    out["reply_tokens"] = dist([x for _, _, _, r in dialogues for x in r])
    out["user_tokens"] = dist([x for _, _, u, _ in dialogues for x in u])
    return out


def measure(path, tok):
    raw = []
    bad = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            m = alternate(json.loads(line).get("messages", []))
            if m is None:
                bad += 1
                continue
            raw.append(m)
    texts = [x["content"] for m in raw for x in m]
    lens = []
    for i in range(0, len(texts), 20000):
        lens += [len(e.ids) for e in tok.encode_batch(texts[i:i + 20000], add_special_tokens=False)]
    dialogues, p = [], 0
    for m in raw:
        n = len(m)
        ln = lens[p:p + n]
        p += n
        users, replies = ln[0::2], ln[1::2]
        dialogues.append((split_of(m), [3 + u + r for u, r in zip(users, replies)], users, replies))
    rec = {"corpus": path.replace(os.sep, "/"), "sha256": C.sha256_file(path), "dropped_not_alternating": bad,
           "all": summarise(dialogues), "sft_split": summarise([d for d in dialogues if d[0] == "sft"]),
           "long_gap_echo": long_gap_echo(raw)}
    return rec


WORD = re.compile(r"[a-z]{5,}")


def long_gap_echo(raw, df_max=0.005):
    """A crude proxy for long-gap dependencies in natural dialogue: among dialogues of 12+ turns, the share of
    turns 10+ (user message and reply together) that use a content word last seen in turns 1-4, absent from
    every turn in between. Content word: 5+ letters, in at most df_max of the corpus's messages (so common
    words do not count). Word overlap is not proof of a dependency; the number bounds how often natural
    long dialogue even revisits early content."""
    df = {}
    n_msg = 0
    for m in raw:
        for x in m:
            n_msg += 1
            for w in set(WORD.findall(x["content"].lower())):
                df[w] = df.get(w, 0) + 1
    cap = df_max * n_msg
    late = echo = dialogues = 0
    for m in raw:
        turns = [set(w for w in WORD.findall((m[2 * k]["content"] + " " + m[2 * k + 1]["content"]).lower())
                     if df.get(w, 0) <= cap) for k in range(len(m) // 2)]
        if len(turns) < 12:
            continue
        dialogues += 1
        early = set().union(*turns[:4])
        for k in range(9, len(turns)):
            between = set().union(*turns[4:k])
            late += 1
            if (turns[k] & early) - between:
                echo += 1
    return {"dialogues_12_plus": dialogues, "turns_10_plus": late, "turns_echoing_turns_1_4_only": echo,
            "rate": round(echo / late, 4) if late else None, "df_max": df_max}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("corpora", nargs="+")
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--state", default="")
    a = ap.parse_args(argv)
    state = a.state or a.out + ".state.jsonl"
    done = {}
    if os.path.exists(state):
        for line in open(state, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                done[r["corpus"]] = r
    tok = llama3_bpe.from_gguf(a.vocab)
    for path in a.corpora:
        key = path.replace(os.sep, "/")
        if key in done:
            continue
        rec = measure(path, tok)
        with open(state, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rec) + "\n")
        done[key] = rec
        s = rec["all"]
        print(f"{os.path.basename(os.path.dirname(path))}: {s['dialogues']} dialogues, median {s['turns']['p50']} "
              f"turns, {s['reach_12']['dialogues']} reach 12", flush=True)
    recs = [done[p.replace(os.sep, "/")] for p in a.corpora]

    def pooled(part):
        keys = ("dialogues", "total_tokens", "total_turns")
        tot = {k: sum(r[part][k] for r in recs) for k in keys}
        for m in MARKS:
            tot[f"reach_{m}"] = {k: sum(r[part][f"reach_{m}"][k] for r in recs)
                                 for k in recs[0][part][f"reach_{m}"] if not k.startswith("tokens_to")}
        for T in TS:
            tot[f"sft_rule_T{T}"] = {k: sum(r[part][f"sft_rule_T{T}"][k] for r in recs)
                                     for k in recs[0][part][f"sft_rule_T{T}"]}
        return tot

    rep = {"tool": "tools/dialogue_lengths.py", "vocab": a.vocab,
           "vocab_sha256": C.sha256_file(a.vocab),
           "turn": "one exchange (user message + assistant reply)",
           "exchange_tokens": "3 + |U| + |A| (chat template markers), tools/chat_data.py _sized",
           "fits_T": "prefix of whole exchanges within T + 1 tokens (tools/chat_data.py all_turns)",
           "pooled": {"all": pooled("all"), "sft_split": pooled("sft_split")},
           "per_corpus": {os.path.basename(os.path.dirname(r["corpus"])): r for r in recs}}
    tmp = a.out + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rep, f, indent=1)
        f.write("\n")
    os.replace(tmp, a.out)
    print(json.dumps(rep["pooled"]["all"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
