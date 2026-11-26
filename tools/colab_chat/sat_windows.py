#!/usr/bin/env python3
"""The trainer's window draws replayed at home, for SAT's label build (the registration's section 4,
AI_ML/chat_screens_claude/base2/token_ladder/symbols_as_targets/REGISTRATION.md in the author's research tree).
Claude Opus 5.5, 26 Nov 2026.

tools/mtstudio.cpp, for an lm run: std::mt19937 rng(123 + 1000003 * seed); each training row takes one draw, in the
order step 1 .. steps, micro-step 0 .. accum - 1, row 0 .. batch - 1; the window's offset is draw % limit, where
limit = val_start - T - 1 and val_start = n - n / 20, n being the main corpus's length after the state-token
insertion; a window is the post-insertion ids at .. at + T (T + 1 tokens: inputs and their targets). With
train.symbol_targets twins, a row whose global index g = (step - 1) * batch * accum + micro * batch + b has
g % every == 0 before cut_step reads the twin corpus instead (its draw is still taken). replay() returns, for steps
1 .. cut_step - 1, every non-twin row's window and the raw (pre-insertion) positions it covers, so the tagger can
label exactly the text those windows read. tests/test_sat_windows.py checks it against the trainer's
MICROTORCH_LOG_WINDOWS=1 log.

numpy's MT19937 with legacy seeding yields std::mt19937's raw 32-bit outputs (mt19937_raw; tested against the
standard's 10,000th output for the default seed, 4123659995).

The state-token insertion is computed as runs (a raw position and the ids inserted before it) from the special
tokens alone, so a corpus of billions of ids needs no second copy: insertion_runs() follows nn::insert_state_ids
(and tools/state_tokens.py), which the test compares on a synthetic corpus.
"""
import argparse
import json
import sys

import numpy as np

NONE, USER, REPLY = 0, 1, 2
CHUNK = 1 << 26


def mt19937_raw(seed, n):
    """The first n outputs of std::mt19937(seed)."""
    bg = np.random.MT19937(0)
    bg._legacy_seeding(int(seed) & 0xFFFFFFFF)
    return bg.random_raw(int(n)).astype(np.uint64)


def insertion_runs(raw, k, eot, user, assistant):
    """(at, cum): the raw positions before which ids are inserted, sorted, and the ids inserted up to and including
    each, as nn::insert_state_ids inserts them (k state ids after an <|endoftext|> that closes a reply; an
    <|endoftext|> and k state ids before a <|user|> that arrives while a reply is open)."""
    raw = raw if isinstance(raw, np.ndarray) else np.asarray(raw)
    idx_parts, tok_parts = [], []
    for c0 in range(0, len(raw), CHUNK):           # in chunks, so a memory-mapped corpus of billions of ids fits
        c = np.asarray(raw[c0:c0 + CHUNK])
        i = np.flatnonzero((c == eot) | (c == user) | (c == assistant))
        idx_parts.append(i + c0)
        tok_parts.append(c[i])
    idx = np.concatenate(idx_parts) if idx_parts else np.zeros(0, np.int64)
    toks = np.concatenate(tok_parts) if tok_parts else np.zeros(0, np.int64)
    phase = NONE
    runs = {}
    for i, t in zip(idx.tolist(), toks.tolist()):
        if t == user:
            if phase == REPLY:
                runs[i] = runs.get(i, 0) + 1 + k
            phase = USER
        elif t == assistant:
            if phase == USER:
                phase = REPLY
        else:  # eot
            if phase == REPLY:
                runs[i + 1] = runs.get(i + 1, 0) + k
            phase = NONE
    at = np.array(sorted(runs), dtype=np.int64)
    cum = np.cumsum(np.array([runs[a] for a in at.tolist()], dtype=np.int64))
    return at, cum


class InsertMap:
    """Raw positions to post-insertion positions, as the trainer's InsertMap."""

    def __init__(self, n_raw, at, cum):
        self.n_raw, self.at, self.cum = int(n_raw), at, cum
        self.inserted = int(cum[-1]) if len(cum) else 0

    @property
    def n_post(self):
        return self.n_raw + self.inserted

    def post(self, r):
        j = int(np.searchsorted(self.at, r, side="right"))
        return int(r) + (int(self.cum[j - 1]) if j else 0)

    def raw_from(self, p):
        """The first raw position whose post-insertion position is >= p (n_raw if none)."""
        lo, hi = 0, self.n_raw
        while lo < hi:
            mid = (lo + hi) // 2
            if self.post(mid) >= p:
                hi = mid
            else:
                lo = mid + 1
        return lo


def insert_map(raw, k=0, eot=-1, user=-1, assistant=-1):
    if k == 0:
        return InsertMap(len(raw), np.zeros(0, np.int64), np.zeros(0, np.int64))
    at, cum = insertion_runs(raw, k, eot, user, assistant)
    return InsertMap(len(raw), at, cum)


def replay(n_raw, imap, T, seed, batch, accum, cut_step, twin_every=0):
    """Every non-twin row's window of steps 1 .. cut_step - 1: dicts {step, micro, b, g, draw, at, raw_lo, raw_hi},
    raw positions [raw_lo, raw_hi) being those whose post-insertion position falls in [at, at + T]."""
    n = imap.n_post
    val_start = n - n // 20
    limit = val_start - T - 1
    rows = batch * accum
    draws = mt19937_raw(123 + 1000003 * seed, (cut_step - 1) * rows)
    out = []
    for g, d in enumerate(draws.tolist()):
        if twin_every and g % twin_every == 0:
            continue
        step, rem = divmod(g, rows)
        micro, b = divmod(rem, batch)
        at = d % limit
        out.append({"step": step + 1, "micro": micro, "b": b, "g": g, "draw": d, "at": at,
                    "raw_lo": imap.raw_from(at), "raw_hi": imap.raw_from(at + T + 1)})
    return out, {"tokens_after": n, "val_start": val_start, "limit": limit}


def read_ids(path):
    """A PKIDS001 file's ids (as the trainer's read_corpus reads .ids), memory-mapped read-only."""
    with open(path, "rb") as f:
        head = f.read(24)
    if head[:8] != b"PKIDS001":
        raise ValueError(f"{path}: not a PKIDS001 id file")
    width = int.from_bytes(head[8:12], "little")
    count = int.from_bytes(head[16:24], "little")
    dt = {2: "<u2", 4: "<u4"}[width]
    return np.memmap(path, dtype=dt, mode="r", offset=24, shape=(count,))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ids")
    ap.add_argument("--T", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--accum", type=int, required=True)
    ap.add_argument("--cut-step", type=int, required=True)
    ap.add_argument("--twin-every", type=int, default=0)
    ap.add_argument("--state-k", type=int, default=0)
    ap.add_argument("--eot", type=int, default=-1)
    ap.add_argument("--user", type=int, default=-1)
    ap.add_argument("--assistant", type=int, default=-1)
    ap.add_argument("--out", required=True, help="JSONL of the windows")
    a = ap.parse_args(argv)
    raw = read_ids(a.ids)
    imap = insert_map(raw, a.state_k, a.eot, a.user, a.assistant)
    wins, info = replay(len(raw), imap, a.T, a.seed, a.batch, a.accum, a.cut_step, a.twin_every)
    with open(a.out, "w", encoding="utf-8") as f:
        for w in wins:
            f.write(json.dumps(w) + "\n")
    print(json.dumps({**info, "windows": len(wins), "out": a.out}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
