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


class SpanIndex:
    """An SATL file's spans (sat_labels.read's), sorted by raw offset, for window lookups."""

    def __init__(self, spans):
        self.spans = sorted(spans, key=lambda s: s["raw_offset"])
        self.offs = np.array([s["raw_offset"] for s in self.spans], dtype=np.int64)

    def overlapping(self, lo, end):
        """The spans that overlap raw positions [lo, end)."""
        i = int(np.searchsorted(self.offs, lo, side="right"))
        if i > 0 and self.offs[i - 1] + len(self.spans[i - 1]["codes"]) > lo:
            i -= 1
        while i < len(self.spans) and self.offs[i] < end:
            yield self.spans[i]
            i += 1


def window_targets(index, imap, at, T):
    """(role targets, pointer targets, pointers dropped) of the window whose inputs are post-insertion positions
    at .. at + T - 1, by the trainer's rule (sat_window): labelled codes on raw positions inside the window; a pointer
    counts when its position is inside and its gold is too, and is dropped when its gold lies before the window."""
    lo, end = imap.raw_from(at), imap.raw_from(at + T)
    role = ptr = dropped = 0
    if lo >= end:
        return 0, 0, 0
    for s in index.overlapping(lo, end):
        off, codes = s["raw_offset"], s["codes"]
        a, b = max(lo, off) - off, min(end, off + len(codes)) - off
        role += sum(1 for c in codes[a:b] if c & 8)
        for pos, back in s["pointers"]:
            rt = off + pos
            if lo <= rt < end:
                if rt - back < lo:
                    dropped += 1
                else:
                    ptr += 1
    return role, ptr, dropped


def window_nearest(index, imap, at, T):
    """(nearest-right, nearest-wrong) among the window's kept pointer targets (SATW): a target is nearest-right when
    the nearest earlier mention-bit position in the window (the recency rule, row-local as the trainer and the label
    build's receipt compute it) is its gold. Mentions of every span overlapping the window count."""
    lo, end = imap.raw_from(at), imap.raw_from(at + T)
    if lo >= end:
        return 0, 0
    spans = list(index.overlapping(lo, end))
    men = []
    for s in spans:
        off, codes = s["raw_offset"], np.frombuffer(bytes(s["codes"]), np.uint8)
        a, b = max(lo, off) - off, min(end, off + len(codes)) - off
        men.append(off + a + np.flatnonzero(codes[a:b] & 16))
    men = np.concatenate(men) if men else np.zeros(0, np.int64)
    right = wrong = 0
    for s in spans:
        off = s["raw_offset"]
        for pos, back in s["pointers"]:
            rt, rg = off + pos, off + pos - back
            if not (lo <= rt < end) or rg < lo:
                continue                                   # outside the window, or dropped
            i = int(np.searchsorted(men, rt)) - 1          # the nearest mention before the target
            if i >= 0 and int(men[i]) == rg:
                right += 1
            else:
                wrong += 1
    return right, wrong


def expected_counts(index, imap, windows, T, nearest=False):
    """The replay's windows' target counts summed: the expectation for the trainer's natural_role_targets,
    natural_pointer_targets and natural_pointers_dropped over steps 1 .. cut_step - 1 (REGISTRATION section 10, V8);
    with nearest, also natural_pointer_nearest_right and natural_pointer_nearest_wrong (SATW)."""
    tot = [0, 0, 0]
    nr = [0, 0]
    for w in windows:
        for i, v in enumerate(window_targets(index, imap, w["at"], T)):
            tot[i] += v
        if nearest:
            for i, v in enumerate(window_nearest(index, imap, w["at"], T)):
                nr[i] += v
    out = {"natural_role_targets": tot[0], "natural_pointer_targets": tot[1], "natural_pointers_dropped": tot[2],
           "natural_rows": len(windows)}
    if nearest:
        out.update(natural_pointer_nearest_right=nr[0], natural_pointer_nearest_wrong=nr[1])
    return out


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
