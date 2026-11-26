"""tools/colab_chat/sat_windows.py, SAT's home replay of the trainer's window draws, against the trainer's own
MICROTORCH_LOG_WINDOWS=1 log on a synthetic corpus with turn markers (state ids inserted). Claude Opus 5.5, 26 Nov 2026.
The trainer part needs MTSTUDIO_BIN; the generator and insertion parts run without it."""
import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools" / "colab_chat"))
import sat_windows as W  # noqa: E402
import test_sat_targets as S  # noqa: E402

needs_bin = S.needs_bin


def test_numpy_mt19937_is_std_mt19937():
    # The standard's check value: the 10,000th output of a default-constructed std::mt19937 (seed 5489).
    assert int(W.mt19937_raw(5489, 10000)[-1]) == 4123659995
    # The trainer's seeds are 123 + 1000003 * seed, taken mod 2^32 by std::mt19937's constructor.
    assert W.mt19937_raw(123 + 1000003 * 1, 3).dtype == np.uint64


@pytest.mark.parametrize("seed,chunk", [(0, 1 << 26), (1, 37), (2, 1)])
def test_insertion_runs_match_state_tokens(seed, chunk, monkeypatch, tmp_path):
    monkeypatch.setattr(W, "CHUNK", chunk)                  # the chunked scan, at chunk edges too
    raw = S.main_raw(400, seed)
    S.write_ids(tmp_path / "r.ids", raw)
    assert W.read_ids(tmp_path / "r.ids").tolist() == raw
    post, pos = S.post_positions(raw)
    m = W.insert_map(np.array(raw), S.K, S.EOT, S.USER, S.ASST)
    assert m.n_post == len(post)
    assert [m.post(r) for r in range(len(raw))] == pos
    r = random.Random(seed)
    for p in [r.randrange(len(post)) for _ in range(200)] + [0, len(post) - 1]:
        want = next((i for i, q in enumerate(pos) if q >= p), len(raw))
        assert m.raw_from(p) == want


@needs_bin
@pytest.mark.parametrize("twins", [True, False])
def test_replay_equals_the_trainers_windows(data, twins):
    """Every non-twin row of steps 1 .. cut_step - 1: the same draw and offset as the trainer logged, and the raw span
    the replay returns is exactly the raw text the trainer's window [at, at + T] holds (the rest are state ids or
    inserted <|endoftext|>s). Twin rows are absent from the replay and are the logged twin rows."""
    tmp, info = data
    out = S.full(tmp, "sat_full" if twins else "sat_notwins", S.sat_cfg(tmp, twins=twins))
    raw = np.array(S.SL_read_ids(tmp / "main.ids"))
    post, pos = S.post_positions(raw.tolist())
    imap = W.insert_map(raw, S.K, S.EOT, S.USER, S.ASST)
    wins, meta = W.replay(len(raw), imap, S.T, 3, S.BATCH, S.ACCUM, S.CUT, S.EVERY if twins else 0)
    assert meta["tokens_after"] == info["n_post"] and meta["val_start"] == info["val_start"]
    logged = {}
    for w in S.evs(out, "windows"):
        assert w["lim"] == meta["limit"] or w["step"] > S.CUT
        for b in range(S.BATCH):
            logged[(w["step"], w["micro"], b)] = (w["raw"][b], w["at"][b], w["twin"][b])
    want_keys = {(s, k, b) for s in range(1, S.CUT) for k in range(S.ACCUM) for b in range(S.BATCH)}
    got = {(x["step"], x["micro"], x["b"]): x for x in wins}
    twin_keys = {k for k in want_keys if logged[k][2]}
    assert set(got) == want_keys - twin_keys and (bool(twin_keys) == twins)
    posset = {p: r for r, p in enumerate(pos)}
    for key, x in got.items():
        d, at, tw = logged[key]
        assert (x["draw"], x["at"], tw) == (d, at, 0)
        held = [posset[p] for p in range(at, at + S.T + 1) if p in posset]
        assert held == list(range(x["raw_lo"], x["raw_hi"]))
        assert all(post[p] >= S.NV or post[p] == S.EOT for p in range(at, at + S.T + 1) if p not in posset)


def step_sums(events, keys):
    """Per-step symbol_targets counts over steps 1 .. cut_step - 1, de-duplicated by step (a restarted run's come last)."""
    by = {}
    for e in events:
        if e.get("event") == "step" and e["step"] < S.CUT:
            by[e["step"]] = e["symbol_targets"]
    assert sorted(by) == list(range(1, S.CUT))
    return {k: sum(v[k] for v in by.values()) for k in keys}


@needs_bin
def test_natural_counts_equal_the_replays_expectation(data):
    """Explore 6, GPT 5 and 9: the trainer's natural targets, counted from the rows it built, summed over steps
    1 .. cut_step - 1, equal exactly what sat_windows.py's replay and the label file predict; twin rows sum to the
    schedule; and a run ended while a checkpoint was being written (so steps are logged twice) gives the same sums."""
    tmp, _ = data
    out = S.full(tmp, "sat_full", S.sat_cfg(tmp))
    raw = np.array(S.SL_read_ids(tmp / "main.ids"))
    imap = W.insert_map(raw, S.K, S.EOT, S.USER, S.ASST)
    wins, _ = W.replay(len(raw), imap, S.T, 3, S.BATCH, S.ACCUM, S.CUT, S.EVERY)
    _, spans = S.SL.read(tmp / "A.satl")
    index = W.SpanIndex(spans)
    want = W.expected_counts(index, imap, wins, S.T)
    assert want["natural_pointer_targets"] > 0 and want["natural_pointers_dropped"] > 0
    got = step_sums(S.evs(out, "step"), list(want))
    assert got == want
    tw = step_sums(S.evs(out, "step"), ["twin_rows"])["twin_rows"]
    assert tw == -(-(S.CUT - 1) * S.BATCH * S.ACCUM // S.EVERY) == len(range(0, (S.CUT - 1) * S.BATCH * S.ACCUM, S.EVERY))
    # window_targets against the independent label mapping of test_sat_targets (tools/state_tokens.py's insertion).
    _, pos = S.post_positions(raw.tolist())
    for w in wins[:20]:
        lab, ptr, dropped = S.expected_window_labels(spans, pos, w["at"], 0)
        assert W.window_targets(index, imap, w["at"], S.T) == (sum(v[2] for v in lab.values()), len(ptr), dropped)
    # Ended while the checkpoint at step 8 was being written: it resumes from 4 and logs 5 to 8 again. The stream a
    # reader stitches across the restart holds those steps twice; de-duplicated by step, the sums are unchanged.
    cut = tmp / "counts_resume"
    p = S.spec(tmp, cut, S.sat_cfg(tmp))
    assert S.run(p, MICROTORCH_CKPT_CRASH="mkdir:8").returncode != 0
    before = (cut / "events.jsonl").read_text()
    assert S.run(p).returncode == 0
    stitched = S.stitched_events(before, (cut / "events.jsonl").read_text())
    logged = [e["step"] for e in stitched if e.get("event") == "step"]
    assert len(logged) > len(set(logged))
    assert step_sums(stitched, list(want) + ["twin_rows"]) == {**want, "twin_rows": tw}
    assert step_sums(S.evs(cut, "step"), list(want) + ["twin_rows"]) == {**want, "twin_rows": tw}


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("satw")
    return tmp, S.setup(tmp)
