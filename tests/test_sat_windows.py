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


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("satw")
    return tmp, S.setup(tmp)
