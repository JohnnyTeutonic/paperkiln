"""experiments/tiny_chat_2026-10/make_jobs.py: the registered dialogue-level rule and the matched
shapes (PREREGISTRATION.md sections 5-6)."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_jobs", ROOT / "experiments" / "tiny_chat_2026-10" / "make_jobs.py")
mj = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mj)


def test_levels_nominal_when_four_copies_cover_a_mix():
    d = mj.dialogue_levels(int(mj.MIX_BYTES / (4 * 0.95)) + 10)
    assert d["mode"] == "shares" and d["levels"] == [0, 25, 50, 100] and d["e1_share"] == 50


def test_levels_by_copies_when_the_corpus_is_short():
    d = mj.dialogue_levels(110_000_000)  # ~0.13 of a mix per copy
    assert d["mode"] == "copies" and d["levels"] == [0, 13, 26, 52] and d["e1_share"] == 52
    d = mj.dialogue_levels(60_000_000)
    assert d["levels"] == [0, 7, 14, 28] and d["e1_share"] == 28


def test_levels_refuse_a_corpus_too_small_to_separate():
    with pytest.raises(SystemExit):
        mj.dialogue_levels(5_000_000)


def test_shapes_matched_within_one_percent():
    for sid, s in mj.shapes().items():
        target = 20e6 if s["size"] == "20m" else 60e6
        assert abs(s["nonemb"] - target) / target < 0.01, sid
        assert s["d"] % 64 == 0 and s["heads"] * 64 == s["d"] and s["d_ff"] % 32 == 0
