"""Every evaluation set in corpus/eval/ is frozen: it has a pin in
corpus/hashes/ and matches it byte for byte (charter 0001 §9, Phase E)."""
import glob
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "corpus", "fetchers"))
import _common as C  # noqa: E402

SETS = sorted(p for p in glob.glob(os.path.join(ROOT, "corpus", "eval", "*"))
              if p.endswith((".jsonl", ".txt")))


def test_there_are_frozen_sets():
    assert len(SETS) >= 11


@pytest.mark.parametrize("path", SETS, ids=os.path.basename)
def test_set_matches_its_pin(path):
    name = os.path.basename(path)
    eid = os.path.splitext(name)[0]
    want = C.expected_hashes(eid)
    assert name in want, f"{name} has no pin in corpus/hashes/{eid}.sha256"
    assert C.sha256_file(path) == want[name], f"{name} changed after freezing: version it instead"
