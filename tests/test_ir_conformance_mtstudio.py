"""The keramos conformance suite (spec/conformance/, spec/IR_SPEC.md section 11) against mtstudio through
tools/ir_conformance_mtstudio.py: every case mtstudio can run passes and none fails. Needs MTSTUDIO_BIN
(ir_eval and expr_eval beside it add the recurrence, digest and expr cases)."""
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import ir_conformance_mtstudio as mt  # noqa: E402


@pytest.mark.skipif(not os.environ.get("MTSTUDIO_BIN"), reason="set MTSTUDIO_BIN")
def test_suite_passes_against_mtstudio():
    counts = mt.ir_conformance.run_suite(mt.MtstudioAdapter())
    assert counts["FAIL"] == 0 and counts["PASS"] > 0, counts
