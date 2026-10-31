"""Semantic landed-audit checks without models, provider calls or altered research."""

import json
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import hs1_parity_reconstruction_audit as audit  # noqa: E402


def fixture():
    request = {"key": "target", "prompt_ids": [2, 3], "continuation_ids": [4]}
    scored = {
        "key": "target",
        "continuation_ids": [4],
        "over_ctx": False,
        "argmax": [4],
        "logprobs": [-0.1],
        "sum": -0.1,
    }
    values = np.zeros((1, 16384), dtype="<f4")
    values[0, 0], values[0, 4], values[0, 5] = 100.0, 2.0, 1.999999
    return [request], [scored], values.tobytes()


def test_valid_raw_argmax_and_chosen_margin():
    requests, rows, raw = fixture()
    result = audit.audit_chunk(requests, rows, rows, raw, {"target"})
    assert result[0]["positions"][0]["reconstructed_scorer_argmax"] == 4
    assert result[0]["positions"][0]["argmax_minus_saved"] == 0.0


@pytest.mark.parametrize(
    "field,value",
    [
        ("key", "other"),
        ("continuation_ids", [5]),
        ("over_ctx", True),
        ("argmax", [5]),
        ("sum", float("nan")),
    ],
)
def test_semantic_mismatch_refused(field, value):
    requests, rows, raw = fixture()
    changed = json.loads(json.dumps(rows))
    changed[0][field] = value
    with pytest.raises(ValueError):
        audit.audit_chunk(requests, changed, changed, raw, {"target"})


def test_tap_numerical_change_refused():
    requests, rows, raw = fixture()
    changed = json.loads(json.dumps(rows))
    changed[0]["logprobs"] = [-0.100001]
    with pytest.raises(ValueError):
        audit.audit_chunk(requests, rows, changed, raw, {"target"})


def test_missing_rows_and_nonfinite_raw_refused():
    requests, rows, raw = fixture()
    with pytest.raises(ValueError):
        audit.audit_chunk(requests, [], [], raw, {"target"})
    values = np.frombuffer(raw, dtype="<f4").copy()
    values[4] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        audit.audit_chunk(requests, rows, rows, values.tobytes(), {"target"})
