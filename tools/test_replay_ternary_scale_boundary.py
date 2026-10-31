import copy
import json
import pytest
from replay_ternary_scale_boundary import PREFIX, VALUES, replay

FRESH = {"kind": "fresh", "scale": 1.4545455, "zero_scale": 100000}
FIXED = {
    "kind": "deferred",
    "actual_cuda": True,
    "scale": 1.4545455,
    "host_stale_before": True,
    "host_stale_after": False,
    "values": VALUES,
}
OLD = {
    "kind": "deferred",
    "actual_cuda": True,
    "scale": 100000,
    "host_stale_before": True,
    "host_stale_after": True,
    "values": [0] * 6,
}


def save(tmp_path, rows):
    p = tmp_path / "raw.log"
    p.write_text("\n".join(PREFIX + json.dumps(r) for r in rows) + "\n")
    return p


@pytest.mark.parametrize("mode,row,rc", [("old", OLD, 1), ("fixed", FIXED, 0)])
def test_actual_pair(tmp_path, mode, row, rc):
    assert replay(save(tmp_path, [FRESH, row]), mode, True, rc)["verified"]


@pytest.mark.parametrize("mode", ["old", "fixed"])
def test_fresh(tmp_path, mode):
    assert replay(save(tmp_path, [FRESH]), mode, False, 0)["verified"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("actual_cuda", False),
        ("actual_cuda", 1),
        ("scale", float("nan")),
        ("scale", float("inf")),
        ("scale", 0),
        ("scale", True),
        ("values", VALUES[:-1]),
        ("values", VALUES + [0]),
        ("values", [0] * 6),
        ("values", float("nan")),
        ("values", [float("nan")] * 6),
        ("host_stale_after", True),
        ("host_stale_after", 0),
        ("kind", "wrong"),
    ],
)
def test_mutation_rejected(tmp_path, field, value):
    row = copy.deepcopy(FIXED)
    row[field] = value
    with pytest.raises((ValueError, TypeError)):
        replay(save(tmp_path, [FRESH, row]), "fixed", True, 0)


@pytest.mark.parametrize(
    "rows,rc",
    [([FRESH], 0), ([FRESH, FIXED, FIXED], 0), ([FRESH, FIXED], 1), ([FRESH, OLD], 0)],
)
def test_inventory_or_exit_rejected(tmp_path, rows, rc):
    with pytest.raises(ValueError):
        replay(save(tmp_path, rows), "fixed", True, rc)


def test_old_must_reproduce_failure(tmp_path):
    with pytest.raises(ValueError):
        replay(save(tmp_path, [FRESH, FIXED]), "old", True, 1)


def test_duplicate_raw_key_rejected(tmp_path):
    p = save(tmp_path, [FRESH, FIXED])
    p.write_text(
        p.read_text().replace('"scale": 1.4545455', '"scale": 0, "scale": 1.4545455')
    )
    with pytest.raises(ValueError):
        replay(p, "fixed", True, 0)
