"""Parser rejection controls; stored CPU output is not CUDA evidence."""

import copy
import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from replay_host_helper_gate import replay


@pytest.fixture
def raw():
    return json.loads(
        (Path(__file__).parent / "fixtures/host_helper_gate_cpu.json").read_text()
    )


def test_cpu_full_array_control(raw):
    assert replay(raw)["pass"]


def test_cpu_cannot_substitute_cuda(raw):
    with pytest.raises(ValueError):
        replay(raw, True)


@pytest.mark.parametrize(
    "change",
    [
        "zero_gradient",
        "nan_gradient",
        "nan_output",
        "omit_gradient",
        "wrong_shape",
        "omit_case",
        "duplicate_case",
        "fixture_drift",
        "false_poison_proof",
    ],
)
def test_mutation_refused(raw, change):
    row = raw["cases"][0]
    if change == "zero_gradient":
        row["gradients"]["a"]["values"] = [0] * 6
    elif change == "nan_gradient":
        row["gradients"]["a"]["values"][0] = float("nan")
    elif change == "nan_output":
        row["output"]["values"][0] = float("nan")
    elif change == "omit_gradient":
        del row["gradients"]["a"]
    elif change == "wrong_shape":
        row["output"]["shape"] = [1, 6]
    elif change == "omit_case":
        raw["cases"].pop()
    elif change == "duplicate_case":
        raw["cases"][1] = copy.deepcopy(row)
    elif change == "fixture_drift":
        row["inputs"]["a"]["values"][0] += 1
    else:
        row["stale_gradient_poisoned"] = True
    with pytest.raises((ValueError, KeyError)):
        replay(raw)


@pytest.mark.parametrize(
    "case,name",
    [
        (0, "a"),
        (1, "a"),
        (1, "b"),
        (2, "a"),
        (2, "b"),
        (3, "a"),
        (4, "a"),
        (5, "a"),
        (6, "a"),
        (6, "w"),
        (7, "a"),
        (7, "w"),
        (8, "a"),
        (8, "w"),
    ],
)
@pytest.mark.parametrize("mutation", ["omit", "zero"])
def test_every_gradient_tensor_is_required_nonzero(raw, case, name, mutation):
    item = raw["cases"][case]["gradients"]
    if mutation == "omit":
        del item[name]
    else:
        item[name]["values"] = [0] * len(item[name]["values"])
    with pytest.raises(ValueError):
        replay(raw)
