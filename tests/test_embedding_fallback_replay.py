"""Offline mutation fixtures use synthetic CUDA labels; no device evidence is invented."""

import copy
import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "embedding_replay", ROOT / "tools/replay_embedding_fallback.py"
)
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
CPU = r.load(ROOT / "docs/receipts/embedding_fallback_20261031/cpu.jsonl")


def gpu_rows():
    rows = copy.deepcopy(CPU)
    for row in rows:
        row["actual_cuda"] = True
        if "case" in row:
            row["stale_operand_proved"] = row["host_poison"] = (
                row["computed_table"] and row["deferred"]
            )
            row["ops_enabled"] = row["case"] == "deferred_device"
    return rows


def test_frozen_cpu_and_synthetic_label_control():
    assert r.replay(CPU)["cases"] == 7
    assert r.replay(gpu_rows(), True)["cases"] == 7


@pytest.mark.parametrize(
    "name",
    [
        "empty_ids",
        "negative_id",
        "large_id",
        "deferred_device",
        "write_through_fallback",
        "leaf_fallback",
        "all",
    ],
)
def test_old_failure_mode_cannot_attest_controls(name):
    rows = gpu_rows()
    if name != "all":
        rows = [next(x for x in rows if x.get("case") == name), dict(rows[-1], cases=1)]
    with pytest.raises(ValueError):
        r.replay(rows, True, name, True)


def test_cpu_cannot_attest_old_cuda_failure():
    with pytest.raises(ValueError):
        r.replay([CPU[0], dict(CPU[-1], cases=1)], False, "deferred_fallback", True)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate",
        "shape",
        "input",
        "upstream",
        "nan_output",
        "nan_grad",
        "zero_output",
        "zero_grad",
        "backend",
        "mode",
        "bounds",
        "empty",
    ],
)
def test_mutations_refuse(mutation):
    rows = gpu_rows()
    if mutation == "missing":
        rows.pop(1)
    elif mutation == "duplicate":
        rows[1] = copy.deepcopy(rows[0])
    elif mutation == "shape":
        rows[0]["output_shape"] = [1, 9]
    elif mutation == "input":
        rows[0]["input"] = [0] * 12
    elif mutation == "upstream":
        rows[0]["upstream"] = [0] * 9
    elif mutation == "nan_output":
        rows[0]["output"][0] = float("nan")
    elif mutation == "nan_grad":
        rows[0]["input_gradient"][0] = float("nan")
    elif mutation == "zero_output":
        rows[0]["output"] = [0] * 9
    elif mutation == "zero_grad":
        rows[0]["input_gradient"] = [0] * 12
    elif mutation == "backend":
        rows[0]["actual_cuda"] = False
    elif mutation == "mode":
        rows[0]["ops_enabled"] = True
    elif mutation == "bounds":
        rows[5]["bounds_rejected"] = False
    elif mutation == "empty":
        rows[4]["empty_rejected"] = False
    with pytest.raises(ValueError):
        r.replay(rows, True)


def test_synthetic_old_numeric_failure_is_specific():
    rows = gpu_rows()
    bad = rows[0]
    bad["output"] = [0] * 9
    bad["pass"] = False
    bad["max_abs_error"] = 0.75
    summary = dict(rows[-1], cases=1, pass_=False)
    summary["pass"] = False
    result = r.replay([bad, summary], True, "deferred_fallback", True)
    assert result["old_numeric_failure_verified"] is True
