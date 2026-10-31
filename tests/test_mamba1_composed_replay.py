"""Parser attacks include zeroing every complete parameter-gradient tensor."""

import copy
import functools
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "mamba_ref", ROOT / "tools/replay_mamba1_composed.py"
)
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)
D = ROOT / "docs/receipts/mamba1_composed_fd_20261031"
ROWS = ref.load(D / "cpu.jsonl")
FIXTURE = json.loads((D / "weights.json").read_text())
ORIGINAL_FD = ref.finite_differences


@functools.lru_cache(None)
def oracle(depth):
    weights = {
        name: np.array(p["values"], dtype=np.float64).reshape(p["shape"])
        for name, p in ROWS[depth - 1]["parameters"].items()
    }
    return ORIGINAL_FD(weights, depth)


@pytest.fixture(autouse=True)
def cached_reference(monkeypatch):
    # Cache only the already frozen weights. Mutated fixture weights reject before this point.
    monkeypatch.setattr(ref, "finite_differences", lambda weights, depth: oracle(depth))


def test_complete():
    result = ref.replay(copy.deepcopy(ROWS), FIXTURE)
    if sum(case["parameter_elements"] for case in result["cases"]) != 1224:
        raise ValueError("inventory")


ZERO_CASES = [(i, name) for i, row in enumerate(ROWS[:2]) for name in row["parameters"]]


@pytest.mark.parametrize("case,name", ZERO_CASES)
def test_each_gradient_tensor_zero_rejects(case, name):
    rows = copy.deepcopy(ROWS)
    rows[case]["parameters"][name]["gradient"] = [0.0] * len(
        rows[case]["parameters"][name]["gradient"]
    )
    with pytest.raises(ValueError, match="gradient bound"):
        ref.replay(rows, FIXTURE)


@pytest.mark.parametrize("kind", range(17))
def test_registered_attacks(kind):
    rows = copy.deepcopy(ROWS)
    first = rows[0]
    name = "layers.0.dt_proj.weight"
    if kind == 0:
        first["depth"] = True
    if kind == 1:
        first["seq_len"] = 8
    if kind == 2:
        first["ids"][1] = True
    if kind == 3:
        first["targets"][0] = 0
    if kind == 4:
        first["parameters"][name]["shape"] = [1, 16]
    if kind == 5:
        first["parameters"]["layers.0.norm.weight"]["shape"] = [True, 8]
    if kind == 6:
        first["parameters"].pop(name)
    if kind == 7:
        first["parameters"][name]["values"][0] += 0.01
    if kind == 8:
        first["parameters"][name]["gradient"][0] = float("nan")
    if kind == 9:
        first["parameters"][name]["gradient"] = [1.0]
    if kind == 10:
        first["logits"][0] += 0.01
    if kind == 11:
        first["loss"] += 0.01
    if kind == 12:
        rows[-1]["pass"] = 1
    if kind == 13:
        rows.pop()
    if kind == 14:
        first["gpu_n"] = {"selective_scan_fwd": 1}
    if kind == 15:
        first["actual_cuda"] = True
    if kind == 16:
        first["parameters"][name]["gradient"][0] = True
    with pytest.raises(ValueError):
        ref.replay(rows, FIXTURE)


def synthetic_cuda():
    rows = copy.deepcopy(ROWS)
    rows[-1]["actual_cuda"] = True
    for depth, row in enumerate(rows[:2], 1):
        row["actual_cuda"] = True
        row["gpu_n"] = {
            "selective_scan_fwd": depth,
            "selective_scan_bwd": depth,
            "conv1d_causal_fwd": depth,
            "conv1d_causal_bwd": 3 * depth,
            "rmsnorm_fwd": depth + 1,
            "rmsnorm_bwd": 2 * (depth + 1),
            "gemm": 12 * depth + 3,
        }
        row["gpu_ms"] = {name: 0.1 for name in row["gpu_n"]}
    return rows


def test_cuda_parser_contract_only():
    ref.replay(synthetic_cuda(), FIXTURE, True)


@pytest.mark.parametrize(
    "op",
    [
        "selective_scan_fwd",
        "selective_scan_bwd",
        "conv1d_causal_fwd",
        "conv1d_causal_bwd",
        "rmsnorm_fwd",
        "rmsnorm_bwd",
        "gemm",
    ],
)
def test_profile_missing_rejects(op):
    rows = synthetic_cuda()
    rows[0]["gpu_n"].pop(op)
    with pytest.raises(ValueError):
        ref.replay(rows, FIXTURE, True)


def test_bool_profile_rejects():
    rows = synthetic_cuda()
    rows[0]["gpu_n"]["selective_scan_fwd"] = True
    with pytest.raises(ValueError):
        ref.replay(rows, FIXTURE, True)


def test_ce_difference_identity():
    rng = np.random.default_rng(19)
    minus = rng.normal(size=(8, 7))
    plus = minus + 0.001 * rng.normal(size=(8, 7))

    def ce(x):
        return float(
            np.mean(np.log(np.exp(x).sum(axis=1)) - x[np.arange(8), ref.TARGETS])
        )

    if abs(ref.ce_difference(plus, minus) - (ce(plus) - ce(minus))) > 1e-14:
        raise ValueError("stable CE equivalence")
