"""CPU raw reference and synthetic parser negatives; no CUDA execution."""

import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "replay", ROOT / "tools/replay_checkpoint_copy.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def raw(variant, case):
    return json.loads(
        (
            ROOT
            / "docs/receipts/checkpoint_copy_gate_20261031/cpu"
            / (variant + "_" + case + "_cpu.json")
        ).read_text()
    )


def simulated_gpu(case, old=False):
    d = raw("old" if old else "fixed", case)
    d["actual_cuda"] = True
    d["input_stale_before_checkpoint"] = case == "input"
    d["input_stale_before_backward"] = case == "input"
    for i, o in enumerate(d["observations"]):
        o["input_stale"] = case == "input" and i == 0
        o["output_stale_before_local_materialize"] = True
        o["output_stale_on_return"] = case == "output"
    d["profile"] = {"gemm": 1}
    if old:
        d["pass"] = False
        d["error"] = (
            "DEVCHECK: host read of device-stale tensor at checkpoint " + case + " copy"
        )
        d["observations"] = d["observations"][:1]
        d.pop("gradients")
        d.pop("parameters")
        d.pop("profile")
        if case == "output":
            d.pop("input_stale_before_backward")
    return d


@pytest.mark.parametrize("variant", ["old", "fixed"])
@pytest.mark.parametrize("case", ["output", "input"])
def test_cpu_controls(variant, case):
    m.replay(raw(variant, case), variant, case)


@pytest.mark.parametrize("case", ["output", "input"])
@pytest.mark.parametrize("old", [False, True])
def test_synthetic_cuda_parser_acceptance(case, old):
    m.replay(simulated_gpu(case, old), "old" if old else "fixed", case, True)


@pytest.mark.parametrize(
    "case,name",
    [("output", "x"), ("output", "w"), ("input", "a"), ("input", "b"), ("input", "w")],
)
@pytest.mark.parametrize("attack", ["omit", "zero", "nan", "shape"])
def test_full_gradient_tamper(case, name, attack):
    d = raw("fixed", case)
    if attack == "omit":
        del d["gradients"][name]
    elif attack == "zero":
        d["gradients"][name]["values"] = [0] * len(d["gradients"][name]["values"])
    elif attack == "nan":
        d["gradients"][name]["values"][0] = float("nan")
    else:
        d["gradients"][name]["shape"] = [1, len(d["gradients"][name]["values"])]
    with pytest.raises(ValueError):
        m.replay(d, "fixed", case)


@pytest.mark.parametrize("case", ["output", "input"])
@pytest.mark.parametrize("attack", ["guard", "stale", "phase", "backend", "passed"])
def test_old_failure_proof_tamper(case, attack):
    d = simulated_gpu(case, True)
    if attack == "guard":
        d["error"] = "unrelated failure"
    elif attack == "stale":
        d["observations"][0]["output_stale_before_local_materialize"] = False
    elif attack == "phase":
        d["observations"][0]["grad_enabled"] = True
    elif attack == "backend":
        d["actual_cuda"] = 1
    else:
        d["pass"] = True
    with pytest.raises(ValueError):
        m.replay(d, "old", case, True)


@pytest.mark.parametrize("case", ["output", "input"])
@pytest.mark.parametrize("attack", ["fixture", "profile", "input", "output", "loss"])
def test_fixed_proof_tamper(case, attack):
    d = simulated_gpu(case)
    if attack == "fixture":
        d["parameters"]["w"]["values"][0] += 0.01
    elif attack == "profile":
        d["profile"] = {"gemm": True}
    elif attack == "input":
        d["observations"][1]["input_stale"] = True
    elif attack == "output":
        d["logits"]["values"][0] += 0.01
    else:
        d["loss"]["values"][0] += 0.01
    with pytest.raises(ValueError):
        m.replay(d, "fixed", case, True)


def test_exact_two_line_patch():
    old = (ROOT / "tests/checkpoint_copy_baseline.cpp").read_text()
    fixed = (ROOT / "src/autograd.cpp").read_text()
    for line in [
        "        device::materialize(inner->data);\n",
        "        device::materialize(xin->data);\n",
    ]:
        assert fixed.count(line) == 1
        fixed = fixed.replace(line, "")
    assert old == fixed
