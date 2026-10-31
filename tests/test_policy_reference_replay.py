"""CPU controls and synthetic parser negatives; no actual CUDA outcomes."""

import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "replay", ROOT / "tools/replay_policy_reference.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
CASES = ["mixed", "policy_only", "kl_only"]


def raw(variant, case):
    return json.loads(
        (
            ROOT
            / "docs/receipts/policy_reference_gate_20261101/cpu"
            / (variant + "_" + case + "_cpu.json")
        ).read_text()
    )


def synthetic(variant, case):
    d = raw(variant, case)
    d.update(
        actual_cuda=True,
        poisoned=True,
        reference_stale_before=True,
        reference_stale_after_op=variant == "old",
        profile={"gemm": 1},
    )
    if variant == "old":
        loss, g = m.reference(d["logits"]["values"], [9.0] * 6, d["w"], d["k"])
        d["loss"]["values"] = loss
        d["gradient"]["values"] = g
    return d


@pytest.mark.parametrize("variant", ["old", "fixed"])
@pytest.mark.parametrize("case", CASES)
def test_cpu(variant, case):
    m.replay(raw(variant, case), variant, case)


@pytest.mark.parametrize("variant", ["old", "fixed"])
@pytest.mark.parametrize("case", CASES)
def test_synthetic_cuda_parser(variant, case):
    m.replay(synthetic(variant, case), variant, case, True)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("attack", ["zero", "nan", "shape", "missing"])
def test_gradient_tamper(case, attack):
    d = raw("fixed", case)
    if attack == "zero":
        d["gradient"]["values"] = [0.0] * 6
    elif attack == "nan":
        d["gradient"]["values"][0] = float("nan")
    elif attack == "shape":
        d["gradient"]["shape"] = [3, 2]
    else:
        del d["gradient"]
    with pytest.raises((ValueError, KeyError)):
        m.replay(d, "fixed", case)


@pytest.mark.parametrize("case", ["mixed", "kl_only"])
@pytest.mark.parametrize(
    "attack",
    [
        "correct_result",
        "arbitrary_wrong",
        "host_repaired",
        "no_poison",
        "backend",
        "bool_profile",
    ],
)
def test_causal_old_proof(case, attack):
    d = synthetic("old", case)
    if attack == "correct_result":
        d["loss"] = raw("old", case)["loss"]
        d["gradient"] = raw("old", case)["gradient"]
    elif attack == "arbitrary_wrong":
        d["loss"]["values"][0] += 1.0
    elif attack == "host_repaired":
        d["reference_stale_after_op"] = False
    elif attack == "no_poison":
        d["poisoned"] = False
    elif attack == "backend":
        d["actual_cuda"] = 1
    else:
        d["profile"] = {"gemm": True}
    with pytest.raises(ValueError):
        m.replay(d, "old", case, True)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("attack", ["fixture", "stale", "target", "weight"])
def test_fixed_provenance(case, attack):
    d = synthetic("fixed", case)
    if attack == "fixture":
        d["reference"]["values"][0] += 0.01
    elif attack == "stale":
        d["reference_stale_after_op"] = True
    elif attack == "target":
        d["targets"] = [0, 0]
    else:
        d["w"] = [1.0, 1.0]
    with pytest.raises(ValueError):
        m.replay(d, "fixed", case, True)


def test_exact_two_line_patch():
    old = (ROOT / "tests/policy_reference_baseline.cpp").read_text()
    fixed = (ROOT / "src/ops.cpp").read_text()
    for line in [
        "    device::materialize(ref_logp);\n",
        '    MT_DEVCHECK_HOST_READ(ref_logp, "policy_gradient_kl reference copy");\n',
    ]:
        assert fixed.count(line) == 1
        fixed = fixed.replace(line, "")
    assert fixed == old
