"""Mutation tests use synthetic CUDA labels only to exercise offline rejection paths."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "replay_model", ROOT / "tools/replay_gpt2_precision_gate.py"
)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)
RAW = ROOT / "docs/receipts/gpt2_attention_precision_20261031/model_gate_cpu.json"
CPU = json.loads(RAW.read_text())


def synthetic_cuda():
    r = copy.deepcopy(CPU)
    r["actual_cuda"] = True
    r["bf16_profile_counts"] = {"gemm_bf16": 24}
    r["training_profile_counts"] = {"gemm_bf16": 384}
    return r


def test_independent_cpu_forward_receipt_and_synthetic_identical_parser_fixture():
    assert replay.replay(CPU)["cpu_forward_error"] < 3e-5
    assert replay.replay(CPU, synthetic_cuda())["cpu_cuda_gradient_error"] == 0


@pytest.mark.parametrize("name", sorted(CPU["gradients"]))
def test_every_nonzero_expected_gradient_tensor_cannot_be_zeroed(name):
    r = synthetic_cuda()
    r["gradients"][name]["values"] = [0] * len(r["gradients"][name]["values"])
    with pytest.raises(ValueError):
        replay.replay(CPU, r)


@pytest.mark.parametrize("name", sorted(CPU["gradients"]))
def test_nonfinite_gradient_refused(name):
    r = synthetic_cuda()
    r["gradients"][name]["values"][0] = float("nan")
    with pytest.raises(ValueError):
        replay.replay(CPU, r)


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "missing_both",
        "shape",
        "initial_count",
        "training_count",
        "backend",
        "loss",
        "default",
        "initial_weight",
    ],
)
def test_fixed_inventory_profiler_and_outcome_mutations_refused(change):
    r = synthetic_cuda()
    name = next(iter(r["gradients"]))
    if change == "missing":
        r["gradients"].pop(name)
    elif change == "missing_both":
        r["gradients"].pop(name)
        r["parameters"].pop(name)
    elif change == "shape":
        r["gradients"][name]["cols"] += 1
    elif change == "initial_count":
        r["bf16_profile_counts"]["gemm_bf16"] = 25
    elif change == "training_count":
        r["training_profile_counts"]["gemm_bf16"] = 383
    elif change == "backend":
        r["actual_cuda"] = False
    elif change == "loss":
        r["losses"][0] += 1
    elif change == "default":
        r["default_profile_counts"]["gemm_bf16"] = 1
    else:
        r["parameters"][name]["values"][0] += 1
    with pytest.raises(ValueError):
        replay.replay(CPU, r)


@pytest.mark.parametrize("kind", ["ember", "manifest", "escape", "artifact"])
def test_host_replay_refuses_tampered_binding_before_model_execution(tmp_path, kind):
    import sys

    sys.path.insert(0, str(ROOT / "tools"))
    import replay_gpt2_precision_home as home

    ember = tmp_path / "ember"
    ember.write_bytes(b"synthetic fixture, never executed")
    inputs = tmp_path / "inputs.json"
    inputs.write_text("{}")
    mh = home.sha(inputs)
    summary = {"pass": True, "manifest_sha256": mh, "artifacts": {}}
    (tmp_path / "hardware.log").write_text("NVIDIA L4, synthetic rejection fixture")
    if kind == "escape":
        summary["artifacts"]["../outside"] = "bad"
    elif kind == "artifact":
        (tmp_path / "raw.json").write_text("{}")
        summary["artifacts"]["raw.json"] = "bad"
    (tmp_path / "gate_summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        home.replay(
            tmp_path,
            "bad" if kind == "manifest" else mh,
            ember,
            "bad" if kind == "ember" else home.sha(ember),
        )


def test_vm_manifest_mismatch_cannot_reach_hardware_or_subprocess(
    tmp_path, monkeypatch
):
    vm_spec = importlib.util.spec_from_file_location(
        "vm_model_gate", ROOT / "tools/colab_chat/gpt2_precision_gate.py"
    )
    vm = importlib.util.module_from_spec(vm_spec)
    vm_spec.loader.exec_module(vm)
    (tmp_path / "inputs.json").write_text("{}")

    def forbidden(*a, **kw):
        raise AssertionError("must not dispatch anything")

    monkeypatch.setattr(vm.subprocess, "Popen", forbidden)
    out = tmp_path / "out"
    vm.run(tmp_path, "wrong-frozen-hash", out)
    result = json.loads((out / "gate_summary.json").read_text())
    assert (
        result["pass"] is False
        and result["commands"] == []
        and (out / "out.zip").exists()
    )
