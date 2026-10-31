"""Independent reference and rejection controls; synthetic rows are not device evidence."""

import copy
import importlib.util
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "flex_reference", ROOT / "tools/replay_flex_precision.py"
)
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)


def raw_array(value):
    return {"shape": list(value.shape), "values": value.reshape(-1).tolist()}


@pytest.fixture(scope="module")
def synthetic():
    rows = []
    for layers, hidden, precision in ref.SHAPES:
        shapes = ref.expected_shapes(layers, hidden)
        params = {n: ref.fixed_parameter(n, shape) for n, shape in shapes.items()}
        logits, loss, grads = ref.full_model(params, layers, precision == "bf16")
        rows.append(
            {
                "layers": layers,
                "d_ff": hidden,
                "precision": precision,
                "d": 8,
                "heads": 2,
                "vocab": 7,
                "seq_len": 4,
                "ids": ref.IDS,
                "targets": ref.TARGETS,
                "parameters": {n: raw_array(v) for n, v in params.items()},
                "gradients": {n: raw_array(v) for n, v in grads.items()},
                "logits": raw_array(logits),
                "loss": raw_array(loss),
                "profile": {},
                "default_proof": {
                    "legacy_fp32_logits_and_all_gradients_bit_equal": True,
                    "IR_precision_independent": True,
                    "reload_fp32_bit_equal": True,
                },
            }
        )
    return {"pass": True, "actual_cuda": False, "cases": rows}


def test_reference_self_control(synthetic):
    assert ref.replay(synthetic)["pass"]


def test_CPU_cannot_substitute_cuda(synthetic):
    with pytest.raises(ValueError):
        ref.replay(synthetic, True)


@pytest.mark.parametrize(
    "mutation",
    [
        "zero_logits",
        "nan_logits",
        "shape",
        "ids",
        "width",
        "omit_case",
        "duplicate_case",
        "fixture",
        "false_default",
        "false_reload",
        "backend",
    ],
)
def test_registered_contracts_reject(synthetic, mutation):
    doc = copy.deepcopy(synthetic)
    row = doc["cases"][1]
    if mutation == "zero_logits":
        row["logits"]["values"] = [0] * 56
    elif mutation == "nan_logits":
        row["logits"]["values"][0] = float("nan")
    elif mutation == "shape":
        row["logits"]["shape"] = [7, 8]
    elif mutation == "ids":
        row["ids"] = [1] * 8
    elif mutation == "width":
        row["d_ff"] = 13
    elif mutation == "omit_case":
        doc["cases"].pop()
    elif mutation == "duplicate_case":
        doc["cases"][2] = copy.deepcopy(row)
    elif mutation == "fixture":
        row["parameters"]["head.weight"]["values"][0] += 1 / 64
    elif mutation == "false_default":
        row["default_proof"]["legacy_fp32_logits_and_all_gradients_bit_equal"] = False
    elif mutation == "false_reload":
        row["default_proof"]["reload_fp32_bit_equal"] = False
    else:
        doc["actual_cuda"] = True
    with pytest.raises(ValueError):
        ref.replay(doc)


GRADIENT_CASES = [
    (i, name)
    for i, (layers, hidden, _) in enumerate(ref.SHAPES)
    for name in ref.expected_shapes(layers, hidden)
]


@pytest.mark.parametrize("case,name", GRADIENT_CASES)
@pytest.mark.parametrize("mutation", ["omit", "zero", "nan"])
def test_every_full_gradient_tensor_required(synthetic, case, name, mutation):
    doc = copy.deepcopy(synthetic)
    values = doc["cases"][case]["gradients"]
    if mutation == "omit":
        del values[name]
    elif mutation == "zero":
        values[name]["values"] = [0] * len(values[name]["values"])
    else:
        values[name]["values"][0] = float("nan")
    with pytest.raises(ValueError):
        ref.replay(doc)


@pytest.mark.parametrize(
    "mutation", ["missing", "too_few", "boolean", "FP32_contamination"]
)
def test_future_cuda_six_gemm_profile_required(synthetic, mutation):
    # Only a parser unit control; changing identities never creates GPU evidence.
    doc = copy.deepcopy(synthetic)
    doc["actual_cuda"] = True
    for row in doc["cases"]:
        row["profile"] = {
            "gemm_bf16": row["layers"] * 12 if row["precision"] == "bf16" else 0
        }
    if mutation == "missing":
        doc["cases"][1]["profile"] = {}
    elif mutation == "too_few":
        doc["cases"][1]["profile"]["gemm_bf16"] -= 1
    elif mutation == "boolean":
        doc["cases"][0]["profile"]["gemm_bf16"] = False
    else:
        doc["cases"][0]["profile"]["gemm_bf16"] = 1
    with pytest.raises(ValueError):
        ref.replay(doc, True)


def test_independent_backward_engine_binary64_finite_differences(monkeypatch):
    # FP32-only reference promoted to binary64 for derivative validation. This
    # is not a finite difference of discontinuous BF16 bit rounding.
    monkeypatch.setattr(ref, "f32", lambda x: np.asarray(x, dtype=np.float64))
    parameters = {
        n: ref.fixed_parameter(n, s) for n, s in ref.expected_shapes(1, 12).items()
    }
    _, _, grads = ref.full_model(parameters, 1, False)
    h = 1e-5
    for name, value in parameters.items():
        # Every tensor, three positions, including both embedding tables,
        # both normalizations, all projections, FFN, final norm and head.
        for flat in sorted({0, value.size // 2, value.size - 1}):
            index = np.unravel_index(flat, value.shape)
            original = value[index]
            value[index] = original + h
            plus = float(ref.full_model(parameters, 1, False)[1][0, 0])
            value[index] = original - h
            minus = float(ref.full_model(parameters, 1, False)[1][0, 0])
            value[index] = original
            fd = (plus - minus) / (2 * h)
            assert abs(fd - grads[name][index]) / (1 + abs(fd)) < 1e-7, (
                name,
                index,
                fd,
                grads[name][index],
            )


@pytest.mark.parametrize("mutation", ["bool_geometry", "bool_shape", "bool_token"])
def test_registered_integer_types_reject_bool(synthetic, mutation):
    doc = copy.deepcopy(synthetic)
    row = doc["cases"][0]
    if mutation == "bool_geometry":
        row["layers"] = True
    elif mutation == "bool_shape":
        row["parameters"]["ln_f.weight"]["shape"] = [True, 8]
    else:
        row["ids"][-1] = True
    with pytest.raises(ValueError):
        ref.replay(doc)
