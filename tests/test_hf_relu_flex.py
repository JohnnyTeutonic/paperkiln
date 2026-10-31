"""Plain Flex ReLU mapping and exceptional-value diagnostic."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from test_hf_export import flex_gpt2_fixture
import hf_export


@pytest.mark.parametrize("tied", [False, True])
def test_actual_relu_flex_transformers_roundtrip(tmp_path, tied):
    binary = os.environ.get("MTSTUDIO_BIN")
    if not binary:
        pytest.skip("MTSTUDIO_BIN required for actual CPU parity")
    pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM
    from test_thirdparty_ir import spec_of

    root = Path(__file__).resolve().parents[1]
    custom = {
        "d": 16,
        "layers": 2,
        "heads": 2,
        "d_ff": 24,
        "tie_head": tied,
        "norm": "layernorm",
        "activation": "relu",
        "position": "learned",
    }
    spec = spec_of(
        tmp_path, "relu", "gpt2-nano", custom, steps=3, formats=("safetensors", "gguf")
    )
    out = tmp_path / "relu"

    def run(*args):
        p = subprocess.run(
            list(map(str, args)),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=120,
        )
        assert p.returncode == 0, p.stdout + p.stderr

    run(binary, "run", spec)
    events = [
        json.loads(line) for line in (out / "events.jsonl").read_text().splitlines()
    ]
    model_event = next(e for e in events if e.get("event") == "model")
    assert (
        model_event["family"] == "flex"
        and model_event["activation"] == "relu"
        and "gelu_approx" not in model_event
    )
    run(sys.executable, root / "tools/hf_export.py", out)
    config = json.loads((out / "hf/config.json").read_text())
    assert (
        config["activation_function"] == "relu"
        and config["tie_word_embeddings"] is tied
    )
    loaded = AutoModelForCausalLM.from_pretrained(
        out / "hf", attn_implementation="eager"
    )
    assert (
        loaded.get_input_embeddings().weight.data_ptr()
        == loaded.get_output_embeddings().weight.data_ptr()
    ) is tied
    for i, prompt in enumerate(["the cat", "cat .\nthe", "the unknown café"]):
        receipt = tmp_path / f"relu-{i}.json"
        run(
            sys.executable,
            root / "tools/hf_export_verify.py",
            out / "hf",
            out,
            "--spec",
            spec,
            "--prompt",
            prompt,
            "--tokens",
            8,
            "--mtstudio",
            binary,
            "--receipt",
            receipt,
        )
        run(sys.executable, root / "tools/hf_export_verify.py", "--replay", receipt)


@pytest.mark.parametrize(
    "extra",
    [
        {"gelu_approx": "tanh"},
        {"gelu_approx": "none"},
        {"gelu_approx": None},
        {"activation": "silu"},
        {"activation": "expr:x * sigmoid(x)"},
        {"activation_inputs": [{"kind": "up"}]},
        {"position": "sinusoidal"},
        {"norm": "rmsnorm"},
        {"embed": {"expr": "expr:x*2"}},
        {"logits": "expr:x*2"},
        {"residual": "highway"},
        {"norm_placement": {"attn": "post", "ffn": "post"}},
    ],
)
def test_relu_retains_unmapped_variant_refusals(extra):
    model, _ = flex_gpt2_fixture()
    with pytest.raises(ValueError):
        hf_export.validate_gpt2({**model, "activation": "relu", **extra})


def test_native_relu_finite_reference_and_exception_diagnostic(tmp_path):
    import numpy as np
    import torch

    build = os.environ.get("MICROTORCH_CPU_BUILD")
    if not build:
        pytest.skip("MICROTORCH_CPU_BUILD required for native activation diagnostic")
    root = Path(__file__).resolve().parents[1]
    cpp = tmp_path / "relu_diagnostic.cpp"
    cpp.write_text(r"""
#include "microtorch/ops.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
using namespace microtorch;
int main(){
 Matrix x(1,404);
 for(int j=0;j<=400;++j)x(0,j)=static_cast<float>(-20.0+0.1*j);
 x(0,401)=std::numeric_limits<float>::quiet_NaN();x(0,402)=-0.f;x(0,403)=0.f;
 NoGrad ng;auto y=ops::relu(make_var(x));
 std::cout<<std::isnan(y->data(0,401))<<' '<<std::signbit(y->data(0,402))<<'\n';
 std::cout<<std::setprecision(9);
 for(int j=0;j<=400;++j)std::cout<<y->data(0,j)<<' ';
 std::cout<<'\n';
}
""")
    exe = tmp_path / "relu_diagnostic"
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-O2",
            "-fopenmp",
            "-I",
            str(root / "include"),
            "-I",
            str(root / "third_party/transformer_cpp/include"),
            str(cpp),
            str(Path(build) / "libmicrotorch.a"),
            str(Path(build) / "transformer_cpp/libtransformer_core.a"),
            "-o",
            str(exe),
        ],
        check=True,
    )
    p = subprocess.run([str(exe)], capture_output=True, text=True, check=True)
    flags, values = p.stdout.splitlines()
    assert (
        flags == "0 0"
    )  # Source masks NaN and clears negative zero; diagnostic, not parity.
    torch_edge = torch.relu(torch.tensor([float("nan"), -0.0]))
    assert torch.isnan(torch_edge[0]) and torch.signbit(torch_edge[1])
    x = np.array([-20.0 + 0.1 * j for j in range(401)], dtype=np.float32)
    reference = np.maximum(0.0, x.astype(np.float64)).astype(np.float32)
    actual = np.fromstring(values, sep=" ", dtype=np.float32)
    np.testing.assert_array_equal(actual, reference)
    np.testing.assert_array_equal(torch.relu(torch.from_numpy(x)).numpy(), reference)
