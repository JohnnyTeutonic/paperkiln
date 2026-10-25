"""ops::depth_attend (llama_attnres's depth attention) against the anchor, python/attn_res_reference.py.

test_llama_attnres (a C++ test binary) writes fixed cases with ATTNRES_REF_DUMP=FILE: S sources [T, d], a
pseudo-query w [1, d], a cotangent, the mixture scale (1, or S as under scale "sources"), the fused op's output
and its gradients with respect to every source and the query (loss = mean(out * cotangent)). Here the same
inputs go through the reference's own _depth_attend (Eq. 9: RMSNormed keys, raw values, softmax over the
source list) in float64 with torch autograd, the reference's key epsilon set to mtstudio's (1e-5), and the
product scaled as llama_attnres does. The fp32 op must match to rounding: output and every gradient within
2e-5 of the largest reference magnitude (fp32 sums over up to S x d = 10,560 terms at the arm's width; measured
worst 7.1e-6, a source gradient at S 33, d 320; the cases' worst is printed).

Needs torch and a built test_llama_attnres (ATTNRES_TEST_BIN, or ./build/test_llama_attnres); skipped otherwise.
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
BIN = os.environ.get("ATTNRES_TEST_BIN") or str(ROOT / "build" / "test_llama_attnres")
torch = pytest.importorskip("torch")
needs_bin = pytest.mark.skipif(not Path(BIN).exists(), reason="set ATTNRES_TEST_BIN to a built test_llama_attnres")
TOL = 2e-5


def reference():
    spec = importlib.util.spec_from_file_location("attn_res_reference", ROOT / "python" / "attn_res_reference.py")
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    return ref


@needs_bin
def test_fused_op_matches_the_reference(tmp_path):
    out = tmp_path / "cases.json"
    p = subprocess.run([BIN], env={**os.environ, "ATTNRES_REF_DUMP": str(out)}, capture_output=True, text=True,
                       timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
    cases = json.loads(out.read_text())
    assert len(cases) == 5
    ref = reference()
    worst = 0.0
    for c in cases:
        eps = c["eps"]
        ref._rmsnorm.__defaults__ = (eps,)  # the reference's key epsilon (1e-6) set to mtstudio's
        f64 = dict(dtype=torch.float64)
        src = [torch.tensor(s, **f64, requires_grad=True) for s in c["sources"]]
        w = torch.tensor(c["w"], **f64, requires_grad=True)
        y = ref._depth_attend(w[0], src) * c["scale"]
        cot = torch.tensor(c["cotangent"], **f64)
        (y * cot).mean().backward()

        def rel(got, want):
            got = torch.tensor(got, **f64)
            return float((got - want).abs().max() / max(float(want.abs().max()), 1e-30))
        errs = {"out": rel(c["out"], y.detach()), "grad_w": rel(c["grad_w"], w.grad)}
        errs["grad_sources"] = max(rel(g, s.grad) for g, s in zip(c["grad_sources"], src))
        print(f"S {c['S']}, T {c['T']}, d {c['d']}, scale {c['scale']:g}: " +
              ", ".join(f"{k} {v:.2e}" for k, v in errs.items()))
        worst = max(worst, *errs.values())
    print(f"worst relative error against the reference: {worst:.2e}")
    assert worst < TOL, worst
