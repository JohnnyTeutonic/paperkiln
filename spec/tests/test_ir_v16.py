"""Revision 1.6 (IR_SPEC 3.8, 5.14, 5.16): the reference against independent float64 statements of the published
mechanisms.

- moe_ffn's router with scoring "sigmoid", selection_bias and groups (DeepSeek-V3, arXiv 2412.19437, eqs. 12-16;
  auxiliary-loss-free load balancing, Wang et al. 2024, arXiv 2408.15664): scores s = sigmoid(u . e), the bias
  enters the top-k selection only, node-limited routing keeps the experts of the top groups by the sum of their
  two largest biased scores, gates are the selected scores normalised.
- YaRN (Peng et al. 2023, arXiv 2309.00071, section 3.2 "NTK-by-parts" and 3.4): tools/gguf_ir.py's divisors
  against the paper's ramp written out here.
- apply: a node skipped in an application leaves its value as the earlier writes left it.
"""
from pathlib import Path
import math
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import gguf_ir  # noqa: E402
import ir_numpy  # noqa: E402

RNG = np.random.default_rng(1606)


def r(*shape, s=1.0):
    return (s * RNG.standard_normal(shape)).astype(np.float32)


def moe64(x, R, G, U, D, k, scoring, bias, groups, renorm):
    x, R, G, U, D = (np.asarray(a, np.float64) for a in (x, R, G, U, D))
    out = np.zeros((x.shape[0], D.shape[1]))
    E = R.shape[0]
    for t in range(x.shape[0]):
        z = R @ x[t]
        s = 1 / (1 + np.exp(-z)) if scoring == "sigmoid" else np.exp(z - z.max()) / np.exp(z - z.max()).sum()
        c = s + (np.asarray(bias, np.float64) if bias is not None else 0.0)
        allowed = set(range(E))
        if groups:
            n, top = groups
            per = E // n
            gscore = [sorted(c[g * per:(g + 1) * per], reverse=True)[:2] for g in range(n)]
            keep = sorted(range(n), key=lambda g: (-sum(gscore[g]), g))[:top]
            allowed = {e for g in keep for e in range(g * per, (g + 1) * per)}
        sel = sorted(sorted(allowed, key=lambda e: (-c[e], e))[:k])
        gates = np.array([s[e] for e in sel])
        if renorm:
            gates = gates / gates.sum()
        for gte, e in zip(gates, sel):
            hg = G[e] @ x[t]
            out[t] += gte * (D[e] @ (hg / (1 + np.exp(-hg)) * (U[e] @ x[t])))
    return out


def test_router_matches_float64():
    worst = 0.0
    for scoring, bias_on, groups, renorm in (("sigmoid", False, None, True), ("sigmoid", True, (2, 1), True),
                                             ("softmax", True, None, False), ("sigmoid", True, (4, 2), False)):
        E, k = 8, 2
        x, R = r(9, 6), r(E, 6)
        G, U, D = r(E, 5, 6), r(E, 5, 6), r(E, 6, 5)
        bias = r(E, s=0.3) if bias_on else None
        node = {"op": "moe_ffn", "in": ["x"], "router": "r", "experts": {"gate": "g", "up": "u", "down": "d"},
                "n_experts": E, "top_k": k, "gate_norm": "renorm" if renorm else "raw", "activation": "silu",
                "scoring": scoring, "out": "y"}
        W = {"r": R, "g": G, "u": U, "d": D}
        if bias_on:
            node["selection_bias"], W["sb"] = "sb", bias
        if groups:
            node["groups"] = {"n": groups[0], "top": groups[1]}
        env = {"x": x}
        ir_numpy.run_nodes([node], env, W, i=0)
        want = moe64(x, R, G, U, D, k, scoring, bias, groups, renorm)
        rel = float(np.abs(env["y"] - want).max() / max(1.0, np.abs(want).max()))
        worst = max(worst, rel)
        assert rel < 2e-6, (scoring, bias_on, groups, renorm, rel)
    assert worst < 2e-6


def yarn64(dim, base, factor, orig, beta_fast=32.0, beta_slow=1.0):
    """Peng et al. 2023, section 3.2: r(d) = L / lambda_d, lambda_d = 2 pi base^(2d/|D|); h(theta_d) = (1 - gamma)
    theta_d / s + gamma theta_d with gamma 0 for r < alpha and 1 for r > beta, linear between (in the dimension
    index, as the released code: the correction dimensions of beta_fast and beta_slow rotations, floor and
    ceiling). Returns the divisors theta_d / h(theta_d)."""
    def dim_of(rot):  # the d at which r(d) = rot
        return dim * math.log(orig / (rot * 2 * math.pi)) / (2 * math.log(base))
    lo, hi = max(math.floor(dim_of(beta_fast)), 0), min(math.ceil(dim_of(beta_slow)), dim - 1)
    hi = hi + 0.001 if hi == lo else hi
    out = []
    for d in range(dim // 2):
        gamma = 1.0 - min(1.0, max(0.0, (d - lo) / (hi - lo)))
        out.append(1.0 / ((1.0 - gamma) / factor + gamma))
    return out


def test_yarn_factors_match_the_paper():
    for dim, base, factor, orig in ((64, 10000.0, 40.0, 4096), (64, 10000.0, 40.0, 4096), (8, 10000.0, 4.0, 16),
                                    (4, 10000.0, 4.0, 8), (128, 500000.0, 8.0, 8192)):
        got = gguf_ir.yarn_factors(dim, base, factor, orig)
        want = yarn64(dim, base, factor, orig)
        assert len(got) == dim // 2
        assert max(abs(a - b) / b for a, b in zip(got, want)) < 1e-7
    # DeepSeek-V2-Lite (factor 40 over 4096, 64 rope dims): high frequencies extrapolate, low ones interpolate.
    f = gguf_ir.yarn_factors(64, 10000.0, 40.0, 4096)
    assert f[0] == 1.0 and f[-1] == 40.0


def test_apply_skips_a_node_per_application():
    doc = {"block": {"repeat": 2}}
    nodes = [{"op": "affine", "in": ["x"], "scale": 2.0, "shift": 0.0, "out": "x",
              "apply": {"per_layer": [True, False]}},
             {"op": "affine", "in": ["x"], "scale": 3.0, "shift": 1.0, "out": "x",
              "apply": {"per_layer": [False, True]}}]
    x = r(3, 4)
    env = {"x": x}
    for i in range(doc["block"]["repeat"]):
        ir_numpy.run_nodes(nodes, env, {}, i=i)
    want = (x * np.float32(2)) * np.float32(3) + np.float32(1)
    assert np.array_equal(env["x"], want.astype(np.float32))
