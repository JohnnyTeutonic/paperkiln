"""Revision 1.5 (IR_SPEC 5.5, 5.14, 5.15, 5.20): the reference against independent float64 statements of the
published formulas, on random inputs.

- mla_attention (DeepSeek-V2, arXiv 2405.04434, eqs. 9-19): the reference computes the absorbed form (the query
  through k_b, values through v_b after the weighted sum); the float64 statement decompresses every key and value
  first (k_nope = W_UK c, v = W_UV c, then standard attention over [k_nope | k_rope]), as the paper writes it.
- attention softcap (Gemma 2, arXiv 2408.00118 section 2): cap * tanh(s / cap) on the scaled score.
- rope freq_factors (LongRoPE, arXiv 2402.13753; Phi-3, arXiv 2404.14219): the angle pos * theta^(-2j/n) / f_j,
  cos and sin times attn_factor, the long set when the sequence's length exceeds original_ctx.
- split groups: column bookkeeping, exact.
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import ir_numpy  # noqa: E402

RNG = np.random.default_rng(1505)


def r(*shape, s=1.0):
    return (s * RNG.standard_normal(shape)).astype(np.float32)


def run(node, env, W):
    env = dict(env)
    ir_numpy.run_nodes([node], env, W, i=0)
    return env


def rel(a, b):
    return float(np.abs(np.asarray(a, np.float64) - b).max() / max(1.0, np.abs(b).max()))


def mla64(qn, qr, c, kr, kb, vb, H, dn, dr, scale, window=0, sinks=0):
    """Decompressed MLA in float64: per head, k = [W_UK c | kr], v = W_UV c; softmax over the visible keys."""
    T = qn.shape[0]
    qn, qr, c, kr, kb, vb = (np.asarray(a, np.float64) for a in (qn, qr, c, kr, kb, vb))
    p, j = np.arange(T)[:, None], np.arange(T)[None, :]
    hide = j > p
    if window:
        hide |= (j >= sinks) & (j < p - window + 1)
    out = []
    for h in range(H):
        k_nope = c @ kb[h]  # [T, dn]: kb[h] is [r, dn], so W_UK = kb[h]^T
        v = c @ vb[h].T  # [T, dv]
        q = np.concatenate([qn[:, h * dn:(h + 1) * dn], qr[:, h * dr:(h + 1) * dr]], axis=1)
        k = np.concatenate([k_nope, kr], axis=1)
        s = (q @ k.T) * scale
        s[hide] = -np.inf
        a = np.exp(s - s.max(axis=1, keepdims=True))
        a /= a.sum(axis=1, keepdims=True)
        out.append(a @ v)
    return np.concatenate(out, axis=1)


def test_mla_attention_matches_the_decompressed_float64_form():
    for H, r_, dn, dr, dv, T, win, sinks in ((2, 6, 4, 4, 3, 9, 0, 0), (4, 20, 16, 8, 12, 17, 0, 0),
                                             (3, 5, 2, 2, 7, 12, 4, 1), (1, 3, 2, 2, 2, 5, 0, 0)):
        node = {"op": "mla_attention", "in": ["qn", "qr", "c", "kr"], "state": "kv", "heads": H, "kv_rank": r_,
                "qk_nope": dn, "qk_rope": dr, "v_head_dim": dv, "k_b": "kb", "v_b": "vb",
                "scale": float(np.float32(1 / np.sqrt(dn + dr))), "causal": True, "window": win, "sinks": sinks,
                "out": "a"}
        env = {"qn": r(T, H * dn), "qr": r(T, H * dr), "c": r(T, r_), "kr": r(T, dr)}
        W = {"kb": r(H, r_, dn, s=0.5), "vb": r(H, dv, r_, s=0.5)}
        got = run(node, env, W)["a"]
        want = mla64(env["qn"], env["qr"], env["c"], env["kr"], W["kb"], W["vb"], H, dn, dr, node["scale"], win, sinks)
        assert got.shape == (T, H * dv)
        assert rel(got, want) < 2e-6, rel(got, want)


def test_mla_attention_decode_rows_equal_prefill():
    """A row's output depends on the rows up to it only (the kv slot holds c and kr per position)."""
    H, r_, dn, dr, dv, T = 2, 6, 4, 4, 3, 8
    node = {"op": "mla_attention", "in": ["qn", "qr", "c", "kr"], "state": "kv", "heads": H, "kv_rank": r_,
            "qk_nope": dn, "qk_rope": dr, "v_head_dim": dv, "k_b": "kb", "v_b": "vb", "scale": 0.35, "causal": True,
            "window": 0, "sinks": 0, "out": "a"}
    env = {"qn": r(T, H * dn), "qr": r(T, H * dr), "c": r(T, r_), "kr": r(T, dr)}
    W = {"kb": r(H, r_, dn), "vb": r(H, dv, r_)}
    full = run(node, env, W)["a"]
    for t in range(1, T + 1):
        part = run(node, {k: v[:t] for k, v in env.items()}, W)["a"]
        assert rel(part[-1], full[t - 1].astype(np.float64)) < 1e-6


def test_attention_softcap_matches_float64():
    H, hd, T, cap = 2, 4, 9, 0.5
    node = {"op": "attention", "in": ["q", "k", "v"], "state": "kv", "heads": H, "kv_heads": H, "head_dim": hd,
            "causal": True, "scale": 0.5, "window": 0, "sinks": 0, "softcap": cap, "out": "a"}
    q, k, v = r(T, H * hd), r(T, H * hd), r(T, H * hd)
    got = run(node, {"q": q, "k": k, "v": v}, {})["a"]
    out = []
    for h in range(H):
        sl = slice(h * hd, (h + 1) * hd)
        s = (q[:, sl].astype(np.float64) @ k[:, sl].T.astype(np.float64)) * 0.5
        s = cap * np.tanh(s / cap)
        s[np.arange(T)[None, :] > np.arange(T)[:, None]] = -np.inf
        a = np.exp(s - s.max(axis=1, keepdims=True))
        out.append((a / a.sum(axis=1, keepdims=True)) @ v[:, sl].astype(np.float64))
    want = np.concatenate(out, axis=1)
    assert rel(got, want) < 2e-6
    # The cap binds: without it the output is another.
    plain = run(dict(node, softcap=None), {"q": q, "k": k, "v": v}, {})["a"]
    assert rel(plain, want) > 1e-2


def rope64(x, pos, H, hd, n_rot, theta, f, m, pairing):
    """HF's rotate_half (neox) or adjacent pairs, inv_freq_j = 1 / (f_j * theta^(2j/n_rot)), cos and sin times m."""
    x = np.asarray(x, np.float64).reshape(len(pos), H, hd).copy()
    j = np.arange(n_rot // 2)
    ang = np.asarray(pos, np.float64)[:, None] / (np.asarray(f, np.float64) * theta ** (2.0 * j / n_rot))[None, :]
    c, s = (np.cos(ang) * m)[:, None, :], (np.sin(ang) * m)[:, None, :]
    if pairing == "neox":
        a, b = x[..., :n_rot // 2].copy(), x[..., n_rot // 2:n_rot].copy()
        x[..., :n_rot // 2], x[..., n_rot // 2:n_rot] = a * c - b * s, a * s + b * c
    else:
        a, b = x[..., 0:n_rot:2].copy(), x[..., 1:n_rot:2].copy()
        x[..., 0:n_rot:2], x[..., 1:n_rot:2] = a * c - b * s, a * s + b * c
    return x.reshape(len(pos), H * hd)


def test_rope_freq_factors_short_long_and_attn_factor_match_float64():
    H, hd, n_rot, theta = 2, 8, 4, 10000.0
    fs, fl = (1 + 0.5 * np.abs(r(2))).astype(np.float32), (2 + 4 * np.abs(r(2))).astype(np.float32)
    base = {"op": "rope", "in": ["q", "k", "pos"], "heads": H, "kv_heads": 1, "head_dim": hd, "n_rot": n_rot,
            "theta": theta, "rope_heads": "all", "pairing": "neox", "out": ["q", "k"], "attn_factor": 1.25,
            "freq_factors": {"short": "fs", "long": "fl", "original_ctx": 6}}
    for T, f in ((6, fs), (7, fl), (12, fl)):
        q, k = r(T, H * hd), r(T, hd)
        env = run(base, {"q": q, "k": k, "pos": np.arange(T)}, {"fs": fs, "fl": fl})
        assert rel(env["q"], rope64(q, np.arange(T), H, hd, n_rot, theta, f, 1.25, "neox")) < 1e-6
        assert rel(env["k"], rope64(k, np.arange(T), 1, hd, n_rot, theta, f, 1.25, "neox")) < 1e-6
    # In a forward pass the set follows the sequence's length, not a row's position: every row of a 7-token
    # sequence uses the long factors, the first 6 rows included.
    doc_env = {"q": r(7, H * hd), "k": r(7, hd), "pos": np.arange(7), "_seq_len": 7}
    env = run(base, doc_env, {"fs": fs, "fl": fl})
    assert rel(env["q"][:6], rope64(doc_env["q"][:6], np.arange(6), H, hd, n_rot, theta, fl, 1.25, "neox")) < 1e-6
    # One tensor, interleaved pairs.
    one = dict(base, pairing="interleaved", freq_factors="f", attn_factor=1.0)
    q = r(5, H * hd)
    env = run(one, {"q": q, "k": r(5, hd), "pos": np.array([0, 3, 4, 9, 30])}, {"f": fl})
    assert rel(env["q"], rope64(q, [0, 3, 4, 9, 30], H, hd, n_rot, theta, fl, 1.0, "interleaved")) < 1e-6


def test_split_groups_is_exact_bookkeeping():
    x = r(3, 12)
    env = run({"op": "split", "in": ["x"], "sizes": [2, 1], "groups": 4, "out": ["a", "b"]}, {"x": x}, {})
    g = x.reshape(3, 4, 3)
    assert np.array_equal(env["a"], g[:, :, :2].reshape(3, 8))
    assert np.array_equal(env["b"], g[:, :, 2:].reshape(3, 4))
