"""The paperkiln IR's numpy reference kernels (include/microtorch/ir.hpp), as a module: tools/ir_run.py
runs paperkiln GGUFs with them, tools/thirdparty_parity.py runs third-party GGUFs whose IR
tools/gguf_ir.py builds from llama.cpp metadata.

    run_nodes(nodes, env, W, i=None, no_window=False, wi=None)   # one embed / block / head node list
    forward(doc, W, ids, last_only=False, trace=None)   # logits [T, vocab] (or the last row only)
    load_weights(path)                                  # {name: f32 array [.., in]} from any GGUF type

Every op computes in f32. Beyond the ops tools/ir_run.py documents:
  scale       y = (x * w) + b per feature, w and b [d] (b may be null): the product rounded to f32,
              then the sum, never fused; a weight or bias whose length is not x's width stops,
              naming the tensor.
  expr        a runtime-defined op, evaluated by tools/ir_expr.py.
For third-party graphs:
  attention   kv_heads < heads (grouped-query): query head h reads kv head h // (heads // kv_heads).
  rope        pairing "neox": the pair (x[j], x[j + n_rot/2]) within a head (llama.cpp NEOX, HF
              rotate_half), "interleaved" the adjacent pair (x[2j], x[2j+1]); n_rot (default
              head_dim) rotates only a head's first n_rot dims (partial rotary); the angle is
              pos * theta^(-2j/n_rot); kv_heads (default heads) is k's head count.
  rmsnorm     group g (default the whole row): normalise each consecutive g columns on their own
              (per-head q/k norm), weight [g] shared by every group; weight_offset c (default 0):
              the weight read as (w + c), rounded to f32 once (a load-time transform; Gemma's 1 + w).
  rope        freq_scale s (default 1): the angle is (pos * s) * theta^(-2j/n_rot) (linear scaling).
  conv1d_causal, selective_scan  Mamba-1's ops (include/microtorch/ir.hpp).
Any attribute of a block node (and of a block state slot) may be {"per_layer": [v_0, .., v_{L-1}]}:
block i reads v_i. The block's own key "weight_index" {"per_layer": [m_0, .., m_{L-1}]} (tied-block
recursion) makes application i read its tensors from blk.{m_i}.* (run_nodes' wi); everything else,
per-layer attributes and state included, stays indexed by i.
A block whose first node is depth_route (Mixture-of-Recursions, causal token choice; kv slots with
kv_rows "active") routes at step 0's first application and runs each recursion step's applications on
its active rows only (forward's trace receives the depths).
"""
from __future__ import annotations

import math

import numpy as np

ERF = np.vectorize(math.erf)


def load_weights(path):
    """Every tensor of a GGUF as f32, shaped [.., in] (numpy order = reversed GGUF dims), dequantised
    by the gguf package (f16, bf16, q8_0, k-quants ...)."""
    import gguf
    r = gguf.GGUFReader(path)
    W = {}
    for t in r.tensors:
        shape = [int(x) for x in reversed(t.shape.tolist())]
        if t.tensor_type == gguf.GGMLQuantizationType.F32:
            a = np.asarray(t.data, dtype=np.float32)
        else:
            a = gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float32)
        W[t.name] = a.reshape(shape)
    return W


def per_layer(nd, i):
    """The node with every {"per_layer": [...]} attribute replaced by block i's entry."""
    if i is None or not any(isinstance(v, dict) and "per_layer" in v for v in nd.values()):
        return nd
    return {k: v["per_layer"][i] if isinstance(v, dict) and "per_layer" in v else v for k, v in nd.items()}


def softplus(x):
    """llama.cpp ggml_ssm_scan's: x <= 20 ? log1p(exp(x)) : x (torch's softplus, threshold 20)."""
    return np.where(x <= np.float32(20), np.log1p(np.exp(np.minimum(x, np.float32(20)))), x).astype(np.float32)


def _rope(x, pos, heads, hd, n_rot, theta, pairing, freq_scale=1.0):
    t = x.reshape(len(pos), heads, hd)
    half = n_rot // 2
    j = np.arange(half)
    ang = (pos[:, None].astype(np.float64) * float(freq_scale)) * (float(theta) ** (-2.0 * j / n_rot))[None, :]
    c, s = np.cos(ang).astype(np.float32)[:, None, :], np.sin(ang).astype(np.float32)[:, None, :]
    out = t.copy()
    if pairing == "neox":
        a, b = t[..., :half], t[..., half:n_rot]
        out[..., :half] = a * c - b * s
        out[..., half:n_rot] = a * s + b * c
    elif pairing == "interleaved":
        a, b = t[..., 0:n_rot:2], t[..., 1:n_rot:2]
        out[..., 0:n_rot:2] = a * c - b * s
        out[..., 1:n_rot:2] = a * s + b * c
    else:
        raise SystemExit(f"ir_numpy: rope pairing {pairing!r}")
    return out.reshape(len(pos), heads * hd).astype(np.float32)


def weight_index(doc, i):
    """The stored layer whose tensors block application i reads (ir.hpp weight_index; i when absent)."""
    wmap = doc["block"].get("weight_index")
    if wmap is None:
        return i
    m = wmap["per_layer"]
    if len(m) != doc["block"]["repeat"] or not all(isinstance(x, int) and x >= 0 for x in m):
        raise SystemExit("ir_numpy: block weight_index must list block.repeat integers >= 0")
    return m[i]


def run_nodes(nodes, env, W, i=None, no_window=False, wi=None):
    """wi: the stored layer whose tensors "{i}" names (default i, the application)."""
    ti = i if wi is None else wi

    def name(key):
        return key.replace("{i}", str(ti)) if ti is not None else key

    def w(key):
        return W[name(key)]

    for nd in nodes:
        nd = per_layer(nd, i)
        op, ins = nd["op"], nd["in"]
        x = env[ins[0]]
        if op == "embedding":
            y = w(nd["weight"])[x]
        elif op == "pos_embedding":
            y = x + w(nd["weight"])[env[ins[1]]]
        elif op == "linear":
            wt = W.get(name(nd["weight"]))
            if wt is None:
                wt = W[nd["tied_to"]]
            y = x @ wt.T
            if nd.get("bias"):
                y = y + w(nd["bias"])
        elif op == "scale":
            wt = w(nd["weight"]).reshape(-1)
            bias = w(nd["bias"]).reshape(-1) if nd.get("bias") else None
            for tname, t in ((nd["weight"], wt), (nd.get("bias"), bias)):
                if t is not None and t.shape[0] != x.shape[1]:
                    raise SystemExit(f"ir_numpy: scale tensor {name(tname)} has length {t.shape[0]}, "
                                     f"not the width {x.shape[1]} of {ins[0]}")
            y = (x * wt).astype(np.float32)
            if bias is not None:
                y = (y + bias).astype(np.float32)
        elif op == "expr":
            import ir_expr
            y = ir_expr.evaluate(nd["expr"], [env[n] for n in ins])
        elif op == "layernorm":
            mu = x.mean(axis=1, keepdims=True)
            var = ((x - mu) ** 2).mean(axis=1, keepdims=True)
            y = (x - mu) / np.sqrt(var + np.float32(nd["eps"])) * w(nd["weight"]) + w(nd["bias"])
        elif op == "rmsnorm":
            g = nd.get("group") or x.shape[1]
            t = x.reshape(x.shape[0], x.shape[1] // g, g)
            wt = w(nd["weight"])
            if nd.get("weight_offset"):
                wt = (wt + np.float32(nd["weight_offset"])).astype(np.float32)
            t = t / np.sqrt((t * t).mean(axis=2, keepdims=True) + np.float32(nd["eps"])) * wt
            y = t.reshape(x.shape)
        elif op == "conv1d_causal":
            # Depthwise causal conv over time, per channel c: out[t, c] = sum_k w[c, k] xx[t + k, c]
            # (k ascending), xx = the kernel-1 earlier inputs (the state; 0 at the start) then x;
            # bias added to the finished sum.
            K = nd["kernel"]
            wt = w(nd["weight"]).reshape(x.shape[1], K)
            xx = np.concatenate([np.zeros((K - 1, x.shape[1]), np.float32), x], axis=0)
            y = np.zeros_like(x)
            for k in range(K):
                y = (y + wt[:, k] * xx[k:k + x.shape[0]]).astype(np.float32)
            if nd.get("bias"):
                y = y + w(nd["bias"])
        elif op == "selective_scan":
            # Mamba-1 (ggml_ssm_scan): delta = softplus(dt); s = s * exp(delta A) + B (x delta);
            # y = sum_n s C (n ascending); + D x.
            xs, dt, bm, cm = (env[n] for n in ins)
            A = w(nd["a"]).reshape(x.shape[1], nd["d_state"])
            delta = softplus(dt) if nd.get("dt_softplus", True) else dt
            s = np.zeros_like(A)
            y = np.zeros_like(xs)
            for t in range(xs.shape[0]):
                s = (s * np.exp(delta[t][:, None] * A) + bm[t][None, :] * (xs[t] * delta[t])[:, None]).astype(np.float32)
                y[t] = s @ cm[t]
            if nd.get("d"):
                y = y + xs * w(nd["d"]).reshape(-1)
        elif op == "gelu":
            if nd["approx"] == "tanh":
                y = 0.5 * x * (1 + np.tanh(np.float32(math.sqrt(2 / math.pi)) * (x + np.float32(0.044715) * x ** 3)))
            else:
                y = 0.5 * x * (1 + ERF(x / np.float32(math.sqrt(2))).astype(np.float32))
        elif op == "silu":
            y = x / (1 + np.exp(-x))
        elif op in ("add", "mul", "sub"):
            y = x + env[ins[1]] if op == "add" else x * env[ins[1]] if op == "mul" else x - env[ins[1]]
        elif op == "relu":
            y = np.maximum(x, np.float32(0))
        elif op == "sigmoid":
            y = np.float32(1) / (np.float32(1) + np.exp(-x))
        elif op == "affine":
            y = (x * np.float32(nd["scale"])).astype(np.float32) + np.float32(nd["shift"])
        elif op == "mul_col":
            y = x * env[ins[1]]  # [T, w] * [T, 1]
        elif op == "rms_row":
            y = np.sqrt((x * x).sum(axis=1, keepdims=True) / np.float32(x.shape[1]) + np.float32(nd["eps"]))
        elif op == "linear_attention":
            # elu+1 features; num/den running sums over the tokens (a
            # sequential scan, np.cumsum), then phi(q) num / max(den, floor).
            assert nd["feature_map"] == "elu_plus_one" and nd["causal"]
            one = np.float32(1)

            def phi(a):
                return np.where(a > 0, a + one, (np.exp(a) - one) + one).astype(np.float32)

            q, k, v = (env[n] for n in ins)
            pk = phi(k)
            num = np.cumsum(pk * v, axis=0, dtype=np.float32)
            den = np.cumsum(pk, axis=0, dtype=np.float32)
            y = (phi(q) * num) / np.maximum(den, np.float32(nd["den_floor"]))
        elif op == "list":
            env[nd["out"]] = [env[n] for n in ins]
            continue
        elif op == "append":
            env[nd["out"]] = env[ins[0]] + [env[ins[1]]]
            continue
        elif op == "depth_attend":
            src = list(env[ins[0]]) + [env[n] for n in ins[1:]]
            if len(src) == 1:
                y = src[0]
            else:
                wq, eps = w(nd["weight"]), np.float32(nd["eps"])
                logits = []
                for s in src:
                    r = np.float32(1) / np.sqrt((s * s).sum(axis=1, keepdims=True) / np.float32(s.shape[1]) + eps)
                    logits.append(((s * r) * wq).sum(axis=1))
                lg = np.stack(logits, axis=1)
                e = np.exp(lg - lg.max(axis=1, keepdims=True))
                a = e / e.sum(axis=1, keepdims=True)
                y = src[0] * a[:, 0:1]
                for j in range(1, len(src)):
                    y = y + src[j] * a[:, j:j + 1]
        elif op == "ssm_scan":
            A, b, c, d = (w(nd[n]) for n in ("a", "b", "c", "d"))
            s = np.zeros(nd["n"], dtype=np.float32)
            y = np.zeros_like(x)
            for t in range(x.shape[0]):
                s = (A @ s + b * x[t]).astype(np.float32)
                y[t] = c * s + d[0] * x[t]
        elif op == "split":
            outs, at = [], 0
            for s in nd["sizes"]:
                outs.append(x[:, at:at + s])
                at += s
            assert at == x.shape[1]
            for o, v in zip(nd["out"], outs):
                env[o] = v
            continue
        elif op == "rope":
            if nd["rope_heads"] != "all":
                raise SystemExit("ir_numpy: rope_heads other than 'all'")
            H, hd, pos = nd["heads"], nd["head_dim"], env[ins[2]]
            n_rot = nd.get("n_rot") or hd
            for o, src, heads in zip(nd["out"], ins[:2], (H, nd.get("kv_heads") or H)):
                env[o] = _rope(env[src], pos, heads, hd, n_rot, nd["theta"], nd["pairing"], nd.get("freq_scale", 1.0))
            continue
        elif op == "moe_ffn":
            # Softmax over the k selected router logits (= renormalised top-k
            # probabilities); ties to the lower expert index; experts summed in
            # ascending index. gate_norm "raw" uses the full softmax instead.
            k = nd["top_k"]
            logits = x @ w(nd["router"]).T
            g, u, d = (w(nd["experts"][n]) for n in ("gate", "up", "down"))  # [E][out][in]
            y = np.zeros((x.shape[0], d.shape[1]), dtype=np.float32)
            for t_i in range(x.shape[0]):
                lt = logits[t_i]
                sel = sorted(np.argsort(-lt, kind="stable")[:k])
                if nd["gate_norm"] == "renorm":
                    e = np.exp(lt[sel] - lt[sel].max())
                    gates = e / e.sum()
                else:
                    p = np.exp(lt - lt.max())
                    gates = (p / p.sum())[sel]
                for gate, e_i in zip(gates, sel):
                    hgate = g[e_i] @ x[t_i]
                    h = hgate / (1 + np.exp(-hgate)) * (u[e_i] @ x[t_i])
                    y[t_i] += np.float32(gate) * (d[e_i] @ h)
        elif op == "attention":
            q, k, v = (env[n] for n in ins)
            H, hd, T = nd["heads"], nd["head_dim"], x.shape[0]
            KV = nd["kv_heads"]
            assert nd["causal"] and H % KV == 0
            outs = []
            # Hidden keys: j > p, and, with a window, keys that are neither a
            # sink (j < sinks) nor among the last `window` (j >= p - window + 1).
            p, j = np.arange(T)[:, None], np.arange(T)[None, :]
            mask = j > p
            win = 0 if no_window else nd["window"]
            if win:
                mask |= (j >= nd["sinks"]) & (j < p - win + 1)
            for h in range(H):
                kh = h // (H // KV)
                s = (q[:, h * hd:(h + 1) * hd] @ k[:, kh * hd:(kh + 1) * hd].T) * np.float32(nd["scale"])
                s[mask] = -np.inf
                s = np.exp(s - s.max(axis=1, keepdims=True))
                s /= s.sum(axis=1, keepdims=True)
                outs.append(s @ v[:, kh * hd:(kh + 1) * hd])
            y = np.concatenate(outs, axis=1)
        else:
            raise SystemExit(f"ir_numpy: no kernel for op {op}")
        env[nd["out"]] = y.astype(np.float32)


def _routed_blocks(doc, W, env, no_window, trace):
    """The block under a depth_route node (ir.hpp; Mixture-of-Recursions, causal token choice): route at
    step 0's first application, run each step's applications on its active rows only (their positions,
    so attention and RoPE see exactly those rows), then x = x_in + g (x - x_in) for them."""
    route, nodes = doc["block"]["nodes"][0], doc["block"]["nodes"][1:]
    if route.get("route") != "token_choice" or route.get("gate") not in ("score", "none"):
        raise SystemExit("ir_numpy: depth_route supports route token_choice and gate score | none")
    for s in doc["block"].get("state", []):
        if s.get("kind") == "kv" and s.get("kv_rows") != "active":
            raise SystemExit("ir_numpy: depth_route needs every kv slot's kv_rows to be active")
    steps, L = route["step"]["per_layer"], doc["block"]["repeat"]
    T = env["x"].shape[0]
    depth = gate = x_in = act = sub = None
    for i in range(L):
        k = steps[i]
        if k < 0:
            run_nodes(nodes, env, W, i, no_window=no_window, wi=weight_index(doc, i))
            continue
        first = i == 0 or steps[i - 1] != k
        last = i == L - 1 or steps[i + 1] != k
        if k == 0 and first:
            logits = (env["x"] @ W[route["weight"]].T).astype(np.float32)  # [T, R]
            j = np.argmax(logits, axis=1)  # the first largest: ties to the lower index
            e = np.exp(logits - logits.max(axis=1, keepdims=True))
            p = (e / e.sum(axis=1, keepdims=True)).astype(np.float32)
            depth = j + 1
            gate = p[np.arange(T), j][:, None]
            srt = np.sort(logits, axis=1)
            env["rec_margin"] = srt[:, -1] - srt[:, -2] if logits.shape[1] > 1 else np.full(T, np.inf)
            env[route["out"]] = depth[:, None].astype(np.float32)
            if trace is not None:
                trace.extend(int(d) for d in depth)
        if first:
            x_in, act = env["x"], np.nonzero(depth > k)[0]
            sub = {"x": x_in[act], "pos": env["pos"][act]}
        if act.size == 0:
            continue
        run_nodes(nodes, sub, W, i, no_window=no_window, wi=weight_index(doc, i))
        if last:
            x = x_in.copy()
            if route["gate"] == "score":
                d = (sub["x"] - x_in[act]).astype(np.float32)
                x[act] = (x_in[act] + (gate[act] * d).astype(np.float32)).astype(np.float32)
            else:
                x[act] = sub["x"]
            env["x"] = x


def forward(doc, W, ids, last_only=False, no_window=False, trace=None):
    """Logits [T, vocab] for one sequence from position 0; last_only runs the head on the last row
    only (every head op is row-wise), the cheap way to decode greedily. trace (a list): under a
    depth_route block, each token's depth is appended to it."""
    env = {"tokens": np.array(ids), "pos": np.arange(len(ids))}  # pos unused by graphs without positions
    run_nodes(doc["embed"]["nodes"], env, W, no_window=no_window)
    blk = doc["block"]["nodes"]
    if blk and blk[0]["op"] == "depth_route":
        _routed_blocks(doc, W, env, no_window, trace)
    else:
        for i in range(doc["block"]["repeat"]):
            run_nodes(blk, env, W, i, no_window=no_window, wi=weight_index(doc, i))
    if last_only:
        for k in list(env):
            if isinstance(env[k], np.ndarray) and env[k].ndim == 2 and env[k].shape[0] == len(ids):
                env[k] = env[k][-1:]
    run_nodes(doc["head"]["nodes"], env, W, no_window=no_window)
    return env["logits"]
