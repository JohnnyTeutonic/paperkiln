"""The paperkiln IR's numpy reference kernels (IR_SPEC.md), as a module: engines' gates run GGUFs with
embedded documents through them (paperkiln's tools/ir_run.py and tools/thirdparty_parity.py among them),
and third-party GGUFs whose document tools/gguf_ir.py builds from llama.cpp metadata.

    run_nodes(nodes, env, W, i=None, no_window=False, wi=None, states=None)   # one embed / block / head node list
    forward(doc, W, ids, last_only=False, trace=None, states=None)   # logits [T, vocab] (or the last row only)
    load_weights(path)                                  # {name: f32 array [.., in]} from any GGUF type

Every op computes in f32 (IR_SPEC.md sections 5 to 7). Notes on some of them:
  scale       y = (x * w) + b per feature, w and b [d] (b may be null): the product rounded to f32,
              then the sum, never fused; a weight or bias whose length is not x's width stops,
              naming the tensor.
  expr        a runtime-defined op, evaluated by tools/ir_expr.py.
For third-party graphs:
  attention   kv_heads < heads (grouped-query): query head h reads kv head h // (heads // kv_heads).
              position_bias {weight, buckets} (revision 1.3; T5's relative bias): each head adds
              W[buckets[min(p - j, D - 1)], h] to its scaled score (W [B, heads]; p, j the positions).
  rope        pairing "neox": the pair (x[j], x[j + n_rot/2]) within a head (llama.cpp NEOX, HF
              rotate_half), "interleaved" the adjacent pair (x[2j], x[2j+1]); n_rot (default
              head_dim) rotates only a head's first n_rot dims (partial rotary); the angle is
              pos * theta^(-2j/n_rot); kv_heads (default heads) is k's head count.
  rmsnorm     group g (default the whole row): normalise each consecutive g columns on their own
              (per-head q/k norm), weight [g] shared by every group; weight_offset c (default 0):
              the weight read as (w + c), rounded to f32 once (a load-time transform; Gemma's 1 + w).
  rope        freq_scale s (default 1): the angle is (pos * s) * theta^(-2j/n_rot) (linear scaling).
  conv1d_causal, selective_scan  Mamba-1's ops (IR_SPEC.md 6.6, 6.7).
  recurrence  a per-token state update declared in the document (IR_SPEC.md 6.10; tools/ir_recurrence.py).
Any attribute of a block node (and of a block state slot) may be {"per_layer": [v_0, .., v_{L-1}]}:
block i reads v_i. The block's own key "weight_index" {"per_layer": [m_0, .., m_{L-1}]} (tied-block
recursion) makes application i read its tensors from blk.{m_i}.* (run_nodes' wi); everything else,
per-layer attributes and state included, stays indexed by i. A node with "tensor_index": "application"
(llama_attnres under recursion: each application's own pseudo-queries) reads "{i}" as the application i
instead (IR_SPEC.md 3.8, family llama_attnres).
A block whose first node is depth_route (Mixture-of-Recursions, causal token choice; kv slots with
kv_rows "active") routes at step 0's first application and runs each recursion step's applications on
its active rows only (forward's trace receives the depths).
Per-turn state tokens (IR_SPEC.md 6.8): an embedding node with "state_weight"
reads row v - vocab of that tensor for an id v >= vocab; an attention node with "mask": "turn_state" lets
the query at p see the key at j <= p iff j is a state position or in p's segment (turn_state_codes below,
from the token ids, k = the state_weight's rows). A kv slot's "evict" changes no value (the engine may
drop rows no later query sees), so the prefill reference ignores it.
states (a dict): each recurrent op records its slot's tensors after the last row, as
states[slot][application] = {tensor: array} (the decode state a prefill leaves; IR_SPEC.md 6.3).
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


def turn_state_codes(ids, vocab, k):
    """ops::turn_state_codes for one sequence: 2 * s + c per position, c = 1 for a state id (>= vocab),
    s = the complete state blocks (k consecutive ids vocab .. vocab + k - 1) strictly before it."""
    out, seg, run = [], 0, 0
    for t in ids:
        state = t >= vocab
        out.append(2 * seg + (1 if state else 0))
        if not state:
            run = 0
        else:
            j = t - vocab
            run = run + 1 if j == run else (1 if j == 0 else 0)
            if run == k:
                seg, run = seg + 1, 0
    return out


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


class WidthError(ValueError):
    """IR_SPEC 10.1 rule 37: a value's width does not fit the node reading it (a run-time refusal)."""


def _width(cond, section, msg):
    if not cond:
        raise WidthError(f"IR_SPEC {section} (rule 37): {msg}")


def weight_index(doc, i):
    """The stored layer whose tensors block application i reads (ir.hpp weight_index; i when absent)."""
    wmap = doc["block"].get("weight_index")
    if wmap is None:
        return i
    m = wmap["per_layer"]
    if len(m) != doc["block"]["repeat"] or not all(isinstance(x, int) and x >= 0 for x in m):
        raise SystemExit("ir_numpy: block weight_index must list block.repeat integers >= 0")
    return m[i]


def run_nodes(nodes, env, W, i=None, no_window=False, wi=None, states=None):
    """wi: the stored layer whose tensors "{i}" names (default i, the application). states: see the module
    docstring (recurrent slots' tensors after the last row, per application)."""

    def keep(nd, **tensors):
        if states is not None:
            states.setdefault(nd["state"], {})[i] = {k: np.array(v, dtype=np.float32) for k, v in tensors.items()}
    ti = i if wi is None else wi
    cur = [ti]  # the index "{i}" means for the current node (tensor_index "application": i)

    def name(key):
        return key.replace("{i}", str(cur[0])) if cur[0] is not None else key  # cur: see tensor_index

    def w(key):
        return W[name(key)]

    for nd in nodes:
        nd = per_layer(nd, i)
        cur[0] = i if nd.get("tensor_index") == "application" else ti
        op, ins = nd["op"], nd["in"]
        x = env[ins[0]]
        if op == "embedding":
            tab = w(nd["weight"])
            if nd.get("state_weight"):
                st = w(nd["state_weight"])
                V = tab.shape[0]
                if (x >= V + st.shape[0]).any() or (x < 0).any():
                    raise SystemExit("ir_numpy: an id past the state ids")
                y = np.where((x >= V)[:, None], st[np.clip(x - V, 0, st.shape[0] - 1)], tab[np.clip(x, 0, V - 1)])
            else:
                y = tab[x]
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
                if t is not None:
                    _width(t.shape[0] == x.shape[1], "5.4", f"scale tensor {name(tname)} has length "
                           f"{t.shape[0]}, not the width {x.shape[1]} of {ins[0]}")
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
            _width(x.shape[1] % g == 0, "5.6", f"rmsnorm group {g} does not divide the width {x.shape[1]} of {ins[0]}")
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
            keep(nd, x=xx[-(K - 1):])
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
            keep(nd, s=s)
        elif op == "gelu":
            if nd["approx"] == "tanh":
                y = 0.5 * x * (1 + np.tanh(np.float32(math.sqrt(2 / math.pi)) * (x + np.float32(0.044715) * x ** 3)))
            else:
                y = 0.5 * x * (1 + ERF(x / np.float32(math.sqrt(2))).astype(np.float32))
        elif op == "silu":
            y = x / (1 + np.exp(-x))
        elif op in ("add", "mul", "sub"):
            _width(np.shape(x) == np.shape(env[ins[1]]), "5.10", f"{op} reads {ins[0]} {list(np.shape(x))} and "
                   f"{ins[1]} {list(np.shape(env[ins[1]]))}, not one shape")
            y = x + env[ins[1]] if op == "add" else x * env[ins[1]] if op == "mul" else x - env[ins[1]]
        elif op == "relu":
            y = np.where(x > np.float32(0), x, np.float32(0))  # IR_SPEC 5.9: relu(-0) = relu(NaN) = +0
        elif op == "sigmoid":
            y = np.float32(1) / (np.float32(1) + np.exp(-x))
        elif op == "affine":
            y = (x * np.float32(nd["scale"])).astype(np.float32) + np.float32(nd["shift"])
        elif op == "mul_col":
            _width(np.shape(env[ins[1]]) == (x.shape[0], 1), "5.12", f"mul_col's {ins[1]} is "
                   f"{list(np.shape(env[ins[1]]))}, not [{x.shape[0]}, 1]")
            y = x * env[ins[1]]  # [T, w] * [T, 1]
        elif op == "rms_row":
            y = np.sqrt((x * x).sum(axis=1, keepdims=True) / np.float32(x.shape[1]) + np.float32(nd["eps"]))
        elif op == "recurrence":
            # IR_SPEC.md 6.10: the declared per-token update, t = 0 .. T-1 from the initial state.
            import ir_recurrence
            prog = ir_recurrence.compile_node(nd)
            y, st = ir_recurrence.run(prog, [env[n] for n in ins], {n: w(t) for n, _, t in prog.weights})
            keep(nd, **st)
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
            keep(nd, num=num[-1].reshape(nd["heads"], nd["head_dim"]), den=den[-1].reshape(nd["heads"], nd["head_dim"]))
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
                wq, eps = w(nd["weight"]), np.float32(nd.get("eps", 1e-05))  # IR_SPEC 5.18: eps defaults to 1e-5
                logits = []
                for s in src:
                    _width(s.shape == src[0].shape, "5.18", "depth_attend's sources are not of one shape")
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
            keep(nd, s=s)
        elif op == "split":
            outs, at = [], 0
            _width(sum(nd["sizes"]) == x.shape[1], "5.5", f"split sizes {nd['sizes']} do not sum to the width "
                   f"{x.shape[1]} of {ins[0]}")
            for s in nd["sizes"]:
                outs.append(x[:, at:at + s])
                at += s
            for o, v in zip(nd["out"], outs):
                env[o] = v
            continue
        elif op == "rope":
            H, hd, pos = nd["heads"], nd["head_dim"], env[ins[2]]
            n_rot = nd.get("n_rot") or hd
            for o, src, heads in zip(nd["out"], ins[:2], (H, nd.get("kv_heads") or H)):
                _width(env[src].shape[1] == heads * hd, "5.14", f"rope's {src} has width {env[src].shape[1]}, not "
                       f"{heads} heads of {hd}")
                if nd["rope_heads"] == "first":  # IR_SPEC 5.14: head 0 of q and of k rotates, the rest pass through
                    out = env[src].copy()
                    out[:, :hd] = _rope(env[src][:, :hd], pos, 1, hd, n_rot, nd["theta"], nd["pairing"],
                                        nd.get("freq_scale", 1.0))
                    env[o] = out
                else:
                    env[o] = _rope(env[src], pos, heads, hd, n_rot, nd["theta"], nd["pairing"],
                                   nd.get("freq_scale", 1.0))
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
            m = nd.get("mask", "causal")
            if m == "turn_state":
                if nd["window"] or nd["sinks"] or "_ts_code" not in env:
                    raise SystemExit("ir_numpy: mask turn_state needs window 0, sinks 0 and a state embedding")
                c = np.asarray(env["_ts_code"])
                seg, st = c >> 1, c & 1
                mask |= ~((st[None, :] == 1) | (seg[None, :] == seg[:, None]))
            elif m != "causal":
                raise SystemExit(f"ir_numpy: attention mask {m!r}")
            win = 0 if no_window else nd["window"]
            if win:
                mask |= (j >= nd["sinks"]) & (j < p - win + 1)
            pb = nd.get("position_bias")
            if pb:  # IR_SPEC 5.15 (revision 1.3): W[buckets[min(p - j, D - 1)], h], p and j the rows' positions
                bk = np.asarray(pb["buckets"], dtype=np.int64)
                tab = w(pb["weight"])
                _width(tab.ndim == 2 and tab.shape[1] == H and tab.shape[0] > int(bk.max()), "5.15",
                       f"position_bias {pb['weight']} has shape {list(tab.shape)}, not [B, heads {H}] with B > "
                       f"every bucket")
                ps = np.asarray(env["pos"], dtype=np.int64) if "pos" in env else np.arange(T)
                b_of = bk[np.clip(ps[:, None] - ps[None, :], 0, len(bk) - 1)]
            for h in range(H):
                kh = h // (H // KV)
                s = (q[:, h * hd:(h + 1) * hd] @ k[:, kh * hd:(kh + 1) * hd].T) * np.float32(nd["scale"])
                if pb:
                    s = (s + tab[b_of, h]).astype(np.float32)
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


def forward(doc, W, ids, last_only=False, no_window=False, trace=None, states=None):
    """Logits [T, vocab] for one sequence from position 0; last_only runs the head on the last row
    only (every head op is row-wise), the cheap way to decode greedily. trace (a list): under a
    depth_route block, each token's depth is appended to it."""
    env = {"tokens": np.array(ids), "pos": np.arange(len(ids))}  # pos unused by graphs without positions
    run_nodes(doc["embed"]["nodes"], env, W, no_window=no_window)
    sw = [n["state_weight"] for n in doc["embed"]["nodes"] if n.get("op") == "embedding" and n.get("state_weight")]
    if sw:  # the turn_state mask's codes, from the ids alone
        env["_ts_code"] = turn_state_codes(list(ids), doc["model"]["vocab"], W[sw[0]].shape[0])
    blk = doc["block"]["nodes"]
    if blk and blk[0]["op"] == "depth_route":
        _routed_blocks(doc, W, env, no_window, trace)
    else:
        for i in range(doc["block"]["repeat"]):
            run_nodes(blk, env, W, i, no_window=no_window, wi=weight_index(doc, i), states=states)
    if last_only:
        for k in list(env):
            if isinstance(env[k], np.ndarray) and env[k].ndim == 2 and env[k].shape[0] == len(ids):
                env[k] = env[k][-1:]
            elif isinstance(env[k], list):  # a depth_attend bank (attnres, llama_attnres): per-row entries
                env[k] = [e[-1:] if e.ndim == 2 and e.shape[0] == len(ids) else e for e in env[k]]
    run_nodes(doc["head"]["nodes"], env, W, no_window=no_window)
    return env["logits"]
