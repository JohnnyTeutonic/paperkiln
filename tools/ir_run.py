"""Reference interpreter for the paperkiln IR embedded in a GGUF (numpy, f32
tensors only): an independent check that the exported IR alone reproduces
mtstudio's greedy replies. Each step re-runs the last ctx tokens from
position 0, as mtstudio does. Ops: embedding, pos_embedding, linear,
layernorm, rmsnorm, gelu, silu, add, mul, split, rope, moe_ffn, attention
(causal, or a sliding window with sinks); any other op stops with its name.
40/40 vs mtstudio on chat_sftfinal_v3, moe_tiny_v3 and gpt2_tiny (20 Oct
2026), and swa_tiny with replies up to 200 tokens.

    python tools/ir_run.py model.gguf receipt.jsonl tools/ [out.jsonl]
        [--max-new N] [--no-window]

receipt.jsonl rows need "i", "prompt" and "mtstudio" (the recorded reply).
--max-new: the generation cap mtstudio was asked for (default 60).
--no-window: run every attention node with window 0 (dense causal), the
control that shows the window changes replies."""
import json
import math
import sys

import numpy as np

argv, MAX_NEW, NO_WINDOW = [], 60, False
_it = iter(sys.argv[1:])
for _a in _it:
    if _a == "--max-new":
        MAX_NEW = int(next(_it))
    elif _a == "--no-window":
        NO_WINDOW = True
    else:
        argv.append(_a)

sys.path.insert(0, argv[2])
import bpe  # noqa: E402
import ir_check  # noqa: E402

path = argv[0]
meta, tensors, data_start, raw, _ = ir_check.read_gguf(path)
doc = json.loads(meta["paperkiln.ir"][1])
W = {}
for name, dims, typ, off in tensors:
    assert typ == 0, name
    n = int(np.prod(dims))
    W[name] = np.frombuffer(raw, dtype=np.float32, count=n, offset=data_start + off).reshape(list(reversed(dims)))

ERF = np.vectorize(math.erf)


def run_nodes(nodes, env, i=None):
    def w(key):
        return W[key.replace("{i}", str(i))] if i is not None else W[key]

    for nd in nodes:
        op, ins = nd["op"], nd["in"]
        x = env[ins[0]]
        if op == "embedding":
            y = w(nd["weight"])[x]
        elif op == "pos_embedding":
            y = x + w(nd["weight"])[env[ins[1]]]
        elif op == "linear":
            wt = W.get(nd["weight"].replace("{i}", str(i))) if "{i}" in nd["weight"] else W.get(nd["weight"])
            if wt is None:
                wt = W[nd["tied_to"]]
            y = x @ wt.T
            if nd.get("bias"):
                y = y + w(nd["bias"])
        elif op == "layernorm":
            mu = x.mean(axis=1, keepdims=True)
            var = ((x - mu) ** 2).mean(axis=1, keepdims=True)
            y = (x - mu) / np.sqrt(var + np.float32(nd["eps"])) * w(nd["weight"]) + w(nd["bias"])
        elif op == "rmsnorm":
            y = x / np.sqrt((x * x).mean(axis=1, keepdims=True) + np.float32(nd["eps"])) * w(nd["weight"])
        elif op == "gelu":
            if nd["approx"] == "tanh":
                y = 0.5 * x * (1 + np.tanh(np.float32(math.sqrt(2 / math.pi)) * (x + np.float32(0.044715) * x ** 3)))
            else:
                y = 0.5 * x * (1 + ERF(x / np.float32(math.sqrt(2))).astype(np.float32))
        elif op == "silu":
            y = x / (1 + np.exp(-x))
        elif op in ("add", "mul"):
            y = x + env[ins[1]] if op == "add" else x * env[ins[1]]
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
            # Adjacent pairs (x[2j], x[2j+1]) within each head, angle
            # pos * theta^(-2j/head_dim); rope_heads "all" only.
            assert nd["rope_heads"] == "all" and nd["pairing"] == "interleaved"
            H, hd, pos = nd["heads"], nd["head_dim"], env[ins[2]]
            j = np.arange(hd // 2)
            ang = pos[:, None] * (float(nd["theta"]) ** (-2.0 * j / hd))[None, :]
            c, s = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
            for o, src in zip(nd["out"], ins[:2]):
                t = env[src].reshape(len(pos), H, hd // 2, 2)
                a, b = t[..., 0], t[..., 1]
                r = np.stack([a * c[:, None, :] - b * s[:, None, :], a * s[:, None, :] + b * c[:, None, :]], axis=-1)
                env[o] = r.reshape(len(pos), H * hd).astype(np.float32)
            continue
        elif op == "moe_ffn":
            # Softmax over the k selected router logits (= renormalised top-k
            # probabilities); ties to the lower expert index; experts summed in
            # ascending index. gate_norm "raw" uses the full softmax instead.
            E, k = nd["n_experts"], nd["top_k"]
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
            assert nd["kv_heads"] == H and nd["causal"]
            outs = []
            # Hidden keys: j > p, and, with a window, keys that are neither a
            # sink (j < sinks) nor among the last `window` (j >= p - window + 1).
            p, j = np.arange(T)[:, None], np.arange(T)[None, :]
            mask = j > p
            win = 0 if NO_WINDOW else nd["window"]
            if win:
                mask |= (j >= nd["sinks"]) & (j < p - win + 1)
            for h in range(H):
                s = (q[:, h * hd:(h + 1) * hd] @ k[:, h * hd:(h + 1) * hd].T) * np.float32(nd["scale"])
                s[mask] = -np.inf
                s = np.exp(s - s.max(axis=1, keepdims=True))
                s /= s.sum(axis=1, keepdims=True)
                outs.append(s @ v[:, h * hd:(h + 1) * hd])
            y = np.concatenate(outs, axis=1)
        else:
            raise SystemExit(f"ir_run: no kernel for op {op}")
        env[nd["out"]] = y.astype(np.float32)


def forward(ids):
    env = {"tokens": np.array(ids), "pos": np.arange(len(ids))}
    run_nodes(doc["embed"]["nodes"], env)
    for i in range(doc["block"]["repeat"]):
        run_nodes(doc["block"]["nodes"], env, i)
    run_nodes(doc["head"]["nodes"], env)
    return env["logits"]


ctx = doc["model"]["ctx"]
tok = bpe.read_gguf(path)
unk = doc["tokenizer"]["specials"].get("unk")
stops = {doc["tokenizer"]["specials"][r] for r in ("eot", "user", "assistant") if r in doc["tokenizer"]["specials"]}
rows = [json.loads(line) for line in open(argv[1], encoding="utf-8")]
# The span past which a window hides keys: window + sinks (0 = no window).
attn = [n for n in doc["block"]["nodes"] if n["op"] == "attention"]
span = max((n["window"] + n["sinks"]) if n["window"] else 0 for n in attn) if attn else 0
same, out_rows, first, past = 0, [], [], 0
for r in rows:
    ids = tok.encode("<|user|>" + r["prompt"] + "<|assistant|>")
    out = []
    for step in range(MAX_NEW):
        lg = forward(ids[-ctx:])[-1].copy()
        if step == 0:
            first.append(lg.copy())
        if unk is not None:
            lg[unk] = -np.inf  # mtstudio bans <unk> from generation
        t = int(lg.argmax())
        if t in stops:
            break
        ids.append(t)
        out.append(t)
    reply = tok.decode(out).strip()
    ok = reply == r["mtstudio"].strip()
    same += ok
    past += bool(span) and len(ids) > span
    out_rows.append({"i": r["i"], "ir_numpy": reply, "same": ok, "tokens": len(out), "seq_len": len(ids)})
    if not ok:
        print("DIFF", r["i"], repr(reply[:80]), "| mtstudio:", repr(r["mtstudio"][:80]))
mode = " (window disabled)" if NO_WINDOW else ""
print(f"numpy IR interpreter{mode} vs mtstudio greedy: {same}/{len(rows)} identical replies")
if span:
    print(f"sequences longer than window + sinks = {span}: {past}/{len(rows)}; "
          f"longest {max(o['seq_len'] for o in out_rows)} tokens (ctx {ctx})")
if len(argv) > 3:
    with open(argv[3], "w", encoding="utf-8") as f:
        for o in out_rows:
            f.write(json.dumps(o) + "\n")
    np.save(argv[3] + ".first_logits.npy", np.stack(first))
