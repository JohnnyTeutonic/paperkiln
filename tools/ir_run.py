"""Reference interpreter for the paperkiln IR embedded in a GGUF (numpy, f32
tensors only): an independent check that the exported IR alone reproduces
mtstudio's greedy replies. Each step re-runs the last ctx tokens from
position 0, as mtstudio does (recurrent ops therefore scan from a zero
state every step). Ops: embedding, pos_embedding, linear, layernorm,
rmsnorm, gelu, silu, relu, sigmoid, add, sub, mul, mul_col, affine, scale,
rms_row, split, rope, moe_ffn, attention (causal, or a sliding window with
sinks), linear_attention, list, append, depth_attend, ssm_scan, expr
(spec/tools/ir_expr.py); any other op stops with its name. 40/40 vs mtstudio on chat_sftfinal_v3, moe_tiny_v3
and gpt2_tiny (20 Oct 2026), and swa_tiny with replies up to 200 tokens.

    python tools/ir_run.py model.gguf receipt.jsonl tools/ [out.jsonl]
        [--max-new N] [--no-window]

receipt.jsonl rows need "i", "prompt" and "mtstudio" (the recorded reply).
--max-new: the generation cap mtstudio was asked for (default 60).
--no-window: run every attention node with window 0 (dense causal), the
control that shows the window changes replies.
--repeat-penalty P [--repeat-last-n N]: the contract repetition penalty
(LOCAL_LLM_STACK.md section 3 point 4; default N 64), applied before the
<unk> ban, so a receipt recorded with it can be checked here."""
import json
import sys

import numpy as np

argv, MAX_NEW, NO_WINDOW, REP_P, REP_N = [], 60, False, 1.0, 64
_it = iter(sys.argv[1:])
for _a in _it:
    if _a == "--max-new":
        MAX_NEW = int(next(_it))
    elif _a == "--no-window":
        NO_WINDOW = True
    elif _a == "--repeat-penalty":
        REP_P = float(next(_it))
    elif _a == "--repeat-last-n":
        REP_N = int(next(_it))
    else:
        argv.append(_a)

sys.path.insert(0, argv[2])
from pathlib import Path  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spec" / "tools"))  # keramos (the IR reference)
import bpe  # noqa: E402
import ir_check  # noqa: E402
import ir_numpy  # noqa: E402  (the op kernels)

path = argv[0]
meta, tensors, data_start, raw, _ = ir_check.read_gguf(path)
doc = json.loads(meta["paperkiln.ir"][1])
W = {}
for name, dims, typ, off in tensors:
    assert typ == 0, name
    n = int(np.prod(dims))
    W[name] = np.frombuffer(raw, dtype=np.float32, count=n, offset=data_start + off).reshape(list(reversed(dims)))


def forward(ids):
    return ir_numpy.forward(doc, W, ids, no_window=NO_WINDOW)


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
        if REP_P != 1.0 and REP_N > 0:
            # Each distinct id among the last REP_N conditioning ids, stop set
            # exempt: l > 0 ? l / p : l * p, one float32 operation.
            p32 = np.float32(REP_P)
            for t_id in set(ids[-ctx:][-REP_N:]) - stops:
                lg[t_id] = lg[t_id] / p32 if lg[t_id] > 0 else lg[t_id] * p32
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
