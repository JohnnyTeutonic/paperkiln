#!/usr/bin/env python3
"""Where an L4 training step's time goes, per op class, for the tiny-chat shapes and pk_base_v1.

    python3 tools/step_profile.py run --out OUT --vocab VOCAB.gguf [--shapes all] [--configs fp32]
                                      [--steps 40] [--warm 10]
    python3 tools/step_profile.py table OUT/profile.json

`run` trains each shape for --steps steps on a synthetic corpus (uniform random ids: a step's cost
does not depend on the text) twice per config:
  1. clean: the production environment (jobs_vm.py's ENV), throughput from the step lines' arrival
     times between step --warm and the last step, as tiny_chat_run.py's probe measures it;
  2. profiled: the same plus MICROTORCH_STEP_TIMING=1 (per-phase wall time, a device sync at each
     phase mark) and MICROTORCH_GPU_PROFILE=1 (CUDA events around every device op and GEMM,
     summed per entry point). Per step, the wall time is the sum of the phases and the GPU time
     the sum of the ops; the difference is time the GPU sat idle inside the step: host work,
     launch gaps, host syncs and copies outside an op.
The entry points are grouped into classes (CLASSES below): the LM head's GEMMs (forward and
backward, tagged "gemm_head" by nn::Llama) and the cross-entropy/softmax kernels ("loss") each
have their own class, apart from the other GEMMs and the elementwise ops. Writes OUT/profile.json and prints the
table. A config other than fp32 sets train.precision / train.attention_kernel and is refused
(recorded, not run) unless the binary records the setting in its model event.
"""
import argparse
import json
import os
import random
import struct
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.environ.get("JOBS_MTSTUDIO", "/content/pk/build_cuda/mtstudio")
ENV = {"MICROTORCH_DEVICE": "cuda", "MICROTORCH_DEVICE_OPS": "1", "MICROTORCH_ATTN_BLOCKED": "1",
       "MICROTORCH_STEP_RESIDENCY": "1", "MICROTORCH_DEFER_DOWNLOADS": "1", "MICROTORCH_CUBLAS": "1",
       "OMP_NUM_THREADS": "4"}  # tools/colab_chat/jobs_vm.py ENV: what every production run trains with
SPECS = os.path.join(REPO, "experiments", "tiny_chat_2026-10", "specs")
SHAPES = ["20m_L2", "20m_L4", "20m_L8", "20m_L16", "20m_L32", "60m_L4", "60m_L8", "60m_L16", "60m_L32",
          "pk_base_v1"]
# pk_base_v1 (C:/ml_artifacts/next_base/base/base_spec.json): d768 L12, batch 16, accum 1, T 512.
PK_BASE = {"arch": {"preset": "llama-tiny", "custom": {"d": 768, "layers": 12, "heads": 12, "d_ff": 2048}},
           "data": {"vocab_cap": 16384, "T": 512, "max_tokens": 0},
           "train": {"lr": 0.0006, "batch": 16, "seed": 1, "optimizer": "adamw", "gradmap_every": 0,
                     "lr_schedule": {"warmup_steps": 2000, "decay": "cosine", "min_lr_frac": 0.1}}}
CONFIGS = {"fp32": {}, "bf16": {"precision": "bf16"}, "flash": {"attention_kernel": "flash"},
           "bf16_flash": {"precision": "bf16", "attention_kernel": "flash"}}
# Device entry point (MICROTORCH_GPU_PROFILE's op names) -> class. Unlisted names count as "other".
CLASSES = {
    "gemm": ["gemm", "gemm_bf16"],  # every GEMM but the LM head's
    "head_gemm": ["gemm_head", "gemm_head_bf16", "transpose"],  # logits = h E^T and its backward, E^T itself
    "bf16_convert": ["bf16_convert"],
    "attention_fwd": ["attn_heads_fwd", "attn_heads_fwd_ts", "attn_blocked_fwd", "attn_masked_softmax",
                      "swa_masked_softmax", "attn_flash_fwd"],
    "attention_bwd": ["attn_heads_bwd", "attn_blocked_bwd", "attn_softmax_bwd_inplace", "attn_flash_bwd"],
    "norms": ["rmsnorm_fwd", "rmsnorm_bwd", "layernorm_fwd", "layernorm_bwd"],
    "loss": ["ce_fwd", "ce_bwd", "ce_fwd_masked", "ce_bwd_masked", "logsumexp_fwd", "logsumexp_bwd"],
    "optimizer": ["adamw_step", "adamw_step_dev", "sgd_step", "sgd_step_dev", "sumsq"],
    "embedding": ["embed_gather", "embed_scatter_add"],
    "elementwise": ["add", "sub", "mul", "scale", "axpy", "fill", "sigmoid_fwd", "sigmoid_bwd", "silu_fwd",
                    "silu_bwd", "swiglu_fwd", "swiglu_bwd", "gelu_fwd", "gelu_bwd", "softmax_fwd", "softmax_bwd",
                    "rope_fwd", "rope_bwd", "slice_cols", "place_cols", "concat_cols", "accumulate_cols",
                    "gather_rows", "scatter_rows", "scatter_rows_multi", "scatter_rows_into", "mul_col_fwd",
                    "mul_col_dx", "mul_col_dc"],
}
OP_CLASS = {op: c for c, ops in CLASSES.items() for op in ops}


def shape_spec(name):
    if name == "pk_base_v1":
        return json.loads(json.dumps(PK_BASE))
    with open(os.path.join(SPECS, f"shape_{name}.json"), encoding="utf-8") as f:
        s = json.load(f)
    s.pop("_shape", None)
    return s


def synth_corpus(path, vocab=16384, n=8_000_000, seed=1):
    """A PKIDS001 file (tools/mtstudio.cpp read_corpus): uniform random uint16 ids."""
    if os.path.exists(path):
        return path
    rng = random.Random(seed)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(b"PKIDS001" + struct.pack("<IIQ", 2, vocab, n))
        chunk = 1 << 20
        for k in range(0, n, chunk):
            m = min(chunk, n - k)
            f.write(struct.pack(f"<{m}H", *(rng.randrange(vocab) for _ in range(m))))
    os.replace(tmp, path)
    return path


def events(run_dir):
    p = os.path.join(run_dir, "events.jsonl")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def run_once(spec, run_dir, env_extra, warm):
    """mtstudio run; returns (rc, steady seconds per step from line arrival times, events, log tail)."""
    os.makedirs(run_dir, exist_ok=True)
    spec = dict(spec, out_dir=run_dir)
    sp = run_dir.rstrip("/") + "_spec.json"
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=1)
    env = {k: v for k, v in os.environ.items() if not k.startswith("MICROTORCH_")}
    env.update(ENV)
    env.update(env_extra)
    marks, tail = {}, []
    p = subprocess.Popen([BIN, "run", sp], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                         env=env)
    for line in p.stdout:
        tail = (tail + [line])[-40:]
        if line.startswith('{"event":"step"'):
            try:
                st = json.loads(line)["step"]
            except (ValueError, KeyError):
                continue
            now = time.monotonic()
            if st >= warm:
                marks.setdefault("first", (st, now))
            marks["last"] = (st, now)
    rc = p.wait()
    sps = None
    if "first" in marks and marks["last"][0] > marks["first"][0]:
        (s0, t0), (s1, t1) = marks["first"], marks["last"]
        sps = (t1 - t0) / (s1 - s0)
    return rc, sps, events(run_dir), "".join(tail)


def breakdown(evs, warm):
    """Mean per-step ms: wall (sum of phases), phases, GPU per op and per class, and the idle rest."""
    steps = [e for e in evs if e.get("event") == "step" and e.get("step", 0) > warm and "t_ms" in e]
    if not steps:
        return None
    n = len(steps)
    phases, ops = {}, {}
    for e in steps:
        for k, v in e["t_ms"].items():
            phases[k] = phases.get(k, 0.0) + v / n
        for k, v in e.get("gpu_ms", {}).items():
            ops[k] = ops.get(k, 0.0) + v / n
    # "other" is eval/checkpoint/logging since the previous step: outside the training step.
    wall = sum(v for k, v in phases.items() if k != "other")
    classes = {}
    for op, ms in ops.items():
        c = OP_CLASS.get(op, "other")
        classes[c] = classes.get(c, 0.0) + ms
    gpu = sum(ops.values())
    return {"steps": n, "wall_ms": wall, "gpu_ms": gpu, "idle_ms": wall - gpu,
            "phases_ms": dict(sorted(phases.items())), "class_ms": dict(sorted(classes.items())),
            "op_ms": dict(sorted(ops.items(), key=lambda kv: -kv[1])),
            "peak_mb": max(e.get("peak_mb", 0.0) for e in steps)}


def config_recorded(evs, cfg):
    """True when the binary's model event carries every key the config sets (an older binary
    ignores unknown spec keys, which would time fp32 under another name)."""
    model = next((e for e in evs if e.get("event") == "model"), {})
    return all(model.get(k) == v for k, v in cfg.items())


def gpu_name():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                              text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None


def git_head():
    try:
        return subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None


def cmd_run(a):
    os.makedirs(a.out, exist_ok=True)
    corpus = synth_corpus(os.path.join(a.out, "synth.ids"))
    shapes = SHAPES if a.shapes == "all" else a.shapes.split(",")
    configs = a.configs.split(",")
    path = os.path.join(a.out, "profile.json")
    res = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    res.update({"commit": git_head(), "gpu": gpu_name(), "env": ENV, "steps": a.steps, "warm": a.warm})
    res.setdefault("runs", {})
    for shape in shapes:
        for cfg_name in configs:
            key = f"{shape}/{cfg_name}"
            if key in res["runs"] and res["runs"][key].get("done"):
                continue  # resumable: a finished cell is kept
            cfg = CONFIGS[cfg_name]
            spec = shape_spec(shape)
            spec["name"] = f"prof_{shape}_{cfg_name}"
            spec["data"].update({"corpus": corpus, "vocab": a.vocab})
            spec["train"].update({"steps": a.steps, "eval_every": 10 ** 7, "eval_windows": 1,
                                  "checkpoint_every": 10 ** 7}, **cfg)
            spec["export"] = {"formats": []}
            cell = {"shape": shape, "config": cfg_name, "arch": spec["arch"]["custom"],
                    "tokens_per_step": spec["train"]["batch"] * spec["train"].get("accum", 1) * spec["data"]["T"]}
            base = os.path.join(a.out, "runs", f"{shape}_{cfg_name}")
            rc, sps, evs, tail = run_once(spec, base + "_clean", {}, a.warm)
            cell.update({"clean_rc": rc, "clean_s_per_step": sps})
            if cfg and not config_recorded(evs, cfg):
                cell.update({"done": True, "refused": "this binary does not record " + json.dumps(cfg)})
            elif rc != 0:
                cell.update({"done": True, "log_tail": tail})
            else:
                if sps:
                    cell["clean_tok_per_s"] = cell["tokens_per_step"] / sps
                rc2, _, evs2, tail2 = run_once(spec, base + "_prof",
                                               {"MICROTORCH_STEP_TIMING": "1", "MICROTORCH_GPU_PROFILE": "1"}, a.warm)
                cell.update({"prof_rc": rc2, "profile": breakdown(evs2, a.warm), "done": True})
                if rc2 != 0:
                    cell["log_tail"] = tail2
                steps = [e for e in evs if e.get("event") == "step"]
                cell["losses"] = [e.get("loss") for e in steps]
            res["runs"][key] = cell
            with open(path + ".tmp", "w", encoding="utf-8") as f:
                json.dump(res, f, indent=1)
            os.replace(path + ".tmp", path)
            print(json.dumps({k: cell.get(k) for k in ("shape", "config", "clean_rc", "clean_tok_per_s",
                                                         "refused")}), flush=True)
    print(table(res))
    return 0


def table(res):
    cls = list(CLASSES) + ["other"]
    head = ["shape", "config", "tok/s", "ms/step", "gpu ms"] + cls + ["idle ms", "peak MB"]
    rows = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for key, c in res.get("runs", {}).items():
        p = c.get("profile")
        if not p:
            rows.append(f"| {c['shape']} | {c['config']} | " + (c.get("refused") or f"rc {c.get('clean_rc')}") +
                        " |" * (len(head) - 2))
            continue
        w = p["wall_ms"]
        pct = [f"{p['class_ms'].get(k, 0.0):.0f} ({100 * p['class_ms'].get(k, 0.0) / w:.0f}%)" for k in cls]
        rows.append("| " + " | ".join([c["shape"], c["config"], f"{c.get('clean_tok_per_s') or 0:.0f}",
                                       f"{w:.0f}", f"{p['gpu_ms']:.0f}"] + pct +
                                      [f"{p['idle_ms']:.0f} ({100 * p['idle_ms'] / w:.0f}%)",
                                       f"{p['peak_mb']:.0f}"]) + " |")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--vocab", required=True)
    r.add_argument("--shapes", default="all")
    r.add_argument("--configs", default="fp32")
    r.add_argument("--steps", type=int, default=40)
    r.add_argument("--warm", type=int, default=10)
    t = sub.add_parser("table")
    t.add_argument("profile")
    a = ap.parse_args()
    if a.cmd == "run":
        return cmd_run(a)
    print(table(json.load(open(a.profile, encoding="utf-8"))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
