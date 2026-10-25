"""mtstudio end to end on a CUDA VM: the device SFT evaluation and the gpt2 family under deferral.

LOCAL_LLM_STACK.md section 11, items 5.2 and 5.4 (defect D1, docs/CUDA_PHASE_B2.md). Runs the
VM's CUDA mtstudio on synthetic word-vocabulary data in pairs that differ in one switch, with the
job's MICROTORCH_* production environment otherwise (device op set, step residency, deferred
downloads, cuBLAS, blocked attention):

  sft     MICROTORCH_DEVICE_EVAL=0 (the windowless evaluation) vs default (the evaluation in a
          step window). Pass: every eval event of the default run says eval_device true and the
          other's false; each val_loss within 1e-6 relative (test_cuda_ops leg 9's bound for the
          device masked cross-entropy); training losses bit-identical (the evaluation path cannot
          touch training). Families: llama (eval_batch 1 and 4), gpt2 (ParityLM), flex (4 layers).
  timing  the same pair at two real shapes (d384 L6 T512 vocab 8192, the 20M/fw class; d768 L12
          T512 vocab 16384, pk_base_v1's), 100 held-out examples of ~370 tokens: eval_s of each,
          and the same parity rule.
  lm      MICROTORCH_DEFER_DOWNLOADS=1 vs 0 for the gpt2 family (exact, sliding-window, Kimi, SRD)
          and flex (layernorm/gelu, 4 layers), under MALLOC_CHECK_=3 MALLOC_PERTURB_=85 so a write
          into freed memory aborts. Pass: both runs exit 0; the first 10 training losses within
          1e-4 relative (tests/test_gpt2_device.cpp's step bound); every val_loss within 1e-3
          relative (60 steps of rounding-level drift); the loss falls.

Writes OUT/device_eval_gate.json ({"ok", "legs": [...]}) and each run's events and log under OUT.
Usage: python3 tools/device_eval_gate.py --bin /content/pk/build_cuda/mtstudio --out DIR [--only sft,lm,timing]
"""
import argparse
import json
import math
import os
import random
import struct
import subprocess
import sys
import time

EOT = "<|endoftext|>"


def gguf_str(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def write_vocab_gguf(path, tokens, eos_id):
    """Tokenizer-only GGUF (v3) with a word vocabulary (tests/test_mtstudio_chat.py's writer)."""
    meta = [gguf_str("tokenizer.ggml.model") + struct.pack("<I", 8) + gguf_str("word"),
            gguf_str("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, len(tokens))
            + b"".join(gguf_str(t) for t in tokens),
            gguf_str("tokenizer.ggml.eos_token_id") + struct.pack("<II", 4, eos_id)]
    with open(path, "wb") as f:
        f.write(struct.pack("<IIQQ", 0x46554747, 3, 0, len(meta)) + b"".join(meta))


def vocab(d, n_words):
    words = [f"w{i}" for i in range(n_words - 2)]
    write_vocab_gguf(os.path.join(d, "v.gguf"), ["<unk>", EOT] + words, 1)
    return words


def sft_data(d, words, n, plen, clen, seed):
    """n {prompt, completion} rows; each completion is a fixed function of its prompt (learnable)."""
    rng = random.Random(seed)
    k = len(words)
    with open(os.path.join(d, "ex.jsonl"), "w", encoding="utf-8") as f:
        for _ in range(n):
            p = [rng.randrange(k) for _ in range(rng.randint(plen // 2, plen))]
            c = [(p[i % len(p)] * 7 + 3) % k for i in range(rng.randint(clen // 2, clen))]
            f.write(json.dumps({"prompt": " ".join(words[i] for i in p),
                                "completion": " ".join(words[i] for i in c)}) + "\n")


def lm_corpus(d, words, n_tokens, seed):
    rng = random.Random(seed)
    k, out = len(words), []
    while len(out) < n_tokens:  # short repeated motifs: a text the model can learn in 60 steps
        m = [rng.randrange(k) for _ in range(rng.randint(3, 8))]
        out += m * rng.randint(2, 4)
    with open(os.path.join(d, "corpus.txt"), "w", encoding="utf-8") as f:
        f.write(" ".join(words[i] for i in out) + "\n")


def spec(d, name, arch, data, train):
    s = {"name": name, "arch": arch, "data": {"vocab": os.path.join(d, "v.gguf"), **data},
         "train": {"gradmap_every": 0, "seed": 3, **train}, "export": {"formats": []},
         "out_dir": os.path.join(d, name)}
    p = os.path.join(d, name + ".spec.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=1)
    return p


def run(binary, spec_path, env_over, log, timeout=7200):
    base = {k: v for k, v in os.environ.items()}
    base.setdefault("MICROTORCH_DEVICE", "cuda")
    base.setdefault("MICROTORCH_DEVICE_OPS", "1")
    base.setdefault("MICROTORCH_STEP_RESIDENCY", "1")
    base.setdefault("MICROTORCH_DEFER_DOWNLOADS", "1")
    base.setdefault("MICROTORCH_CUBLAS", "1")
    env = {**base, **env_over}
    t0 = time.time()
    try:
        p = subprocess.run([binary, "run", spec_path], env=env, capture_output=True, text=True, timeout=timeout)
        rc, text = p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired as e:
        rc, text = -1, f"timeout: {e}"
    with open(log, "w", encoding="utf-8") as f:
        f.write(text)
    s = json.load(open(spec_path, encoding="utf-8"))
    ev_path = os.path.join(s["out_dir"], "events.jsonl")
    events = [json.loads(x) for x in open(ev_path, encoding="utf-8") if x.strip()] if os.path.exists(ev_path) else []
    return {"rc": rc, "seconds": round(time.time() - t0, 1), "events": events,
            "env": {k: v for k, v in env.items() if k.startswith("MICROTORCH_") or k.startswith("MALLOC_")},
            "tail": text[-3000:] if rc != 0 else ""}


def steps_of(r):
    return [e["loss"] for e in r["events"] if e.get("event") == "step"]


def evals_of(r):
    return [e for e in r["events"] if e.get("event") == "eval"]


def relerr(a, b):
    return abs(a - b) / max(abs(b), 1e-30)


def sft_pair(binary, out, name, arch, data, train, words_n=64, n=400, plen=10, clen=10):
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    w = vocab(d, words_n)
    sft_data(d, w, n, plen, clen, seed=11)
    data = {"vocab_cap": words_n, "sft_examples": os.path.join(d, "ex.jsonl"), **data}
    sp_a = spec(d, "host_eval", arch, data, {"objective": "sft", **train})
    sp_b = spec(d, "device_eval", arch, data, {"objective": "sft", **train})
    a = run(binary, sp_a, {"MICROTORCH_DEVICE_EVAL": "0"}, os.path.join(d, "host_eval.log"))
    b = run(binary, sp_b, {}, os.path.join(d, "device_eval.log"))
    ea, eb = evals_of(a), evals_of(b)
    la, lb = steps_of(a), steps_of(b)
    vrel = [relerr(y["val_loss"], x["val_loss"]) for x, y in zip(ea, eb)]
    leg = {"leg": "sft", "name": name, "arch": arch, "train": train, "examples": n,
           "rc": [a["rc"], b["rc"]], "seconds": [a["seconds"], b["seconds"]],
           "evals": len(ea), "val_loss_host": [x["val_loss"] for x in ea], "val_loss_device": [y["val_loss"] for y in eb],
           "val_rel_max": max(vrel) if vrel else None,
           "eval_device_flags": [[x.get("eval_device") for x in ea], [y.get("eval_device") for y in eb]],
           "eval_s_host": [x.get("eval_s") for x in ea], "eval_s_device": [y.get("eval_s") for y in eb],
           "train_losses_bitwise_equal": la == lb, "steps": len(la), "env": [a["env"], b["env"]]}
    leg["ok"] = (a["rc"] == 0 and b["rc"] == 0 and len(ea) > 0 and len(ea) == len(eb)
                 and all(x.get("eval_device") is False for x in ea) and all(y.get("eval_device") is True for y in eb)
                 and max(vrel) <= 1e-6 and la == lb and len(la) == train["steps"])
    if not leg["ok"]:
        leg["tails"] = [a["tail"], b["tail"]]
    return leg


def lm_pair(binary, out, name, arch):
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    w = vocab(d, 64)
    lm_corpus(d, w, 40000, seed=5)
    data = {"vocab_cap": 64, "corpus": os.path.join(d, "corpus.txt"), "T": 32}
    train = {"steps": 60, "lr": 3e-3, "batch": 4, "eval_every": 20, "checkpoint_every": 1000}
    malloc = {"MALLOC_CHECK_": "3", "MALLOC_PERTURB_": "85"}
    sp_a = spec(d, "write_through", arch, data, train)
    sp_b = spec(d, "deferred", arch, data, train)
    a = run(binary, sp_a, {"MICROTORCH_DEFER_DOWNLOADS": "0", **malloc}, os.path.join(d, "write_through.log"))
    b = run(binary, sp_b, {"MICROTORCH_DEFER_DOWNLOADS": "1", **malloc}, os.path.join(d, "deferred.log"))
    la, lb = steps_of(a), steps_of(b)
    ea, eb = evals_of(a), evals_of(b)
    srel = [relerr(y, x) for x, y in zip(la[:10], lb[:10])]
    vrel = [relerr(y["val_loss"], x["val_loss"]) for x, y in zip(ea, eb)]
    finite = all(math.isfinite(x) for x in la + lb)
    leg = {"leg": "lm", "name": name, "arch": arch, "rc": [a["rc"], b["rc"]], "seconds": [a["seconds"], b["seconds"]],
           "steps": [len(la), len(lb)], "step_rel_max_first10": max(srel) if srel else None,
           "val_rel_max": max(vrel) if vrel else None,
           "loss_first_last": [[la[0], la[-1]] if la else None, [lb[0], lb[-1]] if lb else None],
           "env": [a["env"], b["env"]]}
    leg["ok"] = (a["rc"] == 0 and b["rc"] == 0 and len(la) == len(lb) == train["steps"] and finite
                 and max(srel) <= 1e-4 and len(ea) == len(eb) > 0 and max(vrel) <= 1e-3 and lb[-1] < lb[0])
    if not leg["ok"]:
        leg["tails"] = [a["tail"], b["tail"]]
    return leg


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=os.environ.get("JOBS_MTSTUDIO", "/content/pk/build_cuda/mtstudio"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="sft,lm,timing")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    only = set(a.only.split(","))
    legs = []
    res_path = os.path.join(a.out, "device_eval_gate.json")

    def save():
        res = {"ok": bool(legs) and all(x["ok"] for x in legs), "failed": [x["name"] for x in legs if not x["ok"]],
               "bin": a.bin, "legs": legs}
        tmp = res_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
        os.replace(tmp, res_path)
        return res

    small = {"steps": 40, "lr": 3e-3, "batch": 4, "eval_every": 20, "checkpoint_every": 1000}
    if "sft" in only:
        for name, arch, extra in (
                ("sft_llama", {"preset": "llama-tiny", "custom": {"d": 64, "layers": 2, "heads": 4}}, {}),
                ("sft_llama_eb4", {"preset": "llama-tiny", "custom": {"d": 64, "layers": 2, "heads": 4}},
                 {"eval_batch": 4}),
                ("sft_gpt2", {"preset": "gpt2-nano", "custom": {"d": 64, "heads": 4}}, {}),
                ("sft_flex", {"preset": "gpt2-small", "custom": {"d": 64, "heads": 4}}, {})):
            legs.append(sft_pair(a.bin, a.out, name, arch, {"T": 32}, {**small, **extra}))
            print(json.dumps({k: legs[-1].get(k) for k in ("name", "ok", "val_rel_max", "eval_s_host", "eval_s_device")}),
                  flush=True)
            save()
    if "lm" in only:
        for name, arch in (("lm_gpt2_exact", {"preset": "gpt2-nano", "custom": {"d": 64, "heads": 4}}),
                           ("lm_gpt2_swa", {"preset": "swa-tiny", "custom": {"d": 64, "heads": 4, "window": 8}}),
                           ("lm_gpt2_kimi", {"preset": "kimi-tiny", "custom": {"d": 64, "heads": 4}}),
                           ("lm_gpt2_srd", {"preset": "srd-tiny", "custom": {"d": 64, "heads": 4}}),
                           ("lm_flex4", {"preset": "gpt2-small", "custom": {"d": 64, "heads": 4}})):
            legs.append(lm_pair(a.bin, a.out, name, arch))
            print(json.dumps({k: legs[-1].get(k) for k in ("name", "ok", "step_rel_max_first10", "val_rel_max", "rc")}),
                  flush=True)
            save()
    if "timing" in only:
        one = {"steps": 2, "lr": 1e-3, "batch": 2, "eval_every": 2, "checkpoint_every": 1000}
        for name, arch, words_n in (
                ("timing_d384", {"preset": "llama-tiny", "custom": {"d": 384, "layers": 6, "heads": 6}}, 8192),
                ("timing_d768", {"preset": "llama-tiny", "custom": {"d": 768, "layers": 12, "heads": 12}}, 16384)):
            legs.append(sft_pair(a.bin, a.out, name, arch, {"T": 512}, one, words_n=words_n, n=2000, plen=230,
                                 clen=250))
            print(json.dumps({k: legs[-1].get(k) for k in ("name", "ok", "val_rel_max", "eval_s_host", "eval_s_device")}),
                  flush=True)
            save()
    res = save()
    print(json.dumps({"ok": res["ok"], "failed": res["failed"]}))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
