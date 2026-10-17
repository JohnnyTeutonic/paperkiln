#!/usr/bin/env python3
"""Training-throughput probe on one Colab GPU VM (the chat-model sizing
question, 17 Oct 2026). VM side: run detached by tools/colab_throughput_probe.sh.

How fast does mtstudio train as width, depth and batch grow, on the CUDA
build the studies use (MICROTORCH_DEVICE=cuda, MICROTORCH_DEVICE_OPS=1)?
The banked transfer runs give ~630 tokens/s at d=512, L=2, batch 4 on an
L4, with one process using ~1.2 cores and ~180 MiB of GPU. Whether larger
batches buy throughput decides whether a chat-scale model is trainable
without engine work first.

Everything comes from the pushed repository: paperkiln master and
coalfire.cpp (transformer_cpp) are cloned, and the corpus is fetched and
hash-verified by corpus/fetchers. Nothing is uploaded. Writes
/content/probe/results.json and /content/probe/PROBE_DONE.
"""
import json
import os
import subprocess
import time

ROOT = "/content/probe"
PK = "https://github.com/JohnnyTeutonic/paperkiln.git"
CF = "https://github.com/JohnnyTeutonic/coalfire.cpp.git"
STEPS = 20
GRID = [  # (d, layers, heads, batch)
    (256, 4, 4, 4), (256, 4, 4, 16), (256, 4, 4, 64),
    (384, 6, 6, 16), (384, 6, 6, 64),
    (512, 8, 8, 4), (512, 8, 8, 16), (512, 8, 8, 64),
]
# BLOCKED=1 (argv) runs every point with MICROTORCH_ATTN_BLOCKED=1 and writes
# results_blocked.json: the second probe, after the stacked path was found to
# cost O((B*T)^2).
import sys
BLOCKED = len(sys.argv) > 1 and sys.argv[1] == "blocked"


def sh(cmd, log=None, timeout=None, env=None):
    with open(log, "a") if log else open(os.devnull, "w") as f:
        return subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, timeout=timeout, env=env).returncode


def main():
    global RES
    RES = "results_blocked.json" if BLOCKED else "results.json"
    os.makedirs(ROOT, exist_ok=True)
    log = f"{ROOT}/probe.log"
    out = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "runs": []}
    sh(f"cd {ROOT} && rm -rf microtorch transformer_cpp && git clone -q --depth 1 {PK} microtorch && "
       f"git clone -q --depth 1 {CF} transformer_cpp", log)
    out["commit"] = subprocess.run(f"git -C {ROOT}/microtorch rev-parse HEAD", shell=True, capture_output=True,
                                   text=True).stdout.strip()
    out["gpu"] = subprocess.run("nvidia-smi -L", shell=True, capture_output=True, text=True).stdout.strip()
    rc = sh(f"mkdir -p {ROOT}/microtorch/build_cuda && cd {ROOT}/microtorch/build_cuda && "
            f"cmake .. -DCMAKE_BUILD_TYPE=Release -DMICROTORCH_CUDA=ON && make mtstudio -j$(nproc)", log)
    if rc:
        out["error"] = "build failed"
        json.dump(out, open(f"{ROOT}/{RES}", "w"), indent=1)
        open(f"{ROOT}/PROBE_DONE", "w").write("BUILD_FAILED")
        return
    mt = f"{ROOT}/microtorch"
    sh(f"cd {mt} && pip -q install huggingface_hub pandas pyarrow && python3 corpus/fetchers/tinystories.py valid_v1 && "
       f"python3 corpus/fetchers/vocab_tinystories_valid.py", log)
    corpus = f"{mt}/corpus/data/tinystories_valid_v1/tinystories_valid.txt"
    vocab = f"{mt}/corpus/data/vocab_tinystories_valid_v1/tinystories_vocab.gguf"
    out["data_ok"] = os.path.exists(corpus) and os.path.exists(vocab)
    env = dict(os.environ, MICROTORCH_DEVICE="cuda", MICROTORCH_DEVICE_OPS="1", OMP_NUM_THREADS="4")
    if BLOCKED:
        env["MICROTORCH_ATTN_BLOCKED"] = "1"
    out["attention_impl"] = "blocked" if BLOCKED else "stacked"
    for d, L, H, B in GRID:
        name = f"d{d}_L{L}_b{B}"
        spec = {"name": name, "arch": {"preset": "llama-tiny", "custom": {"d": d, "layers": L, "heads": H}},
                "data": {"corpus": corpus, "vocab": vocab, "vocab_cap": 4096, "T": 256, "max_tokens": 2000000},
                "train": {"steps": STEPS, "lr": 0.001, "batch": B, "eval_every": 1000, "checkpoint_every": 100000,
                          "gradmap_every": 0, "seed": 1},
                "export": {"formats": []}, "out_dir": f"{ROOT}/runs/{name}"}
        sp = f"{ROOT}/{name}.json"
        json.dump(spec, open(sp, "w"))
        t0 = time.time()
        try:
            rc = sh(f"{mt}/build_cuda/mtstudio run {sp}", f"{ROOT}/{name}.log", timeout=900, env=env)
        except subprocess.TimeoutExpired:
            rc = "timeout"
        wall = time.time() - t0
        row = {"d": d, "layers": L, "heads": H, "batch": B, "T": 256, "steps": STEPS, "rc": rc, "wall_s": wall}
        rp = f"{ROOT}/runs/{name}/result.json"
        if os.path.exists(rp):
            r = json.load(open(rp))
            row.update({k: r.get(k) for k in ("tokens_per_second", "params", "wall_seconds", "best_val")})
        row["gpu_mem"] = subprocess.run("nvidia-smi --query-gpu=memory.used --format=csv,noheader", shell=True,
                                        capture_output=True, text=True).stdout.strip()
        out["runs"].append(row)
        json.dump(out, open(f"{ROOT}/{RES}", "w"), indent=1)
    out["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(out, open(f"{ROOT}/{RES}", "w"), indent=1)
    open(f"{ROOT}/PROBE_DONE", "w").write("OK")


if __name__ == "__main__":
    main()
