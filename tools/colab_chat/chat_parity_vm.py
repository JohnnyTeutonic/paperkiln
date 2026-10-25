#!/usr/bin/env python3
"""Greedy parity (LOCAL_LLM_STACK.md section 3, point 1) of a model this session trained, on the VM:
ember.cpp's server (tiny_chat_run.EMBER_COMMIT, CUDA and CPU) against `mtstudio chat` on the CPU,
full recompute, the procedure of ember's experiments/pk_base_v1_2026-10-24/gate.py.

    python3 tools/colab_chat/chat_parity_vm.py OUT JOB [N=40]

1. mtstudio chat on JOB's own spec (/content/chat/JOB_spec.json) with data.vocab set to the run's
   f32 GGUF (the vocabulary the export embedded), MICROTORCH_DEVICE=cpu; /health must name the model.
2. CUDA leg: ember's server on JOB's f32 GGUF, all layers on the GPU, EMBER_GEMM_TF32=0; ember's own
   tools/chat_parity.py (from ember's tree, unchanged) on the first N frozen prompts
   (tinychat_hf_prompts_v1, greedy, 60 new tokens): reply text, token count, stop reason, and ember's
   prompt token count against paperkiln's tokeniser. OUT/parity_cuda.jsonl.
3. CPU leg: the same GGUF on ember's CPU path (GPU layers 0), the CUDA leg's prompts replayed through
   ember alone and compared with the recorded mtstudio replies (gate.py's "ref" job). OUT/parity_cpu.jsonl.
Writes OUT/parity_summary.json (counts per leg, the GGUF's sha256, both commits)."""
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request

C = "/content/chat"
REPO = "/content/pk"
MTS = f"{REPO}/build_cuda/mtstudio"
PE, PM = 8097, 8098
PROMPTS = f"{REPO}/corpus/eval/tinychat_hf_prompts_v1.jsonl"


def health(port):
    try:
        return json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2))
    except Exception:
        return None


def post(port, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=900))


def start(argv, port, log, env):
    if health(port) is not None:
        sys.exit(f"port {port} already answers /health; refusing to share it")
    p = subprocess.Popen(argv, stdout=open(log, "w"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env)
    for _ in range(900):
        if p.poll() is not None:
            sys.exit(f"{argv[0]} on {port} exited {p.returncode}; see {log}")
        h = health(port)
        if h is not None:
            return p, h
        time.sleep(1)
    p.kill()
    sys.exit(f"{argv[0]} on {port} did not come up")


def stop(p):
    p.terminate()
    try:
        p.wait(30)
    except subprocess.TimeoutExpired:
        p.kill()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    out, job = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    os.makedirs(out, exist_ok=True)
    spec = json.load(open(f"{C}/{job}_spec.json"))
    name = spec["name"]
    gguf = os.path.join(spec["out_dir"], name + ".gguf")
    spec["data"]["vocab"] = gguf
    # The reference is FP32 with the tiled kernel (mtstudio applies these only to `run` today; the
    # model trained under BF16 GEMMs is served in FP32 by both engines).
    for k in ("precision", "attention_kernel"):
        spec.get("train", {}).pop(k, None)
    spec_path = f"{out}/mtstudio_chat_spec.json"
    json.dump(spec, open(spec_path, "w"), indent=1)
    r = subprocess.run([sys.executable, f"{REPO}/tools/colab_chat/ensure_ember_vm.py", f"{out}/ember_build.json",
                        f"{out}/ember_build.log"], cwd=REPO)
    if r.returncode != 0:
        sys.exit("ember build failed")
    eb = json.load(open(f"{out}/ember_build.json"))
    ember_bin = eb["bin"]
    parity = os.path.join(os.path.dirname(os.path.dirname(ember_bin)), "src", "tools", "chat_parity.py")
    base_env = {k: v for k, v in os.environ.items() if not k.startswith("MICROTORCH_")}
    procs = []
    summary = {"model": name, "gguf": gguf, "gguf_sha256": sha256(gguf), "ember": eb,
               "paperkiln_commit": subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True,
                                                  text=True).stdout.strip(),
               "prompts": PROMPTS, "n": n, "max_new": 60, "mtstudio": "CPU, full recompute"}
    try:
        pm, hm = start([MTS, "chat", spec_path, "--port", str(PM)], PM, f"{out}/mtstudio.log",
                       {**base_env, "MICROTORCH_DEVICE": "cpu", "OMP_NUM_THREADS": str(os.cpu_count() or 4)})
        procs.append(pm)
        if hm.get("model") != name:
            sys.exit(f"mtstudio /health names {hm.get('model')!r}, not {name!r}")
        summary["mtstudio_health"] = hm
        for leg, layers in (("cuda", "-1"), ("cpu", "0")):
            t0 = time.time()
            pe, he = start([ember_bin, gguf, str(PE), "127.0.0.1", "-", layers], PE, f"{out}/ember_{leg}.log",
                           {**base_env, "EMBER_GEMM_TF32": "0"})
            if os.path.realpath(he.get("model", "")) != os.path.realpath(gguf):
                stop(pe)
                sys.exit(f"ember /health names {he.get('model')!r}, not {gguf}")
            rec_path = f"{out}/parity_{leg}.jsonl"
            try:
                if leg == "cuda":
                    env = {**base_env, "PARITY_EMBER_PORT": str(PE), "PARITY_MTS_PORT": str(PM), "PARITY_GGUF": gguf,
                           "PARITY_PROMPTS": PROMPTS, "PARITY_MAX_NEW": "60", "PAPERKILN_TOOLS": f"{REPO}/tools"}
                    p = subprocess.run([sys.executable, parity, str(n), rec_path], env=env, capture_output=True,
                                       text=True)
                    open(f"{out}/chat_parity_cuda.log", "w").write(p.stdout + p.stderr)
                    if p.returncode != 0:
                        sys.exit(f"chat_parity.py failed ({p.returncode}); see chat_parity_cuda.log")
                    rows = [json.loads(x) for x in open(rec_path)]
                    ref_rows = rows
                else:
                    rows = []
                    for r0 in ref_rows:
                        e = post(PE, {"user_input": r0["prompt"], "max_new_tokens": 60, "temperature": 1.0,
                                      "top_k": 1, "top_p": 1.0, "seed": 0})
                        rows.append({**r0, "ember": e["reply"], "ember_tokens": e["tokens"],
                                     "ember_stop": e["stop_reason"], "ember_prompt_tokens": e["prompt_tokens"],
                                     "ember_cuda": r0["ember"]})
                    with open(rec_path, "w") as f:
                        for x in rows:
                            f.write(json.dumps(x) + "\n")
            finally:
                stop(pe)
            summary[leg] = {"n": len(rows), "health": he, "seconds": round(time.time() - t0, 1),
                            "reply_identical": sum(x["ember"] == x["mtstudio"] for x in rows),
                            "token_count": sum(x["ember_tokens"] == x["mts_tokens"] for x in rows),
                            "stop_reason": sum(x["ember_stop"] == x["mts_stop"] for x in rows),
                            "prompt_tokens": sum(x["ember_prompt_tokens"] == x["bpe_py_prompt_tokens"] for x in rows),
                            "distinct_replies": len({x["mtstudio"] for x in rows})}
            print(leg, json.dumps(summary[leg]), flush=True)
    finally:
        for p in procs:
            stop(p)
    json.dump(summary, open(f"{out}/parity_summary.json", "w"), indent=1)
    ok = all(summary[leg][k] == n for leg in ("cuda", "cpu") for k in ("reply_identical", "token_count",
                                                                        "stop_reason", "prompt_tokens"))
    print("PARITY", "PASS" if ok else "FAIL", flush=True)


if __name__ == "__main__":
    main()
