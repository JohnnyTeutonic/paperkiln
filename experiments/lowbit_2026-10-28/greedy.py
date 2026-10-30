#!/usr/bin/env python3
"""Greedy agreement of quantised GGUFs with their F32 file on the 40 frozen chat prompts, in ember on the CPU
(LOCAL_LLM_WORK R3; experiments/lowbit_2026-10-28).

    python3 greedy.py run TAG VARIANT MODEL.gguf OUT_DIR      replies of one file (resumes per prompt)
    python3 greedy.py compare OUT_DIR TAG                     each variant against TAG's f32 replies

ember's tinyllama_server (EMBER_BIN; ember 757e36b, which reads Q8_0, Q4_0, Q5_0/Q5_1 and real Q4_K/Q5_K/Q6_K
blocks and multiplies dequantised values in F32), GPU layers 0 and CUDA hidden (no local GPU), /chat with
top_k 1 and max_new_tokens 60 on tinychat_hf_prompts_v1.jsonl, as ember's k-quant gate
(ember.cpp/experiments/kquant_2026-10-25/gate.py). Per variant: replies identical to F32, first reply token
identical (each reply re-encoded with the model's own vocabulary), and the mean common prefix in tokens.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "tools"))
EMBER = os.environ.get("EMBER_BIN", os.path.expanduser("~/r3_ember/tinyllama_server_757e36b"))
PROMPTS = os.environ.get("R3_PROMPTS", "/mnt/c/ml_artifacts/chat_base/sft/tinychat_hf_prompts_v1.jsonl")
PORT = int(os.environ.get("R3_PORT", "8131"))
MAX_NEW = 60


def post(body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))


def health():
    try:
        return urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).status == 200
    except OSError:
        return False


def body(prompt, max_new):
    return {"user_input": prompt, "max_new_tokens": max_new, "temperature": 1.0, "top_k": 1, "top_p": 1.0, "seed": 0}


def serve(gguf, out, tag, extra_env):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "2"),
           **extra_env}
    log = open(os.path.join(out, f"{tag}.server.log"), "a")
    p = subprocess.Popen([EMBER, gguf, str(PORT), "127.0.0.1", "-", "0"], env=env, stdout=log,
                         stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=out)
    for _ in range(900):
        if health():
            return p
        if p.poll() is not None:
            raise SystemExit(f"ember exited (rc {p.returncode}); {out}/{tag}.server.log")
        time.sleep(1)
    p.kill()
    raise SystemExit("ember did not come up")


def stop(p):
    p.terminate()
    try:
        p.wait(60)
    except subprocess.TimeoutExpired:
        p.kill()


def run(tag, variant, gguf, out):
    """First-token logits of the 40 prompts (TINYLLAMA_LOGITS_DUMP), then the 60-token greedy replies."""
    os.makedirs(out, exist_ok=True)
    rows = [json.loads(x)["prompt"] for x in open(PROMPTS, encoding="utf-8")][:40]
    name = f"{tag}_{variant}"
    dump = os.path.join(out, name + ".logits")
    if not (os.path.exists(dump) and sum(1 for _ in open(dump)) == len(rows)):
        if os.path.exists(dump):
            os.remove(dump)
        p = serve(gguf, out, name + "_probe", {"TINYLLAMA_LOGITS_DUMP": dump})
        try:
            for pr in rows:
                post(body(pr, 1))
        finally:
            stop(p)
    path = os.path.join(out, name + ".jsonl")
    done = sum(1 for _ in open(path)) if os.path.exists(path) else 0
    if done >= len(rows):
        return
    p = serve(gguf, out, name, {})
    try:
        with open(path, "a") as f:
            for i in range(done, len(rows)):
                e = post(body(rows[i], MAX_NEW))
                f.write(json.dumps({"i": i, "reply": e["reply"], "tokens": e["tokens"], "stop": e["stop_reason"],
                                    "prompt_tokens": e["prompt_tokens"]}) + "\n")
                f.flush()
    finally:
        stop(p)


def logits(path):
    return [[float(x) for x in line.split()] for line in open(path)]


def compare(out, tag, gguf_for_vocab):
    import llama3_bpe
    tok = llama3_bpe.from_gguf(gguf_for_vocab)
    enc = lambda s: tok.encode(s or "", add_special_tokens=False).ids  # noqa: E731
    ref = [json.loads(x) for x in open(os.path.join(out, f"{tag}_f32.jsonl"))]
    res = {}
    for v in ("q8_0_all", "q4_k_m", "q4_k_s", "q4_0"):
        p = os.path.join(out, f"{tag}_{v}.jsonl")
        if not os.path.exists(p):
            continue
        rows = [json.loads(x) for x in open(p)]
        same = first = 0
        pre = []
        for a, b in zip(ref, rows):
            same += a["reply"] == b["reply"]
            ia, ib = enc(a["reply"]), enc(b["reply"])
            first += (ia[:1] == ib[:1])
            k = 0
            while k < min(len(ia), len(ib)) and ia[k] == ib[k]:
                k += 1
            pre.append(k)
        la, lb = logits(os.path.join(out, f"{tag}_f32.logits")), logits(os.path.join(out, f"{tag}_{v}.logits"))
        top = sum(max(range(len(x)), key=x.__getitem__) == max(range(len(y)), key=y.__getitem__) for x, y in zip(la, lb))
        dmax = max(max(abs(p - q) for p, q in zip(x, y)) for x, y in zip(la, lb))
        rng = max(max(x) - min(x) for x in la)
        res[v] = {"n": len(rows), "replies_identical": same, "first_token_identical": first,
                  "first_token_argmax_identical": top, "max_abs_dlogit": round(dmax, 4),
                  "max_abs_dlogit_over_range": round(dmax / rng, 5),
                  "mean_common_prefix_tokens": round(sum(pre) / max(len(pre), 1), 2),
                  "f32_mean_reply_tokens": round(sum(len(enc(a["reply"])) for a in ref) / len(ref), 2)}
    return res


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(*sys.argv[2:6])
    else:
        print(json.dumps(compare(sys.argv[2], sys.argv[3], sys.argv[4]), indent=1))
