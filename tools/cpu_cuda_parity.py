"""CUDA mtstudio against the CPU build on trained models (LOCAL_LLM_STACK.md section 11, item 4.1).

    python3 tools/cpu_cuda_parity.py --cpu-bin build_cpu/mtstudio --cuda-bin build_cuda/mtstudio \\
        --model live=/content/eval/live/spec.json --model moe=... --out DIR [--n 40] [--max-new 60]

Per model, the two builds of the same commit, the CUDA one under the production environment
(MICROTORCH_DEVICE=cuda, device op set, step residency, deferred downloads, cuBLAS, blocked
attention: what the evaluations of section 11 item 2.1 run), the CPU one with no MICROTORCH_*:
  greedy  `mtstudio chat` on the frozen prompts (corpus/eval/tinychat_hf_prompts_v1.jsonl, the first
          n), top_k 1, max_new tokens: reply text, token count and stop reason, N/N.
  logits  `mtstudio score --logits` on each prompt + the CPU build's greedy reply (+ <|endoftext|>
          when it stopped on it): the argmax at every scored position equal on both builds, and the
          largest |logit difference| over the CPU row's logit range (ROUNDING: 1e-5 for models this
          shallow, section 5.3) and the largest |log-prob difference|, both reported.
  sft     the reference bits/byte path (tools/chat_eval.py ref_bpb): a 1-step lr-0 SFT run whose
          held-out evaluation is the n prompts' references; val_loss within 1e-5 relative (on CUDA
          this is the device SFT evaluation, item 5.2).
Pass: greedy N/N, top-1 at every position, logits within 1e-5 of the range, sft within 1e-5.
Writes OUT/cpu_cuda_parity.json and per-model files under OUT/<tag>/.
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import bpe  # noqa: E402

PROD = {"MICROTORCH_DEVICE": "cuda", "MICROTORCH_DEVICE_OPS": "1", "MICROTORCH_STEP_RESIDENCY": "1",
        "MICROTORCH_DEFER_DOWNLOADS": "1", "MICROTORCH_CUBLAS": "1", "MICROTORCH_ATTN_BLOCKED": "1"}
EOT = "<|endoftext|>"


DRY = False  # --dry-cpu: the "cuda" side runs with the CPU environment (a local test of this tool)


def envs():
    base = {k: v for k, v in os.environ.items() if not k.startswith("MICROTORCH_")}
    return {"cpu": base, "cuda": base if DRY else {**base, **PROD}}


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))


def serve(binary, spec, env, log):
    port = free_port()
    p = subprocess.Popen([binary, "chat", spec, "--port", str(port)], env=env, stdout=open(log, "w"),
                         stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < 600:
        if p.poll() is not None:
            raise SystemExit(f"{binary} chat {spec} exited {p.returncode}; see {log}")
        try:
            h = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5))
            return p, port, h
        except OSError:
            time.sleep(2)
    p.kill()
    raise SystemExit(f"{binary} chat {spec}: no /health after 600 s")


def greedy(bins, spec, prompts, max_new, d):
    out = {}
    for b in ("cpu", "cuda"):
        p, port, h = serve(bins[b], spec, envs()[b], os.path.join(d, f"chat_{b}.log"))
        try:
            rows = []
            for i, pr in enumerate(prompts):
                r = post(f"http://127.0.0.1:{port}/chat", {"user_input": pr, "max_new_tokens": max_new,
                                                            "temperature": 1.0, "top_k": 1, "top_p": 1.0, "seed": 0})
                rows.append({"i": i, "reply": r["reply"], "tokens": r["tokens"], "stop": r["stop_reason"]})
            out[b] = {"health": h, "rows": rows}
        finally:
            p.terminate()
            p.wait(60)
    same = [a == c for a, c in zip(out["cpu"]["rows"], out["cuda"]["rows"])]
    with open(os.path.join(d, "greedy.jsonl"), "w", encoding="utf-8") as f:
        for a, c in zip(out["cpu"]["rows"], out["cuda"]["rows"]):
            f.write(json.dumps({"cpu": a, "cuda": c, "same": a == c}) + "\n")
    return out, sum(same)


def logits(bins, spec, prompts, cpu_rows, vocab_n, d):
    req = os.path.join(d, "score_in.jsonl")
    keep = []
    with open(req, "w", encoding="utf-8") as f:
        for pr, r in zip(prompts, cpu_rows):
            cont = r["reply"] + (EOT if r["stop"] == "eos" else "")
            if not cont:
                continue
            keep.append(r["i"])
            f.write(json.dumps({"messages": [{"role": "user", "content": pr}], "continuation": cont,
                                "key": r["i"]}) + "\n")
    res = {}
    for b in ("cpu", "cuda"):
        o, lf = os.path.join(d, f"score_{b}.jsonl"), os.path.join(d, f"logits_{b}.f32")
        p = subprocess.run([bins[b], "score", spec, "--in", req, "--out", o, "--logits", lf], env=envs()[b],
                           capture_output=True, text=True, timeout=7200)
        if p.returncode != 0:
            raise SystemExit(f"score on {b} failed: {p.stderr[-2000:]}")
        recs = [json.loads(x) for x in open(o, encoding="utf-8") if x.strip()]
        res[b] = (recs, np.fromfile(lf, dtype="<f4").reshape(-1, vocab_n))
    (rc, lc), (rg, lg) = res["cpu"], res["cuda"]
    if lc.shape != lg.shape:
        raise SystemExit(f"logits shapes differ: {lc.shape} vs {lg.shape}")
    rng = lc.max(axis=1) - lc.min(axis=1)
    err = np.abs(lg.astype(np.float64) - lc.astype(np.float64)).max(axis=1) / rng
    top1 = sum(int(a == b) for x, y in zip(rc, rg) for a, b in zip(x["argmax"], y["argmax"]))
    npos = sum(len(x["argmax"]) for x in rc)
    dlp = max(abs(a - b) for x, y in zip(rc, rg) for a, b in zip(x["logprobs"], y["logprobs"]))
    # the smallest top-1 margin on the CPU side, so a top-1 disagreement can be told from a near-tie
    srt = np.sort(lc, axis=1)
    margin = float((srt[:, -1] - srt[:, -2]).min() / rng[np.argmin(srt[:, -1] - srt[:, -2])])
    for f in ("logits_cpu.f32", "logits_cuda.f32"):
        os.remove(os.path.join(d, f))  # large; the numbers above are the receipt
    return {"scored": len(keep), "positions": npos, "top1_equal": top1, "logit_err_max": float(err.max()),
            "logit_err_median": float(np.median(err)), "logprob_diff_max": dlp, "cpu_min_top1_margin_rel": margin,
            "argmax_cont_cpu": sum(int(a == c) for x in rc for a, c in zip(x["argmax"], x["continuation_ids"]))}


def sft_eval(bins, spec_path, rows, d):
    spec = json.load(open(spec_path, encoding="utf-8"))
    tok = bpe.read_gguf(spec["data"]["vocab"])
    room = spec["data"]["T"] + 1
    ex = [{"prompt": r["prompt"], "completion": r["reference"]} for r in rows
          if len(tok.encode(r["prompt"])) + len(tok.encode(r["reference"])) + 8 < room]
    path = os.path.join(d, "ref.jsonl")
    with open(path, "w", encoding="utf-8", newline="\n") as f:  # the held-out tail (1/20) is exactly ex
        for e in [ex[0]] * (19 * len(ex)) + ex:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    out = {}
    for b in ("cpu", "cuda"):
        ev = {"name": f"refeval_{b}", "arch": spec["arch"],
              "data": {"vocab": spec["data"]["vocab"], "vocab_cap": spec["data"].get("vocab_cap", 8192),
                       "T": spec["data"]["T"], "sft_examples": path},
              "train": {"steps": 1, "lr": 0.0, "batch": 1, "eval_every": 1, "checkpoint_every": 1000,
                        "gradmap_every": 0, "seed": 1, "objective": "sft",
                        "init_from": os.path.join(spec["out_dir"], spec["name"] + ".safetensors")},
              "export": {"formats": []}, "out_dir": os.path.join(d, f"refeval_{b}")}
        sp = os.path.join(d, f"refeval_{b}.json")
        json.dump(ev, open(sp, "w"))
        p = subprocess.run([bins[b], "run", sp], env=envs()[b], capture_output=True, text=True, timeout=7200)
        if p.returncode != 0:
            raise SystemExit(f"sft eval on {b} failed: {p.stderr[-2000:]}")
        e = next(json.loads(x) for x in open(os.path.join(ev["out_dir"], "events.jsonl")) if '"eval"' in x)
        out[b] = e
    vc, vg = out["cpu"]["val_loss"], out["cuda"]["val_loss"]
    return {"examples": len(ex), "val_cpu": vc, "val_cuda": vg, "rel": abs(vg - vc) / abs(vc),
            "cuda_eval_device": out["cuda"].get("eval_device"), "cuda_eval_s": out["cuda"].get("eval_s")}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-bin", required=True)
    ap.add_argument("--cuda-bin", required=True)
    ap.add_argument("--model", action="append", required=True, help="TAG=spec.json")
    ap.add_argument("--prompts", default=os.path.join(ROOT, "corpus/eval/tinychat_hf_prompts_v1.jsonl"))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--max-new", type=int, default=60)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-cpu", action="store_true", help="both sides on the CPU environment (tests this tool)")
    a = ap.parse_args(argv)
    global DRY
    DRY = a.dry_cpu
    os.makedirs(a.out, exist_ok=True)
    rows = [json.loads(x) for x in open(a.prompts, encoding="utf-8") if x.strip()][:a.n]
    prompts = [r["prompt"] for r in rows]
    bins = {"cpu": a.cpu_bin, "cuda": a.cuda_bin}
    res = {"n": len(prompts), "max_new": a.max_new, "prompts": a.prompts, "cuda_env": PROD, "models": {}}
    path = os.path.join(a.out, "cpu_cuda_parity.json")
    for m in a.model:
        tag, spec = m.split("=", 1)
        d = os.path.join(a.out, tag)
        os.makedirs(d, exist_ok=True)
        t0 = time.time()
        g, same = greedy(bins, spec, prompts, a.max_new, d)
        rec = {"spec": spec, "health_cuda": g["cuda"]["health"], "greedy_identical": same}
        rec["logits"] = logits(bins, spec, prompts, g["cpu"]["rows"], g["cpu"]["health"]["vocab"], d)
        rec["sft"] = sft_eval(bins, spec, rows, d)
        L = rec["logits"]
        rec["ok"] = (same == len(prompts) and L["top1_equal"] == L["positions"] and L["logit_err_max"] <= 1e-5
                     and rec["sft"]["rel"] <= 1e-5)
        rec["seconds"] = round(time.time() - t0, 1)
        res["models"][tag] = rec
        print(json.dumps({"model": tag, "ok": rec["ok"], "greedy": same, "top1": [L["top1_equal"], L["positions"]],
                          "logit_err_max": L["logit_err_max"], "sft_rel": rec["sft"]["rel"]}), flush=True)
        res["ok"] = all(x["ok"] for x in res["models"].values())
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
        os.replace(path + ".tmp", path)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
