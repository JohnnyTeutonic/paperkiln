#!/usr/bin/env python3
"""pk_sft_v1 in Ollama (LOCAL_LLM_STACK.md section 11 item 6), CPU, in WSL, nothing pulled or pushed.

    python3 ollama_gate.py --ollama ~/ollama/bin/ollama --work ~/cp_ollama --out OUT [--threads 2]

1. The Modelfile is paperkiln's (tools/chat_template.py modelfile(), byte-identical to the one mtstudio
   writes beside an F32 export, tests/test_chat_template.cpp), FROM the F32 GGUF; `ollama create
   pk-sft-v1 -f Modelfile` imports it into a private OLLAMA_MODELS under WORK.
2. Rendering: the 60 conversations of llamacpp_gate.py (single turn, multi-turn, a system message) go
   through /api/chat and, rendered by tools/chat_template.py render(), through /api/generate raw: the
   prompt lengths (UTF-8 bytes) that OLLAMA_DEBUG=1 logs must equal paperkiln's rendering's, and the 24-token greedy
   continuations must be equal.
3. Chat: /api/chat greedy (temperature 0, top_k 1, repeat_penalty 1, num_ctx 512, num_predict 60) on the 40
   frozen prompts against ember's and mtstudio's pk_sft_v1 replies
   (C:/ml_artifacts/general_chat/parity/parity_cpu.jsonl).
The model is removed (`ollama rm`) and WORK/models deleted at the end; the receipts stay in OUT.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(MT, "tools"))
sys.path.insert(0, HERE)
import chat_template as CT  # noqa: E402
import llama3_bpe  # noqa: E402
from llamacpp_gate import PROMPTS, REF_SFT, SFT, conversations, jl, save, sha256  # noqa: E402

HOST = "127.0.0.1:11535"
NAME = "pk-sft-v1"


def post(path, body, timeout=900):
    req = urllib.request.Request(f"http://{HOST}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ollama", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    build = os.path.join(a.work, "build")
    os.makedirs(build, exist_ok=True)
    link = os.path.join(build, "pk_sft_v1.gguf")
    if not os.path.lexists(link):
        os.symlink(SFT, link)
    mf = CT.modelfile("pk_sft_v1.gguf")
    open(os.path.join(build, "Modelfile"), "w", encoding="utf-8").write(mf)
    open(os.path.join(a.out, "pk_sft_v1.Modelfile"), "w", encoding="utf-8").write(mf)
    env = {**os.environ, "OLLAMA_HOST": HOST, "OLLAMA_MODELS": os.path.join(a.work, "models"), "OLLAMA_DEBUG": "1",
           "OLLAMA_NOPRUNE": "1", "OLLAMA_NUM_PARALLEL": "1", "OLLAMA_MAX_LOADED_MODELS": "1",
           "OLLAMA_KEEP_ALIVE": "30m", "CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": ""}
    log_path = os.path.join(a.out, "ollama_serve.log")
    logf = open(log_path, "w")
    srv = subprocess.Popen([a.ollama, "serve"], env=env, stdout=logf, stderr=subprocess.STDOUT)
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://{HOST}/api/version", timeout=5)
                break
            except Exception:
                time.sleep(0.5)
        version = json.load(urllib.request.urlopen(f"http://{HOST}/api/version"))["version"]
        cr = subprocess.run([a.ollama, "create", NAME, "-f", "Modelfile"], cwd=build, env=env, capture_output=True,
                            text=True)
        open(os.path.join(a.out, "ollama_create.log"), "w").write(cr.stdout + cr.stderr)
        if cr.returncode != 0:
            raise SystemExit(f"ollama create failed: {cr.stderr[-800:]}")
        show = post("/api/show", {"model": NAME})
        opts = {"temperature": 0.0, "top_k": 1, "top_p": 1.0, "min_p": 0.0, "repeat_penalty": 1.0,
                "num_ctx": 512, "seed": 0, "num_thread": a.threads}

        # 2. rendering. Ollama 0.40's debug log gives each llama-server completion request's prompt_len, not
        # its text, so each conversation goes through /api/chat and, rendered by paperkiln, through
        # /api/generate with raw (no template): the two prompts must have the same length as paperkiln's
        # rendering, and the two greedy continuations must be equal.
        tok = llama3_bpe.from_gguf(SFT)

        def call(path, body):
            mark = os.path.getsize(log_path)
            out = post(path, body)
            time.sleep(0.3)
            logf.flush()
            with open(log_path, encoding="utf-8", errors="replace") as f:
                f.seek(mark)
                lens = re.findall(r'msg="llama-server completion request".*?prompt_len=(\d+)', f.read())
            return out, int(lens[-1]) if lens else None

        trows = []
        for i, m in enumerate(conversations()):
            mine = CT.render(m, generation_prompt=True)
            c, lc = call("/api/chat", {"model": NAME, "messages": m, "stream": False,
                                       "options": {**opts, "num_predict": 24}})
            g, lg = call("/api/generate", {"model": NAME, "prompt": mine, "raw": True, "stream": False,
                                           "options": {**opts, "num_predict": 24}})
            row = {"i": i, "turns": sum(x["role"] == "user" for x in m), "system": m[0]["role"] == "system",
                   "paperkiln_bytes": len(mine.encode("utf-8")), "paperkiln_tokens": len(tok.encode(mine, add_special_tokens=False).ids),
                   "chat_prompt_len": lc, "raw_prompt_len": lg, "chat_prompt_eval_count": c.get("prompt_eval_count"),
                   "chat_reply": c["message"]["content"], "raw_reply": g["response"]}
            row["length_identical"] = lc == lg == len(mine.encode("utf-8"))  # Go's len: UTF-8 bytes
            row["reply_identical"] = row["chat_reply"] == row["raw_reply"]
            trows.append(row)
        # 3. chat against ember
        refs = jl(REF_SFT)
        crows = []
        for i, r in enumerate(jl(PROMPTS)[:40]):
            ref = refs[i]
            out = post("/api/chat", {"model": NAME, "messages": [{"role": "user", "content": r["prompt"]}],
                                     "stream": False, "options": {**opts, "num_predict": 60}})
            reply = out["message"]["content"]
            crows.append({"i": i, "prompt": r["prompt"], "ollama": reply, "ember": ref["ember"],
                          "mtstudio": ref["mtstudio"], "reply_identical": reply == ref["ember"],
                          "done_reason": out.get("done_reason"), "ember_stop": ref["ember_stop"],
                          "eval_count": out.get("eval_count"), "ember_tokens": ref["ember_tokens"],
                          "prompt_eval_count": out.get("prompt_eval_count"),
                          "ember_prompt_tokens": ref["ember_prompt_tokens"]})
        save(a.out, "ollama_template.jsonl", trows)
        save(a.out, "ollama_chat.jsonl", crows)
        save(a.out, "ollama_summary.json", {
            "ollama": version, "model": SFT, "sha256": sha256(SFT),
            "modelfile_sha256": hashlib.sha256(mf.encode()).hexdigest(),
            "show_template": show.get("template", "")[:200], "show_parameters": show.get("parameters"),
            "show_details": show.get("details"), "options": opts,
            "template": {"n": len(trows), "with_system": sum(r["system"] for r in trows),
                         "prompt_length_identical": sum(r["length_identical"] for r in trows),
                         "chat_equals_raw_render": sum(r["reply_identical"] for r in trows),
                         "chat_prompt_eval_count_equals_paperkiln_tokens": sum(
                             r["chat_prompt_eval_count"] == r["paperkiln_tokens"] for r in trows),
                         "differ": [r["i"] for r in trows if not (r["length_identical"] and r["reply_identical"])]},
            "chat": {"n": len(crows), "reply_identical": sum(r["reply_identical"] for r in crows),
                     "prompt_tokens_identical": sum(r["prompt_eval_count"] == r["ember_prompt_tokens"] for r in crows),
                     "eval_count_identical": sum(r["eval_count"] == r["ember_tokens"] for r in crows),
                     "eval_count_identical_less_eos": sum(
                         (r["eval_count"] or 0) - (r["done_reason"] == "stop") == r["ember_tokens"] for r in crows),
                     "done_reasons": sorted(set(str(r["done_reason"]) for r in crows)),
                     "differ": [r["i"] for r in crows if not r["reply_identical"]]}})
    finally:
        subprocess.run([a.ollama, "rm", NAME], env=env, capture_output=True)
        srv.send_signal(signal.SIGINT)
        try:
            srv.wait(30)
        except subprocess.TimeoutExpired:
            srv.kill()
        logf.close()
        shutil.rmtree(os.path.join(a.work, "models"), ignore_errors=True)
    print("done", a.out)


if __name__ == "__main__":
    main()
