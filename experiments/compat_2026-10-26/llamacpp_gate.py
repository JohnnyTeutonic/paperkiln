#!/usr/bin/env python3
"""pk_base_v1 and pk_sft_v1 in llama.cpp b6000 (LOCAL_LLM_STACK.md section 11 item 6), all on the CPU in WSL.

    python3 llamacpp_gate.py --llamacpp ~/llamacpp_b6000_build/build/bin --out ~/cp_compat/out [--threads 2]

Legs (each writes OUT/<leg>.json or .jsonl; a finished leg is not rerun):
  load        llama-server loads the F32 and Q4_K_M files of pk_base_v1 and the F32 pk_sft_v1 (/health 200,
              /props), with the tensor and vocabulary counts it logs.
  tokenize    llama.cpp's /tokenize (parse_special on, add_special off) against paperkiln's tokeniser,
              tools/llama3_bpe.py (the gates' reference), and the C++ llama3_tokenizer.hpp (tools/llama3_encode,
              built here), on a text sample: the 200 prompts of tinychat_hf_prompts_v1 (raw and rendered by paperkiln/1), 40 UltraChat prompts, 200 FineWeb-Edu
              paragraphs, 100 OASST lines, edge cases and the rendered chat prompts. /detokenize round trip.
  greedy_f32  /completion on the 40 frozen prompts rendered by paperkiln/1 ("<|user|>P<|assistant|>"), 60 new
              tokens, argmax (temperature 0, top_k 1), the <unk> id banned as paperkiln and ember ban it, f32 KV
              cache; the reply cut at the first stop id (<|endoftext|> eos, <|user|>/<|assistant|> turn) and
              compared with ember's and mtstudio's recorded replies (ember.cpp/experiments/pk_base_v1_2026-10-24/
              pk_base_v1_parity.jsonl): text, token count (stop token excluded), stop reason, prompt tokens.
  greedy_f32_kvf16  the same with llama.cpp's default f16 KV cache (what a user gets without -ctk/-ctv).
  greedy_q4km the Q4_K_M file against ember's Q4_K_M replies (ember.cpp/experiments/kquant_2026-10-25/).
  greedy_q4km_dequant  the Q4_K_M file dequantised to F32 by llama-quantize (ggml's dequantize_row_*, the stored
              values IR_SPEC 4.7 names), run as F32, against the same ember replies: llama.cpp's CPU kernels
              multiply Q4_K/Q6_K blocks by activations quantised to Q8_K, ember by the stored f32 values, so this
              leg separates the stored weights from llama.cpp's activation quantisation.
  chat_sft    llama-server --jinja on pk_sft_v1: /apply-template on 60 conversations (single turn, multi-turn,
              system message) against tools/chat_template.py render(); /v1/chat/completions greedy on the 40
              frozen prompts against ember's and mtstudio's recorded pk_sft_v1 replies
              (C:/ml_artifacts/general_chat/parity/parity_cpu.jsonl).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import signal
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MT = os.path.dirname(os.path.dirname(HERE))
WS = os.path.dirname(MT)
sys.path.insert(0, os.path.join(MT, "tools"))
import chat_template as CT  # noqa: E402
import llama3_bpe  # noqa: E402

BASE = "/mnt/c/ml_artifacts/next_base/base/pk_base_v1.gguf"
BASE_Q4 = "/mnt/c/ml_artifacts/next_base/base/pk_base_v1_q4_k_m.gguf"
SFT = "/mnt/c/ml_artifacts/general_chat/sft/pk_sft_v1.gguf"
PROMPTS = os.path.join(MT, "corpus/eval/tinychat_hf_prompts_v1.jsonl")
ULTRA = os.path.join(MT, "experiments/thirdparty_2026-10-22/gemma3_270m_it_long/prompts_ultrachat_100_139.jsonl")
FINEWEB = os.path.join(MT, "corpus/data/fineweb_edu_sample_v1/pretrain_train.txt")
OASST = os.path.join(MT, "corpus/data/human_oasst_v1/text.txt")
REF_BASE = os.path.join(WS, "ember.cpp/experiments/pk_base_v1_2026-10-24/pk_base_v1_parity.jsonl")
REF_Q4 = os.path.join(WS, "ember.cpp/experiments/kquant_2026-10-25/q4_k_m_replies.jsonl")
REF_SFT = "/mnt/c/ml_artifacts/general_chat/parity/parity_cpu.jsonl"
STOP_IDS = {1: "eos", 2: "turn", 3: "turn"}
MAX_NEW = 60
PORT = 8131


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def post(path, body, timeout=600):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def get(path):
    return json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=30))


class Server:
    def __init__(self, a, model, extra, log):
        self.argv = [os.path.join(a.llamacpp, "llama-server"), "-m", model, "--host", "127.0.0.1", "--port",
                     str(PORT), "-t", str(a.threads), "-tb", str(a.threads), "-c", "512", "-np", "1",
                     "--no-webui", *extra]
        self.log = log

    def __enter__(self):
        self.f = open(self.log, "w")
        self.p = subprocess.Popen(self.argv, stdout=self.f, stderr=subprocess.STDOUT)
        for _ in range(600):
            try:
                if get("/health").get("status") == "ok":
                    return self
            except Exception:
                pass
            if self.p.poll() is not None:
                raise SystemExit(f"llama-server exited {self.p.returncode}; see {self.log}")
            time.sleep(0.5)
        raise SystemExit("llama-server did not become healthy")

    def __exit__(self, *exc):
        self.p.send_signal(signal.SIGINT)
        try:
            self.p.wait(30)
        except subprocess.TimeoutExpired:
            self.p.kill()
        self.f.close()


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def done(out, name):
    return os.path.exists(os.path.join(out, name))


def save(out, name, obj):
    tmp = os.path.join(out, name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        if isinstance(obj, list):
            for r in obj:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        else:
            json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(tmp, os.path.join(out, name))


def log_facts(log):
    keep = []
    for line in open(log, errors="replace"):
        if any(k in line for k in ("model type", "model params", "n_vocab", "file type", "file format",
                                   "loaded meta data", "n_ctx_train", "n_embd ", "n_layer", "print_info: vocab type",
                                   "EOG token", "BOS token", "EOS token", "UNK token", "PAD token", "chat template",
                                   "special tokens cache size", "token to piece cache size", "CPU_REPACK",
                                   "KV self size", "llama_kv_cache_unified: size", "f_norm_rms_eps", "pre tokenizer",
                                   "load_tensors", "srv  init")):
            keep.append(line.rstrip())
    return keep


# ---------------------------------------------------------------- legs
def leg_load(a):
    res = {}
    for name, model in (("pk_base_v1_f32", BASE), ("pk_base_v1_q4_k_m", BASE_Q4), ("pk_sft_v1_f32", SFT)):
        log = os.path.join(a.out, f"server_load_{name}.log")
        with Server(a, model, [], log):
            props = get("/props")
            res[name] = {"model": model, "sha256": sha256(model), "health": "ok",
                         "chat_template_from_file": bool(props.get("chat_template")),
                         "props_n_ctx": props.get("default_generation_settings", {}).get("n_ctx")}
        res[name]["log"] = log_facts(log)
    save(a.out, "load.json", res)


def sample_texts():
    recs = []
    for r in jl(PROMPTS):
        recs.append(("tinychat_hf_prompts_v1", r["prompt"]))
    for r in jl(ULTRA):
        recs.append(("ultrachat40", r["prompt"]))
    paras, buf = [], ""
    with open(FINEWEB, encoding="utf-8") as f:
        for line in f:
            buf += line
            if len(buf) > 1200:
                paras.append(buf)
                buf = ""
            if len(paras) >= 200:
                break
    recs += [("fineweb200", p) for p in paras]
    with open(OASST, encoding="utf-8") as f:
        lines = [l for l in f.read().split("\n") if l.strip()]
    rnd = random.Random(20261026)
    recs += [("oasst100", l) for l in rnd.sample(lines, 100)]
    edge = ["", " ", "  ", "\n", "\n\n\n", "\t", " \t \n ", "a", "Hello", " Hello", "Hello ", "HELLO world",
            "I'M HERE, YOU'RE THERE, WE'LL SEE", "don't won't can't I've you'd they're", "1", "12", "123", "1234",
            "12345678901234567890", "3.14159 2,718 1e-10 0x1F", "$100.00 or 50% off!!!", "a\r\nb\rc\nd",
            "trailing spaces   ", "   leading spaces", "tabs\tand\tmore\t\t", "naïve café résumé façade",
            "Ünïcödé ÄÖÜ ß ẞ", "日本語のテキスト", "中文文本测试", "한국어 텍스트", "Русский текст", "Ελληνικά κείμενα",
            "עברית", "العربية", "emoji 😀🎉👍🏽 family 👨‍👩‍👧", "math ∑∫√∞ ≤ ≥ ≠", "code: def f(x):\n    return x**2\n",
            "<html><body>Hi</body></html>", "URL https://example.com/a?b=c&d=e#f", "email a.b@c.de",
            "...", "!!!???", "--- *** ___", "''' \"\"\" ``` ", "\u00a0nbsp\u00a0", "zero\u200bwidth",
            "combining e\u0301 a\u0308", "<|user|>hi<|assistant|>hello<|endoftext|>", "<unk> literal",
            "text <|endoftext|> more", "<|user|>", "<|assistant|>", "<|endoftext|>", "< |user| >"]
    recs += [("edge", e) for e in edge]
    for r in jl(PROMPTS):
        recs.append(("rendered_paperkiln1", CT.prompt([], r["prompt"])))
    return recs


def build_encoder(a):
    exe = os.path.join(a.out, "llama3_encode")
    if not os.path.exists(exe):
        subprocess.run(["g++", "-O2", "-std=c++17", "-I", os.path.join(MT, "include"),
                        os.path.join(MT, "tools/llama3_encode.cpp"), "-o", exe], check=True)
    return exe


def leg_tokenize(a):
    tok = llama3_bpe.from_gguf(BASE)
    recs = sample_texts()
    exe = build_encoder(a)
    cpp_in = b"\0".join(t.encode("utf-8") for _, t in recs)
    p = subprocess.run([exe, BASE], input=cpp_in, capture_output=True, check=True)
    cpp = [[int(x) for x in line.split()] for line in p.stdout.decode().split("\n")[:len(recs)]]
    rows, by = [], {}
    with Server(a, BASE, [], os.path.join(a.out, "server_tokenize.log")):
        for i, (src, text) in enumerate(recs):
            ll = post("/tokenize", {"content": text, "add_special": False})["tokens"]
            py = tok.encode(text, add_special_tokens=False).ids
            det = post("/detokenize", {"tokens": ll})["content"] if ll else ""
            row = {"i": i, "source": src, "chars": len(text), "tokens": len(py), "llamacpp_eq_py": ll == py,
                   "cpp_eq_py": cpp[i] == py, "detok_roundtrip": det == text}
            if ll != py or cpp[i] != py:
                k = next((j for j in range(min(len(ll), len(py))) if ll[j] != py[j]), min(len(ll), len(py)))
                row.update({"text": text, "first_diff": k, "llamacpp": ll[max(0, k - 3):k + 5],
                            "py": py[max(0, k - 3):k + 5], "cpp": cpp[i][max(0, k - 3):k + 5]})
            if not row["detok_roundtrip"]:
                row["text"], row["detok"] = text, det
            rows.append(row)
            b = by.setdefault(src, {"records": 0, "tokens": 0, "llamacpp_eq_py": 0, "cpp_eq_py": 0,
                                    "detok_roundtrip": 0})
            b["records"] += 1
            b["tokens"] += len(py)
            for k in ("llamacpp_eq_py", "cpp_eq_py", "detok_roundtrip"):
                b[k] += row[k]
    save(a.out, "tokenize.jsonl", rows)
    tot = {k: sum(b[k] for b in by.values()) for k in ("records", "tokens", "llamacpp_eq_py", "cpp_eq_py",
                                                       "detok_roundtrip")}
    save(a.out, "tokenize_summary.json", {"vocab": BASE, "by_source": by, "total": tot,
                                          "reference": "tools/llama3_bpe.py (HF tokenizers configured as llama.cpp "
                                                       "encodes); tools/llama3_encode (llama3_tokenizer.hpp) built here",
                                          "llamacpp": "/tokenize, parse_special on, add_special off"})


def complete(prompt_text):
    r = post("/completion", {"prompt": prompt_text, "n_predict": MAX_NEW, "temperature": 0.0, "top_k": 1,
                             "top_p": 1.0, "min_p": 0.0, "typical_p": 1.0, "repeat_penalty": 1.0,
                             "presence_penalty": 0.0, "frequency_penalty": 0.0, "dry_multiplier": 0.0,
                             "xtc_probability": 0.0, "logit_bias": [[0, False]], "seed": 0, "cache_prompt": False,
                             "return_tokens": True, "n_probs": 0, "samplers": ["top_k", "temperature"]})
    ids = r.get("tokens") or []
    cut, stop = len(ids), "length"
    for j, t in enumerate(ids):
        if t in STOP_IDS:
            cut, stop = j, STOP_IDS[t]
            break
    return ids[:cut], stop, r.get("tokens_evaluated"), r


def greedy_leg(a, name, model, ref_path, extra, ref_key):
    tok = llama3_bpe.from_gguf(BASE)
    refs = jl(ref_path)
    rows = []
    with Server(a, model, extra, os.path.join(a.out, f"server_{name}.log")):
        for r in jl(PROMPTS)[:40]:
            i = len(rows)
            ref = refs[i]
            ptxt = CT.prompt([], r["prompt"])
            ids, stop, n_prompt, raw = complete(ptxt)
            reply = tok.decode(ids, skip_special_tokens=False)
            rr = ref[ref_key]
            rt = ref.get("ember_tokens", ref.get("tokens"))
            rs = ref.get("ember_stop", ref.get("stop"))
            rp = ref.get("ember_prompt_tokens", ref.get("prompt_tokens"))
            row = {"i": i, "prompt": r["prompt"], "llamacpp": reply, "ref": rr, "llamacpp_tokens": len(ids),
                   "ref_tokens": rt, "llamacpp_stop": stop, "ref_stop": rs, "llamacpp_prompt_tokens": n_prompt,
                   "ref_prompt_tokens": rp, "reply_identical": reply == rr, "llamacpp_content": raw.get("content")}
            if reply != rr:
                k = next((j for j in range(min(len(reply), len(rr))) if reply[j] != rr[j]), min(len(reply), len(rr)))
                row["first_char_diff"] = k
            if "mtstudio" in ref:
                row["ref_mtstudio_equal"] = ref["mtstudio"] == rr
            rows.append(row)
    save(a.out, f"{name}.jsonl", rows)
    s = {"model": model, "sha256": sha256(model), "ref": ref_path, "ref_key": ref_key, "server_extra": extra,
         "n": len(rows), "reply_identical": sum(r["reply_identical"] for r in rows),
         "token_count_identical": sum(r["llamacpp_tokens"] == r["ref_tokens"] for r in rows),
         "stop_identical": sum(r["llamacpp_stop"] == r["ref_stop"] for r in rows),
         "prompt_tokens_identical": sum(r["llamacpp_prompt_tokens"] == r["ref_prompt_tokens"] for r in rows),
         "differ": [r["i"] for r in rows if not r["reply_identical"]]}
    save(a.out, f"{name}_summary.json", s)


def conversations():
    rnd = random.Random(7)
    ps = [r["prompt"] for r in jl(PROMPTS)]
    refs = [r["reference"] for r in jl(PROMPTS)]
    convs = []
    for k in range(20):
        convs.append([{"role": "user", "content": ps[k]}])
    for k in range(20):
        n = 1 + k % 4
        m = []
        for j in range(n):
            m += [{"role": "user", "content": ps[(k + j) % 40]}, {"role": "assistant", "content": refs[(k + j) % 40]}]
        m.append({"role": "user", "content": ps[(k + n) % 40]})
        convs.append(m)
    for k in range(20):
        sysm = rnd.choice(["You are a helpful assistant.", "Answer in one sentence.", "Be kind.\nBe brief.",
                           "Speak like a pirate."])
        m = [{"role": "system", "content": sysm}]
        for j in range(k % 3):
            m += [{"role": "user", "content": ps[(k + j) % 40]}, {"role": "assistant", "content": refs[(k + j) % 40]}]
        m.append({"role": "user", "content": ps[(k + 7) % 40]})
        convs.append(m)
    return convs


def leg_chat(a):
    tok = llama3_bpe.from_gguf(SFT)
    refs = jl(REF_SFT)
    trows, crows = [], []
    with Server(a, SFT, ["--jinja", "-ctk", "f32", "-ctv", "f32"], os.path.join(a.out, "server_chat_sft.log")):
        props = get("/props")
        for i, m in enumerate(conversations()):
            mine = CT.render(m, generation_prompt=True)
            got = post("/apply-template", {"messages": m})["prompt"]
            ids_ll = post("/tokenize", {"content": got, "add_special": False})["tokens"]
            ids_pk = tok.encode(mine, add_special_tokens=False).ids
            trows.append({"i": i, "turns": sum(x["role"] == "user" for x in m),
                          "system": m[0]["role"] == "system", "identical": got == mine, "ids_identical": ids_ll == ids_pk,
                          **({} if got == mine else {"llamacpp": got, "paperkiln": mine})})
        for i, r in enumerate(jl(PROMPTS)[:40]):
            ref = refs[i]
            body = {"messages": [{"role": "user", "content": r["prompt"]}], "max_tokens": MAX_NEW, "temperature": 0.0,
                    "top_k": 1, "top_p": 1.0, "min_p": 0.0, "repeat_penalty": 1.0, "seed": 0, "cache_prompt": False,
                    "logit_bias": [[0, False]], "samplers": ["top_k", "temperature"]}
            out = post("/v1/chat/completions", body)
            ch = out["choices"][0]
            reply = ch["message"]["content"]
            crows.append({"i": i, "prompt": r["prompt"], "llamacpp": reply, "ember": ref["ember"],
                          "mtstudio": ref["mtstudio"], "reply_identical": reply == ref["ember"],
                          "finish_reason": ch.get("finish_reason"), "ember_stop": ref["ember_stop"],
                          "completion_tokens": out.get("usage", {}).get("completion_tokens"),
                          "ember_tokens": ref["ember_tokens"],
                          "prompt_tokens": out.get("usage", {}).get("prompt_tokens"),
                          "ember_prompt_tokens": ref["ember_prompt_tokens"]})
    save(a.out, "chat_template.jsonl", trows)
    save(a.out, "chat_sft.jsonl", crows)
    save(a.out, "chat_sft_summary.json", {
        "model": SFT, "sha256": sha256(SFT), "server": "llama-server --jinja -ctk f32 -ctv f32",
        "chat_template_in_props": (props.get("chat_template") or "")[:120],
        "apply_template": {"n": len(trows), "identical": sum(r["identical"] for r in trows),
                           "ids_identical": sum(r["ids_identical"] for r in trows)},
        "chat": {"n": len(crows), "reply_identical": sum(r["reply_identical"] for r in crows),
                 "prompt_tokens_identical": sum(r["prompt_tokens"] == r["ember_prompt_tokens"] for r in crows),
                 # llama.cpp's usage counts the end-of-text token it stopped on; ember's `tokens` does not
                 "completion_tokens_identical_less_eos": sum(
                     r["completion_tokens"] - (r["finish_reason"] == "stop") == r["ember_tokens"] for r in crows),
                 "finish_stop_vs_eos": sum((r["finish_reason"] == "stop") == (r["ember_stop"] in ("eos", "turn"))
                                           for r in crows),
                 "differ": [r["i"] for r in crows if not r["reply_identical"]]}})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llamacpp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--legs", default="load,tokenize,greedy_f32,greedy_f32_kvf16,greedy_q4km,greedy_q4km_dequant,chat_sft")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    legs = a.legs.split(",")
    f32kv = ["-ctk", "f32", "-ctv", "f32"]
    if "load" in legs and not done(a.out, "load.json"):
        leg_load(a)
    if "tokenize" in legs and not done(a.out, "tokenize_summary.json"):
        leg_tokenize(a)
    if "greedy_f32" in legs and not done(a.out, "greedy_f32_summary.json"):
        greedy_leg(a, "greedy_f32", BASE, REF_BASE, f32kv, "ember")
    if "greedy_f32_kvf16" in legs and not done(a.out, "greedy_f32_kvf16_summary.json"):
        greedy_leg(a, "greedy_f32_kvf16", BASE, REF_BASE, [], "ember")
    if "greedy_q4km" in legs and not done(a.out, "greedy_q4km_summary.json"):
        greedy_leg(a, "greedy_q4km", BASE_Q4, REF_Q4, f32kv, "reply")
    if "greedy_q4km_dequant" in legs and not done(a.out, "greedy_q4km_dequant_summary.json"):
        deq = os.path.join(a.out, "pk_base_v1_q4_k_m_dequant_f32.gguf")
        if not os.path.exists(deq):
            with open(os.path.join(a.out, "dequant.log"), "w") as lf:
                subprocess.run([os.path.join(a.llamacpp, "llama-quantize"), "--allow-requantize", BASE_Q4, deq + ".tmp",
                                "F32", str(a.threads)], check=True, stdout=lf, stderr=subprocess.STDOUT)
            os.replace(deq + ".tmp", deq)
        greedy_leg(a, "greedy_q4km_dequant", deq, REF_Q4, f32kv, "reply")
        os.remove(deq)  # scratch: its sha256 is in the summary
    if "chat_sft" in legs and not done(a.out, "chat_sft_summary.json"):
        leg_chat(a)
    print("done", a.out)


if __name__ == "__main__":
    main()
