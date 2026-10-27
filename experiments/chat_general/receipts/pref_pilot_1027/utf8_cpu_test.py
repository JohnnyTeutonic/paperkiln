#!/usr/bin/env python3
"""Preference pilot: sample_candidates.py's U+FFFD rule against a CPU ember on pk_sft_v1.

    python3 utf8_cpu_test.py SERVER_BIN GGUF REPO OUT_JSON

1. Greedy (top_k 1) replies to prompts that draw non-ASCII text, at every max_new_tokens in 1..MAXN,
   until a reply cut by the budget ends inside a character (reply ends in U+FFFD, stop "length").
   Each reply is checked against the vocabulary's decode of the returned ids (Python's
   errors="replace" semantics), and no request may answer HTTP 500.
2. sample_candidates.sample() on that prompt at that budget, through the real server: near-greedy
   (temperature 0.01, all four runs the same cut reply) gives fewer_than_two_distinct with four
   "trimmed" runs; the pilot's temperatures give records whose candidates hold no U+FFFD.
3. sample_candidates.py main on the first 3 eligible test-split prompts at the pilot's settings
   (128 new tokens, ctx 512), output checked for format and for U+FFFD.
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

BIN, GGUF, REPO, OUT = sys.argv[1:5]
PORT = 8137
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "tools", "posttrain"))
sys.path.insert(0, os.path.join(REPO, "corpus"))
import bpe  # noqa: E402
import sample_candidates as S  # noqa: E402
from chat_sample_ember import post  # noqa: E402

URL = f"http://127.0.0.1:{PORT}"
PROMPTS = ["Write the Japanese word for thank you in Japanese script.",
           "Translate 'good morning, my friend' into Russian.",
           "What is the Greek letter for pi? Write the symbol.",
           "Write a short sentence with three emoji in it.",
           "Translate 'the weather is nice today' into Chinese characters.",
           "Write the German word for street, with its special letter."]
MAXN = 48


def health():
    try:
        return json.load(urllib.request.urlopen(URL + "/health", timeout=2))
    except Exception:
        return None


def main():
    tok = bpe.load_vocab(GGUF)
    srv = subprocess.Popen([BIN, GGUF, str(PORT), "127.0.0.1", "-", "-1"], stdout=open(OUT + ".ember.log", "w"),
                           stderr=subprocess.STDOUT)
    rep = {"bin": BIN, "gguf": GGUF}
    try:
        for _ in range(300):
            if health() is not None or srv.poll() is not None:
                break
            time.sleep(1)
        rep["health"] = health()
        requests = http500 = decode_mismatch = 0
        found = None
        fffd_seen = []
        for user in PROMPTS:
            for n in range(1, MAXN + 1):
                body = {"user_input": user, "history": [], "temperature": 1.0, "top_k": 1, "top_p": 1.0, "seed": 1,
                        "max_new_tokens": n, "return_ids": True}
                try:
                    r = post(URL, body)
                except urllib.error.HTTPError as e:
                    http500 += e.code == 500
                    continue
                requests += 1
                if tok.decode(r["ids"]) != r["reply"]:
                    decode_mismatch += 1
                if "�" in r["reply"]:
                    fffd_seen.append({"user": user, "max_new": n, "stop": r["stop_reason"], "reply": r["reply"][-12:],
                                      "status": S.utf8_status(r["reply"], r["stop_reason"])[1]})
                if found is None and r["stop_reason"] == "length" and S.utf8_status(r["reply"], "length")[1] == "trimmed":
                    found = {"user": user, "max_new": n, "reply": r["reply"], "ids_tail": r["ids"][-4:],
                             "trimmed": S.utf8_status(r["reply"], "length")}
                if r["stop_reason"] != "length":
                    break
            if found:
                break
        rep.update(greedy_requests=requests, http500=http500, decode_mismatch=decode_mismatch, found=found,
                   fffd_seen=fffd_seen[:20])
        if found:
            ids = tok.encode(f"<|user|>{found['user']}<|assistant|>")
            ctx = 512
            near = list(S.sample([("t#0", found["user"])], lambda b: post(URL, b), 4, [0.01], 5, found["max_new"], 1.0,
                                 tok.encode, ctx))
            rep["near_greedy"] = near
            pilot = list(S.sample([("t#0", found["user"])], lambda b: post(URL, b), 4, [0.8, 0.9, 1.0, 0.9], 31,
                                  found["max_new"], 1.0, tok.encode, ctx))
            rep["pilot_temps"] = pilot
            sweep = []
            for n2 in range(found["max_new"], found["max_new"] + 12):
                sweep += list(S.sample([("t#0", found["user"])], lambda b: post(URL, b), 4, [0.8, 0.9, 1.0, 0.9],
                                       31 + n2, n2, 1.0, tok.encode, ctx))
            rep["sweep"] = sweep
            rep["prompt_tokens"] = len(ids)
        # 3. the CLI end to end on real prompts
        if len(sys.argv) > 5 and sys.argv[5] == "nocli":
            raise StopIteration
        out = OUT + ".candidates.jsonl"
        for p in (out, out + ".skipped.jsonl"):
            if os.path.exists(p):
                os.remove(p)
        p = subprocess.run([sys.executable, "tools/posttrain/sample_candidates.py", "--url", URL, "--vocab", GGUF,
                            "--out", out, "--prompts", "3", "--max-new", "128", "--ctx", "512",
                            "--ember-commit", "7526469695ee3f07f2b5d6e60568cd3e09529c84", "--model-sha256", "test"],
                           cwd=REPO, capture_output=True, text=True)
        rep["cli_rc"] = p.returncode
        rep["cli_counts"] = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else p.stderr[-2000:]
        rows = [json.loads(x) for x in open(out, encoding="utf-8")] if os.path.exists(out) else []
        rep["cli_rows"] = len(rows)
        rep["cli_fffd_in_candidates"] = sum("�" in c for r in rows for c in r["candidates"])
        rep["cli_runs_utf8"] = [x["utf8"] for r in rows for x in r["sampler"]["runs"]]
        rep["cli_stops"] = [x["stop"] for r in rows for x in r["sampler"]["runs"]]
        rep["cli_config"] = rows[0]["sampler"]["config"] if rows else None
    except StopIteration:
        pass
    finally:
        srv.terminate()
        srv.wait(timeout=30)
    json.dump(rep, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: rep.get(k) for k in ("greedy_requests", "http500", "decode_mismatch", "cli_rc", "cli_rows",
                                              "cli_fffd_in_candidates")}))


if __name__ == "__main__":
    main()
