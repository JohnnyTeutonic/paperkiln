#!/usr/bin/env python3
"""Fill the verdict cache that LLMJudgeVerifier reads
(include/microtorch/posttrain/verifier.hpp; charter 0001 §6 and §12.3).

    python tools/posttrain/judge_sidecar.py plan --requests R.jsonl --cache C.jsonl
    python tools/posttrain/judge_sidecar.py run  --requests R.jsonl --cache C.jsonl \\
        --judge-id MODEL --rubric tools/posttrain/rubrics/tinyarith_v1.txt \\
        --env-file PATH --max-calls 600

Requests are {"prompt", "completion"} lines. Each verdict is appended to the
cache as {"prompt", "completion", "judge_id", "rubric_version", "rubric_sha256",
"reward", "rationale", "response_sha256", "created"}. The rubric version is the
rubric file's name plus the first 12 hex digits of its sha256, so editing a
rubric can never silently mix verdicts.

Rules this script enforces:
- No credentials are searched for. `run` reads exactly one key
  (OPENAI_API_KEY) from the file named by --env-file and never prints it.
- --max-calls is required and is a hard cap on API requests in one run.
- Pairs already in the cache for this judge and rubric are skipped
  (idempotent, append-only).
- A response that is not JSON, has no verdict in {0, 1}, or has an empty
  rationale is NOT written to the cache. It goes to <cache>.failures.jsonl,
  so the C++ verifier sees "no verdict on record" rather than an invented
  reward.
- The judge never sees a reference answer: that is what makes it a judge
  rather than an exact-match check.

Standard library only (urllib).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys
import urllib.request

API_URL = "https://api.openai.com/v1/chat/completions"
KEY_NAME = "OPENAI_API_KEY"


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def rubric_identity(path):
    with open(path, "rb") as f:
        body = f.read()
    h = sha256_bytes(body)
    name = os.path.splitext(os.path.basename(path))[0]
    return body.decode("utf-8"), f"{name}@{h[:12]}", h


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_key(env_file):
    """KEY_NAME from a dotenv-style file; no other line is interpreted."""
    with open(env_file, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s.startswith("export "):
                s = s[len("export "):].lstrip()
            if s.startswith(KEY_NAME + "="):
                v = s.split("=", 1)[1].strip()
                if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
                    v = v[1:-1]
                if v:
                    return v
    raise SystemExit(f"{KEY_NAME} not found in the designated env file")


def openai_transport(key, model):
    def call(messages):
        body = json.dumps({"model": model, "messages": messages, "temperature": 0,
                           "response_format": {"type": "json_object"}}).encode()
        req = urllib.request.Request(API_URL, data=body, method="POST", headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
        content = json.loads(raw)["choices"][0]["message"]["content"]
        return content, raw
    return call


def parse_verdict(content):
    """(reward, rationale) or raise ValueError: nothing is ever defaulted."""
    j = json.loads(content)
    v = j.get("verdict")
    if v not in (0, 1) or isinstance(v, bool):
        raise ValueError(f"verdict must be 0 or 1, got {v!r}")
    r = j.get("rationale")
    if not isinstance(r, str) or not r.strip():
        raise ValueError("empty rationale")
    return float(v), r.strip()


def pending(requests, cache, judge_id, rubric_version):
    done = {(c["prompt"], c["completion"]) for c in cache
            if c.get("judge_id") == judge_id and c.get("rubric_version") == rubric_version}
    seen, out = set(), []
    for r in requests:
        k = (r["prompt"], r["completion"])
        if k not in done and k not in seen:
            seen.add(k)
            out.append(r)
    return out


def run(requests, cache_path, judge_id, rubric_path, max_calls, transport, now=None):
    if max_calls is None or max_calls < 0:
        raise SystemExit("--max-calls is required")
    rubric, rubric_version, rubric_sha = rubric_identity(rubric_path)
    todo = pending(requests, read_jsonl(cache_path), judge_id, rubric_version)
    calls = ok = failed = 0
    stamp = now or (lambda: _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"))
    with open(cache_path, "a", encoding="utf-8", newline="\n") as cache, \
            open(cache_path + ".failures.jsonl", "a", encoding="utf-8", newline="\n") as fails:
        for r in todo:
            if calls >= max_calls:
                break
            calls += 1
            messages = [{"role": "system", "content": rubric},
                        {"role": "user", "content": json.dumps(
                            {"problem": r["prompt"], "response": r["completion"]})}]
            try:
                content, raw = transport(messages)
                reward, rationale = parse_verdict(content)
            except Exception as e:  # recorded, never converted into a verdict
                failed += 1
                fails.write(json.dumps({"prompt": r["prompt"], "completion": r["completion"],
                                        "judge_id": judge_id, "rubric_version": rubric_version,
                                        "error": f"{type(e).__name__}: {e}", "created": stamp()}) + "\n")
                continue
            ok += 1
            cache.write(json.dumps({"prompt": r["prompt"], "completion": r["completion"],
                                    "judge_id": judge_id, "rubric_version": rubric_version,
                                    "rubric_sha256": rubric_sha, "reward": reward,
                                    "rationale": rationale,
                                    "response_sha256": sha256_bytes(raw if isinstance(raw, bytes)
                                                                    else str(raw).encode()),
                                    "created": stamp()}, ensure_ascii=False) + "\n")
    return {"pending": len(todo), "calls": calls, "verdicts": ok, "failures": failed,
            "left": len(todo) - calls, "rubric_version": rubric_version}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["plan", "run"])
    ap.add_argument("--requests", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--judge-id")
    ap.add_argument("--rubric", default=os.path.join(os.path.dirname(__file__), "rubrics", "tinyarith_v1.txt"))
    ap.add_argument("--env-file")
    ap.add_argument("--max-calls", type=int)
    a = ap.parse_args(argv)
    reqs = read_jsonl(a.requests)
    _, rubric_version, _ = rubric_identity(a.rubric)
    if a.cmd == "plan":
        todo = pending(reqs, read_jsonl(a.cache), a.judge_id or "", rubric_version)
        print(json.dumps({"requests": len(reqs), "pending": len(todo), "judge_id": a.judge_id,
                          "rubric_version": rubric_version, "would_call": min(len(todo), a.max_calls)
                          if a.max_calls is not None else len(todo)}))
        return 0
    if not a.judge_id or not a.env_file or a.max_calls is None:
        ap.error("run needs --judge-id, --env-file and --max-calls")
    out = run(reqs, a.cache, a.judge_id, a.rubric, a.max_calls,
              openai_transport(read_key(a.env_file), a.judge_id))
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
