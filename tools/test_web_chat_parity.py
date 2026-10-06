#!/usr/bin/env python3
"""Parity: the browser engine (web/chat/paperkiln.js) against mtstudio.

    python3 tools/test_web_chat_parity.py --run RUN_DIR [--run RUN_DIR2 ...]
        [--mtstudio build/mtstudio] [--tokens 40] [--no-chat]

Run from the directory mtstudio trains from (relative spec paths such as
data.vocab resolve against it). For every run directory:

1. export it with tools/export_web_chat.py into a temporary folder;
2. greedy continuations, token for token: `mtstudio sample <spec> --prompt P
   --temp 0 --topk 1` versus paperkiln.js under Node for the same prompts,
   including one prompt longer than the context window (the sliding window
   path) and one with out-of-vocabulary words (id 0);
3. chat replies (unless --no-chat): `mtstudio chat` on a local port, POST
   /chat at temperature 0 with a fixed seed, versus Chat.reply() at
   temperature 0, for single and multi-turn histories (reply text and stop
   reason);
4. RoPE head coverage (llama runs with more than one head): the manifest's
   rope_heads ("all" or "first", from the run's record) is what makes the
   match; the same export with the other value must NOT reproduce
   mtstudio's greedy continuations, so the check cannot pass by accident.
   Pass one run of each kind to cover both (a run trained before
   arch.rope_heads existed is "first"; a new run is "all").

Needs `node` on PATH (WSL: source ~/.nvm/nvm.sh). Exit status 0 only when
every comparison matches. Standard library only.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
JS = HERE.parent / "web" / "chat" / "paperkiln.js"
sys.path.insert(0, str(HERE))
import export_web_chat  # noqa: E402

PROMPTS = [
    "user: hello ! assistant:",
    "user: how are you ? assistant:",
    "user: what do you like to eat ? assistant: i like",
    "user: good morning ! assistant: hi ! nice to see you . user: where did you go last night ? assistant:",
    "user: Tell me about the XYLOPHONE zqx, please! assistant:",  # case + unknown words
    "hello",
]
CHATS = [
    {"history": [], "user_input": "hello !"},
    {"history": [], "user_input": "how is the weather today ?"},
    {"history": [{"role": "user", "content": "hi !"},
                 {"role": "assistant", "content": "hi ! nice to see you ."}],
     "user_input": "what do you like to eat ?"},
]

DRIVER = r"""
const fs = require("fs");
const P = require(process.argv[2]);
const dir = process.argv[3];
const job = JSON.parse(fs.readFileSync(process.argv[4], "utf8"));
const manifest = JSON.parse(fs.readFileSync(dir + "/model/manifest.json", "utf8"));
const weights = fs.readFileSync(dir + "/model/" + manifest.weights.file);
const vocab = JSON.parse(fs.readFileSync(dir + "/model/" + manifest.vocab.file, "utf8"));
const e = P.fromParts(manifest, weights, vocab);
(async () => {
  const samples = job.prompts.map(p => {
    const ids = e.vocab.tokenize(p, 100000);
    return e.chat.continueIds(ids, job.tokens, { temperature: 0, topK: 1 }).map(i => e.vocab.tokens[i]);
  });
  const chats = [];
  for (const c of job.chats)
    chats.push(await e.chat.reply(c.history, c.user_input, { maxNewTokens: job.chat_tokens, temperature: 0 }));
  process.stdout.write(JSON.stringify({ samples, chats, V: e.model.V, params: e.model.params }));
})().catch(err => { console.error(err.stack || String(err)); process.exit(1); });
"""


def long_prompt(T: int) -> str:
    """A dialogue longer than T word tokens, so generation slides the window."""
    turn = "user: how are you ? assistant: i am fine , thank you . "
    p = ""
    while len(p.split()) < T + 12:
        p += turn
    return p + "user: hello ! assistant:"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def mtstudio_sample(mtstudio: str, spec: Path, prompt: str, tokens: int) -> list[str]:
    r = subprocess.run([mtstudio, "sample", str(spec), "--prompt", prompt, "--tokens", str(tokens),
                        "--temp", "0", "--topk", "1"], capture_output=True, text=True, check=True)
    lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
    line = lines[-1]
    if not line.startswith(prompt):
        raise RuntimeError(f"unexpected mtstudio sample output: {line!r}")
    return line[len(prompt):].split()


def mtstudio_chat(mtstudio: str, run: Path, chats: list, chat_tokens: int) -> list[dict]:
    port = free_port()
    log = tempfile.TemporaryFile()
    proc = subprocess.Popen([mtstudio, "chat", str(run), "--port", str(port)],
                            stdout=log, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(600):
            if proc.poll() is not None:
                log.seek(0)
                raise RuntimeError("mtstudio chat exited: " + log.read().decode(errors="replace"))
            try:
                urllib.request.urlopen(base + "/health", timeout=2).read()
                break
            except OSError:
                time.sleep(0.2)
        out = []
        for c in chats:
            body = json.dumps({"user_input": c["user_input"], "history": c["history"],
                               "temperature": 0, "max_new_tokens": chat_tokens,
                               "seed": 1}).encode()
            req = urllib.request.Request(base + "/chat", data=body,
                                         headers={"Content-Type": "application/json"})
            out.append(json.loads(urllib.request.urlopen(req, timeout=600).read()))
        return out
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def run_js(site: Path, job_path: Path, tmp: str) -> dict | None:
    drv = Path(tmp) / "driver.js"
    drv.write_text(DRIVER, encoding="utf-8")
    r = subprocess.run(["node", str(drv), str(JS), str(site), str(job_path)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr)
        return None
    return json.loads(r.stdout)


def check_run(run: Path, mtstudio: str, tokens: int, chat: bool, vocab: str | None,
              modes: set) -> bool:
    run = run.resolve()
    ok = True
    with tempfile.TemporaryDirectory(prefix="webchat-parity-") as tmp:
        site = Path(tmp) / "site"
        m = export_web_chat.export(run, site, vocab)
        rope = f", heads={m['heads']}, rope_heads={m['rope_heads']}" if m["rope_heads"] else ""
        print(f"== {m['name']}: family {m['family']}, d={m['d']}, layers={m['layers']}, "
              f"T={m['T']}, vocab {m['vocab_size']}, weights {m['weights']['bytes']:,} bytes"
              + rope)
        # A spec copy whose out_dir is this run, wherever it now lives.
        spec = json.loads((run / "spec.json").read_text(encoding="utf-8"))
        spec["out_dir"] = str(run)
        spec_path = Path(tmp) / "spec.json"
        spec_path.write_text(json.dumps(spec), encoding="utf-8")

        prompts = PROMPTS + [long_prompt(m["T"])]
        job = {"prompts": prompts, "tokens": tokens, "chats": CHATS if chat else [],
               "chat_tokens": 60}
        job_path = Path(tmp) / "job.json"
        job_path.write_text(json.dumps(job), encoding="utf-8")
        t0 = time.time()
        js = run_js(site, job_path, tmp)
        if js is None:
            return False
        print(f"   paperkiln.js: {js['params']:,} parameters, {time.time() - t0:.1f} s under Node")

        wants = [mtstudio_sample(mtstudio, spec_path, p, tokens) for p in prompts]
        for p, got, want in zip(prompts, js["samples"], wants):
            same = got == want
            ok &= same
            shown = p if len(p) < 60 else p[:28] + " ... " + p[-24:]
            n = len(want)
            first = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b),
                         None if len(got) == n else min(len(got), n))
            print(f"   sample {'MATCH' if same else 'DIFF '} {n:3d} tokens  {shown!r}"
                  + ("" if same else f"  first difference at token {first}"))
            if not same:
                print(f"      mtstudio: {' '.join(want)}\n      js:       {' '.join(got)}")

        if chat:
            server = mtstudio_chat(mtstudio, run, CHATS, 60)
            for c, s, j in zip(CHATS, server, js["chats"]):
                same = s["reply"] == j["reply"] and s["stop_reason"] == j["stop_reason"]
                ok &= same
                turns = len(c["history"]) // 2 + 1
                print(f"   chat   {'MATCH' if same else 'DIFF '} {turns} turn(s) {c['user_input']!r}"
                      f" -> {s['reply']!r} [{s['stop_reason']}]")
                if not same:
                    print(f"      js: {j['reply']!r} [{j['stop_reason']}]")

        if m["rope_heads"] and m["heads"] > 1:
            # Counterfactual: the same weights under the other coverage.
            other = "first" if m["rope_heads"] == "all" else "all"
            alt = Path(tmp) / "site_alt"
            shutil.copytree(site, alt)
            man = json.loads((alt / "model" / "manifest.json").read_text(encoding="utf-8"))
            man["rope_heads"] = other
            (alt / "model" / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
            alt_job = Path(tmp) / "job_alt.json"
            alt_job.write_text(json.dumps(dict(job, chats=[])), encoding="utf-8")
            js_alt = run_js(alt, alt_job, tmp)
            if js_alt is None:
                return False
            n_diff = sum(g != w for g, w in zip(js_alt["samples"], wants))
            discriminates = n_diff > 0
            ok &= discriminates
            print(f"   rope   {'OK   ' if discriminates else 'FAIL '} manifest rope_heads="
                  f"{m['rope_heads']} matches; rope_heads={other} differs on {n_diff}/"
                  f"{len(wants)} prompts")
            modes.add(m["rope_heads"])
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="paperkiln.js versus mtstudio, token for token")
    ap.add_argument("--run", action="append", type=Path, required=True,
                    help="mtstudio run directory (repeatable)")
    ap.add_argument("--mtstudio", default="build/mtstudio")
    ap.add_argument("--tokens", type=int, default=40)
    ap.add_argument("--vocab", help="vocab GGUF override for export_web_chat")
    ap.add_argument("--no-chat", action="store_true", help="skip the mtstudio chat comparison")
    args = ap.parse_args(argv)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    modes: set = set()
    results = [check_run(r, args.mtstudio, args.tokens, not args.no_chat, args.vocab, modes)
               for r in args.run]
    if modes:
        print("rope_heads covered: " + ", ".join(sorted(modes)))
    print("PARITY OK" if all(results) else "PARITY FAILED")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
