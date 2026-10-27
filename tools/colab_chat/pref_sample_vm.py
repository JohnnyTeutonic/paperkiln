#!/usr/bin/env python3
"""Candidates for the Claude-judged preference stage on the VM (experiments/chat_general/PLAN.md,
preference step; LOCAL_LLM_STACK.md section 11, item 2.2): ember.cpp's CUDA server (EMBER_COMMIT,
default tiny_chat_run.EMBER_COMMIT; EMBER_GEMM_TF32=0 and EMBER_DETERMINISTIC=1, so a recorded seed
draws the same reply again on the same GPU and build) on the model's f32 GGUF, and
tools/posttrain/sample_candidates.py drawing k = 4 replies per smol-smoltalk test-split prompt.

    python3 tools/colab_chat/pref_sample_vm.py OUT TAG N_PROMPTS [MAX_NEW=64] [EMBER_COMMIT]

TAG is a chat_eval_stage.py tag: NAME (an uploaded NAME.gguf / NAME.safetensors with its eval spec)
or NAME@JOB (a model this session trained). The context passed to the sampler is the model's T.
Writes OUT/candidates.jsonl (+ .skipped.jsonl; resumable: a rerun continues it) and
OUT/sample_summary.json (the sampler's counts, the GGUF's sha256, ember's build record)."""
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request

REPO = "/content/pk"
PORT = 8099


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def health():
    try:
        return json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2))
    except Exception:
        return None


def main():
    out, tag, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    max_new = sys.argv[4] if len(sys.argv) > 4 else "64"
    commit = sys.argv[5] if len(sys.argv) > 5 else ""
    os.makedirs(out, exist_ok=True)
    # The test split (and the train split it is checked against): pinned, sha256-verified by the fetcher.
    subprocess.run([sys.executable, "corpus/fetchers/smol_smoltalk.py", "qa_v1"], cwd=REPO, check=True)
    spec_path = subprocess.run([sys.executable, "tools/colab_chat/chat_eval_stage.py", tag], cwd=REPO, check=True,
                               capture_output=True, text=True).stdout.strip().splitlines()[-1]
    spec = json.load(open(spec_path))
    gguf = os.path.join(spec["out_dir"], spec["name"] + ".gguf")
    subprocess.run([sys.executable, "tools/colab_chat/ensure_ember_vm.py", f"{out}/ember_build.json",
                    f"{out}/ember_build.log", commit], cwd=REPO, check=True)
    eb = json.load(open(f"{out}/ember_build.json"))
    if health() is not None:
        sys.exit(f"port {PORT} already answers /health")
    env = {k: v for k, v in os.environ.items() if not k.startswith("MICROTORCH_")}
    srv = subprocess.Popen([eb["bin"], gguf, str(PORT), "127.0.0.1", "-", "-1"], stdout=open(f"{out}/ember.log", "w"),
                           stderr=subprocess.STDOUT, env={**env, "EMBER_GEMM_TF32": "0", "EMBER_DETERMINISTIC": "1"})
    try:
        h = None
        for _ in range(300):
            h = health()
            if h is not None or srv.poll() is not None:
                break
            time.sleep(1)
        if not h or os.path.realpath(h.get("model", "")) != os.path.realpath(gguf):
            sys.exit(f"ember /health names {h and h.get('model')}, not {gguf}")
        if h.get("deterministic") is not True:
            sys.exit("ember /health does not report deterministic mode")
        digest = sha256(gguf)
        p = subprocess.run([sys.executable, "tools/posttrain/sample_candidates.py", "--url", f"http://127.0.0.1:{PORT}",
                            "--vocab", gguf, "--out", f"{out}/candidates.jsonl", "--prompts", str(n),
                            "--max-new", max_new, "--ctx", str(spec["data"]["T"]), "--ember-commit", eb["commit"],
                            "--model-sha256", digest], cwd=REPO, capture_output=True, text=True)
        open(f"{out}/sample.log", "w").write(p.stdout + p.stderr)
        if p.returncode != 0:
            sys.exit(f"sample_candidates.py failed ({p.returncode}); see sample.log")
        counts = json.loads(p.stdout.strip().splitlines()[-1])
    finally:
        srv.terminate()
        srv.wait(timeout=30)
    json.dump({"tag": tag, "model": spec["name"], "gguf_sha256": digest, "ember": eb, "health": h, "counts": counts},
              open(f"{out}/sample_summary.json", "w"), indent=1)
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
