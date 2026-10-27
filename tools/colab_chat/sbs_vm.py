#!/usr/bin/env python3
"""The judged side-by-side's sampling on the VM (tools/posttrain/side_by_side.py; experiments/
chat_general/PLAN.md, the preference step's gate): two ember.cpp CUDA servers (EMBER_COMMIT,
EMBER_GEMM_TF32=0, EMBER_DETERMINISTIC=1), one per model, and side_by_side.py sample over the
smol-smoltalk val split. The judging runs at home.

    python3 tools/colab_chat/sbs_vm.py OUT TAG_A TAG_B N_PROMPTS MAX_NEW EMBER_COMMIT

TAGs are chat_eval_stage.py tags (NAME, or NAME@JOB for a model this session trained); both models
must share a vocabulary (the prompt-id check runs against A's GGUF). Writes OUT/sbs.jsonl,
OUT/sbs_summary.json (counts, both GGUFs' sha256, ember's build record) and OUT/ember_{a,b}.log."""
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request

REPO = "/content/pk"
PORTS = (8101, 8102)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def health(port):
    try:
        return json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2))
    except Exception:
        return None


def stage(tag):
    path = subprocess.run([sys.executable, "tools/colab_chat/chat_eval_stage.py", tag], cwd=REPO, check=True,
                          capture_output=True, text=True).stdout.strip().splitlines()[-1]
    spec = json.load(open(path))
    return spec, os.path.join(spec["out_dir"], spec["name"] + ".gguf")


def main():
    out, ta, tb, n, max_new, commit = sys.argv[1:7]
    os.makedirs(out, exist_ok=True)
    subprocess.run([sys.executable, "corpus/fetchers/smol_smoltalk.py", "qa_v1"], cwd=REPO, check=True)
    (sa, ga), (sb, gb) = stage(ta), stage(tb)
    if sa["data"]["T"] != sb["data"]["T"]:
        sys.exit(f"the two models' contexts differ ({sa['data']['T']} vs {sb['data']['T']})")
    subprocess.run([sys.executable, "tools/colab_chat/ensure_ember_vm.py", f"{out}/ember_build.json",
                    f"{out}/ember_build.log", commit], cwd=REPO, check=True)
    eb = json.load(open(f"{out}/ember_build.json"))
    env = {k: v for k, v in os.environ.items() if not k.startswith("MICROTORCH_")}
    env.update(EMBER_GEMM_TF32="0", EMBER_DETERMINISTIC="1")
    srvs, hs = [], []
    try:
        for port, gguf, side in zip(PORTS, (ga, gb), "ab"):
            if health(port) is not None:
                sys.exit(f"port {port} already answers /health")
            srvs.append(subprocess.Popen([eb["bin"], gguf, str(port), "127.0.0.1", "-", "-1"],
                                         stdout=open(f"{out}/ember_{side}.log", "w"), stderr=subprocess.STDOUT, env=env))
            h = None
            for _ in range(300):
                h = health(port)
                if h is not None or srvs[-1].poll() is not None:
                    break
                time.sleep(1)
            if not h or os.path.realpath(h.get("model", "")) != os.path.realpath(gguf) or h.get("deterministic") is not True:
                sys.exit(f"ember on {port}: /health {h}")
            hs.append(h)
        p = subprocess.run([sys.executable, "tools/posttrain/side_by_side.py", "sample",
                            "--url-a", f"http://127.0.0.1:{PORTS[0]}", "--url-b", f"http://127.0.0.1:{PORTS[1]}",
                            "--name-a", ta, "--name-b", tb, "--vocab", ga, "--out", f"{out}/sbs.jsonl",
                            "--prompts", n, "--max-new", max_new, "--ctx", str(sa["data"]["T"]),
                            "--ember-commit", eb["commit"]], cwd=REPO, capture_output=True, text=True)
        open(f"{out}/sbs.log", "w").write(p.stdout + p.stderr)
        if p.returncode != 0:
            sys.exit(f"side_by_side.py failed ({p.returncode}); see sbs.log")
        counts = json.loads(p.stdout.strip().splitlines()[-1])
    finally:
        for s in srvs:
            s.terminate()
            s.wait(timeout=30)
    json.dump({"models": [ta, tb], "gguf_sha256": [sha256(ga), sha256(gb)], "ember": eb, "health": hs,
               "counts": counts}, open(f"{out}/sbs_summary.json", "w"), indent=1)
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
