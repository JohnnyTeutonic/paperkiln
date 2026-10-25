#!/usr/bin/env python3
"""Stage a model of the benchmarks campaign on the VM (tools/colab_chat/bench_vm.sh).

    python3 bench_stage.py TAG [--gguf]      prints the staged spec's path

TAG's spec is experiments/benchmarks_2026-10-26/specs/TAG.spec.json; its weights are the upload
/content/chat/upload/TAG.safetensors, linked (never moved) as /content/eval/TAG/<name>.safetensors, and
its data.vocab an upload of the same file name. --gguf also writes the F32 GGUF beside it with `mtstudio
export` (the GGUF ember serves), unless one is already there. Appends {tag, spec, safetensors and GGUF
sha256} to /content/eval/staged.jsonl, so the receipts can be held against the home files' sha256s."""
import hashlib
import json
import os
import subprocess
import sys

U = "/content/chat/upload"
REPO = "/content/pk"
MT = f"{REPO}/build_cuda/mtstudio"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    tag = sys.argv[1]
    want_gguf = "--gguf" in sys.argv[2:]
    s = json.load(open(f"{REPO}/experiments/benchmarks_2026-10-26/specs/{tag}.spec.json"))
    up = f"{U}/{os.path.basename(s['data']['vocab'])}"
    if not os.path.exists(up):
        sys.exit(f"{tag}: vocabulary {up} not uploaded")
    s["data"]["vocab"] = up
    d = s["out_dir"] = f"/content/eval/{tag}"
    os.makedirs(d, exist_ok=True)
    st, link = f"{U}/{tag}.safetensors", f"{d}/{s['name']}.safetensors"
    if not os.path.exists(st):
        sys.exit(f"{tag}: {st} not uploaded")
    if not os.path.islink(link):
        if os.path.exists(link):
            os.remove(link)
        os.symlink(st, link)
    path = f"{d}/spec.json"
    with open(path, "w") as f:
        json.dump(s, f, indent=1)
    rec = {"tag": tag, "spec": path, "name": s["name"], "safetensors_sha256": sha256(st)}
    g = f"{d}/{s['name']}.gguf"
    if want_gguf:
        if not os.path.exists(g):
            with open(f"{d}/export.log", "a") as log:
                p = subprocess.run([MT, "export", path, "--from", st, "--formats", "gguf"], stdout=log,
                                   stderr=subprocess.STDOUT)
            if p.returncode != 0 or not os.path.exists(g):
                sys.exit(f"{tag}: mtstudio export --formats gguf failed (rc {p.returncode}; {d}/export.log)")
        rec["gguf_sha256"] = sha256(g)
    with open("/content/eval/staged.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(path)


if __name__ == "__main__":
    main()
