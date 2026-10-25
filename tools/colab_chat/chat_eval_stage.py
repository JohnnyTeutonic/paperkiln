#!/usr/bin/env python3
"""Stage the models of a chat_eval_vm.sh run on the VM, and write its manifest.

    python3 chat_eval_stage.py TAG                 prints the eval spec path for TAG
    python3 chat_eval_stage.py --manifest OUT EMBER_JSON TAG...

TAG = NAME: an uploaded model (/content/chat/upload/NAME.gguf, NAME.safetensors), its eval spec
NAME.spec.json uploaded or in experiments/chat_general/eval_specs/. The spec's data.vocab becomes
an upload of the same file name when there is one, else the VM checkout's path, as does
data.corpus; out_dir becomes /content/eval/NAME, where the model files are symlinked under the
spec's name (never moved, so the uploads stay where the runner and later jobs expect them).
TAG = NAME@JOB: a model this session trained, job JOB's own spec /content/chat/JOB_spec.json.
The manifest records the paperkiln commit, ember's build record, the VM's provenance.json and,
per model, the spec and the sha256 of the GGUF and safetensors it served."""
import hashlib
import json
import os
import subprocess
import sys

C = "/content/chat"
U = f"{C}/upload"
REPO = "/content/pk"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stage(tag):
    if "@" in tag:
        name, job = tag.split("@", 1)
        path = f"{C}/{job}_spec.json"
        s = json.load(open(path))
        for ext in ("gguf", "safetensors"):
            f = os.path.join(s["out_dir"], f"{s['name']}.{ext}")
            if not os.path.exists(f):
                sys.exit(f"{tag}: {f} missing")
        return path, s
    d = f"/content/eval/{tag}"
    os.makedirs(d, exist_ok=True)
    src = next((p for p in (f"{U}/{tag}.spec.json", f"{REPO}/experiments/chat_general/eval_specs/{tag}.spec.json")
                if os.path.exists(p)), None)
    if src is None:
        sys.exit(f"{tag}: no eval spec uploaded or in experiments/chat_general/eval_specs/")
    s = json.load(open(src))
    for k in ("vocab", "corpus"):
        if k not in s["data"]:
            continue
        p = s["data"][k]
        up = f"{U}/{os.path.basename(p)}"
        vm = f"{REPO}/" + p.split("/microtorch/", 1)[1] if "/microtorch/" in p else p
        s["data"][k] = up if k == "vocab" and os.path.exists(up) else vm
        if not os.path.exists(s["data"][k]):
            sys.exit(f"{tag}: data.{k} {p} is neither uploaded nor on the VM")
    s["out_dir"] = d
    for ext in ("gguf", "safetensors"):
        f, link = f"{U}/{tag}.{ext}", f"{d}/{s['name']}.{ext}"
        if not os.path.exists(f):
            sys.exit(f"{tag}: {f} missing (upload it as {tag}.{ext})")
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink(f, link)
    path = f"{d}/spec.json"
    json.dump(s, open(path, "w"), indent=1)
    return path, s


def manifest(out, ember_json, tags):
    head = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    m = {"paperkiln_commit": head, "ember": json.load(open(ember_json)), "models": {}}
    prov = f"{C}/provenance.json"
    if os.path.exists(prov):
        m["provenance"] = json.load(open(prov))
    for t in tags:
        path, s = stage(t)
        files = {ext: os.path.realpath(os.path.join(s["out_dir"], f"{s['name']}.{ext}")) for ext in ("gguf", "safetensors")}
        m["models"][t.split("@")[0]] = {"tag": t, "spec": path, "name": s["name"], "arch": s.get("arch"),
                                        "vocab": s["data"]["vocab"], "T": s["data"]["T"],
                                        **{f"{ext}_sha256": sha256(p) for ext, p in files.items()},
                                        "files": files}
    json.dump(m, open(out, "w"), indent=1)
    print(json.dumps({t: (v["name"], v["gguf_sha256"][:8]) for t, v in m["models"].items()}))


if __name__ == "__main__":
    if sys.argv[1] == "--manifest":
        manifest(sys.argv[2], sys.argv[3], sys.argv[4:])
    else:
        print(stage(sys.argv[1])[0])
