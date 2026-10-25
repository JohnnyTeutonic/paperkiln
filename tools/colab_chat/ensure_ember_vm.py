#!/usr/bin/env python3
"""ember.cpp's server on the VM at tiny_chat_run.EMBER_COMMIT, built once per VM (or RunPod volume)
by tiny_chat_run.ensure_ember, for jobs outside the tiny-chat programme (chat_eval_vm.sh,
chat_parity_vm.py).

    python3 tools/colab_chat/ensure_ember_vm.py OUT_JSON LOG

Writes the build record (bin, commit, repo, cuda, ...) to OUT_JSON; exits 1 when ember cannot be
built (the reason in OUT_JSON and LOG)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tiny_chat_run  # noqa: E402


def main():
    out, log = sys.argv[1], sys.argv[2]
    rec = tiny_chat_run.ensure_ember(log)
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps(rec))
    if "error" in rec or not rec.get("cuda"):
        sys.exit(f"ember: {rec.get('error') or 'built without CUDA'}")


if __name__ == "__main__":
    main()
