#!/usr/bin/env python3
"""Operational two-pod fence bound to one exact reviewed Mamba1 driver."""
import hashlib
import json
from pathlib import Path
import runpy
import sys
PINS = {'/home/jonat/pk_mamba_composed_20261031/tools/colab_chat/runner_multi.py': '273abf725d9ec79c3d43a5fa485ae282cb76671d808c1b8fa9893239e017b3ea', '/home/jonat/pk_mamba_composed_20261031/tools/colab_chat/runpod_api.py': 'e0812f37b743e7c60a20c18a9b97b63b5a2fe692541ec3a6442bb361e5065e3b', '/home/jonat/pk_mamba_composed_20261031/tools/colab_chat/runpod_transport.py': '4e9e99b1642f1e0b88edc6a8180fadef91cf0bd73749898e541429cb1ae70690', '/home/jonat/pk_mamba_composed_20261031/tools/colab_chat/boot_commands_sparse.sh': '9bc4abc080b2b6bb0ac1811b54db57da245450e11ef453752c39b0c2ff08cfe6', '/home/jonat/pk_mamba_composed_20261031/tools/colab_chat/mamba1_composed_20261031.json': 'a78c6802f274fc92803fd63f8bae8068e4e6caf0b746ba03206a51dd9a0750c5', '/home/jonat/mamba1_composed_gate_stage_v1_20261031/launch_review.json': '0bad88f38c799c8c81ca2335ebb1883e204a036fc350e1f67d6428853eff6dc0'}
REVIEW = Path(__file__).resolve().parent / "launch_review.json"

def install_ceiling(runner):
    for path, expected in PINS.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError("operational driver binding changed")
    sys.path.insert(0, str(Path(runner).resolve().parent))
    import runpod_api
    if Path(runpod_api.__file__).resolve() != Path(runner).resolve().parent / "runpod_api.py":
        raise ValueError("unexpected RunPod API module")
    runpod_api.MAX_PODS = 2
    return runpod_api

def main():
    approved = json.loads(REVIEW.read_text())["driver_argv"][1:]
    if sys.argv[1:] != approved:
        raise ValueError("driver arguments differ from reviewed launch")
    runner = approved[0]
    install_ceiling(runner)
    sys.argv = approved
    runpy.run_path(runner, run_name="__main__")

if __name__ == "__main__":
    main()
