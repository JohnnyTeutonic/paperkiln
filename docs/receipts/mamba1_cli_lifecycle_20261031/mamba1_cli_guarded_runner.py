#!/usr/bin/env python3
"""Operational two-pod fence bound to one exact reviewed Mamba1 driver."""
import hashlib
import json
from pathlib import Path
import runpy
import sys
PINS = {'/home/jonat/pk_mamba_composed_20261031/tools/colab_chat/runner_multi.py': '273abf725d9ec79c3d43a5fa485ae282cb76671d808c1b8fa9893239e017b3ea', '/home/jonat/pk_mamba_composed_20261031/tools/colab_chat/runpod_api.py': 'e0812f37b743e7c60a20c18a9b97b63b5a2fe692541ec3a6442bb361e5065e3b', '/home/jonat/pk_mamba_composed_20261031/tools/colab_chat/runpod_transport.py': '4e9e99b1642f1e0b88edc6a8180fadef91cf0bd73749898e541429cb1ae70690', '/home/jonat/pk_mamba_composed_20261031/tools/colab_chat/boot_commands_sparse.sh': '9bc4abc080b2b6bb0ac1811b54db57da245450e11ef453752c39b0c2ff08cfe6', '/home/jonat/pk_mamba_composed_20261031/tools/colab_chat/mamba1_cli_lifecycle_20261031.json': 'a4036c60bb0cde0659983e57b9d1fde41805d45493c4bbebe6060697b4f425bd', '/home/jonat/mamba1_cli_gate_stage_v1_20261031/launch_review.json': '9da5d13339fa54461a696529bbcf3b82fa2e57859421a6bbcd624ff7bb607f74', '/home/jonat/mamba1_cli_gate_stage_v1_20261031/host_replay_guard.py': '2bd08768a03f41d91b3efb0a2f2eaecd31b9a180ba2d4c5b53c2487d0bf55f1f', '/home/jonat/mamba1_cli_gate_stage_v1_20261031/host_pins.json': '5de70f399efda83d461c1304504b60404115727388245fca83f431a22d2f073f'}
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
