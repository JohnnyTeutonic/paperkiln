#!/usr/bin/env python3
"""Operational two-pod fence bound to one exact reviewed Mamba1 driver."""
import hashlib
import json
from pathlib import Path
import runpy
import sys
PINS = {'/home/jonat/pk_host_helper_launch_20261031/tools/colab_chat/runner_multi.py': '273abf725d9ec79c3d43a5fa485ae282cb76671d808c1b8fa9893239e017b3ea', '/home/jonat/pk_host_helper_launch_20261031/tools/colab_chat/runpod_api.py': 'e0812f37b743e7c60a20c18a9b97b63b5a2fe692541ec3a6442bb361e5065e3b', '/home/jonat/pk_host_helper_launch_20261031/tools/colab_chat/runpod_transport.py': '4e9e99b1642f1e0b88edc6a8180fadef91cf0bd73749898e541429cb1ae70690', '/home/jonat/pk_host_helper_launch_20261031/tools/colab_chat/boot_commands_sparse.sh': '9bc4abc080b2b6bb0ac1811b54db57da245450e11ef453752c39b0c2ff08cfe6', '/home/jonat/pk_host_helper_launch_20261031/tools/colab_chat/jobs_vm.py': 'db8c6c3d9370cda09723b6421a4363d707d6fa2ba8c1fc3deeb87c23f8b558ae', '/home/jonat/pk_host_helper_launch_20261031/tools/colab_chat/host_helper_gate_20261031.json': '38ebee612422f18c083aeb038bc2e05db08d53745f66b8e407f78d9f8ed8fac8', '/home/jonat/host_helper_launch_v1_20261031/launch_review.json': 'c653da66a7ec4e19ea71c923c35ad4fd014fa10b7ca809bc30929890b5a73062', '/home/jonat/host_helper_launch_v1_20261031/host_replay_guard.py': 'e6a524d1af6f05c91a4ceee8928e35e770210cda66096a5d5d1aa76006b32b4e', '/home/jonat/host_helper_gate_stage_unlaunched_v3_20261031/inputs.json': '3ab069e1f144aae1461a1a0c6bb3c1bfd584b25d8f1e6fe41d7caba1a7458ed7', '/home/jonat/host_helper_gate_stage_unlaunched_v3_20261031/host_helper_gate.py': '4d0049082b89976388749101a64e205f748c587719363baac0ad48c8fced9422', '/home/jonat/host_helper_gate_stage_unlaunched_v3_20261031/replay_host_helper_gate.py': '8d7d7962a88372734cf6af31539ca8254d3954db708d503646122a9797e722a0', '/home/jonat/host_helper_gate_stage_unlaunched_v3_20261031/protocol.json': '9911d9e092cc265350a92936eaf7149691f5305950e5abe78b03c7b42226ccaf', '/home/jonat/host_helper_gate_stage_unlaunched_v3_20261031/source.tgz': '95ccb8d606a7db85300e82e58ec5380976e64fd0f5b4b005723d680e1291b200'}
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
