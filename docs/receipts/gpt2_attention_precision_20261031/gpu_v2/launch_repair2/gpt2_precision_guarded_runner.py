#!/usr/bin/env python3
"""Operational two-pod window fence for one root-approved frozen driver only."""
import hashlib
import json
from pathlib import Path
import runpy
import sys
PINS = {'/home/jonat/pk_gpt2_model_transport_20261031/tools/colab_chat/runner_multi.py': '273abf725d9ec79c3d43a5fa485ae282cb76671d808c1b8fa9893239e017b3ea', '/home/jonat/pk_gpt2_model_transport_20261031/tools/colab_chat/runpod_api.py': 'e0812f37b743e7c60a20c18a9b97b63b5a2fe692541ec3a6442bb361e5065e3b', '/home/jonat/pk_gpt2_model_transport_20261031/tools/colab_chat/runpod_transport.py': '4e9e99b1642f1e0b88edc6a8180fadef91cf0bd73749898e541429cb1ae70690', '/home/jonat/gpt2_precision_gate_launch_repair2_20261031/launch_review.json': '54ee3db44c4c260464f9527e0eb7c32e62b187dac7d1a3a390487c10c193104b'}
REVIEW = Path(__file__).resolve().parent / 'launch_review.json'


def install_ceiling(runner):
    for path, expected in PINS.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError('operational driver binding changed')
    sys.path.insert(0, str(Path(runner).resolve().parent))
    import runpod_api
    runpod_api.MAX_PODS = 2
    return runpod_api


def main():
    approved = json.loads(REVIEW.read_text())['driver_argv'][1:]
    if sys.argv[1:] != approved:
        raise ValueError('driver arguments differ from reviewed launch')
    runner = approved[0]
    install_ceiling(runner)
    sys.argv = approved
    runpy.run_path(runner, run_name='__main__')


if __name__ == '__main__':
    main()
