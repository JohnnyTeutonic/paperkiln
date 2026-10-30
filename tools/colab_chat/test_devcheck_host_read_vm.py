"""Offline VM gate protocol tests: no CUDA calls or paid resources."""
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('devcheck_vm', Path(__file__).with_name('devcheck_host_read_vm.py'))
vm = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(vm)
FILES = ['CMakeLists.txt', 'include/microtorch/device_cache.hpp', 'src/cuda_resident.cu', 'src/autograd.cpp',
         'src/nn.cpp', 'src/depth_attend.cpp', 'src/safetensors.cpp', 'src/llama.cpp', 'src/llama_attnres.cpp',
         'src/moe.cpp', 'src/ops.cpp', 'tests/test_devcheck_host_read.cpp']


@pytest.mark.parametrize('mode,passed', [('good', True), ('wrong_gpu', False), ('build_error', False),
                                      ('cpu_fallback', False), ('regression_error', False)])
def test_gate_requires_real_proof_and_always_lands_all_declared_outputs(tmp_path, monkeypatch, mode, passed):
    root, out = tmp_path / 'source', tmp_path / 'out'
    for name in FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture, not source evidence')
    monkeypatch.setattr(vm.subprocess, 'check_output', lambda *a, **k: 'fixture-commit\n')

    class Process:
        def __init__(self, argv, **kwargs):
            self.argv = [str(a).replace(chr(92), '/') for a in argv]
            name = self.argv[0]
            if name == 'nvidia-smi':
                text = 'NVIDIA A100,123\n' if mode == 'wrong_gpu' else 'NVIDIA L4,123\n'
            elif name.endswith('/test_devcheck_host_read'):
                text = json.dumps({'pass': True, 'devcheck_enabled': True, 'cuda_integration': mode != 'cpu_fallback',
                                   'stale_and_released_predicate_negatives': True, 'macro_evaluation_checked': True}) + '\n'
            else:
                text = 'offline fixture\n'
            kwargs['stdout'].write(text)

        def wait(self, timeout):
            if mode == 'build_error' and '--build' in self.argv:
                return 1
            if mode == 'regression_error' and self.argv[0].endswith('/test_gpt2_device'):
                return 1
            return 0

    monkeypatch.setattr(vm.subprocess, 'Popen', Process)
    assert vm.run(out, root=root) == 0  # Transport completion is not a scientific pass.
    result = json.loads((out / 'gate_summary.json').read_text())
    assert result['pass'] is passed
    if not passed:
        assert result['errors']
    job = json.loads(Path(__file__).with_name('devcheck_host_read_20261030.json').read_text())['jobs'][0]
    assert all((out / name).is_file() for name in job['outputs'])
