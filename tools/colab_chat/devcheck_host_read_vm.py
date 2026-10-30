"""Bounded command-only RunPod DEVCHECK gate; diagnostic completion is not a pass."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


def run(out, root=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(root) if root else Path(__file__).resolve().parents[2]
    for name in ('hardware', 'nvcc', 'configure', 'build', 'host_read_negative_positive',
                 'gpt2_device_regression', 'depth_attend_regression'):
        (out / f'{name}.log').write_text('NOT_RUN\n')
    build = Path('/content/devcheck_build')
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    report = {'pass': False, 'scope': 'audited non-WIP host-read boundaries; ops.cpp/mtstudio/parity_model not audited',
              'cuda_required': True, 'devcheck_required': True, 'commands': [], 'errors': []}

    def command(name, argv, timeout=1200, extra_env=None):
        with (out / f'{name}.log').open('w') as log:
            p = subprocess.Popen(argv, cwd=root, env={**env, **(extra_env or {})},
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                rc = p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
                raise ValueError(name + ' timed out') from None
        report['commands'].append({'name': name, 'argv': list(map(str, argv)), 'returncode': rc})
        if rc:
            raise ValueError(f'{name} failed with exit {rc}')
        return (out / f'{name}.log').read_text(errors='replace')

    try:
        report['commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
        report['source_sha256'] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in
            ['CMakeLists.txt', 'include/microtorch/device_cache.hpp', 'src/cuda_resident.cu', 'src/autograd.cpp',
             'src/nn.cpp', 'src/depth_attend.cpp', 'src/safetensors.cpp', 'src/llama.cpp', 'src/llama_attnres.cpp',
             'src/moe.cpp', 'src/ops.cpp', 'tests/test_devcheck_host_read.cpp']}
        gpu = command('hardware', ['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'], 30)
        if len(gpu.strip().splitlines()) != 1 or not gpu.startswith('NVIDIA L4,'):
            raise ValueError('requires exactly one NVIDIA L4')
        command('nvcc', ['nvcc', '--version'], 30)
        command('configure', ['cmake', '-S', str(root), '-B', str(build), '-DCMAKE_BUILD_TYPE=Release',
                             '-DMICROTORCH_CUDA=ON', '-DMICROTORCH_DEVCHECK=ON', '-DMICROTORCH_BUILD_PYTHON=OFF',
                             '-DCMAKE_CUDA_ARCHITECTURES=89'], 300)
        command('build', ['cmake', '--build', str(build), '--target', 'test_devcheck_host_read',
                          'test_gpt2_device', 'test_llama_attnres', '--parallel', '4'], 1200)
        text = command('host_read_negative_positive', [str(build / 'test_devcheck_host_read'), '--require-cuda'], 120)
        proof = json.loads(text.strip().splitlines()[-1])
        if not all(proof.get(key) is True for key in ('pass', 'devcheck_enabled', 'cuda_integration',
                    'stale_and_released_predicate_negatives', 'macro_evaluation_checked')):
            raise ValueError('missing actual CUDA/DEVCHECK/negative integration proof')
        report['integration'] = proof
        command('gpt2_device_regression', [str(build / 'test_gpt2_device')], 300, {'MICROTORCH_DEVICE': 'cuda'})
        command('depth_attend_regression', [str(build / 'test_llama_attnres')], 300, {'MICROTORCH_DEVICE': 'cuda'})
        report['pass'] = True
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        report['errors'].append(str(exc))
    (out / 'gate_summary.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({'pass': report['pass'], 'errors': report['errors']}))
    # Always return diagnostics normally: gate_summary.pass is the scientific verdict.
    return 0


if __name__ == '__main__':
    raise SystemExit(run(sys.argv[1]))
