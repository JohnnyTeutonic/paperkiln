#!/usr/bin/env python3
"""Remote-only diagnostic packet; requires separate serial watchdog admission.

Exit zero means a complete diagnostic package returned, not that the gate passed.
Both execution.gate_pass and local raw-leg replay are required for a passing claim.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import tarfile
import time

from host_read_boundary_gate_v1 import legs, replay

SOURCE_FILES = ['src/device.cpp', 'tests/test_gemm_fallback_materialization.cpp',
                'tests/gemm_fallback_baseline.cpp', 'tests/test_legacy_matmul_materialization.cpp',
                'docs/receipts/legacy_matmul_materialization_20261102/baseline_device.cpp',
                'tools/colab_chat/host_read_boundary_gate_v1.py',
                'tools/colab_chat/run_host_read_boundaries_l4_v1.py', 'CMakeLists.txt']


def read_source_pin(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError('Expected source file must be regular')
    with path.open('rb') as stream:
        raw = stream.read(43)
    if len(raw) not in (40, 41) or (len(raw)==41 and raw[-1:]!=b'\n'):
        raise ValueError('Expected source file must contain exactly one full identifier')
    value = raw.decode('ascii').strip()
    if len(value)!=40 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Expected source identifier differs')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--expected-source')
    group.add_argument('--expected-source-file', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    # Refuse local laptop execution before any device probe or compilation.
    if root != Path('/content/pk') or not Path('/content/chat').is_dir():
        raise ValueError('This packet runs only in an admitted remote /content/pk worker')
    if args.expected_source_file is not None:
        args.expected_source = read_source_pin(args.expected_source_file)
    if len(args.expected_source) != 40 or any(c not in '0123456789abcdef' for c in args.expected_source):
        raise ValueError('Expected source must be a full Git object identifier')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any((out/name).exists() for name in ['execution.json', 'results.tar.gz', 'source_hardware.json']):
        raise FileExistsError('Preserve prior attempt; fresh diagnostic output required')
    started = time.monotonic()
    execution = {'kind': 'HOST_READ_BOUNDARY_EXECUTION_V1', 'diagnostic_package_complete': False,
                 'gate_pass': False, 'expected_source': args.expected_source, 'operational_error': None,
                 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}

    def command(argv, log=None, limit=60):
        remaining = 2700 - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError('45-minute packet deadline reached')
        env = os.environ | {'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '1'}
        if log is None:
            return subprocess.run(argv, cwd=root, env=env, capture_output=True, text=True,
                                  timeout=min(limit, remaining), check=True).stdout
        with (out/log).open('x', encoding='utf-8') as stream:
            return subprocess.run(argv, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                  timeout=min(limit, remaining)).returncode

    status = 1
    try:
        head = command(['git', 'rev-parse', 'HEAD']).strip()
        if head != args.expected_source:
            raise ValueError('Worker source differs from admission')
        command(['git', 'diff', '--quiet', 'HEAD'])
        hardware = command(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader']).strip().splitlines()
        if hardware != ['NVIDIA L4']:
            raise ValueError('Exactly one NVIDIA L4 is required')
        binding = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in SOURCE_FILES}
        (out/'source_hardware.json').write_text(json.dumps({'source_commit': head, 'hardware': hardware,
                                                          'source_sha256': binding}, indent=2)+'\n')
        build = Path('/content/host_read_boundary_build_v1')
        rc = command(['cmake', '-S', str(root), '-B', str(build), '-DMICROTORCH_CUDA=ON',
                      '-DMICROTORCH_DEVCHECK=ON', '-DCMAKE_BUILD_TYPE=Release',
                      '-DCMAKE_CUDA_ARCHITECTURES=89'], 'configure.txt', limit=180)
        if rc:
            raise ValueError('CMake configuration failed')
        targets = ['gemm_fallback_old', 'gemm_fallback_fixed', 'legacy_matmul_old', 'legacy_matmul_fixed']
        rc = command(['cmake', '--build', str(build), '--target', *targets, '--parallel', '2'],
                     'build.txt', limit=1800)
        if rc:
            raise ValueError('CUDA compilation failed')
        execution['binary_sha256'] = {name: hashlib.sha256((build/name).read_bytes()).hexdigest() for name in targets}
        # Each completed leg is durable evidence; builds are incremental on the volume.
        for leg in legs():
            target = ('gemm_fallback_' if leg['family']=='gemm' else 'legacy_matmul_') + leg['variant']
            argv = [str(build/target), '--require-cuda', '--transition', leg['transition']]
            if leg['b1']:
                argv.append('--b1')
            rc = command(argv, leg['name']+'.txt', limit=30)
            (out/(leg['name']+'.rc')).write_text(str(rc)+'\n')
        command(['git', 'diff', '--quiet', 'HEAD'])
        if command(['git', 'rev-parse', 'HEAD']).strip() != head:
            raise ValueError('Worker source changed during execution')
        if any(hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest for name,digest in binding.items()):
            raise ValueError('Bound source bytes changed during execution')
        result = replay(out)
        (out/'gate_summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        execution.update(diagnostic_package_complete=True, gate_pass=result['pass'])
        status = 0
    except Exception as error:
        execution['operational_error'] = type(error).__name__ + ': ' + str(error)[:600]
    finally:
        execution['elapsed_seconds'] = time.monotonic() - started
        (out/'execution.json').write_text(json.dumps(execution, indent=2, allow_nan=False)+'\n')
        members = sorted(p for p in out.iterdir() if p.is_file() and not p.is_symlink() and p.name!='results.tar.gz')
        with tarfile.open(out/'results.tar.gz', 'w:gz') as archive:
            for member in members:
                archive.add(member, arcname=member.name, recursive=False)
    print(json.dumps(execution, allow_nan=False))
    return status


if __name__ == '__main__':
    raise SystemExit(main())
