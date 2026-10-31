"""Strict one-shot actual-L4 stale-host fallback gate. Run only through reviewed transport."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import tarfile
import time

NEGATIVE = (
    'gelu_forward_x', 'gelu_backward_x', 'silu_backward_x',
    'softmax_forward_x', 'softmax_backward_y', 'sigmoid_forward_x', 'sigmoid_backward_y',
    'rmsnorm_forward_x', 'rmsnorm_forward_w', 'rmsnorm_backward_x', 'rmsnorm_backward_w',
    'group_forward_w', 'group_backward_w',
)
CONTROLS = ('silu_forward_x', 'group_forward_x', 'group_backward_x')
CASES = NEGATIVE + CONTROLS

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def rows(text):
    return [json.loads(line) for line in text.splitlines() if line.startswith('{')]

def require_stale(text, case):
    proof = {'case': case, 'actual_cuda': True, 'stale_operand_proved': True,
             'host_poison': True, 'ops_disabled_before_fallback': True}
    if proof not in rows(text):
        raise ValueError('missing actual-device stale operand/fallback proof')

def quantities(case):
    return ('forward', 'input_gradient', 'weight_gradient') if case.startswith(('rmsnorm_', 'group_')) else ('forward', 'input_gradient')

def check_numeric(row, case, quantity, passing):
    if row.get('case') != case or row.get('quantity') != quantity:
        raise ValueError('numeric case or quantity differs')
    error = row.get('normalized_max_error')
    if type(error) not in (float, int) or not math.isfinite(error) or error < 0:
        raise ValueError('nonfinite or missing numerical diagnostic')
    if row.get('numeric_reference_pass') is not passing or (error < 2e-5) is not passing:
        raise ValueError('numerical diagnostic disagrees with fixed threshold')

def check_baseline(rc, text, case):
    if rc != 1 or case not in NEGATIVE:
        raise ValueError('baseline must fail a registered numerical check with exit 1')
    require_stale(text, case)
    numeric = [x for x in rows(text) if 'quantity' in x]
    expected_quantity = 'forward' if '_forward_' in case else 'input_gradient'
    expected_prefix = list(quantities(case))[:list(quantities(case)).index(expected_quantity) + 1]
    if [x.get('quantity') for x in numeric] != expected_prefix:
        raise ValueError('baseline failed a different numerical stage')
    for row in numeric[:-1]:
        check_numeric(row, case, row['quantity'], True)
    check_numeric(numeric[-1], case, expected_quantity, False)
    if not re.search(r'^CHECK failed .*test_fallback_materialization\.cpp:[0-9]+: maximum < 2e-5$', text, re.MULTILINE):
        raise ValueError('baseline failed wrong assertion')
    if any(x.get('pass') is True for x in rows(text)):
        raise ValueError('baseline unexpectedly passed')

def check_fixed(rc, text, case):
    if rc != 0:
        raise ValueError('fixed/control executable failed')
    require_stale(text, case)
    parsed = rows(text)
    numeric = [x for x in parsed if 'quantity' in x]
    if [x.get('quantity') for x in numeric] != list(quantities(case)):
        raise ValueError('missing/duplicate numerical reference checks')
    for row in numeric:
        check_numeric(row, case, row['quantity'], True)
    if {'pass': True, 'case': case, 'actual_cuda': True, 'independent_forward_backward': True} not in parsed:
        raise ValueError('missing case reference proof')
    if parsed[-1] != {'pass': True, 'actual_cuda': True, 'devcheck_enabled': True, 'cases': 1}:
        raise ValueError('missing strict CUDA/DEVCHECK final proof')

def run(upload, expected_manifest, out, root, build, max_seconds=1500):
    upload, out, root, build = map(Path, (upload, out, root, build))
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError('gate evidence is immutable; choose a new reviewed attempt')
    deadline = time.monotonic() + max_seconds
    report = {'pass': False, 'commands': [], 'errors': [], 'source_restored': False,
              'scope': 'conditional runtime device-op fallback only; no default-on corruption or performance claim'}
    original = None
    source = root / 'src/ops.cpp'
    logs = []
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
    def command(name, argv, timeout, allowed=(0,)):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError('gate wall deadline expired')
        log = out / (name + '.log'); logs.append(log)
        with log.open('w') as stream:
            child = subprocess.Popen(list(map(str, argv)), cwd=root, env=env, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            try:
                rc = child.wait(timeout=min(timeout, remaining))
            except (subprocess.TimeoutExpired, InterruptedError):
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait()
                raise ValueError(name + ' timed out or interrupted') from None
        report['commands'].append({'name': name, 'argv': list(map(str, argv)), 'returncode': rc})
        text = log.read_text(errors='replace')
        if rc not in allowed:
            raise ValueError(name + ' failed: exit ' + str(rc))
        return rc, text
    try:
        manifest_path = upload / 'inputs.json'
        if sha(manifest_path) != expected_manifest:
            raise ValueError('reviewed manifest hash differs')
        manifest = json.loads(manifest_path.read_text())
        report['inputs'] = manifest
        if manifest['negative_cases'] != list(NEGATIVE) or manifest['control_cases'] != list(CONTROLS):
            raise ValueError('registered case matrix differs')
        for filename, expected in manifest['files'].items():
            if Path(filename).name != filename or sha(upload / filename) != expected:
                raise ValueError('frozen input differs: ' + filename)
        root.mkdir(parents=True, exist_ok=True)
        if any(root.iterdir()):
            raise ValueError('source destination must be empty')
        with tarfile.open(upload / 'source.tgz') as archive:
            for member in archive.getmembers():
                if member.name.startswith('/') or '..' in Path(member.name).parts or not (member.isdir() or member.isfile()):
                    raise ValueError('unsafe archive member')
            archive.extractall(root)
        for filename, expected in manifest['source_files'].items():
            if sha(root / filename) != expected:
                raise ValueError('frozen extracted source differs: ' + filename)
        if sha(Path(__file__)) != manifest['files']['fallback_gate.py']:
            raise ValueError('executing gate differs from manifest')
        original = source.read_bytes()
        _, hardware = command('hardware', ['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'], 30)
        if len(hardware.strip().splitlines()) != 1 or not hardware.startswith('NVIDIA L4,'):
            raise ValueError('requires one actual NVIDIA L4')
        command('nvcc', ['nvcc', '--version'], 30)
        command('configure', ['cmake', '-S', root, '-B', build, '-DCMAKE_BUILD_TYPE=Release',
                             '-DMICROTORCH_CUDA=ON', '-DMICROTORCH_DEVCHECK=ON',
                             '-DMICROTORCH_BUILD_PYTHON=OFF', '-DCMAKE_CUDA_ARCHITECTURES=89'], 180)
        source.write_bytes((upload / 'baseline_ops.cpp').read_bytes())
        report['baseline_source_sha256'] = sha(source)
        command('baseline_build', ['cmake', '--build', build, '--target', 'test_fallback_materialization', '--parallel', '4'], 750)
        binary = build / 'test_fallback_materialization'
        report['baseline_binary_sha256'] = sha(binary)
        report['baseline'] = {}
        for case in NEGATIVE:
            rc, text = command('baseline_' + case, [binary, '--require-cuda', '--case', case], 30, (1,))
            check_baseline(rc, text, case)
            report['baseline'][case] = {'registered_numerical_failure': True}
        for case in CONTROLS:
            rc, text = command('control_' + case, [binary, '--require-cuda', '--case', case], 30)
            check_fixed(rc, text, case)
        source.write_bytes(original)
        report['fixed_source_sha256'] = sha(source)
        command('fixed_build', ['cmake', '--build', build, '--target', 'test_fallback_materialization', '--parallel', '4'], 300)
        report['fixed_binary_sha256'] = sha(binary)
        if report['fixed_binary_sha256'] == report['baseline_binary_sha256']:
            raise ValueError('old/fixed executable bytes identical')
        report['fixed'] = {}
        for case in CASES:
            rc, text = command('fixed_' + case, [binary, '--require-cuda', '--case', case], 30)
            check_fixed(rc, text, case)
            report['fixed'][case] = {'independent_forward_backward': True}
        report['pass'] = True
    except (OSError, ValueError, KeyError, IndexError, subprocess.SubprocessError) as exc:
        report['errors'].append(str(exc))
    finally:
        if original is not None:
            temporary = source.with_suffix('.restore'); temporary.write_bytes(original); temporary.replace(source)
            report['source_restored'] = source.read_bytes() == original
            if not report['source_restored']:
                report['pass'] = False; report['errors'].append('fixed source restoration failed')
        report['logs'] = {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size} for p in logs}
        temporary = out / 'gate_summary.partial'; temporary.write_text(json.dumps(report, indent=2) + '\n'); temporary.replace(out / 'gate_summary.json')
    print(json.dumps({'pass': report['pass'], 'errors': report['errors']}))
    return 0  # Transport completion is deliberately distinct from numerical PASS.

def interrupted(signum, frame):
    raise InterruptedError('termination signal ' + str(signum))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--upload', required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--root', default='/content/fallback_source')
    parser.add_argument('--build', default='/content/fallback_build')
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(run(args.upload, args.manifest_sha256, args.out, args.root, args.build))
