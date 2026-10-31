"""Offline verification of the bounded L4 bias gate; does not execute device work."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tarfile

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def audit(home, stage):
    home, stage = (Path(home), Path(stage))
    report = json.loads((home / 'gate_summary.json').read_text())
    manifest = json.loads((stage / 'bias_inputs.json').read_text())
    if not (report['inputs'] == manifest and report['pass'] is True and (report['errors'] == [])):
        raise ValueError("offline evidence contract differs: report['inputs'] == manifest and report['pass'] is True and (report['errors'] == [])")
    if not (report['source_restored'] is True and report['baseline_expected_numerical_failure'] is True):
        raise ValueError("offline evidence contract differs: report['source_restored'] is True and report['baseline_expected_numerical_failure'] is True")
    for name, expected in manifest['files'].items():
        if not sha(stage / name) == expected:
            raise ValueError('offline evidence contract differs: sha(stage / name) == expected')
    for name, expected in report['logs'].items():
        if not (sha(home / name) == expected['sha256'] and (home / name).stat().st_size == expected['bytes']):
            raise ValueError("offline evidence contract differs: sha(home / name) == expected['sha256'] and (home / name).stat().st_size == expected['bytes']")
    commands = report['commands']
    if not [x['name'] for x in commands] == ['hardware', 'nvcc', 'configure', 'baseline_build', 'baseline', 'fixed_build', 'fixed', 'parity_export']:
        raise ValueError("offline evidence contract differs: [x['name'] for x in commands] == ['hardware', 'nvcc', 'configure', 'baseline_build', 'baseline', 'fixed_build', 'fixed', 'parity_export']")
    if not [x['returncode'] for x in commands] == [0, 0, 0, 0, 1, 0, 0, 0]:
        raise ValueError("offline evidence contract differs: [x['returncode'] for x in commands] == [0, 0, 0, 0, 1, 0, 0, 0]")
    if not (home / 'hardware.log').read_text().strip().startswith('NVIDIA L4,'):
        raise ValueError("offline evidence contract differs: (home / 'hardware.log').read_text().strip().startswith('NVIDIA L4,')")
    flags = commands[2]['argv']
    if not all((x in flags for x in ('-DMICROTORCH_CUDA=ON', '-DMICROTORCH_DEVCHECK=ON', '-DCMAKE_CUDA_ARCHITECTURES=89'))):
        raise ValueError("offline evidence contract differs: all((x in flags for x in ('-DMICROTORCH_CUDA=ON', '-DMICROTORCH_DEVCHECK=ON', '-DCMAKE_CUDA_ARCHITECTURES=89')))")
    if not all((x['argv'][-1] == '--require-cuda' for x in (commands[4], commands[6], commands[7]))):
        raise ValueError("offline evidence contract differs: all((x['argv'][-1] == '--require-cuda' for x in (commands[4], commands[6], commands[7])))")
    baseline = (home / 'baseline.log').read_text()
    proof = {'actual_cuda': True, 'stale_computed_bias': True, 'host_poison': True}
    if not proof in [json.loads(x) for x in baseline.splitlines() if x.startswith('{')]:
        raise ValueError("offline evidence contract differs: proof in [json.loads(x) for x in baseline.splitlines() if x.startswith('{')]")
    expression = 'std::fabs(y->data(i, j) - expected) / (1 + std::fabs(expected)) < 2e-5'
    if not re.search('^CHECK failed .*test_add_bias_materialization\\.cpp:[0-9]+: ' + re.escape(expression) + '$', baseline, re.MULTILINE):
        raise ValueError("offline evidence contract differs: re.search('^CHECK failed .*test_add_bias_materialization\\\\.cpp:[0-9]+: ' + re.escape(expression) + '$', baseline, re.MULTILINE)")
    for name, keys in [('fixed', ('pass', 'cuda_integration', 'devcheck_enabled', 'forward_and_backward_reference')), ('parity_export', ('pass', 'cuda_integration', 'devcheck_enabled', 'stale_and_released_predicate_negatives', 'macro_evaluation_checked'))]:
        result = json.loads((home / (name + '.log')).read_text().strip().splitlines()[-1])
        if not (result == report[name] and all((result.get(k) is True for k in keys))):
            raise ValueError('offline evidence contract differs: result == report[name] and all((result.get(k) is True for k in keys))')
    if not report['baseline_ops_sha256'] == sha(stage / 'bias_baseline_ops.cpp') == 'e3942b4564d035243be9f8084025222d7f93a01c794a2e40b3ef2e86cde0f42e':
        raise ValueError("offline evidence contract differs: report['baseline_ops_sha256'] == sha(stage / 'bias_baseline_ops.cpp') == 'e3942b4564d035243be9f8084025222d7f93a01c794a2e40b3ef2e86cde0f42e'")
    with tarfile.open(stage / 'bias_source.tgz') as source:
        for name, expected in [('src/ops.cpp', '4d6fe6c7041e02fcc93d1cb29f0be15f8658315f91fc64a1b89f31473956ed85'), ('tests/test_add_bias_materialization.cpp', '3c3dcb65ce8006ff2721fe3e145550c33448c1be97179945fcd55fc6ccf53383'), ('CMakeLists.txt', 'b406da7371bc32ca8517ab4a543b772e34a6e2b4650c271ffe4b5e81d550ab65')]:
            if not hashlib.sha256(source.extractfile(name).read()).hexdigest() == expected:
                raise ValueError('offline evidence contract differs: hashlib.sha256(source.extractfile(name).read()).hexdigest() == expected')
        if not source.getmember('studio/chat.html').size > 0:
            raise ValueError("offline evidence contract differs: source.getmember('studio/chat.html').size > 0")
    if not report['fixed_ops_sha256'] == '4d6fe6c7041e02fcc93d1cb29f0be15f8658315f91fc64a1b89f31473956ed85':
        raise ValueError("offline evidence contract differs: report['fixed_ops_sha256'] == '4d6fe6c7041e02fcc93d1cb29f0be15f8658315f91fc64a1b89f31473956ed85'")
    if not report['baseline_binary_sha256'] != report['fixed_binary_sha256']:
        raise ValueError("offline evidence contract differs: report['baseline_binary_sha256'] != report['fixed_binary_sha256']")
    return {'pass': True, 'scope': 'Offline exact source/input/log/command/proof replay; no fresh device execution', 'report_sha256': sha(home / 'gate_summary.json'), 'manifest_sha256': sha(stage / 'bias_inputs.json'), 'raw_report': report, 'raw_test_logs': {n: (home / n).read_text() for n in ('baseline.log', 'fixed.log', 'parity_export.log')}, 'limitations': ['Only the stale computed bias add_bias seam and registered parity-export stale/released guards were tested.', 'No stale-x, arbitrary graph, model training, performance or endurance claim.']}
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--home', type=Path, required=True)
    p.add_argument('--stage', type=Path, required=True)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    result = audit(a.home, a.stage)
    if a.output:
        if a.output.exists():
            raise FileExistsError('immutable audit receipt exists')
        a.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = a.output.with_suffix('.partial')
        temporary.write_text(json.dumps(result, indent=1) + '\n')
        temporary.replace(a.output)
    print(json.dumps({'pass': True, 'report_sha256': result['report_sha256']}))
