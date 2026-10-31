"""Offline strict replay of immutable actual-L4 fallback gate evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import fallback_gate as gate

def replay(folder, manifest_path, expected_manifest):
    folder, manifest_path = Path(folder), Path(manifest_path)
    if gate.sha(manifest_path) != expected_manifest:
        raise ValueError('reviewed manifest differs')
    manifest = json.loads(manifest_path.read_text())
    report = json.loads((folder / 'gate_summary.json').read_text())
    if report.get('pass') is not True or report.get('source_restored') is not True or report.get('errors') != []:
        raise ValueError('raw gate did not pass with source restoration')
    if report.get('inputs') != manifest:
        raise ValueError('report inputs differ from reviewed manifest')
    if report.get('baseline_source_sha256') != manifest['files']['baseline_ops.cpp']:
        raise ValueError('baseline source differs')
    if report.get('fixed_source_sha256') != manifest['source_files']['src/ops.cpp']:
        raise ValueError('fixed source differs')
    for key in ('baseline_binary_sha256', 'fixed_binary_sha256'):
        if len(report.get(key, '')) != 64:
            raise ValueError('missing binary hash')
    if report['baseline_binary_sha256'] == report['fixed_binary_sha256']:
        raise ValueError('binary bytes identical')
    expected_names = ['hardware', 'nvcc', 'configure', 'baseline_build']
    expected_names += ['baseline_' + c for c in gate.NEGATIVE]
    expected_names += ['control_' + c for c in gate.CONTROLS]
    expected_names += ['fixed_build'] + ['fixed_' + c for c in gate.CASES]
    commands = report.get('commands', [])
    if [x.get('name') for x in commands] != expected_names:
        raise ValueError('command inventory/order differs')
    if set(report.get('logs', {})) != {x + '.log' for x in expected_names}:
        raise ValueError('log inventory differs')
    for name, bound in report['logs'].items():
        path = folder / name
        if gate.sha(path) != bound['sha256'] or path.stat().st_size != bound['bytes']:
            raise ValueError('raw log differs: ' + name)
    hardware = (folder / 'hardware.log').read_text().strip().splitlines()
    if len(hardware) != 1 or not hardware[0].startswith('NVIDIA L4,'):
        raise ValueError('actual one-L4 proof missing')
    by_name = {x['name']: x for x in commands}
    configure = by_name['configure']['argv']
    for flag in ('-DMICROTORCH_CUDA=ON', '-DMICROTORCH_DEVCHECK=ON', '-DCMAKE_CUDA_ARCHITECTURES=89'):
        if flag not in configure:
            raise ValueError('strict build flag missing')
    for name, record in by_name.items():
        rc = record['returncode']
        if name.startswith('baseline_') and name != 'baseline_build':
            case = name[len('baseline_'):]
            gate.check_baseline(rc, (folder / (name + '.log')).read_text(), case)
        elif name.startswith(('fixed_', 'control_')) and name != 'fixed_build':
            case = name.split('_', 1)[1]
            if record['argv'][1:] != ['--require-cuda', '--case', case]:
                raise ValueError('registered executable invocation differs')
            gate.check_fixed(rc, (folder / (name + '.log')).read_text(), case)
        elif rc != 0:
            raise ValueError('setup/build command failed')
    if set(report.get('baseline', {})) != set(gate.NEGATIVE) or set(report.get('fixed', {})) != set(gate.CASES):
        raise ValueError('summary case inventories differ')
    return {'pass': True, 'scope': 'offline raw-log, command, source-manifest and numerical-proof replay',
            'manifest_sha256': expected_manifest, 'report_sha256': gate.sha(folder / 'gate_summary.json'),
            'negative_cases': len(gate.NEGATIVE), 'control_cases': len(gate.CONTROLS), 'fixed_cases': len(gate.CASES)}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = replay(args.folder, args.manifest, args.manifest_sha256)
    with Path(args.output).open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps(result))
