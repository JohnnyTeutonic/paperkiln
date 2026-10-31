"""Replay landed gate evidence without CUDA, API calls, or transport assumptions."""
import importlib.util
import json
from pathlib import Path
import re
import sys
P=Path(__file__).with_name('conv1d_gate_vm_v1.py')
s=importlib.util.spec_from_file_location('conv_gate',P);gate=importlib.util.module_from_spec(s);s.loader.exec_module(gate)

def replay(directory):
    directory=Path(directory)
    report=json.loads((directory/'gate_summary.json').read_text())
    if set(report.get('logs',{}))!={n+'.log' for n in gate.LOGS}:
        raise ValueError('missing raw log bindings')
    for name,record in report['logs'].items():
        path=directory/name
        if gate.sha(path)!=record['sha256'] or path.stat().st_size!=record['bytes']:
            raise ValueError('raw log differs: '+name)
    if report.get('source_commit')!=gate.SOURCE_COMMIT:
        raise ValueError('source pin differs')
    verdict={'integrity':True,'scientific_pass':False,'source_commit':gate.SOURCE_COMMIT,
             'reported_errors':report.get('errors',[]),'binary_binding':'recorded executable SHA256; binaries are not downloaded',
             'speedup_claim':False,'selective_scan_device_claim':False}
    # A landed, hash-consistent failure remains a failure. Missing numerical
    # phases cannot become successful merely because transport marked HOME.
    if report.get('pass') is not True:
        return verdict
    if report.get('errors')!=[]:raise ValueError('successful report has errors')
    if report.get('source_sha256')!=gate.CRITICAL:raise ValueError('source hashes differ')
    inputs=report.get('inputs',{})
    if inputs.get('source_commit')!=gate.SOURCE_COMMIT:raise ValueError('uploaded source pin differs')
    if inputs.get('files',{}).get('conv1d_source.tgz')!=gate.ARCHIVE_SHA:raise ValueError('archive binding differs')
    if inputs.get('files',{}).get('conv1d_gate_vm_v1.py')!=gate.sha(P):raise ValueError('VM program binding differs')
    commands=report.get('commands',[])
    if [c.get('name') for c in commands]!=list(gate.LOGS) or any(c.get('returncode')!=0 for c in commands):
        raise ValueError('actual successful command sequence differs')
    if commands[4]['argv'][-1:]!=['--require-cuda']:raise ValueError('strict actual-CUDA invocation missing')
    configure=commands[2]['argv']
    for flag in ('-DMICROTORCH_CUDA=ON','-DMICROTORCH_DEVCHECK=ON','-DMICROTORCH_BUILD_PYTHON=OFF','-DCMAKE_CUDA_ARCHITECTURES=89'):
        if flag not in configure:raise ValueError('configure proof differs: '+flag)
    hardware=(directory/'hardware.log').read_text().strip().splitlines()
    if len(hardware)!=1 or not hardware[0].startswith('NVIDIA L4,'):raise ValueError('actual L4 evidence differs')
    if 'release ' not in (directory/'nvcc.log').read_text():raise ValueError('nvcc version evidence missing')
    strict=gate.parse_proof((directory/'strict.log').read_text())
    thirdparty=gate.check_thirdparty((directory/'thirdparty.log').read_text())
    if strict!=report.get('strict') or thirdparty!=report.get('thirdparty'):raise ValueError('numerical proof differs from raw logs')
    binaries=report.get('binary_sha256',{})
    if set(binaries)!={'test_conv1d_cuda','test_thirdparty_ops'} or any(not re.fullmatch('[0-9a-f]{64}',value) for value in binaries.values()):
        raise ValueError('missing executable hash bindings')
    verdict.update(scientific_pass=True,strict=strict,thirdparty=thirdparty,recorded_binary_sha256=binaries)
    return verdict

if __name__=='__main__':
    result=replay(sys.argv[1]);print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['scientific_pass'] else 1)
