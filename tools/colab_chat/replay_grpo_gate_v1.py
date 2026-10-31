"""Offline replay of original numeric failures and fixed actual-device helper evidence."""
import importlib.util
import json
from pathlib import Path
import re
import sys
P=Path(__file__).with_name('grpo_gate_vm_v1.py')
s=importlib.util.spec_from_file_location('gate',P);gate=importlib.util.module_from_spec(s);s.loader.exec_module(gate)

def replay(directory):
    directory=Path(directory);report=json.loads((directory/'gate_summary.json').read_text())
    names=('hardware','nvcc','configure','baseline_build','fixed_build')+tuple(phase+'_'+case for phase in ('baseline','fixed') for case in gate.CASES)
    if set(report.get('logs',{}))!={n+'.log' for n in names}:raise ValueError('raw log inventory differs')
    for name,item in report['logs'].items():
        p=directory/name
        if gate.sha(p)!=item['sha256'] or p.stat().st_size!=item['bytes']:raise ValueError('raw log differs: '+name)
    result={'integrity':True,'scientific_pass':False,'scope':'two real CLI host consumers only; no training-quality claim','reported_errors':report.get('errors',[])}
    if report.get('pass') is not True:return result
    if report.get('errors')!=[] or report.get('source_restored') is not True:raise ValueError('positive report has errors or unrestored source')
    if report.get('source_commit')!=gate.SOURCE_COMMIT:raise ValueError('source pin differs')
    if report.get('baseline_mtstudio_sha256')!=gate.BASELINE_SHA or report.get('fixed_mtstudio_sha256')!=gate.FIXED_SHA:raise ValueError('source phase hashes differ')
    files=report.get('inputs',{}).get('files',{})
    for name,expected in [('grpo_source.tgz',gate.ARCHIVE_SHA),('grpo_baseline_mtstudio.cpp',gate.BASELINE_SHA),('grpo_gate_vm_v1.py',gate.sha(P))]:
        if files.get(name)!=expected:raise ValueError('input binding differs: '+name)
    commands=report.get('commands',[]);by_name={x['name']:x for x in commands}
    if len(commands)!=len(names) or set(by_name)!=set(names):raise ValueError('command inventory differs')
    hardware=(directory/'hardware.log').read_text().strip().splitlines()
    if len(hardware)!=1 or not hardware[0].startswith('NVIDIA L4,'):raise ValueError('actual L4 evidence differs')
    if 'release ' not in (directory/'nvcc.log').read_text():raise ValueError('nvcc version evidence missing')
    for flag in ('-DMICROTORCH_CUDA=ON','-DMICROTORCH_DEVCHECK=ON','-DMICROTORCH_BUILD_PYTHON=OFF','-DCMAKE_CUDA_ARCHITECTURES=89'):
        if flag not in by_name['configure']['argv']:raise ValueError('configure differs: '+flag)
    fixed={}
    for case in gate.CASES:
        original=by_name['baseline_'+case];new=by_name['fixed_'+case]
        if original['argv'][-3:]!=['--require-cuda','--case',case] or new['argv'][-3:]!=['--require-cuda','--case',case]:raise ValueError('actual-device case invocation missing')
        text=(directory/('baseline_'+case+'.log')).read_text();gate.check_baseline(original['returncode'],text,case)
        if new['returncode']!=0:raise ValueError('fixed case failed')
        fixed_text=(directory/('fixed_'+case+'.log')).read_text()
        proof=gate.parse_proof(fixed_text,('pass','actual_cuda','devcheck_enabled','no_grad_context','cpu_reference','stale_producer','host_poison'))
        rows=[json.loads(x) for x in fixed_text.splitlines() if x.startswith('{')]
        producer={'actual_cuda':True,'stale_producer':True,'host_poison':True,'no_grad_context':True,'deferred_window':True}
        if producer not in rows:raise ValueError('actual NoGrad/open-window evidence missing')
        required='sampler_ids_logp_rng' if case=='sample' else 'all_softmax_rows'
        if proof.get('case')!=case or proof.get(required) is not True or proof!=report['fixed'][case]:raise ValueError('fixed reference proof differs')
        fixed[case]=proof
    for name in ('hardware','nvcc','configure','baseline_build','fixed_build'):
        if by_name[name]['returncode']!=0:raise ValueError('setup/build failed')
    binaries=[report.get(k,'') for k in ('baseline_binary_sha256','fixed_binary_sha256')]
    if any(not re.fullmatch('[0-9a-f]{64}',v) for v in binaries) or binaries[0]==binaries[1]:raise ValueError('baseline/fixed executable hash evidence differs')
    result.update(scientific_pass=True,source_commit=gate.SOURCE_COMMIT,original_numeric_failures=list(gate.CASES),fixed=fixed,binary_binding='recorded executable SHA256; binary bytes are not downloaded')
    return result

if __name__=='__main__':
    result=replay(sys.argv[1]);print(json.dumps(result,indent=2));raise SystemExit(0 if result['scientific_pass'] else 1)
