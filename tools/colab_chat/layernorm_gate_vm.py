"""Actual L4 LayerNorm fallback baseline/fix comparison; transport completion is not PASS."""
import hashlib
import json
import os
import re
from pathlib import Path
import signal
import subprocess
import sys
import tarfile

BASELINE_SHA = '9578ee8cfc3cd5c98f201df71834291a37b73585a114733a335cf7779ccf2abd'
FIXED_SHA = '94e4d38b17066362dc9e40d7ea64705ccb3834b89034756f3d929453236dd81a'
TEST_SHA = 'e13d9a0cc7e8013ed908db448327dcb2e71d3abf6d050764488dbf94ea251535'
CMAKE_SHA = '217b24c7e3278d723b98e0aa59ce982eac72ea66abeddbd257f33a0bb9f76550'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def parse_proof(text, keys):
    proof = json.loads(text.strip().splitlines()[-1])
    if not all(proof.get(key) is True for key in keys):
        raise ValueError('missing explicit actual CUDA/DEVCHECK proof')
    return proof

CASES = ('forward_x','forward_gamma','forward_beta','backward_gamma')
FORWARD_EXPRESSION = 'std::fabs(y->data(i, j) - expected[i * C + j]) / (1 + std::fabs(expected[i * C + j])) < 2e-5'
BACKWARD_EXPRESSION = 'std::fabs(x0->grad(i, j) - dx[i * C + j]) / (1 + std::fabs(dx[i * C + j])) < 2e-5'

def check_baseline(rc, text, case):
    if rc != 1 or case not in CASES:
        raise ValueError('baseline must fail registered numeric check with exit1')
    rows=[json.loads(x) for x in text.splitlines() if x.startswith('{')]
    proof={'case':case,'actual_cuda':True,'stale_operand_proved':True,'host_poison':True,'ops_disabled_before_fallback':True}
    if proof not in rows:
        raise ValueError('baseline lacks actual stale device operand/fallback proof')
    expression=BACKWARD_EXPRESSION if case=='backward_gamma' else FORWARD_EXPRESSION
    pattern=r'^CHECK failed .*test_layernorm_materialization\.cpp:[0-9]+: '+re.escape(expression)+r'$'
    if not re.search(pattern,text,re.MULTILINE):
        raise ValueError('baseline failed wrong assertion or stage')

def run(out, upload=Path('/content/chat/upload'), root=Path('/content/layernorm_source'), build=Path('/content/layernorm_build')):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    report = {'pass': False, 'scope': 'LayerNorm fallback stale x/gamma/beta forward and stale gamma backward',
              'commands': [], 'errors': [], 'source_restored': False}
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
    source = root / 'src/ops.cpp'; original = None
    names = ('hardware','nvcc','configure','baseline_build','fixed_build') + tuple(phase+'_'+case for phase in ('baseline','fixed') for case in CASES)
    for name in names: (out / (name + '.log')).write_text('NOT_RUN\n')
    def command(name, argv, timeout, allowed=(0,)):
        with (out / (name + '.log')).open('w') as log:
            child = subprocess.Popen(list(map(str,argv)), cwd=root, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            try: rc = child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                try: child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait()
                raise ValueError(name + ' timed out') from None
        report['commands'].append({'name': name, 'argv': list(map(str,argv)), 'returncode': rc})
        text = (out / (name + '.log')).read_text(errors='replace')
        if rc not in allowed: raise ValueError(name + ' failed with exit ' + str(rc))
        return rc, text
    try:
        manifest = json.loads((upload/'layernorm_inputs.json').read_text())
        report['inputs'] = manifest
        for filename, expected in manifest['files'].items():
            if sha(upload/filename) != expected: raise ValueError('input hash differs: ' + filename)
        root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(upload/'layernorm_source.tgz') as archive:
            for member in archive.getmembers():
                if member.name.startswith('/') or '..' in Path(member.name).parts or member.issym() or member.islnk():
                    raise ValueError('unsafe source archive')
            archive.extractall(root)
        if sha(source)!=FIXED_SHA or sha(root/'tests/test_layernorm_materialization.cpp')!=TEST_SHA or sha(root/'CMakeLists.txt')!=CMAKE_SHA:
            raise ValueError('frozen source/test/CMake differs')
        if sha(upload/'layernorm_baseline_ops.cpp')!=BASELINE_SHA: raise ValueError('baseline source differs')
        original = source.read_bytes()
        _, hardware = command('hardware', ['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],30)
        if len(hardware.strip().splitlines())!=1 or not hardware.startswith('NVIDIA L4,'): raise ValueError('requires one actual L4')
        command('nvcc',['nvcc','--version'],30)
        command('configure',['cmake','-S',root,'-B',build,'-DCMAKE_BUILD_TYPE=Release','-DMICROTORCH_CUDA=ON',
                             '-DMICROTORCH_DEVCHECK=ON','-DMICROTORCH_BUILD_PYTHON=OFF','-DCMAKE_CUDA_ARCHITECTURES=89'],180)
        source.write_bytes((upload/'layernorm_baseline_ops.cpp').read_bytes())
        report['baseline_ops_sha256']=sha(source)
        if report['baseline_ops_sha256']!=BASELINE_SHA: raise ValueError('baseline phase source differs')
        command('baseline_build',['cmake','--build',build,'--target','test_layernorm_materialization','--parallel','4'],600)
        report['baseline_binary_sha256']=sha(build/'test_layernorm_materialization')
        report['baseline']={}
        for case in CASES:
            rc,text=command('baseline_'+case,[build/'test_layernorm_materialization','--require-cuda','--case',case],30,(1,))
            check_baseline(rc,text,case)
            report['baseline'][case]={'returncode':rc,'registered_numerical_failure':True}
        report['baseline_expected_numerical_failure']=True
        temporary=source.with_suffix('.restore');temporary.write_bytes(original);temporary.replace(source)
        report['fixed_ops_sha256']=sha(source)
        if report['fixed_ops_sha256']!=FIXED_SHA: raise ValueError('fixed phase source differs')
        command('fixed_build',['cmake','--build',build,'--target','test_layernorm_materialization','--parallel','4'],300)
        report['fixed_binary_sha256']=sha(build/'test_layernorm_materialization')
        report['fixed']={}
        for case in CASES:
            _,text=command('fixed_'+case,[build/'test_layernorm_materialization','--require-cuda','--case',case],30)
            proof=parse_proof(text,('pass','cuda_integration','devcheck_enabled'))
            if proof.get('cases')!=1:raise ValueError('fixed case inventory differs')
            rows=[json.loads(x) for x in text.splitlines() if x.startswith('{')]
            if not any(x.get('case')==case and x.get('pass') is True and x.get('forward_backward_reference') is True for x in rows):
                raise ValueError('fixed specific case reference proof absent')
            report['fixed'][case]=proof
        if report['baseline_binary_sha256']==report['fixed_binary_sha256']: raise ValueError('baseline and fixed executable hashes unexpectedly identical')
        report['pass']=True
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        report['errors'].append(str(exc))
    finally:
        if original is not None:
            temporary=source.with_suffix('.restore');temporary.write_bytes(original);temporary.replace(source)
            report['source_restored']=sha(source)==FIXED_SHA
            if not report['source_restored']:report['pass']=False;report['errors'].append('fixed source restoration failed')
        report['logs']={name+'.log':{'sha256':sha(out/(name+'.log')),'bytes':(out/(name+'.log')).stat().st_size} for name in names}
        temporary=out/'gate_summary.partial';temporary.write_text(json.dumps(report,indent=1)+'\n');temporary.replace(out/'gate_summary.json')
    print(json.dumps({'pass':report['pass'],'errors':report['errors']}))
    return 0

if __name__ == '__main__':
    raise SystemExit(run(sys.argv[1]))
