"""Actual L4 stale-bias baseline/fix comparison; transport completion is not PASS."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile

BASELINE_SHA = 'e3942b4564d035243be9f8084025222d7f93a01c794a2e40b3ef2e86cde0f42e'
FIXED_SHA = '4d6fe6c7041e02fcc93d1cb29f0be15f8658315f91fc64a1b89f31473956ed85'
TEST_SHA = '3c3dcb65ce8006ff2721fe3e145550c33448c1be97179945fcd55fc6ccf53383'
CMAKE_SHA = 'b406da7371bc32ca8517ab4a543b772e34a6e2b4650c271ffe4b5e81d550ab65'

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

def check_baseline(rc, text):
    if rc != 1 or '"actual_cuda":true,"stale_computed_bias":true,"host_poison":true' not in text:
        raise ValueError('baseline is not actual CUDA poisoned stale bias exit1')
    if 'CHECK failed std::fabs(y->data(i, j) - expected)' not in text:
        raise ValueError('baseline did not fail the registered forward numerical assertion')

def run(out, upload=Path('/content/chat/upload'), root=Path('/content/bias_source'), build=Path('/content/bias_build')):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    report = {'pass': False, 'scope': 'add_bias actual stale computed bias; fixed parity export host-read guards',
              'commands': [], 'errors': [], 'source_restored': False}
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
    source = root / 'src/ops.cpp'; original = None
    names = ('hardware','nvcc','configure','baseline_build','baseline','fixed_build','fixed','parity_export')
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
        manifest = json.loads((upload/'bias_inputs.json').read_text())
        report['inputs'] = manifest
        for filename, expected in manifest['files'].items():
            if sha(upload/filename) != expected: raise ValueError('input hash differs: ' + filename)
        root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(upload/'bias_source.tgz') as archive:
            for member in archive.getmembers():
                if member.name.startswith('/') or '..' in Path(member.name).parts or member.issym() or member.islnk():
                    raise ValueError('unsafe source archive')
            archive.extractall(root)
        if sha(source)!=FIXED_SHA or sha(root/'tests/test_add_bias_materialization.cpp')!=TEST_SHA or sha(root/'CMakeLists.txt')!=CMAKE_SHA:
            raise ValueError('frozen source/test/CMake differs')
        if sha(upload/'bias_baseline_ops.cpp')!=BASELINE_SHA: raise ValueError('baseline source differs')
        original = source.read_bytes()
        _, hardware = command('hardware', ['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],30)
        if len(hardware.strip().splitlines())!=1 or not hardware.startswith('NVIDIA L4,'): raise ValueError('requires one actual L4')
        command('nvcc',['nvcc','--version'],30)
        command('configure',['cmake','-S',root,'-B',build,'-DCMAKE_BUILD_TYPE=Release','-DMICROTORCH_CUDA=ON',
                             '-DMICROTORCH_DEVCHECK=ON','-DMICROTORCH_BUILD_PYTHON=OFF','-DCMAKE_CUDA_ARCHITECTURES=89'],300)
        source.write_bytes((upload/'bias_baseline_ops.cpp').read_bytes())
        command('baseline_build',['cmake','--build',build,'--target','test_add_bias_materialization','--parallel','4'],900)
        report['baseline_binary_sha256']=sha(build/'test_add_bias_materialization')
        rc,text=command('baseline',[build/'test_add_bias_materialization','--require-cuda'],120,(1,))
        check_baseline(rc,text); report['baseline_expected_numerical_failure']=True
        temporary=source.with_suffix('.restore');temporary.write_bytes(original);temporary.replace(source)
        command('fixed_build',['cmake','--build',build,'--target','test_add_bias_materialization','test_devcheck_host_read','--parallel','4'],900)
        report['fixed_binary_sha256']=sha(build/'test_add_bias_materialization')
        report['parity_binary_sha256']=sha(build/'test_devcheck_host_read')
        _,text=command('fixed',[build/'test_add_bias_materialization','--require-cuda'],120)
        report['fixed']=parse_proof(text,('pass','cuda_integration','devcheck_enabled','forward_and_backward_reference'))
        _,text=command('parity_export',[build/'test_devcheck_host_read','--require-cuda'],120)
        report['parity_export']=parse_proof(text,('pass','cuda_integration','devcheck_enabled','stale_and_released_predicate_negatives','macro_evaluation_checked'))
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
