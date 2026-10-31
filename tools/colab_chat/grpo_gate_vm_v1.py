"""Actual L4 GRPO CLI host consumers baseline/fix comparison; transport completion is not PASS."""
import datetime
import hashlib
import json
import os
import re
from pathlib import Path
import signal
import subprocess
import sys
import tarfile

SOURCE_COMMIT = '226c5bbe2677ee66edb8d12b80f51ba2da684859'
ARCHIVE_SHA = '5e9a4236143a20f0ef3ec97884448d912af125eedb9ef0b12cfc389da0a12335'
DEADLINE = datetime.datetime(2026,10,31,3,5,tzinfo=datetime.timezone.utc)
BASELINE_SHA = '15b0fbc8ceb5f48a10d53b62b755c3c26cd8ac5e041c90b8b8f613ba40b5fb7d'
FIXED_SHA = 'c50b8cfb86fcd88f8f2ee06c56e824e3ca7e82ea009f04740962b8a7eb6200ef'
TEST_SHA = 'e413b8db4f02889f20d87a4e8a2fd2bb23847078447570390715f1c9ca989d1c'
CMAKE_SHA = '0dd1b48503e686371734131dd67dadf0a2981bfad4f094ef17b554b180bb6ba3'

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

CASES = ('sample', 'logsoftmax')
EXPRESSIONS = {'sample': 'std::fabs(actual.logp - expected.logp) < 1e-12',
               'logsoftmax': 'std::fabs(actual(i, j) - expected[i * 4 + j]) < 1e-6'}
def check_baseline(rc, text, case):
    if rc != 1 or case not in CASES:raise ValueError('baseline must fail exact registered numeric assertion with exit1')
    rows=[json.loads(x) for x in text.splitlines() if x.startswith('{')]
    proof={'actual_cuda':True,'stale_producer':True,'host_poison':True,'no_grad_context':True,'deferred_window':True}
    if proof not in rows:raise ValueError('baseline lacks actual stale CUDA NoGrad/open-window proof')
    pattern=r'^CHECK failed .*test_studio_grpo_materialization\.cpp:[0-9]+: '+re.escape(EXPRESSIONS[case])+r'$'
    if not re.search(pattern,text,re.MULTILINE):raise ValueError('baseline failed wrong assertion or stage')

def run(out, upload=Path('/content/chat/upload'), root=Path('/content/grpo_source'), build=Path('/content/grpo_build')):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError("existing gate evidence is immutable; operational review required")
    report = {'pass': False, 'scope': 'sample_pure and log_softmax_rows host boundaries only; no training-quality claim',
              'commands': [], 'errors': [], 'source_restored': False}
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
    source = root / 'tools/mtstudio.cpp'; original = None
    names = ('hardware','nvcc','configure','baseline_build','fixed_build') + tuple(phase+'_'+case for phase in ('baseline','fixed') for case in CASES)
    for name in names: (out / (name + '.log')).write_text('NOT_RUN\n')
    def command(name, argv, timeout, allowed=(0,)):
        remaining=(DEADLINE-datetime.datetime.now(datetime.timezone.utc)).total_seconds()
        if remaining<=0: raise ValueError('absolute deadline expired')
        timeout=min(timeout,remaining)
        with (out / (name + '.log')).open('w') as log:
            child = subprocess.Popen(list(map(str,argv)), cwd=root, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            try: rc = child.wait(timeout=timeout)
            except (subprocess.TimeoutExpired, InterruptedError):
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
        manifest = json.loads((upload/'grpo_inputs.json').read_text())
        report['inputs'] = manifest
        if manifest.get('source_commit') != SOURCE_COMMIT: raise ValueError('source pin differs')
        for filename, expected in manifest['files'].items():
            if sha(upload/filename) != expected: raise ValueError('input hash differs: ' + filename)
        if sha(upload/'grpo_source.tgz') != ARCHIVE_SHA: raise ValueError('frozen archive differs')
        root.mkdir(parents=True, exist_ok=True)
        if any(root.iterdir()): raise ValueError('source destination must be empty')
        with tarfile.open(upload/'grpo_source.tgz') as archive:
            for member in archive.getmembers():
                if member.name.startswith('/') or '..' in Path(member.name).parts or not (member.isfile() or member.isdir()):
                    raise ValueError('unsafe source archive')
            archive.extractall(root)
        if sha(source)!=FIXED_SHA or sha(root/'tests/test_studio_grpo_materialization.cpp')!=TEST_SHA or sha(root/'CMakeLists.txt')!=CMAKE_SHA:
            raise ValueError('frozen source/test/CMake differs')
        if sha(upload/'grpo_baseline_mtstudio.cpp')!=BASELINE_SHA: raise ValueError('baseline source differs')
        if sha(upload/'grpo_source.tgz') != ARCHIVE_SHA: raise ValueError('frozen archive differs')
        report['source_commit']=SOURCE_COMMIT
        original = source.read_bytes()
        _, hardware = command('hardware', ['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],30)
        if len(hardware.strip().splitlines())!=1 or not hardware.startswith('NVIDIA L4,'): raise ValueError('requires one actual L4')
        command('nvcc',['nvcc','--version'],30)
        command('configure',['cmake','-S',root,'-B',build,'-DCMAKE_BUILD_TYPE=Release','-DMICROTORCH_CUDA=ON',
                             '-DMICROTORCH_DEVCHECK=ON','-DMICROTORCH_BUILD_PYTHON=OFF','-DCMAKE_CUDA_ARCHITECTURES=89'],180)
        source.write_bytes((upload/'grpo_baseline_mtstudio.cpp').read_bytes())
        report['baseline_mtstudio_sha256']=sha(source)
        if report['baseline_mtstudio_sha256']!=BASELINE_SHA: raise ValueError('baseline phase source differs')
        command('baseline_build',['cmake','--build',build,'--target','test_studio_grpo_materialization','--parallel','4'],600)
        report['baseline_binary_sha256']=sha(build/'test_studio_grpo_materialization')
        report['baseline']={}
        for case in CASES:
            rc,text=command('baseline_'+case,[build/'test_studio_grpo_materialization','--require-cuda','--case',case],30,(1,))
            check_baseline(rc,text,case)
            report['baseline'][case]={'returncode':rc,'registered_numerical_failure':True}
        report['baseline_expected_numerical_failure']=True
        temporary=source.with_suffix('.restore');temporary.write_bytes(original);temporary.replace(source)
        report['fixed_mtstudio_sha256']=sha(source)
        if report['fixed_mtstudio_sha256']!=FIXED_SHA: raise ValueError('fixed phase source differs')
        command('fixed_build',['cmake','--build',build,'--target','test_studio_grpo_materialization','--parallel','4'],300)
        report['fixed_binary_sha256']=sha(build/'test_studio_grpo_materialization')
        report['fixed']={}
        for case in CASES:
            _,text=command('fixed_'+case,[build/'test_studio_grpo_materialization','--require-cuda','--case',case],30)
            proof=parse_proof(text,('pass','actual_cuda','devcheck_enabled','no_grad_context','cpu_reference','stale_producer','host_poison'))
            rows=[json.loads(x) for x in text.splitlines() if x.startswith('{')]
            producer={'actual_cuda':True,'stale_producer':True,'host_poison':True,'no_grad_context':True,'deferred_window':True}
            if producer not in rows: raise ValueError('fixed actual NoGrad/open-window producer proof missing')
            if proof.get('case')!=case:raise ValueError('fixed case inventory differs')
            required='sampler_ids_logp_rng' if case=='sample' else 'all_softmax_rows'
            if proof.get(required) is not True:raise ValueError('fixed specific reference proof missing')
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

def interrupted(signum, frame):
    raise InterruptedError('termination signal '+str(signum))

if __name__ == '__main__':
    signal.signal(signal.SIGTERM,interrupted)
    raise SystemExit(run(sys.argv[1]))
