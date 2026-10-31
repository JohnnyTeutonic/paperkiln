"""Bounded actual L4 convolution gate. Transport completion never implies PASS."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile

SOURCE_COMMIT = '1ae3472e9fbd247cfb15ae60ca1899fc95ea0a9a'
ARCHIVE_SHA = 'd2d27b08d2e2886a93c8a027024247a4d6a72120868dc90ea7f026933421e644'
CRITICAL = {'CMakeLists.txt': '6b4bf87dc7ef3691a7e2487eeb1420eb8a286ec08e5dd2f25fdadfefa88aa455', 'include/microtorch/device_cache.hpp': 'e92273a45e095a0d60b65b4ce50f2346a6fa54101ed79071709a8aeac54b70e6', 'include/microtorch/ops.hpp': 'f48cad0682eff17302f5dfc9b274f9ef042207a52257b0fe3257a0f5ccaacd87', 'src/cuda_ops.cu': '9559249aae0f7698cb5ccb98acf1e030437e94bb19e3296eeca7709e9c4b89d7', 'src/conv1d_kernels.cuh': '056f3c40e0ed9558216156ea37c56a997cbe17392d22ef6e72a6d16f0738754e', 'src/device_cache.cpp': 'ddbadf767defdac314553b0256ef5e3a80481bce293225dc5497040843106b4b', 'src/ops.cpp': 'c11303411bcf217702635a06d6047db6e1c65779dad917da628ebbe28c9ac3f9', 'tests/test_conv1d_cuda.cpp': '9002d77bcdeba5e702b409515bb3019978733caa5f039b7d7d3f054921993833', 'tests/test_thirdparty_ops.cpp': '70944525d77fb52d1fd06230c3f7321b2d55eb88847619757822285e010fc2c2', 'studio/chat.html': 'bb9d2be154e274436d28f4e7fb758815ec3031071d5700a5d653659517abd4dc'}
DEADLINE = datetime.datetime(2026, 10, 31, 3, 0, tzinfo=datetime.timezone.utc)
PROOF_KEYS = ('pass', 'actual_cuda', 'devcheck_enabled', 'dense_binary64', 'finite_difference',
              'f32_order_bitwise', 'all_gradient_masks', 'poisoned_stale_inputs',
              'deferred_outputs', 'deferred_gradients', 'device_to_host_backward_fallback',
              'public_autograd_traversal', 'write_through')
LOGS = ('hardware', 'nvcc', 'configure', 'build', 'strict', 'thirdparty')

class GateError(ValueError):
    pass

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def parse_proof(text):
    proof = json.loads(text.strip().splitlines()[-1])
    if not isinstance(proof, dict) or not all(proof.get(k) is True for k in PROOF_KEYS):
        raise GateError('missing explicit actual CUDA/DEVCHECK/convolution proof')
    if type(proof.get('cases')) is not int or proof['cases'] != 120:
        raise GateError('strict convolution matrix must contain exactly 120 cases')
    return proof

def check_thirdparty(text):
    required = ('third-party ops (CUDA device ops)', '1. conv1d_causal (K 4, 2 sequences)',
                '4. mamba1 (d 16, Di 32, N 4, K 4)', 'all third-party op checks passed')
    if not all(fragment in text for fragment in required):
        raise GateError('missing actual CUDA selected third-party/Mamba regression proof')
    # This regression selects the real CUDA device, but selective_scan remains
    # a host fallback. It does not establish a fully device-resident Mamba path.
    return {'pass': True, 'cuda_device_selected': True, 'mamba_regression': True,
            'selective_scan_device_claim': False}

def safe_extract(archive_path, root):
    if root.exists() and any(root.iterdir()):
        raise GateError('source destination must be empty')
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            if member.name.startswith('/') or '..' in Path(member.name).parts or not (member.isfile() or member.isdir()):
                raise GateError('unsafe source archive')
        archive.extractall(root)

def run(out, upload=Path('/content/chat/upload'), root=Path('/content/conv1d_source'), build=Path('/content/conv1d_build')):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    report={'pass':False,'source_commit':SOURCE_COMMIT,'scope':'ordered conv1d_causal CUDA seam only',
            'commands':[],'errors':[],'selective_scan_device_claim':False,'speedup_claim':False}
    env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
         'MICROTORCH_DEVICE':'cuda','MICROTORCH_DEVICE_OPS':'1',
         'MICROTORCH_STEP_RESIDENCY':'0','MICROTORCH_DEFER_DOWNLOADS':'0'}
    for name in LOGS:(out/(name+'.log')).write_text('NOT_RUN\n')
    def command(name, argv, timeout):
        remaining=(DEADLINE-datetime.datetime.now(datetime.timezone.utc)).total_seconds()
        if remaining<=0:raise GateError('absolute deadline expired before '+name)
        with (out/(name+'.log')).open('w') as log:
            child=subprocess.Popen(list(map(str,argv)),cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:rc=child.wait(timeout=min(timeout,remaining))
            except (subprocess.TimeoutExpired,GateError):
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                raise GateError(name+' timed out/interrupted') from None
        report['commands'].append({'name':name,'argv':list(map(str,argv)),'returncode':rc})
        if rc!=0:raise GateError(name+' failed with exit '+str(rc))
        return (out/(name+'.log')).read_text(errors='replace')
    try:
        manifest=json.loads((upload/'conv1d_inputs.json').read_text());report['inputs']=manifest
        if manifest['source_commit']!=SOURCE_COMMIT:raise GateError('source pin differs')
        for filename,expected in manifest['files'].items():
            if sha(upload/filename)!=expected:raise GateError('upload hash differs: '+filename)
        if sha(upload/'conv1d_source.tgz')!=ARCHIVE_SHA:raise GateError('archive differs from frozen gate')
        safe_extract(upload/'conv1d_source.tgz',root)
        report['source_sha256']={name:sha(root/name) for name in CRITICAL}
        if report['source_sha256']!=CRITICAL:raise GateError('frozen source/test/kernel/CMake differs')
        hardware=command('hardware',['nvidia-smi','--query-gpu=name,driver_version,uuid,memory.total','--format=csv,noheader'],30)
        if len(hardware.strip().splitlines())!=1 or not hardware.startswith('NVIDIA L4,'):raise GateError('requires exactly one actual NVIDIA L4')
        command('nvcc',['nvcc','--version'],30)
        command('configure',['cmake','-S',root,'-B',build,'-DCMAKE_BUILD_TYPE=Release','-DMICROTORCH_CUDA=ON',
                             '-DMICROTORCH_DEVCHECK=ON','-DMICROTORCH_BUILD_PYTHON=OFF','-DCMAKE_CUDA_ARCHITECTURES=89'],300)
        command('build',['cmake','--build',build,'--target','test_conv1d_cuda','test_thirdparty_ops','--parallel','4'],1200)
        report['binary_sha256']={name:sha(build/name) for name in ('test_conv1d_cuda','test_thirdparty_ops')}
        report['strict']=parse_proof(command('strict',[build/'test_conv1d_cuda','--require-cuda'],180))
        report['thirdparty']=check_thirdparty(command('thirdparty',[build/'test_thirdparty_ops'],180))
        # Recheck source after every command, so a build cannot silently mutate it.
        if any(sha(root/name)!=expected for name,expected in CRITICAL.items()):raise GateError('source changed during gate')
        report['pass']=True
    except (OSError,ValueError,KeyError,IndexError,subprocess.SubprocessError,tarfile.TarError) as exc:
        report['errors'].append(str(exc))
    finally:
        report['logs']={name+'.log':{'sha256':sha(out/(name+'.log')),'bytes':(out/(name+'.log')).stat().st_size} for name in LOGS}
        temporary=out/'gate_summary.partial';temporary.write_text(json.dumps(report,indent=2)+'\n');temporary.replace(out/'gate_summary.json')
    print(json.dumps({'pass':report['pass'],'errors':report['errors']}))
    return 0

def interrupted(signum, frame):
    raise GateError('received termination signal '+str(signum))

if __name__=='__main__':
    signal.signal(signal.SIGTERM,interrupted)
    raise SystemExit(run(sys.argv[1]))
