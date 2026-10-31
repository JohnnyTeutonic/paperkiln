"""One-attempt source-bound VM payload. No provider/launch/config APIs."""
import argparse,hashlib,json,os,signal,subprocess,tarfile,threading,time,traceback,zipfile
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def main():
    a=argparse.ArgumentParser();a.add_argument('--upload',type=Path,required=True);a.add_argument('--manifest-sha256',required=True);a.add_argument('--out',type=Path,required=True);a=a.parse_args();out=a.out;out.mkdir(exist_ok=True,parents=True)
    if any(out.iterdir()):raise ValueError('preserve existing evidence')
    src=out.parent/(out.name+'_source');build=out.parent/(out.name+'_build');summary={'status':'ERROR_OR_INCOMPLETE','commands':[]};deadline=time.monotonic()+1800;done=threading.Event();lock=threading.Lock()
    def pack():
        with lock:
            tmp=out/'summary.json.tmp';tmp.write_text(json.dumps(summary,indent=2)+'\n');tmp.replace(out/'summary.json');tmp=out/'out.zip.tmp'
            with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,strict_timestamps=False) as z:
                for p in sorted(out.rglob('*')):
                    if p.is_file() and p.name not in ['out.zip','out.zip.tmp','summary.json.tmp']:z.write(p,p.relative_to(out))
            tmp.replace(out/'out.zip')
    def periodic():
        while not done.wait(120):pack()
    def command(name,cmd,timeout):
        left=deadline-time.monotonic()
        if left<=0:raise TimeoutError('science deadline')
        with (out/(name+'.log')).open('w') as f:
            p=subprocess.Popen(list(map(str,cmd)),stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
            try:p.wait(timeout=min(timeout,left))
            except BaseException:
                if p.poll() is None:
                    os.killpg(p.pid,signal.SIGTERM)
                    try:p.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                raise
        summary['commands'].append({'name':name,'returncode':p.returncode})
        if p.returncode:raise ValueError(name+' failed')
    def stop(sig,frame):raise TimeoutError('terminated '+str(sig))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);thread=threading.Thread(target=periodic,daemon=True);thread.start()
    try:
        if sha(a.upload/'inputs.json')!=a.manifest_sha256:raise ValueError('input manifest pin')
        m=json.loads((a.upload/'inputs.json').read_text())
        if m['limits']!={'max_usd':1,'max_hours':1,'science_seconds':1800,'gpu':'NVIDIA L4','gpus':1}:raise ValueError('resource scope')
        for name,h in m['files'].items():
            if Path(name).name!=name or sha(a.upload/name)!=h:raise ValueError('flat input/hash '+name)
        src.mkdir(exist_ok=False)
        with tarfile.open(a.upload/'source.tgz') as t:
            for x in t.getmembers():
                if Path(x.name).is_absolute() or '..' in Path(x.name).parts or not (x.isfile() or x.isdir()):raise ValueError('unsafe archive')
            t.extractall(src)
        actual={str(p.relative_to(src)) for p in src.rglob('*') if p.is_file()}
        if actual!=set(m['source_files']):raise ValueError('source inventory')
        for name,h in m['source_files'].items():
            if sha(src/name)!=h:raise ValueError('source pin '+name)
        if sha(src/'tools/flex_cli_lifecycle_vm.py')!=sha(a.upload/'flex_cli_lifecycle_vm.py') or sha(__file__)!=m['files']['flex_cli_lifecycle_vm.py']:raise ValueError('executing/source VM mismatch')
        if sha(src/'docs/receipts/flex_cli_lifecycle_20261101/protocol.json')!=sha(a.upload/'protocol.json'):raise ValueError('source/upload protocol mismatch')
        (out/'inputs.json').write_bytes((a.upload/'inputs.json').read_bytes())
        command('hardware',['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],30)
        lines=(out/'hardware.log').read_text().strip().splitlines()
        if len(lines)!=1 or not lines[0].startswith('NVIDIA L4,'):raise ValueError('single L4 required')
        command('configure',['cmake','-S',src,'-B',build,'-DCMAKE_BUILD_TYPE=Release','-DMICROTORCH_CUDA=ON','-DMICROTORCH_DEVCHECK=ON','-DMICROTORCH_BUILD_PYTHON=OFF','-DCMAKE_CUDA_ARCHITECTURES=89'],300)
        command('build',['cmake','--build',build,'--target','mtstudio','flex_precision_gate','--parallel','2'],1000)
        command('python_dependencies',['python3','-m','pip','install','numpy==1.26.4','pytest==8.3.5'],120)
        (out/'binaries.json').write_text(json.dumps({name:sha(build/name) for name in ['mtstudio','flex_precision_gate']},indent=2)+'\n')
        command('actual_cli',['python3',src/'tools/flex_cli_lifecycle.py','--binary',build/'mtstudio','--model-gate',build/'flex_precision_gate','--require-cuda','--out',out/'lifecycle'],300)
        command('independent_raw_replay',['python3',src/'tools/replay_flex_cli_lifecycle.py',out/'lifecycle','--require-cuda'],120)
        for name,h in m['source_files'].items():
            if sha(src/name)!=h:raise ValueError('final source drift')
        summary.update(status='PASS',complete=True,actual_cuda=True,source_commit=m['source_commit'])
    except BaseException as e:
        summary.update(status='ERROR_OR_INCOMPLETE',complete=False,error=repr(e));(out/'exception.txt').write_text(traceback.format_exc())
    finally:done.set();thread.join(timeout=5);pack()
    return 0 if summary['status']=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())
