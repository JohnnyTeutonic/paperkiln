"""GNU/Linux CPU-only boundary proof; supplied library source must match consumers."""
from pathlib import Path
import argparse,hashlib,json,subprocess,os
p=argparse.ArgumentParser();p.add_argument('--library-build',type=Path,required=True);p.add_argument('--library-source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
R=Path(__file__).resolve().parents[3];O=a.out.resolve();O.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
consumers=['src/ops.cpp','src/autograd.cpp','src/device_cache.cpp','include/microtorch/autograd.hpp','include/microtorch/ops.hpp','include/microtorch/device_cache.hpp','include/microtorch/device.hpp','include/microtorch/primitives.hpp']
proof={f:{'tested_sha256':sha(R/f),'library_source_sha256':sha(a.library_source/f)} for f in consumers}
assert all(v['tested_sha256']==v['library_source_sha256'] for v in proof.values())
flags=['-std=c++17','-O3','-DNDEBUG','-ffp-contract=off','-fopenmp','-I'+str(R/'include'),'-I'+str(R/'tests'),'-I'+str(R/'third_party/transformer_cpp/include'),'-I'+str(R/'third_party/transformer_cpp/third_party')]
def run(cmd,name):
 with (O/name).open('w') as f:subprocess.run(cmd,check=True,stdout=f,stderr=subprocess.STDOUT)
run(['g++',*flags,'-DGEMM_FALLBACK_WRAP','-c',str(R/'tests/test_gemm_fallback_materialization.cpp'),'-o',str(O/'gate.o')],'compile_test.txt')
for v,src in [('baseline',R/'tests/gemm_fallback_baseline.cpp'),('fixed',R/'src/device.cpp')]:
 run(['g++',*flags,'-c',str(src),'-o',str(O/(v+'.o'))],'compile_'+v+'.txt')
 run(['g++','-fopenmp',str(O/'gate.o'),str(O/(v+'.o')),str(a.library_build/'libmicrotorch.a'),str(a.library_build/'transformer_cpp/libtransformer_core.a'),'-pthread','-Wl,--wrap=_ZN10microtorch6device11materializeERK6Matrix','-o',str(O/v)],'link_'+v+'.txt')
results=[]
for v in ['baseline','fixed']:
 for mode in ['fresh','synthetic']:
  proc=subprocess.run([str(O/v)]+(['--synthetic'] if mode=='synthetic' else []),text=True,capture_output=True,env=os.environ|{'OMP_NUM_THREADS':'1'})
  (O/(v+'_'+mode+'.txt')).write_text(proc.stdout+proc.stderr)
  rows=[json.loads(x.split('GEMM_FALLBACK_JSON ',1)[1]) for x in proc.stdout.splitlines() if x.startswith('GEMM_FALLBACK_JSON ')]
  assert len(rows)==5
  expected=v=='fixed' or mode=='fresh'
  assert (proc.returncode==0)==expected and all(row['pass']==expected for row in rows)
  results.append({'variant':v,'mode':mode,'returncode':proc.returncode,'rows':rows})
(O/'cpu_validation.json').write_text(json.dumps({'actual_cuda':False,'linked_source_proof':proof,'library_sha256':sha(a.library_build/'libmicrotorch.a'),'baseline_device_source_sha256':sha(R/'tests/gemm_fallback_baseline.cpp'),'fixed_device_source_sha256':sha(R/'src/device.cpp'),'test_source_sha256':sha(R/'tests/test_gemm_fallback_materialization.cpp'),'results':results},indent=2)+'\n')
print('20cells checked: baseline synthetic5FAIL,fixed synthetic5PASS,fresh10PASS; actual_cuda=false')
