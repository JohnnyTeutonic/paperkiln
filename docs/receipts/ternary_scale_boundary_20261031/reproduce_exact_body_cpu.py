import hashlib,json,os,pathlib,subprocess
root=pathlib.Path('/home/jonat/pk_host_reader_audit_20261031b')
out=pathlib.Path('/home/jonat/ternary_scale_boundary_proof_20261031');out.mkdir(exist_ok=True)
base='5f489fdd7c5c05eb16434b4f6c96e84252d28dee'
old=subprocess.check_output(['git','show',base+':src/ops.cpp'],cwd=root).decode()
fixed=(root/'src/ops.cpp').read_text()
receipt={'base':base,'validation':'exact production function bodies only; full-object/CUDA validation pending','cuda_executed':False,'runs':{},'hashes':{}}
lib=pathlib.Path('/home/jonat/pk_publication_build_20261030/libmicrotorch.a')
core=pathlib.Path('/home/jonat/pk_publication_build_20261030/transformer_cpp/libtransformer_core.a')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for p in [lib,core,root/'tests/test_ternary_scale_boundary.cpp']:receipt['hashes'][str(p)]=sha(p)
for mode,src in [('old',old),('fixed',fixed)]:
 start=src.index('float ternary_scale(const Matrix& w) {');end=src.index('\n}',start)+2
 body=src[start:end]
 source=out/(mode+'.cpp');source.write_text('#include <cmath>\n#include <algorithm>\n#include "microtorch/device.hpp"\n#include "microtorch/ops.hpp"\nnamespace microtorch::ops {\n'+body+'\n}\n')
 receipt['hashes'][mode+'_full_ops']=hashlib.sha256(src.encode()).hexdigest();receipt['hashes'][str(source)]=sha(source)
 for emulated in [False,True]:
  label=mode+('_emulated' if emulated else '_fresh')
  exe=out/label
  cmd=['c++','-std=c++17','-O0','-ffp-contract=off','-DMICROTORCH_DEVCHECK','-Dternary_scale=boundary_ternary_scale','-I'+str(root/'include'),'-I'+str(root/'third_party/transformer_cpp/include'),'-I'+str(root/'tests'),str(source),str(root/'tests/test_ternary_scale_boundary.cpp'),str(lib),str(core),'-fopenmp','-o',str(exe)]
  if emulated:cmd+=['-DTERNARY_BOUNDARY_WRAP=1','-Wl,--wrap=_ZN10microtorch6device11materializeERK6Matrix']
  c=subprocess.run(cmd,capture_output=True,text=True);(out/(label+'.compile.log')).write_text(c.stdout+c.stderr)
  if c.returncode:raise RuntimeError(c.stderr)
  r=subprocess.run(['stdbuf','-oL',str(exe)],capture_output=True,text=True)
  (out/(label+'.log')).write_text(r.stdout+r.stderr)
  receipt['runs'][label]={'command':cmd,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
  if (mode=='old' and emulated):
   if r.returncode==0 or 'actual=100000' not in r.stdout:raise RuntimeError('old boundary failure absent')
  elif r.returncode:raise RuntimeError(label+' failed: '+r.stdout+r.stderr)
receipt['proof_script_sha256']=sha(pathlib.Path(__file__))
(root/'docs/receipts/ternary_scale_boundary_20261031/exact_body_cpu_proof.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:{'returncode':v['returncode'],'stdout':v['stdout']} for k,v in receipt['runs'].items()},indent=2))
