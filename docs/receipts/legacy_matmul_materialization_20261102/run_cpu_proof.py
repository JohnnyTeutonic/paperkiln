from pathlib import Path
import hashlib,json,subprocess,os
r=Path('/home/jonat/codex_sfull_checkpoint_repair_20261101')
o=Path('/home/jonat/legacy_matmul_cpu_proof_20261102');o.mkdir(exist_ok=True)
assert not (o/'validation.json').exists(), 'Completed proof is immutable'
lib=Path('/home/jonat/pk_dpo_checkpoint_build_20261101')
cached=Path('/home/jonat/pk_dpo_checkpoint_20261101')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(lib/'libmicrotorch.a')=='bbaf128454147357574c1f623b0b909c28536be660b3abe2041eb360f25d67f8'
proof={}
for p in [*r.glob('src/*.cpp'),*r.glob('include/microtorch/*.hpp')]:
 rel=p.relative_to(r)
 if str(rel) in ['src/device.cpp','include/microtorch/dpo_startup_phases.hpp']:continue
 pair={'current':sha(p),'cached_source':sha(cached/rel)}
 assert pair['current']==pair['cached_source'],str(rel)
 proof[str(rel)]=pair
flags=['-std=c++17','-O3','-DNDEBUG','-ffp-contract=off','-fopenmp','-I'+str(r/'include'),'-I'+str(r/'tests'),'-I'+str(r/'third_party/transformer_cpp/include'),'-I'+str(r/'third_party/transformer_cpp/third_party'),'-I/home/jonat/pk_publication_build_20261030/_deps/json-src/include']
commands=[]
def run(cmd,name):
 commands.append(cmd)
 with (o/name).open('w') as f:subprocess.run(cmd,check=True,stdout=f,stderr=subprocess.STDOUT)
run(['g++',*flags,'-DLEGACY_MATMUL_WRAP','-c',str(r/'tests/test_legacy_matmul_materialization.cpp'),'-o',str(o/'test.o')],'compile_test.txt')
for name,src in [('old',r/'docs/receipts/legacy_matmul_materialization_20261102/baseline_device.cpp'),('fixed',r/'src/device.cpp')]:
 run(['g++',*flags,'-c',str(src),'-o',str(o/(name+'.o'))],'compile_'+name+'.txt')
 run(['g++','-fopenmp',str(o/'test.o'),str(o/(name+'.o')),str(lib/'libmicrotorch.a'),str(lib/'transformer_cpp/libtransformer_core.a'),'-pthread','-Wl,--wrap=_ZN10microtorch6device11materializeERK6Matrix','-o',str(o/name)],'link_'+name+'.txt')
results=[]
for name in ['old','fixed']:
 for mode in ['fresh','synthetic']:
  proc=subprocess.run([str(o/name)]+(['--synthetic'] if mode=='synthetic' else []),capture_output=True,text=True,env=os.environ|{'OMP_NUM_THREADS':'1'})
  (o/(name+'_'+mode+'.txt')).write_text(proc.stdout+proc.stderr)
  rows=[json.loads(x.split('LEGACY_MATMUL_JSON ',1)[1]) for x in proc.stdout.splitlines() if x.startswith('LEGACY_MATMUL_JSON ')]
  assert len(rows)==5
  wanted=name=='fixed' or mode=='fresh'
  assert proc.returncode==(0 if wanted else 1)
  assert all(row['pass'] is wanted and row['actual_cuda'] is False and row['synthetic_authority'] is (mode=='synthetic') for row in rows)
  if not wanted:assert all(row['error']=='independent numerical reference differs' for row in rows)
  results.append({'variant':name,'mode':mode,'returncode':proc.returncode,'rows':rows})
# Existing quantisation/LoRA and Muon consumers, linked to the repaired primitive.
for name in ['test_lora_quant','test_muon']:
 p=r/'tests'/(name+'.cpp')
 if p.exists():
  run(['g++',*flags,str(p),str(o/'fixed.o'),str(lib/'libmicrotorch.a'),str(lib/'transformer_cpp/libtransformer_core.a'),'-pthread','-o',str(o/name)],'link_'+name+'.txt')
  run([str(o/name)],name+'.txt')
report={'kind':'LEGACY_MATMUL_CPU_AUTHORITY_PROOF','actual_cuda':False,'source_sha256':{str(p.relative_to(r)):sha(p) for p in [r/'src/device.cpp',r/'tests/test_legacy_matmul_materialization.cpp',r/'docs/receipts/legacy_matmul_materialization_20261102/baseline_device.cpp']},'cached_library_sha256':sha(lib/'libmicrotorch.a'),'cached_source_proof':proof,'commands':commands,'results':results,'limits':'Synthetic CPU authority emulation; no real CUDA producer/transfer, no default historical training invalidity or exhaustive audit claim.'}
(o/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
print('20 cells verified: old synthetic five numerical failures, fixed synthetic five passes, ten fresh passes; existing consumers pass; actual_cuda=false')
