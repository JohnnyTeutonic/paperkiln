import gzip, hashlib, json, os, re, subprocess, sys
from pathlib import Path
root=Path('/home/jonat/pk_publication_20261030')
build=Path('/home/jonat/pk_publication_build_20261030')
receipt=root/'docs/receipts/stack_integration_final_20261031'
receipt.mkdir(exist_ok=False)
files=['CMakeLists.txt','src/ops.cpp','src/cuda_ops.cu','src/conv1d_kernels.cuh','tests/test_add_bias_materialization.cpp','tests/test_layernorm_host.cpp','tests/test_layernorm_materialization.cpp','tests/test_conv1d_cuda.cpp','tools/mtstudio.cpp','tests/test_studio_grpo_materialization.cpp']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
bindings={f:sha(root/f) for f in files}
report={'pass':False,'source_commit':head,'source_sha256':bindings,'commands':[],'limits':'CPU integration only; actual CUDA and timing claims remain bound to separate receipts.'}
env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MICROTORCH_DEVICE':'cpu','MICROTORCH_DEVICE_OPS':'0','MICROTORCH_STEP_RESIDENCY':'0','MICROTORCH_DEFER_DOWNLOADS':'0','MTSTUDIO_BIN':str(build/'mtstudio'),'MICROTORCH_CPU_BUILD':str(build)}
steps=[('configure',['cmake','-S',str(root),'-B',str(build),'-DMICROTORCH_CUDA=OFF','-DMICROTORCH_DEVCHECK=ON','-DMICROTORCH_BUILD_PYTHON=OFF'],120),('build',['cmake','--build',str(build),'--parallel','2'],900),('ctest',['ctest','--test-dir',str(build),'--output-on-failure','-j','1'],300),('python',['python3','-m','pytest','tests/test_hf_export.py','tests/test_hf_export_evidence.py','tests/test_hf_export_publication.py','tests/test_hf_export_training_modifiers.py','tests/test_hf_tied_flex.py','tests/test_hf_exact_gelu.py','tests/test_hf_relu_flex.py','tests/test_evaluation_probes_cli.py','-q'],300),('grpo',['python3','-m','pytest','tests/test_mtstudio_grpo.py','-q'],300)]
try:
 for name,command,timeout in steps:
  print('START '+name,flush=True)
  with (receipt/(name+'.log')).open('xb') as log:
   result=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
  report['commands'].append({'name':name,'argv':command,'returncode':result.returncode,'log_sha256':sha(receipt/(name+'.log'))})
  if result.returncode:raise RuntimeError(name+' failed')
  print('PASS '+name,flush=True)
 raw=subprocess.check_output([str(build/'bench_layernorm_cpu'),'dump'],cwd=root,env=env,timeout=60)
 expected=gzip.decompress((root/'docs/receipts/cpu_layernorm_20261031/baseline_bits.json.gz').read_bytes())
 report['current_layernorm_bits']={'equal_to_frozen_baseline':raw==expected,'raw_sha256':hashlib.sha256(raw).hexdigest(),'frozen_raw_sha256':hashlib.sha256(expected).hexdigest(),'cases':len(json.loads(raw)['cases']),'models':len(json.loads(raw)['models'])}
 if raw!=expected:
  (receipt/'mismatching_layernorm_bits.json.gz').write_bytes(gzip.compress(raw,mtime=0))
  raise RuntimeError('current LayerNorm bit dump differs from frozen baseline')
 ctest=(receipt/'ctest.log').read_text();python=(receipt/'python.log').read_text()
 if not re.search(r'100% tests passed, 0 tests failed out of 57',ctest):raise RuntimeError('unexpected CTest inventory')
 if not re.search(r'302 passed, 1 skipped',python):raise RuntimeError('unexpected HF/probe inventory')
 if not re.search(r'6 passed',(receipt/'grpo.log').read_text()):raise RuntimeError('unexpected GRPO end-to-end inventory')
 report['ctest_passed']=57;report['python_passed']=302;report['python_skipped']=1;report['grpo_end_to_end_passed']=6
 if {f:sha(root/f) for f in files}!=bindings:raise RuntimeError('source drift during checks')
 report['binary_sha256']={name:sha(build/name) for name in ['mtstudio','test_add_bias_materialization','test_layernorm_host','test_layernorm_materialization','test_conv1d_cuda','test_studio_grpo_materialization','bench_layernorm_cpu']}
 report['pass']=True
except Exception as error:report['error']=str(error)
finally:
 temporary=receipt/'validation.partial';temporary.write_text(json.dumps(report,indent=2)+'\n');temporary.replace(receipt/'validation.json')
 print(json.dumps({'pass':report['pass'],'receipt':str(receipt),'error':report.get('error')}),flush=True)
sys.exit(0 if report['pass'] else 1)
