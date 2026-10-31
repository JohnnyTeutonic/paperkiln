from pathlib import Path
import subprocess,json,hashlib,os
repo=Path('/home/jonat/pk_host_helper_audit_20261031');out=Path('/home/jonat/host_helper_controls_20261031');out.mkdir(exist_ok=True);build=Path('/home/jonat/pk_publication_build_20261030');h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();libs=[build/'libmicrotorch.a',build/'transformer_cpp/libtransformer_core.a'];before={str(p):h(p) for p in libs};common=['c++','-std=c++17','-O0','-DMICROTORCH_DEVCHECK','-ffp-contract=off','-fopenmp','-I'+str(repo/'include'),'-I'+str(repo/'third_party/transformer_cpp/include'),'-I'+str(repo/'third_party/transformer_cpp/third_party'),'-I'+str(build/'_deps/json-src/include')]
# Compile audited ops directly into executable so archive's ops.o is not used.
subprocess.run(common+['-c',str(repo/'src/ops.cpp'),'-o',str(out/'audited_ops.o')],check=True,stdout=(out/'compile.log').open('w'),stderr=subprocess.STDOUT)
results={}
for name in ['test_inplace','test_expr']:
 cmd=common+[str(repo/'tests'/(name+'.cpp')),str(out/'audited_ops.o'),*[str(p) for p in libs],'-o',str(out/name)]
 with (out/(name+'.build.log')).open('w') as log:subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
 with (out/(name+'.log')).open('w') as log:p=subprocess.run([str(out/name)],stdout=log,stderr=subprocess.STDOUT,timeout=120)
 if p.returncode:raise RuntimeError(name+' failed')
 results[name]={'pass':True,'executable_sha256':h(out/name),'log_sha256':h(out/(name+'.log'))}
assert before=={str(p):h(p) for p in libs},'libraries changed during control build'
result={'pass':True,'kind':'CPU_CONTROLS_NOT_CUDA_AUTHORITY_PROOF','source_commit':'42823c8191d6f4188ffebf8aa565565d56de9fd2','audited_ops_compiled_directly':True,'support_library_sha256':before,'results':results,'source_sha256':{str(p.relative_to(repo)):h(p) for p in [repo/'src/ops.cpp',repo/'src/cuda_resident.cu',repo/'src/autograd.cpp',repo/'include/microtorch/ops.hpp',repo/'include/microtorch/device_cache.hpp',repo/'tests/test_inplace.cpp',repo/'tests/test_expr.cpp']},'paid_calls':0,'local_gpu_calls':0};(out/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
