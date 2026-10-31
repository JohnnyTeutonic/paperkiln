from pathlib import Path
import os,json,subprocess,time,hashlib
art=Path('/home/jonat/hs1_parity_diagnosis_20261031');inventory=json.loads((art/'inventory.json').read_text());binary=Path('/home/jonat/pk_publication_build_20261030/mtstudio');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();env={**os.environ,'MICROTORCH_DEVICE':'cpu','MICROTORCH_DEVICE_OPS':'0','MICROTORCH_STEP_RESIDENCY':'0','MICROTORCH_DEFER_DOWNLOADS':'0','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
p=art/'cpu_diagnostic_progress.json';state=json.loads(p.read_text()) if p.exists() else {'binary':str(binary),'binary_sha256':sha(binary),'source_commit':'16654d9656539ceeb8a7dbe8ec63e0c2255910ec','scope':'New CPU diagnostic; not recovered original GPU logits','jobs':{}}
assert sha(binary)==state['binary_sha256']
for run,v in inventory['runs'].items():
 for mode in ['full','kv']:
  key=run+'_'+mode
  if key in state['jobs']:continue
  output=art/(key+'.scores.jsonl');logits=art/(key+'.logits.f32');log=art/(key+'.log');assert not output.exists() and not logits.exists()
  cmd=[str(binary),'score',v['spec'],'--in',str(art/(run+'.requests.jsonl')),'--out',str(output),'--logits',str(logits)]+(['--kv-cache'] if mode=='kv' else [])
  start=time.time()
  with log.open('xb') as f:r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=600)
  result={'command':cmd,'returncode':r.returncode,'seconds':time.time()-start,'log_sha256':sha(log)}
  if output.exists():result['scores_sha256']=sha(output)
  if logits.exists():result['logits_sha256']=sha(logits)
  state['jobs'][key]=result;q=p.with_suffix('.tmp');q.write_text(json.dumps(state,indent=2)+'\n');q.replace(p);print(key,result,flush=True)
  if r.returncode:raise RuntimeError(key+' failed, preserve output')
assert sha(binary)==state['binary_sha256']
