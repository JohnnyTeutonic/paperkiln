from pathlib import Path
import hashlib,importlib.util,json,subprocess,sys,tempfile,concurrent.futures
sys.dont_write_bytecode=True
source=Path('/mnt/c/ml_artifacts/kvq/stage-e9-gate-v2-20261101/e9_runpod_job.py');spec=importlib.util.spec_from_file_location('v2wrapper',source);w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
results=[]
def snapshot(root):return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
for case in ['negative_science','positive_science','partial_science','negative_jobout','GATE_DONE','GATE_FAILED','PHASES','gate_vm.log','gate_out.zip','gate_out_123.zip','existing_claim']:
 with tempfile.TemporaryDirectory() as tmp:
  base=Path(tmp);content=base/'content';content.mkdir();job=base/'job';job.mkdir();out=content/'out';out.mkdir()
  p=out/'gate_summary.json' if case in ['negative_science','positive_science'] else out/'partial.json' if case=='partial_science' else job/'gate_summary.json' if case=='negative_jobout' else content/('.e9_gate_attempt.json' if case=='existing_claim' else case)
  p.write_bytes(b'{"original":"preserve-exactly"}\n');before=snapshot(base)
  try:w.run(job,content)
  except (ValueError,OSError):pass
  else:raise AssertionError('prior evidence accepted '+case)
  assert snapshot(base)==before;results.append({'case':case,'refused_before_write':True,'all_prior_bytes_preserved':True})
for setup in ['missing_staging','stub_scientific_negative','stub_scientific_positive']:
 with tempfile.TemporaryDirectory() as tmp:
  base=Path(tmp);content=base/'content';content.mkdir();job=base/'job';job.mkdir() # exact jobs_vm empty-dir contract
  if setup!='missing_staging':
   upload=content/'chat/upload';upload.mkdir(parents=True);manifest={'formats':['f16'],'deadline_minutes':95,'phases':['g1','g2','g3','g4','tfd_qwen3','tfd_qwen25'],'baseline_ember_commit':'3ada969788f4d5ea65f5b81b29c22c30b7f25c97','files':{}}
   for name in w.FILES:
    data=(manifest['baseline_ember_commit']+'\n').encode() if name=='HEAD' else b'fixture';(upload/name).write_bytes(data);manifest['files'][name]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
   (upload/'STAGING.json').write_text(json.dumps(manifest))
  positive=setup=='stub_scientific_positive';calls=[]
  def stub(cmd,**kwargs):
   calls.append(cmd);(content/'out/gate_summary.json').write_text(json.dumps({'pass':positive,'errors':[]}));(content/'gate_vm.log').write_text('stub only: no GPU or model execution\n')
   if positive:(content/'GATE_DONE').touch()
   return subprocess.CompletedProcess(cmd,0 if positive else 1)
  w.subprocess.run=stub
  rc=w.run(job,content);execution=json.loads((job/'execution.json').read_text());assert rc==0 and execution['gate_pass'] is positive and (job/'out.zip').is_file()
  assert (content/'.e9_gate_attempt.json').is_file();before=snapshot(base)
  try:w.run(job,content)
  except (ValueError,OSError):pass
  else:raise AssertionError('second attempt permitted')
  assert snapshot(base)==before
  assert len(calls)==(0 if setup=='missing_staging' else 1)
  results.append({'case':setup,'transport_rc':rc,'gate_pass':execution['gate_pass'],'second_attempt_refused_without_changes':True,'VM_invocations_are_stubs':True})
# Exclusive claim, including two simultaneous starts on an otherwise empty content root.
with tempfile.TemporaryDirectory() as tmp:
 content=Path(tmp);a=content/'job_a';b=content/'job_b'
 def claim(path):
  try:w.claim_attempt(path,content);return 'claimed'
  except FileExistsError:return 'refused'
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:states=list(pool.map(claim,[a,b]))
 assert sorted(states)==['claimed','refused'];results.append({'case':'simultaneous_attempt_claim','exactly_one_winner':True})
report={'status':'PASS','wrapper_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'cases':results,'case_count':len(results),'jobs_vm_source_contract':'manager only creates empty jobout; pid/rc/STATUS/trainlog are outside jobout (jobs_vm.py command branch)','provider_GPU_model_actions':0}
out=Path('/home/jonat/e9_quality_hardening_20261101/wrapper_negative_tests.json');out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':'PASS','cases':len(results),'wrapper_sha256':report['wrapper_sha256']},indent=2))
