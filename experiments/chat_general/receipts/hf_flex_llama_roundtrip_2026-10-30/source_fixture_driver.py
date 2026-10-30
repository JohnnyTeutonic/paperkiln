import pathlib,subprocess,json,sys,struct,os
repo=pathlib.Path('/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/microtorch');sys.path.insert(0,str(repo/'tools'))
import hf_export
import hf_export_verify
home=pathlib.Path('/mnt/c/ml_artifacts/hf_flex_llama_roundtrip_20261030')
checkpoint=home/'checkpoint.json'
def record(stage):
 p=checkpoint.with_suffix('.tmp');p.write_text(json.dumps({'stage':stage})+'\n');os.replace(p,checkpoint)
def invoke(main,args):
 old=sys.argv;sys.argv=args
 try:main()
 finally:sys.argv=old
for tied in (False,):
 name='tied' if tied else 'untied';p=home/name;run=p/'run';spec=p/'spec.json'
 events=run/'events.jsonl'
 done=events.exists() and any(json.loads(line).get('event')=='done' for line in events.read_text().splitlines())
 if not done:
  record(name+' training');subprocess.run(['/home/jonat/hf_flex_llama_clean_20261030/build/mtstudio','run',str(spec)],check=True)
 rows=[json.loads(line) for line in events.read_text().splitlines()];model=next(r for r in rows if r['event']=='model')
 assert model['family']=='flex' and model['norm']=='rmsnorm' and model['activation']=='swiglu' and model['position']=='rope'
 assert model.get('tie_head',False)==tied and model['rope_theta']==1234.0
 with (run/f'hf_flex_llama_{name}.safetensors').open('rb') as f:
  n=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(n))
 assert header.get('__metadata__') is None and ('head.weight' in header)==(not tied)
 record(name+' export');invoke(hf_export.main,['hf_export.py',str(run)])
 for i,prompt in enumerate(['once upon a time',"HELLO!! it's (a test)",'the little boy went home']):
  receipt=p/f'roundtrip_{i}.json'
  if receipt.exists():continue
  record(name+f' verify{i}')
  invoke(hf_export_verify.main,['hf_export_verify.py',str(run/'hf'),str(run),'--spec',str(spec),'--prompt',prompt,'--tokens','8','--mtstudio','/home/jonat/hf_flex_llama_clean_20261030/build/mtstudio','--receipt',str(receipt)])
record('complete');checkpoint.unlink()
