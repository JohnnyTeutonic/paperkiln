from pathlib import Path
import hashlib,json,os,subprocess,sys
repo=Path('/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/microtorch')
home=Path('/mnt/c/ml_artifacts/hf_flex_llama_roundtrip_20261030')
output=home/'evidence_replay_final';output.mkdir(exist_ok=True)
run=home/'untied/run'; spec=home/'untied/spec.json'
binary='/home/jonat/hf_flex_llama_clean_20261030/build/mtstudio'
assert hashlib.sha256(Path(binary).read_bytes()).hexdigest()=='fe1be5499e9592f4864377cbd6b292d7dd7c32d2dc9ef2e062c145a547914f00'
for i,prompt in enumerate(['once upon a time',"HELLO!! it's (a test)",'the little boy went home']):
 target=output/f'roundtrip_{i}.json'
 cmd=[sys.executable,str(repo/'tools/hf_export_verify.py'),str(run/'hf'),str(run),'--spec',str(spec),'--prompt',prompt,'--tokens','8','--mtstudio',binary,'--receipt',str(target)]
 if not target.exists():subprocess.run(cmd,check=True)
 subprocess.run([sys.executable,str(repo/'tools/hf_export_verify.py'),'--replay',str(target)],check=True)
 value=json.loads(target.read_text());old=json.loads((home/'untied'/f'roundtrip_{i}.json').read_text())
 for field in ('prompt_ids','continuation_ids','mtstudio_argmax','max_abs_logit_error'):
  assert value[field]==old[field],(i,field,value[field],old[field])
 print('HISTORICAL_IDS_AND_DELTA_MATCH',i,flush=True)
print('ALL_THREE_REPLAYED',flush=True)
