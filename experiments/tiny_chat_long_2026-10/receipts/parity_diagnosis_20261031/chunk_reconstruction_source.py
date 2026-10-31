from pathlib import Path
import sys,json,hashlib
root=Path('/home/jonat/pk_hs1_parity_diagnosis_20261031');art=Path('/home/jonat/hs1_parity_diagnosis_20261031');sys.path.insert(0,str(root/'tools'));import carry12_eval as ce
items=[json.loads(x) for x in (root/'corpus/eval/carry12_v1.jsonl').read_text().splitlines()];affected=json.loads((art/'requests.json').read_text());inv=json.loads((art/'inventory.json').read_text());receipt=[]
for run,v in inv['runs'].items():
 targets={x['request']['key'] for x in affected if x['run']==run};gens={x['id']:x for x in map(json.loads,Path(v['gen']).read_text().splitlines())};tok=ce.ce.Tok('/mnt/c/ml_artifacts/capacity/vocab/vocab_llama3_bpe.gguf');tok.state=ce.ce.check_state_spec(v['original_spec'],v['gguf']);n=0;chunk=[];selected=[]
 def finish(part,index):
  present=targets.intersection(x[0] for x in part)
  if not present:return
  path=art/f'{run}.chunk{index:04d}.requests.jsonl';data=''.join(json.dumps({'key':k,'prompt_ids':p,'continuation_ids':c})+'\n' for k,p,c in part);path.write_text(data)
  selected.append({'chunk_index':index,'requests':len(part),'target_keys':sorted(present),'file':str(path),'sha256':hashlib.sha256(data.encode()).hexdigest()})
 for it in items:
  for request in ce.requests([it],tok,{it['id']:gens[it['id']]}):
   chunk.append(request);n+=1
   if len(chunk)==256:finish(chunk,(n-1)//256);chunk=[]
 if chunk:finish(chunk,(n-1)//256)
 assert targets=={k for x in selected for k in x['target_keys']}
 # Exact selected gate rows must equal independently reconstructed single-item requests.
 for x in selected:
  rows={q['key']:q for q in map(json.loads,Path(x['file']).read_text().splitlines())}
  for a in affected:
   if a['run']==run and a['request']['key'] in rows:assert rows[a['request']['key']]==a['request']
 receipt.append({'run':run,'all_requests':n,'chunk_size':256,'score_batch_tokens':16384,'state_unpacked':run.startswith('c12_Sgist'),'selected_chunks':selected});print(run,n,len(selected),flush=True)
(art/'chunk_inventory.json').write_text(json.dumps(receipt,indent=2)+'\n')
