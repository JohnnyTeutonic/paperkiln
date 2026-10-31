from pathlib import Path
import sys,json,hashlib,subprocess
root=Path('/home/jonat/pk_hs1_parity_diagnosis_20261031');out=Path('/home/jonat/hs1_parity_diagnosis_20261031');out.mkdir(exist_ok=False)
sys.path.insert(0,str(root/'tools'));import carry12_eval as ce
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
audit=root/'experiments/tiny_chat_long_2026-10/receipts/hs1_five_seed_20261030/history_argmax_audit.json';affected=json.loads(audit.read_text())['nonidentical'];corpus=root/'corpus/eval/carry12_v1.jsonl';items={x['id']:x for x in map(json.loads,corpus.read_text().splitlines())};sources={str(audit):sha(audit),str(corpus):sha(corpus)}
for rel in ['tools/carry12_eval.py','tools/carry5_eval.py','tools/chat_eval.py','tools/state_tokens.py','tools/llama3_bpe.py']:
 p=root/rel;sources[rel]=sha(p)
 old=subprocess.check_output(['git','-C',str(root),'show','3dc818b2:'+rel]);assert old==p.read_bytes(),rel+' differs historical'
rows=[]; runs={}
for x in affected:
 run=f"c12_{x['cell']}_r{x['seed']}";base=Path('/mnt/c/ml_artifacts/tiny_chat')/run/post if False else Path('/mnt/c/ml_artifacts/tiny_chat')/run/f'post_{run}'
 genp=base/'carry12/m/gen.jsonl';gens={y['id']:y for y in map(json.loads,genp.read_text().splitlines())};it=items[x['id']];name=f"tc_sft_{x['cell']}_r{x['seed']}";gguf=base/'sft'/(name+'.gguf');weight=base/'sft'/(name+'.safetensors');specp=base/'sft_spec.json';spec=json.loads(specp.read_text());tok=ce.ce.Tok(str(gguf));tok.state=ce.ce.check_state_spec(str(specp),str(gguf));requests=ce.requests([it],tok,{x['id']:gens[x['id']]});key=f"{x['id']}|gate_self|{x['gate_turn']}";matching=[r for r in requests if r[0]==key];assert len(matching)==1
 _,p,c=matching[0];row={'key':key,'prompt_ids':p,'continuation_ids':c};rows.append({'run':run,'request':row,'source_mismatch':x})
 if run not in runs:
  local=json.loads(json.dumps(spec));local['out_dir']=str(base/'sft');local['data']['vocab']='/mnt/c/ml_artifacts/capacity/vocab/vocab_llama3_bpe.gguf';(out/(run+'.spec.json')).write_text(json.dumps(local,indent=2)+'\n');runs[run]={'spec':str(out/(run+'.spec.json')),'original_spec':str(specp),'gen':str(genp),'gguf':str(gguf),'weights':str(weight),'sources_sha256':{str(q):sha(q) for q in [specp,genp,gguf,weight,base/'gen_engine.json']}}
for run in runs:
 (out/(run+'.requests.jsonl')).write_text(''.join(json.dumps(x['request'])+'\n' for x in rows if x['run']==run))
(out/'requests.json').write_text(json.dumps(rows,indent=2)+'\n');(out/'inventory.json').write_text(json.dumps({'sources_sha256':sources,'runs':runs,'missing_historical_score_chunks':True,'no_history_or_result_changes':True},indent=2)+'\n');print([(x['run'],len(x['request']['prompt_ids']),len(x['request']['continuation_ids'])) for x in rows])
