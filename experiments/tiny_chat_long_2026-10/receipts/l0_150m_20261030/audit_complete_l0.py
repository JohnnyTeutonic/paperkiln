import pathlib,json,hashlib,subprocess,sys,math,datetime
repo=pathlib.Path('/home/jonat/pk_publication_20261030');art=pathlib.Path('/mnt/c/ml_artifacts/tiny_chat');folder=art/'l0_colab/l0_pick_e1_150m_L8/pick_e1_150m_L8';tmp=pathlib.Path('/mnt/c/Users/jonat/AppData/Local/Temp');pin='82c86c96bd0086ae47e35ab31370b5fb98ae28ed'
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
r=json.loads((folder/'l0.json').read_text());assert r['stage']=='L0' and r['smoke'] is False and r['model']['T']==512 and r['tag']=='pick_e1_150m_L8'
inputs={'weights':(art/'e1_150m/pick_e1_150m_L8/model.safetensors','02f8b00e2253272eded24341bd8bf93b11abcd6378ef693756a088e285c2f7fd'),'gguf':(art/'e1_150m/pick_e1_150m_L8/model.gguf','99944e664225984d93577620cb40ea8f4b74406dbbeb8b0cc586872bb31345c5'),'spec':(art/'e1_150m/pick_e1_150m_L8/spec.json','1506cca15af986cfabdb555e4da4ce3d7eccfbd4dbc065970c3ad2a17741ccec'),'vocab':(pathlib.Path('/mnt/c/ml_artifacts/capacity/vocab/vocab_llama3_bpe.gguf'),'0b880aa073a7e27c0770d5ecc5215b55d6d415080d20bd3dc7b169ecabadf198')}
for p,h in inputs.values():assert sha(p)==h,p
assert r['model']['safetensors_sha256']==inputs['weights'][1] and r['model']['gguf_sha256']==inputs['gguf'][1]
files=['tools/carry12_l0.py','tools/carry12_eval.py','tools/carry12_build.py','tools/carry5_eval.py','tools/chat_eval.py','tools/llama3_bpe.py','tools/colab_chat/tiny_chat_run.py','tools/colab_chat/carry12_l0_e1_150m_v1.json','experiments/tiny_chat_long_2026-10/STAGE_L0.md','corpus/eval/carry12_v1.jsonl']
source={}
for p in files:
 blob=subprocess.check_output(['git','-C',str(repo),'show',pin+':'+p]);source[p]=hashlib.sha256(blob).hexdigest()
assert r['registration_sha256']==source[files[-2]] and r['set']['sha256']==source[files[-1]] and r['set']['pinned'] is True
assert r['eval']['scorer']=='mtstudio' and r['eval']['weights_are_weights'] is True and r['eval']['every']==1 and r['eval']['spot']==20
argv=r['eval']['argv'];assert argv[argv.index('--max-new')+1]=='24' and argv[argv.index('--self-scope')+1]=='target' and '--no-near-self' not in argv
eng=r['descriptive']['gen_engine'];assert eng['engine']=='ember' and eng['spot']['pass'] is True and eng['spot']['items_identical']==20 and eng['spot']['replies_identical']==404
rows=[json.loads(x) for x in (folder/'eval/pick_e1_150m_L8/items.jsonl').read_text().splitlines()];gates=[v for x in rows for v in x['gate']];assert len(gates)==22956 and all(v is True for v in gates) and not any(x['server_errors'] for x in rows)
raw=subprocess.run([sys.executable,str(tmp/'codex_carry_raw_audit.py'),str(folder/'eval/pick_e1_150m_L8/items.jsonl')],capture_output=True,text=True,check=True);(tmp/'l0_raw_credit_audit.json').write_text(raw.stdout);raw=json.loads(raw.stdout)['runs'][0];assert raw['hl0']==r['hl0']['D'] and raw['hl0']['n']==1320
assert round(raw['twin_primary'],5)==r['validity_gate']['twin_primary'] and round(raw['context_twin_minus_noctx'],5)==r['validity_gate']['twin_minus_noctx_context'] and r['validity_gate']['pass'] is True
assert r['hl0']['verdict']=='SUPPORTED' and raw['hl0']['ci95'][1]<0 and raw['hl0']['mean']<=-.03
s=json.loads(pathlib.Path('/home/jonat/pk_tc_l0_150m_runner/state.json').read_text());assert s['campaign_home'] is True and all(v.get('home') for v in s['jobs'].values()) and not s['runpod'].get('pod') and not s['runpod'].get('volume')
sys.path.insert(0,str(repo/'tools/colab_chat'));import runpod_api as a;assert 'j73owoxlpwzpoy' not in {p['id'] for p in a.list_pods()}
records={str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file()}
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Single original150M pick; descriptive extension, not a new multi-seed confirmatory result or inference replay.','launch_commit':pin,'selected_registered_source_blobs_sha256':source,'model_inputs_sha256':{k:{'path':str(p),'sha256':h} for k,(p,h) in inputs.items()},'artifact_files_sha256':records,'raw_score_audit':raw,'generation':{'items':1452,'spot_items_identical':20,'spot_replies_identical':404,'full_argmax_gates_passed':22956,'server_errors':0},'hl0_recorded_verdict':r['hl0']['verdict'],'home':{'campaign_home':True,'all_jobs_home':True,'pod':None,'volume':None,'pod_absent_from_provider':True,'cumulative_usd':a.ledger_usd(s['runpod']['ledger']),'state_sha256':sha('/home/jonat/pk_tc_l0_150m_runner/state.json')},'incident':'Original partial evaluation was lost in the documented volume incident; this is the authorized unchanged-recipe recovery. Prior failure receipts remain unmodified.'}
(tmp/'l0_independent_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':'VERIFIED_HOME','credits':raw['raw_fc_credits_recomputed'],'hl0':raw['hl0'],'verdict':r['hl0']['verdict'],'home':out['home']}))
