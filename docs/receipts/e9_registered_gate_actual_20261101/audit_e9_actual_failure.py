from pathlib import Path
import hashlib,json,stat,tarfile,zipfile,subprocess,os
home=Path('/mnt/c/ml_artifacts/kvq/ember-e9-gate-20261101/e9_gate');out=Path('/home/jonat/e9_actual_failure_audit_20261101');out.mkdir(exist_ok=False)
stage=Path('/mnt/c/ml_artifacts/kvq/stage-e9-gate-v3-20261101')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
science=out/'science';science.mkdir();inventory=[]
with zipfile.ZipFile(home/'out.zip') as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
 for i in z.infolist():
  p=Path(i.filename);assert not p.is_absolute() and '..' not in p.parts and not stat.S_ISLNK(i.external_attr>>16)
  if i.is_dir():continue
  data=z.read(i);dest=science/p;dest.parent.mkdir(exist_ok=True,parents=True);dest.write_bytes(data);inventory.append({'path':i.filename,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
assert sha(science/'STAGING.json')==sha(stage/'STAGING.json')=='17b87afcd732e357949861cfe9c701fc566b4fffc27ae40522e7335557dafddf'
for name in ['execution.json','gate_summary.json']:assert (science/name).read_bytes()==(home/name).read_bytes()
m=json.loads((science/'STAGING.json').read_text());assert sha(stage/'ember.tgz')==m['files']['ember.tgz']['sha256']=='731f51f8ed21872581ee4fe08886547bd496b46766d3560f613675386dad8cd7'
extract=['ember/tools/e9_gate_check.py','ember/tools/e9_quality.cpp']+[f'ember/experiments/kv_quant_r2_2026-10-28/receipts/colab_l4_eval/eval_{model}/tf_prefill.json' for model in ['qwen3','qwen25']]
source_pins={}
with tarfile.open(stage/'ember.tgz') as t:
 for name in extract:
  data=t.extractfile(name).read();p=out/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);source_pins[name]=hashlib.sha256(data).hexdigest()
checker=out/'ember/tools/e9_gate_check.py'
cmd=['python3',str(checker),'summary',str(science),'--phases',','.join(m['phases']),'--formats','f16']
p=subprocess.run(cmd,capture_output=True,text=True,env=os.environ|{'PYTHONDONTWRITEBYTECODE':'1'});(out/'frozen_checker.stdout').write_text(p.stdout);(out/'frozen_checker.stderr').write_text(p.stderr);assert p.returncode==1
replay=json.loads(p.stdout);(out/'frozen_checker_result.json').write_text(json.dumps(replay,indent=2)+'\n')
returned=json.loads((home/'gate_summary.json').read_text());execution=json.loads((home/'execution.json').read_text())
assert replay['pass'] is False and returned['pass'] is False and execution=={'gate_pass':False,'process_exit_code':1,'complete_marker':False}
assert replay['teacher_forcing_comparison']==returned['teacher_forcing_comparison'] and replay['errors']==['qwen3: decode metric exceeds corresponding prefill observation','qwen25: decode metric exceeds corresponding prefill observation']
violations=[];counts={k:{'passed':0,'failed':0} for k in ['max_rel','mean_rel','ppl_rel_diff']}
for model,rows in replay['teacher_forcing_comparison'].items():
 for row in rows:
  for metric in counts:
   x=row[metric];counts[metric]['passed' if x['within_prefill_observed'] else 'failed']+=1
   if not x['within_prefill_observed']:
    a,b=abs(x['decode']),abs(x['prefill']);violations.append({'model':model,'input':row['input'],'context':row['context'],'metric':metric,'decode':x['decode'],'stored_prefill':x['prefill'],'absolute_metric_excess':a-b,'absolute_ratio':a/b if b else None})
g3={};g4={}
for model in ['qwen3','qwen25']:
 g3[model]={fmt:json.loads((science/'g3'/f'{model}_{fmt}.json').read_text()) for fmt in ['f32','f16']}
 g4[model]=json.loads((science/'g4'/f'{model}.json').read_text())
 for fmt,data in g3[model].items():assert data['pass'] is True and data['on_gpu'] is True and data['top1_agree']=='288/288' and data['greedy_identical']=='12/12'
 assert g4[model]['pass'] is True and {r['i'] for r in g4[model]['rows']}==set(range(12))
checks={str(p.relative_to(science)):json.loads(p.read_text()) for folder in ['g1','g2'] for p in (science/folder).glob('*.check.json')}
labels=(science/'gates.txt').read_text().splitlines();bytes_labels=[s for s in labels if s.startswith(('g1 ','g2 '))];assert len(bytes_labels)==14 and all(' IDENTICAL ' in s for s in bytes_labels)
# Validate retained full dump hashes for each actual compared pair, without claiming removed raw bins replayable.
for model in ['qwen3','qwen25']:
 for det in [0,1]:
  for kind in ['short','long']:
   a=checks[f'g1/{model}_{kind}_det{det}_base.bin.check.json'];b=checks[f'g1/{model}_{kind}_det{det}_new.bin.check.json'];assert a['sha256']==b['sha256'] and a['bytes']==b['bytes']
 assert checks[f'g1/{model}_long_det1_new.bin.check.json']['sha256']==checks[f'g1/{model}_long_det1_new_run2.bin.check.json']['sha256']
 for kind in ['short','long']:assert checks[f'g2/{model}_f16_{kind}_run1.bin.check.json']['sha256']==checks[f'g2/{model}_f16_{kind}_run2.bin.check.json']['sha256']
bench=json.loads((science/'eval_qwen25_8064/bench.json').read_text());assert len(bench)==1 and bench[0]['ctx']==8064 and bench[0]['steps']==128
for model in ['qwen3','qwen25']:
 assert json.loads((science/f'eval_{model}/meta.json').read_text())['device']=='cuda'
 assert {r['name'] for r in json.loads((science/f'eval_{model}/tf_decode.json').read_text())}=={'book','chat'}
log=(home/'vm.log').read_text();phases=['g1','g2','g3','g4','tfd_qwen3','tfd_qwen25','bench_qwen25_8064'];assert all('=== phase '+name+' done ' in log for name in phases) and 'FAILED: receipt validation' in log
report={'status':'VALID_SCIENTIFIC_NEGATIVE_REPRODUCED','setup_failure':False,'timeout':False,'failed_stage':'final receipt validation: teacher-forced decode versus frozen historical prefill observations, both models','execution':execution,'canonical_zip':str(home/'out.zip'),'canonical_zip_sha256':sha(home/'out.zip'),'canonical_zip_bytes':(home/'out.zip').stat().st_size,'CRC_all_members_verified':True,'raw_member_count':len(inventory),'raw_uncompressed_bytes':sum(x['bytes'] for x in inventory),'stage_manifest_sha256':sha(science/'STAGING.json'),'candidate_archive_sha256':sha(stage/'ember.tgz'),'baseline_commit':m['baseline_ember_commit'],'frozen_checker_source_and_reference_sha256':source_pins,'frozen_checker_errors':replay['errors'],'all_reached_phases':phases,'g1_g2':{'byte_comparisons':14,'all_identical':True,'dump_integrity_receipts':len(checks),'full_retained_hashes_match_comparison_pairs':True,'raw_bin_arrays_removed_by_frozen_script':True},'g3':{model:{fmt:{k:data[k] for k in ['pass','tol','worst_prefill_rel','worst_decode_rel','on_gpu','top1_agree','greedy_identical']} for fmt,data in formats.items()} for model,formats in g3.items()},'g4':{model:{k:data[k] for k in ['pass','limit','worst_prefill_rel','worst_decode_rel','n','len','seed']} for model,data in g4.items()},'teacher_forcing_metric_counts':counts,'teacher_forcing_violations':violations,'bench_qwen25_8064_executed':True,'bench':bench,'proof_scope':'Offline integrity/checker replay; no model execution or new device measurements. Maxerror controls passed, but conjunction with mean/PPL historical ceilings failed.','causality':'No engine defect proven. Decode and prefill may differ through legitimate numerical paths and source/runtime history; this audit does not establish which mechanism caused these differences.','next_step_if_authorised':'Preserve failed verdict. A separate prospectively registered matched-source/current-device prefill-versus-decode diagnostic could locate present path differences; never substitute fresh controls for frozen historical ceilings retroactively or loosen thresholds to obtain a pass. No concrete engine fix justified by this receipt alone.','provider_GPU_config_or_Claude_actions':0,'canonical_receipts_modified':False,'inventory':inventory}
(out/'independent_failure_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':report['status'],'ZIPsha':report['canonical_zip_sha256'],'metric_counts':counts,'violations':len(violations),'report_sha256':sha(out/'independent_failure_audit.json')},indent=2))
