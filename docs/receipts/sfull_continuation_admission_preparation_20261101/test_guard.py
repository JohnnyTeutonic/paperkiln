from pathlib import Path
import importlib.util,copy,json,tempfile,datetime
r=Path('/home/jonat/sfull_continuation_admission_v4_20261101');sp=importlib.util.spec_from_file_location('g',r/'guarded_continuation.py');g=importlib.util.module_from_spec(sp);sp.loader.exec_module(g)
base=json.loads((r/'probe_review.json').read_text());passed=[]
with tempfile.TemporaryDirectory() as td:
 t=Path(td);s=t/'state';s.mkdir();orig=Path(base['state_dir'])/'state.json';(s/'state.json').write_bytes(orig.read_bytes());res=json.loads(Path(base['reservation_path']).read_text());res['reservations'][0]['state_dir']=str(s);(t/'res.json').write_text(json.dumps(res));base['state_dir']=str(s);base['reservation_path']=str(t/'res.json');base['pins']={str(p):g.sha(p) for p in [base['runner_argv'][base['runner_argv'].index('--upload')+1].split('=',1)[1]]};base['runner_argv'][base['runner_argv'].index('--state-dir')+1]=str(s)
 def check(v):
  f=t/'review.json';f.write_text(json.dumps(v));return g.validate(f,g.sha(f),v['runner_argv'][1:])
 assert check(base);passed.append('complete_valid_contract')
 for flag,val in [('--max-usd','8'),('--gpu','A100'),('--max-hours','24'),('--volume-gb','100'),('--commit','a'*40),('--boot-script','evil.sh')]:
  v=copy.deepcopy(base);v['runner_argv'][v['runner_argv'].index(flag)+1]=val
  try:check(v)
  except ValueError:passed.append('reject_'+flag)
  else:raise AssertionError(flag)
 for kind in ['omitted_prior_ledger','changed_prior_ledger','protected_owned_volume','protected_retired_volume','protected_owned_volume_dict','protected_retired_volume_dict','already_home','duplicate_ledger']:
  state=json.loads(orig.read_text())
  if kind=='omitted_prior_ledger':state['runpod']['ledger']['pods'].pop()
  if kind=='changed_prior_ledger':state['runpod']['ledger']['pods'][0]['usd_per_hr']=0
  if kind=='protected_owned_volume':state['runpod']['volume']=base['protected_old_volume']
  if kind=='protected_retired_volume':state['runpod']['retired_volumes']=[base['protected_old_volume']]
  if kind=='protected_owned_volume_dict':state['runpod']['volume']={'id':base['protected_old_volume'],'gb':80,'dc':'EU-RO-1'}
  if kind=='protected_retired_volume_dict':state['runpod']['retired_volumes']=[{'id':base['protected_old_volume'],'gb':80,'dc':'EU-RO-1'}]
  if kind=='already_home':state['campaign_home']=True
  if kind=='duplicate_ledger':state['runpod']['ledger']['pods'].append(copy.deepcopy(state['runpod']['ledger']['pods'][0]))
  (s/'state.json').write_text(json.dumps(state))
  try:check(base)
  except ValueError:passed.append('reject_'+kind)
  else:raise AssertionError(kind)
 (s/'state.json').write_bytes(orig.read_bytes())
 for kind in ['historical_path_reuse','expired_deadline','wrong_numerical','missing_pin','reservation_reset']:
  v=copy.deepcopy(base)
  if kind=='historical_path_reuse':v['original_art_dir']=v['art_dir']
  if kind=='expired_deadline':v['deadline_utc']='2020-01-01T00:00:00Z';v['runner_argv'][v['runner_argv'].index('--deadline-utc')+1]=v['deadline_utc']
  if kind=='wrong_numerical':v['numerical_commit']='b'*40
  if kind=='missing_pin':v['pins'][str(r/'guarded_continuation.py')]='0'*64
  if kind=='reservation_reset':
   v['reservation_path']=str(t/'res_bad.json');bad=copy.deepcopy(res);bad['reservations'].append(copy.deepcopy(bad['reservations'][0]));Path(v['reservation_path']).write_text(json.dumps(bad))
  try:check(v)
  except ValueError:passed.append('reject_'+kind)
  else:raise AssertionError(kind)
 # Re-sign fixtures deliberately to exercise semantic rather than hash checks.
 for kind in ['carried_ledger_not_original','absolute_window_reset','symlink_pin']:
  v=copy.deepcopy(base)
  if kind=='carried_ledger_not_original':
   prior=json.loads(Path(v['carried_ledger']).read_text());prior['pods'].pop();bad=t/'bad_prior.json';bad.write_text(json.dumps(prior));v['carried_ledger']=str(bad)
  if kind=='absolute_window_reset':
   v['probe_only']=False;v['admitted_at_utc']='2020-01-01T00:00:00Z'
  if kind=='symlink_pin':
   link=t/'pin_link';link.symlink_to(r/'guarded_continuation.py');v['pins'][str(link)]=g.sha(link)
  try:check(v)
  except ValueError:passed.append('reject_'+kind)
  else:raise AssertionError(kind)
(r/'guard_validation.json').write_text(json.dumps({'passed':len(passed),'cases':passed,'provider_calls':0,'gpu_calls':0},indent=2)+'\n');print('S-full admission guard: '+str(len(passed))+' cases pass')
