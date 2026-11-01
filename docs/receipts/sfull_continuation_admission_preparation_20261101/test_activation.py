from pathlib import Path
import tempfile,json,hashlib,datetime,types,importlib.util,sys
src=Path('/home/jonat/sfull_continuation_admission_v4_20261101');spec=importlib.util.spec_from_file_location('activation',src/'activate_watchdog.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);passed=[]
for case in ['positive','wrong_argv','wrong_review_hash','stale','other_enabled','duplicate_state','active_provider','exhausted_cap','guard_source_drift']:
 with tempfile.TemporaryDirectory() as td:
  root=Path(td);state=root/'runner';state.mkdir();(state/'state.json').write_text('{"runpod":{"ledger":{}}}')
  m.ROOT=root;m.CONFIG=root/'config.json';m.CONFIG.write_text('{"drivers":[]}');guard=root/'guarded_continuation.py';guard.write_text('def validate(*a): pass\ndef validate_checkpoint(*a):pass\n')
  argv=['python3','runner_multi.py','--state-dir',str(state)];r={'dispatch_authorized':True,'probe_only':False,'runner_argv':argv,'state_dir':str(state),'source_checkout':str(root),'deadline_utc':'2099-01-01T00:00:00Z','pins':{str(guard):hashlib.sha256(guard.read_bytes()).hexdigest()}};review=root/'admission.json';review.write_text(json.dumps(r));h=hashlib.sha256(review.read_bytes()).hexdigest();cmd=['python3',str(guard),'--review',str(review),'--review-sha256',h,'--']+argv[1:];d={'review':str(review),'review_sha256':h,'guard_argv':cmd};now=datetime.datetime.now(datetime.timezone.utc);(root/'serial_dispatch_check.json').write_text(json.dumps({'checked_at_utc':now.isoformat()}))
  fake=types.SimpleNamespace(list_pods=lambda:[],ledger_usd=lambda l:0);sys.modules['runpod_api']=fake
  if case=='wrong_argv':d['guard_argv'][-1]='wrong'
  if case=='wrong_review_hash':d['review_sha256']='0'*64
  if case=='stale':(root/'serial_dispatch_check.json').write_text(json.dumps({'checked_at_utc':(now-datetime.timedelta(minutes=5)).isoformat()}))
  if case=='other_enabled':m.CONFIG.write_text('{"drivers":[{"name":"other GPU","enabled":true}]}')
  if case=='duplicate_state':m.CONFIG.write_text(json.dumps({'drivers':[{'state_dir':str(state),'enabled':False}]}))
  if case=='active_provider':fake.list_pods=lambda:[{'id':'foreign'}]
  if case=='guard_source_drift':guard.write_text('raise AssertionError("unverified source executed")\n')
  if case=='exhausted_cap':fake.ledger_usd=lambda l:7
  (root/'dispatch.json').write_text(json.dumps(d));before=m.CONFIG.read_bytes()
  try:m.activate()
  except ValueError:
   assert case!='positive';assert m.CONFIG.read_bytes()==before;assert not(root/'watchdog_activation.json').exists();passed.append(case)
  else:
   assert case=='positive';out=json.loads(m.CONFIG.read_text());assert len(out['drivers'])==1 and out['drivers'][0]['enabled'];passed.append(case)
sys.modules.pop('runpod_api',None)
(src/'activation_validation.json').write_text(json.dumps({'cases_passed':len(passed),'cases':passed,'isolated_fake_provider_only':True,'actual_provider_calls':0,'actual_watchdog_mutations':0},indent=2)+'\n');print('Activation routing: '+str(len(passed))+' isolated cases pass')
