from pathlib import Path
import importlib.util,json,hashlib,tempfile,datetime,copy
r=Path('/home/jonat/e9_matched_v2_admission_20261101');spec=importlib.util.spec_from_file_location('guard',r/'guarded_runner.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
with tempfile.TemporaryDirectory() as td:
 p=Path(td);j=p/'review.json';deadline=(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(hours=2)).isoformat()
 argv=['python3',str(r/'source/tools/colab_chat/runner_multi.py'),'--backend','runpod','--gpu','L4','--cloud','SECURE','--volume-gb','30','--max-usd','2','--max-hours','2','--commit','a'*40,'--state-dir',str(p/'state'),'--art',str(p/'art'),'--deadline-utc',deadline]
 base={'kind':'ROOT_E9_MATCHED_SERIAL_ADMISSION_V2','status':'APPROVED_AFTER_OFFLINE_AND_SERIAL_CHECKS','driver_argv':argv,'state_dir':str(p/'state'),'art_dir':str(p/'art'),'deadline_utc':deadline,'pins':{},'upload_names':[]}
 def validate(v,args=None):
  j.write_text(json.dumps(v));return g.validate(j,g.sha(j),args or v['driver_argv'][1:])
 assert validate(base)==argv[1:];cases=0
 for flag,val in [('--max-usd','3'),('--max-hours','4'),('--volume-gb','40'),('--gpu','A100')]:
  v=copy.deepcopy(base);v['driver_argv'][v['driver_argv'].index(flag)+1]=val
  try:validate(v)
  except ValueError:cases+=1
  else:raise AssertionError(flag)
 v=copy.deepcopy(base);v['pins'][str(r/'guarded_runner.py')]='0'*64
 try:validate(v)
 except ValueError:cases+=1
 else:raise AssertionError('pin')
 (p/'state').mkdir();(p/'state/state.json').write_text('{"campaign_home":true}')
 try:validate(base)
 except ValueError:cases+=1
 else:raise AssertionError('home')
 assert cases==6
 (r/'guard_validation.json').write_text(json.dumps({'positive_validations':1,'negative_refusals':cases,'gpu_calls':0,'launch_authorized':False},indent=2)+'\n');print('E9 guard: positive and six refusal cases pass; no provider calls')
