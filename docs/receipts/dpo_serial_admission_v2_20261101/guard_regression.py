import sys,json,datetime,tempfile,hashlib,importlib.util
from pathlib import Path
p=Path('/home/jonat/dpo_root_admission_20261101/guarded_runner.py')
s=importlib.util.spec_from_file_location('guard',p);g=importlib.util.module_from_spec(s);s.loader.exec_module(g)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
passed=[]
with tempfile.TemporaryDirectory(prefix='dpo_guard_test_') as td:
 t=Path(td);payload=t/'x';payload.write_text('frozen');review=t/'review.json'
 argv=['/tmp/runner_multi.py','--backend','runpod','--gpu','L4','--cloud','SECURE','--volume-gb','30','--max-usd','5','--max-hours','4','--boot-script','tools/colab_chat/boot_dpo_checkpoint_v1.sh','--commit','a'*40,'--state-dir','/tmp/dpo-test-state','--deadline-utc',(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(hours=4)).isoformat(),'--upload','model='+str(payload)]
 r={'kind':'ROOT_DPO_SERIAL_ADMISSION_V1','status':'APPROVED_AFTER_OFFLINE_AND_SERIAL_CHECKS','driver_argv':['python3']+argv,'deadline_utc':argv[argv.index('--deadline-utc')+1],'state_dir':'/tmp/dpo-test-state','pins':{str(payload):sha(payload),str(p):sha(p)},'upload_names':['model']}
 def write():review.write_text(json.dumps(r));return sha(review)
 h=write();assert g.validate(review,h,argv)==argv;passed.append('valid_fixture')
 def rejects(label,fn):
  try:fn()
  except (ValueError,KeyError):passed.append(label);return
  raise AssertionError(label)
 rejects('review_digest',lambda:g.validate(review,'0'*64,argv))
 for flag in ['--max-usd','--max-hours','--commit','--state-dir','--boot-script']:
  bad=argv.copy();bad[bad.index(flag)+1]='changed';rejects(flag,lambda:g.validate(review,h,bad))
 payload.write_text('changed');rejects('source_drift',lambda:g.validate(review,h,argv));payload.write_text('frozen')
 r['status']='PROPOSED';h=write();rejects('unapproved',lambda:g.validate(review,h,argv));r['status']='APPROVED_AFTER_OFFLINE_AND_SERIAL_CHECKS'
 old=r['deadline_utc'];r['deadline_utc']='2000-01-01T00:00:00Z';r['driver_argv'][r['driver_argv'].index('--deadline-utc')+1]=r['deadline_utc'];h=write();rejects('expired',lambda:g.validate(review,h,r['driver_argv'][1:]))
 r['deadline_utc']=old;r['driver_argv'][r['driver_argv'].index('--deadline-utc')+1]=old
 payload.unlink();target=t/'target';target.write_text('frozen');payload.symlink_to(target);h=write();rejects('symlink_upload',lambda:g.validate(review,h,r['driver_argv'][1:]))
Path('/home/jonat/dpo_root_admission_20261101/guard_tests.json').write_text(json.dumps({'passed':passed,'count':len(passed),'provider_calls':0,'gpu_calls':0},indent=2)+'\n')
print('Guard rejection checks passed',len(passed))
