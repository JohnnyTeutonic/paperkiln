import copy,datetime,hashlib,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
import pytest
import prepare_dispatch as prep
ROOT=Path(__file__).resolve().parent

def module(path):
 spec=importlib.util.spec_from_file_location('reviewed_guard',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def write_review(path,record):path.write_text(json.dumps(record));return prep.sha(path)
@pytest.fixture
def prospective(tmp_path):
 plan=json.loads((ROOT/'plan_snapshot.json').read_text());plan['state_dir']=str(tmp_path/'state');plan['art_dir']=str(tmp_path/'art')
 serial=tmp_path/'serial.json';serial.write_text(json.dumps(dict(checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),gpu_campaigns_active=0,previous_campaign_home=True,previous_owned_resources_deleted=True)))
 result=prep.prepare(plan,tmp_path/'dispatch',serial,ROOT/'proposed_guarded_runner.py')
 path=Path(result['review']);record=json.loads(path.read_text());return plan,serial,result,path,record

def test_good_frozen_source_and_original_guard_accept(prospective):
 _,_,result,path,r=prospective
 assert module(ROOT/'proposed_guarded_runner.py').validate(path,prep.sha(path),r['driver_argv'][1:])==r['driver_argv'][1:]
 assert module(ROOT/'original_guard.py').validate(path,prep.sha(path),r['driver_argv'][1:])==r['driver_argv'][1:]

@pytest.mark.parametrize('bad',['dirty_artifact','dirty_state','wrong_source','long_deadline','symlink_pin','unpinned_upload'])
def test_proposed_refuses_gaps_original_accepts(prospective,bad,tmp_path):
 _,_,_,path,r=prospective
 if bad=='dirty_artifact':Path(r['art_dir']).mkdir();(Path(r['art_dir'])/'gate_summary.json').write_text('{"pass":true}')
 elif bad=='dirty_state':Path(r['state_dir']).mkdir();(Path(r['state_dir'])/'state.json').write_text('{"runpod":{"volume":"foreign-volume"}}')
 elif bad=='wrong_source':r['driver_argv'][r['driver_argv'].index('--commit')+1]='a'*40
 elif bad=='long_deadline':
  r['deadline_utc']=(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(days=1)).isoformat();r['driver_argv'][r['driver_argv'].index('--deadline-utc')+1]=r['deadline_utc']
 elif bad=='symlink_pin':
  target=tmp_path/'payload';target.write_text('pin');link=tmp_path/'symlink';link.symlink_to(target);r['pins'][str(link)]=prep.sha(target)
 elif bad=='unpinned_upload':
  target=tmp_path/'payload';target.write_text('arbitrary');i=r['driver_argv'].index('--upload');r['driver_argv'][i+1]='ember.tgz='+str(target)
 digest=write_review(path,r)
 assert module(ROOT/'original_guard.py').validate(path,digest,r['driver_argv'][1:])
 with pytest.raises(ValueError):module(ROOT/'proposed_guarded_runner.py').validate(path,digest,r['driver_argv'][1:])

@pytest.mark.parametrize('bad',['dirty_artifact','dirty_state','stale_serial','active_gpu','not_home','resources_left','input_drift'])
def test_dispatch_refuses_before_writing_review(prospective,tmp_path,bad):
 plan,serial,_,_,_=prospective
 if bad in ('dirty_artifact','dirty_state'):Path(plan['art_dir' if bad=='dirty_artifact' else 'state_dir']).mkdir()
 elif bad=='input_drift':plan['pins'][plan['uploads']['HEAD']]='0'*64
 else:
  s=json.loads(serial.read_text())
  if bad=='stale_serial':s['checked_at_utc']=(datetime.datetime.now(datetime.timezone.utc)-datetime.timedelta(minutes=3)).isoformat()
  elif bad=='active_gpu':s['gpu_campaigns_active']=1
  elif bad=='not_home':s['previous_campaign_home']=False
  elif bad=='resources_left':s['previous_owned_resources_deleted']=False
  serial.write_text(json.dumps(s))
 with pytest.raises(ValueError):prep.prepare(plan,tmp_path/'refused',serial,ROOT/'proposed_guarded_runner.py')
 assert not (tmp_path/'refused').exists()

def test_pinned_anchor_allows_same_admission_resume_and_refuses_alien(prospective):
 _,_,_,path,r=prospective;m=module(ROOT/'proposed_guarded_runner.py');digest=prep.sha(path)
 state=Path(r['state_dir']);state.mkdir();anchor=state/'E9_ADMISSION_ANCHOR.json';anchor.write_text(json.dumps({'review_sha256':digest}));(state/'state.json').write_text('{"runpod":{"volume":"owned-later"}}');Path(r['art_dir']).mkdir()
 assert m.validate(path,digest,r['driver_argv'][1:])
 anchor.write_text(json.dumps({'review_sha256':'wrong'}))
 with pytest.raises(ValueError,match='anchor differs'):m.validate(path,digest,r['driver_argv'][1:])

def test_real_guard_no_provider_owner_recognized_by_frozen_sweeper(prospective):
 _,_,result,path,r=prospective
 sys.path.insert(0,str(Path(r['source'])/'tools/colab_chat'));import runpod_sweep as sweep
 env=dict(os.environ,E9_OWNER_PROBE='1');process=subprocess.Popen(result['guard_argv'],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 try:
  assert process.stdout.readline().strip()=='E9_OWNER_PROBE_VALIDATED_NO_PROVIDER'
  state=Path(r['state_dir']);state.mkdir();(state/'runner.pid').write_text(str(process.pid))
  assert sweep.runner_alive(str(state))
  (state/'runner.pid').write_text('2147483647');assert not sweep.runner_alive(str(state))
 finally:
  process.terminate();process.wait(timeout=5);process.stdout.close();process.stderr.close()
 assert not (state/'state.json').exists()
