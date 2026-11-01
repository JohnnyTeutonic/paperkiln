import copy,hashlib,json,socket,types
from pathlib import Path
import pytest
import offline_intake_guard as guard
ROOT=Path(__file__).resolve().parent

def records():return tuple(json.loads(p.read_text()) for p in [guard.ORIGINAL,guard.SUPPLEMENT,guard.BINDING])

def test_actual_adopted_supplement_and_frozen_sources_are_valid_not_ready():
 proof=guard.verify_sources()
 assert proof['status']=='SOURCE_LINKAGE_VALID_NOT_SAMPLING_READY'
 assert proof['sampling_readiness_established'] is False
 assert proof['changed_consumers']==5 and proof['extra_consumers']==3
 assert proof['judging_launch_authorized'] is False

@pytest.mark.parametrize('bad',['not_adopted','wrong_kind','after_outcomes','modified_original','sources_changed','launch_authorized','waive_gates','automatic_replacement','raised_stage_cap','wrong_adoption_hash','wrong_binding_hash','wrong_source_commit','consumer_changed','consumer_omitted','extra_omitted','binding_source_changed','binding_original_changed'])
def test_linkage_mutations_fail_closed(bad):
 original,supplement,binding=records()
 if bad=='not_adopted':supplement['adopted']=False
 elif bad=='wrong_kind':supplement['kind']='PROPOSAL'
 elif bad=='after_outcomes':supplement['before_dpo_training_and_outcomes']=False
 elif bad=='modified_original':supplement['original_adoption_modified']=True
 elif bad=='sources_changed':supplement['original_heldout_operational_sources_unchanged']=False
 elif bad=='launch_authorized':supplement['judging_launch_authorized']=True
 elif bad=='waive_gates':supplement['all_five_original_gates_required']=False
 elif bad=='automatic_replacement':supplement['automatic_model_replacement']=True
 elif bad=='raised_stage_cap':supplement['limits_unchanged']['stage_usd']=26
 elif bad=='wrong_adoption_hash':supplement['original_adoption_sha256']='0'*64
 elif bad=='wrong_binding_hash':supplement['v4_training_binding_sha256']='0'*64
 elif bad=='wrong_source_commit':supplement['v4_training_source_commit']='0'*40
 elif bad=='consumer_changed':next(iter(supplement['changed_consumer_hashes'].values()))['v4_actual_sha256']='0'*64
 elif bad=='consumer_omitted':supplement['changed_consumer_hashes'].pop(next(iter(supplement['changed_consumer_hashes'])))
 elif bad=='extra_omitted':supplement['extra_bound_consumer_hashes']={}
 elif bad=='binding_source_changed':binding['source_commit']='0'*40
 elif bad=='binding_original_changed':binding['heldout_adoption_sha256']='0'*64
 with pytest.raises(ValueError):guard.validate_linkage(original,supplement,binding)

@pytest.mark.parametrize('target',['original','supplement','binding','policy'])
def test_exact_anchor_byte_mutation_refused(tmp_path,target):
 p={'original':guard.ORIGINAL,'supplement':guard.SUPPLEMENT,'binding':guard.BINDING,'policy':guard.POLICY}[target]
 q=tmp_path/'changed.json';q.write_bytes(p.read_bytes()+b'\n')
 with pytest.raises(ValueError,match='hash differs'):guard.check_file(q,guard.sha(p))

def test_missing_actual_sampling_never_imports_builder_or_claims_ready(monkeypatch,tmp_path):
 proof=guard.verify_sources()
 monkeypatch.setattr(guard,'frozen_modules',lambda:pytest.fail('builder imported before actual receipts exist'))
 plan=json.loads((ROOT/'intake_plan.json').read_text())
 plan['builder_argv'][plan['builder_argv'].index('--sampling')+1]=str(tmp_path/'missing-actual-sampling.json')
 p=tmp_path/'plan.json';p.write_text(json.dumps(plan))
 out=tmp_path/'instance.json'
 with pytest.raises(ValueError,match='canonical regular input'):guard.prepare(p,out)
 assert not out.exists() and proof['sampling_readiness_established'] is False

def test_changed_training_binding_rejected_before_builder(monkeypatch,tmp_path):
 monkeypatch.setattr(guard,'frozen_modules',lambda:pytest.fail('unexpected builder import'))
 plan=json.loads((ROOT/'intake_plan.json').read_text());fake=tmp_path/'binding.json';fake.write_bytes(guard.BINDING.read_bytes()+b'\n')
 plan['builder_argv'][plan['builder_argv'].index('--training-binding')+1]=str(fake)
 p=tmp_path/'plan.json';p.write_text(json.dumps(plan))
 with pytest.raises(ValueError,match='adoption/policy/binding'):guard.prepare(p,tmp_path/'instance.json')

def test_source_pin_escape_and_symlink_refused(tmp_path):
 with pytest.raises(ValueError,match='escapes'):guard.source_path('../../../../outside',True)
 p=tmp_path/'file';p.write_text('unchanged');link=tmp_path/'alias';link.symlink_to(p)
 with pytest.raises(ValueError,match='regular'):guard.check_file(link,guard.sha(p))

def test_offline_barrier_blocks_keys_provider_and_socket_then_restores():
 oldkey=lambda p:'synthetic';oldpost=lambda *a:'synthetic';api=types.SimpleNamespace(load_key=oldkey,http_post=oldpost)
 original_connection=socket.create_connection
 with guard.offline_provider_barrier(api):
  for f,args in [(api.load_key,('never-read',)),(api.http_post,()),(socket.create_connection,(('localhost',1),))]:
   with pytest.raises(RuntimeError,match='offline intake forbids'):f(*args)
 assert api.load_key is oldkey and api.http_post is oldpost and socket.create_connection is original_connection

def test_cli_has_no_campaign_dispatch_entrypoint():
 import subprocess,sys
 result=subprocess.run([sys.executable,str(ROOT/'offline_intake_guard.py'),'campaign'],capture_output=True,text=True)
 assert result.returncode==2 and 'invalid choice' in result.stderr
