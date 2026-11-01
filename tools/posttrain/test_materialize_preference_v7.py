"""Offline immutable dataset, actual stop metadata and exact recipe checks."""
import copy
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import materialize_preference_v7 as m
from dpo_vm_preflight_v7 import verify,sha

def fixture(tmp_path):
    proposal=tmp_path/'proposal.json';proposal.write_text(json.dumps(dict(reviewed_failure_exclusions=[{'key':'f1'},{'key':'f2'}],selection_rule='whole set',downstream_condition='all eligible',bias_limitation='missingness')))
    protocol=tmp_path/'protocol.json';cache=tmp_path/'cache';ledger=tmp_path/'ledger';vocab=tmp_path/'vocab';adopt=tmp_path/'adoption.json'
    for p in [cache,ledger,vocab,Path(str(cache)+'.failures.jsonl')]:p.write_text('fixture')
    protocol.write_text(json.dumps(dict(dataset_decision_file=str(proposal),cache=str(cache),ledger=str(ledger),rubric='unused')))
    decision=json.loads(proposal.read_text())
    adoption=dict(kind='ROOT_ADOPTED_PAIRED_DATASET_POLICY_V7',adopted=True,proposal_file_sha256=sha(proposal),protocol_file_sha256=sha(protocol),eligible_candidate_sets=3018,eligible_target_rankings=6036,original_target_rankings=6040,original_full_valid_completeness=False,training_recipe_and_five_part_acceptance_unchanged=True,new_failure_auto_exclusion=False,reviewed_exclusions=decision['reviewed_failure_exclusions'])
    adoption.update({k:decision[k] for k in ['selection_rule','downstream_condition','bias_limitation']});adopt.write_text(json.dumps(adoption))
    status=dict(valid_api_rankings=6036,missing_rankings=4,eligible_missing_rankings=0,preserved_failed_rankings=2,unattempted_paired_exclusions=2)
    return protocol,proposal,adopt,vocab,status

@pytest.mark.parametrize('key',['proposal_file_sha256','protocol_file_sha256','eligible_candidate_sets','adopted','new_failure_auto_exclusion','reviewed_exclusions'])
def test_adoption_cannot_be_reinterpreted(tmp_path,key):
    p,proposal,a,v,s=fixture(tmp_path);ad=json.loads(a.read_text());ad[key]='wrong'
    with pytest.raises(ValueError):m.verify_adoption(p,proposal,ad)

def test_materialization_preserves_cut_replies_file_order_and_last_five_percent(monkeypatch,tmp_path):
    protocol,proposal,adopt,vocab,status=fixture(tmp_path)
    monkeypatch.setattr(m.campaign,'inspect',lambda p:([],status))
    records=[dict(prompt='p'+str(i),chosen='good reply',rejected='bad reply',chosen_stop='length',rejected_stop='eos') for i in range(20)]
    def pairs(req,cache,jid,rubric,out):
        Path(out).write_bytes(m.encode_rows(records));return dict(pairs=20,unranked=2,out=out)
    monkeypatch.setattr(m.api.judge,'pairs',pairs)
    out=tmp_path/'output';report=m.materialize(protocol,adopt,vocab,out,encode=lambda s:s.split())
    assert m.api.rows(out/'dpo_pairs.jsonl')==records
    counts=report['token_filter_report'];assert counts['train_pairs']==19 and counts['val_pairs']==1 and counts['chosen_finished']==0
    assert report['chosen_finished_is_breakdown_only'] and not report['training_launched']
    with pytest.raises(FileExistsError):m.materialize(protocol,adopt,vocab,out,encode=lambda s:s.split())

def test_incomplete_eligible_data_never_reaches_pair_extraction(monkeypatch,tmp_path):
    protocol,proposal,adopt,vocab,status=fixture(tmp_path);status['eligible_missing_rankings']=1
    monkeypatch.setattr(m.campaign,'inspect',lambda p:([],status))
    monkeypatch.setattr(m.api.judge,'pairs',lambda *a:pytest.fail('incomplete data reached pairs'))
    with pytest.raises(ValueError):m.materialize(protocol,adopt,vocab,tmp_path/'output',encode=lambda s:s.split())

def test_exact_original_recipe_and_expected_two_passes():
    base={'arch':{'preset':'llama-tiny'},'data':{'vocab_cap':16384}}
    s=m.training_spec(base,{'train_pairs':101},'weights','vocab','pairs')
    assert s['train']['steps']==26 and s['train']['batch']==8 and s['train']['lr']==2e-6
    assert s['train']['po']==dict(weight_mode='uniform',beta=.1,nll=.2)
    assert s['train']['lr_schedule']==dict(warmup_steps=20,decay='cosine',min_lr_frac=.1)
    assert s['train']['precision']=='fp32' and s['train']['optimizer']=='adamw'
    assert s['train']['init_from']=='weights' and s['data']['pref_pairs']=='pairs'
    assert s['export']['formats']==['safetensors','gguf'] and base=={'arch':{'preset':'llama-tiny'},'data':{'vocab_cap':16384}}

def test_vm_preflight_rejects_upload_source_or_spec_drift(tmp_path):
    upload=tmp_path/'upload';repo=tmp_path/'repo';upload.mkdir();repo.mkdir()
    (upload/'weights').write_text('weights');(repo/'source').write_text('source');spec=upload/'spec';spec.write_text('{}')
    import tarfile
    with tarfile.open(upload/'training_source.tar','w') as tar:tar.add(repo/'source',arcname='source')
    b=dict(source_commit='frozen',spec_sha256=sha(spec),upload_sha256={'weights':sha(upload/'weights'),'training_source.tar':sha(upload/'training_source.tar')},source_files_sha256={'source':sha(repo/'source')})
    assert verify(b,spec,upload,repo)['training_launched'] is False
    for field,key in [('spec_sha256',None),('upload_sha256','weights'),('source_files_sha256','source')]:
        bad=copy.deepcopy(b)
        if key:bad[field][key]='bad'
        else:bad[field]='bad'
        with pytest.raises(ValueError):verify(bad,spec,upload,repo)


def test_training_instance_requires_bound_root_freeze_and_emits_serial_standard_jobs(monkeypatch,tmp_path):
    import build_dpo_instance_v7 as b
    dataset=tmp_path/'dataset';dataset.mkdir();(dataset/'dpo_pairs.jsonl').write_text('pairs')
    manifest=dict(kind='IMMUTABLE_ELIGIBLE_API_PREFERENCE_DATASET_V7',readiness=dict(eligible_complete=True,original_full_valid_complete=False,eligible_candidate_sets=3018,excluded_candidate_sets=2),dpo_pairs_sha256=sha(dataset/'dpo_pairs.jsonl'),token_filter_report=dict(train_pairs=101))
    (dataset/'dataset_manifest.json').write_text(json.dumps(manifest))
    base=tmp_path/'base';base.write_text(json.dumps({'arch':{'preset':'llama-tiny'}}))
    checkpoint=tmp_path/'checkpoint';checkpoint.write_text('weights');vocab=tmp_path/'vocab';vocab.write_text('vocab');archive=tmp_path/'archive';archive.write_text('archive')
    source=tmp_path/'source';source.mkdir();files={}
    for name in b.REQUIRED_SOURCE_FILES:
        p=source/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text("EMBER_COMMIT='frozen-ember'\n" if name.endswith('tiny_chat_run.py') else name);files[name]=sha(p)
    heldout=tmp_path/'heldout.json';heldout.write_text(json.dumps({name:{'path':str(vocab),'sha256':sha(vocab)} for name in ['vocab','val_split','train_split']}))
    adopted=tmp_path/'heldout_adoption.json';adopted.write_text(json.dumps(dict(kind='ROOT_ADOPTED_API_HELDOUT_POLICY_V1',adopted=True,before_dpo_outcomes=True,policy_file_sha256=sha(b.FROZEN_HELDOUT_POLICY),operational_source_sha256=b.heldout_runner.source_hashes(),heldout_source_data_sha256={name:sha(vocab) for name in ['vocab','val_split','train_split']})))
    admission=dict(ember_commit='frozen-ember',heldout_input_pins_sha256=sha(heldout),heldout_adoption_sha256=sha(adopted),kind='ROOT_DPO_SOURCE_FREEZE_V1',ready=True,source_commit='frozen',source_files_sha256=files,dataset_manifest_sha256=sha(dataset/'dataset_manifest.json'),model_spec_sha256=sha(base),checkpoint_sha256=sha(checkpoint),vocab_sha256=sha(vocab),source_archive_sha256=sha(archive))
    ap=tmp_path/'admission';ap.write_text(json.dumps(admission))
    monkeypatch.setattr(b.subprocess,'check_output',lambda *a,**k:'frozen\n');monkeypatch.setattr(b.subprocess,'run',lambda *a,**k:None)
    import io,types
    monkeypatch.setattr(b.subprocess,'Popen',lambda *a,**k:types.SimpleNamespace(stdout=io.BytesIO(b'archive'),wait=lambda:0))
    out=tmp_path/'packet';binding=b.build(dataset,base,checkpoint,vocab,source,archive,ap,heldout,adopted,out)
    jobs=json.loads((out/'jobs_v1.json').read_text())
    # Execute emitted argv through the actual entry point; stray source paths
    # previously failed tuple-unpacking before bound-input verification.
    import dpo_vm_preflight_v7 as preflight
    import sys
    command=[arg.replace('{upload}',str(out)).replace('{out}',str(out/'preflight')) for arg in jobs['jobs'][0]['cmd']]
    seen=[]
    monkeypatch.setattr(preflight,'verify',lambda binding,spec,upload,repo:seen.append((binding,spec,upload,repo)) or {'kind':'DPO_VM_PREFLIGHT_PASSED'})
    monkeypatch.setattr(sys,'argv',command[1:])
    preflight.main()
    assert len(seen)==1 and seen[0][1]==out/'pk_dpo_v1.spec.json'
    assert (out/'preflight/preflight.json').exists()
    assert jobs['max_parallel']==1 and jobs['jobs'][1]['after']=='dpo_preflight'
    assert jobs['jobs'][1]['spec']['train']['steps']==26 and binding['activated'] is False
    assert set(json.loads((out/'upload_manifest.json').read_text())['files'])=={'pksft.safetensors','vocab.gguf','dpo_pairs.jsonl','training_source.tar','dpo_binding.json','pk_dpo_v1.spec.json','dialogues_val.jsonl','dialogues_train.jsonl','DPO_RUNTIME_JOBS_V1.json','dpo_checkpoint_contract_v1.json'}
    bad=copy.deepcopy(admission);bad['checkpoint_sha256']='wrong'
    with pytest.raises(ValueError):b.verify_source_admission(bad,source,dataset/'dataset_manifest.json',base,checkpoint,vocab,archive)
    (source/'tools/mtstudio.cpp').write_text('modified')
    with pytest.raises(ValueError):b.verify_source_admission(admission,source,dataset/'dataset_manifest.json',base,checkpoint,vocab,archive)
