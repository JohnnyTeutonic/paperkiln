"""Hash-bound DPO manager integration, no training or provider access."""
import copy,json,os,subprocess,sys
from pathlib import Path
import pytest
import build_dpo_instance_v7 as b
from pref_artifact_io_v1 import sha


def packet(tmp_path,commit):
    spec=b.training_spec({'arch':{'preset':'llama-tiny'}},{'train_pairs':101},'/content/chat/upload/pksft.safetensors','/content/chat/upload/vocab.gguf','/content/chat/upload/dpo_pairs.jsonl')
    jobs={'max_parallel':1,'jobs':[{'name':'dpo_preflight','cmd':['python3','preflight']},{'name':'pk_dpo_v1','after':'dpo_preflight','spec':spec,'init_file':'pksft.safetensors'}]}
    binding=dict(source_commit=commit,dataset_manifest_sha256='d'*64,uploads={},upload_sha256={'pksft.safetensors':'a'*64,'vocab.gguf':'b'*64,'dpo_pairs.jsonl':'c'*64})
    return spec,jobs,binding


def test_packet_has_independent_contract_jobs_and_binding_hashes(monkeypatch,tmp_path):
    spec,jobs,binding=packet(tmp_path,'e'*40);seen=[];monkeypatch.setattr(b.subprocess,'run',lambda *a,**k:seen.append(a))
    added=b.bind_checkpoint_packet(tmp_path,spec,jobs,binding,tmp_path)
    assert set(added)=={'DPO_RUNTIME_JOBS_V1.json','dpo_checkpoint_contract_v1.json'}
    assert binding['runtime_jobs_sha256']==sha(tmp_path/'jobs_v1.json')
    assert binding['checkpoint_contract_sha256']==sha(tmp_path/'dpo_checkpoint_contract_v1.json')
    contract=json.loads((tmp_path/'dpo_checkpoint_contract_v1.json').read_text())
    assert contract['runtime_spec']==spec and contract['source_commit']=='e'*40
    assert set(contract['inputs_sha256'])=={spec['data']['vocab'],spec['data']['pref_pairs'],spec['train']['init_from']}
    c=jobs['jobs'][1]['dpo_full_checkpoint_v1']
    assert c['max_gap_seconds']==180 and c['every_steps']==10 and c['steps']==26
    assert c['contract_sha256']==binding['checkpoint_contract_sha256'] and seen
    assert binding['boot_script']=='tools/colab_chat/boot_dpo_checkpoint_v1.sh'


@pytest.fixture
def owner():
    root=Path(os.environ.get('DPO_CHECKPOINT_SOURCE_ROOT',Path(__file__).resolve().parents[2]))
    if not (root/'tools/colab_chat/dpo_checkpoint_manager_v1.py').exists():pytest.skip('checkpoint owner source not installed; set DPO_CHECKPOINT_SOURCE_ROOT')
    sys.path.insert(0,str(root/'tools/colab_chat'))
    import dpo_checkpoint_manager_v1 as manager
    commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    return root,commit,manager


def test_real_owner_accepts_exact_emitted_packet(owner,tmp_path):
    root,commit,manager=owner;spec,jobs,binding=packet(tmp_path,commit)
    b.bind_checkpoint_packet(tmp_path,spec,jobs,binding,root)
    contract,digest=manager.validate_jobs(root,tmp_path/'jobs_v1.json',tmp_path)
    assert digest==binding['checkpoint_contract_sha256'] and contract['runtime_spec']==spec


@pytest.mark.parametrize('change',['jobs','contract','binding_source','recipe','third_train'])
def test_real_owner_refuses_changed_bound_operational_packet(owner,tmp_path,change):
    root,commit,manager=owner;spec,jobs,binding=packet(tmp_path,commit)
    b.bind_checkpoint_packet(tmp_path,spec,jobs,binding,root)
    if change=='jobs':(tmp_path/'jobs_v1.json').write_text('{}')
    if change=='contract':(tmp_path/'dpo_checkpoint_contract_v1.json').write_text('{}')
    if change=='binding_source':
        binding['source_commit']='f'*40;(tmp_path/'dpo_binding.json').write_text(json.dumps(binding))
    if change in ['recipe','third_train']:
        altered=json.loads((tmp_path/'jobs_v1.json').read_text())
        if change=='recipe':altered['jobs'][1]['spec']['train']['po']['beta']=.2
        else:altered['jobs'].append(copy.deepcopy(altered['jobs'][1]))
        (tmp_path/'jobs_v1.json').write_text(json.dumps(altered))
        binding['runtime_jobs_sha256']=sha(tmp_path/'jobs_v1.json');(tmp_path/'dpo_binding.json').write_text(json.dumps(binding))
    with pytest.raises((ValueError,KeyError)):manager.validate_jobs(root,tmp_path/'jobs_v1.json',tmp_path)
