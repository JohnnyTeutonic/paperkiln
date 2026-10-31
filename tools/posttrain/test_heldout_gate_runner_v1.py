"""Offline replay of all dispositions, fixed order seeds and fail-stop restart boundaries."""
import copy
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import heldout_sampling_receipt_v1 as sampling
import heldout_gate_runner_v1 as runner

def encoder(text):return list(range(300 if 'too long' in text else 4))

def sample_fixture():
    items=[('long','too long'),('same','same'),('different','different')]
    def send(which):
        def reply(body):
            user=body['user_input'];text='identical' if user=='same' else which
            return dict(prompt_ids=encoder('<|user|>'+user+'<|assistant|>'),reply=text,stop_reason='eos',tokens=2)
        return reply
    return sampling.capture(items,send('a'),send('b'),encoder,'frozen-ember')

def test_original_index_seeds_and_explicit_skips_no_refill():
    r=sample_fixture()
    assert [d['status'] for d in r['dispositions']]==['over_context','identical','ok']
    assert len(r['requests'])==1 and len(r['raw_calls'])==4
    assert [c['body']['seed'] for c in r['raw_calls']]==[72,72,73,73]
    assert sampling.replay(r,encoder,r['items'],require_300=False)==r['requests']
    with pytest.raises(ValueError,match='300'):sampling.replay(r,encoder)

@pytest.mark.parametrize('change',['drop','refill','order','request','skip','raw'])
def test_replay_rejects_selected_or_silently_incomplete_sampling(change):
    r=sample_fixture()
    if change=='drop':r['dispositions'].pop()
    if change=='refill':r['items'].append(['new','replacement'])
    if change=='order':r['items'].reverse()
    if change=='request':r['requests'][0]['candidates'].reverse()
    if change=='skip':r['dispositions'][1]['status']='over_context'
    if change=='raw':r['raw_calls'].append(copy.deepcopy(r['raw_calls'][-1]))
    with pytest.raises((ValueError,KeyError)):sampling.replay(r,encoder,require_300=False)

@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(runner.memory,'wait_for_memory',lambda *a:None)
    monkeypatch.setattr(runner.api,'load_key',lambda *a:pytest.fail('credential accessor used'))
    monkeypatch.setattr(runner.api,'http_post',lambda *a,**k:pytest.fail('provider used'))
    monkeypatch.setattr(runner.sampling.sbs,'count',lambda *a:dict(unranked=0,b_win_share=.7,b_win_share_wilson95=[.6,.8]))

def policy():return {'rubric_path':'fixture','budget_migration':{'legacy_cache':'fixture','prior_api_ledgers':['prior']}}

def test_complete_only_after_both_order_calls_and_no_duplicate_restart(tmp_path):
    calls=[];instance=dict(cache='fixture',ledger='fixture',env_file='unused')
    def inspect(_):return policy(),[],dict(calls=len(calls),missing_rankings=2-len(calls),valid_rankings=len(calls))
    def dispatch(*args,**kwargs):
        assert args[6]==2-len(calls) and args[7:11]==(.25,200,0,'heldout_gate')
        assert kwargs['stop_on_failure'] is True
        n=args[6];calls.extend(range(n));return dict(calls=n,failed_rankings=0,pending_reservations=0)
    assert runner.campaign(instance,tmp_path,inspect,dispatch)==0 and len(calls)==2
    stopped=json.loads((tmp_path/'STOPPED').read_text());assert stopped['all_five_gates_required']
    assert stopped['kind']=='COMPLETE_HELDOUT_JUDGING_NOT_MODEL_ADOPTION'
    assert runner.campaign(instance,tmp_path,inspect,dispatch)==0 and len(calls)==2

@pytest.mark.parametrize('failure',['incomplete','unknown','budget'])
def test_failure_or_unknown_never_becomes_tie_retry_or_exclusion(tmp_path,failure):
    calls=[];instance=dict(cache='fixture',ledger='fixture',env_file='unused')
    def inspect(_):return policy(),[],dict(calls=0,missing_rankings=2)
    def dispatch(*args,**kwargs):
        calls.append(1)
        if failure!='incomplete':raise RuntimeError('PRIVATE_FAILURE')
        return dict(calls=1,failed_rankings=1,pending_reservations=0)
    assert runner.campaign(instance,tmp_path,inspect,dispatch)==2
    before={p.name:p.read_bytes() for p in tmp_path.iterdir()}
    assert runner.campaign(instance,tmp_path,inspect,dispatch)==2 and len(calls)==1
    assert before=={p.name:p.read_bytes() for p in tmp_path.iterdir()}
    assert 'PRIVATE' not in (tmp_path/'HALTED').read_text()

def test_registered_cap_cannot_stop_as_success_with_missing_rankings(tmp_path):
    calls=[]
    def inspect(_):return policy(),[],dict(calls=600,missing_rankings=1)
    assert runner.campaign({},tmp_path,inspect,lambda *a,**k:calls.append(1))==2
    assert not calls and (tmp_path/'HALTED').exists()
