"""Offline tests only. Provider dispatch and credentials are always replaced."""
import copy
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent))
import chat_judge_campaign_continuation_v7 as c

@pytest.fixture(autouse=True)
def headroom(monkeypatch):
    monkeypatch.setattr(c, "memory_available_bytes", lambda: 2 * 1024**3)

def protocol():
    return {"max_new_attempts":3468,"batch_calls":24,"minimum_mem_available_bytes":1024**3,
       "cache":"unused","legacy_cache":"unused","rubric":"unused","ledger":"unused","env_file":"unused",
       "reserve_per_call_usd":.25,"max_usd":200,"gate_reserve_usd":25,"prior_api_ledgers":[],
       "reviewed_failure_exclusions":[{"key":"f1","paired_unattempted_key":"p1"},{"key":"f2","paired_unattempted_key":"p2"}]}

def test_restart_only_dispatches_remaining_and_never_claims_full_completeness(tmp_path):
    p=protocol();calls=[]
    def inspect(_):
        return [],dict(attempts_since_start=len(calls),missing_rankings=3472-len(calls),
                       eligible_missing_rankings=3468-len(calls),valid_api_rankings=2568+len(calls))
    def interrupt(*args,**kw):
        calls.extend(range(3));raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):c.campaign(p,tmp_path,interrupt,inspect)
    def dispatch(*args,**kw):
        assert kw['excluded_failed_keys']==('f1','f2')
        assert kw['excluded_unattempted_keys']==('p1','p2')
        assert args[6]==min(24,3468-len(calls))
        n=args[6];calls.extend(range(n));return dict(calls=n,failed_rankings=0,pending_reservations=0)
    assert c.campaign(p,tmp_path,dispatch,inspect)==0
    stopped=json.loads((tmp_path/'STOPPED').read_text())
    assert len(calls)==3468 and stopped['valid_api_rankings']==6036
    assert stopped['missing_rankings']==4 and stopped['eligible_judging_complete']
    assert stopped['full_judging_complete'] is False and not (tmp_path/'COMPLETE').exists()
    assert c.campaign(p,tmp_path,dispatch,inspect)==0 and len(calls)==3468

@pytest.mark.parametrize('failure',['unknown','incomplete','budget'])
def test_new_problem_halts_and_never_retries(tmp_path,failure):
    calls=[]
    def inspect(_):return [],dict(attempts_since_start=0,missing_rankings=5,eligible_missing_rankings=1)
    def dispatch(*a,**kw):
        calls.append(1)
        if failure!='incomplete':raise RuntimeError('PRIVATE_PROVIDER_OR_BUDGET')
        return dict(calls=1,failed_rankings=1,pending_reservations=0)
    assert c.campaign(protocol(),tmp_path,dispatch,inspect)==2
    assert (tmp_path/'HALTED').exists() and (tmp_path/'STOPPED').exists()
    before={p.name:p.read_bytes() for p in tmp_path.iterdir()}
    assert c.campaign(protocol(),tmp_path,dispatch,inspect)==2 and len(calls)==1
    assert before=={p.name:p.read_bytes() for p in tmp_path.iterdir()}
    assert 'PRIVATE' not in (tmp_path/'HALTED').read_text()

def pair_fixture():
    requests=[dict(prompt='case'+str(i),candidates=['one','two'],uri='fixture'+str(i)) for i in range(2)]
    version='fixture';gates=[]
    for r in requests:
        jobs=[(o,c.api.judge.key_of(r['prompt'],r['candidates'],o,c.api.profile_id(),version)) for _,o in c.api.judge.jobs([r])]
        gates.append(dict(candidate_case_sha256=c.api.digest(r),planned_orders=[o for o,k in jobs],key=jobs[0][1],paired_unattempted_key=jobs[1][1]))
    return dict(reviewed_failure_exclusions=gates),requests,version

def test_both_failed_sets_are_excluded_without_inspecting_rankings():
    p,r,v=pair_fixture()
    assert len(c.verify_reviewed_pairs(p,r,v,[],set()))==4

@pytest.mark.parametrize('change',['case','key','orders','attempted','ranked','overlap','unknown'])
def test_no_outcome_selection_or_reranking_through_exclusion_change(change):
    p,r,v=pair_fixture();g=p['reviewed_failure_exclusions'][1];history=[];done=set()
    if change=='case':g['candidate_case_sha256']='wrong'
    if change=='key':g['paired_unattempted_key']='foreign'
    if change=='orders':g['planned_orders'].reverse()
    if change=='attempted':history=[dict(event='reserve',key=g['paired_unattempted_key'])]
    if change=='ranked':done={g['paired_unattempted_key']}
    if change=='overlap':p['reviewed_failure_exclusions'][1]=copy.deepcopy(p['reviewed_failure_exclusions'][0])
    if change=='unknown':history=[dict(event='reserve',id='x',key='unknown'),dict(event='retain_unknown_ceiling',id='x')]
    with pytest.raises(ValueError):c.verify_reviewed_pairs(p,r,v,history,done)

@pytest.mark.parametrize('field,value',[('max_new_attempts',3469),('initial_training_attempts',2524),('initial_valid_api_rankings',2569),('max_usd',201),('gate_reserve_usd',24),('reserve_per_call_usd',.26),('minimum_mem_available_bytes',1)])
def test_frozen_limits_cannot_expand(monkeypatch,field,value):
    p=protocol();p.update(kind='FROZEN_FAILSTOP_API_PAIRED_V7_3468',source_sha256={},profile_sha256=c.api.digest(c.api.PROFILE),initial_training_attempts=2525,initial_valid_api_rankings=2568)
    p[field]=value;monkeypatch.setattr(c,'source_hashes',lambda:{})
    with pytest.raises(ValueError,match='limits'):c.inspect(p)

def test_attempt_exhaustion_with_incomplete_data_halts(tmp_path):
    calls=[]
    def inspect(_):return [],dict(attempts_since_start=3468,missing_rankings=5,eligible_missing_rankings=1)
    assert c.campaign(protocol(),tmp_path,lambda *a,**k:calls.append(1),inspect)==2
    assert not calls and (tmp_path/'HALTED').exists()


def test_downstream_dataset_gate_rejects_silent_incompleteness():
    from preference_dataset_gate_v7 import require_ready
    status=dict(valid_api_rankings=6036,missing_rankings=4,eligible_missing_rankings=0,preserved_failed_rankings=2,unattempted_paired_exclusions=2)
    assert require_ready(status)['original_full_valid_complete'] is False
    for key in status:
        bad=dict(status);bad[key]-=1
        with pytest.raises(ValueError):require_ready(bad)
