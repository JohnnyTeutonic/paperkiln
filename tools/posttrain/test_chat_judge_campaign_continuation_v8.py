"""Offline tests only. Provider dispatch and credentials are always replaced."""
import copy
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent))
import chat_judge_campaign_continuation_v8 as c

@pytest.fixture(autouse=True)
def headroom(monkeypatch):
    monkeypatch.setattr(c, "memory_available_bytes", lambda: 2 * 1024**3)

def protocol():
    return {"max_new_attempts":3259,"batch_calls":24,"minimum_mem_available_bytes":1024**3,
       "cache":"unused","legacy_cache":"unused","rubric":"unused","ledger":"unused","env_file":"unused",
       "reserve_per_call_usd":.25,"max_usd":200,"gate_reserve_usd":25,"prior_api_ledgers":[],
       "fresh_dispatch_gate":{"key":c.FRESH_KEY},"reviewed_failure_exclusions":[{"key":"f1","paired_unattempted_key":"p1"},{"key":"f2","paired_unattempted_key":"p2"}]}

def test_restart_only_dispatches_remaining_and_never_claims_full_completeness(tmp_path):
    p=protocol();calls=[]
    def inspect(_):
        return [],dict(explicit_fresh_dispatch_remaining=not calls,attempts_since_start=len(calls),missing_rankings=3263-len(calls),
                       eligible_missing_rankings=3259-len(calls),valid_api_rankings=2777+len(calls))
    def interrupt(*args,**kw):
        calls.extend(range(3));raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):c.campaign(p,tmp_path,interrupt,inspect)
    def dispatch(*args,**kw):
        assert kw['excluded_failed_keys']==('f1','f2')
        assert kw['excluded_unattempted_keys']==('p1','p2')
        assert args[6]==(1 if not calls else min(24,3259-len(calls)))
        n=args[6];calls.extend(range(n));return dict(calls=n,failed_rankings=0,pending_reservations=0)
    assert c.campaign(p,tmp_path,dispatch,inspect)==0
    stopped=json.loads((tmp_path/'STOPPED').read_text())
    assert len(calls)==3259 and stopped['valid_api_rankings']==6036
    assert stopped['missing_rankings']==4 and stopped['eligible_judging_complete']
    assert stopped['full_judging_complete'] is False and not (tmp_path/'COMPLETE').exists()
    assert c.campaign(p,tmp_path,dispatch,inspect)==0 and len(calls)==3259

@pytest.mark.parametrize('failure',['unknown','incomplete','budget'])
def test_new_problem_halts_and_never_retries(tmp_path,failure):
    calls=[]
    def inspect(_):return [],dict(explicit_fresh_dispatch_remaining=True,attempts_since_start=0,missing_rankings=5,eligible_missing_rankings=1)
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
    return dict(fresh_dispatch_gate={"key":c.FRESH_KEY},reviewed_failure_exclusions=gates),requests,version

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

@pytest.mark.parametrize('field,value',[('max_new_attempts',3260),('initial_training_attempts',2734),('initial_valid_api_rankings',2778),('max_usd',201),('gate_reserve_usd',24),('reserve_per_call_usd',.26),('minimum_mem_available_bytes',1)])
def test_frozen_limits_cannot_expand(monkeypatch,field,value):
    p=protocol();p.update(kind='FROZEN_EXPLICIT_UNTRANSMITTED_API_V8_3259',source_sha256={},profile_sha256=c.api.digest(c.api.PROFILE),initial_training_attempts=2735,initial_valid_api_rankings=2777)
    p[field]=value;monkeypatch.setattr(c,'source_hashes',lambda:{})
    with pytest.raises(ValueError,match='limits'):c.inspect(p)

def test_attempt_exhaustion_with_incomplete_data_halts(tmp_path):
    calls=[]
    def inspect(_):return [],dict(attempts_since_start=3259,missing_rankings=5,eligible_missing_rankings=1)
    assert c.campaign(protocol(),tmp_path,lambda *a,**k:calls.append(1),inspect)==2
    assert not calls and (tmp_path/'HALTED').exists()


def test_downstream_dataset_gate_rejects_silent_incompleteness():
    from preference_dataset_gate_v7 import require_ready
    status=dict(valid_api_rankings=6036,missing_rankings=4,eligible_missing_rankings=0,preserved_failed_rankings=2,unattempted_paired_exclusions=2)
    assert require_ready(status)['original_full_valid_complete'] is False
    for key in status:
        bad=dict(status);bad[key]-=1
        with pytest.raises(ValueError):require_ready(bad)


def trace_fixture():
    address=dict(family=2,ip='192.0.2.1',port=443)
    trace=dict(schema=1,stage='connection_or_tls_failed',http_write_started=False,
      binding=dict(reservation_id=c.PENDING_ID,request_sha256='a'*64),
      connect_failure=dict(type='EndpointConnectionError',errno=101),dns=[address],
      attempts=[dict(address,failure=dict(type='OSError',errno=101))])
    error=dict(error_type='URLError',reason_type='EndpointConnectionError',errno=101,charge='unknown_reservation_retained')
    reservation=dict(event='reserve',id=c.PENDING_ID,key=c.FRESH_KEY,reserve_usd=.25)
    return trace,error,reservation


def test_exact_proven_zero_http_write_allows_review_classification_only():
    trace,error,reservation=trace_fixture()
    assert c.prove_untransmitted(trace,error,reservation,'a'*64) is None


@pytest.mark.parametrize('change',['write','missing_write','stage','request','reservation','key','charge','errno','empty_dns','missing_attempt','attempt_errno','address','reason','reserve'])
def test_proof_rejects_posted_unknown_or_changed_transport(change):
    trace,error,reservation=trace_fixture()
    if change=='write':trace['http_write_started']=True
    if change=='missing_write':del trace['http_write_started']
    if change=='stage':trace['stage']='response_read'
    if change=='request':trace['binding']['request_sha256']='b'*64
    if change=='reservation':reservation['id']='different'
    if change=='key':reservation['key']='different'
    if change=='charge':error['charge']='zero'
    if change=='errno':trace['connect_failure']['errno']=110
    if change=='empty_dns':trace['dns']=[]
    if change=='missing_attempt':trace['attempts']=[]
    if change=='attempt_errno':trace['attempts'][0]['failure']['errno']=110
    if change=='address':trace['attempts'][0]['ip']='192.0.2.2'
    if change=='reason':error['reason_type']='TimeoutError'
    if change=='reserve':reservation['reserve_usd']=0
    with pytest.raises(ValueError):c.prove_untransmitted(trace,error,reservation,'a'*64)


def history_fixture():
    _,_,reservation=trace_fixture()
    return [reservation,dict(event='retain_unknown_ceiling',id=c.PENDING_ID,ceiling_usd=.25)]


def test_retained_unknown_and_exactly_one_fresh_settlement():
    h=history_fixture();assert c.fresh_history(h,[]) is True
    h.extend([dict(event='reserve',id='fresh',key=c.FRESH_KEY),dict(event='settle',id='fresh',cost_usd=.01)])
    assert c.fresh_history(h,[dict(key=c.FRESH_KEY,budget_reservation_id='fresh')]) is False
    # Full ceiling is retained alongside fresh actual usage, never called zero.
    assert sum(r.get('ceiling_usd',0)+r.get('cost_usd',0) for r in h)==.26


@pytest.mark.parametrize('change',['third','pending','failure','wrong_accept','missing_retention','reduced_retention','old_response'])
def test_exception_is_never_an_automatic_retry_or_release(change):
    h=history_fixture();rows=[]
    if change in ['third','pending','failure','wrong_accept']:
        h.append(dict(event='reserve',id='fresh',key=c.FRESH_KEY))
        if change!='pending':h.append(dict(event='settle',id='fresh',cost_usd=.01))
        if change not in ['pending','failure']:rows=[dict(key=c.FRESH_KEY,budget_reservation_id='fresh')]
    if change=='third':h.append(dict(event='reserve',id='third',key=c.FRESH_KEY))
    if change=='wrong_accept':rows[0]['budget_reservation_id']=c.PENDING_ID
    if change=='missing_retention':h.pop()
    if change=='reduced_retention':h[-1]['ceiling_usd']=0
    if change=='old_response':rows=[dict(key=c.FRESH_KEY,budget_reservation_id=c.PENDING_ID)]
    with pytest.raises(ValueError):c.fresh_history(h,rows)


def test_first_fresh_dispatch_is_a_single_isolated_call(tmp_path):
    calls=[]
    def inspector(_):return [],dict(explicit_fresh_dispatch_remaining=not calls,attempts_since_start=len(calls),missing_rankings=5-len(calls),eligible_missing_rankings=1-len(calls))
    def dispatch(*args,**kw):
        assert args[6]==1;calls.append(1);return dict(calls=1,failed_rankings=0,pending_reservations=0)
    assert c.campaign(protocol(),tmp_path,dispatch,inspector)==0 and len(calls)==1


@pytest.mark.parametrize('change',['permission','key','request','trace','source','count','cohort','mate','future','ceiling'])
def test_exact_root_authorization_cannot_expand(change):
    p={'fresh_dispatch_gate':{'request':{'sha256':'a'*64},'transport':{'sha256':'b'*64},'pre_disposition_ledger_sha256':'c'*64,'retention_authorization':{'sha256':'d'*64}}}
    auth=c.fresh_authorization_expected(p)
    c.verify_fresh_authorization(auth,p)
    fields={'permission':('authorised',False),'key':('key','other'),'request':('request_sha256','other'),'trace':('transport_sha256','other'),'source':('source_sha256',{}),'count':('max_fresh_dispatches',2),'cohort':('eligible_target_rankings',6034),'mate':('preserve_accepted_mate',False),'future':('future_unknown_auto_retry',True),'ceiling':('unknown_ceiling_retained_usd',0)}
    key,value=fields[change];auth[key]=value
    with pytest.raises(ValueError):c.verify_fresh_authorization(auth,p)


@pytest.mark.parametrize('field',['kind','pending_id','key','accepted_mate_key','candidate_case_sha256'])
def test_protocol_cannot_select_a_different_fresh_exception(field):
    g=dict(kind='ONE_SOURCE_BOUND_PROVEN_UNTRANSMITTED_EXCEPTION_V8',pending_id=c.PENDING_ID,key=c.FRESH_KEY,accepted_mate_key=c.MATE_KEY,candidate_case_sha256=c.CASE_SHA)
    g[field]='different'
    with pytest.raises(ValueError,match='gate identity'):c.verify_fresh_gate({'fresh_dispatch_gate':g})
