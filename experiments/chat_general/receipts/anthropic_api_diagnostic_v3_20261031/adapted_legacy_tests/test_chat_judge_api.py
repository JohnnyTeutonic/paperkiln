"""Independent fake-HTTP tests: no model calls or credentials."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chat_judge_api as api  # noqa: E402


@pytest.fixture
def files(tmp_path):
    env = tmp_path / 'designated.env'
    env.write_text('OTHER=must-not-load\nANTHROPIC_API_KEY="test-only-not-a-real-key"\n')
    rubric = tmp_path / 'chat_v1.txt'
    rubric.write_text('Rank the replies. JSON only.\n')
    legacy = tmp_path / 'legacy.jsonl'
    api.append(legacy, {'cost_usd': 20.073})
    return dict(cache=tmp_path/'api.jsonl', legacy_cache=legacy, rubric=rubric,
                ledger=tmp_path/'budget.jsonl', env_file=env)


def requests():
    return [{'prompt': '<|user|>Hello<|assistant|>', 'candidates': ['Hello.', 'Goodbye.'], 'uri': 'probe'}]


def response(**changes):
    obj = {'model': api.MODEL, 'id': 'msg_fixture', 'stop_reason': 'end_turn',
           'usage': {'input_tokens': 1000, 'output_tokens': 600, 'output_tokens_details': {'thinking_tokens': 500}},
           'content': [{'type': 'thinking', 'thinking': ''},
                       {'type': 'text', 'text': '{"ranking":["A","B"],"rationale":"A replies to the greeting."}'}]}
    obj.update(changes)
    return obj


def fake(obj=None, status=200):
    calls = []
    def post(key, body, **_diagnostic_trace):
        calls.append((key, body))
        return status, api.encoded(response() if obj is None else obj)
    post.calls = calls
    return post


def test_hidden_thinking_is_billed_by_total_provider_output():
    assert api.usage_cost(response()) == pytest.approx(.016)
    assert api.usage_cost(response(content=[])) == pytest.approx(.016)


def test_cache_rates_reconcile_without_double_counting():
    u = {'input_tokens': 100, 'output_tokens': 200, 'cache_read_input_tokens': 300,
         'cache_creation_input_tokens': 500,
         'cache_creation': {'ephemeral_5m_input_tokens': 400, 'ephemeral_1h_input_tokens': 100}}
    assert api.usage_cost(response(usage=u)) == pytest.approx(.00726)


@pytest.mark.parametrize('usage', [None, {}, {'input_tokens': 2},
    {'input_tokens': -1, 'output_tokens': 2}, {'input_tokens': True, 'output_tokens': 2},
    {'input_tokens': 1, 'output_tokens': 1025}, {'input_tokens': 1, 'output_tokens': float('nan')},
    {'input_tokens': 1, 'output_tokens': 1, 'speed': 'fast'},
    {'input_tokens': 1, 'output_tokens': 1, 'inference_geo': 'us'},
    {'input_tokens': 1, 'output_tokens': 1, 'server_tool_use': {'web_search_requests': 1}},
    {'input_tokens': 1, 'output_tokens': 1, 'cache_creation_input_tokens': 2,
     'cache_creation': {'ephemeral_1h_input_tokens': 3}}])
def test_unverifiable_pricing_rejected(usage):
    with pytest.raises(ValueError):
        api.usage_cost(response(usage=usage))


def test_exact_returned_model_required():
    with pytest.raises(ValueError, match='model'):
        api.usage_cost(response(model='claude-sonnet-5-5'))


def test_exact_payload_no_tools_or_hidden_system(files):
    post = fake()
    result = api.run(requests(), **files, max_calls=1, post=post)
    assert result['rankings_added'] == 1
    key, body = post.calls[0]
    assert key == 'test-only-not-a-real-key'
    assert body['model'] == 'claude-opus-5-5'
    assert body['system'] == files['rubric'].read_text()
    assert body['messages'] == [{'role': 'user', 'content': api.judge.render(
        requests()[0]['prompt'], requests()[0]['candidates'], [0, 1])}]
    assert 'tools' not in body and 'temperature' not in body
    rec = api.rows(files['cache'])[0]
    assert rec['key'] != rec['semantic_request_key']
    assert rec['judge_id'].startswith('claude-opus-5-5@anthropic-messages/')
    assert rec['transport_receipt']['usage']['output_tokens'] == 600
    for p in files['ledger'].parent.glob('budget.raw/*'):
        assert key.encode() not in p.read_bytes()
    assert api.rows(files['ledger'])[-1]['cost_usd'] == pytest.approx(.016)


def test_resume_skips_own_profile_rankings(files):
    post = fake()
    api.run(requests(), **files, max_calls=2, post=post)
    assert len(post.calls) == 2
    assert api.run(requests(), **files, max_calls=2, post=post)['calls'] == 0
    assert len(post.calls) == 2


def test_cli_cache_verdicts_are_not_api_cache_hits(files):
    _, version, _ = api.judge.rubric_identity(str(files['rubric']))
    req = requests()[0]
    for order in ([0, 1], [1, 0]):
        api.append(files['legacy_cache'], {'key': api.judge.key_of(req['prompt'], req['candidates'], order,
                   api.MODEL, version), 'cost_usd': .01})
    post = fake()
    assert api.run(requests(), **files, max_calls=2, post=post)['calls'] == 2


def test_unknown_historical_failure_reserves_remain(files):
    api.append(str(files['legacy_cache'])+'.failures.jsonl', {'error': 'unknown outcome'})
    baseline = api.legacy_account(files['legacy_cache'])
    # The largest known record here is the whole historical charge fixture.
    assert baseline['unknown_failure_reserve_usd'] == 20.073
    with pytest.raises(ValueError, match='reserve'):
        api.run(requests(), **files, max_calls=1, cap=45, gate_reserve=5, post=fake())


def test_gate_and_prior_spend_admission_before_http(files):
    post = fake()
    with pytest.raises(ValueError, match='reserve'):
        api.run(requests(), **files, max_calls=1, cap=45, post=post)
    assert not post.calls
    assert len(api.rows(files['ledger'])) == 1


def test_repeated_calls_cannot_pass_cumulative_cap(files):
    post = fake()
    api.run(requests(), **files, max_calls=1, cap=45.33, post=post)
    with pytest.raises(ValueError, match='reserve'):
        api.run(requests(), **files, max_calls=1, cap=45.33, post=post)
    assert len(post.calls) == 1
    _, paid = api.Budget(files['ledger'], api.legacy_account(files['legacy_cache']), 45.33, 25).state()
    assert paid == pytest.approx(.016)



@pytest.mark.parametrize('status', [400, 401, 404, 429, 500])
def test_http_errors_record_status_and_keep_reservation(files, status):
    post = fake({'error': {'type': 'fixture_error'}}, status=status)
    with pytest.raises(RuntimeError, match=str(status)):
        api.run(requests(), **files, max_calls=1, post=post)
    pending, paid = api.Budget(files['ledger'], api.legacy_account(files['legacy_cache'])).state()
    assert len(pending) == 1 and paid == 0
    with pytest.raises(ValueError, match='unresolved'):
        api.run(requests(), **files, max_calls=1, post=post)
    assert len(post.calls) == 1


def test_timeout_is_not_free_and_never_retried(files):
    calls = []
    def timeout(key, body):
        calls.append(body)
        raise TimeoutError('do not expose secret transport details')
    with pytest.raises(RuntimeError, match='unknown'):
        api.run(requests(), **files, max_calls=2, post=timeout)
    assert len(calls) == 1
    assert len(api.Budget(files['ledger'], api.legacy_account(files['legacy_cache'])).state()[0]) == 1


@pytest.mark.parametrize('changes', [{'stop_reason': 'max_tokens'},
    {'content': [{'type': 'text', 'text': 'not a ranking'}]},
    {'content': [{'type': 'text', 'text': '[]'}]}])
def test_invalid_rankings_still_settle_bill(files, changes):
    result = api.run(requests(), **files, max_calls=1, post=fake(response(**changes)))
    assert result['failed_rankings'] == 1 and result['rankings_added'] == 0
    assert not api.rows(files['cache'])
    assert api.rows(str(files['cache'])+'.failures.jsonl')[0]['cost_usd'] == pytest.approx(.016)


def test_wrong_model_or_missing_usage_keeps_full_reserve(files):
    with pytest.raises(RuntimeError, match='unverifiable'):
        api.run(requests(), **files, max_calls=1, post=fake(response(model='different-model')))
    pending, paid = api.Budget(files['ledger'], api.legacy_account(files['legacy_cache'])).state()
    assert len(pending) == 1 and paid == 0


def test_changed_legacy_snapshot_blocks_resume(files):
    api.run(requests(), **files, max_calls=1, post=fake())
    api.append(files['legacy_cache'], {'cost_usd': 1})
    with pytest.raises(ValueError, match='snapshot changed'):
        api.run(requests(), **files, max_calls=1, post=fake())


def test_profiles_cannot_silently_mix(files):
    api.Budget(files['ledger'], api.legacy_account(files['legacy_cache']))
    api.append(files['cache'], {'judge_id': api.MODEL, 'key': 'legacy'})
    with pytest.raises(ValueError, match='binding'):
        api.run(requests(), **files, max_calls=1, post=fake())


def test_key_requires_designated_nonempty_name(tmp_path):
    p = tmp_path/'env'
    p.write_text('SOME_OTHER_KEY=not-the-right-key\n')
    with pytest.raises(ValueError, match='missing'):
        api.load_key(p)


def test_lock_prevents_concurrent_campaign_dispatch(tmp_path):
    p = tmp_path/'lock'
    with api.exclusive(p):
        with pytest.raises(BlockingIOError):
            with api.exclusive(p):
                pass

def test_missing_legacy_cache_cannot_zero_prior_spend(files):
    with pytest.raises(ValueError, match='legacy cache'):
        api.legacy_account(files['legacy_cache'].with_name('missing.jsonl'))


def test_probe_is_one_request_under_quarter_dollar(files):
    with pytest.raises(ValueError, match='probe'):
        api.run(requests(), **files, max_calls=2, stage='probe', post=fake())
    with pytest.raises(ValueError, match='probe'):
        api.run(requests(), **files, max_calls=1, stage='probe', reserve=.251, post=fake())


def test_existing_api_cache_cannot_restart_with_empty_budget(files):
    api.append(files['cache'], {'judge_id': api.profile_id(), 'key': 'test',
                             'transport_profile_sha256': api.digest(api.PROFILE)})
    with pytest.raises(ValueError, match='binding'):
        api.run(requests(), **files, max_calls=1, post=fake())


def test_probe_restart_stays_within_total_quarter_dollar(files):
    post = fake()
    api.run(requests(), **files, max_calls=1, stage='probe', post=post)
    with pytest.raises(ValueError, match='stage attempt'):
        api.run(requests(), **files, max_calls=1, stage='probe', post=post)
    assert len(post.calls) == 1


def rewrite(path, records):
    import json
    Path(path).write_text(''.join(json.dumps(r)+'\n' for r in records))


def test_new_existing_empty_ledger_cannot_reset_api_cost(files):
    post = fake()
    api.run(requests(), **files, max_calls=1, post=post)
    other = files['ledger'].with_name('other.jsonl')
    other.touch()
    changed = dict(files, ledger=other)
    with pytest.raises(ValueError, match='binding'):
        api.run(requests(), **changed, max_calls=1, post=post)
    assert len(post.calls) == 1
    assert api.Budget(files['ledger'], api.legacy_account(files['legacy_cache'])).state()[1] == .016


def test_same_ledger_cannot_silently_switch_cache(files):
    post = fake()
    api.run(requests(), **files, max_calls=1, post=post)
    changed = dict(files, cache=files['cache'].with_name('other-cache.jsonl'))
    with pytest.raises(ValueError, match='binding'):
        api.run(requests(), **changed, max_calls=1, post=post)
    assert len(post.calls) == 1


def test_truncated_or_reset_ledger_cannot_dispatch_with_existing_cache(files):
    post = fake()
    api.run(requests(), **files, max_calls=1, post=post)
    files['ledger'].write_text('')
    with pytest.raises(ValueError, match='settled reservation'):
        api.run(requests(), **files, max_calls=1, post=post)
    assert len(post.calls) == 1


def test_deleted_cache_does_not_erase_paid_journal_history(files):
    post = fake()
    api.run(requests(), **files, max_calls=1, post=post)
    files['cache'].unlink()
    with pytest.raises(ValueError, match='missing cache'):
        api.run(requests(), **files, max_calls=1, post=post)
    assert len(post.calls) == 1
    assert api.Budget(files['ledger'], api.legacy_account(files['legacy_cache'])).state()[1] == .016


@pytest.mark.parametrize('field,value', [('cost_usd', 0), ('budget_reservation_id', 'not-reserved'),
    ('transport_profile_sha256', 'different'), ('rubric_sha256', 'different'),
    ('response_sha256', 'different'), ('key', 'different'), ('ranking', [1, 0])])
def test_cached_rows_replay_against_paid_raw_receipts(files, field, value):
    post = fake()
    api.run(requests(), **files, max_calls=1, post=post)
    recs = api.rows(files['cache'])
    recs[0][field] = value
    rewrite(files['cache'], recs)
    with pytest.raises(ValueError):
        api.run(requests(), **files, max_calls=1, post=post)
    assert len(post.calls) == 1


def test_failed_rows_are_also_bound_to_settlements(files):
    post = fake(response(stop_reason='max_tokens'))
    api.run(requests(), **files, max_calls=1, post=post)
    failures = str(files['cache'])+'.failures.jsonl'
    recs = api.rows(failures)
    recs[0]['cost_usd'] = 0
    rewrite(failures, recs)
    with pytest.raises(ValueError, match='cost'):
        api.run(requests(), **files, max_calls=1, post=post)
    assert len(post.calls) == 1


def test_raw_provider_changes_are_detected(files):
    post = fake()
    api.run(requests(), **files, max_calls=1, post=post)
    record = api.rows(files['cache'])[0]
    Path(record['transport_receipt']['response']).write_bytes(api.encoded(response(usage={'input_tokens': 1, 'output_tokens': 1})))
    with pytest.raises(ValueError, match='hash'):
        api.run(requests(), **files, max_calls=1, post=post)
    assert len(post.calls) == 1


def test_campaign_requests_and_rubric_cannot_drift(files):
    post = fake()
    api.run(requests(), **files, max_calls=1, post=post)
    altered = requests()
    altered[0]['candidates'][0] = 'different answer'
    with pytest.raises(ValueError, match='binding'):
        api.run(altered, **files, max_calls=1, post=post)
    files['rubric'].write_text('different rubric')
    with pytest.raises(ValueError, match='binding'):
        api.run(requests(), **files, max_calls=1, post=post)
    assert len(post.calls) == 1


def test_cache_lock_is_shared_even_when_ledger_argument_differs(files):
    with api.exclusive(str(files['cache'])+'.lock'):
        with pytest.raises(BlockingIOError):
            api.run(requests(), **files, max_calls=1, post=fake())
        with pytest.raises(BlockingIOError):
            api.run(requests(), **dict(files, ledger=files['ledger'].with_name('other.jsonl')),
                    max_calls=1, post=fake())


def many_requests():
    return [{'prompt': f'<|user|>Hello {i}<|assistant|>',
             'candidates': ['Hello.', 'Goodbye.'], 'uri': str(i)} for i in range(25)]


def test_calibration_48_attempt_bound_persists_across_restarts(files):
    post = fake()
    result = api.run(many_requests(), **files, max_calls=48, stage='calibration', post=post)
    assert result['calls'] == 48
    with pytest.raises(ValueError, match='stage attempt'):
        api.run(many_requests(), **files, max_calls=48, stage='calibration', post=post)
    assert len(post.calls) == 48


def test_calibration_two_dollar_bound_covers_failures(files):
    post = fake(response(stop_reason='max_tokens', usage={'input_tokens': 45000, 'output_tokens': 1000}))
    with pytest.raises(ValueError, match='stage dollar'):
        api.run(many_requests(), **files, max_calls=48, stage='calibration', post=post)
    assert len(post.calls) == 9
    assert api.Budget(files['ledger'], api.legacy_account(files['legacy_cache'])).state()[1] == pytest.approx(1.8)


def test_prior_api_probe_cost_is_verified_and_imported(files, tmp_path):
    post = fake()
    api.run(requests(), **files, max_calls=1, stage='probe', post=post)
    new = dict(files, cache=tmp_path/'new-cache.jsonl', ledger=tmp_path/'new-budget.jsonl')
    result = api.run(requests(), **new, max_calls=1, stage='calibration',
                     prior_api_ledgers=[files['ledger']], post=post)
    assert result['legacy_spend_and_reserve_usd'] == pytest.approx(20.089)
    assert result['api_actual_spend_usd'] == pytest.approx(.016)
    actual = api.legacy_account(files['legacy_cache'], [files['ledger']])
    assert actual['prior_api_spend_usd'] == pytest.approx(.016)


def test_tampered_prior_api_cost_cannot_be_imported(files):
    api.run(requests(), **files, max_calls=1, post=fake())
    history = api.rows(files['ledger'])
    history[-1]['cost_usd'] = 0
    rewrite(files['ledger'], history)
    with pytest.raises(ValueError, match='replayed'):
        api.legacy_account(files['legacy_cache'], [files['ledger']])


def test_future_heldout_reserve_cannot_be_spent_via_unregistered_stage(files):
    with pytest.raises(ValueError, match='stage'):
        api.run(requests(), **files, max_calls=1, stage='heldout_gate', post=fake())

def test_larger_historical_unknown_reserve_is_not_reduced(files):
    api.append(str(files['legacy_cache'])+'.failures.jsonl', {'reserved_cost_usd': 33})
    assert api.legacy_account(files['legacy_cache'])['unknown_failure_reserve_usd'] == 33


def authorize_unknown(files):
    budget = api.Budget(files['ledger'], api.legacy_account(files['legacy_cache']))
    pending, _ = budget.state()
    ident = next(iter(pending))
    rawdir = files['ledger'].with_suffix('.raw')
    return {'kind': 'ROOT_AUTHORISED_RETAIN_UNKNOWN_CEILING', 'pending_id': ident,
            'ledger_sha256': api.file_hash(files['ledger']), 'binding_sha256': api.digest(budget.binding),
            'raw_request_sha256': api.file_hash(rawdir/(ident+'.request.json')),
            'error_sha256': api.file_hash(rawdir/(ident+'.error.json')), 'error_type': 'TimeoutError',
            'reason': 'Independent fixture approval: retain full unknown ceiling permanently.'}


def uncertain_call(files):
    def timeout(key, body):
        raise TimeoutError('fixture')
    with pytest.raises(RuntimeError):
        api.run(requests(), **files, max_calls=1, stage='calibration', post=timeout)


def test_explicit_unknown_recovery_retains_money_and_attempt(files):
    uncertain_call(files)
    auth = authorize_unknown(files)
    budget = api.Budget(files['ledger'], api.legacy_account(files['legacy_cache']))
    event = budget.retain_unknown(auth)
    assert event['ceiling_usd'] == .25
    assert budget.state() == ({}, 0)
    assert api.rows(files['cache']) == []
    post = fake()
    out = api.run(requests(), **files, max_calls=1, stage='calibration', post=post)
    assert out['api_retained_unknown_ceiling_usd'] == .25
    assert out['api_actual_spend_usd'] == .016
    assert len([r for r in api.rows(files['ledger']) if r['event'] == 'reserve']) == 2


def test_retained_ceiling_still_blocks_cap(files):
    uncertain_call(files)
    budget = api.Budget(files['ledger'], api.legacy_account(files['legacy_cache']))
    budget.retain_unknown(authorize_unknown(files))
    # Global contract still includes the retained amount in the available funds.
    budget.cap = 45.4
    with pytest.raises(ValueError, match='cumulative'):
        budget.reserve('another', .25, 'calibration')


def test_unknown_recovery_cannot_double_resolve_or_settle_actual(files):
    uncertain_call(files)
    auth = authorize_unknown(files)
    budget = api.Budget(files['ledger'], api.legacy_account(files['legacy_cache']))
    budget.retain_unknown(auth)
    with pytest.raises(ValueError):
        budget.retain_unknown(auth)
    with pytest.raises(ValueError):
        budget.settle(auth['pending_id'], 0, {})


@pytest.mark.parametrize('field,value', [('raw_request_sha256', 'wrong'), ('error_sha256', 'wrong'),
    ('error_type', 'wrong'), ('pending_id', 'wrong'), ('ledger_sha256', 'wrong'), ('kind', 'wrong')])
def test_unknown_recovery_requires_exact_authorised_evidence(files, field, value):
    uncertain_call(files)
    auth = authorize_unknown(files)
    auth[field] = value
    budget = api.Budget(files['ledger'], api.legacy_account(files['legacy_cache']))
    with pytest.raises((ValueError, KeyError)):
        budget.retain_unknown(auth)
    assert len(budget.state()[0]) == 1


def test_retained_ceiling_tamper_is_detected_on_resume(files):
    uncertain_call(files)
    budget = api.Budget(files['ledger'], api.legacy_account(files['legacy_cache']))
    budget.retain_unknown(authorize_unknown(files))
    history = api.rows(files['ledger'])
    history[-1]['ceiling_usd'] = 0
    rewrite(files['ledger'], history)
    with pytest.raises(ValueError, match='retained unknown'):
        api.run(requests(), **files, max_calls=1, post=fake())


def test_retained_unknown_is_included_in_new_audited_baseline(files, tmp_path):
    uncertain_call(files)
    budget = api.Budget(files['ledger'], api.legacy_account(files['legacy_cache']))
    budget.retain_unknown(authorize_unknown(files))
    baseline = api.legacy_account(files['legacy_cache'], [files['ledger']])
    assert baseline['prior_api_unknown_reserve_usd'] == .25
    assert baseline['prior_api_spend_usd'] == 0