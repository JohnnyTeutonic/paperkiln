import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import types

import pytest

SPEC = importlib.util.spec_from_file_location('financial_guard', Path(__file__).resolve().parents[1] / 'tools/colab_chat/sfull_financial_continuation_v1.py')
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


@pytest.fixture
def sample():
    now = time.time()
    ledger = {'pods': [{'id': 'p', 'start': now - 100, 'stop': now - 1, 'usd_per_hr': .59}],
              'volumes': [{'id': 'working', 'created': now - 100, 'gb': 80}]}
    state = {'jobs': {'pt': {'home': False, 'status': 'RUNNING'}, 'gate': {'home': True, 'status': 'FINISHED'}}, 'runpod': {'pod': None, 'volume': {'id': 'working', 'dc': 'dc'}, 'ledger': ledger}, 'endpoint': None}
    r = {'predecessor_start_ticks': 123, 'working_volume_id': 'working', 'working_volume_dc': 'dc', 'protected_original_volume_id': 'original', 'expected_jobs': ['pt', 'gate'], 'completed_preflight_jobs': ['gate'], 'minimum_home_step': 675}
    log = 'x reached --max-usd 7; stopping\nx session stopped; runner exiting\n'
    return r, state, log


def test_alive_defers_without_resource_or_cost_gate(sample):
    r, state, _ = sample
    assert guard.predecessor_gate(r, state, '', 123, [{'id': 'live'}], 0) == 'DEFER_PREDECESSOR_ALIVE'


def test_budget_boundary_initializes(sample):
    r, state, log = sample
    assert guard.predecessor_gate(r, state, log, None, [], 7.01) == 'INITIALIZE'


@pytest.mark.parametrize('change', ['pod', 'endpoint', 'creating', 'open_ledger', 'volume', 'dc', 'deleted', 'protected'])
def test_nonterminal_or_wrong_volume_refused(sample, change):
    r, s, log = sample
    if change == 'pod': s['runpod']['pod'] = {'id': 'live'}
    elif change == 'endpoint': s['endpoint'] = 'live'
    elif change == 'creating': s['runpod']['creating'] = time.time()
    elif change == 'open_ledger': s['runpod']['ledger']['pods'][0]['stop'] = None
    elif change == 'volume': s['runpod']['volume']['id'] = 'other'
    elif change == 'dc': s['runpod']['volume']['dc'] = 'other'
    elif change == 'deleted': s['runpod']['volume']['deleted'] = time.time()
    else: r['protected_original_volume_id'] = 'working'
    with pytest.raises(guard.Refused): guard.predecessor_gate(r, s, log, None, [], 7.01)


@pytest.mark.parametrize('spent', [0, 6.99, 12, 100, float('nan'), float('inf')])
def test_financial_window_is_bounded(sample, spent):
    r, s, log = sample
    with pytest.raises(guard.Refused): guard.predecessor_gate(r, s, log, None, [], spent)


@pytest.mark.parametrize('log', ['', 'reached --max-usd 7; stopping', 'session stopped; runner exiting', 'reached --max-usd 7; stopping\nTraceback\nsession stopped; runner exiting', 'reached --max-usd 7; stopping\nHALTED\nsession stopped; runner exiting'])
def test_only_clean_budget_exit_can_initialize(sample, log):
    r, s, _ = sample
    with pytest.raises(guard.Refused): guard.predecessor_gate(r, s, log, None, [], 7.01)


def test_foreign_provider_pod_refused(sample):
    r, s, log = sample
    with pytest.raises(guard.Refused): guard.predecessor_gate(r, s, log, None, [{'id': 'foreign'}], 7.01)


def test_completed_campaign_skips_only_with_coherent_home(sample):
    r, s, log = sample
    s['campaign_home'] = True
    with pytest.raises(guard.Refused): guard.predecessor_gate(r, s, log, None, [], 7.01)
    s['jobs']['pt']['home'] = True
    assert guard.predecessor_gate(r, s, log, None, [], 7.01) == 'ALREADY_HOME'
    with pytest.raises(guard.Refused): guard.predecessor_gate(r, s, log, None, [{'id': 'orphan'}], 7.01)


@pytest.mark.parametrize('change', ['failed', 'missing', 'preflight'])
def test_budget_stop_never_waives_operational_failure(sample, change):
    r, s, log = sample
    if change == 'failed': s['jobs']['pt']['status'] = 'FAILED'
    elif change == 'missing': s['jobs'].pop('gate')
    else: s['jobs']['gate']['home'] = False
    with pytest.raises(guard.Refused): guard.predecessor_gate(r, s, log, None, [], 7.01)


def test_ledger_allows_monotonic_lifecycle_but_keeps_costs(sample):
    _, s, _ = sample
    before = s['runpod']['ledger']
    after = copy.deepcopy(before)
    after['volumes'][0]['deleted'] = time.time()
    after['pods'].append({'id': 'new', 'start': time.time(), 'usd_per_hr': .59})
    guard.retain_ledger(before, after)


@pytest.mark.parametrize('change', ['remove', 'price', 'start', 'stop', 'gb', 'duplicate', 'negative', 'nonfinite', 'future_delete'])
def test_ledger_mutations_refused(sample, change):
    _, s, _ = sample
    before = s['runpod']['ledger']; after = copy.deepcopy(before)
    if change == 'remove': after['pods'] = []
    elif change == 'price': after['pods'][0]['usd_per_hr'] = 0
    elif change == 'start': after['pods'][0]['start'] += 1
    elif change == 'stop': after['pods'][0]['stop'] += .1
    elif change == 'gb': after['volumes'][0]['gb'] = 1
    elif change == 'duplicate': after['pods'].append(copy.deepcopy(after['pods'][0]))
    elif change == 'negative': after['pods'].append({'id': 'bad', 'usd_per_hr': -.59, 'start': time.time()})
    elif change == 'nonfinite': after['volumes'][0]['gb'] = float('nan')
    else: after['volumes'][0]['deleted'] = time.time() + 100
    with pytest.raises(guard.Refused): guard.retain_ledger(before, after)


@pytest.fixture
def execution(tmp_path, sample, monkeypatch):
    r, state, log = sample
    state_dir = tmp_path / 'state'; state_dir.mkdir()
    (state_dir / 'state.json').write_text(json.dumps(state))
    (state_dir / 'runner.log').write_text(log)
    for name in ['old_admission', 'old_guard', 'old_reservation']:
        (tmp_path / name).write_text(name)
    r.update({'predecessor_pid': 999999, 'state_dir': str(state_dir), 'original_admission': str(tmp_path / 'old_admission'),
              'original_guard': str(tmp_path / 'old_guard'), 'original_reservation': str(tmp_path / 'old_reservation'),
              'initial_ledger': str(tmp_path / 'initial_ledger.json'), 'runner_argv': ['python3', '/source/runner_multi.py'], 'deadline_utc': '2099-01-01T00:00:00Z'})
    (tmp_path / 'initial_ledger.json').write_text(json.dumps(state['runpod']['ledger']))
    review = tmp_path / 'admission.json'; review.write_text(json.dumps(r)); digest = guard.sha(review)
    monkeypatch.setattr(sys, 'argv', ['guard', '--review', str(review), '--review-sha256', digest, '--', '/source/runner_multi.py'])
    monkeypatch.setattr(guard, 'configuration', lambda *a: (r, {'validate_checkpoint': lambda old: 675}, {'source_checkout': '/source'}))
    monkeypatch.setattr(guard, 'process_identity', lambda pid: None)
    api = types.SimpleNamespace(list_pods=lambda: [], ledger_usd=lambda ledger: 7.1)
    monkeypatch.setitem(sys.modules, 'runpod_api', api)
    calls = []
    monkeypatch.setattr(guard.runpy, 'run_path', lambda *a, **k: calls.append((a, k)))
    return tmp_path, r, state, calls, api


def test_initialization_then_restart_retains_snapshot(execution):
    root, r, state, calls, _ = execution
    guard.main()
    anchor = (root / 'terminal_anchor.json').read_bytes()
    old = (root / 'terminal_v4/state.json').read_bytes()
    assert len(calls) == 1
    guard.main()
    assert len(calls) == 2
    assert (root / 'terminal_anchor.json').read_bytes() == anchor
    assert (root / 'terminal_v4/state.json').read_bytes() == old


def test_early_invocation_never_initializes_or_dispatches(execution, monkeypatch):
    root, _, _, calls, _ = execution
    monkeypatch.setattr(guard, 'process_identity', lambda pid: 123)
    guard.main()
    assert not calls and not (root / 'terminal_v4').exists()
    assert not (root / 'STOPPED').exists()


@pytest.mark.parametrize('change', ['anchor', 'snapshot', 'cost_drop', 'foreign', 'exhausted'])
def test_initialized_restarts_fail_closed(execution, change):
    root, r, state, calls, api = execution
    guard.main()
    if change == 'anchor': (root / 'terminal_anchor.json').write_text('{}')
    elif change == 'snapshot': (root / 'terminal_v4/state.json').write_text('{}')
    elif change == 'cost_drop':
        state['runpod']['ledger']['pods'][0]['usd_per_hr'] = 0
        (Path(r['state_dir']) / 'state.json').write_text(json.dumps(state))
    elif change == 'foreign': api.list_pods = lambda: [{'id': 'foreign'}]
    else: api.ledger_usd = lambda ledger: 12
    with pytest.raises(guard.Refused): guard.main()
    assert len(calls) == 1


def test_interrupted_snapshot_cannot_be_overwritten(execution):
    root, _, _, calls, _ = execution
    (root / 'terminal_v4.pending').mkdir()
    with pytest.raises(guard.Refused): guard.main()
    assert not calls


@pytest.mark.parametrize('pid', [True, 0, -1, '1'])
def test_process_identity_rejects_invalid_pid(pid):
    with pytest.raises(guard.Refused): guard.process_identity(pid)


def test_actual_process_identity_and_dead_pid():
    assert type(guard.process_identity(os.getpid())) is int
    assert guard.process_identity(999999999) is None


@pytest.fixture
def configured(tmp_path):
    state_dir = tmp_path / 'runtime'; state_dir.mkdir()
    (state_dir / 'state.json').write_text('{}')
    old_guard = tmp_path / 'old_guard.py'
    old_guard.write_text('def validate(*args):\n    return None\n')
    reservation = tmp_path / 'reservation.json'; reservation.write_text('{}')
    old_path = tmp_path / 'old_admission.json'
    old = {'runner_argv': ['python3', '/source/runner_multi.py', '--max-usd', '7'],
           'state_dir': str(state_dir), 'deadline_utc': '2099-01-01T00:00:00Z',
           'reservation_path': str(reservation), 'numerical_commit': 'numerical'}
    old_path.write_text(json.dumps(old))
    financial = tmp_path / 'financial.json'
    financial.write_text(json.dumps({'per_seed_cumulative_cap_usd': 12, 'three_seed_planning_reservation_usd': 36,
                                    'seeds2_and3_admitted': False, 'seed': 1, 'numerical_commit': 'numerical',
                                    'authority': 'Explicit human authorisation', 'adopted_at_utc': '2026-11-02T00:00:00Z',
                                    'status': 'ADOPTED_FOR_SEED1_PENDING_SAFE_CAP_BOUNDARY_TRANSITION'}))
    initial = tmp_path / 'ledger.json'; initial.write_text('{"pods": [], "volumes": []}')
    paths = [old_guard, old_path, financial, initial, reservation, Path(guard.__file__).resolve()]
    wanted = copy.deepcopy(old['runner_argv']); wanted[-1] = '12'
    r = {'kind': 'SFULL_FINANCIAL_CONTINUATION_V5', 'status': 'ADMITTED', 'seed': 1,
         'per_seed_cap_usd': 12, 'aggregate_cap_usd': 36, 'seeds2_and3_admitted': False,
         'state_dir': str(state_dir), 'deadline_utc': old['deadline_utc'], 'runner_argv': wanted,
         'original_admission': str(old_path), 'original_guard': str(old_guard),
         'original_reservation': str(reservation), 'financial_amendment': str(financial),
         'initial_ledger': str(initial), 'pins': {str(p): guard.sha(p) for p in paths}}
    review = tmp_path / 'review.json'
    return r, review


def invoke_configuration(r, review):
    review.write_text(json.dumps(r))
    return guard.configuration(review, guard.sha(review), r['runner_argv'][1:])


def test_configuration_validates_all_frozen_anchors(configured):
    r, review = configured
    assert invoke_configuration(r, review)[0] == r


@pytest.mark.parametrize('key', ['original_admission', 'original_guard', 'original_reservation', 'financial_amendment', 'initial_ledger', 'self'])
def test_configuration_requires_every_anchor_pin(configured, key):
    r, review = configured
    path = str(Path(guard.__file__).resolve()) if key == 'self' else r[key]
    r['pins'].pop(path)
    with pytest.raises(guard.Refused): invoke_configuration(r, review)


@pytest.mark.parametrize('key', ['original_guard', 'original_reservation', 'financial_amendment', 'initial_ledger'])
def test_configuration_rejects_anchor_drift(configured, key):
    r, review = configured
    Path(r[key]).write_text('changed')
    with pytest.raises(guard.Refused): invoke_configuration(r, review)


@pytest.mark.parametrize('change', ['cap', 'aggregate', 'future_seed', 'args', 'deadline', 'state', 'halt', 'review_digest'])
def test_configuration_refuses_unreviewed_semantic_changes(configured, change):
    r, review = configured
    if change == 'cap': r['per_seed_cap_usd'] = 13
    elif change == 'aggregate': r['aggregate_cap_usd'] = 37
    elif change == 'future_seed': r['seeds2_and3_admitted'] = True
    elif change == 'args': r['runner_argv'].extend(['--max-hours', '99'])
    elif change == 'deadline': r['deadline_utc'] = '2099-01-02T00:00:00Z'
    elif change == 'state': r['state_dir'] += '-other'
    elif change == 'halt': (Path(r['state_dir']) / 'HALTED').write_text('historical')
    else:
        review.write_text(json.dumps(r))
        with pytest.raises(guard.Refused): guard.configuration(review, '0' * 64, r['runner_argv'][1:])
        return
    with pytest.raises((guard.Refused, FileNotFoundError)): invoke_configuration(r, review)


@pytest.mark.parametrize('kind,end', [('pods', 0), ('pods', float('nan')), ('volumes', 0), ('volumes', float('inf'))])
def test_new_resource_closures_cannot_erase_spend(sample, kind, end):
    _, s, _ = sample; before = s['runpod']['ledger']; after = copy.deepcopy(before)
    row = {'id': 'new', 'start': 1, 'stop': end, 'usd_per_hr': .59} if kind == 'pods' else {'id': 'new', 'created': 1, 'deleted': end, 'gb': 80}
    after[kind].append(row)
    with pytest.raises(guard.Refused): guard.retain_ledger(before, after)


@pytest.mark.parametrize('change', ['pod', 'endpoint', 'creating', 'volume', 'nonfinite_cost'])
def test_restart_requires_stopped_owned_resources(execution, change):
    root, r, state, calls, api = execution
    guard.main()
    if change == 'pod': state['runpod']['pod'] = {'id': 'orphan'}
    elif change == 'endpoint': state['endpoint'] = 'orphan'
    elif change == 'creating': state['runpod']['creating'] = time.time()
    elif change == 'volume': state['runpod']['volume']['id'] = 'other'
    else: api.ledger_usd = lambda ledger: float('nan')
    (Path(r['state_dir']) / 'state.json').write_text(json.dumps(state))
    with pytest.raises(guard.Refused): guard.main()
    assert len(calls) == 1


def test_expired_deadline_cannot_defer_forever(execution, monkeypatch):
    root, r, _, calls, _ = execution
    r['deadline_utc'] = '2000-01-01T00:00:00Z'
    path = root / 'admission.json'; path.write_text(json.dumps(r))
    monkeypatch.setattr(sys, 'argv', ['guard', '--review', str(path), '--review-sha256', guard.sha(path), '--', '/source/runner_multi.py'])
    monkeypatch.setattr(guard, 'process_identity', lambda pid: 123)
    with pytest.raises(guard.Refused): guard.main()
    assert not calls


def test_initial_checkpoint_cannot_regress(execution):
    root, r, _, calls, _ = execution
    r['minimum_home_step'] = 676
    with pytest.raises(guard.Refused): guard.main()
    assert not calls and not (root / 'terminal_v4').exists()
