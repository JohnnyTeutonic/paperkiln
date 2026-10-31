"""Offline prospective held-out migration boundaries; no keys or provider calls."""
from pathlib import Path
import sys
import inspect
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import chat_judge_api_heldout_v1 as gate
import chat_judge_api_refusal_v5 as old

def test_provider_profile_payload_and_accounting_implementation_unchanged():
    assert gate.PROFILE==old.PROFILE and gate.profile_id()==old.profile_id()
    for name in ['usage_cost','request_body','http_post','load_key','legacy_account','reconcile','verify_retained']:
        assert inspect.getsource(getattr(gate,name))==inspect.getsource(getattr(old,name))
    assert gate.judge.parse_ranking is old.judge.parse_ranking

def budget(tmp_path,legacy=0,spent=0):
    binding={'stage_limits':{'heldout_gate':{'attempts':600,'usd':25}}}
    b=gate.Budget(tmp_path/'heldout.jsonl',{'known_spend_usd':legacy},200,0,binding)
    if spent:
        ident=b.reserve('previous',.25,'heldout_gate');b.settle(ident,spent,{})
    return b

@pytest.mark.parametrize('stage',['training_judge','probe','calibration'])
def test_training_probe_and_calibration_stages_cannot_consume_gate(tmp_path,stage):
    with pytest.raises(ValueError,match='stage'):budget(tmp_path).reserve('key',.25,stage)

def test_gate_total_allowance_cannot_expand(tmp_path):
    b=budget(tmp_path)
    # Historical paid records here are synthetic accounting records, not invented provider evidence.
    for i in range(100):
        ident=b.reserve(str(i),.25,'heldout_gate');b.settle(ident,.25,{})
    with pytest.raises(ValueError,match='dollar'):b.reserve('over',.25,'heldout_gate')

def test_original_cumulative_cap_includes_all_legacy_costs(tmp_path):
    with pytest.raises(ValueError,match='cumulative'):budget(tmp_path,legacy=199.9).reserve('over',.25,'heldout_gate')

def test_unknown_dispatch_reservation_blocks_any_next_call(tmp_path):
    b=budget(tmp_path);b.reserve('first',.25,'heldout_gate')
    with pytest.raises(ValueError,match='unresolved'):b.reserve('second',.25,'heldout_gate')

@pytest.mark.parametrize('cap,reserve',[(201,0),(200,25),(199,0)])
def test_only_exact_gate_cap_and_consumed_reserve_contract(tmp_path,cap,reserve):
    with pytest.raises(ValueError):gate.Budget(tmp_path/'x',{},cap,reserve,{})
