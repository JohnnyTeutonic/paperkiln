"""Prospective frozen gate thresholds and completeness boundaries, no model results."""
import copy
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import assess_preference_gates_v1 as a

def metrics():
    return dict(protocol_failure_rates=dict(frozen=.1,everyday=.1,ultrachat=.1),repeat_rates={s:dict(reference=.2,dpo=.2) for s in ['frozen','ultrachat']},qa_correct=dict(reference=30,dpo=27),parity={leg:{k:40 for k in ['n','reply_identical','token_count','stop_reason','prompt_tokens']} for leg in ['cuda','cpu']},judged_gate=dict(unranked=0,b_win_share=.6,b_win_share_wilson95=[.51,.7]))

def test_all_five_frozen_gates_required_and_never_auto_replace():
    v=a.assess(metrics());assert v['adoption_eligible'] and not v['automatic_model_replacement']

@pytest.mark.parametrize('gate',['protocol','repeats','simple_qa','ember_parity','judged_side_by_side'])
def test_one_failed_gate_preserves_negative(gate):
    m=metrics()
    if gate=='protocol':m['protocol_failure_rates']['everyday']=.1001
    if gate=='repeats':m['repeat_rates']['ultrachat']['dpo']=.2001
    if gate=='simple_qa':m['qa_correct']['dpo']=26
    if gate=='ember_parity':m['parity']['cpu']['reply_identical']=39
    if gate=='judged_side_by_side':m['judged_gate']['b_win_share_wilson95'][0]=.5
    v=a.assess(m);assert not v['adoption_eligible'] and v['gates'][gate] is False

@pytest.mark.parametrize('missing',['protocol','repeat','parity','unranked'])
def test_incomplete_evidence_is_not_a_passing_gate(missing):
    m=metrics()
    if missing=='protocol':del m['protocol_failure_rates']['everyday']
    if missing=='repeat':del m['repeat_rates']['frozen']
    if missing=='parity':del m['parity']['cuda']
    if missing=='unranked':m['judged_gate']['unranked']=1
    with pytest.raises(ValueError):a.assess(m)

def test_context_eligibility_is_replayed_and_no_missing_or_duplicate_sample():
    prompts=[{'prompt':'short'},{'prompt':'long'}]
    encode=lambda text:list(range(300 if 'long' in text else 4))
    a.coverage([{'prompt_index':0}],prompts,encode)
    for rows in [[],[{'prompt_index':0},{'prompt_index':0}],[{'prompt_index':1}]]:
        with pytest.raises(ValueError):a.coverage(rows,prompts,encode)


def parity_rows():
    prompts=[{'prompt':'registered '+str(i)} for i in range(40)]
    cuda=[dict(i=i,prompt=p['prompt'],ember='reply '+str(i),mtstudio='reply '+str(i),ember_tokens=3,mts_tokens=3,ember_stop='eos',mts_stop='eos',ember_prompt_tokens=5,bpe_py_prompt_tokens=5) for i,p in enumerate(prompts)]
    cpu=[dict(r,ember_cuda=r['ember']) for r in cuda]
    return cuda,cpu,prompts


def test_registered_raw_parity_inventory_passes():
    cuda,cpu,prompts=parity_rows()
    assert a.parity_coverage(cuda,cpu,prompts)==metrics()['parity']


@pytest.mark.parametrize('defect',['duplicate','missing','reorder','substitute','crossleg_reference','crossleg_cuda','wrong_index','missing_field','bool_token'])
@pytest.mark.parametrize('leg',['cuda','cpu'])
def test_raw_parity_coverage_rejects_evidence_substitution(defect,leg):
    cuda,cpu,prompts=parity_rows();rows=cuda if leg=='cuda' else cpu
    if defect=='duplicate':rows[:]=[copy.deepcopy(rows[0]) for _ in range(40)]
    if defect=='missing':rows.pop()
    if defect=='reorder':rows[0],rows[1]=rows[1],rows[0]
    if defect=='substitute':rows[0]['prompt']='unregistered'
    if defect=='crossleg_reference':rows[0]['mtstudio']='different reference'
    if defect=='crossleg_cuda':cpu[0]['ember_cuda']='different CUDA'
    if defect=='wrong_index':rows[0]['i']=True
    if defect=='missing_field':del rows[0]['mts_stop']
    if defect=='bool_token':rows[0]['mts_tokens']=True
    with pytest.raises(ValueError):a.parity_coverage(cuda,cpu,prompts)
