"""Source-bound held-out campaign. Preparation/import never dispatches; root owns activation."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from pref_artifact_io_v1 import sha
import chat_judge_api_heldout_v1 as api
import chat_judge_campaign_continuation_v7 as memory
import heldout_sampling_receipt_v1 as sampling

HERE=Path(__file__).resolve().parent

def source_hashes():
    return {name:sha(HERE/name) for name in ['heldout_gate_runner_v1.py','heldout_sampling_receipt_v1.py','heldout_sample_vm_v1.py','chat_judge_api_heldout_v1.py','chat_judge_campaign_continuation_v7.py','pref_artifact_io_v1.py','side_by_side.py','sample_candidates.py','chat_judge.py','judge_sidecar.py','chat_judge_transport_diagnostic.py','chat_judge_api_refusal_v5.py','assess_preference_gates_v1.py','../chat_eval.py','../qa_eval.py','../chat_template.py','../bpe.py','../llama3_bpe.py','../chat_sample_ember.py','../../corpus/decontam.py','../../corpus/eval/simple_qa_v1.jsonl','../../corpus/eval/tinychat_hf_prompts_v1.jsonl','../../corpus/eval/chat_prompts_v1.jsonl']}


def load_inputs(instance):
    if instance.get('kind')!='BOUND_HELDOUT_API_INSTANCE_V1' or instance['source_sha256']!=source_hashes():
        raise ValueError('held-out instance/source differs')
    for name in ['policy','adoption','sampling_receipt','sampling_summary','requests','vocab','val_split','train_split']:
        if sha(instance[name]['path'])!=instance[name]['sha256']:raise ValueError('held-out bound input differs: '+name)
    policy=json.loads(Path(instance['policy']['path']).read_text())
    adoption=json.loads(Path(instance['adoption']['path']).read_text())
    if (adoption.get('kind')!='ROOT_ADOPTED_API_HELDOUT_POLICY_V1' or adoption.get('adopted') is not True
        or adoption.get('policy_file_sha256')!=instance['policy']['sha256'] or adoption.get('before_dpo_outcomes') is not True
        or adoption.get('operational_source_sha256')!=source_hashes()
        or adoption.get('heldout_source_data_sha256')!={name:instance[name]['sha256'] for name in ['vocab','val_split','train_split']}):
        raise ValueError('held-out policy requires prospective root adoption')
    for name,expected_sha in policy['source_sha256'].items():
        if sha(HERE/name)!=expected_sha:raise ValueError('frozen held-out policy source differs')
    if policy['profile_sha256']!=api.digest(api.PROFILE) or policy['judge_id']!=api.profile_id():raise ValueError('held-out API profile differs')
    fixed=policy['judging']
    if (fixed['max_calls']!=600 or fixed['max_usd_stage']!=25 or fixed['cumulative_cap_usd']!=200
        or fixed['future_gate_reserve_usd']!=0 or fixed['reserve_per_call_usd']!=.25 or fixed['stage']!='heldout_gate'
        or fixed['batch_calls']!=24):raise ValueError('held-out fixed limits differ')
    if instance['cache']!=fixed['cache'] or instance['ledger']!=fixed['ledger']:raise ValueError('held-out namespace differs')
    if sha(policy['rubric_path'])!=policy['rubric_sha256']:raise ValueError('held-out rubric differs')
    sys.path.insert(0,str(HERE.parent));import bpe
    vocab=bpe.load_vocab(instance['vocab']['path'])
    items,_=sampling.sbs.S.prompts(instance['val_split']['path'],train_turns=sampling.sbs.S.first_turns(instance['train_split']['path']))
    receipt=json.loads(Path(instance['sampling_receipt']['path']).read_text())
    summary=json.loads(Path(instance['sampling_summary']['path']).read_text())
    if (summary['sampling_receipt_sha256']!=instance['sampling_receipt']['sha256'] or summary['requests_sha256']!=instance['requests']['sha256']
        or summary['original_items']!=300 or summary['model_sha256'][0]!=instance['vocab']['sha256']
        or summary['ember_commit']!=receipt['ember_commit'] or summary['dispositions']!=receipt['dispositions']
        or summary['eligible_requests']!=len(receipt['requests'])):raise ValueError('held-out model/sampling summary binding differs')
    requests=sampling.replay(receipt,vocab.encode,items[:300])
    if requests!=api.rows(instance['requests']['path']):raise ValueError('held-out requests differ from complete sampling replay')
    prior=policy['budget_migration']['prior_api_ledgers']
    if set(instance['prior_api_ledger_sha256'])!=set(prior) or len(prior)!=len(set(prior)):
        raise ValueError('prior ledger accounting set differs or duplicates a ledger')
    for path in prior:
        if sha(path)!=instance['prior_api_ledger_sha256'][path]:raise ValueError('completed prior ledger changed')
    if instance['cache'] in prior or instance['ledger'] in prior:raise ValueError('gate namespace overlaps prior training')
    return policy,requests


def inspect(instance):
    policy,requests=load_inputs(instance)
    _,version,rsha=api.judge.rubric_identity(policy['rubric_path'])
    baseline=api.legacy_account(policy['budget_migration']['legacy_cache'],policy['budget_migration']['prior_api_ledgers'])
    if baseline!=instance['legacy_accounting']:raise ValueError('full legacy/prior accounting changed')
    cache,ledger=Path(instance['cache']),Path(instance['ledger'])
    history=api.rows(ledger)
    if history:
        binding=api.verified_binding(cache,ledger,requests,rsha,version)
        budget=api.Budget(ledger,baseline,200,0,binding)
        api.reconcile(cache,ledger,requests,rsha,version)
        pending,paid=budget.state()
        if pending:raise ValueError('unknown held-out dispatch remains reserved')
    else:
        if api.rows(cache) or api.rows(str(cache)+'.failures.jsonl'):raise ValueError('held-out cache lacks bound ledger')
        paid=0
    if api.rows(str(cache)+'.failures.jsonl'):raise ValueError('held-out failed ranking requires separate review')
    records=api.rows(cache);done={r['key'] for r in records}
    planned={api.judge.key_of(r['prompt'],r['candidates'],o,api.profile_id(),version) for r,o in api.judge.jobs(requests)}
    calls=sum(r.get('event')=='reserve' for r in history)
    if len(done)!=len(records) or not done<=planned or calls!=len(done) or calls>600:
        raise ValueError('held-out duplicate, foreign, unknown or incomplete settlement count')
    return policy,requests,{'valid_rankings':len(done),'target_rankings':len(planned),'missing_rankings':len(planned-done),
      'calls':calls,'api_spend_usd':paid,'original_prompt_universe':300,'eligible_requests':len(requests)}


def campaign(instance,state_dir,inspector=None,dispatch=None):
    state=Path(state_dir);state.mkdir(parents=True,exist_ok=True)
    inspector=inspector or inspect;dispatch=dispatch or api.run;identity=api.digest(instance)
    with api.exclusive(state/'campaign.lock'):
        if (state/'HALTED').exists():return 2
        try:
            marker=state/'identity.json'
            if marker.exists() and json.loads(marker.read_text())!={'instance_sha256':identity}:raise ValueError('held-out state belongs to another instance')
            if not marker.exists():memory.save(marker,{'instance_sha256':identity})
            if (state/'STOPPED').exists():
                if json.loads((state/'STOPPED').read_text()).get('instance_sha256')!=identity:raise ValueError('held-out stop identity differs')
                return 0
            while True:
                memory.wait_for_memory(state,1024**3)
                policy,requests,status=inspector(instance)
                memory.save(state/'progress.json',dict(status,instance_sha256=identity))
                if status['missing_rankings']==0:
                    result=sampling.sbs.count(requests,instance['cache'],api.profile_id(),policy['rubric_path'])
                    if result['unranked']:raise ValueError('count reports incomplete held-out rankings')
                    memory.save(state/'STOPPED',{'kind':'COMPLETE_HELDOUT_JUDGING_NOT_MODEL_ADOPTION','instance_sha256':identity,'status':status,'judged_gate':result,'all_five_gates_required':True})
                    return 0
                if status['calls']>=600:raise ValueError('held-out attempt cap exhausted')
                memory.wait_for_memory(state,1024**3)
                result=dispatch(requests,instance['cache'],policy['budget_migration']['legacy_cache'],policy['rubric_path'],
                    instance['ledger'],instance['env_file'],min(24,status['missing_rankings'],600-status['calls']),.25,200,0,'heldout_gate',
                    prior_api_ledgers=policy['budget_migration']['prior_api_ledgers'],stop_on_failure=True)
                if result['failed_rankings'] or result['pending_reservations'] or not result['calls']:
                    raise ValueError('held-out batch failed, unknown or made no progress')
        except Exception as error:
            memory.save(state/'HALTED',dict(memory.safe_error_fields(error,'heldout_campaign'),instance_sha256=identity,requires_review=True,exception_text_suppressed=True))
            memory.save(state/'STOPPED',{'kind':'HALTED_REVIEW_REQUIRED','instance_sha256':identity})
            return 2


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--instance',required=True);ap.add_argument('--instance-sha256',required=True);ap.add_argument('--state-dir',required=True)
    a=ap.parse_args()
    if sha(a.instance)!=a.instance_sha256:raise SystemExit('held-out instance bytes differ')
    instance=json.loads(Path(a.instance).read_text())
    if str(Path(a.state_dir).resolve())!=instance['state_dir']:raise SystemExit('held-out state namespace differs')
    raise SystemExit(campaign(instance,a.state_dir))
if __name__=='__main__':main()
