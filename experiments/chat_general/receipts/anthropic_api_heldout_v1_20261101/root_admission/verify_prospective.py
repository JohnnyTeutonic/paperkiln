"""Read-only prospective policy/source/input verification. No model sampling or dataset materialization."""
import os,sys,json,hashlib,subprocess,time
from pathlib import Path
sys.dont_write_bytecode=True
ACTIVE=Path('/home/jonat/pk_api_continuation_v7_20261101')
PUBLIC=Path('/home/jonat/pk_publication_20261030')
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ACTIVE/'tools/posttrain'));sys.path.insert(0,str(ACTIVE/'tools'))
import heldout_gate_runner_v1 as runner
import build_dpo_instance_v7 as builder
import assess_preference_gates_v1 as assessor
import bpe
from pref_artifact_io_v1 import sha
api=runner.api

def forbidden(*a,**kw):raise RuntimeError('FORBIDDEN_PROVIDER_CREDENTIAL_OR_STATE_MUTATION')
for module in [api,runner.memory.api]:
 for name in ['run','load_key','http_post','append','atomic_blob']:setattr(module,name,forbidden)
import urllib.request
urllib.request.urlopen=forbidden

def save(name,value):
 path=OUT/name
 if path.exists():
  assert json.loads(path.read_text())==value, 'immutable review output differs: '+name
  return
 with path.open('x') as handle:json.dump(value,handle,indent=2,ensure_ascii=False);handle.write('\n')

def head(root):return subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()

policy_rel='experiments/chat_general/receipts/anthropic_api_heldout_v1_20261101/heldout_policy_frozen_for_review.json'
policy_path=ACTIVE/policy_rel;policy=json.loads(policy_path.read_text())
assert sha(policy_path)=='5181e14991c7eb326b32f9bd2d35688946f38156e4a9e1ab53375d83b751ed29'
assert sha(PUBLIC/policy_rel)==sha(policy_path)
source=runner.source_hashes();active_head=head(ACTIVE);public_head=head(PUBLIC);verifier_source_sha256=sha(__file__)
for name,value in source.items():assert sha(PUBLIC/'tools/posttrain'/name)==value
for name,value in policy['source_sha256'].items():assert sha(ACTIVE/'tools/posttrain'/name)==value
assert api.digest(api.PROFILE)==policy['profile_sha256'] and api.profile_id()==policy['judge_id']
assert api.PROFILE==runner.memory.api.PROFILE
assert sha(policy['rubric_path'])==policy['rubric_sha256']
assert sha(policy['registered_rule_source']['path'])==policy['registered_rule_source']['sha256']
assert sha(PUBLIC/'experiments/chat_general/PLAN.md')==policy['registered_rule_source']['sha256']
fixed_sampling=dict(requested_prompts=300,split='smol-smoltalk validation',temperature=.7,seed_rule='71 + original prompt index',max_new_tokens=256,ctx=512,top_k=0,top_p=1.0,ember_deterministic=True,TF32=False,unchanged_skips=['over_context','identical','invalid_utf8'],no_resampling_or_replacement=True)
assert all(policy['sampling'].get(k)==v for k,v in fixed_sampling.items())
fixed_judging=dict(max_requests=300,max_calls=600,batch_calls=24,max_usd_stage=25,cumulative_cap_usd=200,future_gate_reserve_usd=0,reserve_per_call_usd=.25,http_retries=0,sdk_retries=0,stage='heldout_gate')
assert all(policy['judging'].get(k)==v for k,v in fixed_judging.items())
prior=policy['budget_migration']['prior_api_ledgers'];assert len(prior)==2 and len(set(prior))==2
assert policy['judging']['ledger'] not in prior and policy['judging']['cache'] not in prior
for name in ['cache','ledger']:
 path=Path(policy['judging'][name])
 assert not path.exists(), 'heldout namespace already exists; prospective review required'
assert not Path('/mnt/c/ml_artifacts/pref_api_heldout_v1/heldout_cache.jsonl.failures.jsonl').exists()

pins=json.loads((ACTIVE/'experiments/chat_general/receipts/preference_downstream_v7_20261101/existing_input_pins.json').read_text())
for name in ['checkpoint','base_spec','vocab','val_split','train_split']:assert sha(pins[name]['path'])==pins[name]['sha256']
# No prompt contents or model responses are copied to the review packet.
S=runner.sampling.sbs.S
train_turns=S.first_turns(pins['train_split']['path'])
items,dropped=S.prompts(pins['val_split']['path'],train_turns=train_turns)
registered=items[:300];assert len(registered)==300 and len({uri for uri,_ in registered})==300
vocab=bpe.load_vocab(pins['vocab']['path'])
inventory=[]
for index,(uri,user) in enumerate(registered):
 n=len(vocab.encode('<|user|>'+user+'<|assistant|>'))
 inventory.append(dict(original_index=index,uri=uri,prompt_text_sha256=hashlib.sha256(user.encode()).hexdigest(),prompt_tokens=n,seed=71+index,pre_generation_disposition='over_context' if n+256>512 else 'requires_original_sampling_outcome'))
assert [r['seed'] for r in inventory]==list(range(71,371))
original=dict(kind='REGISTERED_ORIGINAL_HELDOUT_UNIVERSE_V1',items=300,original_items_sha256=api.digest(registered),source_filter_dropped=dropped,filtered_validation_items=len(items),context_eligible=sum(r['pre_generation_disposition']!='over_context' for r in inventory),inventory=inventory,no_refill=True,model_responses_generated=0)
save('original_300_universe.json',original)
# Check all five frozen thresholds through the actual pure assessor.
metrics=dict(protocol_failure_rates=dict(frozen=.1,everyday=.1,ultrachat=.1),repeat_rates={s:dict(reference=.2,dpo=.2) for s in ['frozen','ultrachat']},qa_correct=dict(reference=30,dpo=27),parity={leg:{k:40 for k in ['n','reply_identical','token_count','stop_reason','prompt_tokens']} for leg in ['cuda','cpu']},judged_gate=dict(unranked=0,b_win_share=.6,b_win_share_wilson95=[.51,.7]))
assert assessor.assess(metrics)['adoption_eligible']
import copy
for gate in ['protocol','repeats','simple_qa','ember_parity','judged_side_by_side']:
 m=copy.deepcopy(metrics)
 if gate=='protocol':m['protocol_failure_rates']['everyday']=.1001
 if gate=='repeats':m['repeat_rates']['ultrachat']['dpo']=.2001
 if gate=='simple_qa':m['qa_correct']['dpo']=26
 if gate=='ember_parity':m['parity']['cpu']['reply_identical']=39
 if gate=='judged_side_by_side':m['judged_gate']['b_win_share_wilson95'][0]=.5
 v=assessor.assess(m);assert not v['adoption_eligible'] and not v['gates'][gate] and not v['automatic_model_replacement']

extra={'tools/posttrain/build_heldout_instance_v1.py','tools/posttrain/materialize_preference_v8.py','tools/posttrain/build_dpo_instance_v7.py','tools/posttrain/preference_dataset_gate_v7.py','tools/posttrain/dpo_pairs.py'}|builder.REQUIRED_SOURCE_FILES
consumer_hashes={name:sha(PUBLIC/name) for name in sorted(extra)}
for name,value in consumer_hashes.items():
 if name.startswith('tools/posttrain/') or name in ['tools/colab_chat/chat_parity_vm.py','tools/colab_chat/ensure_ember_vm.py','tools/colab_chat/tiny_chat_run.py','tools/colab_chat/chat_eval_vm.sh','tools/colab_chat/chat_eval_stage.py']:
  assert sha(ACTIVE/name)==value, 'operational consumer bytes differ across active/public checkout: '+name
subprocess.run(['git','-C',str(PUBLIC),'diff','--quiet','HEAD','--',*sorted(extra)],check=True)
assert source==runner.source_hashes() and verifier_source_sha256==sha(__file__)
for name,value in source.items():assert sha(PUBLIC/'tools/posttrain'/name)==value
for name in ['checkpoint','base_spec','vocab','val_split','train_split']:assert sha(pins[name]['path'])==pins[name]['sha256']
# Existing consumers require exactly these critical adoption fields. Additional inventory pin is prospective evidence.
adoption=dict(kind='ROOT_ADOPTED_API_HELDOUT_POLICY_V1',adopted=True,before_dpo_outcomes=True,policy_file_sha256=sha(policy_path),operational_source_sha256=source,heldout_source_data_sha256={name:pins[name]['sha256'] for name in ['vocab','val_split','train_split']},original_300_universe_sha256=sha(OUT/'original_300_universe.json'),original_items_sha256=original['original_items_sha256'],registered_original_prompts=300,max_api_attempts=600,max_usd_stage=25,cumulative_cap_usd=200,all_five_original_gates_required=True,no_refill=True,consumer_source_sha256=consumer_hashes)
save('proposed_heldout_adoption.json',adoption)
save('source_and_input_verification.json',dict(status='PASS_READONLY_PROSPECTIVE_NOT_ADOPTED',verifier_source_sha256=verifier_source_sha256,active_source_root=str(ACTIVE),active_commit=active_head,public_source_root=str(PUBLIC),public_commit_begin=public_head,public_commit_end=head(PUBLIC),policy_sha256=sha(policy_path),operational_source_sha256=source,consumer_source_sha256=consumer_hashes,input_pins=pins,source_filter_dropped=dropped,original_items_sha256=original['original_items_sha256'],context_eligible=original['context_eligible'],registered_prompts=300,provider_calls=0,gpu_calls=0,credential_accesses=0,training_dataset_materialized=False,adoption_performed=False,model_responses_generated=0,active_frozen_sources_unchanged=True,final_training_ledger_binding='DEFERRED_UNTIL_ELIGIBLE_JUDGING_COMPLETE',final_dpo_weights_and_pairs_binding='DEFERRED_UNTIL_SOURCE_FREEZE_AND_TRAINING',final_sampling_outcomes='DEFERRED_NO_MODEL_CALLS'))
print(json.dumps(dict(status='PASS',items=300,context_eligible=original['context_eligible'],universe_sha256=original['original_items_sha256'],adoption_sha256=sha(OUT/'proposed_heldout_adoption.json'),public_commit=head(PUBLIC))))
