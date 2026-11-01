"""DPO V4 supplemental source linkage and offline intake only. No campaign entry point.

source-check validates sources, not sampling readiness. prepare runs the unchanged
immutable-instance builder only after actual receipts exist. inspect replays the
unchanged inspector with provider/credential/network access explicitly disabled.
"""
import argparse,contextlib,hashlib,importlib,json,socket,sys
from pathlib import Path
SOURCE=Path('/home/jonat/dpo_startup_admission_v4_20261102/source')
ORIGINAL=Path('/home/jonat/heldout_api_root_admission_20261101/heldout_adoption.json')
SUPPLEMENT=Path('/home/jonat/dpo_heldout_intake_review_v4_20261102/consumer_linkage_adoption_v4.json')
ORIGINAL_SHA='e14520a583bdb308f4c7d5fd2a6ba9036ceb6e0354292f6f31144a0383f39c2c'
SUPPLEMENT_SHA='9bbddcbc85185003aa654654124297f06ce0efa8f4a1fd89d3bfeb2c5576c56a'
BINDING=Path('/home/jonat/dpo_startup_admission_v4_20261102/instance/dpo_binding.json')
BINDING_SHA='0bcde3e10b8389989a4080f79c65922f5daf26c19207555eedb97918a0fa206a'
COMMIT='3f9217c1516e5c08a48994b92128615e8f1d63a6'
POLICY=SOURCE/'experiments/chat_general/receipts/anthropic_api_heldout_v1_20261101/heldout_policy_frozen_for_review.json'
POLICY_SHA='5181e14991c7eb326b32f9bd2d35688946f38156e4a9e1ab53375d83b751ed29'
DATASET=Path('/mnt/c/ml_artifacts/pref_api_20261031/dpo_dataset_v8_20261101/dataset_manifest.json')
TRAINING_PROTOCOL=Path('/home/jonat/pk_api_continuation_v7_20261101/experiments/chat_general/receipts/anthropic_api_continuation_v8_20261101/full_campaign_v8_protocol.json')
SAMPLE_NAMES=('sampling_receipt.json','sampling_summary.json','sampling.journal.jsonl','requests.jsonl')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def regular(path):
 p=Path(path)
 if not p.is_file() or p.is_symlink() or p.resolve()!=p:raise ValueError('canonical regular input required: '+str(p))
 return p

def check_file(path,digest):
 p=regular(path)
 if sha(p)!=digest:raise ValueError('source or input hash differs: '+str(p))
 return p

def source_path(name,post=False):
 p=((SOURCE/'tools/posttrain' if post else SOURCE)/name).resolve()
 if not p.is_relative_to(SOURCE):raise ValueError('source pin escapes frozen checkout')
 return p

def validate_linkage(original,supplement,binding):
 if (original.get('kind')!='ROOT_ADOPTED_API_HELDOUT_POLICY_V1' or original.get('adopted') is not True
     or original.get('before_dpo_outcomes') is not True or original.get('policy_file_sha256')!=POLICY_SHA):raise ValueError('original prospective policy differs')
 if (supplement.get('kind')!='ROOT_ADOPTED_HELDOUT_V4_CONSUMER_LINKAGE' or supplement.get('adopted') is not True
     or supplement.get('before_dpo_training_and_outcomes') is not True or supplement.get('original_adoption_modified') is not False
     or supplement.get('original_heldout_operational_sources_unchanged') is not True
     or supplement.get('original_policy_and_operational_hashes_verified') is not True
     or supplement.get('judging_launch_authorized') is not False
     or supplement.get('all_five_original_gates_required') is not True
     or supplement.get('automatic_model_replacement') is not False
     or supplement.get('limits_unchanged')!={'stage_usd':25,'cumulative_usd':200,'max_attempts':600}):raise ValueError('supplemental scope or limits differ')
 for key,want in {'original_adoption_path':str(ORIGINAL),'original_adoption_sha256':ORIGINAL_SHA,'frozen_policy_sha256':POLICY_SHA,'v4_training_binding_path':str(BINDING),'v4_training_binding_sha256':BINDING_SHA,'v4_training_source_commit':COMMIT}.items():
  if supplement.get(key)!=want:raise ValueError('supplemental linkage differs: '+key)
 if binding.get('kind')!='BOUND_DPO_TRAINING_INSTANCE_V7' or binding.get('source_commit')!=COMMIT or binding.get('heldout_adoption_sha256')!=ORIGINAL_SHA:raise ValueError('training binding linkage differs')
 base=original['consumer_source_sha256'];actual=binding['source_files_sha256']
 changes={name:{'original_adopted_sha256':base[name],'v4_actual_sha256':value} for name,value in actual.items() if name in base and value!=base[name]}
 extras={name:value for name,value in actual.items() if name not in base}
 if changes!=supplement['changed_consumer_hashes'] or extras!=supplement['extra_bound_consumer_hashes']:raise ValueError('consumer change inventory differs')
 return changes,extras

def verify_sources():
 for p,h in [(ORIGINAL,ORIGINAL_SHA),(SUPPLEMENT,SUPPLEMENT_SHA),(BINDING,BINDING_SHA),(POLICY,POLICY_SHA)]:check_file(p,h)
 original=json.loads(ORIGINAL.read_text());supplement=json.loads(SUPPLEMENT.read_text());binding=json.loads(BINDING.read_text())
 changes,extras=validate_linkage(original,supplement,binding)
 check_file(DATASET,binding['dataset_manifest_sha256'])
 data=json.loads(DATASET.read_text());check_file(TRAINING_PROTOCOL,data['input_sha256'][str(TRAINING_PROTOCOL)])
 for name,digest in original['operational_source_sha256'].items():check_file(source_path(name,True),digest)
 consumers=dict(original['consumer_source_sha256']);consumers.update({n:v['v4_actual_sha256'] for n,v in changes.items()});consumers.update(extras)
 for name,digest in consumers.items():check_file(source_path(name),digest)
 for name,digest in binding['source_files_sha256'].items():check_file(source_path(name),digest)
 for name,digest in json.loads(POLICY.read_text())['source_sha256'].items():check_file(source_path(name,True),digest)
 return {'status':'SOURCE_LINKAGE_VALID_NOT_SAMPLING_READY','sampling_readiness_established':False,'source_commit':COMMIT,'original_adoption_sha256':ORIGINAL_SHA,'supplemental_adoption_sha256':SUPPLEMENT_SHA,'training_binding_sha256':BINDING_SHA,'changed_consumers':len(changes),'extra_consumers':len(extras),'provider_calls':0,'credential_reads':0,'judging_launch_authorized':False}

@contextlib.contextmanager
def offline_provider_barrier(api):
 def forbidden(*a,**kw):raise RuntimeError('offline intake forbids provider, credentials and network')
 old=(api.load_key,api.http_post,socket.create_connection,socket.socket.connect)
 api.load_key=api.http_post=socket.create_connection=socket.socket.connect=forbidden
 try:yield
 finally:api.load_key,api.http_post,socket.create_connection,socket.socket.connect=old

def frozen_modules():
 sys.path.insert(0,str(SOURCE/'tools/posttrain'))
 builder=importlib.import_module('build_heldout_instance_v1');runner=importlib.import_module('heldout_gate_runner_v1')
 if Path(builder.__file__).resolve()!=SOURCE/'tools/posttrain/build_heldout_instance_v1.py' or Path(runner.__file__).resolve()!=SOURCE/'tools/posttrain/heldout_gate_runner_v1.py':raise ValueError('unexpected module import location')
 return builder,runner

def prepare(plan_path,out=None):
 proof=verify_sources();plan=json.loads(regular(Path(plan_path).resolve()).read_text());argv=plan['builder_argv']
 if argv[:2]!=['python3',str(SOURCE/'tools/posttrain/build_heldout_instance_v1.py')]:raise ValueError('builder source differs')
 keys=['policy','adoption','sampling','sampling-summary','sampling-journal','training-binding','requests','vocab','val-split','train-split','training-protocol','env-file','state-dir','out']
 if len(argv[2:])!=2*len(keys) or [x for x in argv[2::2]]!=['--'+k for k in keys]:raise ValueError('builder option inventory differs')
 values=dict(zip(keys,argv[3::2]))
 if (values['policy']!=str(POLICY) or values['adoption']!=str(ORIGINAL) or values['training-binding']!=str(BINDING) or values['training-protocol']!=str(TRAINING_PROTOCOL)):raise ValueError('builder adoption/policy/binding differs')
 for key in ['sampling','sampling-summary','sampling-journal','requests']:regular(Path(values[key]))
 original=json.loads(ORIGINAL.read_text())
 for key,name in [('vocab','vocab'),('val-split','val_split'),('train-split','train_split')]:check_file(values[key],original['heldout_source_data_sha256'][name])
 if out is not None:values['out']=str(Path(out).resolve())
 builder,runner=frozen_modules()
 with offline_provider_barrier(runner.api):
  result=builder.build(*[values[k] for k in keys])
  instance=json.loads(Path(values['out']).read_text())
  if instance['training_binding']['sha256']!=BINDING_SHA:raise ValueError('built instance changed training binding')
  status=runner.inspect(instance)[2]
 return dict(proof,status='OFFLINE_INSTANCE_PREPARED_AND_INSPECTED_NOT_ADMITTED',sampling_readiness_established=True,instance_path=values['out'],instance_sha256=sha(values['out']),inspect_status=status,builder_result=result)

def inspect(path,digest):
 proof=verify_sources();check_file(Path(path).resolve(),digest);instance=json.loads(Path(path).read_text())
 if instance['training_binding']!={'path':str(BINDING),'sha256':BINDING_SHA}:raise ValueError('instance training binding differs')
 protocol_sha=json.loads(DATASET.read_text())['input_sha256'][str(TRAINING_PROTOCOL)]
 if instance['training_protocol']!={'path':str(TRAINING_PROTOCOL),'sha256':protocol_sha}:raise ValueError('instance training protocol differs')
 if instance['policy']!={'path':str(POLICY),'sha256':POLICY_SHA} or instance['adoption']!={'path':str(ORIGINAL),'sha256':ORIGINAL_SHA}:raise ValueError('instance policy/adoption differs')
 _,runner=frozen_modules()
 with offline_provider_barrier(runner.api):status=runner.inspect(instance)[2]
 return dict(proof,status='OFFLINE_INSTANCE_INSPECTED_NOT_ADMITTED',sampling_readiness_established=True,inspect_status=status)

def main():
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True);sub.add_parser('source-check');a=sub.add_parser('prepare');a.add_argument('--plan',required=True);a.add_argument('--out');a=sub.add_parser('inspect');a.add_argument('--instance',required=True);a.add_argument('--instance-sha256',required=True);a=p.parse_args()
 result=verify_sources() if a.command=='source-check' else prepare(a.plan,a.out) if a.command=='prepare' else inspect(a.instance,a.instance_sha256)
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
