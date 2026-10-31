"""Offline held-out instance builder after eligible judging and complete sample capture."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from pref_artifact_io_v1 import sha
import heldout_gate_runner_v1 as runner
import chat_judge_campaign_continuation_v7 as training
from preference_dataset_gate_v7 import require_ready
api=runner.api


def build(policy_path,adoption_path,sampling_path,sampling_summary,sampling_journal,training_binding,requests_path,vocab_path,val_split,train_split,training_protocol,env_file,state_dir,out):
    policy=json.loads(Path(policy_path).read_text());adoption=json.loads(Path(adoption_path).read_text())
    if adoption.get('kind')!='ROOT_ADOPTED_API_HELDOUT_POLICY_V1' or adoption.get('adopted') is not True or adoption.get('before_dpo_outcomes') is not True or adoption.get('policy_file_sha256')!=sha(policy_path):
        raise ValueError('exact prospective held-out root adoption required')
    _,status=training.inspect(json.loads(Path(training_protocol).read_text()));require_ready(status)
    paths=dict(policy=policy_path,adoption=adoption_path,sampling_receipt=sampling_path,sampling_summary=sampling_summary,sampling_journal=sampling_journal,training_binding=training_binding,requests=requests_path,vocab=vocab_path,val_split=val_split,train_split=train_split)
    instance={'kind':'BOUND_HELDOUT_API_INSTANCE_V1','source_sha256':runner.source_hashes(),
      **{name:{'path':str(Path(path).resolve()),'sha256':sha(path)} for name,path in paths.items()},
      'training_protocol':{'path':str(Path(training_protocol).resolve()),'sha256':sha(training_protocol)},
      'cache':policy['judging']['cache'],'ledger':policy['judging']['ledger'],'env_file':str(Path(env_file).resolve()),'state_dir':str(Path(state_dir).resolve()),
      'prior_api_ledger_sha256':{path:sha(path) for path in policy['budget_migration']['prior_api_ledgers']},
      'legacy_accounting':api.legacy_account(policy['budget_migration']['legacy_cache'],policy['budget_migration']['prior_api_ledgers'])}
    # Reconstruct original source-filtered 300-item universe and every sample disposition before publishing.
    runner.load_inputs(instance)
    out=Path(out)
    with out.open('xb') as handle:handle.write(api.encoded(instance)+b'\n')
    return {'instance_sha256':sha(out),'provider_calls':0,'credential_accessor_calls':0,'activated':False}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ['policy','adoption','sampling','sampling-summary','sampling-journal','training-binding','requests','vocab','val-split','train-split','training-protocol','env-file','state-dir','out']:ap.add_argument('--'+key,required=True)
    a=ap.parse_args();print(json.dumps(build(a.policy,a.adoption,a.sampling,a.sampling_summary,a.sampling_journal,a.training_binding,a.requests,a.vocab,a.val_split,a.train_split,a.training_protocol,a.env_file,a.state_dir,a.out)))
if __name__=='__main__':main()
