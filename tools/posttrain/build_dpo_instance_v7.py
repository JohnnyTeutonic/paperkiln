"""Build an immutable, unactivated standard-watchdog DPO job after all admissions exist."""
import argparse
import copy
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from pref_artifact_io_v1 import sha
from materialize_preference_v7 import api,training_spec
import heldout_gate_runner_v1 as heldout_runner
FROZEN_HELDOUT_POLICY=Path(__file__).resolve().parents[2]/"experiments/chat_general/receipts/anthropic_api_heldout_v1_20261101/heldout_policy_frozen_for_review.json"


REQUIRED_SOURCE_FILES={"tools/mtstudio.cpp","src/ops.cpp","include/microtorch/ops.hpp",
  "tools/posttrain/dpo_vm_preflight_v7.py","tools/posttrain/materialize_preference_v8.py","tools/posttrain/chat_judge_campaign_continuation_v8.py","tools/posttrain/heldout_sample_vm_v1.py",
  "tools/posttrain/heldout_sampling_receipt_v1.py","tools/posttrain/heldout_sampling_journal_v1.py","tools/posttrain/pref_artifact_io_v1.py","tools/posttrain/assess_preference_gates_v1.py","tools/posttrain/parity_device_evidence_v1.py",
  "tools/colab_chat/chat_eval_vm.sh","tools/colab_chat/chat_parity_vm.py",
  "tools/colab_chat/dpo_checkpoint_core_v1.py","tools/colab_chat/dpo_checkpoint_v1.py",
  "tools/colab_chat/dpo_checkpoint_manager_v1.py","tools/colab_chat/dpo_checkpoint_transport_v1.py",
  "tools/colab_chat/boot_dpo_checkpoint_v1.sh","tools/colab_chat/runner_multi.py","tools/colab_chat/jobs_vm.py",
  "include/microtorch/dpo_checkpoint_bridge_v1.hpp",
  "tools/colab_chat/ensure_ember_vm.py","tools/colab_chat/tiny_chat_run.py",
  "tools/colab_chat/chat_eval_stage.py","experiments/chat_general/eval_specs/pksft.spec.json"}


def verify_source_admission(admission, source_root, dataset, base_spec, checkpoint, vocab, archive):
    required={"kind":"ROOT_DPO_SOURCE_FREEZE_V1","ready":True,
      "dataset_manifest_sha256":sha(dataset),"model_spec_sha256":sha(base_spec),
      "checkpoint_sha256":sha(checkpoint),"vocab_sha256":sha(vocab),
      "source_archive_sha256":sha(archive)}
    if any(admission.get(k)!=v or type(admission.get(k))!=type(v) for k,v in required.items()):
        raise ValueError("root source freeze does not bind exact dataset, model, weights, vocabulary and archive")
    commit=subprocess.check_output(["git","-C",str(source_root),"rev-parse","HEAD"],text=True).strip()
    if admission.get("source_commit")!=commit:raise ValueError("training checkout differs from frozen source commit")
    subprocess.run(["git","-C",str(source_root),"diff","--quiet","HEAD"],check=True)
    process=subprocess.Popen(["git","-C",str(source_root),"archive","--format=tar",commit],stdout=subprocess.PIPE)
    digest=hashlib.sha256()
    for block in iter(lambda:process.stdout.read(1024*1024),b""):digest.update(block)
    process.stdout.close()
    if process.wait()!=0 or digest.hexdigest()!=required["source_archive_sha256"]:
        raise ValueError("training archive is not the exact Git commit archive")
    files=admission.get("source_files_sha256",{})
    if not REQUIRED_SOURCE_FILES<=set(files):
        raise ValueError("source freeze omits actual DPO consumers")
    for name,expected_sha in files.items():
        relative=Path(name)
        if relative.is_absolute() or ".." in relative.parts or sha(Path(source_root)/relative)!=expected_sha:
            raise ValueError("training source file differs from root freeze")
    tree=ast.parse((Path(source_root)/"tools/colab_chat/tiny_chat_run.py").read_text())
    default=[ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign)
      and any(isinstance(target,ast.Name) and target.id=="EMBER_COMMIT" for target in node.targets)]
    if default!=[admission.get("ember_commit")]:raise ValueError("common Ember builder commit differs from root freeze")
    return commit



def bind_checkpoint_packet(out,spec,jobs,binding,source_root):
    """No hash cycle: contract -> jobs -> binding; source admission is independent."""
    out=Path(out);runtime=copy.deepcopy(spec)
    runtime.update(name='pk_dpo_v1',out_dir='/content/chat/pk_dpo_v1')
    runtime['train']['init_from']='/content/chat/upload/pksft.safetensors'
    contract={'kind':'DPO_FULL_STATE_CHECKPOINT_V1','source_commit':binding['source_commit'],
      'runtime_spec':runtime,'inputs_sha256':{
        runtime['data']['vocab']:binding['upload_sha256']['vocab.gguf'],
        runtime['data']['pref_pairs']:binding['upload_sha256']['dpo_pairs.jsonl'],
        runtime['train']['init_from']:binding['upload_sha256']['pksft.safetensors']},
      'dataset_manifest_sha256':binding['dataset_manifest_sha256']}
    cp=out/'dpo_checkpoint_contract_v1.json';cp.write_bytes(api.encoded(contract)+b'\n')
    training=[j for j in jobs['jobs'] if 'spec' in j]
    if len(training)!=1 or training[0]['name']!='pk_dpo_v1':raise ValueError('exactly one DPO training job required')
    training[0]['dpo_full_checkpoint_v1']={'every_steps':10,'max_gap_seconds':180,'steps':runtime['train']['steps'],
      'source_commit':binding['source_commit'],'seed':1,'batch':8,'accum':1,'contract_sha256':sha(cp)}
    jp=out/'jobs_v1.json';jp.write_bytes(api.encoded(jobs)+b'\n')
    binding.update(runtime_jobs_sha256=sha(jp),checkpoint_contract_sha256=sha(cp),
      checkpoint_resume='DPO_FULL_STATE_CHECKPOINT_V1_ATOMIC_MODEL_ADAMW_STATE',
      checkpoint_cadence='Persistent-volume generation advancing within180seconds while training is active; fail-stop on violation, home relay may lag.',
      boot_script='tools/colab_chat/boot_dpo_checkpoint_v1.sh')
    additions={'DPO_RUNTIME_JOBS_V1.json':str(jp.resolve()),'dpo_checkpoint_contract_v1.json':str(cp.resolve())}
    binding['uploads'].update(additions);binding['upload_sha256'].update({name:sha(path) for name,path in additions.items()})
    (out/'dpo_binding.json').write_bytes(api.encoded(binding)+b'\n')
    # Shared owner validation checks normalized runtime spec, source HEAD and both hashes offline.
    code="import sys;sys.path.insert(0,sys.argv[1]+'/tools/colab_chat');from dpo_checkpoint_manager_v1 import validate_jobs;validate_jobs(sys.argv[1],sys.argv[2],sys.argv[3])"
    subprocess.run([sys.executable,'-c',code,str(Path(source_root).resolve()),str(jp.resolve()),str(out.resolve())],check=True)
    return additions


def build(dataset_dir,base_spec,checkpoint,vocab,source_root,source_archive,source_admission,heldout_inputs,heldout_adoption,out_dir):
    dataset_dir=Path(dataset_dir);manifest=dataset_dir/"dataset_manifest.json"
    data=json.loads(manifest.read_text());admission=json.loads(Path(source_admission).read_text())
    if data.get("kind")!="IMMUTABLE_ELIGIBLE_API_PREFERENCE_DATASET_V7" or data.get("readiness")!={"eligible_complete":True,"original_full_valid_complete":False,"eligible_candidate_sets":3018,"excluded_candidate_sets":2}:
        raise ValueError("eligible dataset readiness differs")
    pairs=dataset_dir/"dpo_pairs.jsonl"
    if sha(pairs)!=data["dpo_pairs_sha256"]:raise ValueError("materialized preference pairs changed")
    commit=verify_source_admission(admission,source_root,manifest,base_spec,checkpoint,vocab,source_archive)
    if admission.get("heldout_input_pins_sha256")!=sha(heldout_inputs) or admission.get("heldout_adoption_sha256")!=sha(heldout_adoption):
        raise ValueError("root freeze omits prospective held-out data/policy adoption")
    heldout=json.loads(Path(heldout_inputs).read_text())
    for name in ["vocab","val_split","train_split"]:
        if sha(heldout[name]["path"])!=heldout[name]["sha256"]:raise ValueError("prospective held-out input changed")
    if sha(vocab)!=heldout["vocab"]["sha256"]:raise ValueError("training and held-out winner vocabularies differ")
    adopted=json.loads(Path(heldout_adoption).read_text())
    if (adopted.get("kind")!="ROOT_ADOPTED_API_HELDOUT_POLICY_V1" or adopted.get("adopted") is not True
        or adopted.get("before_dpo_outcomes") is not True or adopted.get("policy_file_sha256")!=sha(FROZEN_HELDOUT_POLICY)
        or adopted.get("operational_source_sha256")!=heldout_runner.source_hashes()
        or adopted.get("heldout_source_data_sha256")!={name:heldout[name]["sha256"] for name in ["vocab","val_split","train_split"]}):
        raise ValueError("held-out rules, operational sources and source data must be adopted before DPO training")
    spec=training_spec(json.loads(Path(base_spec).read_text()),data["token_filter_report"],
        "/content/chat/upload/pksft.safetensors","/content/chat/upload/vocab.gguf","/content/chat/upload/dpo_pairs.jsonl")
    out=Path(out_dir)
    if out.exists():raise FileExistsError("immutable training instance already exists")
    out.mkdir(parents=True)
    (out/"pk_dpo_v1.spec.json").write_bytes(api.encoded(spec)+b"\n")
    uploads={"pksft.safetensors":str(Path(checkpoint).resolve()),"vocab.gguf":str(Path(vocab).resolve()),
      "dpo_pairs.jsonl":str(pairs.resolve()),"training_source.tar":str(Path(source_archive).resolve()),
      "dialogues_val.jsonl":heldout["val_split"]["path"],"dialogues_train.jsonl":heldout["train_split"]["path"]}
    hashes={name:sha(path) for name,path in uploads.items()}
    binding={"kind":"BOUND_DPO_TRAINING_INSTANCE_V7","source_commit":commit,
      "source_admission_sha256":sha(source_admission),"dataset_manifest_sha256":sha(manifest),
      "spec_sha256":sha(out/"pk_dpo_v1.spec.json"),"ember_commit":admission["ember_commit"],
      "heldout_adoption_sha256":sha(heldout_adoption),"uploads":uploads,"upload_sha256":hashes,
      "source_files_sha256":admission["source_files_sha256"],"steps":spec["train"]["steps"],
      "checkpoint_interval_steps":10,"checkpoint_interval_requires_measured_wall_seconds_at_most":180,
      "selected_checkpoint":"final","activated":False,"provider_or_gpu_calls":0,
      "whole_completion_chain_ready":False,"unready_step":"Root final instance admission and serial paid launch remain separate; API heldout gate starts only from bound returned sampling evidence."}
    # Named atomic checkpoint manager owns training; no claim about historical standard relay.
    jobs={"max_parallel":1,"jobs":[{"name":"dpo_preflight","cmd":["python3","tools/posttrain/dpo_vm_preflight_v7.py","{upload}/dpo_binding.json","{upload}/pk_dpo_v1.spec.json","{out}"],"outputs":["preflight.json"]},
      {"name":"pk_dpo_v1","after":"dpo_preflight","spec":spec,"init_file":"pksft.safetensors"},
      {"name":"dpo_parity","after":"pk_dpo_v1","cmd":["python3","tools/colab_chat/chat_parity_vm.py","{out}","pk_dpo_v1","40"],"outputs":["parity_summary.json","parity_cuda.jsonl","parity_cpu.jsonl","ember_cuda.log","ember_cpu.log","ember_build.json"]},
      {"name":"dpo_eval","after":"dpo_parity","cmd":["bash","tools/colab_chat/chat_eval_vm.sh","{out}","pksft","pk_dpo_v1@pk_dpo_v1"],"outputs":["chat_eval_results.tar.gz","manifest.json"]},
      {"name":"dpo_heldout_sample","after":"dpo_eval","cmd":["python3","tools/posttrain/heldout_sample_vm_v1.py","{out}","{upload}/dpo_binding.json"],"outputs":["sampling_receipt.json","requests.jsonl","sampling_summary.json","sampling.journal.jsonl","ember_build.json"]}]}
    additions=bind_checkpoint_packet(out,spec,jobs,binding,source_root)
    all_uploads={**uploads,**additions,"dpo_binding.json":str((out/"dpo_binding.json").resolve()),
      "pk_dpo_v1.spec.json":str((out/"pk_dpo_v1.spec.json").resolve())}
    (out/"upload_manifest.json").write_bytes(api.encoded({"files":all_uploads,
      "sha256":{name:sha(path) for name,path in all_uploads.items()},"activated":False})+b"\n")
    return binding


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ['dataset-dir','base-spec','checkpoint','vocab','source-root','source-archive','source-admission','heldout-inputs','heldout-adoption','out-dir']:ap.add_argument('--'+key,required=True)
    a=ap.parse_args();result=build(a.dataset_dir,a.base_spec,a.checkpoint,a.vocab,a.source_root,a.source_archive,a.source_admission,a.heldout_inputs,a.heldout_adoption,a.out_dir)
    print(json.dumps({'source_commit':result['source_commit'],'steps':result['steps'],'activated':False}))
if __name__=='__main__':main()
