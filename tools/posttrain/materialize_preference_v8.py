"""Offline, immutable preference materialization; never calls a provider or training binary."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parent))
from pref_artifact_io_v1 import sha
import chat_judge_campaign_continuation_v8 as campaign
import dpo_pairs
from preference_dataset_gate_v7 import require_ready
api=campaign.api


def verify_adoption(protocol_path,proposal,adoption,operational_adoption,adoption_path):
    from materialize_preference_v7 import verify_adoption as original_policy
    protocol=json.loads(Path(protocol_path).read_text())
    decision=original_policy(Path(protocol['fresh_dispatch_gate']['v7_protocol']['path']),proposal,adoption)
    expected={'kind':'ROOT_ADOPTED_V8_PROVEN_UNTRANSMITTED_CONTINUATION','adopted':True,
      'protocol_file_sha256':sha(protocol_path),'source_sha256':campaign.source_hashes(),
      'original_dataset_adoption_sha256':sha(adoption_path),'eligible_target_rankings':6036,
      'eligible_candidate_sets':3018,'preserve_accepted_mate':True,'no_new_candidate_exclusions':True,
      'training_recipe_and_five_part_acceptance_unchanged':True,'future_unknown_auto_retry':False}
    if any(operational_adoption.get(k)!=v or type(operational_adoption.get(k))!=type(v) for k,v in expected.items()):
        raise ValueError('prospective V8 operational adoption differs; scientific dataset policy remains V7')
    return decision


def encode_rows(records):
    return b"".join(api.encoded(r)+b"\n" for r in records)


def materialize(protocol_path, adoption_path, operational_adoption_path, vocab_path, out_dir, encode=None):
    protocol_path,adoption_path,vocab_path=map(Path,[protocol_path,adoption_path,vocab_path])
    protocol=json.loads(protocol_path.read_text());proposal=Path(protocol["dataset_decision_file"])
    verify_adoption(protocol_path,proposal,json.loads(adoption_path.read_text()),json.loads(Path(operational_adoption_path).read_text()),adoption_path)
    watched=[protocol_path,proposal,adoption_path,Path(operational_adoption_path),Path(protocol["fresh_dispatch_gate"]["v7_protocol"]["path"]),vocab_path,Path(protocol["cache"]),Path(protocol["ledger"]),Path(protocol["cache"]+".failures.jsonl")]
    before={str(p):sha(p) for p in watched}
    requests,status=campaign.inspect(protocol);readiness=require_ready(status)
    if encode is None:
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
        import bpe
        encode=bpe.load_vocab(str(vocab_path)).encode
    out_dir=Path(out_dir)
    if out_dir.exists():raise FileExistsError("immutable dataset output already exists")
    out_dir.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pref_v7_",dir=out_dir.parent) as scratch:
        temp=Path(scratch);raw=temp/"agreement_pairs.jsonl"
        pairs_report=api.judge.pairs(requests,str(protocol["cache"]),api.profile_id(),protocol["rubric"],str(raw))
        if pairs_report["unranked"]!=2:raise ValueError("pair extractor missing-set count differs from adopted exclusions")
        records=api.rows(raw)
        if any(r.get(k) not in ["eos","length","turn"] for r in records for k in ["chosen_stop","rejected_stop"]):
            raise ValueError("sampled stop metadata missing or invalid; no default EOS inference")
        kept,report=dpo_pairs.build(records,encode,512)
        if report["train_pairs"]<1 or report["val_pairs"]<1:raise ValueError("insufficient training/validation pairs")
        final=temp/"dpo_pairs.jsonl";final.write_bytes(encode_rows(kept))
        if before!={str(p):sha(p) for p in watched}:raise ValueError("bound input changed during materialization")
        pairs_report["out"]="agreement_pairs.jsonl"
        manifest={"kind":"IMMUTABLE_ELIGIBLE_API_PREFERENCE_DATASET_V7","readiness":readiness,
          "operational_continuation_version":"V8_ONE_PROVEN_UNTRANSMITTED_FRESH_DISPATCH","scientific_cohort_version":"V7_6036","input_sha256":before,"materializer_sha256":sha(__file__),"pair_filter_report":pairs_report,
          "token_filter_report":report,"agreement_pairs_sha256":sha(raw),"dpo_pairs_sha256":sha(final),
          "file_order_preserved":True,"chosen_finished_is_breakdown_only":True,
          "original_full_valid_completeness":False,"provider_calls":0,"training_launched":False}
        (temp/"dataset_manifest.json").write_bytes(api.encoded(manifest)+b"\n")
        # Same-filesystem rename publishes the complete dataset atomically.
        os.rename(temp,out_dir)
    return manifest


def training_spec(base_spec,counts,checkpoint,vocab,pairs):
    train_pairs=counts["train_pairs"]
    if type(train_pairs) is not int or train_pairs<1:raise ValueError("positive bound training-pair count required")
    spec=copy.deepcopy(base_spec)
    spec["name"]="pk_dpo_v1";spec["out_dir"]="/content/chat/pk_dpo_v1"
    data={"vocab":vocab,"T":512,"pref_pairs":pairs}
    if "vocab_cap" in spec.get("data",{}):data["vocab_cap"]=spec["data"]["vocab_cap"]
    spec["data"]=data
    spec["train"]={"objective":"provenance_po","init_from":checkpoint,"steps":(2*train_pairs+7)//8,
      "batch":8,"accum":1,"lr":2e-6,"optimizer":"adamw","precision":"fp32","attention_precision":"fp32",
      "attention_kernel":"tiled","fused_head":False,"seed":1,"eval_every":10,"checkpoint_every":10,
      "lr_schedule":{"warmup_steps":20,"decay":"cosine","min_lr_frac":.1},
      "po":{"weight_mode":"uniform","beta":.1,"nll":.2}}
    spec["export"]={"formats":["safetensors","gguf"]}
    return spec


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ["protocol","adoption","operational-adoption","vocab","out-dir"]:ap.add_argument("--"+name,required=True)
    a=ap.parse_args();result=materialize(a.protocol,a.adoption,a.operational_adoption,a.vocab,a.out_dir)
    print(json.dumps({"out":a.out_dir,"pairs":result["token_filter_report"]["pairs"],"training_launched":False}))
if __name__=="__main__":main()
