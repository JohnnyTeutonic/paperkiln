from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys
ROOT=Path("/home/jonat/pk_dpo_checkpoint_20261101")
sys.path.insert(0,str(ROOT/"tools/colab_chat"))
import dpo_checkpoint_core_v1 as core
import dpo_checkpoint_v1 as dpo
HOME=Path("/home/jonat/dpo_resume_proof_v2_20261101");HOME.mkdir(exist_ok=False)
BINARY=Path("/home/jonat/pk_dpo_checkpoint_build_20261101/mtstudio")
SOURCE="75dc4497f" # replaced with full source pin before dispatch
SOURCE=subprocess.check_output(["git","-C",str(ROOT),"rev-parse",SOURCE],text=True).strip()
ENV=os.environ|{"MICROTORCH_DEVICE":"cpu","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1"}
VOCAB="/mnt/c/ml_artifacts/capacity/vocab/vocab_llama3_bpe.gguf"
ARCH={"preset":"llama-tiny","rope_heads":"all","custom":{"d":8,"heads":2,"layers":1,"d_ff":16}}
def execute(spec,env,label):
    path=HOME/(label+"_spec.json");path.write_text(json.dumps(spec,indent=2)+"\n")
    result=subprocess.run([str(BINARY),"run",str(path)],env=env,capture_output=True,text=True)
    (HOME/(label+"_stdout.log")).write_text(result.stdout);(HOME/(label+"_stderr.log")).write_text(result.stderr)
    return result
sft=HOME/"initial_sft.jsonl";sft.write_text("".join(json.dumps({"segments":[{"text":"a","loss":False},{"text":" b","loss":True}]})+"\n" for _ in range(20)))
initial={"name":"initial","out_dir":str(HOME/"initial"),"arch":ARCH,"data":{"vocab":VOCAB,"vocab_cap":16384,"sft_examples":str(sft),"T":4,"max_tokens":0},"train":{"objective":"sft","steps":1,"batch":2,"accum":1,"lr":.001,"seed":1,"eval_every":1,"checkpoint_every":1,"eval_windows":1,"optimizer":"adamw"},"export":{"formats":["safetensors"]}}
assert execute(initial,ENV,"initial").returncode==0
base_weights=HOME/"initial/initial.safetensors"
pairs=HOME/"pairs.jsonl";pairs.write_text("".join(json.dumps({"prompt":"a","chosen":" b","rejected":" c","chosen_stop":"eos","rejected_stop":"eos"})+"\n" for _ in range(20)))
out=HOME/"run"
spec={"name":"pk_dpo_v1","out_dir":str(out),"arch":ARCH,"data":{"vocab":VOCAB,"vocab_cap":16384,"pref_pairs":str(pairs),"T":4,"max_tokens":0},"train":{"objective":"provenance_po","steps":20,"batch":8,"accum":1,"lr":2e-6,"seed":1,"eval_every":10,"checkpoint_every":10,"eval_windows":1,"optimizer":"adamw","init_from":str(base_weights),"po":{"beta":.1,"weight_mode":"uniform","nll":.2},"lr_schedule":{"warmup_steps":20,"decay":"cosine","min_lr_frac":.1}},"export":{"formats":["safetensors"]}}
contract={"kind":dpo.KIND,"source_commit":SOURCE,"runtime_spec":spec,"inputs_sha256":{str(p):core.sha(p) for p in [Path(VOCAB),pairs,base_weights]},"dataset_manifest_sha256":"f"*64}
contract_path=HOME/"contract.json";contract_path.write_text(json.dumps(contract,indent=2)+"\n")
publisher=ROOT/"tools/colab_chat/dpo_checkpoint_v1.py"
(HOME/"publisher_snapshot.py").write_bytes(publisher.read_bytes())
(HOME/"core_snapshot.py").write_bytes((publisher.parent/"dpo_checkpoint_core_v1.py").read_bytes())
env=ENV|{"MICROTORCH_DPO_CHECKPOINT_PUBLISH_V1":str(publisher),"DPO_NUMERICAL_SOURCE_COMMIT":SOURCE,"DPO_CHECKPOINT_CONTRACT":str(contract_path)}
assert execute(spec,env,"full").returncode==0
out.rename(HOME/"uninterrupted")
stopper=HOME/"stop_after_ten.py";stopper.write_text("import sys\nsys.path.insert(0,"+repr(str(publisher.parent))+")\nimport dpo_checkpoint_v1 as dpo\ndpo.main()\nif sys.argv[1]=='publish' and int(sys.argv[sys.argv.index('--step')+1])==10:sys.exit(75)\n")
failed=execute(spec,env|{"MICROTORCH_DPO_CHECKPOINT_PUBLISH_V1":str(stopper)},"interrupted")
assert failed.returncode!=0
assert core.current_unlocked(out/".dpo_checkpoint_v1")[1]["step"]==10
home=HOME/"home_bundle";core.copy_current(out/".dpo_checkpoint_v1",home)
out.rename(HOME/"interrupted_output")
# Actual CLI prepare restores the complete home bundle before any resume loading.
core.copy_current(home,out/".dpo_checkpoint_v1")
assert execute(spec,env,"resumed").returncode==0
names=["model.safetensors","optim.safetensors","state.txt"]
equal={n:core.sha(HOME/"uninterrupted"/n)==core.sha(out/n) for n in names}
assert all(equal.values())
assert core.sha(out/"model.safetensors")!=core.sha(base_weights)
events=[json.loads(x) for x in (out/"events.jsonl").read_text().splitlines()]
assert any(e.get("event")=="resume" and e.get("optimizer")=="restored" and e.get("step")==10 for e in events)
final=core.current_unlocked(out/".dpo_checkpoint_v1")[1]
assert final["step"]==20 and final["rng"]["draws"]==160
# Preserve actual missing-optimizer negative: flat output with no atomic generation.
negative=HOME/"negative_flat";negative.mkdir();shutil.copyfile(out/"model.safetensors",negative/"model.safetensors");shutil.copyfile(out/"state.txt",negative/"state.txt")
negative_spec=json.loads(json.dumps(spec));negative_spec["out_dir"]=str(negative)
negative_contract=json.loads(json.dumps(contract));negative_contract["runtime_spec"]=negative_spec
negative_path=HOME/"negative_contract.json";negative_path.write_text(json.dumps(negative_contract))
refused=execute(negative_spec,env|{"DPO_CHECKPOINT_CONTRACT":str(negative_path)},"negative_missing_optimizer")
assert refused.returncode!=0 and "cold resume refused" in refused.stderr
receipt={"status":"PASS_ACTUAL_CPU_PROVENANCE_PO_FULL_HOME_RESUME","source_commit":SOURCE,"binary_sha256":core.sha(BINARY),"publisher_sha256":core.sha(publisher),"core_sha256":core.sha(publisher.parent/"dpo_checkpoint_core_v1.py"),"script_sha256":core.sha(Path(__file__)),"raw_home":str(HOME),"weights_adamw_state_bytes_identical":equal,"checkpoint_step":10,"final_step":20,"rng":final["rng"],"fresh_sft_init_from_used":True,"actual_cold_missing_optimizer_refused":True,"scope":"Tiny Llama D8/L1/T4 batch8 CPU engineering fixture. Actual L4 ten-step cadence, quality and performance are unmeasured."}
(ROOT/"docs/receipts/dpo_checkpoint_readiness_20261101/actual_cpu_continuity_v2.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt,indent=2))
