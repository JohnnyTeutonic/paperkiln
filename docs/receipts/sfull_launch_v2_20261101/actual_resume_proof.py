from pathlib import Path
import hashlib,json,os,subprocess,sys
root=Path('/home/jonat/pk_sfull_operation_v2_20261101');sys.path.insert(0,str(root/'tools/colab_chat'))
import sfull_checkpoint_v2 as cp
home=Path('/home/jonat/sfull_v2_resume_continuity_v4_20261101');home.mkdir(exist_ok=False)
binary=Path('/home/jonat/pk_sfull_numeric_build_v2_20261101/mtstudio');source='1ccb693f55d7328d4a8b0b7238fe8a2cafb6edf1'
publisher=root/'tools/colab_chat/sfull_checkpoint_v2.py'
(home/"publisher_snapshot.py").write_bytes(publisher.read_bytes())
stopper=home/'stop_after_checkpoint.py';stopper.write_text('import sys\nsys.path.insert(0,'+repr(str(publisher.parent))+')\nimport sfull_checkpoint_v2 as cp\ncp.main()\nif int(sys.argv[sys.argv.index("--step")+1])==4:sys.exit(75)\n')
sft=home/'sft.jsonl';sft.write_text(''.join(json.dumps({'segments':[{'text':a,'loss':False},{'text':b,'loss':True}]})+'\n' for a,b in [('a b',' c d'),('c d',' a b'),('b a',' d c'),('d c',' b a'),('a c',' b d'),('c a',' d b')]*8))
base={'name':'resume_fixture','arch':{'preset':'llama-tiny','rope_heads':'all','custom':{'d':8,'heads':2,'layers':1,'d_ff':16},'state_tokens':{'k':1,'mask':'causal'}},'data':{'vocab':'/mnt/c/ml_artifacts/capacity/vocab/vocab_llama3_bpe.gguf','vocab_cap':16384,'corpus':'/home/jonat/gpt2_precision_checks_20261031/corpus.txt','T':4,'max_tokens':0},'train':{'steps':8,'batch':2,'accum':1,'lr':.001,'seed':17,'eval_every':4,'checkpoint_every':2,'eval_windows':2,'optimizer':'adamw','lr_schedule':{'warmup_steps':2,'decay':'cosine','min_lr_frac':.1}},'export':{'formats':['safetensors']}}
env=os.environ|{'MICROTORCH_DEVICE':'cpu','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MICROTORCH_CHECKPOINT_PUBLISH_V2':str(publisher),'S_FULL_NUMERICAL_COMMIT':source}
results=[]
for objective in ['lm','sft']:
 def execute(label,writer):
  out=home/objective/label;out.mkdir(parents=True,exist_ok=True)
  spec=json.loads(json.dumps(base));spec['out_dir']=str(out)
  if objective=='sft':spec['train']['objective']='sft';spec['data']['sft_examples']=str(sft)
  sp=out/'spec.json';sp.write_text(json.dumps(spec,indent=2)+'\n')
  p=subprocess.run([str(binary),'run',str(sp)],env=env|{'MICROTORCH_CHECKPOINT_PUBLISH_V2':str(writer)},capture_output=True,text=True)
  (out/'stdout.log').write_text(p.stdout);(out/'stderr.log').write_text(p.stderr)
  return out,p
 full,a=execute('uninterrupted',publisher)
 if a.returncode:raise RuntimeError('full '+objective+': '+a.stdout[-800:]+a.stderr[-800:])
 interrupted,b=execute('interrupted',stopper)
 if b.returncode==0:raise RuntimeError('missing intentional interruption')
 item=cp.current_unlocked(interrupted/'.checkpoint_v2')
 if item[1]['step']!=4:raise RuntimeError('interrupted checkpoint not4')
 backup=home/objective/'home_backup';cp.copy_current(interrupted/'.checkpoint_v2',backup)
 resumed=home/objective/'resumed';resumed.mkdir()
 cp.copy_current(backup,resumed/'.checkpoint_v2');cp.restore_flat(resumed/'.checkpoint_v2',resumed,source)
 resumed,c=execute('resumed',publisher)
 if c.returncode:raise RuntimeError('resume '+objective+': '+c.stdout[-800:]+c.stderr[-800:])
 equal={n:cp.sha(full/n)==cp.sha(resumed/n) for n in ("model.safetensors","optim.safetensors","state.txt")}
 if not all(equal.values()):raise RuntimeError('uninterrupted/resumed bytes differ '+str(equal))
 final=cp.current_unlocked(resumed/'.checkpoint_v2')[1]
 if final['step']!=8 or final['rng']['draws']!=16:raise RuntimeError('final optimizer/RNG count differs')
 if '"optimizer":"restored"' not in c.stdout.replace(' ','') and '"optimizer": "restored"' not in c.stdout:raise RuntimeError('no restored optimizer event in stdout')
 results.append({'objective':objective,'intentional_interruption_returncode':b.returncode,'checkpoint_step':4,'final_step':8,'full_model_optimizer_state_bytes_identical':equal,'rng_reconstruction':final['rng'],'checkpoint_manifest_sha256':cp.sha(cp.current_unlocked(resumed/'.checkpoint_v2')[0]/'manifest.json')})
receipt={'status':'PASS_ACTUAL_CPU_LM_SFT_UNINTERRUPTED_VS_HOME_RESTORED','source_commit':source,'binary_sha256':cp.sha(binary),'publisher_sha256':cp.sha(publisher),'proof_script_sha256':cp.sha(Path(__file__)),'raw_home':str(home),'results':results,'scope':'Tiny actual Llama LM/SFT only; checkpoint2/stop4/final8 engineering fixture. Production25/L4/cadence and performance not validated.'}
(root/'docs/receipts/sfull_launch_v2_20261101/actual_cpu_resume_continuity_v4.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
