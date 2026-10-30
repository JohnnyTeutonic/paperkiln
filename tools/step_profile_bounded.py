"""Fail-closed operational wrapper for the existing descriptive BF16/flash profiler."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tarfile
import time
import step_profile as profile

ROOT=Path(__file__).resolve().parents[1]
CONFIGS=('fp32','bf16','flash','bf16_flash')
SHAPES=('20m_L2','20m_L4','20m_L8','20m_L16','20m_L32','60m_L4','60m_L8','60m_L16','60m_L32','pk_base_v1')
STEPS,WARM=30,8

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def atomic(p,data):
 p=Path(p);t=p.with_suffix(p.suffix+'.incoming');t.write_text(json.dumps(data,indent=1)+'\n');t.replace(p)
def finite(x):return type(x) in (int,float) and math.isfinite(x)

def source_inventory():
 commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
 subprocess.run(['git','-C',str(ROOT),'diff','--quiet','HEAD','--'],check=True)
 files=subprocess.check_output(['git','-C',str(ROOT),'ls-files','-z']).decode().split('\0')
 return {'commit':commit,'files':{p:sha(ROOT/p) for p in files if p and (ROOT/p).is_file()}}

def gpu_identity():
 return subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True).strip()

def validate_corpus(p):
 p=Path(p)
 with p.open('rb') as f:header=f.read(24)
 if header[:8]!=b'PKIDS001' or struct.unpack('<IIQ',header[8:])!=(2,16384,8_000_000) or p.stat().st_size!=16_000_024:
  raise ValueError('synthetic corpus header/size mismatch')
 return sha(p)

def validate_cell(folder,shape,cfg):
 key=f'{shape}/{cfg}';r=load(folder/'profile.json')['runs'].get(key)
 if (not r or r.get('done') is not True or r.get('clean_rc')!=0 or r.get('prof_rc')!=0
     or r.get('stalled') or r.get('refused') or not finite(r.get('clean_tok_per_s')) or r['clean_tok_per_s']<=0
     or not finite(r.get('clean_s_per_step')) or r['clean_s_per_step']<=0):
  raise ValueError('incomplete/refused/failed profile cell: '+key)
 p=r.get('profile') or {}
 if p.get('steps')!=STEPS-WARM or not all(finite(p.get(k)) for k in ['wall_ms','gpu_ms','idle_ms','peak_mb']) or p['wall_ms']<=0:
  raise ValueError('missing/nonfinite phase telemetry: '+key)
 receipts={'profile_cell_sha256':hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()}
 specs=[]
 for mode in ('clean','prof'):
  base=folder/'runs'/f'{shape}_{cfg}_{mode}';events=load_lines(base/'events.jsonl')
  model=[e for e in events if e.get('event')=='model']
  if len(model)!=1:raise ValueError('missing/duplicate model event: '+key+'/'+mode)
  want=profile.CONFIGS[cfg]
  if (model[0].get('precision','fp32')!=want.get('precision','fp32')
      or model[0].get('attention_kernel','tiled')!=want.get('attention_kernel','tiled')):
   raise ValueError('model did not record requested settings: '+key+'/'+mode)
  steps=[e for e in events if e.get('event')=='step']
  if [e.get('step') for e in steps]!=list(range(1,STEPS+1)) or not all(finite(e.get('loss')) for e in steps):
   raise ValueError('not exact 30 finite-loss steps: '+key+'/'+mode)
  done=[e for e in events if e.get('event')=='done']
  if len(done)!=1 or done[0].get('final_step')!=STEPS or done[0].get('early_stopped'):
   raise ValueError('not complete 30-step run: '+key+'/'+mode)
  if mode=='prof':
   timed=[e for e in steps if e['step']>WARM]
   if any(not e.get('t_ms') or not e.get('gpu_ms') for e in timed):raise ValueError('missing raw profile spans: '+key)
   for e in timed:
    for name in ('t_ms','gpu_ms','gpu_n'):
     if any(not finite(v) or v<0 for v in e.get(name,{}).values()):raise ValueError('invalid raw telemetry: '+key)
  sp=folder/'runs'/f'{shape}_{cfg}_{mode}_spec.json';spec=load(sp);spec.pop('out_dir',None);specs.append(spec)
  receipts[mode]={'events_sha256':sha(base/'events.jsonl'),'spec_sha256':sha(sp),'step_count':len(steps)}
 if specs[0]!=specs[1]:raise ValueError('clean/profiled specifications differ')
 return receipts

def load_lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]

def archive(folder):
 out=folder/'raw_events.tar.gz';tmp=folder/'raw_events.tar.gz.incoming'
 patterns=['*.json','HALTED','cell_logs/*.log','runs/*_spec.json','runs/*/events.jsonl','runs/*/result.json']
 files=sorted({p for pattern in patterns for p in folder.glob(pattern) if p.is_file()})
 with tarfile.open(tmp,'w:gz') as tar:
  for p in files:tar.add(p,arcname=str(p.relative_to(folder)),recursive=False)
 tmp.replace(out)

def run(folder,vocab,binary):
 folder=Path(folder).resolve();folder.mkdir(parents=True,exist_ok=True);vocab=Path(vocab).resolve();binary=Path(binary).resolve()
 statepath=folder/'wrapper_state.json';manifestpath=folder/'source_manifest.json'
 if (folder/'HALTED').exists():
  print('HALTED requires operator review; original failure receipt preserved',file=sys.stderr);return 1
 def environment():return {'source':source_inventory(),'binary_sha256':sha(binary),'vocab_sha256':sha(vocab),
  'gpu':gpu_identity(),'steps':STEPS,'warm':WARM,'run_timeout_s':900,'shapes':profile.SHAPES,'configs':list(CONFIGS),'env':profile.ENV}
 state=None
 try:
  if tuple(profile.SHAPES)!=SHAPES:raise ValueError('profiler shape inventory changed; expected exact ten shapes')
  frozen=environment()
  if statepath.exists():
   state=load(statepath)
   if load(manifestpath)!=frozen or validate_corpus(folder/'synth.ids')!=state['synth_sha256']:
    raise ValueError('frozen source/binary/GPU/settings/corpus changed')
  else:
   if any((folder/n).exists() for n in ('profile.json','synth.ids')):raise ValueError('unmanifested old profile/corpus refused')
   profile.synth_corpus(str(folder/'synth.ids'));digest=validate_corpus(folder/'synth.ids')
   atomic(manifestpath,frozen);state={'status':'RUNNING','created':time.time(),'synth_sha256':digest,'completed':{}}
   atomic(statepath,state)
  (folder/'cell_logs').mkdir(exist_ok=True)
  expected={f'{s}/{c}' for s in profile.SHAPES for c in CONFIGS}
  for shape in profile.SHAPES:
   for cfg in CONFIGS:
    key=f'{shape}/{cfg}'
    if environment()!=frozen or validate_corpus(folder/'synth.ids')!=state['synth_sha256']:
     raise ValueError('source/binary/settings changed between cells')
    if key in state['completed']:
     if validate_cell(folder,shape,cfg)!=state['completed'][key]:raise ValueError('completed cell changed: '+key)
     continue
    if (folder/'profile.json').exists() and key in load(folder/'profile.json').get('runs',{}):
     # Crash handoff: a complete unchanged cell can be accepted after strict validation.
     receipt=validate_cell(folder,shape,cfg)
    else:
     state.update(status='RUNNING',current=key);atomic(statepath,state)
     command=[sys.executable,str(ROOT/'tools/step_profile.py'),'run','--out',str(folder),'--vocab',str(vocab),
              '--shapes',shape,'--configs',cfg,'--steps',str(STEPS),'--warm',str(WARM),'--run-timeout','900']
     with (folder/'cell_logs'/f'{shape}_{cfg}.log').open('a') as log:
      subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,
                     env={**os.environ,'JOBS_MTSTUDIO':str(binary)})
     receipt=validate_cell(folder,shape,cfg)
    state['completed'][key]=receipt;atomic(statepath,state);archive(folder)
  if set(load(folder/'profile.json')['runs'])!=expected or set(state['completed'])!=expected:
   raise ValueError('not exactly40 requested cells')
  state.update(status='COMPLETE',completed_at=time.time());atomic(statepath,state)
  atomic(folder/'completion.json',{'status':'COMPLETE','cells':40,'steps':STEPS,'warm':WARM,
   'profile_sha256':sha(folder/'profile.json'),'source_manifest_sha256':sha(manifestpath),'synth_sha256':state['synth_sha256'],
   'scope':'descriptive per-shape throughput and instrumented op timing; not a new noise-floor gate',
   'negative_idle_cells':[k for k,v in load(folder/'profile.json')['runs'].items() if v['profile']['idle_ms']<0]})
  archive(folder);return 0
 except Exception as e:
  state=state or {'completed':{}};state.update(status='HALTED',error=str(e),checked=time.time());atomic(statepath,state)
  (folder/'HALTED').write_text(str(e)+'\n');archive(folder);print(str(e),file=sys.stderr);return 1

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);p.add_argument('--vocab',required=True);p.add_argument('--binary',required=True)
 a=p.parse_args();raise SystemExit(run(a.out,a.vocab,a.binary))
