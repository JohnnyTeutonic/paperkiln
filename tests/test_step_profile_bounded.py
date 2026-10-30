"""Offline contracts for the bounded profiler; no CUDA, subprocess training or model calls."""
import importlib.util,json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import step_profile_bounded as m

def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))

@pytest.fixture
def cell(tmp_path):
 shape,cfg='20m_L4','bf16_flash'
 r={'done':True,'clean_rc':0,'prof_rc':0,'clean_tok_per_s':100,'clean_s_per_step':1,
    'profile':{'steps':22,'wall_ms':100,'gpu_ms':80,'idle_ms':20,'peak_mb':1}}
 write(tmp_path/'profile.json',{'runs':{shape+'/'+cfg:r}})
 for mode in ('clean','prof'):
  root=tmp_path/'runs'/f'{shape}_{cfg}_{mode}';root.mkdir(parents=True)
  events=[{'event':'model','precision':'bf16','attention_kernel':'flash'}]
  events += [{'event':'step','step':i,'loss':1,'t_ms':{'forward':1},'gpu_ms':{'gemm':1},'gpu_n':{'gemm':1}} for i in range(1,31)]
  events += [{'event':'done','final_step':30,'early_stopped':False}]
  (root/'events.jsonl').write_text('\n'.join(json.dumps(e) for e in events))
  write(tmp_path/'runs'/f'{shape}_{cfg}_{mode}_spec.json',{'out_dir':str(root),'train':{'precision':'bf16','attention_kernel':'flash','steps':30}})
 return tmp_path,shape,cfg

def test_full_both_pass_receipt(cell):
 p,s,c=cell;assert set(m.validate_cell(p,s,c))=={'clean','prof','profile_cell_sha256'}

@pytest.mark.parametrize('kind',['refused','stalled','nonzero','nonfinite','insufficient_profile'])
def test_failed_partial_profile_refused(cell,kind):
 p,s,c=cell;v=json.loads((p/'profile.json').read_text());r=v['runs'][s+'/'+c]
 if kind=='refused':r['refused']='unsupported'
 if kind=='stalled':r['stalled']='prof'
 if kind=='nonzero':r['prof_rc']=1
 if kind=='nonfinite':r['clean_tok_per_s']=float('nan')
 if kind=='insufficient_profile':r['profile']['steps']=21
 write(p/'profile.json',v)
 with pytest.raises(ValueError):m.validate_cell(p,s,c)

@pytest.mark.parametrize('mode',['clean','prof'])
def test_both_pass_model_settings_required(cell,mode):
 p,s,c=cell;f=p/'runs'/f'{s}_{c}_{mode}'/'events.jsonl';rows=m.load_lines(f);rows[0]['precision']='fp32';f.write_text('\n'.join(json.dumps(x) for x in rows))
 with pytest.raises(ValueError,match='settings'):m.validate_cell(p,s,c)

@pytest.mark.parametrize('kind',['missing_step','duplicate_step','nan_loss','missing_timing','bad_finalstep'])
def test_raw_event_contract(cell,kind):
 p,s,c=cell;f=p/'runs'/f'{s}_{c}_prof'/'events.jsonl';rows=m.load_lines(f)
 if kind=='missing_step':rows.pop(10)
 if kind=='duplicate_step':rows[10]['step']=9
 if kind=='nan_loss':rows[10]['loss']=float('inf')
 if kind=='missing_timing':rows[10].pop('gpu_ms')
 if kind=='bad_finalstep':rows[-1]['final_step']=29
 f.write_text('\n'.join(json.dumps(x) for x in rows))
 with pytest.raises(ValueError):m.validate_cell(p,s,c)

def test_spec_changes_reject(cell):
 p,s,c=cell;f=p/'runs'/f'{s}_{c}_prof_spec.json';v=json.loads(f.read_text());v['train']['steps']=31;write(f,v)
 with pytest.raises(ValueError,match='specifications'):m.validate_cell(p,s,c)

def test_archive_keeps_raw_but_excludes_weights_corpus(cell):
 p,_,_=cell;(p/'synth.ids').write_bytes(b'large');(p/'runs/x').mkdir();(p/'runs/x/model.safetensors').write_bytes(b'weights')
 m.archive(p)
 import tarfile
 with tarfile.open(p/'raw_events.tar.gz') as t:
  names=t.getnames();assert any(x.endswith('events.jsonl') for x in names);assert not any(x.endswith(('.ids','.safetensors')) for x in names)

def test_halted_receipt_never_overwritten(tmp_path):
 (tmp_path/'HALTED').write_text('original');(tmp_path/'wrapper_state.json').write_text('{"error":"original","completed":{"x":1}}')
 assert m.run(tmp_path,tmp_path/'absent',tmp_path/'absent')==1
 assert json.loads((tmp_path/'wrapper_state.json').read_text())['completed']=={'x':1}

def test_changed_source_and_binary_refuse_before_train(tmp_path,monkeypatch):
 vocab=tmp_path/'vocab';vocab.write_bytes(b'v');binary=tmp_path/'binary';binary.write_bytes(b'b')
 out=tmp_path/'profile';out.mkdir();write(out/'wrapper_state.json',{'synth_sha256':'original','completed':{}});write(out/'source_manifest.json',{'different':True})
 monkeypatch.setattr(m,'source_inventory',lambda:{'commit':'changed','files':{}});monkeypatch.setattr(m,'gpu_identity',lambda:'L4')
 assert m.run(out,vocab,binary)==1
 assert (out/'HALTED').is_file()

def test_complete_resume_handoff_and_timing_tamper(tmp_path,monkeypatch):
 out=tmp_path/'out';vocab=tmp_path/'v';vocab.write_bytes(b'vocab');binary=tmp_path/'bin';binary.write_bytes(b'binary')
 monkeypatch.setattr(m,'source_inventory',lambda:{'commit':'fixed','files':{'source':'hash'}})
 monkeypatch.setattr(m,'gpu_identity',lambda:'L4,UUID,driver')
 monkeypatch.setattr(m.profile,'synth_corpus',lambda p:Path(p).write_bytes(b'synth'))
 monkeypatch.setattr(m,'validate_corpus',lambda p:m.sha(p))
 calls=[]
 def fake_run(command,**kwargs):
  shape=command[command.index('--shapes')+1];cfg=command[command.index('--configs')+1];calls.append((shape,cfg))
  folder=Path(command[command.index('--out')+1]);p=folder/'profile.json';value=json.loads(p.read_text()) if p.exists() else {'runs':{}}
  value['runs'][shape+'/'+cfg]={'done':True,'clean_rc':0,'prof_rc':0,'clean_tok_per_s':100,'clean_s_per_step':1,
    'profile':{'steps':22,'wall_ms':100,'gpu_ms':80,'idle_ms':20,'peak_mb':1}}
  write(p,value)
  for mode in ('clean','prof'):
   base=folder/'runs'/f'{shape}_{cfg}_{mode}';base.mkdir(parents=True)
   settings=m.profile.CONFIGS[cfg]
   rows=[{'event':'model',**settings}]+[{'event':'step','step':i,'loss':1,'t_ms':{'fwd':1},'gpu_ms':{'gemm':1}} for i in range(1,31)]+[{'event':'done','final_step':30}]
   (base/'events.jsonl').write_text('\n'.join(json.dumps(x) for x in rows))
   write(folder/'runs'/f'{shape}_{cfg}_{mode}_spec.json',{'train':{'steps':30,**settings}})
 monkeypatch.setattr(m.subprocess,'run',fake_run)
 assert m.run(out,vocab,binary)==0 and len(calls)==40
 assert json.loads((out/'completion.json').read_text())['cells']==40
 # A crash after a durable profile cell but before the wrapper checkpoint is recovered without another run.
 state=json.loads((out/'wrapper_state.json').read_text());state['completed'].pop(next(iter(state['completed'])));write(out/'wrapper_state.json',state)
 assert m.run(out,vocab,binary)==0 and len(calls)==40
 # A stored finite timing changed behind an otherwise intact raw event stream is detected.
 value=json.loads((out/'profile.json').read_text());next(iter(value['runs'].values()))['clean_tok_per_s']=200;write(out/'profile.json',value)
 assert m.run(out,vocab,binary)==1 and len(calls)==40
 assert 'changed' in json.loads((out/'wrapper_state.json').read_text())['error']

def test_first_failed_cell_stops_campaign_and_archives(tmp_path,monkeypatch):
 v=tmp_path/'v';v.write_bytes(b'v');b=tmp_path/'b';b.write_bytes(b'b');out=tmp_path/'out';calls=[]
 monkeypatch.setattr(m,'source_inventory',lambda:{'commit':'fixed','files':{}});monkeypatch.setattr(m,'gpu_identity',lambda:'L4')
 monkeypatch.setattr(m.profile,'synth_corpus',lambda p:Path(p).write_bytes(b's'));monkeypatch.setattr(m,'validate_corpus',lambda p:m.sha(p))
 def fail(command,**kwargs):
  calls.append(command);write(out/'profile.json',{'runs':{'20m_L2/fp32':{'done':True,'clean_rc':None,'stalled':'clean','stall':{'last_step':3}}}})
 monkeypatch.setattr(m.subprocess,'run',fail)
 assert m.run(out,v,b)==1 and len(calls)==1
 assert (out/'raw_events.tar.gz').is_file() and (out/'HALTED').is_file()
 assert not (out/'completion.json').exists()

def test_synthetic_header_and_size_checked(tmp_path):
 import struct
 p=tmp_path/'synth.ids';p.write_bytes(b'PKIDS001'+struct.pack('<IIQ',2,16384,8_000_000))
 with pytest.raises(ValueError,match='header/size'):m.validate_corpus(p)
 with p.open('ab') as f:f.truncate(16_000_024)
 assert len(m.validate_corpus(p))==64
 p.write_bytes(b'wrong')
 with pytest.raises(ValueError,match='header/size'):m.validate_corpus(p)
