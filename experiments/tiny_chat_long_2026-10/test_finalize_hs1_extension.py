import importlib.util,json,sys
from pathlib import Path
import pytest
HERE=Path(__file__).parent;sys.path.insert(0,str(HERE))
import finalize_hs1_extension as m

@pytest.fixture
def driver(tmp_path,monkeypatch):
 # Preserve real source inventory, isolate original-trigger I/O from fake artifacts.
 monkeypatch.setattr(m.preparation,'trigger',lambda r:{'verdict':'NOT_RESOLVED'})
 orig=m.common.load_json
 def load(p):
  p=Path(p)
  if p==m.HERE/'receipts/core_2026-10-30/completion.json':return {'input_manifest':{'sha256':{}}}
  return orig(p)
 monkeypatch.setattr(m.common,'load_json',load)
 art=tmp_path/'art';art.mkdir()
 pin=orig(m.HERE/'hs1_extension_launch_pin.json')['commit']
 for c in m.CELLS:
  for s in m.SEEDS:
   base=art/f'c12_{c}_r{s}'
   for part in ('data','gate',f'post_c12_{c}_r{s}'):(base/part).mkdir(parents=True)
   m.common.atomic_json(base/'data/data.ledger.json',{'hash':'unchanged','args':{'seed':1}})
   m.common.atomic_json(base/'gate/gate.json',{'ok':True,'problems':[]})
   row={'mode':'post','cell':c,'seed':s,'Y12':(-.1 if c=='B' else -.2)-s*.001,
        'provenance':{'paperkiln_commit':pin},'carry12':{'context':1024,'pools':{'acc':{
          'primary':{'twin':{'mean':.6}},'context':{'twin':{'mean':.6},'noctx':{'mean':.3}}}}}}
   m.common.atomic_json(base/f'post_c12_{c}_r{s}/summary.json',row)
 return m.Finalizer(art,tmp_path/'state',tmp_path/'out')

def test_exact_ten_ready(driver):
 i=driver.inputs();assert i['ready'] and len(i['selected_cell_seed_pairs'])==10

def test_missing_seed_waits(driver):
 (driver.art/'c12_B_r5/post_c12_B_r5/summary.json').unlink()
 assert driver.tick()=='WAITING'
 assert not (driver.output/'reports').exists()

def test_ledger_mismatch_halts(driver):
 m.common.atomic_json(driver.art/'c12_B_r5/data/data.ledger.json',{'hash':'changed'})
 assert driver.tick()=='HALTED'
 assert (driver.home/'HALTED').is_file()

def test_wrong_pin_halts(driver):
 p=driver.art/'c12_B_r5/post_c12_B_r5/summary.json';r=m.common.load_json(p);r['provenance']['paperkiln_commit']='other';m.common.atomic_json(p,r)
 assert driver.tick()=='HALTED'

def test_duplicate_halts(driver):
 p=driver.art/'copy/post_c12_B_r5';p.mkdir(parents=True);(p/'summary.json').write_bytes((driver.art/'c12_B_r5/post_c12_B_r5/summary.json').read_bytes())
 assert driver.tick()=='HALTED'

def test_source_change_halts(driver,tmp_path,monkeypatch):
 p=tmp_path/'source';p.write_text('old');monkeypatch.setattr(m,'SOURCES',[p]);driver.state();p.write_text('new')
 assert driver.tick()=='HALTED'

def test_exact_helper_analysis_and_reject_partial(driver,tmp_path):
 out=tmp_path/'report.json';m.calculate(driver.art,out);r=m.common.load_json(out)
 assert r['seeds']==[1,2,3,4,5] and r['HS1_Sgist_minus_B']['n']==5
 assert r['HS1_Sgist_minus_B']['verdict']=='SUPPORTED'
 stage=tmp_path/'published';stage.mkdir();m.common.atomic_json(stage/m.REPORT,r);driver.verify_outputs(stage)
 r['seeds']=[1,2,3];m.common.atomic_json(stage/m.REPORT,r)
 with pytest.raises(ValueError):driver.verify_outputs(stage)

def test_publication_handoff(driver,monkeypatch):
 def run(cmd,**kw):
  tail=cmd[cmd.index('--')+1:];m.calculate(tail[tail.index('--art')+1],tail[tail.index('--out')+1])
 monkeypatch.setattr(m.subprocess,'run',run)
 assert driver.tick()=='COMPLETE'
 state=m.common.load_json(driver.state_path);state['status']='ANALYSING';m.common.atomic_json(driver.state_path,state)
 assert driver.tick()=='COMPLETE'
 p=driver.output/'reports'/m.REPORT;p.write_text('{}')
 assert driver.tick()=='HALTED'
