import copy
import importlib.util
import json
from pathlib import Path
import pytest
P=Path(__file__).with_name('prepare_hs1_extension.py')
s=importlib.util.spec_from_file_location('hs1_prepare',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

@pytest.mark.parametrize('cell',m.CELLS)
@pytest.mark.parametrize('seed',m.SEEDS)
def test_exact_frozen_semantic_delta(cell,seed):
 path,new,source=m.build(cell,seed)
 old=json.loads(m.frozen(source));old.pop('_comment',None);new.pop('_comment',None)
 keep={'cuda_gate','data','gate',f'pt_c12_{cell}_r1',f'post_c12_{cell}_r1'}
 old['jobs']=[j for j in old['jobs'] if j['name'] in keep]
 restored=json.loads(json.dumps(new).replace(f'_{cell}_r{seed}',f'_{cell}_r1'))
 for j in restored['jobs']:
  if j['name']=='gate':j['cmd'][j['cmd'].index('--jobs')+1]=source
  if j['name']==f'pt_c12_{cell}_r1':j['spec']['train']['seed']=1
  if j['name']==f'post_c12_{cell}_r1':j['cmd'][j['cmd'].index('--seed')+1]='1'
 assert restored==old
 data=next(j for j in new['jobs'] if j['name']=='data')
 assert data['cmd'][data['cmd'].index('--seed')+1]=='1'
 assert len(new['jobs'])==5

def report():
 return json.loads((m.ROOT/m.REL/'receipts/core_2026-10-30/analysis_carry12.json').read_text())

def test_real_trigger():
 assert m.trigger(report())['sign_agreement']=='3/3'

@pytest.mark.parametrize('kind',['partial','supported','mixed','floor','sfull'])
def test_invalid_trigger_rejected(kind):
 r=report();h=r['carry12']['HS1_Sgist_minus_B']
 if kind=='partial':r['partial']=True
 if kind=='supported':h['verdict']='SUPPORTED'
 if kind=='mixed':h['per_seed'][0]=.1
 if kind=='floor':r['carry12']['validity_gate']['B']=False
 if kind=='sfull':r['carry12']['s_full_condition']['run_s_full']=True
 with pytest.raises(ValueError):m.trigger(r)

def test_only_registered_arms():
 with pytest.raises(ValueError):m.build('TS',4)
 with pytest.raises(ValueError):m.build('B',6)

def test_deterministic_immutable_outputs(tmp_path):
 one=m.write(tmp_path);two=m.write(tmp_path);assert one==two
 p=tmp_path/next(iter(one['jobs']));p.write_text('{}')
 with pytest.raises(ValueError,match='overwrite'):m.write(tmp_path)
