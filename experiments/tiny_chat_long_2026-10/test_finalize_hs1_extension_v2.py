import copy,json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parent))
import finalize_hs1_extension_v2 as m

def proof():return m.common.load_json(m.PROOF)
def reference():return m.common.load_json(m.RECEIPTS/'reference_data.ledger.json')
def resumed():return m.common.load_json(m.RECEIPTS/'resumed_data.ledger.json')

def test_actual_proof_and_exact_hashes():
 m.cached_exception(m.RECEIPTS/'resumed_data.ledger.json',m.RECEIPTS/'reference_data.ledger.json',resumed(),reference())

@pytest.mark.parametrize('kind',['source','output','decontam','eval','fetch','extra_omission','changed_arg','proofidentity'])
def test_any_mismatch_rejects(kind):
 p=proof();r=reference();v=resumed()
 if kind=='source':p['source_dialogues_sha256'][next(iter(p['source_dialogues_sha256']))]='bad'
 if kind=='output':p['outputs']['mix_s35.ids']['sha256']='bad'
 if kind=='decontam':p['decontam_report']['rows_dropped']+=1
 if kind=='eval':p['eval_sha256']['carry12_v1.jsonl']='bad'
 if kind=='fetch':p['fetch_ledger']['sources'].pop('oasst')
 if kind=='extra_omission':v.pop('long_sft')
 if kind=='changed_arg':v['args']['seed']=99
 if kind=='proofidentity':p['resumed_ledger_sha256']='bad'
 with pytest.raises(ValueError):m.verify_proof(p,r,v)

def test_different_ledger_hash_even_operational_path_refuses(tmp_path):
 p=tmp_path/'ledger';v=resumed();v['args']['out']='different';p.write_text(json.dumps(v))
 with pytest.raises(ValueError,match='exact cached'):m.cached_exception(p,m.RECEIPTS/'reference_data.ledger.json',v,reference())

def test_missing_or_mutated_proof_refuses(tmp_path,monkeypatch):
 p=tmp_path/'proof';monkeypatch.setattr(m,'PROOF',p)
 with pytest.raises(ValueError):m.cached_exception(m.RECEIPTS/'resumed_data.ledger.json',m.RECEIPTS/'reference_data.ledger.json',resumed(),reference())
 p.write_bytes(m.RECEIPTS.joinpath('cached_data_audit_20261030.json').read_bytes()+b' ')
 with pytest.raises(ValueError):m.cached_exception(m.RECEIPTS/'resumed_data.ledger.json',m.RECEIPTS/'reference_data.ledger.json',resumed(),reference())

def test_reference_hash_must_match(tmp_path):
 p=tmp_path/'reference';p.write_text(json.dumps(reference()))
 with pytest.raises(ValueError):m.cached_exception(m.RECEIPTS/'resumed_data.ledger.json',p,resumed(),reference())

# Reuse the isolated synthetic ten-report layout; no live campaign artifacts needed.
from test_finalize_hs1_extension import driver

def v2_driver(driver):
 for seed in m.SEEDS:
  path=driver.art/f'c12_Sgist_r{seed}/data/data.ledger.json'
  path.write_bytes((m.RECEIPTS/('resumed_data.ledger.json' if seed==5 else 'reference_data.ledger.json')).read_bytes())
 return m.Finalizer(driver.art,driver.home.parent/'v2state',driver.output.parent/'v2out')

def test_real_exception_inside_complete_input_audit(driver):
 f=v2_driver(driver);value=f.inputs();assert value['ready'] and len(value['selected_cell_seed_pairs'])==10
 assert value['operational_exception']['v1_halt_preserved'] is True
 assert str(m.PROOF) in value['sha256']

def test_different_ledger_hash_halts_driver(driver):
 f=v2_driver(driver);p=f.art/'c12_Sgist_r5/data/data.ledger.json';p.write_bytes(p.read_bytes()+b' ')
 assert f.tick()=='HALTED'

def test_missing_proof_halts_driver(driver,tmp_path,monkeypatch):
 f=v2_driver(driver);monkeypatch.setattr(m,'PROOF',tmp_path/'missing');monkeypatch.setattr(m,'SOURCES',[Path(__file__)])
 assert f.tick()=='HALTED'

def test_historical_builder_blob_is_frozen_not_live():
 assert m.common.sha(m.HISTORICAL_BUILDER)==m.EXPECTED_BUILDER
 assert m.HISTORICAL_BUILDER in m.SOURCES
 assert not any(p.resolve()==m.LIVE_BUILDER.resolve() for p in m.SOURCES)
