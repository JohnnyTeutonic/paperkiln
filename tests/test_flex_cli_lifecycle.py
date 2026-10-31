import importlib.util
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('gate',ROOT/'tools/flex_cli_lifecycle.py');gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
def events():return [{'event':'step','step':i,'loss':1.,'grad_norm':.3,'gpu_n':{'gemm_bf16':12}} for i in range(1,5)]
def test_exact_actual_device_launch_inventory():assert len(gate.validate_steps(events(),1,True))==4
@pytest.mark.parametrize('mutation',['missing','duplicate','zero_grad','nan','bool_count','negative_count','wrong_count','no_profile'])
def test_invalid_lifecycle_rejected(mutation):
 e=events()
 if mutation=='missing':e.pop()
 if mutation=='duplicate':e[-1]['step']=3
 if mutation=='zero_grad':e[0]['grad_norm']=0
 if mutation=='nan':e[0]['loss']=float('nan')
 if mutation=='bool_count':e[0]['gpu_n']['gemm_bf16']=True
 if mutation=='negative_count':e[0]['gpu_n']['other']=-1
 if mutation=='wrong_count':e[0]['gpu_n']['gemm_bf16']=11
 if mutation=='no_profile':e[0].pop('gpu_n')
 with pytest.raises(ValueError):gate.validate_steps(e,1,True)


def test_vm_manifest_failure_cannot_reach_build_or_device(tmp_path):
 import json,subprocess,sys
 upload=tmp_path/'upload';upload.mkdir();(upload/'inputs.json').write_text('{}')
 out=tmp_path/'out';p=subprocess.run([sys.executable,str(ROOT/'tools/flex_cli_lifecycle_vm.py'),'--upload',str(upload),'--manifest-sha256','wrong','--out',str(out)],capture_output=True)
 summary=json.loads((out/'summary.json').read_text())
 assert p.returncode==1 and summary['commands']==[] and summary['complete'] is False and (out/'out.zip').is_file()

def test_vm_prior_evidence_not_overwritten(tmp_path):
 import subprocess,sys
 out=tmp_path/'out';out.mkdir();proof=out/'summary.json';proof.write_text('negative evidence')
 p=subprocess.run([sys.executable,str(ROOT/'tools/flex_cli_lifecycle_vm.py'),'--upload',str(tmp_path),'--manifest-sha256','wrong','--out',str(out)],capture_output=True)
 assert p.returncode!=0 and proof.read_text()=='negative evidence' and len(list(out.iterdir()))==1


def test_reference_runtime_binding(monkeypatch):
 monkeypatch.setattr(gate.np,'__version__','2.3.3')
 with pytest.raises(ValueError,match='requires NumPy 1.26.4'):gate.require_reference_runtime()
 monkeypatch.setattr(gate.np,'__version__','1.26.4')
 gate.require_reference_runtime()
