"""Handcrafted source-literal backend proofs; no CUDA calls."""
import copy
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import parity_device_evidence_v1 as d

NATIVE_CUDA='\n'.join(['Effective CPU layers for this init: 0, Effective GPU layers for this init: 12','Initializing CUDA resources for 12 GPU layers.','cuBLAS handle created successfully.','Successfully loaded all concatenated F32 weights for GPU layers'])
NATIVE_CPU='Effective CPU layers for this init: 12, Effective GPU layers for this init: 0\n'+d.CPU_SKIP
IR_CUDA="[IR] CUDA device path: 80 weights (20 MiB f32), 12 kv slots of 512 positions on the device\nRunning the model through the paperkiln IR interpreter (CUDA; family_hint 'llama-tiny')"
IR_CPU="Running the model through the paperkiln IR interpreter (CPU; family_hint 'llama-tiny')"

@pytest.mark.parametrize('text,leg,expected',[(NATIVE_CUDA,'cuda','native_cuda'),(NATIVE_CPU,'cpu','native_cpu'),(IR_CUDA,'cuda','ir_cuda'),(IR_CPU,'cpu','ir_cpu')])
def test_positive_actual_routes(text,leg,expected):assert d.route(text,leg,12)==expected

@pytest.mark.parametrize('text,leg',[
 ('Effective CPU layers for this init: 0, Effective GPU layers for this init: 12','cuda'),
 (NATIVE_CUDA+'\nCUDA not available; Model will run entirely on CPU.','cuda'),
 (NATIVE_CUDA.replace('cuBLAS handle created successfully.',''),'cuda'),
 (NATIVE_CUDA.replace('Successfully loaded all concatenated F32 weights for GPU layers',''),'cuda'),
 (NATIVE_CUDA.replace('12','11'),'cuda'),(NATIVE_CPU,'cuda'),(NATIVE_CUDA,'cpu'),
 (IR_CUDA.replace('12 kv','11 kv'),'cuda'),(IR_CUDA.replace('[IR] CUDA device path:','unproven requested GPU:'),'cuda'),
 (IR_CPU,'cuda'),(IR_CUDA,'cpu'),(IR_CPU+'\n'+IR_CUDA,'cpu'),
 (NATIVE_CPU.replace(d.CPU_SKIP,''),'cpu'),('','cpu')])
def test_requested_layers_fallback_or_wrong_route_do_not_prove_device(text,leg):
    with pytest.raises(ValueError):d.route(text,leg,12)


def proof():return dict(kind='SOURCE_BOUND_CUDA_BINARY_V1',commit=d.COMMIT,cuda=True,source_sha256=copy.deepcopy(d.SOURCES),binary_sha256='a'*64,compile_flags='CXX_DEFINES = -DHAS_CUDA -DOTHER\n')

@pytest.mark.parametrize('defect',['source','commit','cuda','binary','flags'])
def test_build_source_and_cuda_definition_are_required(defect):
    p=proof()
    if defect=='source':p['source_sha256']['api.cpp']='b'*64
    if defect=='commit':p['commit']='b'*40
    if defect=='cuda':p['cuda']=False
    if defect=='binary':p['binary_sha256']=''
    if defect=='flags':p['compile_flags']='-DHAS_CUDA_OFF'
    with pytest.raises(ValueError):d.check_build(p,d.COMMIT)


def test_per_leg_startup_log_and_binary_binding(tmp_path):
    from pref_artifact_io_v1 import sha
    p=proof();s={'device_build':p}
    for leg,text in [('cuda',NATIVE_CUDA),('cpu',NATIVE_CPU)]:
        log=tmp_path/('ember_'+leg+'.log');log.write_text(text)
        s[leg]={'health':{'layers':12},'device_evidence':dict(binary_sha256=p['binary_sha256'],startup_log_sha256=sha(log),actual_route=d.route(text,leg,12))}
    d.verify(s,tmp_path,d.COMMIT)
    changed=copy.deepcopy(s);changed['cpu']['device_evidence']['binary_sha256']='b'*64
    with pytest.raises(ValueError):d.verify(changed,tmp_path,d.COMMIT)
    (tmp_path/'ember_cpu.log').write_text(NATIVE_CPU+' changed')
    with pytest.raises(ValueError):d.verify(s,tmp_path,d.COMMIT)
