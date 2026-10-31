"""Positive backend evidence for the reviewed Ember source, never inferred from requested layers."""
import json
from pathlib import Path
import re
import subprocess
from pref_artifact_io_v1 import sha

COMMIT='aa50a6112dfd706078e5c7849070f8ca5af620b4'
SOURCES={
 'api.cpp':'9c08ba7dbb220ece062c63a4c833431be16e2fd5e9529db2b59ffb8672936f59',
 'model.cpp':'ee9d5e4ada84a20ccf00e9d3bb5330deee0fd6f8fa84be89c081712c06bfaed6',
 'gpu_initialization.cpp':'33f154bf2a3ed9cde4100ca5494397ce2f0169a08c6eabfc31935a231d96b668',
 'server.cpp':'59d467ad253e063d6cb2d7202f2f7f05523e27b0820047947a1871da452e6f39',
 'graph_ir.cpp':'43c2d5b70b5d4510284a53a8afa0c703538fa508b76d757b92f938ec180c58cd',
 'weight_management.cpp':'e0f93a8b653ef925f068be8b01d210f9b82c21003557adc6adc91fc9968d3710',
 'CMakeLists.txt':'928e8a02f7b73772c779309151ea4b407edd2136228f9e635d1024a1057a317d'}
CPU_SKIP='No layers assigned to GPU (active_num_gpu_layers = 0). Cleaning up existing CUDA resources and skipping GPU initialization.'


def route(text,leg,layers):
    if leg not in ['cuda','cpu'] or type(layers) is not int or layers<=0:raise ValueError('backend leg/layer evidence differs')
    ir=re.findall(r"Running the model through the paperkiln IR interpreter \((CUDA|CPU); family_hint '[^'\n]*'\)",text)
    device=re.findall(r'\[IR\] CUDA device path: ([0-9]+) weights \([0-9]+ MiB f32\), ([0-9]+) kv slots of [0-9]+ positions on the device',text)
    if ir:
        if len(ir)!=1 or ir[0]!=('CUDA' if leg=='cuda' else 'CPU'):raise ValueError('actual IR backend differs')
        if leg=='cuda' and (len(device)!=1 or int(device[0][0])<=0 or int(device[0][1])!=layers):raise ValueError('IR CUDA allocation proof incomplete')
        if leg=='cpu' and device:raise ValueError('CPU IR has CUDA device allocations')
        return 'ir_'+leg
    effective=re.findall(r'Effective CPU layers for this init: ([0-9]+), Effective GPU layers for this init: ([0-9]+)',text)
    expected=[('0',str(layers))] if leg=='cuda' else [(str(layers),'0')]
    if effective!=expected:raise ValueError('actual native layer routing differs')
    if leg=='cuda':
        if ('CUDA not available' in text or CPU_SKIP in text or 'Model will run entirely on CPU' in text
            or 'Initializing CUDA resources for '+str(layers)+' GPU layers.' not in text
            or 'cuBLAS handle created successfully.' not in text
            or 'Successfully loaded all concatenated F32 weights for GPU layers' not in text):raise ValueError('native CUDA allocation/weight proof incomplete')
    elif CPU_SKIP not in text or 'Successfully loaded all concatenated F32 weights for GPU layers' in text or device:
        raise ValueError('native CPU zero-offload proof incomplete')
    return 'native_'+leg


def build_proof(build):
    binary=Path(build['bin']);source=binary.parent.parent/'src'
    if build.get('cuda') is not True or build.get('commit')!=COMMIT:raise ValueError('reviewed CUDA build identity differs')
    head=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if head!=COMMIT:raise ValueError('actual Ember checkout differs')
    subprocess.run(['git','-C',str(source),'diff','--quiet','HEAD'],check=True)
    hashes={name:sha(source/name) for name in SOURCES}
    if hashes!=SOURCES:raise ValueError('reviewed backend source differs')
    flags=(binary.parent/'CMakeFiles/tinyllama_core.dir/flags.make').read_text()
    if not re.search(r'(?<!\w)-DHAS_CUDA(?:\s|$)',flags):raise ValueError('CUDA compile definition missing')
    return {'kind':'SOURCE_BOUND_CUDA_BINARY_V1','commit':head,'cuda':True,'source_sha256':hashes,
      'binary_sha256':sha(binary),'compile_flags':flags}


def check_build(proof,commit):
    if (proof.get('kind')!='SOURCE_BOUND_CUDA_BINARY_V1' or proof.get('commit')!=commit or commit!=COMMIT
        or proof.get('cuda') is not True or proof.get('source_sha256')!=SOURCES
        or not re.fullmatch('[0-9a-f]{64}',proof.get('binary_sha256',''))
        or not re.search(r'(?<!\w)-DHAS_CUDA(?:\s|$)',proof.get('compile_flags',''))):raise ValueError('CUDA binary/source evidence incomplete')


def verify(summary,parity_dir,commit):
    proof=summary['device_build'];check_build(proof,commit)
    for leg in ['cuda','cpu']:
        record=summary[leg]['device_evidence'];path=Path(parity_dir)/('ember_'+leg+'.log')
        actual=route(path.read_text(),leg,summary[leg]['health']['layers'])
        if record!={'binary_sha256':proof['binary_sha256'],'startup_log_sha256':sha(path),'actual_route':actual}:raise ValueError('per-leg startup/binary evidence differs')
