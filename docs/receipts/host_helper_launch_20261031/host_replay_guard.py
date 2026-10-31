#!/usr/bin/env python3
"""Mandatory source-bound post-home replay of both registered backends."""
import hashlib,json,os,subprocess
from pathlib import Path
STAGE=Path('/home/jonat/host_helper_gate_stage_unlaunched_v3_20261031')
LAND=Path('/home/jonat/pk_host_helper_launch_20261031/docs/receipts/host_helper_cuda_20261031/landed_vm')
OUT=LAND.parent/'host_replay'
PYTHON='/home/jonat/pk_bf16_replay_20261031/bin/python'
PINS={'/home/jonat/host_helper_gate_stage_unlaunched_v3_20261031/inputs.json': '3ab069e1f144aae1461a1a0c6bb3c1bfd584b25d8f1e6fe41d7caba1a7458ed7', '/home/jonat/host_helper_gate_stage_unlaunched_v3_20261031/replay_host_helper_gate.py': '8d7d7962a88372734cf6af31539ca8254d3954db708d503646122a9797e722a0'}
def verify():
    observed={}
    for path,want in PINS.items():
        got=hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if got!=want:raise ValueError('frozen host pin changed: '+path)
        observed[path]=got
    return observed
def main():
    before=verify()
    if OUT.exists():raise ValueError('immutable host replay output exists')
    OUT.mkdir()
    version=subprocess.check_output([PYTHON,'-c','import numpy; print(numpy.__version__)'],text=True).strip()
    if version!='1.26.4':raise ValueError('frozen NumPy identity mismatch')
    records=[]
    for mode in ['cpu','cuda']:
        argv=[PYTHON,'-O',str(STAGE/'replay_host_helper_gate.py'),str(LAND/(mode+'.json'))]
        if mode=='cuda':argv.append('--require-cuda')
        p=subprocess.run(argv,capture_output=True,text=True,timeout=60)
        (OUT/(mode+'.stdout')).write_text(p.stdout)
        (OUT/(mode+'.stderr')).write_text(p.stderr)
        if p.returncode:raise ValueError(mode+' replay failed')
        result=json.loads(p.stdout)
        if result.get('pass') is not True or result.get('actual_cuda') is not (mode=='cuda'):raise ValueError('replay backend/result mismatch')
        records.append({'mode':mode,'argv':argv,'returncode':p.returncode,'result':result})
    after=verify()
    (OUT/'result.json').write_text(json.dumps({'pass':True,'numpy':version,'before':before,'after':after,'records':records},indent=2)+'\n')
if __name__=='__main__':main()
