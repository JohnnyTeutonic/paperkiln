#!/usr/bin/env python3
"""Frozen host replay operational guard; no scientific thresholds are changed."""
import argparse, hashlib, json, os, subprocess
from pathlib import Path
STAGE=Path('/home/jonat/mamba1_cli_gate_stage_v1_20261031')
SOURCE=Path('/home/jonat/mamba1_cli_archive_preflight_20261031')
PYTHON='/home/jonat/pk_bf16_replay_20261031/bin/python'
PINS={'/home/jonat/mamba1_cli_gate_stage_v1_20261031/host_pins.json': '5de70f399efda83d461c1304504b60404115727388245fca83f431a22d2f073f', '/home/jonat/mamba1_cli_gate_stage_v1_20261031/inputs.json': 'cc95af6e789f28aecae56bed3fc8c0749984de918cc7de6cbcc97f5eca15b311'}
RECIPE={'steps':3,'batch':2,'seed':37,'lr':0.001,'eval_every':3,'checkpoint_every':3,'gradmap_every':0}
def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def verify():
    for path, expected in PINS.items():
        if digest(path)!=expected: raise ValueError('stage pin drift: '+path)
    pins=json.loads((STAGE/'host_pins.json').read_text())
    observed={}
    for rel, expected in pins['source_files'].items():
        actual=digest(SOURCE/rel)
        if actual!=expected: raise ValueError('host source drift: '+rel)
        observed[rel]=actual
    if digest(STAGE/'mamba1_cli_lifecycle.py')!=pins['source_files']['tools/mamba1_cli_lifecycle.py']: raise ValueError('uploaded helper drift')
    if digest(pins['ember_binary'])!=pins['ember_binary_sha256']: raise ValueError('Ember binary drift')
    return observed
def check_recipe(artifact):
    for depth in (1,2):
        spec=json.loads((artifact/f'depth{depth}/spec.json').read_text())
        if spec['train']!=RECIPE: raise ValueError('registered training recipe mismatch')
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--artifact',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--receipt',type=Path,required=True)
    a=ap.parse_args()
    if a.receipt.exists(): raise ValueError('receipt exists')
    before=verify(); check_recipe(a.artifact)
    pins=json.loads((STAGE/'host_pins.json').read_text())
    argv=[PYTHON,'-O',str(STAGE/'mamba1_cli_lifecycle.py'),'replay','--artifact',str(a.artifact),'--ember',pins['ember_binary'],'--out',str(a.out)]
    env=dict(os.environ,MAMBA1_CLI_SOURCE_ROOT=str(SOURCE))
    result=subprocess.run(argv,env=env,capture_output=True,text=True)
    after=verify()
    record={'argv':argv,'source_root':str(SOURCE),'before':before,'after':after,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'pass':result.returncode==0}
    a.receipt.write_text(json.dumps(record,indent=2)+'\n')
    if result.returncode: raise RuntimeError('frozen host replay failed; see receipt')
if __name__=='__main__': main()
