"""Recompute recorded CLI lifecycle/export evidence without model execution."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'spec/tools'));sys.path.insert(0,str(ROOT/'tools'))
import ir_check
import ir_numpy
import flex_cli_lifecycle as contract
from replay_flex_precision import replay as model_replay

def replay(out,require_cuda=False):
    contract.require_reference_runtime()
    out=Path(out);summary=json.loads((out/'summary.json').read_text())
    if summary.get('status')!='PASS' or summary.get('actual_cuda') is not require_cuda:raise ValueError('incomplete/wrong backend summary')
    model=model_replay(json.loads((out/'model_gradient_raw.json').read_text()),require_cuda)
    cases=[]
    for objective,layers,hidden in [('lm',1,12),('sft',3,20)]:
        d=out/objective;initial=json.loads((d/'recipe_initial.json').read_text());recipe=json.loads((d/'recipe_resume.json').read_text());a=recipe['arch']['custom'];tr=recipe['train']
        if (a['d'],a['heads'],a['layers'],a['d_ff'],tr['batch'],tr['seed'],recipe['data']['T'])!=(8,2,layers,hidden,2,37,8):raise ValueError('registered geometry')
        if initial['train']['steps']!=3 or tr['steps']!=4 or tr['objective']!=objective or tr['attention_precision']!='bf16':raise ValueError('registered lifecycle')
        trained=d/'trained';events=[json.loads(x) for x in (trained/'events.jsonl').read_text().splitlines() if x.strip()];steps=contract.validate_steps(events,layers,require_cuda)
        resumes=[e for e in events if e.get('event')=='resume']
        if len(resumes)!=1 or resumes[0].get('step')!=3 or resumes[0].get('optimizer')!='restored':raise ValueError('warm resume')
        result=json.loads((trained/'result.json').read_text())
        if result['final_step']!=4 or not math.isfinite(result['best_val']):raise ValueError('final completion')
        first=d/'bf16_recipe.f32'
        if first.read_bytes()!=(d/'fp32_recipe.f32').read_bytes():raise ValueError('saved inference precision')
        meta,tensors,offset,raw,_=ir_check.read_gguf(trained/'flex_precision.gguf')
        if meta['paperkiln.train.attention_precision'][1]!='bf16' or any(kind!=0 for _,_,kind,_ in tensors):raise ValueError('export metadata')
        doc=json.loads(meta['paperkiln.ir'][1]);weights={name:np.frombuffer(raw,dtype='<f4',count=int(np.prod(dims)),offset=offset+location).reshape(tuple(reversed(dims))) for name,dims,_,location in tensors}
        actual=np.fromfile(first,dtype='<f4').reshape(8,7);ref=np.concatenate([ir_numpy.forward(doc,weights,ids) for ids in [[1],[1,2,3],[6,5,4,3]]])
        if not np.isfinite(actual).all() or not np.isfinite(ref).all():raise ValueError('nonfinite export rows')
        error=float(np.max(abs(actual-ref)/(1+abs(ref))))
        if error>1e-5 or not np.array_equal(actual.argmax(1),ref.argmax(1)):raise ValueError('independent export rows')
        if (trained/'flex_precision.gguf').read_bytes()!=(d/'reload_export/flex_precision.gguf').read_bytes():raise ValueError('reload export')
        recorded=next(x for x in summary['cases'] if x['objective']==objective)
        if recorded['all_8x7_export_row_error']!=error or recorded['step_grad_norms']!=[e['grad_norm'] for e in steps]:raise ValueError('summary/raw mismatch')
        cases.append({'objective':objective,'steps':4,'row_error':error,'require_cuda':require_cuda})
    return {'status':'REPLAY_VERIFIED','actual_cuda':require_cuda,'cases':cases,'independent_model_gradient':model,'scope':'tiny canonical Flex lifecycle and export; no quality/speed/general variant claim'}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out');p.add_argument('--require-cuda',action='store_true');a=p.parse_args();print(json.dumps(replay(a.out,a.require_cuda),indent=2))
