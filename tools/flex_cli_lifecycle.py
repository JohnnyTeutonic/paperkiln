"""Prospective plain-Flex CLI lifecycle gate, independent of previous frozen gates."""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'spec/tools'))
sys.path.insert(0,str(ROOT/'tests'))
import ir_check
import ir_numpy
spec=importlib.util.spec_from_file_location('cli_fixture',ROOT/'tests/test_flex_attention_precision_cli.py')
fixture_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture_module)


def require_reference_runtime():
    if np.__version__ != '1.26.4':
        raise ValueError('reference runtime requires NumPy 1.26.4; found '+np.__version__)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def validate_steps(events,layers,require_cuda,through=4):
    steps=[e for e in events if e.get('event')=='step']
    if [e.get('step') for e in steps]!=list(range(1,through+1)):raise ValueError('missing/repeated/reordered CLI steps')
    for e in steps:
        if not all(type(e.get(k)) in (int,float) and math.isfinite(e[k]) for k in ['loss','grad_norm']):raise ValueError('nonfinite step')
        if e['grad_norm']<=0:raise ValueError('no gradient signal')
        if require_cuda:
            counts=e.get('gpu_n',{})
            if not counts or any(type(v) is not int or v<0 for v in counts.values()):raise ValueError('invalid actual GPU profile')
            if counts.get('gemm_bf16',0)!=layers*2*6:raise ValueError('BF16 training launch count')
    return steps


def run(binary,model_gate,out,require_cuda):
    require_reference_runtime()
    binary,model_gate,out=Path(binary).resolve(),Path(model_gate).resolve(),Path(out).resolve()
    out.mkdir(exist_ok=False,parents=True)
    env={k:v for k,v in os.environ.items() if not k.startswith('MICROTORCH_')}
    env.update(MICROTORCH_DEVICE='cuda' if require_cuda else 'cpu',MICROTORCH_DEVICE_OPS='1' if require_cuda else '0',MICROTORCH_STEP_RESIDENCY='1' if require_cuda else '0',MICROTORCH_DEFER_DOWNLOADS='1' if require_cuda else '0',MICROTORCH_TF32='0',MICROTORCH_GPU_PROFILE='1' if require_cuda else '0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    commands=[]
    def command(label,args,timeout=120):
        proc=subprocess.run(list(map(str,args)),env=env,capture_output=True,text=True,timeout=timeout)
        (out/(label+'.stdout')).write_text(proc.stdout);(out/(label+'.stderr')).write_text(proc.stderr)
        commands.append({'label':label,'argv':list(map(str,args)),'returncode':proc.returncode})
        if proc.returncode:raise ValueError('actual command failed '+label)
    report={'status':'INCOMPLETE','actual_cuda':require_cuda,'binary_sha256':sha(binary),'model_gate_sha256':sha(model_gate),'cases':[],'commands':commands}
    try:
        model=out/'model_gradient_raw.json';args=[model_gate,'--out',model]
        if require_cuda:args.append('--require-cuda')
        command('model_gradient',args)
        replay=[sys.executable,ROOT/'tools/replay_flex_precision.py',model]
        if require_cuda:replay.append('--require-cuda')
        command('independent_model_gradient',replay)
        for objective,layers,hidden in [('lm',1,12),('sft',3,20)]:
            directory=out/objective;s=fixture_module.fixture(directory,layers,hidden)
            s['train'].update(attention_precision='bf16',objective=objective)
            if objective=='sft':
                examples=directory/'examples.jsonl';examples.write_text(''.join(json.dumps({'segments':[{'text':'the cat','loss':False},{'text':'sat on the mat .','loss':True}]})+'\n' for _ in range(30)));s['data']['sft_examples']=str(examples)
            recipe=directory/'recipe_initial.json';recipe.write_text(json.dumps(s,indent=2)+'\n')
            command(objective+'_train',[binary,'run',recipe])
            trained=Path(s['out_dir']);initial=json.loads((trained/'result.json').read_text())
            if initial['final_step']!=3 or initial.get('train_attention_precision')!='bf16':raise ValueError('training completion/precision')
            for name in ['model.safetensors','optim.safetensors','state.txt']:
                if not (trained/name).is_file():raise ValueError('missing warm checkpoint '+name)
            checkpoints={name:sha(trained/name) for name in ['model.safetensors','optim.safetensors','state.txt']}
            s['train']['steps']=4;s['train']['eval_every']=4;s['train']['checkpoint_every']=4;recipe=directory/'recipe_resume.json';recipe.write_text(json.dumps(s,indent=2)+'\n')
            command(objective+'_resume',[binary,'run',recipe])
            events=[json.loads(line) for line in (trained/'events.jsonl').read_text().splitlines() if line.strip()]
            resume=[e for e in events if e.get('event')=='resume']
            if len(resume)!=1 or resume[0].get('step')!=3 or resume[0].get('optimizer')!='restored':raise ValueError('warm resume not verified')
            steps=validate_steps(events,layers,require_cuda)
            result=json.loads((trained/'result.json').read_text())
            if result['final_step']!=4 or not math.isfinite(result['best_val']):raise ValueError('resumed completion')
            fp=copy.deepcopy(s);fp['train']['attention_precision']='fp32';fp_recipe=directory/'fp32_recipe.json';fp_recipe.write_text(json.dumps(fp,indent=2)+'\n')
            sequences=[[1],[1,2,3],[6,5,4,3]];requests=directory/'requests.jsonl';requests.write_text(''.join(json.dumps({'prompt_ids':[ids[0]],'continuation_ids':ids[1:]+[2]})+'\n' for ids in sequences))
            for label,path in [('bf16_recipe',recipe),('fp32_recipe',fp_recipe)]:command(objective+'_'+label,[binary,'score',path,'--in',requests,'--out',directory/(label+'.jsonl'),'--logits',directory/(label+'.f32')])
            first=directory/'bf16_recipe.f32';second=directory/'fp32_recipe.f32'
            if first.read_bytes()!=second.read_bytes():raise ValueError('saved inference depends on train-only precision')
            actual=np.fromfile(first,dtype='<f4').reshape(8,7);gguf=trained/'flex_precision.gguf';meta,tensors,offset,raw,_=ir_check.read_gguf(gguf)
            if meta['paperkiln.train.attention_precision'][1]!='bf16' or any(kind!=0 for _,_,kind,_ in tensors):raise ValueError('F32 export/train precision metadata')
            doc=json.loads(meta['paperkiln.ir'][1]);weights={name:np.frombuffer(raw,dtype='<f4',count=int(np.prod(dims)),offset=offset+location).reshape(tuple(reversed(dims))) for name,dims,_,location in tensors}
            oracle=np.concatenate([ir_numpy.forward(doc,weights,ids) for ids in sequences])
            if not np.isfinite(actual).all() or not np.isfinite(oracle).all():raise ValueError('nonfinite exported/inference rows')
            error=float(np.max(abs(actual-oracle)/(1+abs(oracle))))
            if error>1e-5 or not np.array_equal(actual.argmax(1),oracle.argmax(1)):raise ValueError('independent saved/export reference')
            exported=directory/'reload_export';command(objective+'_reload_export',[binary,'export',recipe,'--formats','gguf','--out-dir',exported])
            if gguf.read_bytes()!=(exported/'flex_precision.gguf').read_bytes():raise ValueError('checkpoint reload export bytes')
            report['cases'].append({'objective':objective,'layers':layers,'d_ff':hidden,'steps':4,'warm_resume_from':3,'checkpoint_before_resume':checkpoints,'step_grad_norms':[e['grad_norm'] for e in steps],'bf16_gemm_counts':[e.get('gpu_n',{}).get('gemm_bf16',0) for e in steps],'all_8x7_export_row_error':error,'saved_FP32_inference_recipe_bits_equal':True,'reload_export_byte_equal':True,'export_sha256':sha(gguf)})
        report['status']='PASS'
    except BaseException as exc:
        report.update(status='ERROR_OR_INCOMPLETE',error=repr(exc))
    (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--binary',required=True);ap.add_argument('--model-gate',required=True);ap.add_argument('--out',required=True);ap.add_argument('--require-cuda',action='store_true');a=ap.parse_args();r=run(a.binary,a.model_gate,a.out,a.require_cuda);raise SystemExit(0 if r['status']=='PASS' else 1)
