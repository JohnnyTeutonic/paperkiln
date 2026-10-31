"""Frozen five-part preference acceptance; offline evidence replay, never model replacement."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from pref_artifact_io_v1 import sha
import heldout_gate_runner_v1 as heldout
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import chat_eval
import qa_eval
import chat_template as CT


def assess(metrics):
    rates=metrics['protocol_failure_rates']
    if set(rates)!={'frozen','everyday','ultrachat'}:raise ValueError('registered protocol sources incomplete')
    if any(not 0<=r<=1 for r in rates.values()):raise ValueError('invalid protocol rate')
    repeats=metrics['repeat_rates']
    if set(repeats)!={'frozen','ultrachat'}:raise ValueError('registered repeat sources incomplete')
    for values in repeats.values():
        if set(values)!={'reference','dpo'} or any(not 0<=r<=1 for r in values.values()):raise ValueError('invalid repeat rates')
    qa=metrics['qa_correct']
    if set(qa)!={'reference','dpo'} or any(type(n) is not int or not 0<=n<=60 for n in qa.values()):raise ValueError('invalid QA counts')
    parity=metrics['parity']
    if set(parity)!={'cuda','cpu'}:raise ValueError('registered parity legs incomplete')
    required={'n','reply_identical','token_count','stop_reason','prompt_tokens'}
    if any(set(leg)!=required or any(type(n) is not int or not 0<=n<=40 for n in leg.values()) for leg in parity.values()):raise ValueError('parity evidence incomplete')
    judged=metrics['judged_gate']
    if judged['unranked']!=0:raise ValueError('held-out judging incomplete, no gate verdict')
    share=judged['b_win_share'];lower=judged['b_win_share_wilson95'][0]
    gates={'protocol':all(r<=.10 for r in rates.values()),
      'repeats':all(values['dpo']<=values['reference'] for values in repeats.values()),
      'simple_qa':qa['dpo']>=qa['reference']-3,
      'ember_parity':all(all(n==40 for n in leg.values()) for leg in parity.values()),
      'judged_side_by_side':share is not None and share>.5 and lower>.5}
    return {'kind':'FROZEN_FIVE_PART_PREFERENCE_VERDICT_V1','gates':gates,'adoption_eligible':all(gates.values()),
      'automatic_model_replacement':False,'on_failure':'Preserve negative result and retain pk_sft_v1.'}


def coverage(rows,prompts,encode,max_new=64):
    expected={i for i,p in enumerate(prompts) if len(encode(CT.prompt([],p['prompt'])))+max_new<=256}
    indices=[r['prompt_index'] for r in rows]
    if len(indices)!=len(set(indices)) or set(indices)!=expected:raise ValueError('evaluation raw sample coverage differs from unchanged context eligibility')
    if not expected:raise ValueError('empty evaluation denominator')



def parity_coverage(cuda,cpu,prompts):
    """Recompute registered forty-prompt evidence, including the shared CPU reference."""
    if len(prompts)!=40:raise ValueError('registered parity inventory differs')
    reference_fields=['mtstudio','mts_tokens','mts_stop','bpe_py_prompt_tokens']
    required=['i','prompt','ember','mtstudio','ember_tokens','mts_tokens','ember_stop','mts_stop','ember_prompt_tokens','bpe_py_prompt_tokens']
    for rows in [cuda,cpu]:
        if len(rows)!=40:raise ValueError('parity prompt coverage incomplete')
        for i,(row,prompt) in enumerate(zip(rows,prompts)):
            if (any(key not in row for key in required) or type(row['i']) is not int or row['i']!=i
                or row['prompt']!=prompt['prompt']):raise ValueError('parity registered prompt order/identity differs')
            if any(type(row[k]) is not int or row[k]<0 for k in ['ember_tokens','mts_tokens','ember_prompt_tokens','bpe_py_prompt_tokens']):raise ValueError('parity token evidence differs')
            if any(type(row[k]) is not str for k in ['ember','mtstudio','ember_stop','mts_stop']):raise ValueError('parity textual evidence differs')
    for left,right in zip(cuda,cpu):
        if any(left[k]!=right[k] for k in reference_fields) or right.get('ember_cuda')!=left['ember']:
            raise ValueError('parity CPU/CUDA reference correspondence differs')
    return {leg:{'n':len(rows),'reply_identical':sum(r['ember']==r['mtstudio'] for r in rows),
      'token_count':sum(r['ember_tokens']==r['mts_tokens'] for r in rows),'stop_reason':sum(r['ember_stop']==r['mts_stop'] for r in rows),
      'prompt_tokens':sum(r['ember_prompt_tokens']==r['bpe_py_prompt_tokens'] for r in rows)} for leg,rows in [('cuda',cuda),('cpu',cpu)]}


def extract(evaluation_dir,parity_dir,binding,instance,heldout_state,vocab_path):
    R,P=Path(evaluation_dir),Path(parity_dir);root=Path(__file__).resolve().parents[2]
    import bpe
    vocab=bpe.load_vocab(str(vocab_path))
    if sha(vocab_path)!=binding['upload_sha256']['vocab.gguf']:raise ValueError('evaluation vocabulary differs')
    manifest=json.loads((R/'manifest.json').read_text());parity_summary=json.loads((P/'parity_summary.json').read_text())
    if manifest['ember']['commit']!=binding['ember_commit'] or parity_summary['ember']['commit']!=binding['ember_commit']:raise ValueError('evaluation/parity common build differs')
    if manifest['models']['pksft']['gguf_sha256']!=binding['upload_sha256']['vocab.gguf'] or manifest['models']['pk_dpo_v1']['gguf_sha256']!=parity_summary['gguf_sha256']:raise ValueError('evaluated model identity differs')
    if type(parity_summary.get('n')) is not int or parity_summary['n']!=40 or parity_summary.get('max_new')!=60:raise ValueError('parity registered generation extent differs')
    frozen=heldout.api.rows(root/'corpus/eval/tinychat_hf_prompts_v1.jsonl')[:100]
    prompts=heldout.api.rows(root/'corpus/eval/chat_prompts_v1.jsonl')[:200]
    all_rows={}
    for name in ['pksft','pk_dpo_v1']:
        rows1=heldout.api.rows(R/'chat'/(name+'_samples.jsonl'));rows2=heldout.api.rows(R/'chat2'/(name+'_samples.jsonl'))
        coverage(rows1,frozen,vocab.encode);coverage(rows2,prompts,vocab.encode)
        all_rows[name]={'frozen':rows1,**{source:[r for r in rows2 if prompts[r['prompt_index']]['source']==source] for source in ['everyday','ultrachat']}}
    def bad(rows):
        if not rows:raise ValueError('empty protocol denominator')
        return sum(r['stop']=='turn' or any(m in r['completion'] for m in ['<|user|>','<|assistant|>']) for r in rows)/len(rows)
    rates={source:bad(rows) for source,rows in all_rows['pk_dpo_v1'].items()}
    repeats={source:{label:sum(chat_eval.repeats(r['completion']) for r in all_rows[name][source])/len(all_rows[name][source]) for label,name in [('reference','pksft'),('dpo','pk_dpo_v1')]} for source in ['frozen','ultrachat']}
    qs=heldout.api.rows(root/'corpus/eval/simple_qa_v1.jsonl')
    if len(qs)!=60:raise ValueError('registered simple QA set differs')
    correct={}
    for label,name in [('reference','pksft'),('dpo','pk_dpo_v1')]:
        rows=heldout.api.rows(R/'qa'/(name+'_qa.jsonl'));indices=[r['prompt_index'] for r in rows]
        if len(indices)!=60 or set(indices)!=set(range(60)):raise ValueError('simple QA coverage incomplete')
        correct[label]=sum(qa_eval.correct(r['completion'],qs[r['prompt_index']]['answers']) for r in rows)
    parity=parity_coverage(heldout.api.rows(P/'parity_cuda.jsonl'),heldout.api.rows(P/'parity_cpu.jsonl'),frozen[:40])
    _,_,status=heldout.inspect(instance)
    if status['missing_rankings']:raise ValueError('held-out eligible judging incomplete')
    stopped=json.loads((Path(heldout_state)/'STOPPED').read_text())
    if stopped.get('kind')!='COMPLETE_HELDOUT_JUDGING_NOT_MODEL_ADOPTION' or stopped.get('instance_sha256')!=heldout.api.digest(instance):raise ValueError('held-out completion identity differs')
    receipt=json.loads(Path(instance['sampling_receipt']['path']).read_text())
    if receipt['ember_commit']!=binding['ember_commit']:raise ValueError('held-out common build differs')
    sample_summary=json.loads(Path(instance['sampling_summary']['path']).read_text())
    if sample_summary['model_sha256']!=[manifest['models']['pksft']['gguf_sha256'],manifest['models']['pk_dpo_v1']['gguf_sha256']]:raise ValueError('held-out models differ from evaluated final checkpoints')
    result=heldout.sampling.sbs.count(heldout.api.rows(instance['requests']['path']),instance['cache'],heldout.api.profile_id(),json.loads(Path(instance['policy']['path']).read_text())['rubric_path'])
    return dict(protocol_failure_rates=rates,repeat_rates=repeats,qa_correct=correct,parity=parity,judged_gate=result)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['evaluation-dir','parity-dir','training-binding','heldout-instance','heldout-state','vocab','out']:ap.add_argument('--'+name,required=True)
    a=ap.parse_args();binding=json.loads(Path(a.training_binding).read_text());instance=json.loads(Path(a.heldout_instance).read_text())
    metrics=extract(a.evaluation_dir,a.parity_dir,binding,instance,a.heldout_state,a.vocab);verdict=assess(metrics)
    verdict['metrics']=metrics;verdict['assessor_sha256']=sha(__file__)
    with Path(a.out).open('xb') as f:f.write(heldout.api.encoded(verdict)+b'\n')
    print(json.dumps({'adoption_eligible':verdict['adoption_eligible'],'automatic_model_replacement':False}))
if __name__=='__main__':main()
