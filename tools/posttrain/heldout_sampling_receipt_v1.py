"""Capture/replay all original held-out item dispositions without replacing skipped prompts."""
import copy
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import side_by_side as sbs
import chat_judge_api_heldout_v1 as api


def capture(items,send_a,send_b,encode,ember_commit):
    items=list(items);raw=[];records=[];dispositions=[]
    def tracked(which,send):
        def call(body):
            response=send(copy.deepcopy(body))
            raw.append({'model':which,'body':copy.deepcopy(body),'response':copy.deepcopy(response)})
            return response
        return call
    a,b=tracked('pk_sft_v1',send_a),tracked('pk_dpo_v1',send_b)
    config={'models':['pk_sft_v1','pk_dpo_v1'],'ember_commit':ember_commit,'temp':.7,'seed':71,
      'max_new_tokens':256,'ctx':512,'utf8_rule':sbs.S.UTF8_RULE}
    for index,(record,why) in enumerate(sbs.sample(items,a,b,.7,71,256,encode,512)):
        dispositions.append({'prompt_index':index,'uri':items[index][0],'status':why})
        if why=='ok':
            record['sampler'].update(models=config['models'],config=config)
            records.append(record)
    if len(dispositions)!=len(items):raise ValueError('held-out disposition coverage differs')
    return {'kind':'COMPLETE_HELDOUT_SAMPLING_RECEIPT_V1','items':items,'ember_commit':ember_commit,
      'raw_calls':raw,'requests':records,'dispositions':dispositions,'no_refill':True}


def replay(receipt,encode,expected_items=None,require_300=True):
    items=receipt['items']
    if require_300 and len(items)!=300:raise ValueError('original held-out universe must contain 300 items')
    if expected_items is not None and api.encoded(items)!=api.encoded(list(expected_items)):
        raise ValueError('original first 300 source-filtered prompts differ')
    calls=iter(receipt['raw_calls'])
    def send(which):
        def read(body):
            try:record=next(calls)
            except StopIteration:raise ValueError('raw held-out sampling call missing') from None
            if record['model']!=which or record['body']!=body:raise ValueError('held-out sampling request differs')
            return record['response']
        return read
    rebuilt=capture(items,send('pk_sft_v1'),send('pk_dpo_v1'),encode,receipt['ember_commit'])
    if next(calls,None) is not None or api.encoded(rebuilt)!=api.encoded(receipt):
        raise ValueError('sampling replay differs or extra calls/refill present')
    return rebuilt['requests']
