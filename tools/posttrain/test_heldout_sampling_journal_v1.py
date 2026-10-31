"""Offline request-level crash, durable resume and fail-closed restore proofs."""
import copy
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import heldout_sampling_journal_v1 as j
import heldout_sampling_receipt_v1 as sampling

def encode(text):return list(range(300 if 'long' in text else 4))
ITEMS=[('long','long'),('same','same'),('different','different')]
BIND=j.binding_for(ITEMS,encode,['ref','dpo'],'ember','train','sampler','capture','journal')

def response(model,body):
    user=body['user_input'];text='same' if user=='same' else model
    return dict(prompt_ids=encode('<|user|>'+user+'<|assistant|>'),reply=text,stop_reason='eos',tokens=1)

def capture(journal,calls,interrupt=False):
    def send(model):
        def call(body):
            if interrupt and model=='pk_dpo_v1':raise KeyboardInterrupt
            def actual(b):
                calls.append((model,b['seed']));return response(model,b)
            return journal.call(model,body,actual)
        return call
    return sampling.capture(ITEMS,send('pk_sft_v1'),send('pk_dpo_v1'),encode,'ember',journal.item)


def test_safe_interruption_after_result_resumes_without_any_completed_request_repeat(tmp_path):
    path=tmp_path/'remote';calls=[]
    with j.restore(path,None,BIND) as journal:
        with pytest.raises(KeyboardInterrupt):capture(journal,calls,interrupt=True)
    assert calls==[('pk_sft_v1',72)]
    with j.restore(path,None,BIND) as journal:
        receipt=capture(journal,calls);journal.require_complete()
    assert len(calls)==4 and len(set(calls))==4
    assert [d['status'] for d in receipt['dispositions']]==['over_context','identical','ok']
    assert [r['body']['seed'] for r in receipt['raw_calls']]==[72,72,73,73]
    with j.restore(path,None,BIND) as journal:
        def forbidden(body):pytest.fail('completed request repeated')
        rebuilt=sampling.capture(ITEMS,lambda b:journal.call('pk_sft_v1',b,forbidden),lambda b:journal.call('pk_dpo_v1',b,forbidden),encode,'ember',journal.item)
    assert rebuilt==receipt


def test_unknown_request_intent_blocks_resume_without_retry(tmp_path):
    path=tmp_path/'remote';body=dict(seed=71,user_input='user')
    bind=copy.deepcopy(BIND);bind['context_eligible'][0]=True
    with j.restore(path,None,bind) as journal:
        def lost(body):raise RuntimeError('lost process after dispatch')
        with pytest.raises(RuntimeError):journal.call('pk_sft_v1',body,lost)
    before=path.read_bytes()
    with j.restore(path,None,bind) as journal:
        with pytest.raises(ValueError,match='unknown'):journal.call('pk_sft_v1',body,lambda b:pytest.fail('unknown request retried'))
    assert path.read_bytes()==before


@pytest.mark.parametrize('damage',['torn','corrupt','binding'])
def test_partial_corrupt_or_mismatched_journal_refused_without_change(tmp_path,damage):
    path=tmp_path/'remote'
    with j.restore(path,None,BIND):pass
    raw=path.read_bytes();binding=copy.deepcopy(BIND)
    if damage=='torn':path.write_bytes(raw[:-1])
    if damage=='corrupt':path.write_bytes(raw.replace(b'ember',b'other'))
    if damage=='binding':binding['model_sha256']=['different','dpo']
    before=path.read_bytes()
    with pytest.raises(ValueError):j.restore(path,None,binding)
    assert path.read_bytes()==before


def test_remote_home_maximal_complete_prefix_precedence(tmp_path):
    path=tmp_path/'remote';home=tmp_path/'home'
    with j.restore(path,None,BIND):pass
    short=path.read_bytes();home.write_bytes(short)
    with j.restore(path,home,BIND) as journal:capture(journal,[])
    full=path.read_bytes()
    with j.restore(path,home,BIND) as journal:journal.require_complete()
    assert path.read_bytes()==full
    home.write_bytes(full);path.write_bytes(short)
    with j.restore(path,home,BIND) as journal:journal.require_complete()
    assert path.read_bytes()==full
    path.unlink()
    with j.restore(path,home,BIND) as journal:journal.require_complete()
    assert path.read_bytes()==full


def test_divergent_remote_and_home_preserved_for_review(tmp_path):
    remote=tmp_path/'remote';home=tmp_path/'home';bind=copy.deepcopy(BIND);bind['context_eligible'][0]=True;body={'seed':71}
    with j.restore(remote,None,bind):pass
    home.write_bytes(remote.read_bytes())
    with j.restore(remote,None,bind) as journal:journal.call('pk_sft_v1',body,lambda b:{'reply':'one'})
    with j.restore(home,None,bind) as journal:journal.call('pk_sft_v1',body,lambda b:{'reply':'two'})
    before=(remote.read_bytes(),home.read_bytes())
    with pytest.raises(ValueError,match='diverge'):j.restore(remote,home,bind)
    assert before==(remote.read_bytes(),home.read_bytes())


def test_intent_result_and_disposition_are_fsynced_each_time(monkeypatch,tmp_path):
    path=tmp_path/'remote';seen=[];original=j.os.fsync
    def fsync(fd):
        original(fd)
        if path.exists():seen.append(json.loads(path.read_bytes().splitlines()[-1])['kind'])
    monkeypatch.setattr(j.os,'fsync',fsync)
    with j.restore(path,None,BIND) as journal:
        capture(journal,[])
    assert seen.count('intent')==4 and seen.count('result')==4 and seen.count('item')==3


def test_second_owner_and_wrong_global_index_or_model_order_refused(tmp_path):
    path=tmp_path/'remote'
    with j.restore(path,None,BIND) as journal:
        with pytest.raises(BlockingIOError):j.restore(path,None,BIND)
        before=path.read_bytes()
        with pytest.raises(ValueError):journal.call('pk_sft_v1',{'seed':73},lambda b:pytest.fail('wrong index dispatched'))
        assert path.read_bytes()==before


def test_atomic_final_outputs_are_idempotent_and_never_overwrite_mismatch(tmp_path):
    p=tmp_path/'result';j.publish(p,b'complete');j.publish(p,b'complete')
    with pytest.raises(ValueError):j.publish(p,b'different')
    assert p.read_bytes()==b'complete'


def test_registered_300_global_indices_and_invalid_utf8_disposition(tmp_path):
    items=[('uri'+str(i),'user'+str(i)) for i in range(300)]
    binding=j.binding_for(items,encode,['ref','dpo'],'ember','train','sampler','capture','journal')
    calls=[]
    with j.restore(tmp_path/'remote',None,binding) as journal:
        def send(model):
            def call(body):
                def actual(b):
                    calls.append((model,b['seed']))
                    r=response(model,b)
                    if b['seed']==370:r['reply']='bad\ufffd'
                    return r
                return journal.call(model,body,actual)
            return call
        receipt=sampling.capture(items,send('pk_sft_v1'),send('pk_dpo_v1'),encode,'ember',journal.item)
        journal.require_complete()
        assert len(journal.state['items'])==300 and len(journal.state['results'])==600
    assert len(calls)==600 and len(set(calls))==600 and {seed for _,seed in calls}==set(range(71,371))
    assert receipt['dispositions'][-1]['status']=='invalid_utf8'
    assert len(receipt['requests'])==299
