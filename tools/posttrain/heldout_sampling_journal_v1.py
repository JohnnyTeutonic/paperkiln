"""Fsynced request/item journal. Unknown calls, torn lines and divergent restores fail closed."""
import copy
import json
import os
from pathlib import Path
from chat_judge_api_heldout_v1 import encoded,digest


def sync_parent(path):
    fd=os.open(str(Path(path).parent),os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def atomic_replace(path,raw):
    path=Path(path);temp=path.with_name(path.name+'.restore.tmp')
    with temp.open('xb') as handle:handle.write(raw);handle.flush();os.fsync(handle.fileno())
    os.replace(temp,path);sync_parent(path)


def advance(state,record,binding):
    records,intents,results,items=state['records'],state['intents'],state['results'],state['items']
    pending=state['pending'];payload={k:v for k,v in record.items() if k!='record_sha256'}
    if record.get('previous_sha256')!=state['last_sha256'] or record.get('record_sha256')!=digest(payload):raise ValueError('sampling journal hash chain differs')
    kind=record.get('kind')
    if not records:
        if kind!='header' or record.get('binding')!=binding:raise ValueError('sampling journal request/source binding differs')
    elif kind=='intent':
        index=record['prompt_index'];model=record['model'];body=record['body'];key=record['request_key']
        expected=digest({'binding_sha256':digest(binding),'prompt_index':index,'model':model,'body':body})
        if (pending is not None or index!=len(items) or index>=binding['original_items'] or not binding['context_eligible'][index]
            or model not in ['pk_sft_v1','pk_dpo_v1'] or body['seed']!=71+index or key!=expected or key in intents):raise ValueError('sampling intent order, global index or request identity differs')
        completed=[r for r in results.values() if r['prompt_index']==index]
        if len(completed)>=2 or model!=['pk_sft_v1','pk_dpo_v1'][len(completed)]:raise ValueError('sampling model order differs')
        intents[key]=record;pending=key
    elif kind=='result':
        key=record['request_key']
        if key!=pending or key in results:raise ValueError('sampling result lacks unique pending intent')
        before=intents[key]
        if record['prompt_index']!=before['prompt_index'] or record['model']!=before['model'] or record['body']!=before['body']:raise ValueError('sampling result request differs')
        results[key]=record;pending=None
    elif kind=='item':
        index=record['prompt_index'];calls=[r for r in results.values() if r['prompt_index']==index]
        if pending is not None or index!=len(items) or index>=binding['original_items']:raise ValueError('sampling item order differs')
        eligible=binding['context_eligible'][index]
        if (len(calls)!=(2 if eligible else 0) or (record['status']=='over_context')==eligible
            or record['status'] not in ['ok','over_context','identical','invalid_utf8']):raise ValueError('sampling item call/disposition coverage differs')
        items[index]=record
    else:raise ValueError('unknown sampling journal record')
    records.append(record);state['pending']=pending;state['last_sha256']=record['record_sha256']
    return state


def parse(raw,binding):
    if not raw or not raw.endswith(b'\n'):raise ValueError('empty or torn sampling journal')
    state={'records':[],'intents':{},'results':{},'items':{},'pending':None,'last_sha256':None}
    for line in raw.splitlines():
        try:record=json.loads(line)
        except Exception:raise ValueError('invalid sampling journal JSON') from None
        if encoded(record)!=line:raise ValueError('noncanonical or corrupted sampling journal record')
        advance(state,record,binding)
    return state


def restore(remote,home,binding):
    import fcntl
    remote=Path(remote);home=Path(home) if home else None
    remote.parent.mkdir(parents=True,exist_ok=True)
    lock=open(str(remote)+'.lock','a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        r=remote.read_bytes() if remote.exists() else None
        h=home.read_bytes() if home and home.exists() else None
        if r is not None:parse(r,binding)
        if h is not None:parse(h,binding)
        if r is not None and h is not None and not (r.startswith(h) or h.startswith(r)):raise ValueError('remote and home sampling journals diverge')
        if h is not None and (r is None or len(h)>len(r)):
            atomic_replace(remote,h);r=h
        if r is None:
            payload={'kind':'header','binding':binding,'previous_sha256':None}
            payload['record_sha256']=digest(payload)
            with remote.open('xb') as handle:handle.write(encoded(payload)+b'\n');handle.flush();os.fsync(handle.fileno())
            sync_parent(remote)
        return Journal(remote,binding,lock)
    except BaseException:
        lock.close();raise


class Journal:
    def __init__(self,path,binding,lock=None):
        self.path=Path(path);self.binding=binding;self.lock=lock;self.state=parse(self.path.read_bytes(),binding);self.size=self.path.stat().st_size
    def close(self):
        if self.lock is not None:self.lock.close();self.lock=None
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def append(self,payload):
        payload=dict(payload,previous_sha256=self.state['last_sha256']);payload['record_sha256']=digest(payload)
        # Validate one transition in memory; avoid re-reading/parsing every prior response per token request.
        next_state={**self.state,**{key:self.state[key].copy() for key in ['records','intents','results','items']}}
        advance(next_state,payload,self.binding);raw=encoded(payload)+b'\n'
        if self.path.stat().st_size!=self.size:raise ValueError('sampling journal changed outside its owner')
        with self.path.open('ab') as handle:handle.write(raw);handle.flush();os.fsync(handle.fileno())
        self.state=next_state;self.size+=len(raw)
    def call(self,model,body,send):
        index=body['seed']-71
        key=digest({'binding_sha256':digest(self.binding),'prompt_index':index,'model':model,'body':body})
        if key in self.state['results']:return copy.deepcopy(self.state['results'][key]['response'])
        if self.state['pending'] is not None:raise ValueError('unknown sampling request outcome; explicit review required before dispatch')
        if key in self.state['intents']:raise ValueError('unfinished sampling request cannot be repeated')
        self.append({'kind':'intent','prompt_index':index,'model':model,'body':copy.deepcopy(body),'request_key':key})
        response=send(copy.deepcopy(body))
        self.append({'kind':'result','prompt_index':index,'model':model,'body':copy.deepcopy(body),'request_key':key,'response':copy.deepcopy(response)})
        return response
    def item(self,index,record,status):
        payload={'kind':'item','prompt_index':index,'status':status,'record':copy.deepcopy(record)}
        if index in self.state['items']:
            before={k:v for k,v in self.state['items'][index].items() if k not in ['previous_sha256','record_sha256']}
            if before!=payload:raise ValueError('completed sampling item changed')
            return
        self.append(payload)
    def require_complete(self):
        replay=parse(self.path.read_bytes(),self.binding)
        if replay['last_sha256']!=self.state['last_sha256']:raise ValueError('sampling journal changed before publication')
        if self.state['pending'] is not None or len(self.state['items'])!=self.binding['original_items']:raise ValueError('sampling journal is incomplete or unknown')


def binding_for(items,encode,model_sha256,ember_commit,training_binding_sha256,sampler_sha256,capture_sha256,journal_sha256):
    items=list(items)
    return {'kind':'BOUND_REGISTERED_HELDOUT_SAMPLING_JOURNAL_V1','original_items':len(items),
      'items_sha256':digest(items),'context_eligible':[len(encode('<|user|>'+user+'<|assistant|>'))+256<=512 for _,user in items],
      'model_sha256':model_sha256,'ember_commit':ember_commit,'training_binding_sha256':training_binding_sha256,
      'sampler_source_sha256':sampler_sha256,'capture_source_sha256':capture_sha256,'journal_source_sha256':journal_sha256,
      'generation':{'temperature':.7,'seed':71,'max_new_tokens':256,'ctx':512,'top_k':0,'top_p':1.0,'no_refill':True}}


def publish(path,raw):
    path=Path(path)
    if path.exists():
        if path.read_bytes()!=raw:raise ValueError('completed sampling output differs from journal replay')
        return
    temp=path.with_name(path.name+'.publish.tmp')
    if temp.exists():
        if temp.read_bytes()!=raw:raise ValueError('partial/corrupt sampling publication requires review')
    else:
        with temp.open('xb') as handle:handle.write(raw);handle.flush();os.fsync(handle.fileno())
    os.replace(temp,path);sync_parent(path)
