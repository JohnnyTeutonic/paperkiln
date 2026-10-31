"""One standard jobs_vm data job captures all 300 registered item dispositions; root owns launch."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
sys.path.insert(0,str(Path(__file__).resolve().parent))
from pref_artifact_io_v1 import sha
import heldout_sampling_receipt_v1 as sampling
import heldout_sampling_journal_v1 as durable


def health(port):
    try:
        with urllib.request.urlopen('http://127.0.0.1:'+str(port)+'/health',timeout=2) as response:return json.load(response)
    except Exception:return None


def main():
    out,binding_path=map(Path,sys.argv[1:]);out.mkdir(parents=True,exist_ok=True)
    binding=json.loads(binding_path.read_text());upload=binding_path.parent;repo=Path.cwd()
    for name,value in binding['upload_sha256'].items():
        if sha(upload/name)!=value:raise ValueError('held-out uploaded input differs')
    sys.path.insert(0,str(repo/'tools'));import bpe
    vocab=bpe.load_vocab(str(upload/'vocab.gguf'))
    items,dropped=sampling.sbs.S.prompts(str(upload/'dialogues_val.jsonl'),train_turns=sampling.sbs.S.first_turns(str(upload/'dialogues_train.jsonl')))
    items=items[:300]
    if len(items)!=300:raise ValueError('registered filtered held-out universe has fewer than 300 prompts')
    build=out/'ember_build.json'
    subprocess.run([sys.executable,'tools/colab_chat/ensure_ember_vm.py',str(build),str(out/'ember_build.log')],check=True)
    eb=json.loads(build.read_text())
    if eb['commit']!=binding['ember_commit']:raise ValueError('held-out Ember build differs from frozen common build')
    ggufs=[]
    for tag in ['pksft','pk_dpo_v1@pk_dpo_v1']:
        result=subprocess.check_output([sys.executable,'tools/colab_chat/chat_eval_stage.py',tag],text=True).strip().splitlines()[-1]
        spec=json.loads(Path(result).read_text());ggufs.append(Path(spec['out_dir'])/(spec['name']+'.gguf'))
    if sha(ggufs[0])!=binding['upload_sha256']['vocab.gguf']:raise ValueError('held-out reference differs from original sampled winner')
    model_hashes=[sha(g) for g in ggufs]
    journal_binding=durable.binding_for(items,vocab.encode,model_hashes,eb['commit'],sha(binding_path),
        sha(__file__),sha(sampling.__file__),sha(durable.__file__))
    journal_path=out/'sampling.journal.jsonl'
    home=upload/'heldout_sampling.journal.jsonl'
    with durable.restore(journal_path,home,journal_binding) as journal:
        if journal.state['pending'] is not None:raise ValueError('unknown prior sampling request; explicit review required')
        def finish(send_a,send_b):
            receipt=sampling.capture(items,
              lambda body:journal.call('pk_sft_v1',body,send_a),
              lambda body:journal.call('pk_dpo_v1',body,send_b),vocab.encode,eb['commit'],journal.item)
            journal.require_complete();sampling.replay(receipt,vocab.encode,items)
            durable.publish(out/'sampling_receipt.json',sampling.api.encoded(receipt)+b'\n')
            durable.publish(out/'requests.jsonl',b''.join(sampling.api.encoded(r)+b'\n' for r in receipt['requests']))
            summary={'kind':'COMPLETE_REGISTERED_HELDOUT_CAPTURE_V1','original_items':300,'eligible_requests':len(receipt['requests']),
              'dispositions':receipt['dispositions'],'source_filter_dropped':dropped,'model_sha256':model_hashes,
              'sampling_receipt_sha256':sha(out/'sampling_receipt.json'),'requests_sha256':sha(out/'requests.jsonl'),
              'ember_commit':eb['commit'],'api_judging_calls':0,'journal_complete':True,
              'journal_sha256':sha(journal_path),'journal_binding':journal_binding}
            durable.publish(out/'sampling_summary.json',sampling.api.encoded(summary)+b'\n')
        if len(journal.state['items'])==300:
            def forbidden(body):raise ValueError('complete journal must not resample')
            finish(forbidden,forbidden)
            return
        ports=[8701,8702];processes=[];logs=[]
        env={k:v for k,v in os.environ.items() if not k.startswith('MICROTORCH_') and k not in ['EMBER_SPEC','EMBER_DRAFT']}
        env.update(EMBER_GEMM_TF32='0',EMBER_DETERMINISTIC='1')
        try:
            for index,(port,gguf) in enumerate(zip(ports,ggufs)):
                if health(port) is not None:raise ValueError('held-out port already owned; leave existing server alone')
                handle=(out/('model_'+str(index)+'.log')).open('w');logs.append(handle)
                process=subprocess.Popen([eb['bin'],str(gguf),str(port),'127.0.0.1','-','-1'],stdout=handle,stderr=subprocess.STDOUT,env=env)
                processes.append(process)
                status=None
                for _ in range(300):
                    status=health(port)
                    if status is not None or process.poll() is not None:break
                    time.sleep(1)
                if not status or Path(status.get('model','')).resolve()!=gguf.resolve() or status.get('deterministic') is not True:
                    raise ValueError('held-out server model/deterministic health differs')
            def send(port):
                def call(body):
                    request=urllib.request.Request('http://127.0.0.1:'+str(port)+'/chat',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
                    with urllib.request.urlopen(request,timeout=170) as response:return json.load(response)
                return call
            finish(send(ports[0]),send(ports[1]))
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
                    try:process.wait(timeout=30)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
            for handle in logs:handle.close()
if __name__=='__main__':main()
