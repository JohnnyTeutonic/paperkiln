"""No real server/process/network: prove ownership cleanup before any sample on bad health."""
import json
from pathlib import Path
import sys
import types
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import heldout_sample_vm_v1 as vm

def test_bad_owned_server_health_stops_only_owned_process(monkeypatch,tmp_path):
    upload=tmp_path/'upload';upload.mkdir();out=tmp_path/'out';ref=tmp_path/'ref.gguf';ref.write_text('reference');dpo=tmp_path/'dpo.gguf';dpo.write_text('dpo')
    for name in ['vocab.gguf','dialogues_val.jsonl','dialogues_train.jsonl']:(upload/name).write_text('reference' if name=='vocab.gguf' else 'fixture')
    binding=dict(ember_commit='frozen',upload_sha256={name:vm.sha(upload/name) for name in ['vocab.gguf','dialogues_val.jsonl','dialogues_train.jsonl']})
    bp=upload/'binding';bp.write_text(json.dumps(binding));specs=[]
    for name,gguf in [('reference',ref),('dpo',dpo)]:
        sp=tmp_path/(name+'.json');sp.write_text(json.dumps({'out_dir':str(tmp_path),'name':gguf.stem}));specs.append(sp)
    monkeypatch.chdir(tmp_path);monkeypatch.setattr(sys,'argv',['sample',str(out),str(bp)])
    monkeypatch.setitem(sys.modules,'bpe',types.SimpleNamespace(load_vocab=lambda path:types.SimpleNamespace(encode=lambda text:[1])))
    monkeypatch.setattr(vm.sampling.sbs.S,'first_turns',lambda path:set());monkeypatch.setattr(vm.sampling.sbs.S,'prompts',lambda *a,**k:([(str(i),'user') for i in range(300)],{}))
    def run(*a,**k):(out/'ember_build.json').write_text(json.dumps({'commit':'frozen','bin':'fake-server'}))
    monkeypatch.setattr(vm.subprocess,'run',run);paths=iter(specs);monkeypatch.setattr(vm.subprocess,'check_output',lambda *a,**k:str(next(paths)))
    process=types.SimpleNamespace(alive=True,terminated=False)
    process.poll=lambda:None if process.alive else 0
    def terminate():process.terminated=True;process.alive=False
    process.terminate=terminate;process.wait=lambda **k:0
    created=[]
    def popen(*a,**k):created.append(process);return process
    monkeypatch.setattr(vm.subprocess,'Popen',popen)
    readings=iter([None,{'model':str(tmp_path/'wrong'),'deterministic':True}]);monkeypatch.setattr(vm,'health',lambda port:next(readings))
    monkeypatch.setattr(vm.urllib.request,'urlopen',lambda *a,**k:pytest.fail('network sampling reached'))
    with pytest.raises(ValueError,match='health'):vm.main()
    assert len(created)==1 and process.terminated
