import os,sys,json,subprocess,hashlib,time
from pathlib import Path
import numpy as np
ROOT=Path('/home/jonat/pk_studio_probes_20261031');OUT=Path('/home/jonat/author_flex_probe_20261031');OUT.mkdir(exist_ok=True)
BIN=Path('/home/jonat/pk_studio_probes_build_20261031/mtstudio');EMBER=Path('/home/jonat/ember_publication_build_20261030/ir_logits')
os.environ['MTSTUDIO_BIN']=str(BIN)
sys.path[:0]=[str(ROOT/'tests'),str(ROOT/'tools'),str(ROOT/'spec/tools')]
from test_mtstudio_chat import write_vocab_gguf
from test_recursion_ir import parity,weights,prompts
import ir_check,ir_numpy
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
write_vocab_gguf(OUT/'v.gguf',['<unk>','the','cat','sat','on','mat','.'])
(OUT/'corpus.txt').write_text('the cat sat on the mat . '*400)
spec={'name':'author_combined','arch':{'preset':'gpt2-nano','custom':{'d':16,'layers':2,'heads':2,'d_ff':24,'activation':'gelu_erf','position':'sinusoidal','embed':'expr:x*2','embed_before_positions':True,'tie_head':True}},'data':{'vocab':str(OUT/'v.gguf'),'corpus':str(OUT/'corpus.txt'),'T':16,'max_tokens':0},'train':{'steps':3,'batch':1,'seed':9,'lr':.001,'eval_every':3,'checkpoint_every':3,'gradmap_every':0},'export':{'formats':['safetensors','gguf']},'out_dir':str(OUT/'trained')}
p=OUT/'spec.json';p.write_text(json.dumps(spec))
before=sha(BIN)
source_before={f:sha(ROOT/f) for f in ['include/microtorch/ops.hpp','src/ops.cpp','tools/parity_model.hpp','tools/mtstudio.cpp']}
r=subprocess.run([str(BIN),'run',str(p)],capture_output=True,text=True,timeout=90,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'});(OUT/'train.log').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stderr[-1000:]
events=[json.loads(x) for x in (OUT/'trained/events.jsonl').read_text().splitlines()]
f32=Path(next(e['path'] for e in events if e.get('event')=='export' and e.get('format')=='gguf'))
meta,W=weights(f32);doc=json.loads(meta['paperkiln.ir'][1]);irpath=f32.with_suffix('.ir.json')
check=subprocess.run([sys.executable,str(ROOT/'spec/tools/ir_check.py'),str(f32)],capture_output=True,text=True);(OUT/'ir_check.log').write_text(check.stdout+check.stderr);assert check.returncode==0,check.stdout+check.stderr
assert 'output.weight' not in W
assert any(n.get('tied_to')=='token_embd.weight' for n in doc['head']['nodes'])
assert any(n.get('op')=='gelu' and n.get('approx')=='none' for n in doc['block']['nodes'])
seqs=[[1,2],[2,3,4],[1,2,3,4,5,6],[6,5,4,3,2,1]]
(OUT/'ids.json').write_text(json.dumps(seqs))
e=subprocess.run([str(EMBER),str(f32)],input=json.dumps(seqs),capture_output=True,text=True,timeout=90,env={**os.environ,'EMBER_IR':str(irpath),'OMP_NUM_THREADS':'1'});(OUT/'ember.stdout.json').write_text(e.stdout);(OUT/'ember.stderr.log').write_text(e.stderr);assert e.returncode==0,e.stderr[-1200:]
metrics={}
for mode in ['prefill','decode']:
 maxerr=0.;top=rows=0
 for ids,actual in zip(seqs,json.loads(e.stdout)):
  ref=np.asarray(ir_numpy.forward(doc,W,ids));got=np.asarray(actual[mode]);assert got.shape==ref.shape
  err=np.max(np.abs(got-ref))/max(1,float(np.max(np.abs(ref))));maxerr=max(maxerr,float(err));top+=int(np.count_nonzero(got.argmax(1)==ref.argmax(1)));rows+=len(ids)
 assert maxerr<=1e-5 and top==rows,(mode,maxerr,top,rows)
 metrics[mode]={'rounding_error':maxerr,'argmax_equal':top,'rows':rows,'bound':1e-5}
# Same NumPy semantics vs trained source model via score, 40 independent prompts.
# Helper prompt vocabulary is larger, so substitute an explicit valid vocabulary list.
import test_recursion_ir
orig=test_recursion_ir.prompts
test_recursion_ir.prompts=lambda:seqs
try: result=parity(p,OUT,n_new=2)
finally:test_recursion_ir.prompts=orig
assert result[0]<=1e-5 and result[1]==result[2],result[:3]
assert sha(BIN)==before
assert source_before=={f:sha(ROOT/f) for f in source_before}
receipt={'status':'PASS','scope':'Tiny trained CPU combined FLEX tiedhead/exactGELU/embed-before-sinusoidal; no HF/GPU claim','binary_sha256':before,'ember_binary_sha256':sha(EMBER),'spec_sha256':sha(p),'gguf_sha256':sha(f32),'ir_sha256':sha(irpath),'ir_validator':'PASS','missing_output_weight':True,'tied_to':'token_embd.weight','embed_node_ops':[n['op'] for n in doc['embed']['nodes']],'numpy_vs_ember':metrics,'numpy_vs_mtstudio_score':{'max_logprob_rounding':result[0],'greedy_sequences_equal':result[1],'sequences':result[2],'continuation_tokens_per_sequence':2},'source_sha256':{f:sha(ROOT/f) for f in ['include/microtorch/ops.hpp','src/ops.cpp','tools/parity_model.hpp','tools/mtstudio.cpp']},'provider_calls':0,'gpu_calls':0}
(OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
