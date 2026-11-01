"""Controlled old/new lock experiment; tiny fixtures, no throughput extrapolation."""
import contextlib,hashlib,json,multiprocessing as mp,subprocess,sys,types,time,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tools/colab_chat'))
import sfull_checkpoint_v2 as repaired
from test_sfull_checkpoint_v2 import flat,SOURCE
BASE='54cf0861da26'
raw=subprocess.check_output(['git','-C',str(ROOT),'show',BASE+':tools/colab_chat/sfull_checkpoint_v2.py'])
old=types.ModuleType('old_sfull_checkpoint');exec(compile(raw,'frozen_sfull_checkpoint_v2.py','exec'),old.__dict__)
ctx=mp.get_context('fork');results=[]
with tempfile.TemporaryDirectory(prefix='sfull_lock_evidence_') as td:
 for label,core in [('historical',old),('repaired',repaired)]:
  for operation in ['pin','release']:
   run=Path(td)/(label+'-'+operation);flat(run)
   core.publish(run,25,1,2,1,SOURCE);root=run/'.checkpoint_v2'
   generation,_=core.current_unlocked(root);digest=core.sha(generation/'manifest.json')
   if operation=='release':core.pin_inflight(root)
   entered,allow,attempt,acquired,done=[ctx.Event() for _ in range(5)]
   elapsed=ctx.Queue()
   def consumer():
    original=core.validate_generation
    def paused(folder):
     entered.set()
     if not allow.wait(20):raise RuntimeError('barrier timeout')
     return original(folder)
    core.validate_generation=paused
    if operation=='pin':core.pin_inflight(root)
    else:core.release_inflight(root,digest)
   def producer():
    original=core.locked
    @contextlib.contextmanager
    def observed(*args,**kwargs):
     attempt.set()
     with original(*args,**kwargs) as value:
      acquired.set();yield value
    core.locked=observed
    start=time.monotonic()
    for step in [50,75]:
     flat(run,step,step/100);core.publish(run,step,1,2,1,SOURCE)
    elapsed.put(time.monotonic()-start);done.set()
   reader=ctx.Process(target=consumer);writer=ctx.Process(target=producer)
   reader.start()
   try:
    assert entered.wait(5)
    writer.start();assert attempt.wait(5)
    before=done.wait(5 if label=='repaired' else .25)
    assert before==(label=='repaired')
    assert generation.is_dir()
   finally:
    allow.set();reader.join(10);writer.join(10)
    for child in [reader,writer]:
     if child.is_alive():child.terminate();child.join()
   assert reader.exitcode==0 and writer.exitcode==0
   seconds=elapsed.get(timeout=2)
   assert core.current_unlocked(root)[1]['step']==75
   results.append(dict(implementation=label,operation=operation,publisher_attempt_observed=True,publisher_completed_before_verifier_released=before,controlled_publisher_elapsed_seconds=seconds,older_generation_retained_during_verification=True))
report={'status':'PASS','kind':'CONTROLLED_SFULL_LOCK_EVIDENCE_V1','baseline_source_sha256':hashlib.sha256(raw).hexdigest(),'repaired_source_sha256':hashlib.sha256((ROOT/'tools/colab_chat/sfull_checkpoint_v2.py').read_bytes()).hexdigest(),'cells':results,'qualification':'Tiny CPU fixtures and deliberately paused verifier prove lock blocking/removal, not storage throughput or actual L4 cadence. Historical false cells release verifier after250ms observation; elapsed includes this artificial delay. No model, historical checkpoint, GPU, provider or driver state changed.','max_gap_seconds_unchanged':180,'actual_L4_cadence_measured':False}
output=Path(sys.argv[1]);assert not output.exists();output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':'PASS','cells':len(results)}))
