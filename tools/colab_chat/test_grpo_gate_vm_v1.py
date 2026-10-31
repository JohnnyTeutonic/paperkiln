import importlib.util,json,tempfile,unittest,subprocess
from pathlib import Path
P=Path(__file__).with_name('grpo_gate_vm_v1.py');s=importlib.util.spec_from_file_location('gate',P);g=importlib.util.module_from_spec(s);s.loader.exec_module(g)
class Gate(unittest.TestCase):
 def proof(self):return {'actual_cuda':True,'stale_producer':True,'host_poison':True,'no_grad_context':True,'deferred_window':True}
 def test_baseline_exact_numeric_failure(self):
  for case,expression in g.EXPRESSIONS.items():
   text=json.dumps(self.proof())+'\nCHECK failed /source/tests/test_studio_grpo_materialization.cpp:78: '+expression+'\n'
   g.check_baseline(1,text,case)
   for rc,bad in [(0,text),(2,text),(1,text.replace(expression,'wrong')), (1,'CUDA missing')]:
    with self.assertRaises(ValueError):g.check_baseline(rc,bad,case)
 def test_real_check_macro_format(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);cpp=p/'probe.cpp';binary=p/'probe';root=P.parents[2]
   cpp.write_text('#include <cmath>\n#include "tests/check.hpp"\nstruct S {double logp;};\nint main(){S actual{0},expected{1};\n#line 78 "test_studio_grpo_materialization.cpp"\nCHECK(std::fabs(actual.logp - expected.logp) < 1e-12);}\n')
   subprocess.run(['g++','-std=c++17','-I',str(root),str(cpp),'-o',str(binary)],check=True)
   q=subprocess.run([str(binary)],capture_output=True,text=True);g.check_baseline(q.returncode,json.dumps(self.proof())+'\n'+q.stderr,'sample')
 def test_fixed_requires_actual_cuda(self):
  keys=('pass','actual_cuda','devcheck_enabled','no_grad_context','cpu_reference','stale_producer','host_poison')
  valid={k:True for k in keys};g.parse_proof(json.dumps(valid),keys)
  for key in keys:
   for value in (False,1,'true',None):
    with self.assertRaises(ValueError):g.parse_proof(json.dumps({**valid,key:value}),keys)
 def test_setup_failure_is_negative_and_immutable(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);u=p/'uploads';u.mkdir();(u/'grpo_inputs.json').write_text('{"files":{"missing":"x"}}')
   self.assertEqual(g.run(p/'out',u,p/'source',p/'build'),0)
   report=json.loads((p/'out/gate_summary.json').read_text());self.assertFalse(report['pass']);self.assertTrue(report['errors'])
   before=(p/'out/gate_summary.json').read_bytes()
   with self.assertRaises(FileExistsError):g.run(p/'out',u,p/'source',p/'build')
   self.assertEqual(before,(p/'out/gate_summary.json').read_bytes())
 def test_real_slice_and_nograd_producer(self):
  root=P.parents[2];test=(root/'tests/test_studio_grpo_materialization.cpp').read_text()
  self.assertIn('ops::slice_cols(value, 0, 4)',test);self.assertIn('CHECK(!grad_enabled())',test);self.assertIn('CHECK(device::deferring())',test)
  self.assertNotIn('ops::scale(',test)
 def test_job_bounds(self):
  job=json.loads(P.with_name('grpo_gate_20261031.json').read_text());self.assertEqual(job['max_parallel'],1)
  self.assertEqual(job['jobs'][0]['cmd'][:4],['timeout','--signal=TERM','--kill-after=30','1500'])
  self.assertEqual(g.DEADLINE.isoformat(),'2026-10-31T03:05:00+00:00')
if __name__=='__main__':unittest.main()
