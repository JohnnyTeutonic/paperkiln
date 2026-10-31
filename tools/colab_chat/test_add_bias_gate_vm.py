import importlib.util,json,tempfile,unittest,subprocess
from pathlib import Path
P=Path(__file__).with_name('add_bias_gate_vm.py');s=importlib.util.spec_from_file_location('gate',P);g=importlib.util.module_from_spec(s);s.loader.exec_module(g)
class Gate(unittest.TestCase):
 def test_baseline_semantic_contract(self):
  text='{"actual_cuda":true,"stale_computed_bias":true,"host_poison":true}\nCHECK failed /source/tests/test_add_bias_materialization.cpp:72: std::fabs(y->data(i, j) - expected) / (1 + std::fabs(expected)) < 2e-5\n'
  g.check_baseline(1,text)
  for rc,t in [(0,text),(2,text),(1,'CUDA missing'),(1,text.replace('std::fabs(y->data(i, j) - expected)','other assertion'))]:
   with self.assertRaises(ValueError):g.check_baseline(rc,t)
 def test_real_check_macro_line(self):
  with tempfile.TemporaryDirectory() as d:
   root=P.parents[2];cpp=Path(d)/'fail.cpp';binary=Path(d)/'fail'
   cpp.write_text('#include <cmath>\n#include "tests/check.hpp"\nstruct Y {double data(int,int){return 0;}};\nint main(){Y obj;Y* y=&obj;int i=0,j=0;double expected=1;\n#line 72 "test_add_bias_materialization.cpp"\nCHECK(std::fabs(y->data(i, j) - expected) / (1 + std::fabs(expected)) < 2e-5); }\n')
   subprocess.run(['g++','-std=c++17','-I',str(root),str(cpp),'-o',str(binary)],check=True)
   r=subprocess.run([str(binary)],capture_output=True,text=True)
   g.check_baseline(r.returncode,'{"actual_cuda":true,"stale_computed_bias":true,"host_poison":true}\n'+r.stderr)
 def test_fixed_proof_refuses_cpu_missing_false(self):
  for v in ({},{'pass':True},{'pass':True,'cuda_integration':False}):
   with self.assertRaises(ValueError):g.parse_proof(json.dumps(v),('pass','cuda_integration'))
  self.assertTrue(g.parse_proof('{"pass":true,"cuda_integration":true}',('pass','cuda_integration'))['pass'])
 def test_setup_failure_durably_negative(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'uploads').mkdir();(p/'uploads/bias_inputs.json').write_text('{"files":{"missing":"x"}}')
   self.assertEqual(g.run(p/'out',p/'uploads',p/'source',p/'build'),0)
   r=json.loads((p/'out/gate_summary.json').read_text());self.assertFalse(r['pass']);self.assertTrue(r['errors']);self.assertFalse(r['source_restored']);self.assertEqual(len(r['logs']),8)
if __name__=='__main__':unittest.main()
