import importlib.util,json,tempfile,unittest
from pathlib import Path
P=Path(__file__).with_name('add_bias_gate_vm.py');s=importlib.util.spec_from_file_location('gate',P);g=importlib.util.module_from_spec(s);s.loader.exec_module(g)
class Gate(unittest.TestCase):
 def test_baseline_semantic_contract(self):
  text='{"actual_cuda":true,"stale_computed_bias":true,"host_poison":true}\nCHECK failed std::fabs(y->data(i, j) - expected) / x < tol\n'
  g.check_baseline(1,text)
  for rc,t in [(0,text),(2,text),(1,'CUDA missing'),(1,text.replace('std::fabs(y->data(i, j) - expected)','other assertion'))]:
   with self.assertRaises(ValueError):g.check_baseline(rc,t)
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
