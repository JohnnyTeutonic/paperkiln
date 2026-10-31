import importlib.util,json,tempfile,unittest,subprocess
from pathlib import Path
p=Path(__file__).with_name('layernorm_gate_vm.py');s=importlib.util.spec_from_file_location('gate',p);g=importlib.util.module_from_spec(s);s.loader.exec_module(g)
def proof(case):return json.dumps({'case':case,'actual_cuda':True,'stale_operand_proved':True,'host_poison':True,'ops_disabled_before_fallback':True})+'\n'
class Gate(unittest.TestCase):
 def test_distinct_registered_predicates(self):
  for case in g.CASES:
   expr=g.BACKWARD_EXPRESSION if case=='backward_gamma' else g.FORWARD_EXPRESSION
   text=proof(case)+'CHECK failed /source/tests/test_layernorm_materialization.cpp:95: '+expr+'\n';g.check_baseline(1,text,case)
   for rc,t in [(0,text),(2,text),(1,text.replace(expr,'wrong_assertion')),(1,text.replace('"actual_cuda": true','"actual_cuda": false')),(1,text.replace(case,'other_case'))]:
    with self.assertRaises(ValueError):g.check_baseline(rc,t,case)
   other=g.FORWARD_EXPRESSION if case=='backward_gamma' else g.BACKWARD_EXPRESSION
   with self.assertRaises(ValueError):g.check_baseline(1,proof(case)+'CHECK failed test_layernorm_materialization.cpp:95: '+other,case)
 def test_actual_check_macro_producer(self):
  with tempfile.TemporaryDirectory() as d:
   root=p.parents[2];cpp=Path(d)/'fail.cpp';binary=Path(d)/'fail'
   cpp.write_text('#include <cmath>\n#include "tests/check.hpp"\nstruct M { double data(int,int){return 0;} double operator()(int,int){return 0;}};struct V {M grad;double data(int,int){return 0;}};int main(int argc,char**){V obj;V*y=&obj;V*x0=&obj;int i=0,j=0,C=4;double expected[8]={1},dx[8]={1};\n#line 95 "test_layernorm_materialization.cpp"\nif(argc==1){CHECK('+g.FORWARD_EXPRESSION+');}else{CHECK('+g.BACKWARD_EXPRESSION+');}}\n')
   subprocess.run(['g++','-std=c++17','-I',str(root),str(cpp),'-o',str(binary)],check=True)
   for case,args in [('forward_x',[]),('backward_gamma',['backward'])]:
    r=subprocess.run([str(binary),*args],capture_output=True,text=True);g.check_baseline(r.returncode,proof(case)+r.stderr,case)
 def test_fixed_requires_actual_cuda(self):
  for value in ({},{'pass':True,'cuda_integration':False,'devcheck_enabled':True},{'pass':True,'cuda_integration':True,'devcheck_enabled':False}):
   with self.assertRaises(ValueError):g.parse_proof(json.dumps(value),('pass','cuda_integration','devcheck_enabled'))
 def test_missing_inputs_durable_negative(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'uploads').mkdir();(root/'uploads/layernorm_inputs.json').write_text('{"files":{"missing":"x"}}');self.assertEqual(g.run(root/'out',root/'uploads',root/'source',root/'build'),0);r=json.loads((root/'out/gate_summary.json').read_text());self.assertFalse(r['pass']);self.assertTrue(r['errors']);self.assertEqual(len(r['logs']),13)
if __name__=='__main__':unittest.main()
