"""Offline protocol checks; these do not establish actual CUDA execution."""
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

P=Path(__file__).with_name('conv1d_gate_vm_v1.py')
s=importlib.util.spec_from_file_location('conv_gate',P);g=importlib.util.module_from_spec(s);s.loader.exec_module(g)
class Gate(unittest.TestCase):
    def test_explicit_proof_contract(self):
        valid={k:True for k in g.PROOF_KEYS};valid['cases']=120
        self.assertTrue(g.parse_proof(json.dumps(valid))['pass'])
        for key in g.PROOF_KEYS:
            for value in (False,1,'true',None):
                bad={**valid,key:value}
                with self.assertRaises(ValueError):g.parse_proof(json.dumps(bad))
            bad=dict(valid);del bad[key]
            with self.assertRaises(ValueError):g.parse_proof(json.dumps(bad))
        for value in ([],True,None):
            with self.assertRaises(ValueError):g.parse_proof(json.dumps(value))
        for count in (119,121,True,120.0,'120',None):
            with self.assertRaises(ValueError):g.parse_proof(json.dumps({**valid,'cases':count}))
    def test_bounded_job_contract(self):
        job=json.loads(P.with_name('conv1d_gate_20261031.json').read_text())
        self.assertEqual(job['max_parallel'],1);self.assertEqual(len(job['jobs']),1)
        command=job['jobs'][0]['cmd'];self.assertEqual(command[:4],['timeout','--signal=TERM','--kill-after=30','2700'])
        self.assertEqual(job['jobs'][0]['outputs'],['gate_summary.json']+[n+'.log' for n in g.LOGS])
        self.assertEqual(g.DEADLINE.isoformat(),'2026-10-31T03:00:00+00:00')
        self.assertEqual(len(g.SOURCE_COMMIT),40)
    def test_thirdparty_actual_cuda_banner(self):
        text='third-party ops (CUDA device ops)\n1. conv1d_causal (K 4, 2 sequences)\n4. mamba1 (d 16, Di 32, N 4, K 4)\nall third-party op checks passed'
        self.assertFalse(g.check_thirdparty(text)['selective_scan_device_claim'])
        for bad in (text.replace('CUDA device ops','host'),text.replace('all third-party op checks passed',''), ''):
            with self.assertRaises(ValueError):g.check_thirdparty(bad)
    def test_setup_failure_is_durably_negative(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);u=p/'upload';u.mkdir()
            (u/'conv1d_inputs.json').write_text(json.dumps({'source_commit':g.SOURCE_COMMIT,'files':{'missing':'x'}}))
            self.assertEqual(g.run(p/'out',u,p/'source',p/'build'),0)
            report=json.loads((p/'out/gate_summary.json').read_text())
            self.assertFalse(report['pass']);self.assertTrue(report['errors']);self.assertEqual(len(report['logs']),6)
            self.assertFalse((p/'source').exists())
    def test_unsafe_archive_refused(self):
        for name,kind in (('../bad',tarfile.REGTYPE),('/bad',tarfile.REGTYPE),('link',tarfile.SYMTYPE),('hard',tarfile.LNKTYPE)):
            with tempfile.TemporaryDirectory() as d:
                p=Path(d);archive=p/'source.tar'
                with tarfile.open(archive,'w') as f:
                    entry=tarfile.TarInfo(name);entry.type=kind;entry.linkname='bad';f.addfile(entry,io.BytesIO())
                with self.assertRaises(ValueError):g.safe_extract(archive,p/'source')
    def test_producer_has_real_device_copy_seam(self):
        root=P.parents[2]
        test=(root/'tests/test_conv1d_cuda.cpp').read_text()
        producer=test[test.index('auto computed='):test.index('Var cx=computed')]
        self.assertIn('ops::slice_cols(v,0,v->data.cols())',producer)
        self.assertNotIn('ops::scale',producer)
        self.assertIn('CHECK(device::host_stale(q->data))',producer)
        ops=(root/'src/ops.cpp').read_text();ops=ops[ops.index('Var slice_cols('):ops.index('Var place_cols(')] if 'Var place_cols(' in ops else ops[ops.index('Var slice_cols('):ops.index('Var concat_cols(')]
        self.assertIn('device::devops::slice_cols',ops)
        cuda=(root/'src/cuda_ops.cu').read_text();cuda=cuda[cuda.index('bool slice_cols('):cuda.index('bool place_cols(')]
        self.assertIn('if (!active()) return false;',cuda)
        self.assertIn('k_slice_cols<<<',cuda);self.assertIn('dy.finish()',cuda)
    def test_pinned_source_and_assets(self):
        root=P.parents[2]
        for name,digest in g.CRITICAL.items():self.assertEqual(g.sha(root/name),digest,name)
        self.assertIn('studio/chat.html',g.CRITICAL)
if __name__=='__main__':unittest.main()
