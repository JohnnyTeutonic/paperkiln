import copy
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path('/home/jonat/e9_gate_root_admission_20261101')
spec = importlib.util.spec_from_file_location('e9_admission_guard', ROOT/'guarded_runner.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir='/home/jonat', prefix='e9_guard_test_')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.payload = self.root/'payload.txt'; self.payload.write_text('bound payload')
        self.jobs = self.root/'jobs.json'; self.jobs.write_text('{"jobs":[]}')
        deadline = (datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(hours=2)).isoformat()
        self.argv = ['python3','/frozen/tools/colab_chat/runner_multi.py','--commit','a'*40,
                     '--max-usd','3','--max-hours','2','--deadline-utc',deadline,
                     '--upload','payload.txt='+str(self.payload)]
        self.record = {'status':'APPROVED_ONE_ATTEMPT_AFTER_SERIAL_ADMISSION',
                       'driver_argv':self.argv,'bounds':{'deadline_utc':deadline,'scientific_seconds':5700},
                       'pins':{str(self.payload):sha(self.payload)},
                       'registered_job_path':str(self.jobs),'registered_job_sha256':sha(self.jobs)}
    def check(self, record=None, args=None):
        p=self.root/'review.json';p.write_text(json.dumps(record or self.record))
        return guard.validate(p,sha(p),args or (record or self.record)['driver_argv'][1:])
    def test_positive(self): self.assertEqual(self.check(),self.argv[1:])
    def test_args_changed(self):
        with self.assertRaises(ValueError): self.check(args=self.argv[1:]+['extra'])
    def test_resource_bounds(self):
        for flag,value in [('--max-usd','4'),('--max-hours','3')]:
            r=copy.deepcopy(self.record);r['driver_argv'][r['driver_argv'].index(flag)+1]=value
            with self.subTest(flag=flag),self.assertRaises(ValueError):self.check(r)
    def test_short_commit(self):
        r=copy.deepcopy(self.record);r['driver_argv'][3]='a'*9
        with self.assertRaises(ValueError):self.check(r)
    def test_expired(self):
        r=copy.deepcopy(self.record);r['bounds']['deadline_utc']='2000-01-01T00:00:00Z';r['driver_argv'][9]=r['bounds']['deadline_utc']
        with self.assertRaises(ValueError):self.check(r)
    def test_inactive(self):
        r=copy.deepcopy(self.record);r['status']='PREPARED'
        with self.assertRaises(ValueError):self.check(r)
    def test_science_timeout(self):
        r=copy.deepcopy(self.record);r['bounds']['scientific_seconds']=6000
        with self.assertRaises(ValueError):self.check(r)
    def test_payload_drift(self):
        self.payload.write_text('altered')
        with self.assertRaises(ValueError):self.check()
    def test_jobs_drift(self):
        self.jobs.write_text('altered')
        with self.assertRaises(ValueError):self.check()
    def test_nested_upload(self):
        r=copy.deepcopy(self.record);r['driver_argv'][-1]='nested/payload.txt='+str(self.payload)
        with self.assertRaises(ValueError):self.check(r)
    def test_symlink_upload(self):
        q=self.root/'link.txt';q.symlink_to(self.payload)
        r=copy.deepcopy(self.record);r['driver_argv'][-1]='payload.txt='+str(q)
        with self.assertRaises(ValueError):self.check(r)
    def test_review_hash(self):
        p=self.root/'review.json';p.write_text(json.dumps(self.record))
        with self.assertRaises(ValueError):guard.validate(p,'0'*64,self.argv[1:])


if __name__=='__main__': unittest.main()
