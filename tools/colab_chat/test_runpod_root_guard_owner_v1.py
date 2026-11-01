"""Pinned runpy owner proof. All state is synthetic; no provider operations or keys."""
import copy,datetime,hashlib,io,json,os,shutil,subprocess,sys,time
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import runpod_sweep as s


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    state=tmp_path/'state';state.mkdir();script=tmp_path/'guard.py';admission=tmp_path/'admission.json';dependency=tmp_path/'transport.py';dependency.write_text('# pinned synthetic transport\n')
    script.write_text("""import datetime,json,hashlib,sys,time
from pathlib import Path

def validate(record):
 if record['status']!='ROOT_ADMITTED' or record['cap']!=7:raise ValueError('admission bounds')
 deadline=datetime.datetime.strptime(record['deadline_utc'],'%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=datetime.timezone.utc)
 if deadline<=datetime.datetime.now(datetime.timezone.utc):raise ValueError('expired')
 for path,wanted in record['source_pins'].items():
  if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=wanted:raise ValueError('source drift')
 return ['python3',record.get('runner','/frozen/runner_multi.py'),'--state-dir',record['state_dir'],'--deadline-utc',record['deadline_utc']]

if __name__=='__main__':
 validate(json.loads(Path(sys.argv[1]).read_text()))
 print('SYNTHETIC_OWNER_NO_PROVIDER',flush=True)
 time.sleep(30)
""")
    record=dict(status='ROOT_ADMITTED',cap=7,deadline_utc=(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(hours=1)).strftime('%Y-%m-%dT%H:%M:%SZ'),state_dir=str(state),source_pins={str(dependency):sha(dependency)})
    admission.write_text(json.dumps(record));known=dict(kind='seed1',sha256=sha(script),admission=str(admission),admission_sha256=sha(admission),state_dir=str(state))
    monkeypatch.setattr(s,'ADMITTED_ROOT_GUARDS',{str(script):known})
    args=['python3',str(script),str(admission)]
    snap=dict(args=args,start_ticks=12345,started_epoch=time.time()-1,uid=os.getuid(),exe=os.path.realpath(sys.executable))
    return state,script,admission,dependency,record,known,snap


def fake_pid(f,monkeypatch):
    state,script,admission,dependency,record,known,snap=f
    monkeypatch.setattr(s,'process_snapshot',lambda pid:copy.deepcopy(snap))
    return snap


def test_exact_source_admission_state_and_live_pid_owner(fixture,monkeypatch):
    snap=fake_pid(fixture,monkeypatch)
    assert s.admitted_root_guard_pid_alive(123,str(fixture[0]))


@pytest.mark.parametrize('defect',['wrong_script','extra_argument','empty_argument','wrong_admission','script_as_data','wrong_python','wrong_state','false_uid','false_exe','future_start','missing_pid'])
def test_unrelated_or_stale_process_cannot_protect_pod(fixture,monkeypatch,defect):
    snap=fake_pid(fixture,monkeypatch);state=str(fixture[0])
    if defect=='wrong_script':snap['args'][1]='/tmp/guard.py'
    if defect=='extra_argument':snap['args']+=['--extra']
    if defect=='empty_argument':snap['args']+=['']
    if defect=='wrong_admission':snap['args'][2]='/tmp/admission.json'
    if defect=='script_as_data':snap['args']=['python3','/tmp/unrelated.py']+snap['args'][1:]
    if defect=='wrong_python':snap['args'][0]='sh'
    if defect=='wrong_state':state='/tmp/other_state'
    if defect=='false_uid':snap['uid']+=1
    if defect=='false_exe':snap['exe']='/tmp/fake/python3'
    if defect=='future_start':snap['started_epoch']=time.time()+100
    if defect=='missing_pid':monkeypatch.setattr(s,'process_snapshot',lambda pid:(_ for _ in ()).throw(ProcessLookupError()))
    assert not s.admitted_root_guard_pid_alive(123,state)


@pytest.mark.parametrize('defect',['script','admission','source_pin','inactive','cap','symlink'])
def test_source_manifest_or_admission_drift_refused(fixture,monkeypatch,defect):
    state,script,admission,dependency,record,known,snap=fixture;fake_pid(fixture,monkeypatch)
    if defect=='script':script.write_text(script.read_text()+'\n# drift\n')
    if defect=='admission':admission.write_text(admission.read_text()+'\n')
    if defect=='source_pin':dependency.write_text('# changed')
    if defect in ['inactive','cap']:
        record['status' if defect=='inactive' else 'cap']='unapproved' if defect=='inactive' else 8
        admission.write_text(json.dumps(record));known['admission_sha256']=sha(admission)
    if defect=='symlink':
        target=script.with_name('actual.py');script.rename(target);script.symlink_to(target)
    assert not s.admitted_root_guard_pid_alive(123,str(state))


def test_stale_validation_is_not_cached(fixture,monkeypatch):
    fake_pid(fixture,monkeypatch);assert s.admitted_root_guard_pid_alive(123,str(fixture[0]))
    fixture[3].write_text('# drift after approval')
    assert not s.admitted_root_guard_pid_alive(123,str(fixture[0]))


def test_pid_reuse_or_exec_during_validation_refused(fixture,monkeypatch):
    snap=fixture[-1];changed=copy.deepcopy(snap);changed['start_ticks']+=1
    snapshots=iter([copy.deepcopy(snap),changed]);monkeypatch.setattr(s,'process_snapshot',lambda pid:next(snapshots))
    assert not s.admitted_root_guard_pid_alive(123,str(fixture[0]))


def test_existing_live_owner_remains_protected_during_expired_deadline_cleanup(fixture,monkeypatch):
    state,script,admission,dependency,record,known,snap=fixture
    deadline=datetime.datetime.now(datetime.timezone.utc)-datetime.timedelta(minutes=1)
    record['deadline_utc']=deadline.strftime('%Y-%m-%dT%H:%M:%SZ');admission.write_text(json.dumps(record));known['admission_sha256']=sha(admission)
    snap['started_epoch']=deadline.timestamp()-60;fake_pid(fixture,monkeypatch)
    assert s.admitted_root_guard_pid_alive(123,str(state))
    snap['started_epoch']=deadline.timestamp()+1
    assert not s.admitted_root_guard_pid_alive(123,str(state))


def test_real_proc_guard_commandline_and_false_pid(fixture):
    state,script,admission,dependency,record,known,snap=fixture
    python=shutil.which('python3')
    process=subprocess.Popen([python,str(script),str(admission)],stdout=subprocess.PIPE,text=True)
    try:
        assert process.stdout.readline().strip()=='SYNTHETIC_OWNER_NO_PROVIDER'
        assert s.admitted_root_guard_pid_alive(process.pid,str(state))
        assert not s.admitted_root_guard_pid_alive(os.getpid(),str(state))
    finally:
        process.terminate();process.wait(timeout=5);process.stdout.close()


def test_exact_guard_dispatch_in_runner_alive_does_not_accept_embedded_runner_string(fixture,monkeypatch):
    import builtins
    state,script,admission,dependency,record,known,snap=fixture
    (state/'runner.pid').write_text('123');fake_pid(fixture,monkeypatch)
    snap['args']+=['/tmp/runner_multi.py','--state-dir',str(state)]
    oldopen=builtins.open
    def read(path,*a,**kw):
        if str(path)=='/proc/123/cmdline':return io.BytesIO(('\0'.join(snap['args'])+'\0').encode())
        return oldopen(path,*a,**kw)
    monkeypatch.setattr(builtins,'open',read)
    assert not s.runner_alive(str(state))


@pytest.fixture
def recovery(fixture,monkeypatch):
    state,oldguard,oldadmission,dependency,original,oldknown,snap=fixture
    base=state.parent/'runner.py';base.write_text("PINNED_OUTSIDE=1\ndef boot_launch_code():\n    return 'old'\n")
    modified=state.parent/'runner_recovery.py';modified.write_text("PINNED_OUTSIDE=1\ndef boot_launch_code():\n    return 'recovered'\n")
    helper=state.parent/'helper.py';helper.write_text('# reviewed helper\n')
    original['runner']=str(base);oldadmission.write_text(json.dumps(original));oldknown['admission_sha256']=sha(oldadmission)
    script=state.parent/'guarded_recovery.py'
    script.write_text("""import ast,hashlib,importlib.util,json
from pathlib import Path

def outside_boot(source):
 tree=ast.parse(source);nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='boot_launch_code']
 if len(nodes)!=1:raise ValueError('boot inventory')
 n=nodes[0];lines=source.splitlines(keepends=True);return ''.join(lines[:n.lineno-1]+lines[n.end_lineno:])

def validate(record):
 if record['kind']!='ROOT_SFULL_OPERATIONAL_BOOT_RECOVERY_V1' or record['adopted'] is not True:raise ValueError('inactive recovery')
 for path,wanted in record['source_pins'].items():
  if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=wanted:raise ValueError('recovery drift')
 spec=importlib.util.spec_from_file_location('original_sfull_root_guard',record['original_guard'])
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 original=json.loads(Path(record['original_admission']).read_text());argv=module.validate(original)
 modified=record['modified_runner'];helper=record['recovery_helper']
 if any(p not in record['source_pins'] for p in [argv[1],modified,helper]):raise ValueError('missing recovery pin')
 if outside_boot(Path(argv[1]).read_text())!=outside_boot(Path(modified).read_text()):raise ValueError('outside boot drift')
 if record['deadline_utc']!=original['deadline_utc'] or record['state_dir']!=original['state_dir']:raise ValueError('bounds/state changed')
 argv[1]=modified;return argv
""")
    record=dict(kind='ROOT_SFULL_OPERATIONAL_BOOT_RECOVERY_V1',adopted=True,original_guard=str(oldguard),original_admission=str(oldadmission),modified_runner=str(modified),recovery_helper=str(helper),state_dir=str(state),deadline_utc=original['deadline_utc'],source_pins={str(path):sha(path) for path in [oldguard,oldadmission,base,modified,helper]})
    admission=state.parent/'recovery_admission.json';admission.write_text(json.dumps(record))
    known=dict(kind='recovery',sha256=sha(script),admission=str(admission),admission_sha256=sha(admission),state_dir=str(state),original_guard=str(oldguard))
    s.ADMITTED_ROOT_GUARDS[str(script)]=known
    snap['args']=['python3',str(script),str(admission)]
    return fixture,script,admission,record,known,base,modified,helper


def test_reviewed_recovery_only_substitutes_boot_runner(recovery,monkeypatch):
    f,*_=recovery;fake_pid(f,monkeypatch)
    assert s.admitted_root_guard_pid_alive(123,str(f[0]))


@pytest.mark.parametrize('defect',['source','record','inactive','old_guard','old_admission','missing_helper','helper_drift','outside_boot','deadline','state'])
def test_recovery_guard_source_admission_and_scope_negatives(recovery,monkeypatch,defect):
    f,script,admission,record,known,base,modified,helper=recovery;fake_pid(f,monkeypatch)
    if defect=='source':script.write_text(script.read_text()+'\n# drift')
    elif defect=='record':admission.write_text(admission.read_text()+'\n')
    else:
        if defect=='inactive':record['adopted']=False
        if defect=='old_guard':record['original_guard']='/tmp/other_guard.py'
        if defect=='old_admission':record['original_admission']='/tmp/other_admission.json'
        if defect=='missing_helper':del record['source_pins'][str(helper)]
        if defect=='helper_drift':helper.write_text('# changed helper')
        if defect=='outside_boot':
            modified.write_text(modified.read_text().replace('PINNED_OUTSIDE=1','PINNED_OUTSIDE=2'));record['source_pins'][str(modified)]=sha(modified)
        if defect=='deadline':record['deadline_utc']='2099-01-01T00:00:00Z'
        if defect=='state':record['state_dir']='/tmp/other_state'
        admission.write_text(json.dumps(record));known['admission_sha256']=sha(admission)
    assert not s.admitted_root_guard_pid_alive(123,str(f[0]))


def test_recovery_live_cleanup_after_deadline_uses_original_launch_clock(recovery,monkeypatch):
    f,script,admission,record,known,base,modified,helper=recovery
    state,oldguard,oldadmission,dependency,original,oldknown,snap=f
    deadline=datetime.datetime.now(datetime.timezone.utc)-datetime.timedelta(minutes=1)
    original['deadline_utc']=deadline.strftime('%Y-%m-%dT%H:%M:%SZ');oldadmission.write_text(json.dumps(original));oldknown['admission_sha256']=sha(oldadmission)
    record['deadline_utc']=original['deadline_utc'];record['source_pins'][str(oldadmission)]=sha(oldadmission);admission.write_text(json.dumps(record));known['admission_sha256']=sha(admission)
    snap['started_epoch']=deadline.timestamp()-60;fake_pid(f,monkeypatch)
    assert s.admitted_root_guard_pid_alive(123,str(state))
    snap['started_epoch']=deadline.timestamp()+1
    assert not s.admitted_root_guard_pid_alive(123,str(state))


def test_runner_pid_integration_and_pidfile_replacement(fixture,monkeypatch):
    import builtins
    state,*_=fixture;snap=fake_pid(fixture,monkeypatch);(state/'runner.pid').write_text('123')
    oldopen=builtins.open
    def read(path,*a,**kw):
        if str(path)=='/proc/123/cmdline':return io.BytesIO(('\0'.join(snap['args'])+'\0').encode())
        return oldopen(path,*a,**kw)
    monkeypatch.setattr(builtins,'open',read)
    assert s.runner_alive(str(state))
    original=s.validate_root_guard
    def replaced(*a,**kw):
        result=original(*a,**kw);(state/'runner.pid').write_text('456');return result
    monkeypatch.setattr(s,'validate_root_guard',replaced)
    assert not s.runner_alive(str(state))
