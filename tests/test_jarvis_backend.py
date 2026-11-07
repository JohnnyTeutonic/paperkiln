"""runner_multi.py --backend jarvis without spending: jarvis_api's guardrails, JarvisTransport, jarvis_sweep.py,
the prelude and one job driven to home by the real runner. `jl`, ssh and rsync are fakes
(tests/jarvis_fake); nothing here talks to Jarvis or any VM. POSIX only (the sweep reads /proc)."""
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
CC = ROOT / "tools" / "colab_chat"
FAKES = ROOT / "tests" / "jarvis_fake"
sys.path.insert(0, str(CC))
pytestmark = pytest.mark.skipif(os.name == "nt", reason="the sweep reads /proc")

# Before the imports: a test that forgot the fixture finds no jl at all, never the real one.
os.environ["JARVIS_JL"] = "/nonexistent/jl"
os.environ["JARVIS_HOME"] = "/nonexistent/jarvis_home"

import jarvis_api as api  # noqa: E402
import jarvis_sweep as sweep  # noqa: E402
import jarvis_transport as jt  # noqa: E402
import runpod_api  # noqa: E402
import runpod_transport as rt  # noqa: E402

FS_ID = 7


def write_exe(path, body):
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


class Account:
    """The fake account, and the fixture's view of it."""

    def __init__(self, base):
        self.dir = base / "jl"
        self.dir.mkdir()
        self.path = self.dir / "account.json"
        self.path.write_text(json.dumps({"balance": 49.96, "stock": 1, "price": 0.44, "instances": [],
                                         "filesystems": [], "next_id": 1000, "next_fs": FS_ID - 1, "create": "ok"}))

    def get(self):
        return json.loads(self.path.read_text())

    def set(self, **kv):
        d = self.get()
        d.update(kv)
        self.path.write_text(json.dumps(d))

    def calls(self, name=None):
        p = self.dir / "calls.jsonl"
        rows = [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
        return [r for r in rows if name is None or r[0] == name]

    @property
    def instances(self):
        return self.get()["instances"]

    def add_vm(self, mid=2000, name="hand-made", status="Running", **kv):
        vm = {"machine_id": mid, "name": name, "status": status, "gpu_type": "L4", "template": "vm", "num_gpus": 1,
              "is_spot": False, "region": "IN2", "fs_id": FS_ID,
              "ssh_command": "ssh -p 22 ubuntu@203.0.113.7" if status == "Running" else None, **kv}
        self.set(instances=self.instances + [vm])


@pytest.fixture
def acct(tmp_path, monkeypatch):
    """A fake `jl` and a private ~/.jarvis for one test."""
    a = Account(tmp_path)
    jl = tmp_path / "bin" / "jl"
    jl.parent.mkdir()
    write_exe(jl, f"#!/bin/sh\nexec {sys.executable} {FAKES}/fake_jl.py \"$@\"\n")
    home = tmp_path / "dot_jarvis"
    home.mkdir()
    monkeypatch.setenv("FAKE_JL_DIR", str(a.dir))
    monkeypatch.setattr(api, "JL", str(jl))
    monkeypatch.setattr(api, "HOME", str(home))
    monkeypatch.setattr(jt, "REGISTRY", str(home / "runners.json"))
    monkeypatch.setattr(sweep, "REGISTRY", str(home / "runners.json"))
    monkeypatch.setattr(sweep, "KEEP", str(home / "keep.json"))
    monkeypatch.setattr(sweep, "ORPHANS", str(home / "orphans.jsonl"))
    monkeypatch.setattr(jt, "UP_POLL_S", 0)
    monkeypatch.setattr(jt, "DESTROY_BACKOFF_S", 0)
    a.home = home
    a.jl = jl
    return a


class State:
    """runner_multi.State's interface, saved to disk as the real one is."""

    def __init__(self, home):
        self.path = Path(home) / "state.json"
        self.d = {"endpoint": None, "launches": 0, "jobs": {}}

    def save(self):
        self.path.write_text(json.dumps(self.d))


@pytest.fixture
def tr(acct, tmp_path, monkeypatch):
    """An attached transport whose VM answers ssh and whose prelude succeeds."""
    home = tmp_path / "state"
    home.mkdir()
    lines = []
    t = jt.JarvisTransport("L4", 50, 5.0, "abc123", "camp", str(home), lines.append)
    t.lines = lines
    st = State(home)
    t.st = st
    t.R(st)
    monkeypatch.setattr(t, "exec", lambda code, timeout=200: (0, "SSH_OK\n"))
    monkeypatch.setattr(t, "_prelude", lambda vm, rate: True)
    t.ss = st
    return t


def led(t):
    return t.R(t.st)["ledger"]["pods"]


# ---------------------------------------------------------------- jarvis_api
def test_without_the_fixture_there_is_no_jl():
    assert api.JL == "/nonexistent/jl" and api.HOME == "/nonexistent/jarvis_home"
    with pytest.raises(api.JarvisError, match="cannot run jl"):
        api.list_instances()


def test_create_passes_exactly_the_measured_command(acct):
    vm = api.create_vm("camp-1", FS_ID, 0.0, 5.0)
    assert acct.calls("create") == [["create", "--gpu", "L4", "--vm", "--region", "IN2", "--storage", "100",
                                     "--fs-id", str(FS_ID), "--name", "camp-1", "--yes", "--json"]]
    assert vm["usd_per_hr"] == 0.44 and vm["status"] == "Running"


def test_no_code_path_can_ask_for_spot():
    for f in ("jarvis_api.py", "jarvis_transport.py", "jarvis_sweep.py"):
        assert '"--spot"' not in (CC / f).read_text()


@pytest.mark.parametrize("account,args,match", [
    ({"balance": 7.99}, {}, "below"),
    ({"stock": 0}, {}, "no L4 VM free"),
    ({"price": 0.51}, {}, "above"),
    ({}, {"spent": 4.7}, "passes the"),
    ({}, {"max_usd": None}, "ceiling"),
    ({}, {"max_usd": 40.01}, "ceiling"),
    ({}, {"gpu": "L40S"}, "only L4"),
    ({}, {"fs_id": None}, "filesystem"),
    ({}, {"name": "has space"}, "name"),
    ({}, {"name": "x" * 41}, "name"),
])
def test_guardrails_refuse_before_any_create(acct, account, args, match):
    acct.set(**account)
    a = {"name": "camp-1", "fs_id": FS_ID, "spent": 0.0, "max_usd": 5.0, **args}
    with pytest.raises(api.Refused, match=match):
        api.create_vm(a.pop("name"), a.pop("fs_id"), a.pop("spent"), a.pop("max_usd"), **a)
    assert acct.calls("create") == []


def test_live_instances_are_counted_before_every_create(acct):
    acct.add_vm(status="Paused")  # a paused VM still bills its disk, so it counts
    with pytest.raises(api.Refused, match="cap 1"):
        api.create_vm("camp-1", FS_ID, 0.0, 5.0)
    assert acct.calls("create") == []


def test_a_failed_count_stops_the_create(acct):
    acct.set(fail=["list"])
    with pytest.raises(api.JarvisError):
        api.create_vm("camp-1", FS_ID, 0.0, 5.0)
    assert acct.calls("create") == []


def test_refused_is_both_families(acct):
    assert issubclass(api.Refused, runpod_api.Refused) and issubclass(api.JarvisError, runpod_api.RunPodError)


def test_two_runners_cannot_both_pass_the_count(acct):
    with api._lock("create"):
        with pytest.raises(api.Refused, match="in progress"):
            api.create_vm("camp-1", FS_ID, 0.0, 5.0)
    assert acct.calls("create") == []


def test_stock_reads_the_vm_row_not_the_container_row(acct):
    assert api.stock() == {"free": 1, "usd_per_hr": 0.44}
    acct.set(stock=0)
    assert api.stock()["free"] == 0


def test_balance_lets_nothing_personal_out(acct):
    assert api.balance() == 49.96
    assert "SENTINEL" not in json.dumps(api.status())


def test_errors_never_quote_output(acct):
    acct.set(garbage=True)
    with pytest.raises(api.JarvisError) as e:
        api.balance()
    assert "SENTINEL" not in str(e.value) and "not JSON" in str(e.value)


def test_a_silent_jl_is_killed(acct, monkeypatch):
    write_exe(acct.jl, "#!/bin/sh\nexec sleep 30\n")
    t0 = time.monotonic()
    with pytest.raises(api.JarvisError, match="no answer"):
        api._jl(["list"], timeout=0.2)
    assert time.monotonic() - t0 < 5


def test_destroy_believes_the_listing(acct):
    acct.add_vm(2000)
    acct.set(lying_destroy=True)
    assert api.destroy_vm(2000) is False
    acct.set(lying_destroy=False, destroy_fails=1)
    assert api.destroy_vm(2000) is False  # jl failed and the VM stands
    assert api.destroy_vm(2000) is True and acct.instances == []
    assert api.destroy_vm(2000) is True  # jl says "not found"; the listing agrees


def test_destroy_counts_destroying_as_done_and_a_dead_listing_as_unknown(acct):
    acct.add_vm(2000, status="Destroying")
    acct.set(lying_destroy=True)
    assert api.destroy_vm(2000) is True
    acct.set(fail=["list"])
    with pytest.raises(api.JarvisError):
        api.destroy_vm(2000)


@pytest.mark.parametrize("cmd,want", [
    ("ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 2222 ubuntu@1.2.3.4", ("1.2.3.4", 2222)),
    ("ssh -p 22 -l ubuntu 1.2.3.4", ("1.2.3.4", 22)),
    ("ssh ubuntu@host-1.example.org", ("host-1.example.org", 22)),
    ("ssh -p 22 root@1.2.3.4", None),
    ("ssh -p 22 ubuntu@1.2.3.4;rm", None),
    ("ssh 'unterminated", None),
])
def test_ssh_endpoint_parses_and_never_trusts(cmd, want):
    assert api.ssh_endpoint({"status": "Running", "ssh_command": cmd}) == want


def test_no_endpoint_unless_running():
    assert api.ssh_endpoint({"status": "Creating", "ssh_command": "ssh ubuntu@1.2.3.4"}) is None


@pytest.mark.parametrize("change,fault", [
    ({"gpu_type": "L40S"}, "GPU"), ({"is_spot": True}, "spot"), ({"num_gpus": 2}, "GPUs"),
    ({"template": "pytorch"}, "template"), ({"region": "IN1"}, "region"),
])
def test_vm_faults(change, fault):
    good = {"gpu_type": "L4", "num_gpus": 1, "template": "vm", "is_spot": False, "region": "IN2"}
    assert api.vm_faults(good) == []
    assert fault in api.vm_faults({**good, **change})[0]


def test_filesystem_guardrails(acct):
    with pytest.raises(api.Refused, match="outside"):
        api.filesystem_create("f", 49)
    with pytest.raises(api.Refused, match="outside"):
        api.filesystem_create("f", 201)
    with pytest.raises(api.Refused, match="name"):
        api.filesystem_create("x" * 31, 50)
    acct.set(balance=7.0)
    with pytest.raises(api.Refused, match="below"):
        api.filesystem_create("f", 50)
    acct.set(balance=20.0, filesystems=[{"fs_id": 1}, {"fs_id": 2}])
    with pytest.raises(api.Refused, match="cap"):
        api.filesystem_create("f", 50)
    assert not [c for c in acct.calls("filesystem") if c[1] == "create"]


def test_filesystem_create_and_remove(acct):
    fs = api.filesystem_create("camp-fs", 50)
    assert acct.get()["filesystems"][0] == {"fs_id": fs, "fs_name": "camp-fs", "storage": 50, "region": "IN2"}
    assert api.filesystem_remove(fs) is True and acct.get()["filesystems"] == []


def test_ledger_usd():
    now = 1000000.0
    led_ = {"pods": [{"usd_per_hr": 0.44, "start": now - 7200, "stop": now - 3600}, {"usd_per_hr": 0.44, "start": now - 1800}],
            "volumes": [{"gb": 100, "created": now - 30 * 86400}]}
    assert abs(api.ledger_usd(led_, now) - (0.44 + 0.22 + 10.0)) < 1e-9


# ---------------------------------------------------------------- the transport's interface
def test_ssh_and_rsync_use_the_key_the_user_and_a_known_hosts_file_per_vm(tmp_path, monkeypatch):
    t = jt.JarvisTransport("L4", 50, 5.0, "c", "camp", str(tmp_path), print)
    t.st = State(tmp_path)
    t.R()["pod"] = {"id": "1234", "ip": "203.0.113.7", "port": 22}
    seen = []
    monkeypatch.setattr(t, "_run", lambda argv, timeout, stdin_text=None: (seen.append(argv), (0, ""))[1])
    t.exec("print(1)")
    t.put("/a", "/content/b", 10)
    t.get("/content/b", "/a", 10)
    assert "ubuntu@203.0.113.7" in seen[0] and "ubuntu@203.0.113.7:/content/b" in seen[1] + seen[2]
    joined = " ".join(seen[0])
    assert "StrictHostKeyChecking=accept-new" in joined and "StrictHostKeyChecking=no" not in joined
    assert f"UserKnownHostsFile={tmp_path}/known_hosts_1234" in joined
    assert f"-i {jt.KEY}" in joined and "root@" not in joined
    t.R()["pod"]["id"] = "1235"
    assert "known_hosts_1235" in " ".join(t._ssh_opts(t.R()["pod"]))


def test_runpod_argv_is_unchanged_by_the_seams(tmp_path, monkeypatch):
    t = rt.RunPodTransport("L4", "SECURE", 10, None, None, None, "c", "x", str(tmp_path), print)
    t.st = State(tmp_path)
    assert list(t.R()) == ["pod", "volume", "retired_volumes", "ledger"] and "runpod" in t.st.d
    t.R()["pod"] = {"id": "p", "ip": "1.2.3.4", "port": 2022}
    seen = []
    monkeypatch.setattr(t, "_run", lambda argv, timeout, stdin_text=None: (seen.append(argv), (0, ""))[1])
    monkeypatch.setattr(t, "_n", 0)
    t.exec("print(1)")
    t.put("/a", "/content/b", 10)
    t.get("/content/b", "/a", 10)
    opts = ["-i", rt.KEY, "-p", "2022", "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"UserKnownHostsFile={tmp_path}/known_hosts_p", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=4", "-o", "BatchMode=yes", "-o", "LogLevel=ERROR"]
    assert seen[0][:len(opts) + 2] == ["ssh", *opts, "root@1.2.3.4"]
    assert seen[1][-2:] == ["/a", "root@1.2.3.4:/content/b"] and seen[2][-2:] == ["root@1.2.3.4:/content/b", "/a"]


def test_runner_interface_is_complete():
    """Every attribute runner_multi.py calls on a transport, present on the Jarvis one and not abstract."""
    t = jt.JarvisTransport("L4", 50, 5.0, "c", "camp", "/tmp", print)
    assert t.backend == "jarvis" and t.whole_files is True and t.persistent is True and t.gpu == "L4"
    for name in ("create", "listed", "adopt", "exec", "put", "get", "stop", "refresh", "gone", "attach", "release",
                 "over_budget", "upload_whole", "download_whole", "spent"):
        assert callable(getattr(t, name)), name


@pytest.mark.parametrize("kw,match", [
    ({"gpu": "L40S"}, "L4 only"), ({"max_usd": None}, "--max-usd"), ({"max_usd": 41}, "--max-usd"),
    ({"name": "with space"}, "--name"), ({"name": "x" * 28}, "--name"), ({"image": "x/y:z"}, "RunPod options"),
    ({"datacenter": "EU-RO-1"}, "RunPod options"), ({"cloud": "COMMUNITY"}, "RunPod options"),
])
def test_transport_refuses_bad_configuration(kw, match):
    a = {"gpu": "L4", "volume_gb": 50, "max_usd": 5.0, "commit": "c", "name": "camp", "home": "/tmp", "log": print, **kw}
    with pytest.raises(api.Refused, match=match):
        jt.JarvisTransport(**a)


def test_attach_registers_before_anything_is_created(acct, tmp_path, monkeypatch):
    home = tmp_path / "s"
    home.mkdir()
    t = jt.JarvisTransport("L4", 50, 5.0, "c", "camp", str(home), print)
    monkeypatch.setattr(jt.rt.atexit, "register", lambda f: None)
    old = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)}
    try:
        t.attach(State(home))
    finally:
        for s, h in old.items():
            signal.signal(s, h)
    assert json.load(open(jt.REGISTRY))[str(home)]["name"] == "camp"
    assert not (Path.home() / ".jarvis" / "runners.json").exists() or str(home) not in (Path.home() / ".jarvis" / "runners.json").read_text()
    assert acct.calls("create") == []


# ---------------------------------------------------------------- create
def test_create_records_the_vm_filesystem_and_ledger(tr, acct):
    rc, out = tr.create(tr.st)
    assert (rc, out) == (0, "1001")
    R = tr.R(tr.st)
    assert R["pod"]["id"] == "1001" and (R["pod"]["ip"], R["pod"]["port"]) == ("203.0.113.7", 22)
    assert R["creating"] is None and R["volume"]["id"] == FS_ID and tr.st.d["endpoint"] == "1001"
    e = led(tr)[0]
    assert (e["id"], e["name"], e["gpu"], e["usd_per_hr"], e["volume"], e["stop"]) == ("1001", "camp-1", "L4", 0.44, FS_ID, None)
    assert acct.instances[0]["fs_id"] == FS_ID and acct.instances[0]["name"] == "camp-1"
    assert json.loads(tr.st.path.read_text())["jarvis"]["pod"]["id"] == "1001"  # on disk, not only in memory


def test_the_name_is_on_disk_before_jl_runs(tr, acct, monkeypatch):
    seen = {}
    real = api.create_vm

    def spy(name, *a, **k):
        seen.update(json.loads(tr.st.path.read_text())["jarvis"]["creating"])
        return real(name, *a, **k)
    monkeypatch.setattr(api, "create_vm", spy)
    tr.create(tr.st)
    assert seen["name"] == "camp-1"


def test_each_create_gets_its_own_name(tr, acct):
    tr.create(tr.st)
    tr.stop(tr.st)
    tr.create(tr.st)
    assert [c[c.index("--name") + 1] for c in acct.calls("create")] == ["camp-1", "camp-2"]


def test_failed_create_before_the_post_leaves_nothing(tr, acct):
    acct.set(create="refused")
    rc, out = tr.create(tr.st)
    R = tr.R(tr.st)
    assert rc == 1 and "No L4" in out and acct.instances == []
    assert R["creating"] is None and R["pod"] is None and led(tr) == []


def test_guardrail_refusal_clears_the_attempt(tr, acct):
    acct.set(stock=0)
    rc, out = tr.create(tr.st)
    assert rc == 1 and out.startswith("refused") and tr.R(tr.st)["creating"] is None


def test_failed_create_after_the_post_destroys_and_records_the_vm(tr, acct):
    acct.set(create="fail_after_post")
    rc, out = tr.create(tr.st)
    R = tr.R(tr.st)
    assert rc == 1 and acct.instances == []
    assert R["pod"] is None and R["creating"] is None and tr.st.d["endpoint"] is None
    e = led(tr)[0]
    assert e["name"] == "camp-1" and e["stop"] and e["stopped_by"] == "runner" and e["usd_per_hr"] > 0


def test_create_of_unknown_fate_blocks_the_next_one_until_it_is_settled(tr, acct):
    acct.set(create="fail_after_post", fail=[])

    real_settle = tr._settle_creating
    calls = []

    def flaky(st):  # the listing is down right after the failed create
        calls.append(1)
        if len(calls) == 2:
            acct.set(fail=["list"])
        real_settle(st)
    tr._settle_creating = flaky
    rc, _ = tr.create(tr.st)
    assert rc == 1 and tr.R(tr.st)["creating"]["name"] == "camp-1" and len(acct.instances) == 1
    n = len(acct.calls("create"))
    rc, _ = tr.create(tr.st)  # still down: no second create
    assert rc == 1 and len(acct.calls("create")) == n and len(acct.instances) == 1
    acct.set(fail=[], create="ok")
    rc, _ = tr.create(tr.st)  # the listing is back: the first VM dies, is recorded, then the second starts
    assert rc == 0 and [v["name"] for v in acct.instances] == ["camp-2"]
    assert [e["name"] for e in led(tr)] == ["camp-1", "camp-2"] and led(tr)[0]["stop"] and not led(tr)[1]["stop"]


def test_lost_ssh_at_start_destroys_and_records(tr, acct, monkeypatch):
    monkeypatch.setattr(tr, "exec", lambda code, timeout=200: (255, "Connection refused"))
    monkeypatch.setattr(jt, "UP_SECONDS", 0.05)
    rc, out = tr.create(tr.st)
    R = tr.R(tr.st)
    assert rc == 1 and "ssh" in out and acct.instances == [] and R["pod"] is None
    assert led(tr)[0]["stop"] and led(tr)[0]["stopped_by"] == "runner"


def test_vm_vanishing_at_start_is_closed_in_the_ledger(tr, acct, monkeypatch):
    real = api.get_vm
    monkeypatch.setattr(api, "get_vm", lambda i: acct.set(instances=[]) or real(i))
    rc, out = tr.create(tr.st)
    assert rc == 1 and tr.R(tr.st)["pod"] is None and led(tr)[0]["stop"]


def test_failed_prelude_destroys_and_records(tr, acct, monkeypatch):
    monkeypatch.setattr(tr, "_prelude", lambda vm, rate: False)
    rc, out = tr.create(tr.st)
    assert rc == 1 and "prelude" in out and acct.instances == [] and led(tr)[0]["stop"]


def test_two_boot_failures_stop_the_renting(tr, acct, monkeypatch):
    monkeypatch.setattr(tr, "_prelude", lambda vm, rate: False)
    for _ in range(4):
        tr.create(tr.st)
    assert len(acct.calls("create")) == jt.BOOT_FAILS_STOP
    assert "boot_fails" in tr.create(tr.st)[1]


def test_a_good_boot_clears_the_failure_count(tr, acct, monkeypatch):
    monkeypatch.setattr(tr, "_prelude", lambda vm, rate: False)
    tr.create(tr.st)
    monkeypatch.setattr(tr, "_prelude", lambda vm, rate: True)
    assert tr.create(tr.st)[0] == 0 and tr.R(tr.st)["boot_fails"] == 0


def test_a_spot_vm_is_destroyed_before_any_ssh(tr, acct, monkeypatch):
    acct.set(return_spot=True)
    monkeypatch.setattr(tr, "exec", lambda *a, **k: pytest.fail("ssh to a refused VM"))
    rc, out = tr.create(tr.st)
    assert rc == 1 and "spot" in out and acct.instances == [] and led(tr)[0]["stop"]


def test_no_second_vm_while_the_first_stands(tr, acct):
    tr.create(tr.st)
    acct.set(destroy_fails=99)
    tr.stop(tr.st)  # three tries, then it stays recorded
    R = tr.R(tr.st)
    assert R["pod"]["id"] == "1001" and not led(tr)[0]["stop"] and "NOT destroyed" in " ".join(tr.lines)
    rc, out = tr.create(tr.st)
    assert rc == 1 and "not destroyed" in out and len(acct.calls("create")) == 1


def test_deadline_during_the_wait_destroys_the_vm(tr, acct, monkeypatch):
    def late(code, timeout=200):
        raise runpod_api.Refused("runner deadline reached during transport operation")
    monkeypatch.setattr(tr, "exec", late)
    rc, out = tr.create(tr.st)
    assert rc == 1 and "deadline" in out and acct.instances == [] and led(tr)[0]["stop"]


def test_create_timeout_never_outruns_the_deadline(tr):
    tr.deadline = time.time() + 100
    assert 95 <= tr._create_timeout() <= 100
    tr.deadline = time.time() + 5000
    assert tr._create_timeout() == api.CREATE_TIMEOUT
    tr.deadline = None
    assert tr._create_timeout() == api.CREATE_TIMEOUT


# ---------------------------------------------------------------- running
def test_listed_and_gone(tr, acct):
    assert tr.listed() is None and tr.gone(tr.st)  # no VM
    tr.create(tr.st)
    assert tr.listed() == "1001" and not tr.gone(tr.st)
    acct.set(instances=[{**acct.instances[0], "status": "Paused", "ssh_command": None}])
    assert tr.listed() is None and tr.gone(tr.st)
    acct.set(instances=[])
    assert tr.listed() is None and tr.gone(tr.st)


def test_a_failed_listing_never_looks_like_a_dead_vm(tr, acct):
    tr.create(tr.st)
    acct.set(fail=["list"])
    assert tr.listed() is None and not tr.gone(tr.st)
    assert tr.R(tr.st)["pod"]["id"] == "1001"


def test_refresh_follows_a_new_ip(tr, acct):
    tr.create(tr.st)
    vm = acct.instances[0]
    acct.set(instances=[{**vm, "ssh_command": "ssh -p 2200 ubuntu@198.51.100.9"}])
    tr.refresh(tr.st)
    assert (tr.R(tr.st)["pod"]["ip"], tr.R(tr.st)["pod"]["port"]) == ("198.51.100.9", 2200)


def test_lost_ssh_mid_run_replaces_the_vm_instead_of_idling(tr, acct, monkeypatch):
    tr.create(tr.st)
    real = jt.JarvisTransport.exec.__get__(tr)  # the fixture's exec answers; this is the real one over a dead _run
    monkeypatch.setattr(tr, "_run", lambda argv, timeout, stdin_text=None: (255, "Connection timed out"))
    monkeypatch.setattr(tr, "exec", real)
    assert real("print(1)")[0] == 255
    assert tr.listed() == "1001"  # a blip is not a loss
    tr._ssh_down_since = time.time() - jt.SSH_LOST_S - 1
    assert tr.listed() is None and tr.gone(tr.st) and acct.instances == []
    assert tr.R(tr.st)["pod"] is None and led(tr)[0]["stop"]


def test_any_answer_resets_the_lost_ssh_clock(tr, acct, monkeypatch):
    tr.create(tr.st)
    real = jt.JarvisTransport.exec.__get__(tr)
    answers = iter([(255, ""), (1, "traceback")])
    monkeypatch.setattr(tr, "_run", lambda argv, timeout, stdin_text=None: next(answers))
    real("x")
    assert tr._ssh_down_since
    real("x")
    assert tr._ssh_down_since is None


def test_spend_cap_and_balance_floor_stop_a_run(tr, acct):
    tr.create(tr.st)
    assert tr.over_budget(tr.st) is None
    led(tr)[0]["start"] = time.time() - 12 * 3600
    assert "--max-usd" in tr.over_budget(tr.st)
    led(tr)[0]["start"] = time.time()
    acct.set(balance=4.5)
    tr.R(tr.st)["balance_start"] = 4.6  # someone else's spend, or a late top-up: the floor stands regardless
    tr.refresh(tr.st)
    assert "below" in tr.over_budget(tr.st)


def test_real_balance_loss_counts_when_above_the_estimate(tr, acct):
    tr.create(tr.st)
    assert tr.spent(tr.st) < 0.01
    acct.set(balance=49.96 - 6.0)
    tr.refresh(tr.st)
    assert abs(tr.spent(tr.st) - 6.0) < 1e-6 and "--max-usd" in tr.over_budget(tr.st)


def test_a_failed_balance_read_leaves_the_last_one(tr, acct):
    tr.create(tr.st)
    acct.set(fail=["status"])
    tr.refresh(tr.st)
    assert tr._balance == 49.96


def test_over_budget_makes_no_api_call(tr, acct):
    tr.create(tr.st)
    n = len(acct.calls())
    for _ in range(5):
        tr.over_budget(tr.st)
    assert len(acct.calls()) == n  # it runs every 2 s during a transfer


def test_refused_before_a_create_when_over_budget(tr, acct):
    tr.max_usd = 0.01
    tr.R(tr.st)["ledger"]["pods"].append({"id": "x", "usd_per_hr": 1.0, "start": time.time() - 3600, "stop": time.time()})
    rc, out = tr.create(tr.st)
    assert rc == 1 and "refused" in out and acct.calls("create") == []


def test_stop_is_idempotent(tr, acct):
    tr.create(tr.st)
    tr.stop(tr.st)
    n = len(acct.calls("destroy"))
    tr.stop(tr.st)
    assert len(acct.calls("destroy")) == n and tr.st.d["endpoint"] is None


# ---------------------------------------------------------------- filesystem
def test_recorded_filesystem_is_reused(tr, acct):
    tr.create(tr.st)
    tr.stop(tr.st)
    tr.create(tr.st)
    assert len(acct.get()["filesystems"]) == 1 and len(tr.R(tr.st)["ledger"]["volumes"]) == 1


def test_a_filesystem_left_by_a_crash_is_found_by_name(tr, acct):
    acct.set(filesystems=[{"fs_id": 55, "fs_name": "camp-fs", "storage": 50, "region": "IN2"}])
    tr.create(tr.st)
    assert tr.R(tr.st)["volume"]["id"] == 55 and acct.instances[0]["fs_id"] == 55
    assert not [c for c in acct.calls("filesystem") if c[1] == "create"]


def test_a_filesystem_in_another_region_is_refused(tr, acct):
    acct.set(filesystems=[{"fs_id": 55, "fs_name": "camp-fs", "storage": 50, "region": "IN1"}])
    assert "not IN2" in tr.create(tr.st)[1] and acct.calls("create") == []


def test_a_vanished_filesystem_stops_the_campaign(tr, acct):
    tr.create(tr.st)
    tr.stop(tr.st)
    acct.set(filesystems=[])  # what a zero balance does
    rc, out = tr.create(tr.st)
    assert rc == 1 and "no longer listed" in out and len(acct.calls("create")) == 1


def test_release_needs_the_campaign_home(tr, acct):
    tr.create(tr.st)
    tr.stop(tr.st)
    tr.st.d["jobs"] = {"a": {"home": False}}
    with pytest.raises(runpod_api.Refused):
        tr.release(tr.st)
    assert acct.get()["filesystems"]


def test_release_removes_the_filesystem_when_home(tr, acct):
    tr.create(tr.st)
    tr.stop(tr.st)
    tr.st.d.update(jobs={"a": {"home": True}}, campaign_home=True)
    tr.release(tr.st)
    R = tr.R(tr.st)
    assert acct.get()["filesystems"] == [] and R["volume"] is None and R["ledger"]["volumes"][0]["deleted"]


# ---------------------------------------------------------------- jarvis_sweep
def runner_state(base, name, section=None, alive=False):
    """A runner's state dir, registered; alive runs a sleeping process that passes runner_alive's proof."""
    d = base / name
    d.mkdir()
    (d / "state.json").write_text(json.dumps({"endpoint": None, "jobs": {}, "jarvis": section or {}}))
    reg = Path(sweep.REGISTRY)
    r = json.loads(reg.read_text()) if reg.exists() else {}
    r[str(d)] = {"name": name, "registered": time.time()}
    reg.write_text(json.dumps(r))
    p = None
    if alive:
        script = base / "runner_multi.py"
        script.write_text("import time\ntime.sleep(600)\n")
        p = subprocess.Popen([sys.executable, str(script), "--state-dir", str(d)])
        (d / "runner.pid").write_text(str(p.pid))
        for _ in range(100):
            if b"--state-dir" in open(f"/proc/{p.pid}/cmdline", "rb").read():
                break
            time.sleep(0.01)
    return d, p


def stop_all(*ps):
    for p in ps:
        if p:
            p.kill()
            p.wait()


def test_sweep_destroys_a_vm_no_runner_knows_and_writes_it_down(acct, tmp_path):
    acct.add_vm(2000, name="stray")
    assert sweep.main([]) == 0
    assert acct.instances == []
    row = json.loads(Path(sweep.ORPHANS).read_text())
    assert (row["id"], row["name"]) == ("2000", "stray")


@pytest.mark.parametrize("status", ["Paused", "Creating", "Failed"])
def test_sweep_destroys_every_status(acct, status):
    acct.add_vm(2000, status=status)
    assert sweep.main([]) == 0 and acct.instances == []


def test_sweep_dry_run_touches_nothing(acct):
    acct.add_vm(2000)
    assert sweep.main(["--dry-run"]) == 0 and len(acct.instances) == 1 and acct.calls("destroy") == []


@pytest.mark.parametrize("entry", [2000, "2000", "by-hand"])
def test_sweep_leaves_the_keep_list(acct, entry):
    acct.add_vm(2000, name="by-hand")
    Path(sweep.KEEP).write_text(json.dumps([entry]))
    assert sweep.main([]) == 0 and len(acct.instances) == 1


def test_sweep_keeps_a_live_runners_vm(acct, tmp_path):
    acct.add_vm(2000, name="camp-1")
    d, p = runner_state(tmp_path, "camp", {"pod": {"id": "2000"}, "ledger": {"pods": []}}, alive=True)
    try:
        assert sweep.main([]) == 0 and len(acct.instances) == 1
    finally:
        stop_all(p)


def test_sweep_ends_a_dead_runners_vm_and_closes_its_ledger(acct, tmp_path):
    acct.add_vm(2000, name="camp-1")
    d, _ = runner_state(tmp_path, "camp", {"pod": {"id": "2000"}, "ledger": {"pods": [
        {"id": "2000", "name": "camp-1", "usd_per_hr": 0.44, "start": time.time() - 60, "stop": None}]}})
    assert sweep.main([]) == 0 and acct.instances == []
    st = json.loads((d / "state.json").read_text())
    e = st["jarvis"]["ledger"]["pods"][0]
    assert e["stop"] and e["stopped_by"] == "sweep" and st["jarvis"]["pod"] is None and st["endpoint"] is None
    assert not Path(sweep.ORPHANS).exists()


def test_sweep_keeps_a_vm_a_live_runner_is_creating(acct, tmp_path):
    acct.add_vm(2000, name="camp-1", status="Creating")
    d, p = runner_state(tmp_path, "camp", {"creating": {"name": "camp-1", "started": time.time()}}, alive=True)
    try:
        assert sweep.main([]) == 0 and len(acct.instances) == 1
    finally:
        stop_all(p)


def test_sweep_ignores_a_create_too_old_to_be_real(acct, tmp_path):
    acct.add_vm(2000, name="camp-1")
    d, p = runner_state(tmp_path, "camp", {"creating": {"name": "camp-1", "started": time.time() - 2000}}, alive=True)
    try:
        assert sweep.main([]) == 0 and acct.instances == []
    finally:
        stop_all(p)


def test_sweep_records_a_vm_whose_runner_died_mid_create(acct, tmp_path):
    acct.add_vm(2000, name="camp-1")
    t0 = time.time() - 30
    d, _ = runner_state(tmp_path, "camp", {"creating": {"name": "camp-1", "started": t0}})
    assert sweep.main([]) == 0 and acct.instances == []
    R = json.loads((d / "state.json").read_text())["jarvis"]
    e = R["ledger"]["pods"][0]
    assert (e["id"], e["name"], e["start"]) == ("2000", "camp-1", t0) and e["stop"] and R["creating"] is None


def test_sweep_spares_a_live_runner_with_a_torn_state(acct, tmp_path):
    acct.add_vm(2000, name="camp-1")
    d, p = runner_state(tmp_path, "camp", {}, alive=True)
    (d / "state.json").write_text("{torn")
    try:
        assert sweep.main([]) == 0 and len(acct.instances) == 1
    finally:
        stop_all(p)


def test_sweep_reads_the_listing_before_the_states(acct, tmp_path, monkeypatch):
    """A create landing between the two reads must be seen in the state, so the state is read second."""
    d, p = runner_state(tmp_path, "camp", {}, alive=True)
    real = api.list_instances

    def listing_then_create():
        rows = real()
        acct.add_vm(2000, name="camp-1", status="Creating")
        return rows + [{"machine_id": 2000, "name": "camp-1", "status": "Creating"}]

    def state_written_during_listing():
        rows = listing_then_create()
        (d / "state.json").write_text(json.dumps({"jarvis": {"creating": {"name": "camp-1", "started": time.time()}}}))
        return rows
    monkeypatch.setattr(api, "list_instances", state_written_during_listing)
    monkeypatch.setattr(api, "get_vm", lambda i: None)
    try:
        assert sweep.main([]) == 0 and len(acct.instances) == 1 and acct.calls("destroy") == []
    finally:
        stop_all(p)


def test_sweep_stops_on_an_unreadable_registry(acct):
    acct.add_vm(2000, name="camp-1")
    Path(sweep.REGISTRY).write_text("{torn")
    assert sweep.main([]) == 1 and len(acct.instances) == 1 and acct.calls("destroy") == []


def test_sweep_reports_a_failed_listing_and_a_failed_destroy(acct):
    acct.set(fail=["list"])
    assert sweep.main([]) == 1
    acct.set(fail=[])
    acct.add_vm(2000)
    acct.set(lying_destroy=True)
    assert sweep.main([]) == 1 and len(acct.instances) == 1 and not Path(sweep.ORPHANS).exists()


def test_sweep_without_jl_does_nothing(acct, monkeypatch):
    monkeypatch.setattr(api, "JL", "/nonexistent/jl")
    assert sweep.main([]) == 0


def test_sweep_never_touches_filesystems(acct, tmp_path, capsys):
    acct.set(filesystems=[{"fs_id": 9, "fs_name": "stray-fs", "storage": 50, "region": "IN2"}])
    runner_state(tmp_path, "camp", {"volume": {"id": 10}})
    assert sweep.main([]) == 0 and len(acct.get()["filesystems"]) == 1
    assert "left alone" in capsys.readouterr().out


# ---------------------------------------------------------------- whole-process failures
def env_for(acct, tmp_path, **kv):
    """A subprocess environment with the fake jl, ssh and rsync first on PATH."""
    fake_root = tmp_path / "fake"
    fake_root.mkdir(exist_ok=True)
    bin_ = tmp_path / "fakebin"
    bin_.mkdir(exist_ok=True)
    for tool in ("ssh", "rsync"):
        (bin_ / tool).unlink(missing_ok=True)
        (bin_ / tool).symlink_to(FAKES / "fake_ssh.py")
    os.chmod(FAKES / "fake_ssh.py", 0o755)
    return dict(os.environ, JARVIS_JL=str(acct.jl), JARVIS_HOME=str(acct.home), FAKE_JL_DIR=str(acct.dir),
                FAKE_ROOT=str(fake_root), PATH=f"{bin_}:{os.environ['PATH']}", **kv)


def launch_runner(acct, tmp_path, env, jobs=None, extra=()):
    jobs_file = tmp_path / "jobs.json"
    jobs_file.write_text(json.dumps(jobs or {"jobs": [{"name": "data", "cmd": ["x"], "outputs": ["out.txt"]}]}))
    state = tmp_path / "state_dir"
    argv = [sys.executable, str(CC / "runner_multi.py"), "--commit", "abc123", "--jobs", str(jobs_file), "--name", "camp",
            "--art", str(tmp_path / "art"), "--state-dir", str(state), "--backend", "jarvis", "--max-usd", "5",
            "--max-hours", "1", *extra]
    p = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         start_new_session=True)
    return p, state


def wait_for(cond, seconds=30):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


def jarvis_state(state):
    return json.loads((state / "state.json").read_text()).get("jarvis", {})


def test_kill_9_mid_create_leaves_no_vm_unrecorded(acct, tmp_path):
    acct.set(create="hang_after_post")
    p, state = launch_runner(acct, tmp_path, env_for(acct, tmp_path))
    try:
        assert wait_for(lambda: acct.instances), "the VM never appeared"
        assert jarvis_state(state)["creating"]["name"] == "camp-1"
        assert json.load(open(state.parent / "dot_jarvis" / "runners.json"))[str(state)]["name"] == "camp"
        p.kill()
        p.wait()
        assert sweep.main([]) == 0 and acct.instances == []
        R = jarvis_state(state)
        assert R["creating"] is None and R["ledger"]["pods"][0]["name"] == "camp-1" and R["ledger"]["pods"][0]["stop"]
    finally:
        stop_all(p)
        subprocess.run(["pkill", "-f", "[f]ake_jl.py create"], check=False)


def test_kill_9_after_the_vm_is_recorded_leaves_no_vm_unrecorded(acct, tmp_path):
    env = env_for(acct, tmp_path)
    (Path(env["FAKE_ROOT"]) / "ssh_hang").touch()
    p, state = launch_runner(acct, tmp_path, env)
    try:
        assert wait_for(lambda: (state / "state.json").exists() and jarvis_state(state).get("pod"))
        vm = jarvis_state(state)["pod"]["id"]
        p.kill()
        p.wait()
        assert sweep.main([]) == 0 and acct.instances == []
        R = jarvis_state(state)
        assert R["pod"] is None and R["ledger"]["pods"][0]["id"] == vm and R["ledger"]["pods"][0]["stopped_by"] == "sweep"
    finally:
        stop_all(p)
        subprocess.run(["pkill", "-f", f"[{str(tmp_path)[0]}]{str(tmp_path)[1:]}/fakebin/ssh"], check=False)


def test_sigterm_mid_create_destroys_and_records_the_vm(acct, tmp_path):
    acct.set(create="hang_after_post")
    p, state = launch_runner(acct, tmp_path, env_for(acct, tmp_path))
    try:
        assert wait_for(lambda: acct.instances)
        p.send_signal(signal.SIGTERM)
        p.wait(timeout=30)
        assert acct.instances == []
        R = jarvis_state(state)
        assert R["creating"] is None and R["pod"] is None and R["ledger"]["pods"][0]["stop"]
    finally:
        stop_all(p)


def test_runner_exits_if_the_state_ran_another_gpu(acct, tmp_path):
    state = tmp_path / "state_dir"
    state.mkdir()
    (state / "state.json").write_text(json.dumps({"endpoint": None, "launches": 0, "jobs": {}, "gpu": "T4"}))
    p, _ = launch_runner(acct, tmp_path, env_for(acct, tmp_path))
    out, _ = p.communicate(timeout=30)
    assert p.returncode != 0 and "refusing --gpu L4" in out and acct.calls("create") == []


@pytest.mark.parametrize("extra,why", [(("--gpu", "L40S"), "L4 only"), (("--image", "a/b:c"), "RunPod options")])
def test_runner_refuses_a_bad_configuration_before_any_call(acct, tmp_path, extra, why):
    p, _ = launch_runner(acct, tmp_path, env_for(acct, tmp_path), extra=extra)
    out, _ = p.communicate(timeout=30)
    assert p.returncode != 0 and why in out and acct.calls("create") == []


def run_to_end(acct, tmp_path, jobs):
    """The real runner_multi.py, tick 0, against the fakes (the wrapper run_eq.py uses)."""
    env = env_for(acct, tmp_path, FAKE_JOBS=str(tmp_path / "jobs.json"), FAKE_STEPS="200")
    (tmp_path / "jobs.json").write_text(json.dumps(jobs))
    wrap = tmp_path / "wrap.py"
    wrap.write_text("import importlib.util, sys\n"
                    "spec = importlib.util.spec_from_file_location('runner_multi', sys.argv[1])\n"
                    "rm = importlib.util.module_from_spec(spec)\n"
                    "sys.path.insert(0, __import__('os').path.dirname(sys.argv[1]))\n"
                    "spec.loader.exec_module(rm)\n"
                    "rm.TICK = 0\n"
                    "sys.argv = ['runner_multi.py'] + sys.argv[2:]\n"
                    "rm.main()\n")
    state = tmp_path / "state_dir"
    p = subprocess.run([sys.executable, str(wrap), str(CC / "runner_multi.py"), "--commit", "abc123", "--jobs",
                        str(tmp_path / "jobs.json"), "--name", "camp", "--art", str(tmp_path / "art"),
                        "--state-dir", str(state), "--backend", "jarvis", "--max-usd", "5", "--max-hours", "0.2"],
                       env=env, capture_output=True, text=True, timeout=300)
    return p, state


JOB = {"jobs": [{"name": "data", "cmd": ["x"], "outputs": ["out.txt"]},
                {"name": "pt", "model_name": "m", "after": "data", "spec": {}}]}


def test_one_job_runs_to_home_on_the_real_runner(acct, tmp_path):
    p, state = run_to_end(acct, tmp_path, JOB)
    log = (state / "runner.log").read_text()
    assert p.returncode == 0, p.stderr[-2000:] + log[-2000:]
    st = json.loads((state / "state.json").read_text())
    R = st["jarvis"]
    assert "all jobs finished and home" in log and st["campaign_home"] is True
    assert st["gpu"] == "L4" and st["backend"] == "jarvis" and "runpod" not in st
    assert all(j["home"] for j in st["jobs"].values())
    assert acct.instances == [] and acct.get()["filesystems"] == []  # destroyed, then released
    assert R["pod"] is None and R["volume"] is None and R["creating"] is None
    assert len(R["ledger"]["pods"]) == 1 and R["ledger"]["pods"][0]["stop"]
    assert (tmp_path / "art" / "data" / "out.txt").exists()
    assert (tmp_path / "art" / "pt" / "m.safetensors").exists()
    assert not (state / "runner.pid").exists()
    assert len(acct.calls("create")) == 1


def test_a_failed_prelude_on_the_real_runner_leaves_nothing_billing(acct, tmp_path):
    p, state = launch_runner(acct, tmp_path, env_for(acct, tmp_path, FAKE_PRELUDE_RC="3"), extra=("--max-hours", "0.0005"))
    p.communicate(timeout=60)  # about two seconds, then the deadline
    assert len(acct.calls("create")) == 1 and acct.instances == []
    R = jarvis_state(state)
    assert R["pod"] is None and R["boot_fails"] == 1 and R["ledger"]["pods"][0]["stop"]


# ---------------------------------------------------------------- jarvis_prelude.sh
PRELUDE_TOOLS = {
    "sudo": 'while [ "$1" = -n ] || [ "$1" = -H ]; do shift; done\nexec "$@"\n',
    "apt-get": "exit 0\n",
    "cloud-init": "exit ${FAKE_CLOUD_RC:-0}\n",
    "findmnt": 'echo "${FAKE_FSTYPE:-virtiofs}"\n',
    "loginctl": "exit 0\n",
    "ldconfig": "exit 0\n",
    "tee": "cat > /dev/null\n",
    "cmake": 'echo "cmake version ${FAKE_CMAKE:-3.31.6}"\n',
    "curl": 'while [ "$1" != -o ]; do shift; done\ntouch "$2"\n',
    "md5sum": "cat > /dev/null\nexit 0\n",
    "nvidia-smi": 'case "$1" in --query*) echo "${FAKE_GPU:-NVIDIA L4}";; *) echo "| CUDA Version: ${FAKE_DRIVER:-13.0} |";; esac\n',
    "python3": "exit 0\n",
    "sleep": "exit 0\n",
}


@pytest.fixture
def vm(tmp_path):
    """A pretend VM: the prelude's external commands are stubs that log and can be told to fail."""
    bin_ = tmp_path / "vmbin"
    bin_.mkdir()
    for tool, body in PRELUDE_TOOLS.items():
        write_exe(bin_ / tool, '#!/bin/bash\n'
                  f'echo "{tool} $*" >> "$FAKE_LOG"\n'
                  f'if [ "${{FAKE_FAIL:-}}" = {tool} ]; then exit 42; fi\n' + body)
    fs, cuda = tmp_path / "jl_fs", tmp_path / "cuda"
    fs.mkdir()
    log = tmp_path / "vm.log"
    env = dict(os.environ, PATH=f"{bin_}:{os.environ['PATH']}", PK_FS_ROOT=str(fs), PK_CONTENT=str(tmp_path / "content"),
               PK_CUDA_DIR=str(cuda), FAKE_LOG=str(log))

    def nvcc(release):
        (cuda / "bin").mkdir(parents=True, exist_ok=True)
        write_exe(cuda / "bin" / "nvcc", f'#!/bin/sh\necho "Cuda compilation tools, release {release}, V{release}.131"\n')

    def run(**kv):
        return subprocess.run(["bash", str(CC / "jarvis_prelude.sh"), "abc123", "camp"], capture_output=True, text=True,
                              env={**env, **kv}, timeout=60)

    class V:
        pass
    v = V()
    v.run, v.nvcc, v.log, v.fs, v.content, v.bin, v.tmp = run, nvcc, log, fs, tmp_path / "content", bin_, tmp_path
    v.nvcc("12.4")
    return v


def test_prelude_happy_path(vm):
    p = vm.run()
    assert p.returncode == 0, p.stdout + p.stderr
    assert "prelude done" in p.stdout
    assert vm.content.is_symlink() and os.readlink(vm.content) == str(vm.fs / "camp")
    assert (vm.fs / "camp" / "chat" / "upload").is_dir()
    calls = vm.log.read_text()
    assert "curl" not in calls  # nvcc 12.4 is already there
    assert "--root-user-action" not in calls and "cmake==3.31.6" in calls
    assert "DPkg::Lock::Timeout=300" in calls


def test_prelude_clears_a_previous_vms_leftovers(vm):
    chat = vm.fs / "camp" / "chat"
    (chat / "upload" / "job").mkdir(parents=True)
    (chat / "upload" / "job" / "stale.part00").write_text("x")
    (chat / "jobs_vm.pid").write_text("1")
    (chat / "STATUS").write_text("RUNNING")
    (chat / "keep.txt").write_text("x")
    assert vm.run().returncode == 0
    assert not (chat / "upload" / "job" / "stale.part00").exists() and not (chat / "jobs_vm.pid").exists()
    assert not (chat / "STATUS").exists() and (chat / "keep.txt").exists()


def test_prelude_is_idempotent(vm):
    assert vm.run().returncode == 0
    assert vm.run().returncode == 0


def test_prelude_installs_nvcc_when_missing(vm):
    shutil.rmtree(vm.tmp / "cuda")
    real_sudo = (vm.bin / "sudo").read_text()
    # The runfile is a stub: "sudo sh FILE --silent --toolkit" installs nvcc 12.4 at the pretend prefix.
    write_exe(vm.bin / "sudo", real_sudo.replace('exec "$@"', 'if [ "$1" = sh ]; then mkdir -p "$PK_CUDA_DIR/bin"; '
              'printf "#!/bin/sh\\necho release 12.4\\n" > "$PK_CUDA_DIR/bin/nvcc"; chmod +x "$PK_CUDA_DIR/bin/nvcc"; exit 0; fi\n'
              'exec "$@"'))
    p = vm.run()
    assert p.returncode == 0, p.stdout + p.stderr
    calls = vm.log.read_text()
    assert "curl" in calls and "md5sum" in calls and "--silent --toolkit" in calls


@pytest.mark.parametrize("tool", ["findmnt", "cloud-init", "apt-get", "python3", "cmake", "nvidia-smi", "loginctl",
                                  "ldconfig", "tee"])
def test_prelude_stops_at_any_failed_step(vm, tool):
    p = vm.run(FAKE_FAIL=tool)
    assert p.returncode != 0 and "prelude done" not in p.stdout, tool


@pytest.mark.parametrize("tool", ["curl", "md5sum"])
def test_prelude_stops_when_the_toolkit_cannot_be_fetched_or_verified(vm, tool):
    shutil.rmtree(vm.tmp / "cuda")
    p = vm.run(FAKE_FAIL=tool)
    assert p.returncode != 0 and "prelude done" not in p.stdout


def test_prelude_stops_when_sudo_is_refused(vm):
    p = vm.run(FAKE_FAIL="sudo")
    assert p.returncode != 0 and "prelude done" not in p.stdout


@pytest.mark.parametrize("env,why", [
    ({"FAKE_GPU": "NVIDIA L40S"}, "not NVIDIA L4"), ({"FAKE_GPU": "NVIDIA RTX 4090"}, "not NVIDIA L4"),
    ({"FAKE_DRIVER": "11.8"}, "older than 12"), ({"FAKE_FSTYPE": "ext4"}, "not virtiofs"),
    ({"FAKE_CMAKE": "3.22.1"}, "not 3.31.6"), ({"FAKE_CLOUD_RC": "1"}, "cloud-init"),
])
def test_prelude_refuses_the_wrong_machine(vm, env, why):
    p = vm.run(**env)
    assert p.returncode != 0 and why in p.stdout and "prelude done" not in p.stdout


def test_prelude_accepts_cloud_init_warnings(vm):
    assert vm.run(FAKE_CLOUD_RC="2").returncode == 0


def test_prelude_refuses_a_real_content_folder(vm):
    vm.content.mkdir()
    p = vm.run()
    assert p.returncode != 0 and "not a link" in p.stdout


def test_prelude_refuses_another_nvcc_that_will_not_be_replaced(vm):
    vm.nvcc("11.8")
    p = vm.run()  # the stub runfile installs nothing, so 11.8 stays
    assert p.returncode != 0 and "not 12.4" in p.stdout


def test_prelude_checks_the_gpu_before_installing_anything(vm):
    vm.run(FAKE_GPU="NVIDIA L40S")
    assert "apt-get" not in vm.log.read_text()


def test_prelude_removes_a_build_of_another_commit(vm):
    pk = vm.fs / "camp" / "pk"
    subprocess.run(["git", "init", "-q", str(pk)], check=True)
    subprocess.run(["git", "-C", str(pk), "-c", "user.email=a@b", "-c", "user.name=a", "commit", "-q",
                    "--allow-empty", "-m", "x"], check=True)
    assert vm.run().returncode == 0 and not pk.exists()


def test_prelude_pins_the_toolkit_checksum():
    t = (CC / "jarvis_prelude.sh").read_text()
    assert "CUDA_MD5=afc99bab1d8c6579395d851d948ca3c1" in t and "12.4.1_550.54.15" in t
