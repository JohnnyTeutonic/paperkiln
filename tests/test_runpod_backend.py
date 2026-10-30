"""runner_multi.py --backend runpod without spending: runpod_api's guardrails (pod cap, balance floor),
the spend ledger, runpod_sweep.py's decisions (dead runner, unknown pod, live runner, creating, keep
list, volumes of home campaigns), RunPodTransport.stop on a signal, and provenance (jobs_vm.py stamps a
finished training job's result.json; tiny_chat_run.py's summaries carry it). The API is a fake;
nothing here talks to RunPod. POSIX only (the sweep reads /proc)."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
CC = ROOT / "tools" / "colab_chat"
sys.path.insert(0, str(CC))
pytestmark = pytest.mark.skipif(os.name == "nt", reason="the sweep and jobs_vm.py read /proc")

import runpod_api as api  # noqa: E402
import runpod_sweep as sweep  # noqa: E402
import runpod_transport as rt  # noqa: E402


class FakeAPI:
    def __init__(self, pods=(), vols=(), balance=100.0):
        self.pods, self.vols, self.bal = list(pods), list(vols), balance
        self.created, self.terminated, self.deleted = [], [], []

    def install(self, mp):
        mp.setattr(api, "list_pods", lambda: [dict(p) for p in self.pods])
        mp.setattr(api, "list_volumes", lambda: [dict(v) for v in self.vols])
        mp.setattr(api, "balance", lambda: (self.bal, 0.0))
        mp.setattr(api, "terminate_pod", self.terminate)
        mp.setattr(api, "delete_volume", self.delete)
        mp.setattr(api, "rest", self.rest)
        mp.setattr(api, "get_pod", lambda i: next((dict(p) for p in self.pods if p["id"] == i), None))

    def rest(self, method, path, body=None, timeout=60):
        assert (method, path) == ("POST", "/pods")
        pod = {"id": f"pod{len(self.created)}", "name": body["name"], "desiredStatus": "RUNNING", "costPerHr": 0.59}
        self.created.append(body)
        self.pods.append(pod)
        return 201, pod

    def terminate(self, i):
        self.terminated.append(i)
        self.pods = [p for p in self.pods if p["id"] != i]
        return True

    def delete(self, i):
        self.deleted.append(i)
        self.vols = [v for v in self.vols if v["id"] != i]
        return True


def test_create_pod_guardrails(monkeypatch):
    monkeypatch.setattr(api, "MAX_PODS", 4)  # fixture cap; production account limit remains unchanged
    f = FakeAPI(pods=[{"id": i} for i in "abcd"])
    f.install(monkeypatch)
    with pytest.raises(api.Refused, match="cap 4"):
        api.create_pod({"name": "x"})
    f.pods = [{"id": "a"}]
    f.bal = 14.99
    with pytest.raises(api.Refused, match="below"):
        api.create_pod({"name": "x"})
    assert not f.created
    f.bal = 15.0
    assert api.create_pod({"name": "x"})["id"] == "pod0"


def test_ledger_usd():
    t = 1_000_000.0
    led = {"pods": [{"usd_per_hr": 0.6, "start": t, "stop": t + 1800}, {"usd_per_hr": 1.0, "start": t, "stop": None}],
           "volumes": [{"gb": 50, "created": t, "deleted": t + 30 * 86400}]}
    assert api.ledger_usd(led, now=t + 3600) == pytest.approx(0.3 + 1.0 + 50 * 0.07)


def runner_state(tmp, name, pod=None, vol=None, home=False, creating=None, alive=False):
    d = tmp / name
    d.mkdir()
    st = {"endpoint": pod, "launches": 1, "jobs": {"completed": {"home": True}} if home else {}, "campaign_home": home,
          "runpod": {"pod": {"id": pod} if pod else None, "volume": {"id": vol, "dc": "X"} if vol else None,
                     "retired_volumes": [], "creating": creating,
                     "ledger": {"pods": [{"id": pod, "start": 1.0, "stop": None, "usd_per_hr": 0.5}] if pod else [],
                                "volumes": [{"id": vol, "created": 1.0, "gb": 10}] if vol else []}}}
    (d / "state.json").write_text(json.dumps(st))
    proc = None
    if alive:  # a process whose command line looks like this runner's
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "runner_multi.py",
                                 "--state-dir", str(d)])
        (d / "runner.pid").write_text(str(proc.pid))
    else:
        (d / "runner.pid").write_text("999999")
    return str(d), proc


def test_sweep(tmp_path, monkeypatch):
    live, p1 = runner_state(tmp_path, "live", pod="P_live", vol="V_live", alive=True)
    dead, _ = runner_state(tmp_path, "dead", pod="P_dead", vol="V_dead")
    home, _ = runner_state(tmp_path, "home", vol="V_home", home=True)
    mk, p2 = runner_state(tmp_path, "making", creating=time.time(), alive=True)
    reg = {live: {"name": "pk-live"}, dead: {"name": "pk-dead"}, home: {"name": "pk-home"}, mk: {"name": "pk-make"}}
    regf, keepf = tmp_path / "runners.json", tmp_path / "keep.json"
    regf.write_text(json.dumps(reg))
    keepf.write_text(json.dumps(["by-hand"]))
    monkeypatch.setattr(sweep, "REGISTRY", str(regf))
    monkeypatch.setattr(sweep, "KEEP", str(keepf))
    keyf = tmp_path / "key"
    keyf.write_text("k")
    monkeypatch.setattr(api, "KEY_PATH", str(keyf))
    pods = [{"id": "P_live", "name": "pk-live"}, {"id": "P_dead", "name": "pk-dead"},
            {"id": "P_new", "name": "pk-make"}, {"id": "P_orphan", "name": "pk-x"}, {"id": "P_hand", "name": "by-hand"}]
    vols = [{"id": "V_live"}, {"id": "V_dead"}, {"id": "V_home"}, {"id": "V_unknown"}]
    f = FakeAPI(pods=pods, vols=vols)
    f.install(monkeypatch)
    try:
        assert sweep.main(["--dry-run"]) == 0
        assert not f.terminated and not f.deleted
        assert sweep.main([]) == 0
    finally:
        p1.kill()
        p2.kill()
    assert sorted(f.terminated) == ["P_dead", "P_orphan"]
    assert f.deleted == ["V_home"]  # V_dead: not home (resume pending); V_live: alive; V_unknown: nobody's
    st = json.loads((Path(dead) / "state.json").read_text())
    assert st["runpod"]["pod"] is None and st["runpod"]["ledger"]["pods"][0]["stopped_by"] == "sweep"
    assert json.loads((Path(home) / "state.json").read_text())["runpod"]["ledger"]["volumes"][0]["deleted"]


def test_transport_stop_on_signal(tmp_path, monkeypatch):
    f = FakeAPI(pods=[{"id": "P1", "name": "n", "desiredStatus": "RUNNING"}])
    f.install(monkeypatch)
    monkeypatch.setattr(rt, "REGISTRY", str(tmp_path / "runners.json"))

    class St:
        d = {"endpoint": "P1", "runpod": {"pod": {"id": "P1", "ip": "1.2.3.4", "port": 22},
                                          "ledger": {"pods": [{"id": "P1", "start": time.time() - 60,
                                                               "stop": None, "usd_per_hr": 0.6}], "volumes": []}}}

        def save(self):
            pass

    st = St()
    t = rt.RunPodTransport(gpu="L4", cloud="SECURE", volume_gb=10, datacenter=None, image=None, max_usd=0.005,
                           commit="c", name="n", home=str(tmp_path), log=lambda m: None)
    old = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)}
    try:
        t.attach(st)
        with pytest.raises(SystemExit):
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(1)
        assert t.over_budget(st)  # 60 s at $0.6/h = $0.01 > max_usd
        t._atexit()
    finally:
        for s, h in old.items():
            signal.signal(s, h)
    assert f.terminated == ["P1"] and st.d["runpod"]["pod"] is None and st.d["endpoint"] is None
    assert st.d["runpod"]["ledger"]["pods"][0]["stop"]
    assert json.loads((tmp_path / "runners.json").read_text())[str(tmp_path)]["name"] == "n"


TRAINER = """#!{py}
import json, os, sys
spec = json.load(open(sys.argv[2]))
os.makedirs(spec["out_dir"], exist_ok=True)
json.dump({{"final_step": 1}}, open(os.path.join(spec["out_dir"], "result.json"), "w"))
"""


def test_provenance_stamped(tmp_path):
    chat = tmp_path / "chat"
    chat.mkdir()
    trainer = tmp_path / "trainer"
    trainer.write_text(TRAINER.format(py=sys.executable))
    trainer.chmod(0o755)
    env = dict(os.environ, JOBS_CHAT_DIR=str(chat), JOBS_MTSTUDIO=str(trainer), JOBS_REPO=str(tmp_path), JOBS_TICK="0.2")
    (chat / "provider.json").write_text(json.dumps({"provider": "runpod", "image": "img", "image_digest": "sha256:x"}))
    p = subprocess.run([sys.executable, str(CC / "provenance.py"), str(chat / "provenance.json")], env=env,
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    prov = json.loads((chat / "provenance.json").read_text())
    assert prov["provider"] == "runpod" and prov["image_digest"] == "sha256:x" and "driver_version" in prov
    jobs = tmp_path / "jobs.json"
    jobs.write_text(json.dumps({"jobs": [{"name": "pt", "spec": {"train": {}}}]}))
    p = subprocess.run([sys.executable, str(CC / "jobs_vm.py"), str(jobs)], env=env, capture_output=True, text=True,
                       timeout=60)
    assert p.returncode == 0, p.stdout + p.stderr
    assert json.loads((chat / "pt" / "result.json").read_text())["provenance"]["provider"] == "runpod"
    sys.path.insert(0, str(CC))
    import tiny_chat_run as tcr
    os.environ["JOBS_CHAT_DIR"] = str(chat)
    try:
        assert tcr.provenance()["image"] == "img"
    finally:
        del os.environ["JOBS_CHAT_DIR"]
