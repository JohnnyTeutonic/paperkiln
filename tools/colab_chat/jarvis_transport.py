#!/usr/bin/env python3
"""runner_multi.py's Jarvis Labs backend (--backend jarvis; 7 Nov 2026).

RunPodTransport with the provider swapped: ssh, rsync, the whole-file transfers, the deadline and spend
limits on every operation, signals and the ledger's shape come from runpod_transport.py. What differs:

- A VM, not a pod: user ubuntu, so exec and rsync run as that user. /content is a link the prelude makes
  to a folder on the campaign's filesystem (/home/jl_fs/NAME), so paths match Colab's and outlive any VM.
- State lives in state.json under "jarvis"; the registry is ~/.jarvis/runners.json (jarvis_sweep.py).
- Jarvis reuses public IPs with new host keys, so each VM has its own known-hosts file in the state dir
  (the inherited accept-new records the key on first contact and refuses a later change).

No VM goes unrecorded. `jl create` blocks and can fail after its POST, so every create has a unique name
(NAME-SEQ), written to state.json under "creating" before jl runs. _settle_creating() then looks the name
up: a VM found is entered in the ledger from the moment of the attempt and destroyed. It runs after any
failed create, before the next one, and in stop(), so a create whose fate is unknown blocks further
creates. A runner killed outright leaves "creating" for jarvis_sweep.py, which does the same.

A VM the API lists as Running but ssh cannot reach for SSH_LOST_S is destroyed and replaced from the
filesystem: on RunPod that case idles until the deadline. Two VMs in a row that never reach the boot stop
the renting (BOOT_FAILS_STOP), which a watchdog-restarted runner would otherwise repeat until --max-usd.
"""
from __future__ import annotations

import os
import re
import time

import jarvis_api as api
import runpod_transport as rt

KEY = os.path.expanduser("~/.ssh/jarvis_ed25519")
REGISTRY = os.path.join(api.HOME, "runners.json")
PRELUDE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jarvis_prelude.sh")
UP_SECONDS = 300   # create returned Running -> ssh answering (30 s when measured, 7 Nov 2026)
UP_POLL_S = 10
DESTROY_BACKOFF_S = 5
SSH_LOST_S = 900   # continuous ssh failure before a Running VM is given up (AGENTS.md section 5: 15 minutes)
BOOT_FAILS_STOP = 2  # paid VMs in a row that never reached the boot: the third needs a person, not a retry
RUNNER_NAME_RE = re.compile(r"[A-Za-z0-9_-]{1,27}\Z")  # fits "-fs" in Jarvis's 30 characters, so a name is never cut


class JarvisTransport(rt.RunPodTransport):
    backend = "jarvis"
    ssh_user = api.SSH_USER
    ssh_key = KEY
    state_key = "jarvis"

    def __init__(self, gpu, volume_gb, max_usd, commit, name, home, log, cloud="SECURE", datacenter=None, image=None):
        if gpu != api.GPU_TYPE:
            raise api.Refused(f"--backend jarvis runs {api.GPU_TYPE} only, not {gpu}")
        if cloud != "SECURE" or datacenter or image:
            raise api.Refused("--cloud, --datacenter and --image are RunPod options; refused for --backend jarvis")
        if max_usd is None or not 0 < max_usd <= api.SPEND_CEILING:
            raise api.Refused(f"--backend jarvis needs --max-usd, up to {api.SPEND_CEILING:.0f}")
        if not RUNNER_NAME_RE.match(name):
            raise api.Refused(f"--name {name!r}: 1 to 27 letters, digits, hyphens or underscores for Jarvis")
        cap = os.environ.get("JARVIS_ACCOUNT_CAP")
        try:   # account-wide spend since opening, every charge counted (Codex's review, 7 Nov): JARVIS_ACCOUNT_CAP, at most api.ACCOUNT_CAP
            self.account_cap = api.checked_account_cap(float(cap) if cap is not None else api.ACCOUNT_CAP)
        except ValueError:
            raise api.Refused(f"JARVIS_ACCOUNT_CAP {cap!r} is not a number") from None
        super().__init__(gpu, "SECURE", volume_gb, None, "jarvis-vm-template", max_usd, commit, name, home, log)
        self._balance = None         # last balance read; refresh() keeps it, over_budget() must not call jl
        self._ssh_down_since = None  # first of an unbroken run of ssh failures

    # ------------------------------------------------------------ state
    def R(self, st=None):
        r = super().R(st)
        r.setdefault("vm_seq", 0)
        r.setdefault("creating", None)
        r.setdefault("balance_start", None)
        r.setdefault("boot_fails", 0)
        return r

    def _register(self):
        os.makedirs(os.path.dirname(REGISTRY), exist_ok=True)
        with api._lock("registry"):
            reg = rt.load_registry(REGISTRY)
            reg[self.home] = {"name": self.name, "registered": time.time()}
            rt.save_json(REGISTRY, reg)

    def _atexit(self):
        if self.st is not None and (self.R().get("pod") or self.R().get("creating")):
            self.stop(self.st)

    def spent(self, st=None):
        """The ledger's estimate, or what the balance has actually fallen by if that is more (VM disk
        billing is not in the ledger's rate). Other campaigns' spend counts too: that errs towards stopping."""
        R = self.R(st)
        estimate = api.ledger_usd(R["ledger"])
        if R.get("balance_start") is not None and self._balance is not None:
            return max(estimate, R["balance_start"] - self._balance)
        return estimate

    def over_budget(self, st):
        if self.spent(st) >= self.max_usd:
            return f"Jarvis spend ${self.spent(st):.2f} reached --max-usd {self.max_usd}; stopping"
        if self._balance is not None and self._balance < api.MIN_BALANCE_RUNNING:
            return f"Jarvis balance ${self._balance:.2f} is below ${api.MIN_BALANCE_RUNNING:.0f}; stopping"
        if self._balance is not None and api.account_spent(self._balance) >= self.account_cap:
            return f"Jarvis account spend ${api.account_spent(self._balance):.2f} reached its cap ${self.account_cap:.2f}; stopping"
        return None

    # ------------------------------------------------------------ ssh
    def exec(self, code, timeout=200):
        rc, out = super().exec(code, timeout)
        if self.R().get("pod"):
            if rc == 255:  # ssh's own failure: python's and timeout's exits are never 255 here
                self._ssh_down_since = self._ssh_down_since or time.time()
            else:
                self._ssh_down_since = None
        return rc, out

    # ------------------------------------------------------------ the VM
    def _fetch(self):
        pod = self.R().get("pod")
        if not pod:
            self._last = None
            return None
        try:
            v = api.get_vm(pod["id"])
        except api.JarvisError as e:
            self.log(f"Jarvis API: {e}")
            self._last = "error"
            return "error"
        self._last = v
        if v:
            ep = api.ssh_endpoint(v)
            if ep and (pod.get("ip"), pod.get("port")) != ep:
                pod["ip"], pod["port"] = ep
                self.st.save()
        return v

    def refresh(self, st):
        self._fetch()
        try:
            self._balance = api.balance()
        except api.JarvisError as e:
            self.log(f"Jarvis API: {e}")

    def listed(self):
        v = self._fetch()
        if self.R().get("pod") and self._ssh_down_since and time.time() - self._ssh_down_since >= SSH_LOST_S:
            self.log(f"VM {self.R()['pod']['id']}: no ssh for {SSH_LOST_S} s while listed; destroying it")
            self.stop(self.st)
            return None
        if isinstance(v, dict) and api.ssh_endpoint(v):
            return str(v["machine_id"])
        return None

    def gone(self, st):
        v = self._last
        return v is None or (isinstance(v, dict) and v.get("status") in api.DEAD_STATUSES)

    def _record_vm(self, st, vm, start, rate):
        """Into state and the ledger before anything else touches the VM."""
        R = self.R(st)
        vm_id = str(vm["machine_id"])
        ep = api.ssh_endpoint(vm) or (None, None)
        R["pod"] = {"id": vm_id, "ip": ep[0], "port": ep[1], "dc": api.REGION, "created": time.time()}
        R["ledger"]["pods"].append({"id": vm_id, "name": vm.get("name"), "gpu": self.gpu, "usd_per_hr": float(rate),
                                    "dc": api.REGION, "volume": (R.get("volume") or {}).get("id"), "start": start,
                                    "stop": None, "image": self.image, "image_digest": None})
        st.d["endpoint"] = vm_id
        st.save()

    def _settle_creating(self, st):
        """Decide the fate of a create that did not return a VM. Raises (leaving "creating" set, so no
        new create starts) when the listing fails."""
        R = self.R(st)
        c = R.get("creating")
        if not c:
            return
        for vm in api.find_by_name(c["name"]):
            self.log(f"VM {vm.get('machine_id')} ({c['name']}) exists after a failed create; destroying it")
            self._record_vm(st, vm, c["started"], api.MAX_USD_PER_HR)  # the price was not seen: the dearer side
            R["creating"] = None
            self.stop(st)
        R["creating"] = None
        st.save()

    def stop(self, st):
        """Destroy the recorded VM (never pause: its disk would still bill) and close its ledger entry.
        Idempotent. A VM that will not die stays recorded, for the next call or the sweep."""
        R = self.R(st)
        if R.get("creating"):
            try:
                self._settle_creating(st)
            except api.JarvisError as e:
                self.log(f"could not settle create {R['creating']['name']}: {e}; jarvis_sweep.py will")
        pod = R.get("pod")
        if not pod:
            return
        self._in_stop = True
        try:
            for i in range(3):
                try:
                    if api.destroy_vm(pod["id"]):
                        break
                except api.JarvisError as e:
                    self.log(f"destroy {pod['id']}: {e}")
                time.sleep(DESTROY_BACKOFF_S * (i + 1))
            else:
                self.log(f"VM {pod['id']} NOT destroyed; jarvis_sweep.py will retry")
                return
            now = time.time()
            for e in R["ledger"]["pods"]:
                if e["id"] == pod["id"] and not e.get("stop"):
                    e["stop"], e["stopped_by"] = now, "runner"
            R["pod"] = None
            st.d["endpoint"] = None
            st.save()
            self._last, self._ssh_down_since = None, None
            self.log(f"VM {pod['id']} destroyed (Jarvis ledger ${self.spent(st):.2f})")
        finally:
            self._in_stop = False

    def _delete_volume(self, st, v):
        try:
            if api.filesystem_remove(v["id"]):
                v["deleted"] = time.time()
                for e in self.R(st)["ledger"]["volumes"]:
                    if e["id"] == v["id"]:
                        e["deleted"] = v["deleted"]
                st.save()
                self.log(f"filesystem {v['id']} removed")
                return True
        except api.JarvisError as e:
            self.log(f"remove filesystem {v['id']}: {e}")
        return False

    def _filesystem(self, st):
        """The campaign's filesystem: the recorded one, one of its name left by a crash, or a new one. A
        recorded filesystem the API no longer lists is a stop for a person: its checkpoints are gone."""
        R = self.R(st)
        rows = api.list_filesystems()
        fs = R.get("volume")
        if fs:
            if not any(str(f.get("fs_id")) == str(fs["id"]) for f in rows):
                raise api.Refused(f"filesystem {fs['id']} is no longer listed; refusing to start without it")
            return fs
        name = f"{self.name}-fs"
        hit = next((f for f in rows if f.get("fs_name") == name), None)
        if hit and hit.get("region") not in api.REGION_IDS:
            raise api.Refused(f"filesystem {hit['fs_id']} is in {hit.get('region')}, not {api.REGION}")
        fs_id = int(hit["fs_id"]) if hit else api.filesystem_create(name, self.volume_gb)
        fs = {"id": fs_id, "dc": api.REGION, "gb": self.volume_gb, "created": time.time()}
        R["volume"] = fs
        R["ledger"]["volumes"].append(dict(fs, name=name))
        st.save()
        self.log(f"filesystem {fs_id} {'found' if hit else 'created'} in {api.REGION} ({self.volume_gb} GB)")
        return fs

    def _create_timeout(self):
        if self.deadline is None:
            return api.CREATE_TIMEOUT
        return max(30, min(api.CREATE_TIMEOUT, self.deadline - time.time()))

    def create(self, st):
        """(0, VM id) once the VM answers over ssh and the prelude has run; (1, why) otherwise, with
        nothing left running: the runner logs it and retries next tick."""
        R = self.R(st)
        why = self._operation_limit() or self.over_budget(st)
        if why:
            return 1, f"refused: {why}"
        try:
            self._settle_creating(st)
            if R.get("pod"):
                return 1, f"VM {R['pod']['id']} is not destroyed yet; no second VM"
            if R["boot_fails"] >= BOOT_FAILS_STOP:  # a watchdog-restarted runner must not rent VMs in a loop
                return 1, (f"refused: {R['boot_fails']} VMs in a row failed before the boot (runner.log); "
                           "set jarvis.boot_fails to 0 in state.json once the cause is fixed")
            fs = self._filesystem(st)
            if R.get("balance_start") is None:
                self._balance = R["balance_start"] = api.balance()
            why = self._operation_limit() or self.over_budget(st)
            if why:
                return 1, f"refused: {why}"
            R["vm_seq"] += 1
            name = f"{self.name}-{R['vm_seq']}"
            R["creating"] = {"name": name, "started": time.time()}
            st.save()
            start = time.time()
            try:
                vm = api.create_vm(name, fs["id"], self.spent(st), self.max_usd, timeout=self._create_timeout(), account_cap=self.account_cap)
            except api.Refused:
                R["creating"] = None  # nothing was sent to Jarvis
                st.save()
                raise
            except api.JarvisError as e:
                self.log(f"create {name}: {e}")
                self._settle_creating(st)
                raise
            self._record_vm(st, vm, start, vm["usd_per_hr"])
            R["creating"] = None
            st.save()
            self.log(f"VM {vm['machine_id']} created ({name}, {api.GPU_TYPE}, ${vm['usd_per_hr']:.3f}/h, "
                     f"filesystem {fs['id']}); waiting for ssh")
            try:
                faults = api.vm_faults(vm)
                info = None if faults else self._wait_up(str(vm["machine_id"]))
                ok = bool(info) and self._prelude(info, vm["usd_per_hr"])
            except rt.api.Refused:  # the deadline or the cap fell during the wait: no VM outlives that
                self.stop(st)
                raise
            if not ok:
                R["boot_fails"] += 1
                self.stop(st)
                why = f"VM refused ({', '.join(faults)})" if faults else (
                    "prelude failed" if info else "VM did not come up (ssh) in time")
                return 1, f"{why}; destroyed"
            R["boot_fails"] = 0
            st.save()
            return 0, str(vm["machine_id"])
        except rt.api.Refused as e:
            st.save()
            return 1, f"refused: {e}"
        except (rt.api.RunPodError, OSError) as e:
            st.save()
            return 1, f"{type(e).__name__}: {e}"

    def _wait_up(self, vm_id):
        t0 = time.time()
        while time.time() - t0 < UP_SECONDS:
            why = self._operation_limit()
            if why:
                raise api.Refused(why)
            v = self._fetch()
            if v is None:
                self.log(f"VM {vm_id} vanished while starting")
                return None
            if isinstance(v, dict) and api.ssh_endpoint(v):
                rc, out = self.exec("print('SSH_OK')\n", timeout=60)
                if "SSH_OK" in out:
                    self._ssh_down_since = None
                    pod = self.R()["pod"]
                    self.log(f"VM {vm_id} answers at {pod['ip']}:{pod['port']} after {time.time() - t0:.0f} s")
                    return v
            time.sleep(UP_POLL_S)
        self.log(f"VM {vm_id}: no ssh after {UP_SECONDS} s")
        return None

    def _prelude(self, vm, rate):
        provider = {"provider": "jarvis", "machine_id": str(vm.get("machine_id")), "image": self.image,
                    "image_digest": None, "datacenter": api.REGION, "cloud": "on-demand",
                    "gpu_type_id": api.GPU_TYPE, "usd_per_hr": float(rate),
                    "volume": (self.R().get("volume") or {}).get("id"), "memory_gb": vm.get("ram"),
                    "vcpu": vm.get("cores"), "is_spot": bool(vm.get("is_spot"))}
        script = open(PRELUDE).read()
        # The script makes /content, so nothing here may touch it first. Its output is printed before
        # anything can fail, and a failed prelude writes no provider.json.
        rc, out = self.exec(
            "import json, os, subprocess, sys\n"
            f"open('/tmp/pk_prelude.sh', 'w').write({script!r})\n"
            f"p = subprocess.run(['bash', '/tmp/pk_prelude.sh', {self.commit!r}, {self.name!r}],"
            " capture_output=True, text=True)\n"
            "print('PRELUDE_RC', p.returncode)\n"
            "print((p.stdout + p.stderr)[-1500:])\n"
            "sys.stdout.flush()\n"
            "if p.returncode == 0:\n"
            "    open('/content/chat/prelude.log', 'a').write(p.stdout + p.stderr)\n"
            f"    json.dump({provider!r}, open('/content/chat/provider.json.tmp', 'w'), indent=1)\n"
            "    os.replace('/content/chat/provider.json.tmp', '/content/chat/provider.json')\n"
            "    print('PROVIDER_WRITTEN')\n", timeout=1800)
        m = re.search(r"^PRELUDE_RC (-?\d+)$", out, re.M)
        ok = rc == 0 and bool(m) and m.group(1) == "0" and "PROVIDER_WRITTEN" in out
        tail = " | ".join(ln for ln in out.strip().splitlines()[-12:] if ln.strip())
        self.log(f"prelude {'done' if ok else 'FAILED'}: {tail[-600:]}")
        return ok
