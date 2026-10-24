#!/usr/bin/env python3
"""runner_multi.py's RunPod backend (--backend runpod; 24 Oct 2026).

The six operations of runner_multi's Transport on a RunPod pod: create (REST POST /pods behind
runpod_api's guardrails, then the boot prelude runpod_prelude.sh over ssh), listed/alive (GET
/pods/{id}), exec (`ssh pod python3` with the code on stdin), put/get (rsync --partial-dir over the
same ssh; one whole file, no 45 MB parts), stop (DELETE /pods/{id}). State lives in the runner's
state.json under "runpod": the live pod, the campaign's network volume, and a spend ledger (every
pod: id, GPU, $/h, data centre, start, stop; every volume: id, GB, data centre, created, deleted).

The volume (SECURE cloud) is created once per campaign in a data centre with the GPU in stock and
mounted at /content, so /content/chat and /content/pk are Colab's paths and outlive any pod: a
relaunch in the same data centre needs no re-upload, the build is reused (same commit), and jobs
resume from their own checkpoints on the volume (runner_multi.volume_has). When that data centre
has no stock (or two creates there failed) the campaign moves: a new volume elsewhere, a full
prepare(), the old volume deleted once the new pod is up. The volume is deleted when the campaign
is home (release); runpod_sweep.py deletes it too if the runner could not.

The pod is terminated (never just stopped: its container disk would still bill) whenever the runner
stops it: queue empty, deadline, a fatal error, SIGTERM/SIGINT/SIGHUP, or interpreter exit (atexit).
A kill -9 leaves it to runpod_sweep.py, which terminates any pod whose runner is not alive.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import time
import urllib.request

import runpod_api as api

IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"  # official RunPod, CUDA 12.4 devel (nvcc)
# No allowedCudaVersions filter on the create: with it (12.4 to 13.0) RunPod found no L4 host at all (24 Oct
# 2026, EUR-IS-2 and EU-RO-1); the prelude records the host driver's CUDA and refuses one older than 12.0.
KEY = os.path.expanduser("~/.ssh/runpod_ed25519")
REGISTRY = os.path.expanduser("~/.runpod/runners.json")
PRELUDE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runpod_prelude.sh")
# An ssh session does not carry the image's environment (24 Oct 2026: no nvcc on PATH, so the VM-side cmake found
# no CUDA compiler; Colab's kernel has it). Every exec first takes PATH, the library paths and the CUDA/NVIDIA/RUNPOD
# variables from the container's own process 1, so the boot and the jobs it starts see what Colab's would.
ENV_PREAMBLE = (
    "import os as _pk_os\n"
    "try:\n"
    "    for _kv in open('/proc/1/environ', 'rb').read().split(bytes(1)):\n"
    "        _k, _s, _v = _kv.decode('utf-8', 'replace').partition('=')\n"
    "        if _s and (_k in ('PATH', 'LD_LIBRARY_PATH', 'LIBRARY_PATH') or _k.startswith(('CUDA', 'NV', 'RUNPOD_'))):\n"
    "            _pk_os.environ[_k] = _v\n"
    "except OSError:\n"
    "    pass\n"
    "if _pk_os.path.isdir('/usr/local/cuda/bin') and '/usr/local/cuda/bin' not in _pk_os.environ.get('PATH', ''):\n"
    "    _pk_os.environ['PATH'] = '/usr/local/cuda/bin:' + _pk_os.environ.get('PATH', '')\n")
UP_SECONDS = 1500  # create -> ssh answering (image pull included) before the pod is given up
DC_FAILS_MOVE = 2   # failed creates in the volume's data centre before the campaign moves
MIN_RAM, MIN_VCPU = 32, 6  # per GPU: the host floor (RunPod L4 hosts: 6 vCPU/62 GB in EU-RO-1, 12/100 in EUR-IS-2)


def file_sha(path, _cache={}):
    st = os.stat(path)
    k = (path, st.st_size, st.st_mtime_ns)
    if k not in _cache:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(1 << 22), b""):
                h.update(b)
        _cache[k] = h.hexdigest()
    return _cache[k]


def image_digest(image):
    """The tag's digest on Docker Hub, looked up when the pod is created ("repo:tag" only)."""
    m = re.fullmatch(r"([a-z0-9_.-]+)/([a-z0-9_.-]+):([A-Za-z0-9_.-]+)", image)
    if not m:
        return None
    url = f"https://hub.docker.com/v2/repositories/{m.group(1)}/{m.group(2)}/tags/{m.group(3)}"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return json.loads(r.read()).get("digest")
    except (OSError, ValueError):
        return None


def load_registry():
    try:
        return json.load(open(REGISTRY))
    except (OSError, ValueError):
        return {}


def save_json(path, obj):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


class RunPodTransport:
    backend = "runpod"
    whole_files = True

    def __init__(self, gpu, cloud, volume_gb, datacenter, image, max_usd, commit, name, home, log):
        self.gpu, self.cloud, self.volume_gb, self.dc = gpu, cloud, volume_gb, datacenter
        self.image, self.max_usd, self.commit = image or IMAGE, max_usd, commit
        self.name, self.home, self.log = name, home, log
        self.persistent = cloud == "SECURE"  # network volumes exist in Secure Cloud only
        self.st = None
        self._last = None      # the pod as the API last returned it; None = gone; "error" = API failed
        self._in_stop = False
        self._n = 0

    # ------------------------------------------------------------ state
    def R(self, st=None):
        st = st or self.st
        r = st.d.setdefault("runpod", {})
        r.setdefault("pod", None)
        r.setdefault("volume", None)
        r.setdefault("retired_volumes", [])
        r.setdefault("ledger", {"pods": [], "volumes": []})
        return r

    def attach(self, st):
        self.st = st
        self.R(st)
        st.save()
        os.makedirs(os.path.dirname(REGISTRY), exist_ok=True)
        reg = load_registry()
        reg[self.home] = {"name": self.name, "registered": time.time()}
        save_json(REGISTRY, reg)
        atexit.register(self._atexit)
        for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            signal.signal(s, self._signal)

    def _signal(self, signum, frame):
        if self._in_stop:  # never interrupt the terminate call itself
            return
        raise SystemExit(128 + signum)

    def _atexit(self):
        if self.st is not None and self.R().get("pod"):
            self.stop(self.st)

    def spent(self, st=None):
        return api.ledger_usd(self.R(st)["ledger"])

    def over_budget(self, st):
        if self.max_usd is not None and self.spent(st) >= self.max_usd:
            return f"RunPod spend ${self.spent(st):.2f} reached --max-usd {self.max_usd}; stopping"
        return None

    # ------------------------------------------------------------ ssh / rsync
    def _ssh_opts(self, pod):
        return ["-i", KEY, "-p", str(pod["port"]), "-o", "StrictHostKeyChecking=accept-new",
                "-o", f"UserKnownHostsFile={os.path.join(self.home, 'known_hosts_' + pod['id'])}",
                "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=4",
                "-o", "BatchMode=yes", "-o", "LogLevel=ERROR"]

    def _run(self, argv, timeout, stdin_text=None):
        try:
            p = subprocess.run(argv, input=stdin_text, capture_output=True, text=True, timeout=timeout,
                               stdin=None if stdin_text is not None else subprocess.DEVNULL, start_new_session=True)
            return p.returncode, (p.stdout or "") + (p.stderr or "")
        except subprocess.TimeoutExpired:
            return 124, "TIMEOUT"

    def exec(self, code, timeout=200):
        pod = self.R().get("pod")
        if not pod or not pod.get("ip"):
            return 255, "no pod"
        self._n += 1
        code = ENV_PREAMBLE + code
        f = f"/tmp/pkx/{os.getpid()}_{self._n}.py"
        limit = max(60, int(timeout) - 15)
        # The code goes to a file and python3 runs with stdin from /dev/null, so anything it starts in
        # the background (the boot) holds no handle on the ssh channel.
        remote = (f"mkdir -p /tmp/pkx && cat > {f} && {{ cd /content 2>/dev/null || cd /; }} && "
                  f"timeout {limit} python3 {f} < /dev/null; r=$?; rm -f {f}; exit $r")
        return self._run(["ssh", *self._ssh_opts(pod), f"root@{pod['ip']}", remote], timeout, stdin_text=code)

    def _rsync(self, src, dst, timeout, mkdir=None):
        pod = self.R().get("pod")
        if not pod or not pod.get("ip"):
            return 255, "no pod"
        e = " ".join(["ssh", *map(shlex.quote, self._ssh_opts(pod))])
        argv = ["rsync", "--partial-dir=.rsync-partial", "--timeout=180", "-e", e]
        if mkdir:
            argv += ["--rsync-path", f"mkdir -p {shlex.quote(mkdir)} && rsync"]
        return self._run([*argv, src, dst], timeout)

    def put(self, src, dst, timeout):
        pod = self.R().get("pod") or {}
        return self._rsync(src, f"root@{pod.get('ip')}:{dst}", timeout, mkdir=os.path.dirname(dst))

    def get(self, src, dst, timeout):
        pod = self.R().get("pod") or {}
        return self._rsync(f"root@{pod.get('ip')}:{src}", dst, timeout)

    def _remote_shas(self, paths):
        rc, out = self.exec("import hashlib, json, os\n"
                            f"r = {{}}\nfor p in {list(paths)!r}:\n"
                            "    if os.path.isfile(p):\n"
                            "        h = hashlib.sha256()\n"
                            "        with open(p, 'rb') as f:\n"
                            "            for b in iter(lambda: f.read(1 << 22), b''):\n"
                            "                h.update(b)\n"
                            "        r[p] = [os.path.getsize(p), h.hexdigest()]\n"
                            "print('RSHA', json.dumps(r))\n", timeout=600)
        m = re.search(r"^RSHA (.*)$", out, re.M)
        return json.loads(m.group(1)) if m else None

    def upload_whole(self, src, dst):
        """src to dst.part00 (runner_multi and jobs_vm.py reassemble parts), skipped when dst or dst.part00
        already holds the same bytes (a relaunch on the campaign's volume). RuntimeError on failure."""
        want = file_sha(src)
        have = self._remote_shas([dst, dst + ".part00"])
        if have is None:
            raise RuntimeError(f"upload of {src}: the pod did not answer")
        for p in (dst, dst + ".part00"):
            if have.get(p, [None, None])[1] == want:
                self.log(f"{os.path.basename(src)}: already on the volume ({p}); not uploaded")
                return
        for _ in range(3):
            rc, out = self.put(src, dst + ".part00", 7200)
            if rc == 0:
                return
            self.log(f"upload of {src} failed (rc {rc}: {out.strip()[-120:]}); retrying")
        raise RuntimeError(f"upload of {src} failed")

    def download_whole(self, src, dst, timeout):
        have = self._remote_shas([src])
        if not have or src not in have:
            return 1
        size, sha = have[src]
        rc, _ = self.get(src, dst + ".dltmp", timeout)
        if rc != 0 or not os.path.exists(dst + ".dltmp"):
            return 1
        if os.path.getsize(dst + ".dltmp") != size or file_sha(dst + ".dltmp") != sha:
            os.remove(dst + ".dltmp")
            self.log(f"download of {src}: sha256 mismatch; discarded")
            return 1
        os.replace(dst + ".dltmp", dst)
        return 0

    # ------------------------------------------------------------ the pod
    def _fetch(self):
        pod = self.R().get("pod")
        if not pod:
            self._last = None
            return None
        try:
            p = api.get_pod(pod["id"])
        except api.RunPodError as e:
            self.log(f"RunPod API: {e}")
            self._last = "error"
            return "error"
        self._last = p
        if p:
            ep = api.ssh_endpoint(p)
            if ep and (pod.get("ip"), pod.get("port")) != ep:
                pod["ip"], pod["port"] = ep
                self.st.save()
        return p

    def refresh(self, st):
        self._fetch()

    def listed(self):
        p = self._fetch()
        if isinstance(p, dict) and p.get("desiredStatus") == "RUNNING" and api.ssh_endpoint(p):
            return p["id"]
        return None

    def adopt(self, ep):
        return False  # a pod is ours by its recorded id; listed() already looked for it

    def gone(self, st):
        """The API says the recorded pod no longer runs (or none is recorded): proof, unlike Colab's
        listing, so the relaunch need not wait three ticks."""
        p = self._last
        return p is None or (isinstance(p, dict) and p.get("desiredStatus") in ("EXITED", "TERMINATED"))

    def stop(self, st):
        """Terminate the recorded pod and close its ledger entry. Idempotent."""
        R = self.R(st)
        pod = R.get("pod")
        if not pod:
            return
        self._in_stop = True
        try:
            for i in range(3):
                try:
                    if api.terminate_pod(pod["id"]):
                        break
                except api.RunPodError as e:
                    self.log(f"terminate {pod['id']}: {e}")
                time.sleep(5 * (i + 1))
            else:
                self.log(f"pod {pod['id']} NOT terminated; runpod_sweep.py will retry")
                return
            now = time.time()
            for e in R["ledger"]["pods"]:
                if e["id"] == pod["id"] and not e.get("stop"):
                    e["stop"], e["stopped_by"] = now, "runner"
            R["pod"] = None
            st.d["endpoint"] = None
            st.save()
            self.log(f"pod {pod['id']} terminated (RunPod ledger ${self.spent(st):.2f})")
        finally:
            self._in_stop = False

    def release(self, st):
        """The campaign is home: delete its volume(s)."""
        R = self.R(st)
        vols = [v for v in [R.get("volume")] + R["retired_volumes"] if v]
        for v in vols:
            self._delete_volume(st, v)
        if R.get("volume") and R["volume"].get("deleted"):
            R["volume"] = None
        R["retired_volumes"] = [v for v in R["retired_volumes"] if not v.get("deleted")]
        st.save()

    def _delete_volume(self, st, v):
        try:
            if api.delete_volume(v["id"]):
                v["deleted"] = time.time()
                for e in self.R(st)["ledger"]["volumes"]:
                    if e["id"] == v["id"]:
                        e["deleted"] = v["deleted"]
                st.save()
                self.log(f"volume {v['id']} ({v['dc']}) deleted")
                return True
        except api.RunPodError as e:
            self.log(f"delete volume {v['id']}: {e}")
        return False

    def _stocked(self):
        return api.stocked_dcs(self.gpu, MIN_RAM, MIN_VCPU, secure=self.cloud == "SECURE")

    def _retire(self, st, vol):
        """The campaign leaves this volume: deleted now when no pod ever ran on it (nothing to lose), else
        kept until the next pod is up (create) or the campaign is home (release)."""
        R = self.R(st)
        if not any(p.get("volume") == vol["id"] for p in R["ledger"]["pods"]) and self._delete_volume(st, vol):
            return
        R["retired_volumes"].append(vol)
        st.save()

    def _volume_for_create(self, st):
        """(data centre, volume or None) for the next pod; creates the volume when needed."""
        R = self.R(st)
        vol = R.get("volume")
        if not self.persistent:
            stocked = [self.dc] if self.dc else self._stocked()
            if not stocked:
                raise api.RunPodError(f"no {self.gpu} in stock in any data centre")
            return stocked[0], None
        if self.dc:
            if vol and vol["dc"] != self.dc:
                self._retire(st, vol)
                vol = R["volume"] = None
            dc = self.dc
        else:
            stocked = self._stocked()
            if vol and vol["dc"] in stocked and R.get("dc_fails", 0) < DC_FAILS_MOVE:
                return vol["dc"], vol
            if not stocked:
                raise api.RunPodError(f"no {self.gpu} in stock in any data centre")
            if vol:
                others = [d for d in stocked if d != vol["dc"]]
                if not others and vol["dc"] in stocked:
                    return vol["dc"], vol
                self.log(f"{vol['dc']}: no {self.gpu} (stock {stocked}, {R.get('dc_fails', 0)} failed creates); "
                         "moving the campaign to a new volume (full prepare)")
                self._retire(st, vol)
                vol = R["volume"] = None
                stocked = others
            dc = stocked[0]
        if vol:
            return dc, vol
        v = api.create_volume(f"{self.name}-vol", self.volume_gb, dc)
        vol = {"id": v["id"], "dc": dc, "gb": self.volume_gb, "created": time.time()}
        R["volume"] = vol
        R["dc_fails"] = 0
        R["ledger"]["volumes"].append(dict(vol, name=v.get("name")))
        st.save()
        self.log(f"volume {vol['id']} created in {dc} ({self.volume_gb} GB)")
        return dc, vol

    def create(self, st):
        """(0, pod id) once the pod answers over ssh and the prelude has run; (1, why) otherwise
        (the runner logs it and retries next tick). A pod that fails to come up is terminated."""
        R = self.R(st)
        why = self.over_budget(st)
        if why:
            return 1, f"refused: {why}"
        try:
            dc, vol = self._volume_for_create(st)
            pub = open(KEY + ".pub").read().strip()
            body = {"name": self.name, "imageName": self.image, "computeType": "GPU", "cloudType": self.cloud,
                    "gpuTypeIds": [api.gpu_id(self.gpu)], "gpuCount": 1, "dataCenterIds": [dc],
                    "containerDiskInGb": 30 if vol else self.volume_gb,
                    "volumeInGb": 0, "ports": ["22/tcp"], "env": {"PUBLIC_KEY": pub}, "interruptible": False,
                    "minRAMPerGPU": MIN_RAM, "minVCPUPerGPU": MIN_VCPU}
            if vol:
                body.update(networkVolumeId=vol["id"], volumeMountPath="/content")
            else:
                body["supportPublicIp"] = True
            digest = image_digest(self.image)
            R["creating"] = time.time()
            st.save()
            try:
                pod = api.create_pod(body)
            except api.Refused:
                raise
            except api.RunPodError:
                if vol:
                    R["dc_fails"] = R.get("dc_fails", 0) + 1
                raise
            finally:
                R["creating"] = None
            rate = pod.get("costPerHr") or pod.get("adjustedCostPerHr") or api.gpu_price(self.gpu, self.cloud) or 0.0
            R["pod"] = {"id": pod["id"], "ip": None, "port": None, "dc": dc, "created": time.time()}
            R["ledger"]["pods"].append({"id": pod["id"], "name": self.name, "gpu": self.gpu, "usd_per_hr": float(rate),
                                        "dc": dc, "volume": vol["id"] if vol else None, "start": time.time(),
                                        "stop": None, "image": self.image, "image_digest": digest})
            R["dc_fails"] = 0
            st.d["endpoint"] = pod["id"]
            st.save()
            self.log(f"pod {pod['id']} created in {dc} ({self.gpu}, ${float(rate):.3f}/h, {self.cloud}); "
                     f"waiting for ssh")
            info = self._wait_up(pod["id"])
            if not info:
                self.stop(st)
                return 1, "pod did not come up (ssh) in time; terminated"
            ok, tail = self._prelude(info, digest, dc, rate)
            if not ok:
                self.log(f"prelude failed: {tail}")
                self.stop(st)
                return 1, "prelude failed; pod terminated"
            for v in list(R["retired_volumes"]):
                self._delete_volume(st, v)
            R["retired_volumes"] = [v for v in R["retired_volumes"] if not v.get("deleted")]
            st.save()
            return 0, pod["id"]
        except api.Refused as e:
            st.save()
            return 1, f"refused: {e}"
        except (api.RunPodError, OSError) as e:
            st.save()
            return 1, f"{type(e).__name__}: {e}"

    def _wait_up(self, pod_id):
        t0 = time.time()
        while time.time() - t0 < UP_SECONDS:
            p = self._fetch()
            if p is None:
                self.log(f"pod {pod_id} vanished while starting")
                return None
            if isinstance(p, dict) and p.get("desiredStatus") == "RUNNING" and api.ssh_endpoint(p):
                rc, out = self.exec("print('SSH_OK')\n", timeout=60)
                if "SSH_OK" in out:
                    self.log(f"pod {pod_id} answers at {self.R()['pod']['ip']}:{self.R()['pod']['port']} "
                             f"after {time.time() - t0:.0f} s")
                    return p
            time.sleep(15)
        self.log(f"pod {pod_id}: no ssh after {UP_SECONDS} s")
        return None

    def _prelude(self, pod, digest, dc, rate):
        provider = {"provider": "runpod", "pod_id": pod["id"], "image": self.image, "image_digest": digest,
                    "image_digest_source": "Docker Hub tag lookup when the pod was created",
                    "datacenter": dc, "cloud": self.cloud, "gpu_type_id": api.gpu_id(self.gpu),
                    "usd_per_hr": float(rate), "volume": (self.R().get("volume") or {}).get("id"),
                    "machine": {k: (pod.get("machine") or {}).get(k) for k in
                                ("gpuTypeId", "location", "dataCenterId", "secureCloud", "diskThroughputMBps",
                                 "maxDownloadSpeedMbps", "maxUploadSpeedMbps")},
                    "memory_gb": pod.get("memoryInGb"), "vcpu": pod.get("vcpuCount")}
        script = open(PRELUDE).read()
        rc, out = self.exec(
            "import json, os, subprocess\n"
            f"open('/tmp/pk_prelude.sh', 'w').write({script!r})\n"
            f"p = subprocess.run(['bash', '/tmp/pk_prelude.sh', {self.commit!r}], capture_output=True, text=True)\n"
            "os.makedirs('/content/chat', exist_ok=True)\n"
            "open('/content/chat/prelude.log', 'a').write(p.stdout + p.stderr)\n"
            f"json.dump({provider!r}, open('/content/chat/provider.json.tmp', 'w'), indent=1)\n"
            "os.replace('/content/chat/provider.json.tmp', '/content/chat/provider.json')\n"
            "print('PRELUDE_RC', p.returncode)\n"
            "print((p.stdout + p.stderr)[-1500:])\n", timeout=1800)
        m = re.search(r"^PRELUDE_RC (-?\d+)$", out, re.M)
        ok = bool(m) and m.group(1) == "0"
        tail = " | ".join(ln for ln in out.strip().splitlines()[-12:] if ln.strip())
        self.log(f"prelude {'done' if ok else 'FAILED'}: {tail[-600:]}")
        return ok, tail
