#!/usr/bin/env python3
"""A small RunPod client for runner_multi.py --backend runpod and runpod_sweep.py (24 Oct 2026; REST v2 since 17 Nov 2026).

    python3 tools/colab_chat/runpod_api.py status      # balance, spend/h, pods, volumes, L4 stock

The API key is read from ~/.runpod/api_key (one line, mode 600) and sent only as an
`Authorization: Bearer` header from this process: never printed, logged or put on a command line.

REST v2: https://api.runpod.io/v2 (pods, network-volumes, catalog). RunPod retired REST v1
(https://rest.runpod.io/v1) on 15 Nov 2026; its host name stopped resolving on 16 Nov. The v2 shapes
here were read from docs.runpod.io's "Migrate from API v1" page and its v2 OpenAPI schema on 17 Nov
2026, and the read-only calls (list pods, list volumes, a pod and a volume that do not exist, the
catalog, cursor pagination) were checked against the live API that day. What changed against v1:
  lists          {"pods": [...], "pagination": {"nextCursor", "hasNextPage"}} and {"networkVolumes": [...]},
                 cursor-paginated (query `cursor`, `limit`); every page is followed
  a pod          `status` is the actual state (PROVISIONING, STARTING, RUNNING, EXITED, ERROR, TERMINATED),
                 `cost` the hourly rate, `ssh.direct` {host, port} the published 22/tcp (only on GET of one
                 pod, never in a list), `dataCenterId`, `gpu` {id, count, vcpuCount, memory}
  create         a nested body: image, disk, gpu {id, count, minRamPerGpu, minVcpuCountPerGpu}, cloud,
                 mounts {network: [{volumeId, path}]}; unknown fields are rejected with 422; no default ports
                 and no default volume; spot pods, country and bandwidth placement no longer exist
  terminate      DELETE /v2/pods/{id} -> 204
  volumes        /v2/network-volumes; create takes `dataCenter` (was dataCenterId) and a `type`, always given here: the first
                 of VOLUME_TYPES (STANDARD, then HIGH_PERFORMANCE) the data centre offers (21 Nov 2026); delete -> 204
  not used       queued creates (`queue`) are refused: nothing here would own or cancel one
  errors         {"title", "status", "detail"[, "errors"]}
GraphQL (https://api.runpod.io/graphql) is kept for what v2 does not offer: the account balance
(myself { clientBalance currentSpendPerHr }), which RunPod's migration page says to keep there, and the
per-host stock and price queries (gpuTypes), checked live on 17 Nov 2026.

Callers written against v1 keep working: every pod and volume this module returns also carries the v1
field names they read (desiredStatus, costPerHr, publicIp, portMappings, machine, memoryInGb, vcpuCount,
lastStartedAt, networkVolumeId, networkVolume; a volume's dataCenterId), filled from the v2 fields, and create_pod() accepts the flat v1
body and sends its v2 form. A v1 field with no v2 form that would change what is rented (a spot pod,
several GPU types, an unknown field) is refused, never dropped silently.

Guardrails (Jonathan's budget, 24 Oct 2026), enforced by create_pod() itself so no caller can skip
them: at most MAX_PODS pods on the account at once, counted through the API before every create, and
no create while clientBalance is below MIN_BALANCE dollars.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

REST = "https://api.runpod.io/v2"
GRAPHQL = "https://api.runpod.io/graphql"
CATALOG = "https://api.runpod.io/v2/catalog/datacenters"
KEY_PATH = os.path.expanduser(os.environ.get("RUNPOD_KEY_PATH", "~/.runpod/api_key"))
MAX_PODS = 10         # account-wide, every create: a runaway guard, not a budget (Jonathan sets no RunPod pod limit)
MIN_BALANCE = 15.0    # dollars; no create below it
MAX_PAGES = 50        # a list is followed to its last page; more pages than this is a fault, never a short list
GPU_IDS = {"L4": "NVIDIA L4", "T4": "Tesla T4", "A100": "NVIDIA A100 80GB PCIe", "4090": "NVIDIA GeForce RTX 4090",
           "A40": "NVIDIA A40", "L40S": "NVIDIA L40S", "A5000": "NVIDIA RTX A5000"}  # the last three: A31 (13 Nov 2026)


class RunPodError(RuntimeError):
    pass


class Refused(RunPodError):
    """A guardrail refused the action (pod cap, low balance): never retried blindly."""


class Rejected(RunPodError):
    """RunPod refused the request itself: a pod create that failed validation (422), or a volume create it would not process
    (400 or 422). A retry sends the same request (17 Nov 2026: a STANDARD volume asked for in a data centre that has only
    high-performance storage was retried every four minutes and logged as "no A40"). A pod create's 400 is not raised as this:
    the schema calls it "malformed or conflicts with request rules" and does not say what a GPU without capacity returns, so it
    stays an ordinary failure, with RunPod's text."""


def problem_text(d, limit=1500):
    """A v2 error's title, detail and validation errors ({"title", "status", "detail"[, "errors"]}, which _request hands over as
    {"error": the first 2,000 bytes}), up to `limit` characters from its start, so a log line holds what RunPod said and not its
    last 120 characters."""
    raw = d.get("error") if isinstance(d, dict) and isinstance(d.get("error"), str) else json.dumps(d)
    try:
        p = json.loads(raw)
        if isinstance(p, dict) and (p.get("detail") or p.get("title")):
            raw = f"{p.get('title')}: {p.get('detail')}" + (f" {p['errors']}" if p.get("errors") else "")
    except (TypeError, ValueError):
        pass
    return str(raw)[:limit]


def gpu_id(name):
    return GPU_IDS.get(name, name)


def _key():
    with open(KEY_PATH) as f:
        k = f.read().strip()
    if not k:
        raise RunPodError(f"{KEY_PATH} is empty")
    return k


def _request(method, url, body=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", "Bearer " + _key())
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", "paperkiln-runner/1")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if raw.strip() else None)
    except urllib.error.HTTPError as e:
        raw = e.read()[:2000].decode(errors="replace")
        return e.code, {"error": raw}
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise RunPodError(f"{method} {url.split('?')[0]}: {type(e).__name__}: {e}") from None
    except ValueError as e:   # a 2xx body that is not JSON: never taken for an empty answer
        raise RunPodError(f"{method} {url.split('?')[0]}: unreadable response: {e}") from None


def rest(method, path, body=None, timeout=60):
    return _request(method, REST + path, body, timeout)


def graphql(query, variables=None):
    st, d = _request("POST", GRAPHQL, {"query": query, "variables": variables or {}})
    if st != 200 or not d or d.get("errors"):
        raise RunPodError(f"graphql: HTTP {st}: {json.dumps(d)[:400]}")
    return d["data"]


def _list(path, key, what, query=""):
    """Every item of a v2 list: {key: [...], "pagination": {"nextCursor", "hasNextPage"}}, followed to its last page. Anything
    else (a failed page, a body without the list, a cursor that does not advance) raises: a short list would hide a pod from the
    pod cap and from the sweep."""
    out, cursor, seen = [], None, set()
    for _ in range(MAX_PAGES):
        q = "&".join(x for x in (query, ("cursor=" + urllib.parse.quote(cursor, safe="")) if cursor else "") if x)
        st, d = rest("GET", path + ("?" + q if q else ""))
        if st != 200 or not isinstance(d, dict) or not isinstance(d.get(key), list):
            raise RunPodError(f"list {what}: HTTP {st}: {json.dumps(d)[:300]}")
        out += d[key]
        pg = d.get("pagination")
        if not isinstance(pg, dict) or not isinstance(pg.get("hasNextPage"), bool):
            raise RunPodError(f"list {what}: no pagination block: {json.dumps(d)[:200]}")
        if not pg["hasNextPage"]:
            return out
        cursor = pg.get("nextCursor")
        if not isinstance(cursor, str) or not cursor or cursor in seen:
            raise RunPodError(f"list {what}: more pages but no new cursor")
        seen.add(cursor)
    raise RunPodError(f"list {what}: more than {MAX_PAGES} pages")


# ---------------------------------------------------------------- account
def _number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and x == x and abs(x) != float("inf")


def balance():
    """(clientBalance, currentSpendPerHr) in dollars. Both must be numbers in the answer: a spend rate that is missing or null is
    an answer that cannot be read, never zero (a launch's balance floor is built on it)."""
    m = (graphql("query { myself { clientBalance currentSpendPerHr } }") or {}).get("myself")
    if not isinstance(m, dict) or not _number(m.get("clientBalance")) or not _number(m.get("currentSpendPerHr")):
        raise RunPodError(f"balance: the account's balance or spend rate is not a number in the answer: {json.dumps(m)[:200]}")
    return float(m["clientBalance"]), float(m["currentSpendPerHr"])


def gpu_price(gpu, cloud="SECURE"):
    """The on-demand price per GPU-hour RunPod lists for this GPU type (None if unknown)."""
    d = graphql("query($id: String) { gpuTypes(input: {id: $id}) { id securePrice communityPrice } }",
                {"id": gpu_id(gpu)})["gpuTypes"]
    if not d:
        return None
    return d[0].get("securePrice" if cloud == "SECURE" else "communityPrice")


def catalog():
    """The catalog's data centres with their GPU availability: [{"id", "networkVolumeTypes": [...], "gpuAvailability": [...]}]."""
    st, d = _request("GET", CATALOG + "?include=GPU_AVAILABILITY")
    if st != 200 or not isinstance(d, dict) or not isinstance(d.get("dataCenters"), list):
        raise RunPodError(f"catalog: HTTP {st}: {json.dumps(d)[:300]}")
    return d["dataCenters"]


def stock(gpu, cat=None):
    """{data centre id: availability} for the GPU type, from the catalog (NONE/LOW/MEDIUM/HIGH)."""
    want, out = gpu_id(gpu), {}
    for dc in catalog() if cat is None else cat:
        for g in dc.get("gpuAvailability") or []:
            if g.get("id") == want:
                out[dc["id"]] = g.get("availability", "NONE")
    return out


def _tiers(volume_type):
    """A tier or an ordered sequence of tiers, as a tuple."""
    return (volume_type,) if isinstance(volume_type, str) else tuple(volume_type)


def _tier_rank(dc, want):
    """The index in `want` of the first tier this catalog data centre offers, or None when it offers none of them."""
    offered = dc.get("networkVolumeTypes") or []
    return next((i for i, t in enumerate(want) if t in offered), None)


def volume_dcs(volume_type, cat=None):
    """The data centres that offer network volumes of this tier, or of any tier of a sequence (the catalog's networkVolumeTypes)."""
    want = _tiers(volume_type)
    return {dc["id"] for dc in (catalog() if cat is None else cat) if _tier_rank(dc, want) is not None}


def volume_tier(dc_id, cat=None, want=None):
    """The tier a campaign's volume takes in this data centre: the first of `want` (default VOLUME_TYPES) the catalog lists for
    it, or None when it lists none of them (or the data centre is not in the catalog)."""
    want = VOLUME_TYPES if want is None else _tiers(want)
    for dc in catalog() if cat is None else cat:
        if dc.get("id") == dc_id:
            r = _tier_rank(dc, want)
            return None if r is None else want[r]
    return None


RANK = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}


def host_stock(gpu, dc, min_ram=None, min_vcpu=None, secure=True):
    """RunPod's stock status ("High"/"Medium"/"Low") for one GPU in one data centre on hosts with at least
    min_ram GB and min_vcpu vCPUs per GPU, or None when no such host has one free. The catalog's
    availability ignores host size (24 Oct 2026: EU-RO-1 listed L4 as LOW while its L4 hosts have 6 vCPUs,
    and a create asking for 8 found no instance)."""
    inp = {"gpuCount": 1, "secureCloud": secure, "dataCenterId": dc}
    if min_ram:
        inp["minMemoryInGb"] = int(min_ram)
    if min_vcpu:
        inp["minVcpuCount"] = int(min_vcpu)
    d = graphql("query($id: String, $i: GpuLowestPriceInput) { gpuTypes(input: {id: $id}) "
                "{ lowestPrice(input: $i) { stockStatus } } }", {"id": gpu_id(gpu), "i": inp})["gpuTypes"]
    return ((d or [{}])[0].get("lowestPrice") or {}).get("stockStatus")


def stocked_dcs(gpu, min_ram=None, min_vcpu=None, secure=True, volume_type=None):
    """Data centres with the GPU in stock (on hosts of at least min_ram GB / min_vcpu vCPUs per GPU when
    given), best availability first. With volume_type (a tier, or an ordered sequence of tiers such as VOLUME_TYPES), only those
    that also offer network volumes of one of those tiers: a campaign's volume is created where its pod will run, and RunPod
    rejects a tier a data centre does not have (17 Nov 2026: CA-MTL-1, the best-stocked A40 data centre, has only
    high-performance storage). Under a sequence, data centres whose best tier comes earlier in it come first (STANDARD before
    HIGH_PERFORMANCE), then by availability."""
    cat = catalog()
    s = stock(gpu, cat)
    dcs = [dc for dc, a in s.items() if RANK.get(a, 0) > 0]
    tier = {}
    if volume_type:
        want = _tiers(volume_type)
        tier = {d["id"]: r for d in cat if (r := _tier_rank(d, want)) is not None}
        dcs = [dc for dc in dcs if dc in tier]
    if min_ram or min_vcpu:
        dcs = [dc for dc in dcs if host_stock(gpu, dc, min_ram, min_vcpu, secure)]
    return sorted(dcs, key=lambda dc: (tier.get(dc, 0), -RANK[s[dc]], dc))


# ---------------------------------------------------------------- pods
def compat_pod(p):
    """A v2 pod with the v1 field names callers read, filled from its v2 fields (a key the pod already carries is kept).
    desiredStatus is v2's `status`, the actual state: a caller that waits for "RUNNING" now waits for a healthy container, and a
    pod in ERROR is neither running nor gone. publicIp and portMappings come from ssh.direct, the published 22/tcp, which v2
    gives only on a GET of one pod while it runs."""
    if not isinstance(p, dict):
        return p
    p = dict(p)
    gpu = p.get("gpu") if isinstance(p.get("gpu"), dict) else {}
    ssh = p.get("ssh") if isinstance(p.get("ssh"), dict) else {}
    direct = ssh.get("direct") if isinstance(ssh.get("direct"), dict) else {}
    p.setdefault("desiredStatus", p.get("status"))
    p.setdefault("costPerHr", p.get("cost"))
    p.setdefault("lastStartedAt", p.get("startedAt"))
    p.setdefault("publicIp", direct.get("host"))
    p.setdefault("portMappings", {"22": direct["port"]} if direct.get("port") else {})
    p.setdefault("machine", {"dataCenterId": p.get("dataCenterId"), "gpuTypeId": gpu.get("id"),
                             "secureCloud": (p.get("cloud") == "SECURE") if p.get("cloud") else None})
    p.setdefault("memoryInGb", gpu.get("memory"))
    p.setdefault("vcpuCount", gpu.get("vcpuCount"))
    # the network volume a pod mounts (v2: mounts.network[].volumeId): a sweep must see it before it deletes a volume
    net = (p.get("mounts") or {}).get("network") if isinstance(p.get("mounts"), dict) else None
    vid = next((m.get("volumeId") for m in net or [] if isinstance(m, dict) and m.get("volumeId")), None)
    p.setdefault("networkVolumeId", vid)
    p.setdefault("networkVolume", {"id": vid} if vid else None)
    return p


def list_pods():
    """Every pod on the account, cluster member pods included: v2 leaves those out of GET /v2/pods unless asked, and the pod cap
    and the sweep count from this list (a cluster's pods bill like any other; they are ended through their cluster, so a
    terminate of one fails with an error and is logged)."""
    return [compat_pod(p) for p in _list("/pods", "pods", "pods", "includeClusterPods=true")]


def ssh_keys_registered():
    """How many SSH public keys the account has registered (GET /v2/account/ssh-keys -> {"keys": [...]}). v2's startSsh needs at
    least one: "with none registered the flag does nothing and the pod has no SSH access"."""
    st, d = rest("GET", "/account/ssh-keys")
    if st != 200 or not isinstance(d, dict) or not isinstance(d.get("keys"), list):
        raise RunPodError(f"ssh keys: HTTP {st}: {json.dumps(d)[:200]}")
    return len(d["keys"])


def get_pod(pod_id):
    """The pod, or None when the API says it does not exist (terminated). Raises when the API fails."""
    st, d = rest("GET", f"/pods/{pod_id}")
    if st == 200 and isinstance(d, dict) and d.get("id"):
        return compat_pod(d)
    if st == 404:
        return None
    raise RunPodError(f"get pod {pod_id}: HTTP {st}: {json.dumps(d)[:300]}")


def ssh_endpoint(pod):
    """(ip, port) of the pod's public 22/tcp, or None while it initialises."""
    ip, pm = pod.get("publicIp"), pod.get("portMappings") or {}
    port = pm.get("22") or pm.get(22)
    return (ip, int(port)) if ip and port else None


V1_CREATE = {"name", "imageName", "computeType", "cloudType", "gpuTypeIds", "gpuCount", "dataCenterIds", "containerDiskInGb",
             "volumeInGb", "volumeMountPath", "networkVolumeId", "ports", "env", "interruptible", "minRAMPerGPU", "minVCPUPerGPU",
             "supportPublicIp"}


def v2_create_body(body):
    """The v2 create body for a flat v1 body (what runpod_transport.py builds), or the body itself when it is already v2 (it
    carries `image` or `gpu`). Refuses (RunPodError) anything it cannot carry over exactly:
      imageName -> image; containerDiskInGb -> disk; cloudType -> cloud (SECURE or COMMUNITY only);
      gpuTypeIds (one) + gpuCount -> gpu {id, count}; minRAMPerGPU, minVCPUPerGPU -> gpu.minRamPerGpu, gpu.minVcpuCountPerGpu;
      networkVolumeId + volumeMountPath -> mounts.network [{volumeId, path}]; volumeInGb > 0 without a network volume ->
      mounts.persistent {size, path}; volumeInGb 0 -> no mount (v2 adds none by default);
      ports, env, name, dataCenterIds unchanged (v2 has no default ports: the caller's list is the whole list);
      interruptible false and supportPublicIp are dropped: v2 has no spot pods and no public-IP placement field.
    startSsh is set when the body carries PUBLIC_KEY and publishes 22/tcp: v2 says a pod created without it has no SSH access,
    and that the request's own PUBLIC_KEY is kept."""
    if "queue" in body:   # a queued create outlives this call (it deploys when capacity comes) and nothing here owns or cancels it
        raise RunPodError("create pod: queued creates are not used by this client")
    if "image" in body or "gpu" in body:
        return body
    unknown = sorted(set(body) - V1_CREATE)
    if unknown:
        raise RunPodError(f"create pod: fields with no checked v2 form: {unknown}")
    if body.get("computeType", "GPU") != "GPU":
        raise RunPodError(f"create pod: computeType {body.get('computeType')!r} is not carried to v2 here")
    if body.get("interruptible"):
        raise RunPodError("create pod: v2 has no spot (interruptible) pods")
    ids = body.get("gpuTypeIds") or []
    if len(ids) != 1:
        raise RunPodError(f"create pod: v2 takes exactly one GPU type, not {ids}")
    cloud = body.get("cloudType", "SECURE")
    if cloud not in ("SECURE", "COMMUNITY"):
        raise RunPodError(f"create pod: cloud {cloud!r} is not SECURE or COMMUNITY")
    if not body.get("name") or not body.get("imageName"):
        raise RunPodError("create pod: a name and an image are required")
    gpu = {"id": ids[0], "count": int(body.get("gpuCount", 1))}
    if body.get("minRAMPerGPU"):
        gpu["minRamPerGpu"] = int(body["minRAMPerGPU"])
    if body.get("minVCPUPerGPU"):
        gpu["minVcpuCountPerGpu"] = int(body["minVCPUPerGPU"])
    out = {"name": body["name"], "image": body["imageName"], "cloud": cloud, "gpu": gpu}
    if body.get("containerDiskInGb"):
        out["disk"] = int(body["containerDiskInGb"])
    if body.get("dataCenterIds"):
        out["dataCenterIds"] = list(body["dataCenterIds"])
    if body.get("ports"):
        out["ports"] = list(body["ports"])
    if body.get("env"):
        out["env"] = {str(k): str(v) for k, v in body["env"].items()}
    if body.get("networkVolumeId"):
        if not body.get("volumeMountPath"):
            raise RunPodError("create pod: a network volume needs its mount path (v2 has no default)")
        if body.get("volumeInGb"):
            raise RunPodError("create pod: a pod volume and a network volume cannot both be mounted")
        out["mounts"] = {"network": [{"volumeId": body["networkVolumeId"], "path": body["volumeMountPath"]}]}
    elif body.get("volumeInGb"):
        out["mounts"] = {"persistent": {"size": int(body["volumeInGb"]), "path": body.get("volumeMountPath") or "/workspace"}}
    if "PUBLIC_KEY" in (out.get("env") or {}) and "22/tcp" in (out.get("ports") or []):
        out["startSsh"] = True
    return out


def create_pod(body):
    """POST /v2/pods behind the guardrails: refuses (Refused) when MAX_PODS pods exist account-wide or the
    balance is below MIN_BALANCE, or when the pod is to be reached over SSH and the account has no registered key. Takes the
    flat v1 body or a v2 body (v2_create_body); never a queued create. Returns the pod object."""
    send = v2_create_body(body)
    if send.get("startSsh") and ssh_keys_registered() < 1:
        raise Refused("the RunPod account has no registered SSH key: a pod created now could not be reached (v2's startSsh)")
    n = len(list_pods())
    if n >= MAX_PODS:
        raise Refused(f"{n} RunPod pods exist (cap {MAX_PODS}); refusing to create another")
    bal, _ = balance()
    if bal < MIN_BALANCE:
        raise Refused(f"RunPod balance ${bal:.2f} is below ${MIN_BALANCE:.0f}; refusing to create a pod")
    st, d = rest("POST", "/pods", send, timeout=120)
    if st == 422:
        raise Rejected(f"create pod: HTTP {st}: {problem_text(d)}")
    if st not in (200, 201) or not isinstance(d, dict) or not d.get("id"):
        raise RunPodError(f"create pod: HTTP {st}: {problem_text(d)}")
    return compat_pod(d)


def terminate_pod(pod_id):
    """DELETE /v2/pods/{id}; True when it is gone (deleted now, 204, or already: 404 and a GET that confirms it, by a 404 of
    its own or the status TERMINATED)."""
    st, d = rest("DELETE", f"/pods/{pod_id}")
    if st == 204:
        return True
    if st in (200, 404):   # 204 is the documented answer; anything else short of an error is confirmed by a read
        p = get_pod(pod_id)
        return p is None or p.get("status") == "TERMINATED"
    raise RunPodError(f"terminate pod {pod_id}: HTTP {st}: {json.dumps(d)[:300]}")


# ---------------------------------------------------------------- network volumes
def compat_volume(v):
    """A v2 volume with v1's dataCenterId beside v2's dataCenter."""
    if not isinstance(v, dict):
        return v
    v = dict(v)
    if v.get("dataCenterId") is None:
        v["dataCenterId"] = v.get("dataCenter")
    return v


def list_volumes():
    return [compat_volume(v) for v in _list("/network-volumes", "networkVolumes", "volumes")]


# The network volume tiers a campaign accepts, in order of preference: the first one a data centre offers is used there (21 Nov
# 2026: for a day no data centre had an A40 in stock with STANDARD storage, while CA-MTL-1 had A40s and only HIGH_PERFORMANCE
# storage). Every tier here has a rate in VOLUME_USD_PER_GB_MONTH.
VOLUME_TYPES = ("STANDARD", "HIGH_PERFORMANCE")


def create_volume(name, gb, dc, volume_type="STANDARD"):
    """A network volume of the tier given, asked for by name, so a data centre whose default is another tier cannot hand over a
    dearer volume than the ledger prices; one that does not have the tier answers with an error. Only a tier the ledger prices is
    asked for (Refused otherwise, before anything is sent). The volume returned carries `type`, the tier RunPod's answer names
    (None when it names none, which the v2 schema does not allow), and `type_asked`; the ledger prices a volume at the dearer of
    the two, and at the dearest rate when the answer names no tier (volume_entry_rate)."""
    if volume_type not in VOLUME_USD_PER_GB_MONTH:
        raise Refused(f"create volume: tier {volume_type!r} has no rate in VOLUME_USD_PER_GB_MONTH")
    st, d = rest("POST", "/network-volumes", {"name": name, "size": int(gb), "dataCenter": dc, "type": volume_type})
    if st in (400, 422):
        raise Rejected(f"create volume in {dc}: HTTP {st}: {problem_text(d)}")
    if st not in (200, 201) or not isinstance(d, dict) or not d.get("id"):
        raise RunPodError(f"create volume: HTTP {st}: {problem_text(d, 600)}")
    v = compat_volume(d)
    v["type"] = v.get("type") or None
    v["type_asked"] = volume_type
    return v


def delete_volume(vol_id):
    st, d = rest("DELETE", f"/network-volumes/{vol_id}")
    if st == 204:
        return True
    if st in (200, 404):
        return not any(v.get("id") == vol_id for v in list_volumes())
    raise RunPodError(f"delete volume {vol_id}: HTTP {st}: {json.dumps(d)[:300]}")


# Dollars per GB-month by network volume tier, for the ledger's estimate (read 21 Nov 2026).
#   STANDARD          0.07: the rate under 1 TB on https://docs.runpod.io/storage/network-volumes and https://www.runpod.io/pricing.
#   HIGH_PERFORMANCE  0.21: https://www.runpod.io/pricing lists $0.14/GB/mo, but https://docs.runpod.io/storage/high-performance-storage
#                     says "Exact pricing varies by data center", and neither the v2 API (https://api.runpod.io/v2/openapi.json: no
#                     price on a volume or a data centre) nor the docs give the rate per data centre. Priced at three times
#                     STANDARD, 1.5 times the listed rate: a margin for a dearer data centre, not a bound the sources give (at
#                     80 GB the margin is about USD 0.19 a day, against about USD 10 a day for an A40 pod; were the real rate
#                     r, the ledger would be low by 80 * (r - 0.21) / 720 dollars an hour, USD 0.09 an hour at r = 1.00).
VOLUME_USD_PER_GB_MONTH = {"STANDARD": 0.07, "HIGH_PERFORMANCE": 0.21}


def volume_usd_per_gb_month(volume_type):
    """A tier's rate. None is a ledger entry written before the tier choice, priced as STANDARD, the only tier asked for by name
    since 17 Nov 2026 (on 21 Nov 2026 every volume on the account was STANDARD). A tier with no rate here is priced at the dearest
    rate here, never at none."""
    if volume_type is None:
        volume_type = "STANDARD"
    return VOLUME_USD_PER_GB_MONTH.get(volume_type, max(VOLUME_USD_PER_GB_MONTH.values()))


def volume_entry_rate(v):
    """A ledger volume's rate: its tier's, or, when RunPod answered a tier other than the one asked for, the dearer of the two. An
    entry written since the tier choice (it carries `type_asked`) whose answer named no tier is priced at the dearest rate: its
    tier is unconfirmed. An entry without `type_asked` predates the choice (volume_usd_per_gb_month)."""
    if "type_asked" in v and not v.get("type"):
        return max(VOLUME_USD_PER_GB_MONTH.values())
    rate = volume_usd_per_gb_month(v.get("type"))
    if v.get("type_asked"):
        rate = max(rate, volume_usd_per_gb_month(v["type_asked"]))
    return rate


def ledger_usd(led, now=None):
    """Estimated dollars of a runner's ledger: pods at their recorded $/h from start to stop (or now),
    volumes at their tier's VOLUME_USD_PER_GB_MONTH from creation to deletion (or now)."""
    now = now or time.time()
    pods = sum(p["usd_per_hr"] * max(0.0, (p.get("stop") or now) - p["start"]) / 3600 for p in led.get("pods", []))
    vols = sum(v["gb"] * volume_entry_rate(v) * max(0.0, (v.get("deleted") or now) - v["created"]) / (30 * 86400)
               for v in led.get("volumes", []))
    return pods + vols


def status():
    bal, rate = balance()
    out = {"balance": round(bal, 4), "spend_per_hr": rate, "pods": [], "volumes": []}
    for p in list_pods():
        out["pods"].append({k: p.get(k) for k in ("id", "name", "status", "cost", "dataCenterId", "createdAt", "startedAt")}
                           | {"gpu": (p.get("gpu") or {}).get("id")})
    out["volumes"] = [{k: v.get(k) for k in ("id", "name", "size", "dataCenter", "type")} for v in list_volumes()]
    out["L4_stock"] = stock("L4")
    out["L4_price_secure"] = gpu_price("L4")
    return out


if __name__ == "__main__":
    if sys.argv[1:] == ["status"]:
        print(json.dumps(status(), indent=1))
    else:
        sys.exit(__doc__)
