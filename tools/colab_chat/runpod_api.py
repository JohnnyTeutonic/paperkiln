#!/usr/bin/env python3
"""A small RunPod client for runner_multi.py --backend runpod and runpod_sweep.py (24 Oct 2026).

    python3 tools/colab_chat/runpod_api.py status      # balance, spend/h, pods, volumes, L4 stock

The API key is read from ~/.runpod/api_key (one line, mode 600) and sent only as an
`Authorization: Bearer` header from this process: never printed, logged or put on a command line.
REST: https://rest.runpod.io/v1 (pods, networkvolumes). GraphQL: https://api.runpod.io/graphql
(myself { clientBalance currentSpendPerHr }, gpuTypes). Catalog: https://api.runpod.io/v2/catalog
/datacenters?include=GPU_AVAILABILITY (per data centre GPU availability NONE/LOW/MEDIUM/HIGH).
Shapes checked against docs.runpod.io on 24 Oct 2026.

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
import urllib.request

REST = "https://rest.runpod.io/v1"
GRAPHQL = "https://api.runpod.io/graphql"
CATALOG = "https://api.runpod.io/v2/catalog/datacenters"
KEY_PATH = os.path.expanduser(os.environ.get("RUNPOD_KEY_PATH", "~/.runpod/api_key"))
MAX_PODS = 2          # account-wide, every create
MIN_BALANCE = 15.0    # dollars; no create below it
GPU_IDS = {"L4": "NVIDIA L4", "T4": "Tesla T4", "A100": "NVIDIA A100 80GB PCIe", "4090": "NVIDIA GeForce RTX 4090"}


class RunPodError(RuntimeError):
    pass


class Refused(RunPodError):
    """A guardrail refused the action (pod cap, low balance): never retried blindly."""


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


def rest(method, path, body=None, timeout=60):
    return _request(method, REST + path, body, timeout)


def graphql(query, variables=None):
    st, d = _request("POST", GRAPHQL, {"query": query, "variables": variables or {}})
    if st != 200 or not d or d.get("errors"):
        raise RunPodError(f"graphql: HTTP {st}: {json.dumps(d)[:400]}")
    return d["data"]


# ---------------------------------------------------------------- account
def balance():
    """(clientBalance, currentSpendPerHr) in dollars."""
    m = graphql("query { myself { clientBalance currentSpendPerHr } }")["myself"]
    return float(m["clientBalance"]), float(m.get("currentSpendPerHr") or 0.0)


def gpu_price(gpu, cloud="SECURE"):
    """The on-demand price per GPU-hour RunPod lists for this GPU type (None if unknown)."""
    d = graphql("query($id: String) { gpuTypes(input: {id: $id}) { id securePrice communityPrice } }",
                {"id": gpu_id(gpu)})["gpuTypes"]
    if not d:
        return None
    return d[0].get("securePrice" if cloud == "SECURE" else "communityPrice")


def stock(gpu):
    """{data centre id: availability} for the GPU type, from the catalog (NONE/LOW/MEDIUM/HIGH)."""
    st, d = _request("GET", CATALOG + "?include=GPU_AVAILABILITY")
    if st != 200 or not d:
        raise RunPodError(f"catalog: HTTP {st}: {json.dumps(d)[:300]}")
    want, out = gpu_id(gpu), {}
    for dc in d.get("dataCenters", []):
        for g in dc.get("gpuAvailability") or []:
            if g.get("id") == want:
                out[dc["id"]] = g.get("availability", "NONE")
    return out


RANK = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}


def stocked_dcs(gpu):
    """Data centres with the GPU in stock, best availability first."""
    s = stock(gpu)
    return sorted((dc for dc, a in s.items() if RANK.get(a, 0) > 0), key=lambda dc: (-RANK[s[dc]], dc))


# ---------------------------------------------------------------- pods
def list_pods():
    st, d = rest("GET", "/pods")
    if st != 200 or not isinstance(d, list):
        raise RunPodError(f"list pods: HTTP {st}: {json.dumps(d)[:300]}")
    return d


def get_pod(pod_id):
    """The pod, or None when the API says it does not exist (terminated). Raises when the API fails."""
    st, d = rest("GET", f"/pods/{pod_id}")
    if st == 200 and isinstance(d, dict) and d.get("id"):
        return d
    if st in (400, 404):
        return None
    if st == 200 and not d:
        return None
    raise RunPodError(f"get pod {pod_id}: HTTP {st}: {json.dumps(d)[:300]}")


def ssh_endpoint(pod):
    """(ip, port) of the pod's public 22/tcp, or None while it initialises."""
    ip, pm = pod.get("publicIp"), pod.get("portMappings") or {}
    port = pm.get("22") or pm.get(22)
    return (ip, int(port)) if ip and port else None


def create_pod(body):
    """POST /pods behind the guardrails: refuses (Refused) when MAX_PODS pods exist account-wide or the
    balance is below MIN_BALANCE. Returns the pod object."""
    n = len(list_pods())
    if n >= MAX_PODS:
        raise Refused(f"{n} RunPod pods exist (cap {MAX_PODS}); refusing to create another")
    bal, _ = balance()
    if bal < MIN_BALANCE:
        raise Refused(f"RunPod balance ${bal:.2f} is below ${MIN_BALANCE:.0f}; refusing to create a pod")
    st, d = rest("POST", "/pods", body, timeout=120)
    if st not in (200, 201) or not isinstance(d, dict) or not d.get("id"):
        raise RunPodError(f"create pod: HTTP {st}: {json.dumps(d)[:400]}")
    return d


def terminate_pod(pod_id):
    """DELETE /pods/{id}; True when it is gone (deleted now or already)."""
    st, d = rest("DELETE", f"/pods/{pod_id}")
    if st in (200, 204):
        return True
    if st in (400, 404):
        return get_pod(pod_id) is None
    raise RunPodError(f"terminate pod {pod_id}: HTTP {st}: {json.dumps(d)[:300]}")


# ---------------------------------------------------------------- network volumes
def list_volumes():
    st, d = rest("GET", "/networkvolumes")
    if st != 200 or not isinstance(d, list):
        raise RunPodError(f"list volumes: HTTP {st}: {json.dumps(d)[:300]}")
    return d


def create_volume(name, gb, dc):
    st, d = rest("POST", "/networkvolumes", {"name": name, "size": int(gb), "dataCenterId": dc})
    if st not in (200, 201) or not isinstance(d, dict) or not d.get("id"):
        raise RunPodError(f"create volume: HTTP {st}: {json.dumps(d)[:400]}")
    return d


def delete_volume(vol_id):
    st, d = rest("DELETE", f"/networkvolumes/{vol_id}")
    if st in (200, 204):
        return True
    if st in (400, 404):
        return not any(v.get("id") == vol_id for v in list_volumes())
    raise RunPodError(f"delete volume {vol_id}: HTTP {st}: {json.dumps(d)[:300]}")


VOLUME_USD_PER_GB_MONTH = 0.07  # RunPod's network volume rate (under 1 TB), for the ledger's estimate


def ledger_usd(led, now=None):
    """Estimated dollars of a runner's ledger: pods at their recorded $/h from start to stop (or now),
    volumes at VOLUME_USD_PER_GB_MONTH from creation to deletion (or now)."""
    now = now or time.time()
    pods = sum(p["usd_per_hr"] * max(0.0, (p.get("stop") or now) - p["start"]) / 3600 for p in led.get("pods", []))
    vols = sum(v["gb"] * VOLUME_USD_PER_GB_MONTH * max(0.0, (v.get("deleted") or now) - v["created"]) / (30 * 86400)
               for v in led.get("volumes", []))
    return pods + vols


def status():
    bal, rate = balance()
    out = {"balance": round(bal, 4), "spend_per_hr": rate, "pods": [], "volumes": []}
    for p in list_pods():
        out["pods"].append({k: p.get(k) for k in ("id", "name", "desiredStatus", "costPerHr", "publicIp",
                                                  "portMappings", "lastStartedAt")}
                           | {"gpu": (p.get("machine") or {}).get("gpuTypeId") or (p.get("gpu") or {}).get("id")})
    out["volumes"] = list_volumes()
    out["L4_stock"] = stock("L4")
    out["L4_price_secure"] = gpu_price("L4")
    return out


if __name__ == "__main__":
    if sys.argv[1:] == ["status"]:
        print(json.dumps(status(), indent=1))
    else:
        sys.exit(__doc__)
