"""runpod_api.py on RunPod's REST v2 (17 Nov 2026; v1 was retired on 15 Nov 2026). Nothing here talks to RunPod: `_request` is a
fake that answers by method and URL.

Where the shapes come from:
  RECORDED  from read-only calls to the live API on 17 Nov 2026 (ids and names replaced): the empty pod list, the volume list and
            its cursor pagination, the 404 of a pod and of a volume that do not exist, the catalog's data centres, the balance.
  SCHEMA    from docs.runpod.io's v2 OpenAPI schema of the same day (no pod existed to record): the Pod object, the create
            request, 201 on create, 204 on terminate and on volume delete, the 422 of an unknown field.
Claude Fable 5.2, 17 Nov 2026."""
import json
from pathlib import Path
import sys
import urllib.parse

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "colab_chat"))
import runpod_api as api  # noqa: E402

V2 = "https://api.runpod.io/v2"
# RECORDED
EMPTY_PODS = {"pods": [], "pagination": {"hasNextPage": False, "nextCursor": None}}
VOLUME = {"id": "vol0000001", "name": "pk-x-vol", "size": 80, "dataCenter": "EU-RO-1", "type": "STANDARD"}
POD_404 = {"detail": "pod not found", "status": 404, "title": "Not Found"}
VOLUME_404 = {"detail": "network volume not found", "status": 404, "title": "Not Found"}
CATALOG = {"dataCenters": [
    {"id": "EU-SE-1", "name": "EU-SE-1", "region": "EUROPE", "compliance": [], "globalNetwork": True, "networkVolumeTypes": ["STANDARD"],
     "gpuAvailability": [{"availability": "LOW", "id": "NVIDIA A40", "name": "A40"}, {"availability": "HIGH", "id": "NVIDIA L4", "name": "L4"}]},
    {"id": "CA-MTL-1", "name": "CA-MTL-1", "region": "NORTH_AMERICA", "compliance": [], "globalNetwork": True, "networkVolumeTypes": ["STANDARD"],
     "gpuAvailability": [{"availability": "MEDIUM", "id": "NVIDIA A40", "name": "A40"}]},
    {"id": "EU-RO-1", "name": "EU-RO-1", "region": "EUROPE", "compliance": [], "globalNetwork": True, "networkVolumeTypes": ["STANDARD"],
     "gpuAvailability": [{"availability": "NONE", "id": "NVIDIA A40", "name": "A40"}]}]}
BALANCE = {"data": {"myself": {"clientBalance": 165.4733051656, "currentSpendPerHr": 0.048}}}


def pod(pid="pod0000000001", status="RUNNING", direct=True, name="pk-b2cue-ns"):   # SCHEMA: every required field of Pod
    return {"id": pid, "name": name, "status": status, "actions": ["stop", "restart", "terminate"],
            "image": "runpod/pytorch:x", "args": "", "disk": 30, "mounts": {"network": [{"volumeId": "vol0000001", "path": "/content"}]},
            "ports": ["22/tcp"], "env": {"PUBLIC_KEY": "ssh-ed25519 AAAA runner"}, "registry": None, "cloud": "SECURE",
            "dataCenterId": "EU-SE-1", "cudaVersion": "12.4", "template": None, "cost": 0.59, "locked": False,
            "gpu": {"id": "NVIDIA A40", "count": 1, "vcpuCount": 9.0, "memory": 50.0},
            "ssh": {"proxy": {"host": "ssh.runpod.io", "port": 22, "username": pid + "-x", "command": "ssh ..."},
                    "direct": {"host": "203.0.113.7", "port": 40122, "username": "root", "command": "ssh root@203.0.113.7 -p 40122"} if direct else None},
            "runtime": None, "createdAt": "2026-11-17T04:00:00Z", "startedAt": "2026-11-17T04:01:00Z", "globalNetworking": {"enabled": False}}


class Fake:
    """The API by method and URL. `routes` maps (method, path without query) to a function of (query dict, body) -> (status, json)."""

    def __init__(self, mp, routes):
        self.calls, self.routes = [], routes
        mp.setattr(api, "_request", self.request)

    def request(self, method, url, body=None, timeout=60):
        u = urllib.parse.urlsplit(url)
        self.calls.append((method, u.path, dict(urllib.parse.parse_qsl(u.query)), body))
        assert u.scheme == "https" and u.netloc == "api.runpod.io", url          # never the retired rest.runpod.io
        fn = self.routes.get((method, u.path))
        assert fn is not None, f"an unexpected call: {method} {u.path}"
        return fn(dict(urllib.parse.parse_qsl(u.query)), body)


def test_the_client_speaks_to_v2_only():
    assert api.REST == V2 and "rest.runpod.io" not in api.REST + api.GRAPHQL + api.CATALOG


def test_an_empty_pod_list_and_a_wrapped_volume_list(monkeypatch):
    Fake(monkeypatch, {("GET", "/v2/pods"): lambda q, b: (200, EMPTY_PODS),
                       ("GET", "/v2/network-volumes"): lambda q, b: (200, {"networkVolumes": [VOLUME], "pagination": {"hasNextPage": False, "nextCursor": None}})})
    assert api.list_pods() == []
    v = api.list_volumes()
    assert [x["id"] for x in v] == ["vol0000001"] and v[0]["dataCenterId"] == "EU-RO-1" == v[0]["dataCenter"]   # v1's name kept for callers


def test_every_page_of_a_list_is_followed(monkeypatch):
    pages = {None: ({"pods": [pod("a")], "pagination": {"hasNextPage": True, "nextCursor": "c1=="}}),
             "c1==": ({"pods": [pod("b")], "pagination": {"hasNextPage": True, "nextCursor": "c2"}}),
             "c2": ({"pods": [pod("c")], "pagination": {"hasNextPage": False, "nextCursor": None}})}
    f = Fake(monkeypatch, {("GET", "/v2/pods"): lambda q, b: (200, pages[q.get("cursor")])})
    assert [p["id"] for p in api.list_pods()] == ["a", "b", "c"]
    assert [c[2].get("cursor") for c in f.calls] == [None, "c1==", "c2"]          # the cursor goes back as sent, URL-quoted on the wire
    assert all(c[2].get("includeClusterPods") == "true" for c in f.calls)         # every page counts cluster member pods too


@pytest.mark.parametrize("answer, why", [
    ((500, {"error": "x"}), "HTTP 500"),
    ((200, [pod("a")]), "HTTP 200"),                                                        # v1's bare array is not a v2 list
    ((200, {"pods": [pod("a")]}), "no pagination"),
    ((200, {"pods": [pod("a")], "pagination": {"hasNextPage": True, "nextCursor": None}}), "no new cursor"),
])
def test_a_list_that_is_not_whole_raises_and_is_never_taken_for_a_short_list(monkeypatch, answer, why):
    Fake(monkeypatch, {("GET", "/v2/pods"): lambda q, b: answer})
    with pytest.raises(api.RunPodError, match=why):
        api.list_pods()


def test_a_cursor_that_repeats_raises(monkeypatch):
    Fake(monkeypatch, {("GET", "/v2/pods"): lambda q, b: (200, {"pods": [pod("a")], "pagination": {"hasNextPage": True, "nextCursor": "same"}})})
    with pytest.raises(api.RunPodError, match="no new cursor"):
        api.list_pods()


def test_a_pod_carries_the_fields_v1_callers_read(monkeypatch):
    Fake(monkeypatch, {("GET", "/v2/pods/pod0000000001"): lambda q, b: (200, pod())})
    p = api.get_pod("pod0000000001")
    assert p["desiredStatus"] == "RUNNING" == p["status"] and p["costPerHr"] == 0.59 == p["cost"]
    assert api.ssh_endpoint(p) == ("203.0.113.7", 40122)
    assert p["machine"]["dataCenterId"] == "EU-SE-1" and p["memoryInGb"] == 50.0 and p["vcpuCount"] == 9.0
    assert p["networkVolumeId"] == "vol0000001" == p["networkVolume"]["id"]        # a sweep sees the mounted volume before deleting it
    bare = api.compat_pod(dict(pod(), mounts={}))
    assert bare["networkVolumeId"] is None and bare["networkVolume"] is None


def test_a_stopped_pod_that_still_shows_its_ssh_mapping_is_not_running(monkeypatch):
    """SCHEMA (PodSsh.direct): "It can still appear on a stopped pod, so check that `status` is `RUNNING` before connecting"."""
    Fake(monkeypatch, {("GET", "/v2/pods/p"): lambda q, b: (200, pod("p", status="EXITED", direct=True))})
    p = api.get_pod("p")
    assert api.ssh_endpoint(p) == ("203.0.113.7", 40122) and p["desiredStatus"] == "EXITED"   # the transport asks for RUNNING as well
    import runpod_transport as rt
    src = Path(rt.__file__).read_text()
    assert src.count('p.get("desiredStatus") == "RUNNING" and api.ssh_endpoint(p)') == 2       # listed() and _wait_up()


@pytest.mark.parametrize("status", ["PROVISIONING", "STARTING", "ERROR", "EXITED"])
def test_a_pod_that_is_not_running_or_has_no_published_ssh_has_no_endpoint_to_use(monkeypatch, status):
    Fake(monkeypatch, {("GET", "/v2/pods/p"): lambda q, b: (200, pod("p", status=status, direct=False))})
    p = api.get_pod("p")
    assert p["desiredStatus"] == status and api.ssh_endpoint(p) is None            # runpod_transport waits for RUNNING and an endpoint


def test_a_pod_that_does_not_exist_is_none_and_any_other_failure_raises(monkeypatch):
    Fake(monkeypatch, {("GET", "/v2/pods/gone"): lambda q, b: (404, {"error": json.dumps(POD_404)}),
                       ("GET", "/v2/pods/bad"): lambda q, b: (500, {"error": "x"}),
                       ("GET", "/v2/pods/auth"): lambda q, b: (403, {"error": "forbidden"})})
    assert api.get_pod("gone") is None
    for i in ("bad", "auth"):                                                      # never read as "terminated"
        with pytest.raises(api.RunPodError):
            api.get_pod(i)


def v1_body(vol=True):
    """The flat body runpod_transport.create builds."""
    b = {"name": "pk-b2cue-ns", "imageName": "runpod/pytorch:x", "computeType": "GPU", "cloudType": "SECURE",
         "gpuTypeIds": ["NVIDIA A40"], "gpuCount": 1, "dataCenterIds": ["EU-SE-1"], "containerDiskInGb": 30 if vol else 80,
         "volumeInGb": 0, "ports": ["22/tcp"], "env": {"PUBLIC_KEY": "ssh-ed25519 AAAA runner"}, "interruptible": False,
         "minRAMPerGPU": 32, "minVCPUPerGPU": 6}
    if vol:
        b.update(networkVolumeId="vol0000001", volumeMountPath="/content")
    else:
        b["supportPublicIp"] = True
    return b


# SCHEMA: CreatePodRequest's properties (with BaseContainerConfig's and `registry`); additional properties are rejected with 422
V2_CREATE_FIELDS = {"name", "cloud", "cpu", "dataCenterIds", "globalNetworking", "gpu", "mounts", "startJupyter", "startSsh", "templateId",
                    "queue", "args", "cmd", "entrypoint", "disk", "env", "image", "ports", "registry"}
V2_GPU_FIELDS = {"id", "count", "allowedCudaVersions", "minCudaVersion", "minRamPerGpu", "minVcpuCountPerGpu"}


def test_the_flat_v1_body_becomes_the_nested_v2_body_exactly():
    assert api.v2_create_body(v1_body()) == {
        "name": "pk-b2cue-ns", "image": "runpod/pytorch:x", "cloud": "SECURE", "disk": 30, "dataCenterIds": ["EU-SE-1"],
        "gpu": {"id": "NVIDIA A40", "count": 1, "minRamPerGpu": 32, "minVcpuCountPerGpu": 6}, "ports": ["22/tcp"],
        "env": {"PUBLIC_KEY": "ssh-ed25519 AAAA runner"}, "mounts": {"network": [{"volumeId": "vol0000001", "path": "/content"}]},
        "startSsh": True}
    no_vol = api.v2_create_body(v1_body(vol=False))
    assert "mounts" not in no_vol and no_vol["disk"] == 80                         # v2 adds no volume by default
    for b in (api.v2_create_body(v1_body()), no_vol):                              # nothing v2 would reject as an unknown field
        assert set(b) <= V2_CREATE_FIELDS and set(b["gpu"]) <= V2_GPU_FIELDS
        assert set(b["mounts"]["network"][0]) == {"volumeId", "path"} if "mounts" in b else True
    v2 = {"name": "n", "image": "i", "gpu": {"id": "NVIDIA A40"}}
    assert api.v2_create_body(v2) is v2                                            # a v2 body goes through untouched


@pytest.mark.parametrize("change, why", [
    ({"interruptible": True}, "spot"),
    ({"gpuTypeIds": ["NVIDIA A40", "NVIDIA L40S"]}, "exactly one GPU type"),
    ({"gpuTypeIds": []}, "exactly one GPU type"),
    ({"cloudType": "ALL"}, "SECURE or COMMUNITY"),
    ({"computeType": "CPU"}, "computeType"),
    ({"countryCodes": ["SE"]}, "no checked v2 form"),
    ({"allowedCudaVersions": ["12.4"]}, "no checked v2 form"),
    ({"volumeMountPath": None}, "mount path"),
    ({"volumeInGb": 20}, "cannot both"),
    ({"imageName": ""}, "name and an image"),
])
def test_what_cannot_be_carried_to_v2_exactly_is_refused_never_dropped(change, why):
    with pytest.raises(api.RunPodError, match=why):
        api.v2_create_body(dict(v1_body(), **change))


def account(mp, pods=(), bal=165.47, create=None, calls=None, keys=1):
    routes = {("GET", "/v2/pods"): lambda q, b: (200, {"pods": list(pods), "pagination": {"hasNextPage": False, "nextCursor": None}}),
              ("GET", "/v2/account/ssh-keys"): lambda q, b: (200, {"keys": [{"name": "runner"}] * keys}),   # RECORDED: {"keys": [...]}
              ("POST", "/graphql"): lambda q, b: (200, {"data": {"myself": {"clientBalance": bal, "currentSpendPerHr": 0.048}}})}
    if create:
        routes[("POST", "/v2/pods")] = create
    return Fake(mp, routes)


def test_create_sends_the_v2_body_once_and_returns_the_pod(monkeypatch):
    f = account(monkeypatch, create=lambda q, b: (201, pod(status="PROVISIONING", direct=False)))
    p = api.create_pod(v1_body())
    assert p["id"] == "pod0000000001" and p["costPerHr"] == 0.59 and p["desiredStatus"] == "PROVISIONING"
    posts = [c for c in f.calls if c[:2] == ("POST", "/v2/pods")]
    assert len(posts) == 1 and posts[0][3] == api.v2_create_body(v1_body())


def test_the_guardrails_refuse_before_anything_is_sent(monkeypatch):
    f = account(monkeypatch, pods=[pod(str(i)) for i in range(api.MAX_PODS)], create=lambda q, b: pytest.fail("a create was sent"))
    with pytest.raises(api.Refused, match=f"cap {api.MAX_PODS}"):
        api.create_pod(v1_body())
    f = account(monkeypatch, bal=api.MIN_BALANCE - 0.01, create=lambda q, b: pytest.fail("a create was sent"))
    with pytest.raises(api.Refused, match="below"):
        api.create_pod(v1_body())
    f = account(monkeypatch, create=lambda q, b: pytest.fail("a create was sent"))
    with pytest.raises(api.RunPodError, match="spot"):                              # an untranslatable body: nothing is asked or sent
        api.create_pod(dict(v1_body(), interruptible=True))
    assert f.calls == []
    f = account(monkeypatch, keys=0, create=lambda q, b: pytest.fail("a create was sent"))
    with pytest.raises(api.Refused, match="no registered SSH key"):                 # v2: startSsh does nothing without one
        api.create_pod(v1_body())
    for bad in ((500, {"error": "x"}), (200, {"nokeys": []}), (200, [])):           # a key count that cannot be read stops it too
        f = account(monkeypatch, create=lambda q, b: pytest.fail("a create was sent"))
        f.routes[("GET", "/v2/account/ssh-keys")] = lambda q, b, bad=bad: bad
        with pytest.raises(api.RunPodError, match="ssh keys"):
            api.create_pod(v1_body())


def test_a_queued_create_is_refused_in_either_form(monkeypatch):
    f = account(monkeypatch, create=lambda q, b: pytest.fail("a create was sent"))
    for body in ({"name": "n", "image": "i", "gpu": {"id": "NVIDIA A40"}, "queue": {}}, dict(v1_body(), queue={"expiresAt": "2026-12-01T00:00:00Z"})):
        with pytest.raises(api.RunPodError, match="queued creates"):
            api.create_pod(body)
    assert f.calls == []


def test_a_pod_count_that_cannot_be_read_stops_the_create(monkeypatch):
    Fake(monkeypatch, {("GET", "/v2/pods"): lambda q, b: (503, {"error": "x"}), ("POST", "/v2/pods"): lambda q, b: pytest.fail("a create was sent"),
                       ("GET", "/v2/account/ssh-keys"): lambda q, b: (200, {"keys": [{}]})})
    with pytest.raises(api.RunPodError, match="list pods"):
        api.create_pod(v1_body())


@pytest.mark.parametrize("answer", [
    (422, {"error": json.dumps({"title": "Unprocessable Entity", "status": 422, "detail": "x", "errors": ["$: additional properties 'bogus' not allowed"]})}),
    (400, {"error": "bad"}), (202, {"id": "queued1", "status": "QUEUED", "podId": None}), (201, {"name": "no id"}), (201, None)])
def test_a_create_that_did_not_return_a_pod_raises(monkeypatch, answer):
    account(monkeypatch, create=lambda q, b: answer)                               # 202 is a queued pod: this client never asks to queue
    with pytest.raises(api.RunPodError, match="create pod"):
        api.create_pod(v1_body())


def test_terminate_is_true_only_when_the_pod_is_gone(monkeypatch):
    Fake(monkeypatch, {("DELETE", "/v2/pods/p1"): lambda q, b: (204, None)})
    assert api.terminate_pod("p1") is True
    Fake(monkeypatch, {("DELETE", "/v2/pods/p2"): lambda q, b: (404, {"error": json.dumps(POD_404)}),
                       ("GET", "/v2/pods/p2"): lambda q, b: (404, {"error": json.dumps(POD_404)})})
    assert api.terminate_pod("p2") is True                                          # already gone, confirmed by a GET
    Fake(monkeypatch, {("DELETE", "/v2/pods/p3"): lambda q, b: (404, {"error": "x"}), ("GET", "/v2/pods/p3"): lambda q, b: (200, pod("p3"))})
    assert api.terminate_pod("p3") is False                                         # a 404 on delete while the pod still answers
    Fake(monkeypatch, {("DELETE", "/v2/pods/p5"): lambda q, b: (200, None), ("GET", "/v2/pods/p5"): lambda q, b: (200, pod("p5"))})
    assert api.terminate_pod("p5") is False                                         # an undocumented 200 is not taken on trust
    Fake(monkeypatch, {("DELETE", "/v2/pods/p6"): lambda q, b: (200, None), ("GET", "/v2/pods/p6"): lambda q, b: (200, pod("p6", status="TERMINATED", direct=False))})
    assert api.terminate_pod("p6") is True
    for code in (409, 401, 403, 429, 500):
        Fake(monkeypatch, {("DELETE", "/v2/pods/p4"): lambda q, b, code=code: (code, {"error": "x"})})
        with pytest.raises(api.RunPodError, match="terminate pod"):
            api.terminate_pod("p4")


def test_volumes_create_with_v2s_field_and_delete_only_when_gone(monkeypatch):
    f = Fake(monkeypatch, {("POST", "/v2/network-volumes"): lambda q, b: (201, dict(VOLUME, name=b["name"], size=b["size"], dataCenter=b["dataCenter"]))})
    v = api.create_volume("pk-b2cue-ns-vol", 80, "EU-SE-1")
    assert f.calls[0][3] == {"name": "pk-b2cue-ns-vol", "size": 80, "dataCenter": "EU-SE-1", "type": "STANDARD"}   # dataCenter, not v1's
    assert api.VOLUME_TYPES[0] == "STANDARD"                                        # the default tier, asked by name
    assert v["id"] == "vol0000001" and v["dataCenterId"] == "EU-SE-1" and v["type"] == "STANDARD"
    Fake(monkeypatch, {("DELETE", "/v2/network-volumes/v1"): lambda q, b: (204, None)})
    assert api.delete_volume("v1") is True
    listing = lambda q, b: (200, {"networkVolumes": [VOLUME], "pagination": {"hasNextPage": False, "nextCursor": None}})  # noqa: E731
    Fake(monkeypatch, {("DELETE", "/v2/network-volumes/gone"): lambda q, b: (404, {"error": json.dumps(VOLUME_404)}), ("GET", "/v2/network-volumes"): listing})
    assert api.delete_volume("gone") is True
    Fake(monkeypatch, {("DELETE", "/v2/network-volumes/vol0000001"): lambda q, b: (404, {"error": "x"}), ("GET", "/v2/network-volumes"): listing})
    assert api.delete_volume("vol0000001") is False                                 # still listed
    Fake(monkeypatch, {("DELETE", "/v2/network-volumes/v2"): lambda q, b: (200, None), ("GET", "/v2/network-volumes"): listing})
    assert api.delete_volume("v2") is True and api.delete_volume.__name__                              # 200: confirmed by the list
    Fake(monkeypatch, {("DELETE", "/v2/network-volumes/busy"): lambda q, b: (409, {"error": "conflict"})})   # SCHEMA: 409 is documented
    with pytest.raises(api.RunPodError, match="delete volume"):
        api.delete_volume("busy")


def test_stock_reads_the_v2_catalog_and_the_balance_stays_on_graphql(monkeypatch):
    f = Fake(monkeypatch, {("GET", "/v2/catalog/datacenters"): lambda q, b: (200, CATALOG), ("POST", "/graphql"): lambda q, b: (200, BALANCE)})
    assert api.stock("A40") == {"EU-SE-1": "LOW", "CA-MTL-1": "MEDIUM", "EU-RO-1": "NONE"}
    assert f.calls[0][2] == {"include": "GPU_AVAILABILITY"}
    assert api.stocked_dcs("A40") == ["CA-MTL-1", "EU-SE-1"]                        # best first; NONE left out
    assert api.balance() == (165.4733051656, 0.048)


# RECORDED, 17 Nov 2026: the catalog's volume tiers for the two data centres with A40, and RunPod's answer to C1a's first attempt
TIERS = {"dataCenters": [dict(d, networkVolumeTypes={"EU-SE-1": ["STANDARD"], "CA-MTL-1": ["HIGH_PERFORMANCE"], "EU-RO-1": ["STANDARD"]}[d["id"]])
                         for d in CATALOG["dataCenters"]]}
VOLUME_400 = ('{"detail":"Standard storage is not available in data center \\"CA-MTL-1\\". Data centers with standard storage: AP-JP-1, CA-MTL-3, '
              'CA-MTL-4, EU-CZ-1, EU-NL-1, EU-RO-1, EU-SE-1, EUR-IS-1, EUR-IS-2, EUR-IS-3, EUR-IS-4, EUR-NO-1, US-CO-1, US-GA-2, US-MO-2, US-NC-1, '
              'US-NE-1, US-TX-3.","status":400,"title":"Bad Request"}')


def test_a_campaign_with_a_volume_is_placed_only_where_its_storage_tier_exists(monkeypatch):
    """C1a's first attempt: CA-MTL-1 was the best-stocked A40 data centre and has no standard storage."""
    Fake(monkeypatch, {("GET", "/v2/catalog/datacenters"): lambda q, b: (200, TIERS)})
    assert api.stocked_dcs("A40") == ["CA-MTL-1", "EU-SE-1"]                        # stock alone, as before
    assert api.stocked_dcs("A40", volume_type="STANDARD") == ["EU-SE-1"]
    assert api.stocked_dcs("A40", volume_type="HIGH_PERFORMANCE") == ["CA-MTL-1"]
    assert api.volume_dcs("STANDARD", TIERS["dataCenters"]) == {"EU-SE-1", "EU-RO-1"}
    import runpod_transport as rt
    seen = {}
    monkeypatch.setattr(api, "stocked_dcs", lambda gpu, ram, vcpu, secure=True, volume_type=None: seen.update(vt=volume_type) or [])

    class T:
        gpu, cloud = "A40", "SECURE"
    T.persistent = True
    rt.RunPodTransport._stocked(T)
    assert seen["vt"] == api.VOLUME_TYPES == ("STANDARD", "HIGH_PERFORMANCE")       # the transport asks for its accepted tiers, in order
    T.persistent = False
    rt.RunPodTransport._stocked(T)
    assert seen["vt"] is None                                                       # a pod without a network volume goes anywhere


def test_a_request_runpod_rejects_is_a_fault_with_its_whole_text_never_a_shortage(monkeypatch):
    Fake(monkeypatch, {("POST", "/v2/network-volumes"): lambda q, b: (400, {"error": VOLUME_400})})
    with pytest.raises(api.Rejected) as e:
        api.create_volume("pk-b2cue-ns-vol", 80, "CA-MTL-1")
    msg = str(e.value)
    assert "create volume in CA-MTL-1: HTTP 400" in msg and "Standard storage is not available" in msg and msg.endswith("US-TX-3.")
    assert isinstance(e.value, api.RunPodError) and not isinstance(e.value, api.Refused)
    account(monkeypatch, create=lambda q, b: (422, {"error": json.dumps({"title": "Unprocessable Entity", "status": 422, "detail": "bad body",
                                                                         "errors": ["$: additional properties 'bogus' not allowed"]})}))
    with pytest.raises(api.Rejected, match="additional properties 'bogus' not allowed"):
        api.create_pod(v1_body())
    # a pod create's 400 keeps RunPod's text but is an ordinary failure: the schema does not say it is never capacity
    account(monkeypatch, create=lambda q, b: (400, {"error": json.dumps({"title": "Bad Request", "status": 400, "detail": "no instances available"})}))
    with pytest.raises(api.RunPodError, match="Bad Request: no instances available") as e:
        api.create_pod(v1_body())
    assert not isinstance(e.value, api.Rejected)
    for code in (409, 429, 500, 503):                                               # anything else is a failure to retry, not a rejection
        account(monkeypatch, create=lambda q, b, code=code: (code, {"error": "x"}))
        with pytest.raises(api.RunPodError) as e:
            api.create_pod(v1_body())
        assert not isinstance(e.value, api.Rejected)


def test_the_transport_and_the_runner_report_a_rejected_create_as_a_fault(monkeypatch, tmp_path):
    import runner_multi as rm
    import runpod_transport as rt

    class St:
        d = {"runpod": {"pod": None, "volume": None, "retired_volumes": [], "ledger": {"pods": [], "volumes": []}}, "endpoint": None, "launches": 0}

        def save(self):
            pass
    logged = []
    (tmp_path / "k.pub").write_text("ssh-ed25519 AAAA runner")
    monkeypatch.setattr(rt, "KEY", str(tmp_path / "k"))
    t = rt.RunPodTransport.__new__(rt.RunPodTransport)
    t.gpu, t.cloud, t.volume_gb, t.dc, t.image, t.name, t.persistent = "A40", "SECURE", 80, None, "img", "pk-b2cue-ns", True
    t.st, t.log = St(), logged.append
    monkeypatch.setattr(rt.RunPodTransport, "R", lambda self, st=None: St.d["runpod"])
    monkeypatch.setattr(rt.RunPodTransport, "over_budget", lambda self, st: None)
    monkeypatch.setattr(api, "stocked_dcs", lambda *a, **k: ["CA-MTL-1"])
    # a catalog that lists STANDARD for CA-MTL-1 (CATALOG), so the runner asks for that tier and RunPod refuses it
    Fake(monkeypatch, {("POST", "/v2/network-volumes"): lambda q, b: (400, {"error": VOLUME_400}),
                       ("GET", "/v2/catalog/datacenters"): lambda q, b: (200, CATALOG)})
    for n in (1, 2):
        rc, out = t.create(St())
        assert rc == 2 and out.startswith("rejected: create volume in CA-MTL-1: HTTP 400") and "Standard storage is not available" in out
        assert St.d["runpod"]["create_rejected"]["n"] == n and "CREATE REJECTED by RunPod" in logged[-1]
    # the runner's line: a fault, with the reason from its start, and no "no A40"
    lines = []
    monkeypatch.setattr(rm, "T", type("X", (), {"backend": "runpod", "gpu": "A40", "stop": lambda self, st: None, "create": lambda self, st: (2, out)})())
    monkeypatch.setattr(rm, "log", lines.append)
    assert rm.launch(St(), "c", "j", [], []) is False
    assert lines[0].startswith("create FAILED, request rejected by RunPod") and "Standard storage is not available" in lines[0] and "no A40" not in lines[0]
    assert "dc_fails" not in St.d["runpod"]                                          # a rejection says nothing about a data centre's stock
    monkeypatch.setattr(rm, "T", type("X", (), {"backend": "runpod", "gpu": "A40", "stop": lambda self, st: None,
                                                "create": lambda self, st: (1, "RunPodError: no A40 in stock in any data centre")})())
    lines.clear()
    assert rm.launch(St(), "c", "j", [], []) is False and lines[0].startswith("no pod (RunPodError: no A40 in stock")
    monkeypatch.setattr(rm, "T", type("X", (), {"backend": "colab", "gpu": "L4", "stop": lambda self, st: None, "create": lambda self, st: (2, "usage")})())
    lines.clear()
    assert rm.launch(St(), "c", "j", [], []) is False and lines[0].startswith("no pod (usage")       # Colab's exit code 2 is its own


@pytest.mark.parametrize("myself", [{"clientBalance": 165.47}, {"clientBalance": 165.47, "currentSpendPerHr": None}, {"currentSpendPerHr": 0.048},
                                    {"clientBalance": "165", "currentSpendPerHr": 0.048}, None])
def test_a_balance_answer_without_both_numbers_is_an_error_never_a_zero_rate(monkeypatch, myself):
    Fake(monkeypatch, {("POST", "/graphql"): lambda q, b: (200, {"data": {"myself": myself}})})
    with pytest.raises(api.RunPodError, match="balance"):
        api.balance()


def test_a_zero_spend_rate_that_the_api_states_is_zero(monkeypatch):
    Fake(monkeypatch, {("POST", "/graphql"): lambda q, b: (200, {"data": {"myself": {"clientBalance": 20.0, "currentSpendPerHr": 0}}})})
    assert api.balance() == (20.0, 0.0)


def test_a_two_hundred_that_is_not_json_is_an_error(monkeypatch):
    import io
    import urllib.request

    class R(io.BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    monkeypatch.setattr(api, "_key", lambda: "k")
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=60: R(b"<html>gateway</html>"))
    with pytest.raises(api.RunPodError, match="unreadable response"):
        api.list_pods()


def test_the_transport_gives_up_at_once_on_a_pod_in_error(monkeypatch):
    """The migration page: status is now the actual state, "Handle ERROR in polling loops"."""
    import runpod_transport as rt

    class T:
        logged = []

        def _fetch(self):
            return api.compat_pod(pod("p", status="ERROR", direct=False))

        def log(self, m):
            self.logged.append(m)

        def exec(self, *a, **k):
            pytest.fail("ssh was tried on a pod in ERROR")
    monkeypatch.setattr(rt.time, "sleep", lambda s: pytest.fail("the wait went on"))
    t = T()
    assert rt.RunPodTransport._wait_up(t, "p") is None and "ERROR" in t.logged[0]



def test_a_data_centre_given_by_hand_is_checked_for_the_volume_tier_before_anything_is_retired_or_created(monkeypatch):
    import runpod_transport as rt
    # CA-MTL-1 listing no tier at all (since 21 Nov 2026 its HIGH_PERFORMANCE tier is accepted: test_..._high_performance_... below)
    none_there = {"dataCenters": [dict(d, networkVolumeTypes=[] if d["id"] == "CA-MTL-1" else d["networkVolumeTypes"]) for d in TIERS["dataCenters"]]}
    f = Fake(monkeypatch, {("GET", "/v2/catalog/datacenters"): lambda q, b: (200, none_there),
                           ("POST", "/v2/network-volumes"): lambda q, b: pytest.fail("a volume was asked for")})
    R = {"pod": None, "volume": {"id": "vol0000001", "dc": "EU-SE-1", "gb": 80}, "retired_volumes": [], "ledger": {"pods": [], "volumes": []}}

    class T:
        gpu, cloud, volume_gb, name, persistent, dc = "A40", "SECURE", 80, "pk-b2cue-ns", True, "CA-MTL-1"

        def R(self, st=None):
            return R

        def _retire(self, st, vol):
            pytest.fail("the existing volume was retired")
    with pytest.raises(api.RunPodError, match=r"--datacenter CA-MTL-1 offers no network volume of the tiers \['STANDARD', 'HIGH_PERFORMANCE'\]") as e:
        rt.RunPodTransport._volume_for_create(T(), None)
    assert not isinstance(e.value, api.Rejected) and R["volume"]["dc"] == "EU-SE-1"
    assert [c[:2] for c in f.calls] == [("GET", "/v2/catalog/datacenters")]
    T.dc = "EU-SE-1"                                                                # a data centre with the tier: the volume there is reused
    assert rt.RunPodTransport._volume_for_create(T(), None) == ("EU-SE-1", R["volume"])


def test_a_rejected_pod_create_does_not_count_toward_moving_the_campaign(monkeypatch, tmp_path):
    import runpod_transport as rt
    (tmp_path / "k.pub").write_text("ssh-ed25519 AAAA runner")
    monkeypatch.setattr(rt, "KEY", str(tmp_path / "k"))
    monkeypatch.setattr(rt, "image_digest", lambda image: "sha256:x")
    R = {"pod": None, "volume": {"id": "vol0000001", "dc": "EU-SE-1", "gb": 80}, "retired_volumes": [], "ledger": {"pods": [], "volumes": []}}

    class St:
        d = {"runpod": R, "endpoint": None}

        def save(self):
            pass
    t = rt.RunPodTransport.__new__(rt.RunPodTransport)
    t.gpu, t.cloud, t.volume_gb, t.dc, t.image, t.name, t.persistent = "A40", "SECURE", 80, None, "img", "pk-b2cue-ns", True
    t.st, t.log = St(), lambda m: None
    monkeypatch.setattr(rt.RunPodTransport, "R", lambda self, st=None: R)
    monkeypatch.setattr(rt.RunPodTransport, "over_budget", lambda self, st: None)
    monkeypatch.setattr(api, "stocked_dcs", lambda *a, **k: ["EU-SE-1"])
    answer = {}
    account(monkeypatch, create=lambda q, b: answer["a"])
    answer["a"] = (422, {"error": json.dumps({"title": "Unprocessable Entity", "status": 422, "detail": "bad", "errors": ["$.gpu: x"]})})
    assert t.create(St())[0] == 2 and "dc_fails" not in R and R["create_rejected"]["n"] == 1 and R["creating"] is None
    answer["a"] = (400, {"error": json.dumps({"title": "Bad Request", "status": 400, "detail": "no instances available"})})
    rc, out = t.create(St())
    assert rc == 1 and "no instances available" in out and R["dc_fails"] == 1      # an ordinary failed create still counts



# ---------------------------------------------------------------- volume tiers (21 Nov 2026)
# For a day no data centre had an A40 in stock with STANDARD storage, while CA-MTL-1 had A40s and only HIGH_PERFORMANCE storage.
# The runner now accepts VOLUME_TYPES in order: STANDARD where a data centre offers it, else HIGH_PERFORMANCE, priced by tier.
# Claude Opus 5.5, 21 Nov 2026.

def dcs(*rows):
    """A catalog of (id, tiers, A40 availability) rows."""
    return [{"id": i, "name": i, "region": "EUROPE", "compliance": [], "globalNetwork": True, "networkVolumeTypes": list(t),
             "gpuAvailability": [{"availability": a, "id": "NVIDIA A40", "name": "A40"}]} for i, t, a in rows]


MIXED = dcs(("STD-LOW", ["STANDARD"], "LOW"), ("HPS-HIGH", ["HIGH_PERFORMANCE"], "HIGH"), ("NONE-HIGH", [], "HIGH"),
            ("BOTH-MED", ["HIGH_PERFORMANCE", "STANDARD"], "MEDIUM"), ("STD-OUT", ["STANDARD"], "NONE"))


def test_the_accepted_tiers_are_standard_then_high_performance_and_each_has_a_rate():
    assert api.VOLUME_TYPES == ("STANDARD", "HIGH_PERFORMANCE")
    assert set(api.VOLUME_TYPES) <= set(api.VOLUME_USD_PER_GB_MONTH)
    assert api.VOLUME_USD_PER_GB_MONTH["STANDARD"] == 0.07
    hps = api.VOLUME_USD_PER_GB_MONTH["HIGH_PERFORMANCE"]
    assert hps > 0.14 and hps == pytest.approx(3 * api.VOLUME_USD_PER_GB_MONTH["STANDARD"])   # three times STANDARD, above the listed 0.14


def test_a_data_centres_tier_is_the_first_accepted_one_it_offers():
    assert api.volume_tier("STD-LOW", MIXED) == "STANDARD"
    assert api.volume_tier("HPS-HIGH", MIXED) == "HIGH_PERFORMANCE"
    assert api.volume_tier("BOTH-MED", MIXED) == "STANDARD"                          # the catalog's order does not matter
    assert api.volume_tier("NONE-HIGH", MIXED) is None
    assert api.volume_tier("NOT-LISTED", MIXED) is None
    assert api.volume_tier("X", dcs(("X", ["SOMETHING_NEW"], "HIGH"))) is None       # a tier with no rate is never taken
    assert api.volume_dcs(api.VOLUME_TYPES, MIXED) == {"STD-LOW", "HPS-HIGH", "BOTH-MED", "STD-OUT"}
    assert api.volume_dcs("STANDARD", MIXED) == {"STD-LOW", "BOTH-MED", "STD-OUT"}   # one tier by name, as before


def test_the_stock_filter_keeps_data_centres_with_an_accepted_tier_standard_ones_first(monkeypatch):
    Fake(monkeypatch, {("GET", "/v2/catalog/datacenters"): lambda q, b: (200, {"dataCenters": MIXED})})
    assert api.stocked_dcs("A40") == ["HPS-HIGH", "NONE-HIGH", "BOTH-MED", "STD-LOW"]            # stock alone: availability, then name
    assert api.stocked_dcs("A40", volume_type="STANDARD") == ["BOTH-MED", "STD-LOW"]             # one tier: as before
    assert api.stocked_dcs("A40", volume_type="HIGH_PERFORMANCE") == ["HPS-HIGH", "BOTH-MED"]
    # the accepted tiers: STANDARD data centres first (by availability), then HIGH_PERFORMANCE ones; none without either
    assert api.stocked_dcs("A40", volume_type=api.VOLUME_TYPES) == ["BOTH-MED", "STD-LOW", "HPS-HIGH"]


def test_the_host_floor_is_asked_only_of_data_centres_with_an_accepted_tier(monkeypatch):
    asked = []

    def gql(q, b):
        dc = b["variables"]["i"]["dataCenterId"]
        asked.append(dc)
        return 200, {"data": {"gpuTypes": [{"lowestPrice": {"stockStatus": None if dc == "BOTH-MED" else "Low"}}]}}
    Fake(monkeypatch, {("GET", "/v2/catalog/datacenters"): lambda q, b: (200, {"dataCenters": MIXED}), ("POST", "/graphql"): gql})
    assert api.stocked_dcs("A40", 32, 6, volume_type=api.VOLUME_TYPES) == ["STD-LOW", "HPS-HIGH"]  # BOTH-MED has no host big enough
    assert sorted(asked) == ["BOTH-MED", "HPS-HIGH", "STD-LOW"]                                   # NONE-HIGH and STD-OUT never asked


@pytest.mark.parametrize("answer_type", [None, "HIGH_PERFORMANCE"])
def test_a_volume_is_created_in_the_tier_asked_for_and_carries_it(monkeypatch, answer_type):
    def made(q, b):
        v = dict(VOLUME, name=b["name"], size=b["size"], dataCenter=b["dataCenter"])
        v.pop("type")
        return 201, (dict(v, type=answer_type) if answer_type else v)
    f = Fake(monkeypatch, {("POST", "/v2/network-volumes"): made})
    v = api.create_volume("pk-rel-r2-vol", 80, "CA-MTL-1", "HIGH_PERFORMANCE")
    assert f.calls[0][3] == {"name": "pk-rel-r2-vol", "size": 80, "dataCenter": "CA-MTL-1", "type": "HIGH_PERFORMANCE"}
    assert v["type"] == "HIGH_PERFORMANCE"


def test_a_tier_the_ledger_cannot_price_is_never_asked_for(monkeypatch):
    f = Fake(monkeypatch, {("POST", "/v2/network-volumes"): lambda q, b: pytest.fail("a volume was asked for")})
    with pytest.raises(api.RunPodError, match="no rate"):
        api.create_volume("pk-x-vol", 80, "CA-MTL-1", "SOMETHING_NEW")
    assert f.calls == []


def test_the_ledger_prices_each_volume_at_its_tier():
    t, month = 1_000_000.0, 30 * 86400
    led = {"pods": [], "volumes": [
        {"gb": 80, "type": "STANDARD", "created": t, "deleted": t + month},
        {"gb": 80, "type": "HIGH_PERFORMANCE", "created": t, "deleted": t + month},
        {"gb": 80, "created": t, "deleted": t + month},                              # before 21 Nov 2026: STANDARD was the only tier
        {"gb": 10, "type": "SOMETHING_NEW", "created": t, "deleted": t + month}]}    # no rate: the dearest one, never none
    assert api.ledger_usd(led, now=t + month) == pytest.approx(80 * 0.07 + 80 * 0.21 + 80 * 0.07 + 10 * 0.21)
    one_day = {"pods": [{"usd_per_hr": 0.44, "start": t, "stop": t + 86400}],
               "volumes": [{"gb": 80, "type": "HIGH_PERFORMANCE", "created": t, "deleted": None}]}
    assert api.ledger_usd(one_day, now=t + 86400) == pytest.approx(0.44 * 24 + 80 * 0.21 / 30)


def transport(monkeypatch, tmp_path, R, dc=None):
    import runpod_transport as rt

    class St:
        d = {"runpod": R, "endpoint": None}

        def save(self):
            pass
    t = rt.RunPodTransport.__new__(rt.RunPodTransport)
    t.gpu, t.cloud, t.volume_gb, t.dc, t.image, t.name, t.persistent = "A40", "SECURE", 80, dc, "img", "pk-rel-r2", True
    t.logged = []
    t.st, t.log = St(), t.logged.append
    monkeypatch.setattr(rt.RunPodTransport, "R", lambda self, st=None: R)
    return t, St()


def fresh_state():
    return {"pod": None, "volume": None, "retired_volumes": [], "ledger": {"pods": [], "volumes": []}}


def floor_ok(q, b):
    return 200, {"data": {"gpuTypes": [{"lowestPrice": {"stockStatus": "Low"}}]}}


def volume_made(q, b):
    return 201, dict(VOLUME, id="volnew", name=b["name"], size=b["size"], dataCenter=b["dataCenter"], type=b["type"])


def test_with_a40s_only_beside_high_performance_storage_the_campaign_takes_that_tier_and_records_it(monkeypatch, tmp_path):
    """The case of 21 Nov 2026: A40s in CA-MTL-1 (HIGH_PERFORMANCE only) and EU-SE-1 (no volume tier); STANDARD nowhere with A40s."""
    now = dcs(("CA-MTL-1", ["HIGH_PERFORMANCE"], "MEDIUM"), ("EU-SE-1", [], "HIGH"), ("EU-RO-1", ["STANDARD"], "NONE"))
    f = Fake(monkeypatch, {("GET", "/v2/catalog/datacenters"): lambda q, b: (200, {"dataCenters": now}),
                           ("POST", "/graphql"): floor_ok, ("POST", "/v2/network-volumes"): volume_made})
    R = fresh_state()
    t, st = transport(monkeypatch, tmp_path, R)
    dc, vol = t._volume_for_create(st)
    assert dc == "CA-MTL-1" and vol["type"] == "HIGH_PERFORMANCE" and R["volume"] is vol
    assert [c[3] for c in f.calls if c[:2] == ("POST", "/v2/network-volumes")] == [
        {"name": "pk-rel-r2-vol", "size": 80, "dataCenter": "CA-MTL-1", "type": "HIGH_PERFORMANCE"}]
    led = R["ledger"]["volumes"]
    assert len(led) == 1 and led[0]["type"] == "HIGH_PERFORMANCE" and led[0]["id"] == "volnew" and led[0]["dc"] == "CA-MTL-1"
    assert "HIGH_PERFORMANCE" in t.logged[-1]
    assert api.ledger_usd(R["ledger"], now=led[0]["created"] + 30 * 86400) == pytest.approx(80 * 0.21)


def test_standard_storage_is_taken_where_there_is_any_a40_beside_it(monkeypatch, tmp_path):
    now = dcs(("CA-MTL-1", ["HIGH_PERFORMANCE"], "HIGH"), ("EU-SE-1", ["STANDARD"], "LOW"))
    f = Fake(monkeypatch, {("GET", "/v2/catalog/datacenters"): lambda q, b: (200, {"dataCenters": now}),
                           ("POST", "/graphql"): floor_ok, ("POST", "/v2/network-volumes"): volume_made})
    R = fresh_state()
    t, st = transport(monkeypatch, tmp_path, R)
    dc, vol = t._volume_for_create(st)
    assert (dc, vol["type"], R["ledger"]["volumes"][0]["type"]) == ("EU-SE-1", "STANDARD", "STANDARD")
    assert [c[3]["type"] for c in f.calls if c[:2] == ("POST", "/v2/network-volumes")] == ["STANDARD"]


def test_a_campaigns_existing_volume_from_before_the_tier_choice_is_reused_as_it_is(monkeypatch, tmp_path):
    now = dcs(("CA-MTL-1", ["HIGH_PERFORMANCE"], "HIGH"), ("EU-SE-1", ["STANDARD"], "LOW"))
    Fake(monkeypatch, {("GET", "/v2/catalog/datacenters"): lambda q, b: (200, {"dataCenters": now}), ("POST", "/graphql"): floor_ok,
                       ("POST", "/v2/network-volumes"): lambda q, b: pytest.fail("a volume was asked for")})
    R = fresh_state()
    R["volume"] = {"id": "vol0000001", "dc": "EU-SE-1", "gb": 80, "created": 1.0}                 # no "type": created STANDARD
    R["ledger"]["volumes"].append(dict(R["volume"]))
    t, st = transport(monkeypatch, tmp_path, R)
    assert t._volume_for_create(st) == ("EU-SE-1", R["volume"]) and "type" not in R["volume"]
    assert api.ledger_usd(R["ledger"], now=1.0 + 30 * 86400) == pytest.approx(80 * 0.07)


def test_a_data_centre_given_by_hand_with_only_high_performance_storage_gets_that_tier(monkeypatch, tmp_path):
    f = Fake(monkeypatch, {("GET", "/v2/catalog/datacenters"): lambda q, b: (200, TIERS), ("POST", "/v2/network-volumes"): volume_made})
    R = fresh_state()
    t, st = transport(monkeypatch, tmp_path, R, dc="CA-MTL-1")
    dc, vol = t._volume_for_create(st)
    assert (dc, vol["type"]) == ("CA-MTL-1", "HIGH_PERFORMANCE")
    assert [c[3]["type"] for c in f.calls if c[:2] == ("POST", "/v2/network-volumes")] == ["HIGH_PERFORMANCE"]


def test_a_data_centre_that_loses_its_tier_before_the_volume_is_asked_for_is_an_error_not_a_guess(monkeypatch, tmp_path):
    import runpod_transport as rt
    gone = dcs(("CA-MTL-1", [], "HIGH"))
    Fake(monkeypatch, {("GET", "/v2/catalog/datacenters"): lambda q, b: (200, {"dataCenters": gone}),
                       ("POST", "/v2/network-volumes"): lambda q, b: pytest.fail("a volume was asked for")})
    monkeypatch.setattr(rt.RunPodTransport, "_stocked", lambda self: ["CA-MTL-1"])              # the stock was read a moment earlier
    R = fresh_state()
    t, st = transport(monkeypatch, tmp_path, R)
    with pytest.raises(api.RunPodError, match="no longer offers a network volume") as e:
        t._volume_for_create(st)
    assert not isinstance(e.value, api.Rejected) and R["volume"] is None and R["ledger"]["volumes"] == []
