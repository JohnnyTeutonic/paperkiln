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
    assert api.VOLUME_TYPE == "STANDARD"                                            # the tier the ledger's rate is for, asked by name
    assert v["id"] == "vol0000001" and v["dataCenterId"] == "EU-SE-1"
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
