"""Volume preservation using only fake APIs; no credentials or provider calls."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/colab_chat"))
import runpod_transport as rt  # noqa: E402


@pytest.fixture
def setup(tmp_path, monkeypatch):
    transport = rt.RunPodTransport("L4", "SECURE", 10, None, None, None, "c", "test", str(tmp_path), lambda m: None)
    volume = {"id": "old", "dc": "A", "gb": 10, "created": 1}
    state = SimpleNamespace(d={"jobs": {"score": {"home": False}}, "runpod": {
        "volume": volume, "pod": None, "retired_volumes": [],
        "ledger": {"pods": [{"id": "previous", "volume": "old", "stop": 2}],
                   "volumes": [copy.deepcopy(volume)]}}}, save=lambda: None)
    transport.st = state
    calls = {"deleted": [], "created": []}
    monkeypatch.setattr(rt.api, "delete_volume", lambda i: calls["deleted"].append(i) or True)
    monkeypatch.setattr(rt.api, "create_volume", lambda name, gb, dc: calls["created"].append(dc) or {"id": "new"})
    monkeypatch.setattr(transport, "_stocked", lambda: ["B"])
    return transport, state, calls


@pytest.mark.parametrize("fails", [0, 2, 100])
def test_used_volume_never_migrates_for_stock_or_create_failures(setup, fails):
    transport, state, calls = setup
    state.d["runpod"]["dc_fails"] = fails
    assert transport._volume_for_create(state) == ("A", state.d["runpod"]["volume"])
    assert not calls["deleted"] and not calls["created"]
    assert not state.d["runpod"]["retired_volumes"]


def test_used_volume_explicit_dc_switch_refused_without_state_changes(setup):
    transport, state, calls = setup
    transport.dc = "B"
    before = copy.deepcopy(state.d)
    with pytest.raises(rt.api.Refused, match="operator-verified migration"):
        transport._volume_for_create(state)
    assert state.d == before and not calls["deleted"] and not calls["created"]


def test_unused_volume_can_move_and_delete(setup):
    transport, state, calls = setup
    state.d["runpod"]["ledger"]["pods"] = []
    assert transport._volume_for_create(state)[0] == "B"
    assert calls == {"deleted": ["old"], "created": ["B"]}


def test_legacy_unbound_pod_is_treated_as_used(setup):
    transport, state, calls = setup
    state.d["runpod"]["ledger"]["pods"] = [{"id": "legacy"}]
    assert transport._volume_for_create(state)[0] == "A"
    assert not calls["deleted"] and not calls["created"]


def successful_create(setup, monkeypatch, tmp_path):
    transport, state, calls = setup
    public = tmp_path / "key.pub"
    public.write_text("fake-public-key")
    monkeypatch.setattr(rt, "KEY", str(tmp_path / "key"))
    monkeypatch.setattr(rt, "image_digest", lambda image: "fake")
    monkeypatch.setattr(transport, "over_budget", lambda st: None)
    monkeypatch.setattr(rt.api, "create_pod", lambda body: {"id": "replacement", "costPerHr": 0.59})
    monkeypatch.setattr(transport, "_wait_up", lambda pod: {"id": pod})
    monkeypatch.setattr(transport, "_prelude", lambda *args: (True, ""))
    return transport, state, calls


def test_replacement_prelude_never_deletes_used_retired_progress(setup, monkeypatch, tmp_path):
    transport, state, calls = successful_create(setup, monkeypatch, tmp_path)
    retired = {"id": "retired", "dc": "B"}
    state.d["runpod"]["retired_volumes"] = [retired]
    state.d["runpod"]["ledger"]["pods"].append({"id": "prior", "volume": "retired"})
    assert transport.create(state) == (0, "replacement")
    assert not calls["deleted"]
    assert state.d["runpod"]["retired_volumes"] == [retired]


def test_failed_provider_creates_keep_used_volume_on_same_dc(setup, monkeypatch, tmp_path):
    transport, state, calls = successful_create(setup, monkeypatch, tmp_path)
    attempted = []
    def no_stock(body):
        attempted.append((body["dataCenterIds"], body["networkVolumeId"]))
        raise rt.api.RunPodError("no stock")
    monkeypatch.setattr(rt.api, "create_pod", no_stock)
    for _ in range(4):
        assert transport.create(state)[0] == 1
    assert attempted == [(["A"], "old")] * 4
    assert not calls["deleted"] and not calls["created"]


def test_release_requires_all_home_and_no_pod_then_cleans_used_retired(setup):
    transport, state, calls = setup
    state.d["runpod"]["retired_volumes"] = [{"id": "retired", "dc": "B"}]
    with pytest.raises(rt.api.Refused, match="all campaign jobs home"):
        transport.release(state)
    state.d["campaign_home"] = True
    with pytest.raises(rt.api.Refused):
        transport.release(state)
    state.d["jobs"]["score"]["home"] = True
    state.d["runpod"]["pod"] = {"id": "active"}
    with pytest.raises(rt.api.Refused):
        transport.release(state)
    state.d["runpod"]["pod"] = None
    transport.release(state)
    assert calls["deleted"] == ["old", "retired"]
    assert state.d["runpod"]["volume"] is None
    assert state.d["runpod"]["retired_volumes"] == []
