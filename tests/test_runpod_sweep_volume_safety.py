"""Scheduled sweeper cannot equate retired/dead-owner status with completed relay."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/colab_chat"))
import runpod_sweep as sweep  # noqa: E402


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    state = {"campaign_home": False, "endpoint": None, "jobs": {"scoring": {"home": False}},
             "runpod": {"pod": None, "volume": {"id": "replacement"},
                        "retired_volumes": [{"id": "checkpoint-volume"}],
                        "ledger": {"pods": [{"id": "old-pod", "volume": "checkpoint-volume", "stop": 1}],
                                   "volumes": [{"id": "checkpoint-volume"}]}}}
    directory = tmp_path / "runner"
    directory.mkdir()
    state_path = directory / "state.json"
    state_path.write_text(json.dumps(state))
    (directory / "runner.log").write_text("all jobs finished and home\n")
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({str(directory): {"name": "runner"}}))
    key = tmp_path / "fake-key"
    key.write_text("not-a-credential")
    monkeypatch.setattr(sweep, "REGISTRY", str(registry))
    monkeypatch.setattr(sweep, "KEEP", str(tmp_path / "missing-keep"))
    monkeypatch.setattr(sweep.api, "KEY_PATH", str(key))
    monkeypatch.setattr(sweep, "runner_alive", lambda d: False)
    monkeypatch.setattr(sweep.api, "list_pods", lambda: [])
    monkeypatch.setattr(sweep.api, "list_volumes", lambda: [{"id": "checkpoint-volume"}])
    deleted = []
    monkeypatch.setattr(sweep.api, "delete_volume", lambda vid: deleted.append(vid) or True)
    return state, state_path, deleted


def test_dead_owner_retired_unfinished_volume_kept_despite_completion_log(campaign):
    state, path, deleted = campaign
    before = path.read_bytes()
    assert sweep.main([]) == 0
    assert deleted == [] and path.read_bytes() == before


@pytest.mark.parametrize("bad", ["flag", "job", "empty_jobs", "legacy_jobs", "pod", "endpoint", "creating"])
def test_any_missing_completion_proof_preserves_retired_volume(campaign, bad):
    state, path, deleted = campaign
    state["campaign_home"] = True
    state["jobs"]["scoring"]["home"] = True
    if bad == "flag":
        state.pop("campaign_home")
    elif bad == "job":
        state["jobs"]["scoring"]["home"] = False
    elif bad == "empty_jobs":
        state["jobs"] = {}
    elif bad == "legacy_jobs":
        state.pop("jobs")
    elif bad == "pod":
        state["runpod"]["pod"] = {"id": "active"}
    elif bad == "endpoint":
        state["endpoint"] = "active"
    else:
        state["runpod"]["creating"] = 1
    path.write_text(json.dumps(state))
    assert sweep.main([]) == 0 and deleted == []


def test_finished_all_home_retired_volume_cleanup_and_dry_run(campaign):
    state, path, deleted = campaign
    state["campaign_home"] = True
    state["jobs"]["scoring"]["home"] = True
    path.write_text(json.dumps(state))
    assert sweep.main(["--dry-run"]) == 0 and deleted == []
    assert sweep.main([]) == 0 and deleted == ["checkpoint-volume"]
    after = json.loads(path.read_text())
    assert after["runpod"]["retired_volumes"] == []
    assert after["runpod"]["ledger"]["volumes"][0]["deleted"]


def test_latest_unfinished_state_overrules_initial_finished_snapshot(campaign, monkeypatch):
    state, path, deleted = campaign
    state["campaign_home"] = True
    state["jobs"]["scoring"]["home"] = True
    path.write_text(json.dumps(state))
    def volumes():
        updated = copy.deepcopy(state)
        updated["jobs"]["scoring"]["home"] = False
        path.write_text(json.dumps(updated))
        return [{"id": "checkpoint-volume"}]
    monkeypatch.setattr(sweep.api, "list_volumes", volumes)
    assert sweep.main([]) == 0 and deleted == []


def test_provider_still_mounting_volume_blocks_cleanup(campaign, monkeypatch):
    state, path, deleted = campaign
    state["campaign_home"] = True
    state["jobs"]["scoring"]["home"] = True
    path.write_text(json.dumps(state))
    monkeypatch.setattr(sweep.api, "list_pods", lambda: [{"id": "manual", "name": "manual", "networkVolumeId": "checkpoint-volume"}])
    monkeypatch.setattr(sweep.api, "terminate_pod", lambda pid: False)
    assert sweep.main([]) == 0 and deleted == []
