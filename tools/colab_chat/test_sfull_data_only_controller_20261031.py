import datetime as dt
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

p = Path(__file__).with_name("sfull_data_only_controller_20261031.py")
spec = importlib.util.spec_from_file_location("gate", p)
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)


def test_local_fence_does_not_edit_global_source():
    api = SimpleNamespace(MAX_PODS=13)
    assert g.apply_local_fence(api) == 13
    assert api.MAX_PODS == 2


def test_no_overwrite(tmp_path):
    p = tmp_path / "receipt"
    g.save_new(p, {"first": True})
    before = p.read_bytes()
    with pytest.raises(FileExistsError):
        g.save_new(p, {"second": True})
    assert p.read_bytes() == before


def fixture(tmp_path):
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps(
            {"campaign_home": True, "jobs": {"data": {"home": True}}, "runpod": {}}
        )
    )
    now = 1800000000

    def iso(t):
        return dt.datetime.fromtimestamp(t, dt.timezone.utc).isoformat()

    a = {
        "authorisation": "ROOT_APPROVED_SFULL_DATA_ONLY",
        "manifest_sha256": "sha",
        "manual_balance_accessor_checked": True,
        "provider_cleanup_checked": True,
        "published_remote_commit": g.REMOTE,
        "admitted_at_utc": iso(now),
        "manual_balance_checked_utc": iso(now),
        "deadline_utc": iso(now + 4500),
        "serial_predecessors": [{"path": str(state), "sha256": g.digest(state)}],
    }
    path = tmp_path / "admission.json"
    path.write_text(json.dumps(a))
    return {"required_predecessor_states": [str(state)]}, a, path, now, state


def test_admission_good(tmp_path):
    m, a, p, now, _ = fixture(tmp_path)
    assert g.validate_admission(m, "sha", p, now) == a


@pytest.mark.parametrize(
    "kind", ["stale", "long", "wrong_manifest", "pod", "changed_state"]
)
def test_admission_fail_closed(tmp_path, kind):
    m, a, p, now, state = fixture(tmp_path)
    if kind == "stale":
        now += 301
    elif kind == "long":
        a["deadline_utc"] = dt.datetime.fromtimestamp(
            now + 4501, dt.timezone.utc
        ).isoformat()
    elif kind == "wrong_manifest":
        a["manifest_sha256"] = "wrong"
    else:
        s = json.loads(state.read_text())
        s["runpod"]["pod"] = "live" if kind == "pod" else None
        s["changed"] = True
        state.write_text(json.dumps(s))
        if kind == "pod":
            a["serial_predecessors"][0]["sha256"] = g.digest(state)
    p.write_text(json.dumps(a))
    with pytest.raises(ValueError):
        g.validate_admission(m, "sha", p, now)


def test_failed_original_verifier_never_accepts_home(tmp_path):
    out = tmp_path / "sfull_inputs"
    out.mkdir()
    (out / "verified_inputs.json").write_text(json.dumps({"verified": False}))
    with pytest.raises(ValueError, match="original verifier did not pass"):
        g.validate_relay(tmp_path, {}, lambda x: x)


def test_inventory_drift(tmp_path):
    p = tmp_path / "source.py"
    p.write_text("original")
    inventory = {"source.py": g.digest(p)}
    g.validate_inventory(tmp_path, inventory)
    p.write_text("changed")
    with pytest.raises(ValueError):
        g.validate_inventory(tmp_path, inventory)
