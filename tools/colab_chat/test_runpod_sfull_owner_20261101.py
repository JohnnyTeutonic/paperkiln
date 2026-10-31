"""Independent read-only owner-recognition tests; all provider operations absent."""

import builtins
import importlib.util
import io
import json
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/colab_chat"))
spec = importlib.util.spec_from_file_location(
    "sweep_review", ROOT / "tools/colab_chat/runpod_sweep.py"
)
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
STATE = Path("/home/jonat/pk_sfull_data_only_1031_runner")
CONTROLLER = Path(
    "/home/jonat/pk_sfull_data_controller_20261031/tools/colab_chat/sfull_data_only_controller_20261031.py"
)
MANIFEST = Path(
    "/home/jonat/pk_sfull_data_controller_20261031/docs/receipts/sfull_data_only_admission_20261031/manifest.json"
)
SHA = "3640b1595afb5847e66081b7a6fc10766caed7c4fa5b9cb61b2e3957fb1b4ea9"
BASE = [
    "python3",
    str(CONTROLLER),
    "--manifest",
    str(MANIFEST),
    "--manifest-sha256",
    SHA,
    "--admission",
    str(STATE / "admission.json"),
]


@pytest.fixture(autouse=True)
def synthetic_known_controller(tmp_path, monkeypatch):
    """Portable outer-owner contract; no account, real campaign or provider needed.

    The source-pinned synthetic controller checks inventory and anchored launch
    fields. The full frozen controller/real-state matrix is a separate receipt.
    """
    import hashlib

    script = tmp_path / "repo/tools/colab_chat/sfull_controller.py"
    script.parent.mkdir(parents=True)
    controller_code = """
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

def verify_manifest(path, expected, root):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
        raise ValueError('manifest drift')
    m=json.loads(Path(path).read_text())
    if Path(m['controller_root']) != root:
        raise ValueError('root drift')
    for name, wanted in m['source_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != wanted:
            raise ValueError('inventory drift')
    subprocess.check_output(['synthetic-original-tree-check'])
    return m

def timestamp(value):
    return datetime.datetime.fromisoformat(value.replace('Z','+00:00')).timestamp()

def validate_admission(m, expected, path, now, already_admitted):
    a=json.loads(Path(path).read_text())
    if a['authorisation'] != 'ROOT_APPROVED_SFULL_DATA_ONLY' or a['manifest_sha256'] != expected:
        raise ValueError('unapproved launch')
    if not timestamp(a['admitted_at_utc']) <= now < timestamp(a['deadline_utc']):
        raise ValueError('invalid launch time')
    if not already_admitted:
        raise ValueError('must revalidate anchored launch')
    return a
"""
    script.write_text(controller_code)
    transport = script.with_name("runpod_transport.py")
    transport.write_text("# synthetic inventory dependency\n")
    state = tmp_path / "campaign"
    state.mkdir()
    manifest = tmp_path / "manifest.json"

    def h(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    manifest.write_text(
        json.dumps(
            {
                "controller_root": str(script.parents[2]),
                "state_dir": str(state),
                "source_sha256": {
                    str(transport.relative_to(script.parents[2])): h(transport)
                },
            }
        )
    )
    wanted = h(manifest)
    admission = state / "admission.json"
    admission.write_text(
        json.dumps(
            {
                "authorisation": "ROOT_APPROVED_SFULL_DATA_ONLY",
                "manifest_sha256": wanted,
                "admitted_at_utc": "2026-10-31T13:00:00Z",
                "deadline_utc": "2026-10-31T14:15:00Z",
            }
        )
    )
    (state / "admission_anchor.json").write_text(
        json.dumps({"manifest_sha256": wanted, "admission_sha256": h(admission)})
    )
    monkeypatch.setattr(
        s,
        "ADMITTED_CONTROLLERS",
        {
            str(script): {
                "sha256": h(script),
                "manifest": str(manifest),
                "manifest_sha256": wanted,
            }
        },
    )
    monkeypatch.setitem(globals(), "STATE", state)
    monkeypatch.setitem(globals(), "CONTROLLER", script)
    monkeypatch.setitem(globals(), "MANIFEST", manifest)
    monkeypatch.setitem(globals(), "SHA", wanted)
    monkeypatch.setitem(
        globals(),
        "BASE",
        [
            "python3",
            str(script),
            "--manifest",
            str(manifest),
            "--manifest-sha256",
            wanted,
            "--admission",
            str(admission),
        ],
    )
    import subprocess

    monkeypatch.setattr(
        subprocess, "check_output", lambda *a, **k: b"synthetic tree unchanged"
    )


def fake(monkeypatch, argv=None, overrides=None, dead=False):
    argv = BASE if argv is None else argv
    overrides = dict(overrides or {})
    pid = os.getpid()
    overrides[str(STATE / "runner.pid")] = str(pid).encode()
    overrides[f"/proc/{pid}/cmdline"] = (
        None if dead else ("\0".join(argv) + "\0").encode()
    )
    oldopen = builtins.open
    oldio = io.open

    def read(path, mode="r", *args, **kwargs):
        try:
            name = os.fspath(path)
        except TypeError:
            return oldopen(path, mode, *args, **kwargs)
        if name in overrides:
            data = overrides[name]
            if data is None:
                raise FileNotFoundError(name)
            return io.BytesIO(data) if "b" in mode else io.StringIO(data.decode())
        return oldopen(path, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", read)

    def ioread(path, mode="r", *args, **kwargs):
        try:
            name = os.fspath(path)
        except TypeError:
            return oldio(path, mode, *args, **kwargs)
        if name in overrides:
            return read(path, mode, *args, **kwargs)
        return oldio(path, mode, *args, **kwargs)

    monkeypatch.setattr(io, "open", ioread)


def test_exact_live_frozen_controller_is_owner(monkeypatch):
    fake(monkeypatch)
    assert s.runner_alive(str(STATE))


@pytest.mark.parametrize(
    "kind",
    [
        "dead",
        "unknown",
        "same_basename_elsewhere",
        "script_as_data",
        "wrong_manifest_arg",
        "wrong_manifest_sha",
        "wrong_admission_arg",
        "duplicate_manifest",
        "extra_arg",
    ],
)
def test_wrong_process_or_argv_is_not_owner(monkeypatch, kind):
    argv = list(BASE)
    if kind == "unknown":
        argv[1] = "/tmp/unknown_controller.py"
    if kind == "same_basename_elsewhere":
        argv[1] = "/tmp/" + CONTROLLER.name
    if kind == "script_as_data":
        argv = ["python3", "/tmp/unknown.py"] + argv[1:]
    if kind == "wrong_manifest_arg":
        argv[3] = "/tmp/manifest.json"
    if kind == "wrong_manifest_sha":
        argv[5] = "0" * 64
    if kind == "wrong_admission_arg":
        argv[7] = "/tmp/admission.json"
    if kind == "duplicate_manifest":
        argv += ["--manifest", str(MANIFEST)]
    if kind == "extra_arg":
        argv += ["--unreviewed"]
    fake(monkeypatch, argv, dead=kind == "dead")
    assert not s.runner_alive(str(STATE))


@pytest.mark.parametrize(
    "kind",
    [
        "controller_source",
        "manifest_source",
        "inventory_source",
        "anchor_manifest",
        "anchor_admission",
        "admission_source",
    ],
)
def test_frozen_binding_drift_is_not_owner(monkeypatch, kind):
    overrides = {}
    if kind == "controller_source":
        overrides[str(CONTROLLER)] = CONTROLLER.read_bytes() + b"\n# drift\n"
    if kind == "manifest_source":
        overrides[str(MANIFEST)] = MANIFEST.read_bytes() + b"\n"
    if kind == "inventory_source":
        p = CONTROLLER.with_name("runpod_transport.py")
        overrides[str(p)] = p.read_bytes() + b"\n# drift\n"
    if kind.startswith("anchor_"):
        p = STATE / "admission_anchor.json"
        v = json.loads(p.read_text())
        v["manifest_sha256" if kind == "anchor_manifest" else "admission_sha256"] = (
            "0" * 64
        )
        overrides[str(p)] = json.dumps(v).encode()
    if kind == "admission_source":
        p = STATE / "admission.json"
        v = json.loads(p.read_text())
        v["manifest_sha256"] = "0" * 64
        overrides[str(p)] = json.dumps(v).encode()
    fake(monkeypatch, overrides=overrides)
    assert not s.runner_alive(str(STATE))


@pytest.mark.parametrize("equals", [False, True])
def test_standard_runner_behavior_preserved(monkeypatch, equals):
    args = ["python3", "/opt/runner_multi.py"] + (
        ["--state-dir=" + str(STATE)] if equals else ["--state-dir", str(STATE)]
    )
    fake(monkeypatch, args)
    assert s.runner_alive(str(STATE))


def test_standard_wrong_state_refused(monkeypatch):
    fake(monkeypatch, ["python3", "/opt/runner_multi.py", "--state-dir", "/tmp/other"])
    assert not s.runner_alive(str(STATE))


def test_zombie_empty_cmdline_is_not_owner(monkeypatch):
    fake(monkeypatch, [])
    assert not s.runner_alive(str(STATE))


def test_extra_empty_actual_argument_is_not_silently_discarded(monkeypatch):
    fake(monkeypatch, BASE + [""])
    assert not s.runner_alive(str(STATE))


def test_live_cleanup_after_deadline_stays_owned(monkeypatch):
    fake(monkeypatch)
    monkeypatch.setattr(s.time, "time", lambda: 1800000000)
    assert s.runner_alive(str(STATE))


def test_missing_anchor_not_owner(monkeypatch):
    fake(monkeypatch, overrides={str(STATE / "admission_anchor.json"): None})
    assert not s.runner_alive(str(STATE))


def test_original_tree_git_check_failure_is_safe_false(monkeypatch):
    import subprocess

    fake(monkeypatch)

    def bad(*args, **kwargs):
        raise subprocess.CalledProcessError(1, ["git"])

    monkeypatch.setattr(subprocess, "check_output", bad)
    assert not s.runner_alive(str(STATE))


def test_same_validated_manifest_cannot_own_a_different_state(monkeypatch):
    import hashlib

    value = json.loads(MANIFEST.read_text())
    value["state_dir"] = str(STATE.parent / "other_campaign")
    MANIFEST.write_text(json.dumps(value))
    wanted = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    s.ADMITTED_CONTROLLERS[str(CONTROLLER)]["manifest_sha256"] = wanted
    admission = STATE / "admission.json"
    a = json.loads(admission.read_text())
    a["manifest_sha256"] = wanted
    admission.write_text(json.dumps(a))
    (STATE / "admission_anchor.json").write_text(
        json.dumps(
            {
                "manifest_sha256": wanted,
                "admission_sha256": hashlib.sha256(admission.read_bytes()).hexdigest(),
            }
        )
    )
    argv = list(BASE)
    argv[5] = wanted
    fake(monkeypatch, argv)
    assert not s.runner_alive(str(STATE))


def test_hash_anchored_but_unapproved_admission_not_owner(monkeypatch):
    import hashlib

    admission = STATE / "admission.json"
    a = json.loads(admission.read_text())
    a["authorisation"] = "UNAPPROVED"
    admission.write_text(json.dumps(a))
    (STATE / "admission_anchor.json").write_text(
        json.dumps(
            {
                "manifest_sha256": SHA,
                "admission_sha256": hashlib.sha256(admission.read_bytes()).hexdigest(),
            }
        )
    )
    fake(monkeypatch)
    assert not s.runner_alive(str(STATE))
