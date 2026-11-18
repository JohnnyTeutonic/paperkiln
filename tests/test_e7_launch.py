"""E7's start gate (e7_launch.py): one arm after the token ladder, never beside E9.

Claude Opus 5.5 (a fork of the coordinating session), 19 Nov 2026.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CC = ROOT / "tools" / "colab_chat"
sys.path.insert(0, str(CC))
import e7_launch as EL   # noqa: E402

E9_BASE = "6f5fa128"


def git_show(rel):
    return subprocess.run(["git", "show", f"{E9_BASE}:{rel}"], cwd=ROOT, check=True, capture_output=True).stdout


def test_ladder_e9_and_base2_files_are_unchanged():
    for rel in ("tools/colab_chat/ladder_launch.py", "tools/colab_chat/e9_launch.py", "tools/colab_chat/make_ladder.py",
                "tools/colab_chat/make_base2.py", "tools/colab_chat/base2_pins.py", "tools/colab_chat/base2_pins.json",
                "tools/colab_chat/runner_multi.py", "tools/colab_chat/ladder_N1B.json",
                "tools/colab_chat/base2_rung_D1.json", "tools/colab_chat/base2_units_guard.py"):
        assert (ROOT / rel).read_bytes() == git_show(rel), rel


def test_arm_jobs_hours_cap_and_guard():
    assert list(EL.ARMS) == ["pk-e7-E7A"]
    a = EL.ARMS["pk-e7-E7A"]
    assert (ROOT / a["jobs"]).exists()
    assert EL.ARM_WORST_UNITS < EL.GUARD_STOP_AT < EL.CAP_UNITS
    assert EL.GUARD_STOP_AT == 52.0 and EL.CAP_UNITS == 72.0
    EL.install()
    argv = EL.runner_argv("pk-e7-E7A", "a" * 40, "2026-12-05T00:00:00Z")
    assert "tools/colab_chat/e7_E7A.json" in argv and not any("ladder_" in x for x in argv)
    assert argv[argv.index("--max-hours") + 1] == "26"
    assert argv[argv.index("--deadline-utc") + 1] == "2026-12-05T00:00:00Z"
    assert argv[argv.index("--store-campaign") + 1] == "e7-E7A"
    assert EL.LL.LAUNCH_DIR == "/home/jonat/e7_launch" and EL.LL.GUARD["campaign"] == "e7-shape"
    assert EL.LL.WORST_HOURS == 26 and EL.LL.OWN_COPY == "e7_launch.json"


def _arms_in(tmp):
    d = tmp / "pk-e7-E7A"
    d.mkdir()
    return {"pk-e7-E7A": dict(EL.ARMS["pk-e7-E7A"], state_dir=str(d))}


def _k(arms, tmp, ladder="ended", e9=None, monkey=None):
    ld = tmp / "launch"
    ld.mkdir(exist_ok=True)
    ended = (lambda *a, **k: None) if ladder == "ended" else (lambda *a, **k: ladder)
    return EL.decide_arm("pk-e7-E7A", "a" * 40, 0, None, None, None, launch_dir=str(ld), arms=arms, cmdlines=[],
                         ladder_ended=ended, e9=lambda: e9)


def test_waits_while_the_ladder_has_not_ended(tmp_path, monkeypatch):
    monkeypatch.setattr(EL, "_ORIG_DECIDE", lambda *a, **k: ("wait", "delegated"))
    kind, why = _k(_arms_in(tmp_path), tmp_path, ladder="a ladder arm has not stopped")
    assert kind == "wait" and "has not ended" in why


def test_halts_if_e9_is_live(tmp_path, monkeypatch):
    monkeypatch.setattr(EL, "_ORIG_DECIDE", lambda *a, **k: ("wait", "delegated"))
    kind, what = _k(_arms_in(tmp_path), tmp_path, e9="E9 has been admitted (its launch.json exists)")
    assert kind == "halt" and what[0] == "E7_AFTER_E9"


def test_delegates_to_the_ladder_gate_when_clear(tmp_path, monkeypatch):
    seen = {}

    def fake(*a, **k):
        seen.update(k)
        return ("wait", "delegated")
    monkeypatch.setattr(EL, "_ORIG_DECIDE", fake)
    assert _k(_arms_in(tmp_path), tmp_path) == ("wait", "delegated")
    assert seen["launch_dir"].endswith("launch")


def test_a_started_arm_skips_the_first_start_checks(tmp_path, monkeypatch):
    monkeypatch.setattr(EL, "_ORIG_DECIDE", lambda *a, **k: ("exec", ["runner"]))
    arms = _arms_in(tmp_path)
    (Path(arms["pk-e7-E7A"]["state_dir"]) / EL.OWN_COPY).write_text("{}")
    # once started, the ladder's own restart rules apply, even if E9 or the ladder's state has moved since
    assert _k(arms, tmp_path, ladder="not ended", e9="E9 live") == ("exec", ["runner"])


def test_cap_refuses_a_second_start(tmp_path, monkeypatch):
    monkeypatch.setattr(EL, "_ORIG_DECIDE", lambda *a, **k: ("wait", "delegated"))
    arms = _arms_in(tmp_path)
    arms["pk-e7-second"] = dict(arms["pk-e7-E7A"], state_dir=str(tmp_path / "second"))
    (tmp_path / "second").mkdir()
    (Path(arms["pk-e7-second"]["state_dir"]) / EL.OWN_COPY).write_text("{}")
    assert abs(EL.committed_units(arms) - EL.ARM_WORST_UNITS) < 1e-9
    kind, what = _k({"pk-e7-E7A": arms["pk-e7-E7A"], "pk-e7-second": arms["pk-e7-second"]}, tmp_path)
    assert kind == "halt" and what[0] == "E7_CAP"


def test_e9_live_reads_admission_arms_and_builds(tmp_path):
    e9 = tmp_path / "e9"
    e9.mkdir()
    arm = tmp_path / "arm"
    arm.mkdir()
    build = tmp_path / "build"
    build.mkdir()
    arms = {"pk-e9-D2": {"state_dir": str(arm)}}
    builds = {"pk-e9-data-D2": {"state_dir": str(build)}}
    assert EL.e9_live(str(e9), arms, builds) is None
    (build / EL.E9.BUILD_COPY).write_text("{}")
    assert "build" in EL.e9_live(str(e9), arms, builds)
    (build / EL.E9.BUILD_COPY).unlink()
    (arm / EL.E9.OWN_COPY).write_text("{}")
    assert "arm" in EL.e9_live(str(e9), arms, builds)
    (arm / EL.E9.OWN_COPY).unlink()
    (e9 / "launch.json").write_text(json.dumps({}))
    assert "admitted" in EL.e9_live(str(e9), arms, builds)


def test_units_needed_counts_the_one_arm(tmp_path):
    need = EL.units_needed(None, _arms_in(tmp_path), 0.0, None)
    assert abs(need - (26 * 1.94 + EL.LL.UNITS_MARGIN)) < 1e-6, need
