"""Completion-driver tests use only synthetic filesystem reports, never GPU calls."""
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("size_finalizer", Path(__file__).with_name("finalize_extension.py"))
final = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(final)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def ready(tmp_path):
    art = tmp_path / "art"
    cells = (("e5", "e5_s35_r{seed}", "post_e5_s35_r{seed}", "s35"),
             ("e1_60m_L8", "e1_60m_L8_r{seed}", "post_e1_60m_L8_r{seed}", "L8"),
             ("e1_150m_r1", "e1_150m_r{seed}", "post_e1_150m_L8_r{seed}", "L8"))
    for ref, campaign, post, cell in cells:
        write(art / ref / "data/data.ledger.json", {"hash": "fixed", "elapsed": 1})
        for seed in (1, 2, 3, 4, 5):
            folder = campaign.format(seed=seed) if seed > 3 or "150m" in ref else ref
            if seed > 3:
                write(art / folder / "data/data.ledger.json", {"hash": "fixed", "elapsed": 2})
            write(art / folder / post.format(seed=seed) / "summary.json",
                  {"seed": seed, "cell": cell, "primary": -.1})
    return art


def fixture_analysis(staging, seeds=final.SEEDS):
    # Fixture represents only the publication protocol, never a scientific estimate.
    result = {"seeds": seeds, "missing": [], "test_fixture": "NOT_A_SCIENTIFIC_RESULT"}
    write(staging / "analysis_size.json", result)
    write(staging / "analysis_tiny_chat.json", {"partial": False, "E1_150m": result})
    write(staging / "commands.json", ["offline fixture"])


def driver(ready):
    return final.Finalizer(ready, ready.parent / "state")


def test_complete_handoff_and_restart_are_idempotent(ready, monkeypatch):
    d = driver(ready)
    calls = []
    monkeypatch.setattr(d, "run_analyses", lambda staging: (calls.append(staging), fixture_analysis(staging)))
    with final.DriverLock(d.home):
        assert d.tick() == "COMPLETE"
    assert len(calls) == 1
    assert not (d.output / ".incoming").exists()
    result = json.loads((d.output / "reports/completion.json").read_text())
    assert len(result["input_manifest"]["selected_size_seed_pairs"]) == 15
    assert "all jobs finished and home" in (d.home / "runner.log").read_text()
    # Crash after atomic publication, before the completion state was saved.
    s = d.state()
    s["status"] = "ANALYSING"
    final.atomic_json(d.state_path, s)
    assert d.tick() == "COMPLETE"
    assert len(calls) == 1


def test_pending_does_not_merge_or_run_analysis(ready, monkeypatch):
    (ready / "e5_s35_r4/data/data.ledger.json").unlink()
    d = driver(ready)
    monkeypatch.setattr(d, "run_analyses", lambda _: pytest.fail("analysis must not start"))
    assert d.tick() == "WAITING"
    assert not (ready / "e5/post_e5_s35_r4/summary.json").exists()
    assert not (d.output / "reports").exists()


def test_ledger_mismatch_halts_without_analysis(ready, monkeypatch):
    write(ready / "e5_s35_r4/data/data.ledger.json", {"hash": "wrong"})
    d = driver(ready)
    monkeypatch.setattr(d, "run_analyses", lambda _: pytest.fail("analysis must not start"))
    assert d.tick() == "HALTED"
    assert (d.home / "HALTED").exists()
    assert d.tick() == "HALTED"  # Cannot silently clear a failure.


def test_three_seed_output_cannot_publish(ready, monkeypatch):
    d = driver(ready)
    monkeypatch.setattr(d, "run_analyses", lambda p: fixture_analysis(p, [1, 2, 3]))
    assert d.tick() == "HALTED"
    assert not (d.output / "reports").exists()
    assert "all jobs finished and home" not in (d.home / "runner.log").read_text()


def test_missing_initial_seed_halts_even_when_six_extensions_are_ready(ready, monkeypatch):
    (ready / "e5/post_e5_s35_r2/summary.json").unlink()
    d = driver(ready)
    monkeypatch.setattr(d, "run_analyses", lambda _: pytest.fail("analysis must not start"))
    assert d.tick() == "HALTED"
    assert "exactly five" in d.state()["error"]


def test_conflicting_duplicate_in_same_block_is_rejected(ready, monkeypatch):
    write(ready / "e5/post_duplicate/summary.json", {"seed": 1, "cell": "s35", "primary": .9})
    d = driver(ready)
    monkeypatch.setattr(d, "run_analyses", lambda _: pytest.fail("analysis must not start"))
    assert d.tick() == "HALTED"
    assert "duplicate" in d.state()["error"]


def test_source_freeze_is_checked_before_analysis(ready, monkeypatch):
    source = ready.parent / "frozen_helper.py"
    source.write_text("original")
    monkeypatch.setattr(final, "SOURCE_FILES", [source])
    d = driver(ready)
    d.state()
    source.write_text("changed")
    monkeypatch.setattr(d, "run_analyses", lambda _: pytest.fail("analysis must not start"))
    assert d.tick() == "HALTED"
    assert "source/configuration changed" in d.state()["error"]


def test_inputs_changing_during_analysis_cannot_publish(ready, monkeypatch):
    d = driver(ready)
    def changing(staging):
        fixture_analysis(staging)
        write(ready / "e1_150m_r2/post_e1_150m_L8_r2/summary.json", {"seed": 2, "cell": "L8", "primary": .9})
    monkeypatch.setattr(d, "run_analyses", changing)
    assert d.tick() == "HALTED"
    assert not (d.output / "reports").exists()


def test_main_and_standalone_size_blocks_must_agree(ready, monkeypatch):
    d = driver(ready)
    def mismatch(staging):
        fixture_analysis(staging)
        write(staging / "analysis_tiny_chat.json", {"partial": False, "E1_150m": {"seeds": final.SEEDS}})
    monkeypatch.setattr(d, "run_analyses", mismatch)
    assert d.tick() == "HALTED"


def test_duplicate_driver_cannot_hold_same_state_lock(ready):
    d = driver(ready)
    with final.DriverLock(d.home):
        with pytest.raises(OSError):
            with final.DriverLock(d.home):
                pytest.fail("second lock acquired")
    with final.DriverLock(d.home):
        pass


def test_budget_wrapper_required_for_each_analysis(ready, monkeypatch):
    d = driver(ready)
    staging = d.output / ".incoming"
    staging.mkdir()
    calls = []
    monkeypatch.setattr(final.subprocess, "run", lambda argv, **kwargs: calls.append((argv, kwargs)))
    d.run_analyses(staging)
    assert len(calls) == 2
    for argv, kwargs in calls:
        assert argv[1] == str(final.CPU_BUDGET)
        assert argv[argv.index("--cores") + 1] == "1"
        assert "--partial" not in argv
        assert kwargs["check"] and kwargs["timeout"] == 600


def test_altered_publication_is_rejected_on_restart(ready, monkeypatch):
    d = driver(ready)
    monkeypatch.setattr(d, "run_analyses", fixture_analysis)
    assert d.tick() == "COMPLETE"
    write(d.output / "reports/commands.json", ["tampered"])
    assert d.tick() == "HALTED"


def test_recovery_repairs_missing_watchdog_completion_marker(ready, monkeypatch):
    d = driver(ready)
    monkeypatch.setattr(d, "run_analyses", fixture_analysis)
    assert d.tick() == "COMPLETE"
    (d.home / "runner.log").write_text("interrupted before terminal marker\n")
    assert d.tick() == "COMPLETE"
    assert "all jobs finished and home" in (d.home / "runner.log").read_text()
