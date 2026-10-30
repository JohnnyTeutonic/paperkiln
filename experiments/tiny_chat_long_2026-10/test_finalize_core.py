"""Filesystem-only Carry-12 completion gates; no GPU/provider calls."""
import importlib.util
import json
from pathlib import Path

import pytest


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    out = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(out)
    return out


HERE = Path(__file__).parent
f = module("carry_finalizer", HERE / "finalize_core.py")
fixture = module("carry_fixtures", HERE.parents[1] / "tests/test_carry12_analyze.py")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def ready(tmp_path):
    art = tmp_path / "art"
    fixture.core(art)
    for path in art.glob("*/post_*/summary.json"):
        row = f.load_json(path)
        row["mode"] = "post"
        write(path, row)
    ledger = {"args": {"out": "/one"}, "content_sha": "fixed", "rows": 100}
    for seed in f.SEEDS:
        for layout in f.LAYOUTS:
            block = art / f"c12_{layout}_r{seed}"
            write(block / "data/data.ledger.json", ledger)
            write(block / "gate/gate.json", {"ok": True, "problems": []})
        write(art / f"c12_B_r{seed}/data512/data.ledger.json", ledger)
        write(art / f"e1_60m_L8/post_e1_60m_L8_r{seed}/summary.json", {"sft_weights_sha256": "s"})
    write(art / "e1_60m_L8/data/data.ledger.json", ledger)
    return f.Finalizer(art, tmp_path / "state")


def analyze(staging, d):
    # Calls the unchanged registered analysis, using only synthetic inputs.
    m = f.load_module("registered_fixture_analysis", HERE / "analyze.py")
    result = {"partial": False, "carry12": m.analyze(m.load(str(d.art)), e1_home=str(d.art / "e1_60m_L8"))}
    write(staging / "analysis_carry12.json", result)
    write(staging / "commands.json", ["SYNTHETIC_OFFLINE_TEST"])


def test_complete_atomic_restart_and_marker(ready, monkeypatch):
    calls = []
    monkeypatch.setattr(ready, "run_analyses", lambda p: (calls.append(1), analyze(p, ready)))
    assert ready.tick() == "COMPLETE"
    receipt = f.load_json(ready.output / "reports/completion.json")
    assert len(receipt["input_manifest"]["selected_cell_seed_pairs"]) == 21
    assert ready.tick() == "COMPLETE" and calls == [1]
    assert "all jobs finished and home" in (ready.home / "runner.log").read_text()


def test_pending_and_observed_input_freeze(ready, monkeypatch):
    path = ready.art / "c12_X_r3/post_c12_X_r3/summary.json"
    saved = path.read_bytes()
    path.unlink()
    monkeypatch.setattr(ready, "run_analyses", lambda _: pytest.fail("must wait"))
    assert ready.tick() == "WAITING"
    observed = ready.art / "c12_B_r1/post_c12_B_r1/summary.json"
    row = f.load_json(observed)
    row["Y12"] += .01
    write(observed, row)
    path.write_bytes(saved)
    assert ready.tick() == "HALTED"
    assert "previously observed" in ready.state()["error"]


@pytest.mark.parametrize("change", ["extra", "seed", "nan", "ledger", "gate"])
def test_mismatches_halt_without_publication(ready, monkeypatch, change):
    path = ready.art / "c12_X_r3/post_c12_X_r3/summary.json"
    if change == "extra":
        write(ready.art / "other/post_c12_X_r3/summary.json", f.load_json(path))
    elif change in ("seed", "nan"):
        row = f.load_json(path)
        row["seed" if change == "seed" else "Y12"] = 4 if change == "seed" else float("nan")
        write(path, row)
    elif change == "ledger":
        write(ready.art / "c12_X_r3/data/data.ledger.json", {"content_sha": "corrupt", "rows": 100})
    else:
        write(ready.art / "c12_X_r3/gate/gate.json", {"ok": False, "problems": ["failed"]})
    monkeypatch.setattr(ready, "run_analyses", lambda _: pytest.fail("must halt"))
    assert ready.tick() == "HALTED" and ready.tick() == "HALTED"
    assert (ready.home / "HALTED").is_file()
    assert not (ready.output / "reports").exists()


@pytest.mark.parametrize("key,value", [("carry12", True), ("template_sft", True), ("long_sft", "10:4")])
def test_nondefault_data_arguments_are_never_normalized(key, value):
    assert f.data_ledger({"args": {key: value}}) != f.data_ledger({"args": {}})


def test_only_operational_and_explicit_noop_defaults_normalize():
    old = {"args": {"out": "/old"}, "sha256": "a", "rows": 3}
    new = {"args": {"out": "/new", "carry12": False, "template_sft": False, "long_sft": ""},
           "sha256": "a", "rows": 3}
    assert f.data_ledger(old) == f.data_ledger(new)
    for key in ("sha256", "rows"):
        changed = {**new, key: "different"}
        assert f.data_ledger(old) != f.data_ledger(changed)


def test_input_mutation_during_analysis_halts(ready, monkeypatch):
    def run(staging):
        analyze(staging, ready)
        write(ready.art / "c12_X_r3/gate/gate.json", {"ok": False, "problems": ["changed"]})
    monkeypatch.setattr(ready, "run_analyses", run)
    assert ready.tick() == "HALTED"
    assert not (ready.output / "reports").exists()


def test_source_mutation_halts(ready, monkeypatch, tmp_path):
    source = tmp_path / "source"
    source.write_text("a")
    monkeypatch.setattr(f, "SOURCE_FILES", (source,))
    ready.state()
    source.write_text("b")
    assert ready.tick() == "HALTED"
    assert "frozen source" in ready.state()["error"]


def test_partial_output_cannot_publish(ready, monkeypatch):
    def run(staging):
        analyze(staging, ready)
        path = staging / "analysis_carry12.json"
        row = f.load_json(path)
        row["partial"] = True
        write(path, row)
    monkeypatch.setattr(ready, "run_analyses", run)
    assert ready.tick() == "HALTED"
    assert not (ready.output / "reports").exists()


@pytest.mark.parametrize("key", ["corpus_path_sha256", "semantic_time_steps", "human_empathetic_v1/dialogues.jsonl",
                               "hosted_corpus_sha256", "date_annotations", "elapsed_training_steps"])
def test_similarly_named_semantic_fields_are_preserved(key):
    assert f.data_ledger({key: "one"}) != f.data_ledger({key: "two"})
