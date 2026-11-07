"""Winner's PT-only graph preserves its recipe and stops with final output home."""
import copy
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/colab_chat"))
import runner_multi as rm


def graph():
    return json.loads((ROOT / "tools/colab_chat/tc_b1_winner_pt_20261108.json").read_text())


def test_only_continuation_graph_changes():
    old = json.loads((ROOT / "tools/colab_chat/tc_b1_r1.json").read_text())
    expected = copy.deepcopy(old)
    expected["_comment"] = graph()["_comment"]
    expected["jobs"] = [j for j in expected["jobs"] if j["name"] not in ("sft_b1_r1", "items", "bench_b1")]
    pt = next(j for j in expected["jobs"] if j["name"] == "pt_b1_r1")
    pt["pause_rungs"].append("pt_b1_r1")
    assert graph() == expected
    assert pt["spec"]["train"]["steps"] == 60000
    assert 60000 not in pt["snapshot_steps"]


def test_final_home_stops_and_keeps_history(tmp_path, monkeypatch):
    monkeypatch.setattr(rm, "HOME", str(tmp_path))
    st = rm.State()
    pt = next(j for j in graph()["jobs"] if j["name"] == "pt_b1_r1")
    for rung in ("rung_b1_s20000", "rung_b1_s30000"):
        (tmp_path / ("RELEASED_" + rung)).write_text("{}")
    old_halt = tmp_path / "HALTED.rung_b1_s20000"
    old_halt.write_text("historical cutoff")
    checkpoint = tmp_path / "optim.safetensors"
    checkpoint.write_bytes(b"retained optimiser")
    st.job("pt_b1_r1").update(status="FINISHED", home=False)
    assert rm.pause_at_rung(pt, st) is None
    assert not (tmp_path / "HALTED").exists()
    st.job("pt_b1_r1")["home"] = True
    assert rm.pause_at_rung(pt, st) == "pt_b1_r1"
    halt = json.loads((tmp_path / "HALTED").read_text())
    assert halt["kind"] == "CUTOFF_REACHED" and halt["rung"] == "pt_b1_r1"
    assert not st.d.get("campaign_home")
    assert old_halt.read_text() == "historical cutoff"
    assert checkpoint.read_bytes() == b"retained optimiser"
    assert rm.pause_at_rung(pt, st) == "pt_b1_r1"
    assert json.loads((tmp_path / "HALTED").read_text()) == halt
