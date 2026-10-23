"""The MoR block's specs and jobs files (PREREGISTRATION.md amendment 4; make_jobs.py's mor_* functions):
the cells of A4.3, the Stage 0 of A4.4 and the E files' gates. Pure file checks (no binary)."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_jobs", ROOT / "experiments" / "tiny_chat_2026-10" / "make_jobs.py")
mj = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mj)
COLAB = ROOT / "tools" / "colab_chat"
SPECS = ROOT / "experiments" / "tiny_chat_2026-10" / "recursion" / "specs"
CELLS = {"20m_MoR3": ((1, 2, 3, 1), "mor", 8, 6), "20m_MC2": ((1, 2, 2, 1), "fixed", 6, 6),
         "20m_MC3": ((1, 2, 3, 1), "fixed", 8, 8)}
SETTINGS = {"route": "token_choice", "gate": "score", "balance": 0.1, "z_loss": 0.001, "kv_rows": "active"}


def test_mor_cells_match_a4_3():
    ms = mj.mor_shapes(mj.shapes())
    assert set(ms) == set(CELLS)
    for sid, ((P, B, R, C), sched, eff, mean) in CELLS.items():
        s = ms[sid]
        assert (s["d"], s["heads"], s["d_ff"], s["layers"]) == (640, 10, 1760, 4)
        assert s["recursion"] == {"prelude": P, "block": B, "repeats": R, "coda": C}
        assert s["eff_layers"] == eff and s["mean_layers"] == mean
        sp = json.loads((SPECS / f"shape_{sid}.json").read_text())
        rec = sp["arch"]["custom"]["recursion"]
        assert rec["schedule"] == sched and rec["inject"] == "none"
        assert (rec.get("mor") == SETTINGS) == (sched == "mor")
        assert sp["train"]["batch"] == 32 and sp["train"]["accum"] == 1  # effective depth at most 8
        assert sp["_shape"]["effective_layers"] == eff
    assert ms["20m_MoR3"]["router_params"] == 3 * 640


def test_s0_mor_cuda_gate_probes_and_screens():
    doc = json.loads((COLAB / "tiny_chat_s0_mor.json").read_text())
    assert doc["max_parallel"] == 1
    js = doc["jobs"]
    assert [j["name"] for j in js[:3]] == ["cuda_gate", "gate", "data"]
    g = js[1]["cmd"]
    assert g[g.index("--jobs") + 1] == "tools/colab_chat/tiny_chat_s0_mor.json"
    assert g[g.index("--need-pilot") + 1] == "PASS"
    probes = [j for j in js if j["name"].startswith("probe_")]
    screens = [j for j in js if j["name"].startswith("screen_")]
    assert sorted(j["name"] for j in probes) == sorted(f"probe_{k}" for k in CELLS)
    assert len(screens) == 12
    for sid in CELLS:
        lrs = sorted(float(j["cmd"][j["cmd"].index("--lr") + 1]) for j in screens if j["name"].startswith(f"screen_{sid}_"))
        assert lrs == [2.5e-4, 5e-4, 1e-3, 2e-3]
    for j in probes + screens:
        assert j["after"] == ["data", "cuda_gate"] and "--cuda-gate" in j["cmd"]
    for j in screens:
        assert j["cmd"][j["cmd"].index("--seed") + 1] == "101"
        assert j["cmd"][j["cmd"].index("--steps") + 1] == "1500"


def test_e1_mor_file():
    doc = json.loads((COLAB / "tiny_chat_e1_mor.json").read_text())
    pts = [j for j in doc["jobs"] if j["name"].startswith("pt_")]
    assert sorted(j["name"] for j in pts) == sorted(f"pt_e1_mor_{c}_r{s}" for c in ("MoR3", "MC2", "MC3")
                                                    for s in (1, 2, 3))
    for j in pts:
        assert j["snapshot_steps"] == [500, 1000, 2500, 5000]
        rec = j["spec"]["arch"]["custom"]["recursion"]
        assert rec["schedule"] == ("mor" if "MoR3" in j["name"] else "fixed")
    g = next(j for j in doc["jobs"] if j["name"] == "gate")["cmd"]
    assert g[g.index("--choice") + 1] == mj.MOR_CHOICE
    assert g[g.index("--need-cuda-gate") + 1] == mj.MOR_CUDA_GATE
    assert g[g.index("--need-pilot") + 1] == "PASS"


def test_budget_mor_states_the_80_hour_rule():
    t = mj.budget_mor(mj.shapes())
    assert "MoR block" in t and "Jonathan" in t
