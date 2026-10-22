"""experiments/tiny_chat_2026-10/analyze.py on synthetic summaries: the registered contrasts and
decision rules (PREREGISTRATION.md sections 5-6), the floor gate, the data-bar rule, E1's L8 cell
borrowed from E5's 50% cell, and the refusal of incomplete contrasts."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TOOL = [sys.executable, str(ROOT / "experiments" / "tiny_chat_2026-10" / "analyze.py")]


def put(art, block, cell, seed, multi, twin=0.80, chance=0.25, fw=3.5, gold=None):
    d = art / block / f"post_{block}_{cell}_r{seed}"
    d.mkdir(parents=True)
    acc = {"self": multi, "twin": twin, "gold": gold if gold is not None else multi + 0.02, "noctx": 0.25}
    (d / "summary.json").write_text(json.dumps({
        "cell": cell, "seed": seed, "primary": multi - twin, "carry5": {"acc": acc, "chance": chance},
        "panels": {"fineweb": fw, "dialogue": 3.0}, "sft_evals": [[1500, 2.5]]}))


def run(art, *extra, levels=(0, 25, 50, 100), e1=50):
    design = art / "design.json"
    design.write_text(json.dumps({"levels": list(levels), "e1_share": e1, "mode": "shares", "final": True}))
    p = subprocess.run(TOOL + ["--art", str(art), "--design", str(design), *extra], capture_output=True, text=True)
    return p.returncode, (json.loads(p.stdout) if p.returncode == 0 else p.stderr)


def test_e5_and_e1_verdicts(tmp_path):
    art = tmp_path
    noise = {1: 0.0, 2: 0.004, 3: -0.004}
    for r in (1, 2, 3):
        # E5: dialogue share lifts Y by 10 points from 0% to 50%; 100% no better, FineWeb loss worse.
        for cell, m, fw in (("s0", 0.60, 3.50), ("s25", 0.66, 3.52), ("s50", 0.70, 3.55), ("s100", 0.70, 3.90)):
            put(art, "e5", cell, r, m + noise[r], fw=fw)
        # E1: Y rises 2 points per doubling of depth (L8 comes from E5's s50 at 0.70).
        for L, m in ((2, 0.66), (4, 0.68), (16, 0.72), (32, 0.74)):
            put(art, "e1", f"L{L}", r, m + noise[r], fw=3.55)
    rc, res = run(art)
    assert rc == 0, res
    e5 = res["E5"]
    assert e5["H5_primary_YE1_minus_Y0"]["verdict"] == "SUPPORTED"
    assert e5["H5_sat_Ytop_minus_YE1"]["verdict"] in ("EQUIVALENT", "NOT_RESOLVED")
    assert e5["H5_knowledge_fineweb_top_minus_0"]["verdict"] == "SUPPORTED"
    assert e5["data_bar"] == "s50"  # s100 fails the FineWeb rule; s50 is the top Y
    e1 = res["E1_20m"]
    assert abs(e1["H1a_depth_slope_Y"]["mean"] - 0.02) < 1e-9 and e1["H1a_depth_slope_Y"]["verdict"] == "SUPPORTED"
    assert e1["argmax_Y"] == "L32"


def test_floor_and_incomplete(tmp_path):
    for r in (1, 2, 3):
        for cell in ("s0", "s25", "s50", "s100"):
            put(tmp_path, "e5", cell, r, 0.27 + 0.01 * r, twin=0.30, chance=0.25)  # twin within 10 points of chance
    rc, res = run(tmp_path)
    assert rc == 0 and res["E5"]["H5_primary_YE1_minus_Y0"]["verdict"].startswith("FLOOR")
    (tmp_path / "e5" / "post_e5_s25_r3" / "summary.json").unlink()
    rc, err = run(tmp_path)
    assert rc != 0 and "incomplete" in err
    rc, res = run(tmp_path, "--partial")
    assert rc == 0 and res["E5"]["label"] == "DESCRIPTIVE, PARTIAL"


def test_extension_rule(tmp_path):
    # Consistent sign on 3/3 seeds but too noisy to resolve: the rule asks for seeds 4 and 5.
    for r, d in ((1, 0.01), (2, 0.08), (3, 0.03)):
        for cell, m in (("s0", 0.60), ("s25", 0.62), ("s50", 0.60 + d), ("s100", 0.61)):
            put(tmp_path, "e5", cell, r, m)
    rc, res = run(tmp_path)
    h = res["E5"]["H5_primary_YE1_minus_Y0"]
    assert h["verdict"] == "NOT_RESOLVED" and h["extend_to_5_seeds"] is True and h["sign_agreement"] == "3/3"


def test_early_ranking(tmp_path):
    """Amendment 1: at each rung, Spearman's rho of the rung predictor against the final metric over
    runs, its bootstrap interval, and whether the top third of cells by the predictor keeps the final
    best cell. Here the 50% rung orders the runs exactly as their final Y; the 5% rung reverses them."""
    import random
    rng = random.Random(3)
    final = {}
    for r in (1, 2, 3):
        for k, cell in enumerate(("s0", "s25", "s50", "s100")):
            m = 0.60 + 0.03 * k + 0.002 * r
            put(tmp_path, "e5", cell, r, m)
            final[(cell, r)] = m - 0.80
            p = tmp_path / "e5" / f"post_e5_{cell}_r{r}" / "summary.json"
            s = json.loads(p.read_text())
            s["final_rung"] = {"acc_gold": 0.5 + final[(cell, r)], "Y_base": final[(cell, r)]}
            p.write_text(json.dumps(s))
            for step, y in ((500, -final[(cell, r)]), (1000, rng.random()), (2500, rng.random()),
                            (5000, final[(cell, r)])):
                d = tmp_path / "e5" / f"rung_e5_{cell}_r{r}_s{step}"
                d.mkdir(parents=True)
                (d / "rung.json").write_text(json.dumps({
                    "cell": cell, "seed": r, "step": step, "Y_base": y, "acc_gold": 0.5 + y,
                    "panels": {"fineweb": 3.5 + 0.01 * k, "dialogue": 3.0}}))
    rc, res = run(tmp_path)
    assert rc == 0, res
    er = res["early_ranking"]["E5"]
    assert er["label"] == "REGISTERED"
    assert er["rungs"]["50%"]["Y_base->Y"]["verdict"] == "PREDICTIVE" and er["rungs"]["50%"]["Y_base->Y"]["rho"] > 0.99
    assert er["rungs"]["5%"]["Y_base->Y"]["verdict"] == "NOT_PREDICTIVE" and er["rungs"]["5%"]["Y_base->Y"]["rho"] < 0
    assert er["rungs"]["100%"]["Y_base->Y"]["verdict"] == "PREDICTIVE"
    assert er["earliest_predictive"]["Y_base->Y"] in ("10%", "25%", "50%")
    # Missing rungs (a replaced VM): fewer than 80% of the runs makes that rung INCOMPLETE.
    import shutil
    for cell in ("s0", "s25", "s50"):
        shutil.rmtree(tmp_path / "e5" / f"rung_e5_{cell}_r1_s5000")
    rc, res = run(tmp_path)
    assert res["early_ranking"]["E5"]["rungs"]["50%"]["Y_base->Y"]["verdict"] == "INCOMPLETE"
