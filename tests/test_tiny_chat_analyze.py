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
