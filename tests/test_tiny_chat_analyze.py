"""experiments/tiny_chat_2026-10/analyze.py on synthetic summaries: the registered contrasts and
decision rules (PREREGISTRATION.md sections 5-6), the floor gate, the data-bar rule, E1's L8 cell
borrowed from E5's 50% cell, and the refusal of incomplete contrasts."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TOOL = [sys.executable, str(ROOT / "experiments" / "tiny_chat_2026-10" / "analyze.py")]


def put(art, block, cell, seed, multi, twin=0.80, chance=0.25, fw=3.5, gold=None, qalone=0.25, extra=None):
    """A post job's summary as tiny_chat_run.py post writes it after amendment 2: `pool` over the
    primary families (Y's), `ctx` over the context families (the question-alone floor's), and the
    all-items `acc` (instruction included: here inflated by its length prior, as on the live model)."""
    d = art / block / f"post_{block}_{cell}_r{seed}"
    d.mkdir(parents=True)
    g = gold if gold is not None else multi + 0.02
    pool = {"self": multi, "twin": twin, "gold": g, "noctx": qalone}
    allitems = {k: v + 0.08 for k, v in pool.items()}
    fam = {m: {"recall": {"mean": v, "n": 432}, "instruction": {"mean": 0.85, "n": 144}} for m, v in pool.items()}
    (d / "summary.json").write_text(json.dumps({
        "cell": cell, "seed": seed, "primary": multi - twin, "primary_all_items": allitems["self"] - allitems["twin"],
        "carry5": {"acc": allitems, "pool": pool, "ctx": dict(pool), "chance": chance,
                   "pools": {"acc": {"family": fam}, "instruction_complies_gen": {"self": {"mean": 0.1, "n": 144}}}},
        "panels": {"fineweb": fw, "dialogue": 3.0}, "sft_evals": [[1500, 2.5]], **(extra or {})}))


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


def test_question_alone_floor(tmp_path):
    """Amendment 2: a twin well above chance is not enough when the question alone scores as well
    (a prior over the candidates, not reading the statement); the cell is FLOOR."""
    for r in (1, 2, 3):
        for cell in ("s0", "s25", "s50", "s100"):
            q = 0.75 if cell == "s50" else 0.25
            put(tmp_path, "e5", cell, r, 0.70 + 0.01 * r, twin=0.80, qalone=q)
    rc, res = run(tmp_path)
    assert rc == 0, res
    assert res["E5"]["floor_ok"] == {"s0": True, "s25": True, "s50": False, "s100": True}
    assert res["E5"]["H5_primary_YE1_minus_Y0"]["verdict"].startswith("FLOOR")
    # Exactly 10 points above the question alone passes; just under fails.
    for q, ok in ((0.70, True), (0.71, False)):
        art = tmp_path / f"q{q}"
        for r in (1, 2, 3):
            for cell in ("s0", "s25", "s50", "s100"):
                put(art, "e5", cell, r, 0.70 + 0.01 * r, twin=0.80, qalone=q if cell == "s50" else 0.25)
        rc, res = run(art)
        assert res["E5"]["floor_ok"]["s50"] is ok


def test_amended_metric_tables_and_refusal(tmp_path):
    for r in (1, 2, 3):
        for cell, m in (("s0", 0.60), ("s25", 0.62), ("s50", 0.66), ("s100", 0.66)):
            put(tmp_path, "e5", cell, r, m + 0.002 * r)
    rc, res = run(tmp_path)
    assert rc == 0, res
    c = res["E5"]["cells"]["s50"]
    assert abs(c["Y"]["mean"] - (0.664 - 0.80)) < 1e-9 and abs(c["Y_all"]["mean"] - c["Y"]["mean"]) < 1e-9
    assert c["twin_ctx"]["mean"] == 0.80 and c["noctx_ctx"]["mean"] == 0.25
    assert abs(c["twin_minus_qalone_ctx"]["mean"] - 0.55) < 1e-9 and c["instruction_complies_self"]["mean"] == 0.1
    fam = res["E5"]["families"]["s50"]
    assert set(fam) == {"recall", "instruction"} and "length" in fam["instruction"]["note"]
    assert fam["recall"]["noctx"] == 0.25 and abs(fam["recall"]["twin"] - 0.80) < 1e-12
    # A summary written before amendment 2 (no pools) is refused, not read with the old metric.
    p = tmp_path / "e5" / "post_e5_s0_r1" / "summary.json"
    s = json.loads(p.read_text())
    del s["carry5"]["pool"]
    p.write_text(json.dumps(s))
    rc, err = run(tmp_path)
    assert rc != 0 and "amendment 2" in err


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


# ---- amendment 3 (23 Oct 2026): the floor pilot ----
def test_pilot_verdict_and_exclusion(tmp_path):
    out = tmp_path / "verdict.json"
    put(tmp_path, "pilot", "s35", 99, 0.70, twin=0.80, qalone=0.25)
    p = subprocess.run(TOOL + ["--art", str(tmp_path), "--pilot-verdict", str(out)], capture_output=True, text=True)
    v = json.loads(out.read_text())
    assert p.returncode == 0 and v["verdict"] == "PASS" and v["excluded_from_analysis"]
    assert "Y" not in v and "multi_self" not in v  # only the gate's quantities
    for twin, qalone in ((0.34, 0.20), (0.60, 0.51)):  # (i) fails; (ii) fails
        import shutil
        shutil.rmtree(tmp_path / "pilot")
        put(tmp_path, "pilot", "s35", 99, 0.30, twin=twin, qalone=qalone)
        subprocess.run(TOOL + ["--art", str(tmp_path), "--pilot-verdict", str(out)], capture_output=True, text=True)
        assert json.loads(out.read_text())["verdict"] == "FAIL"
    # A pilot-seed summary inside an E block is refused.
    for s in (1, 2, 3):
        for c, m in (("s0", 0.70), ("s25", 0.72), ("s50", 0.75), ("s100", 0.75)):
            put(tmp_path, "e5", c, s, m)
    put(tmp_path, "e5", "s50", 99, 0.75)
    rc, err = run(tmp_path)
    assert rc != 0 and "floor pilot" in err


# ---- amendment 3, A3.3 (23 Oct 2026): the recursion arm, E1-R ----
PLAIN = {"L2": (896, 2, 2528), "L4": (640, 4, 1760), "L8": (448, 8, 1248), "L16": (320, 16, 864),
         "L32": (192, 32, 832)}
REC = {"R2": (0, 4, 2, 0), "R4": (0, 4, 4, 0), "R8": (0, 4, 8, 0), "MC7": (1, 2, 7, 1)}


def arch(cell):
    if cell in PLAIN:
        d, L, dff = PLAIN[cell]
        return {"d": d, "layers": L, "heads": d // 64, "d_ff": dff}
    P, B, R, C = REC[cell]
    return {"d": 640, "layers": 4, "heads": 10, "d_ff": 1760,
            "recursion": {"prelude": P, "block": B, "repeats": R, "coda": C, "schedule": "fixed", "inject": "none"}}


def rec_art(art, r8=False, twin_r4=0.80, hours=None):
    """E5 (its 35% cell is E1's L8), E1's other 20M depths and the recursion cells. Y rises 6 points per
    doubling of passes (L4 -0.12, R2 -0.06, R4 0.00, R8 0.02); plain Y lies on the line
    Y = -0.10 - 0.4 (fw - 3.5); every recursion cell sits at fw 3.55, where the line gives -0.12."""
    noise = {1: 0.0, 2: 0.004, 3: -0.004}
    for r in (1, 2, 3):
        for cell in ("s0", "s17", "s35", "s70"):
            put(art, "e5", cell, r, 0.70, fw=3.50, extra={"arch": arch("L8")})
        for cell, m, fw in (("L2", 0.66, 3.60), ("L4", 0.68, 3.55), ("L16", 0.72, 3.45), ("L32", 0.74, 3.40)):
            put(art, "e1", cell, r, m, fw=fw, extra={"arch": arch(cell)})
        cells = [("R2", 0.74), ("R4", 0.80), ("MC7", 0.78)] + ([("R8", 0.82)] if r8 else [])
        for cell, m in cells:
            ex = {"arch": arch(cell)}
            if hours:
                ex.update({"pt_result": {"wall_seconds": hours[cell] * 3600}, "sft_result": {"wall_seconds": 0.0}})
            put(art, "e1_rec_r8" if cell == "R8" else "e1_rec", cell, r, m + noise[r], fw=3.55,
                twin=twin_r4 if cell == "R4" else 0.80, extra=ex)


def run_rec(art, *extra):
    return run(art, *extra, levels=(0, 17, 35, 70), e1=35)


def test_rec_flops_match_the_registered_table():
    import importlib.util
    spec = importlib.util.spec_from_file_location("an", ROOT / "experiments" / "tiny_chat_2026-10" / "analyze.py")
    an = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(an)
    table = {"L4": 0.19, "R2": 0.32, "R4": 0.58, "R8": 1.09, "MC7": 0.58, "L8": 0.17, "L16": 0.17, "L32": 0.16}
    for c, g in table.items():
        assert abs(an.flops_per_token(arch(c)) / 1e9 - g) < 0.006, c


def test_rec_hypotheses(tmp_path):
    rec_art(tmp_path, hours={"R2": 3.0, "R4": 5.0, "MC7": 5.0})
    rc, res = run_rec(tmp_path)
    assert rc == 0, res
    r = res["E1_rec_20m"]
    assert r["label"] == "REGISTERED (A3.3)" and not r["r8_ran"] and r["seeds"] == [1, 2, 3]
    h1 = r["HR1_passes_slope_Y"]
    assert h1["verdict"] == "SUPPORTED" and abs(h1["mean"] - 0.06) < 0.003
    k = r["HR1_knowledge_passes_slope_fineweb_loss"]
    assert k["verdict"] == "EQUIVALENT" and k["prediction"] == "not falsified"
    h2 = r["HR2_recursion_minus_plain_depth"]  # (0.04 + 0.08) / 2 per seed: above the 3-point SESOI
    assert h2["verdict"] == "SUPPORTED" and abs(h2["mean"] - 0.06) < 0.003
    assert h2["literature_prediction"] == "falsified" and h2["HR2_J"] == "confirmed"
    h3 = r["HR3_residual_from_plain_isoloss_line"]  # residuals 0.06 (R2) and 0.12 (R4)
    assert h3["verdict"] == "SUPPORTED" and abs(h3["mean"] - 0.09) < 0.003 and h3["extrapolations"] == []
    assert abs(h3["per_cell_mean_residual"]["R4"] - 0.12) < 0.003
    assert r["secondary_MC7_minus_R4"]["label"] == "SECONDARY" and "HR4_quadratic_Y_over_log2_passes" not in r
    cost = r["cost"]
    assert abs(cost["R4"]["flops_per_token"] / 1e9 - 0.58) < 0.006 and abs(cost["R2"]["l4_hours_per_run"] - 3.0) < 1e-9
    assert abs(cost["R2"]["Y_per_l4_hour"] - (-0.06 / 3.0)) < 0.002 and cost["L4"]["Y_per_l4_hour"] is None
    # The R8 rule: R2 and R4 pass the gate and Y(R4) - Y(R2) = 6 points >= 1.
    out = tmp_path / "rec_decision.json"
    p = subprocess.run(TOOL + ["--art", str(tmp_path), "--rec-decision", str(out)], capture_output=True, text=True)
    d = json.loads(out.read_text())
    assert p.returncode == 0 and d["run_r8"] is True and abs(d["mean_Y_R4_minus_R2"] - 0.06) < 1e-9


def test_rec_with_r8_and_hr4(tmp_path):
    rec_art(tmp_path, r8=True)
    rc, res = run_rec(tmp_path)
    assert rc == 0, res
    r = res["E1_rec_20m"]
    assert r["r8_ran"] and r["secondary_HR1_slope_with_R8"]["label"] == "SECONDARY"
    h4 = r["HR4_quadratic_Y_over_log2_passes"]  # -0.12 + 0.06 - 0.00 + 0.02
    assert h4["label"].startswith("DESCRIPTIVE") and abs(h4["mean"] + 0.04) < 0.003 and h4["negative"]
    assert "literature_prediction" in r["secondary_HR2_three_pairs"]


def test_rec_floor_and_incomplete(tmp_path):
    rec_art(tmp_path, twin_r4=0.30)  # R4's twin within 10 points of chance: R4 is FLOOR
    rc, res = run_rec(tmp_path)
    assert rc == 0, res
    r = res["E1_rec_20m"]
    assert r["floor_ok"]["R4"] is False
    for h in ("HR1_passes_slope_Y", "HR2_recursion_minus_plain_depth", "HR3_residual_from_plain_isoloss_line"):
        assert r[h]["verdict"].startswith("FLOOR") and "R4" in r[h]["verdict"], h
    assert r["HR1_knowledge_passes_slope_fineweb_loss"]["verdict"] != "FLOOR"  # panel losses are not gated
    out = tmp_path / "rec_decision.json"
    subprocess.run(TOOL + ["--art", str(tmp_path), "--rec-decision", str(out)], capture_output=True, text=True)
    assert json.loads(out.read_text())["run_r8"] is False
    (tmp_path / "e1_rec" / "post_e1_rec_MC7_r2" / "summary.json").unlink()
    rc, err = run_rec(tmp_path)
    assert rc != 0 and "E1-R incomplete" in err
    rc, res = run_rec(tmp_path, "--partial")
    assert rc == 0 and res["E1_rec_20m"]["label"] == "DESCRIPTIVE, PARTIAL"
    (tmp_path / "e1_rec" / "post_e1_rec_R2_r2" / "summary.json").unlink()
    p = subprocess.run(TOOL + ["--art", str(tmp_path), "--rec-decision", str(out)], capture_output=True, text=True)
    assert p.returncode != 0 and "R8 rule needs" in p.stderr
