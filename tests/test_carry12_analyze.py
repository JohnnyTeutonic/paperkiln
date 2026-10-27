"""experiments/tiny_chat_long_2026-10/analyze.py on synthetic post summaries (PREREGISTRATION_DRAFT.md sections 7-9):
every registered contrast and verdict rule (HT, HD, HS1 with HS1-alt and the S-full condition, HS0 and HR when
their cells exist, HX's D_X, correction slope, H-gap, HX-main and isoplot), the validity gate's FLOOR, the
extension rule, the seeds 4-5 read, the refusal of an incomplete core and --partial, L* = 16 (P16 is B), and W512's
reproduction check against E1's home summaries."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TOOL = [sys.executable, str(ROOT / "experiments" / "tiny_chat_long_2026-10" / "analyze.py")]
NOISE = {1: 0.0, 2: 0.004, 3: -0.004, 4: 0.002, 5: -0.002}


def summary(cell, seed, y12, y12_gold=None, twin=0.80, noctx=0.25, rec_gold=0.70, rec_near=0.78, rec_twin=0.85,
            corr=(0.70, 0.0), fw=3.40, gold_by_d=None, sft_sha="s"):
    """A post summary as tiny_chat_run.py post writes it with --carry12-items: Y12 and Y12_gold at the top, and
    carry12_read's pools, derived quantities and curves."""
    y12_gold = y12 + 0.02 if y12_gold is None else y12_gold
    prim = {"twin": twin, "noctx": noctx, "gold": twin + y12_gold, "self": twin + y12, "near": twin - 0.02}
    ctx = dict(prim)
    family = {"gold": {"recall": rec_gold}, "near": {"recall": rec_near}, "twin": {"recall": rec_twin}}
    fd = {"gold": {"correction": {str(d): {"mean": corr[0] + corr[1] * d, "n": 30} for d in range(1, 12)},
                   "recall": {str(d): {"mean": v, "n": 60} for d, v in (gold_by_d or {}).items()}}}

    def rows(d):
        return {m: {"mean": v, "n": 1000} for m, v in d.items()}
    return {"cell": cell, "seed": seed, "primary": -0.04, "Y12": y12 if cell != "W512" else None, "Y12_gold": y12_gold,
            "panels": {"fineweb": fw, "dialogue": 3.0}, "sft_weights_sha256": sft_sha, "pt_weights_sha256": "p",
            "carry12": {"pools": {"acc": {"primary": rows(prim), "context": rows(ctx),
                                          "family": {m: {f: {"mean": v, "n": 480} for f, v in fs.items()}
                                                     for m, fs in family.items()}}},
                        "derived": {"Y12_self": y12, "Y12_gold": y12_gold, "split_cost_gold": -0.02,
                                    "distance_cost_gold": y12_gold + 0.02, "own_replies_cost": y12 - y12_gold},
                        "family_distance": fd, "instruction_complies": {"self": {"mean": 0.4, "n": 72}}}}


def put(art, s, machine=None):
    d = art / (machine or f"c12_{s['cell']}_r{s['seed']}") / f"post_c12_{s['cell']}_r{s['seed']}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps(s))


def core(art, seeds=(1, 2, 3), **over):
    """The registered core with clear effects: HT +6 points, D = B, S-gist -8 points, X +1 point overall with a
    gap gain (far recall +5 points against near +0) rising 1.5 points per turn of distance in correction."""
    for r in seeds:
        n = NOISE[r]
        cells = {"W512": dict(y12=-0.06, y12_gold=-0.06 + n, fw=3.50),
                 "B": dict(y12=-0.03 + n, y12_gold=0.0 + n, fw=3.40, gold_by_d={6: 0.70, 9: 0.60, 12: 0.50}),
                 "D": dict(y12=-0.03 + 1.5 * n, fw=3.40),
                 "Sgist": dict(y12=-0.11 + n, fw=3.41, gold_by_d={6: 0.60, 9: 0.48, 12: 0.40}),
                 "P16": dict(y12=-0.02 + n, fw=3.36, rec_gold=0.70, rec_near=0.78, corr=(0.70, 0.0)),
                 "X": dict(y12=-0.01 + 2 * n, fw=3.30, rec_gold=0.75 + n, rec_near=0.78, corr=(0.66, 0.015 + n))}
        for c, kw in cells.items():
            kw.update({k: (v(n) if callable(v) else v) for k, v in over.get(c, {}).items()})
            put(art, summary(c, r, **kw), machine=f"c12_{'B' if c in ('D', 'W512') else c}_r{r}")


def run(art, *extra):
    p = subprocess.run(TOOL + ["--art", str(art), "--lstar", "8", *extra], capture_output=True, text=True)
    return p.returncode, (json.loads(p.stdout)["carry12"] if p.returncode == 0 else p.stderr)


def test_registered_contrasts_and_verdicts(tmp_path):
    core(tmp_path)
    rc, res = run(tmp_path)
    assert rc == 0, res
    assert res["seeds"] == [1, 2, 3] and res["label"].startswith("REGISTERED")
    assert all(res["validity_gate"][c] for c in ("W512", "B", "D", "Sgist", "P16", "X"))
    ht = res["HT_B_minus_W512_Y12_gold"]
    assert ht["verdict"] == "SUPPORTED" and abs(ht["mean"] - 0.06) < 1e-9 and ht["counter_HT_0"] == "not confirmed"
    hd = res["HD_D_minus_B"]
    assert hd["verdict"] == "EQUIVALENT" and hd["counter_HD_0"].startswith("confirmed")
    hs1 = res["HS1_Sgist_minus_B"]
    assert hs1["verdict"] == "SUPPORTED" and abs(hs1["mean"] + 0.08) < 1e-9 and hs1["HS1_alt"] == "does not hold"
    assert res["s_full_condition"]["s_gist_loses_HS1_SUPPORTED"] is True
    assert res["s_full_condition"]["section_6_text_HS1_CONTRADICTED"] is False
    # pooled over the context families by item counts: recall (60 items) and correction (30, flat at 0.70)
    r9 = res["HS1_retention_ratio_by_distance"]["9"]["ratio"]
    sg, b = (0.48 * 60 + 0.70 * 30) / 90, (0.60 * 60 + 0.70 * 30) / 90
    assert abs(r9 - (sg - 0.25) / (b - 0.25)) < 1e-9
    # HX: D_X = (0.75 - 0.70) - (0.78 - 0.78) = 0.05 > SESOI; the correction gain's slope 0.015 per turn.
    assert abs(res["HX_gap_D_X"]["mean"] - 0.05) < 1e-9 and res["HX_gap_D_X"]["verdict"] == "SUPPORTED"
    sl = res["HX_gap_correction_slope"]
    assert abs(sl["mean"] - 0.015) < 1e-9 and sl["verdict"] == "SUPPORTED" and sl["sesoi"] == 0.01
    assert res["HX_H_gap"]["verdict"] == "SUPPORTED"
    assert res["HX_main_X_minus_P16"]["verdict"] in ("NOT_RESOLVED", "EQUIVALENT", "SIGNIFICANT_BELOW_SESOI")
    iso = res["HX_isoplot_residual"]
    assert iso["n"] == 3 and "thin" in iso and iso["flags"]  # X's loss 3.30 lies outside B's and P16's
    assert "HS0_Sfull_minus_B" not in res and "HR_R_minus_B" not in res  # conditional cells not run
    assert res["secondary_panel_fineweb_X_minus_P16"]["verdict"] == "SUPPORTED"  # 0.06 nats lower


def test_contradicted_and_hs1_alt(tmp_path):
    """D below B beyond doubt is CONTRADICTED (HD's direction is positive); S-gist equal to B is EQUIVALENT, so
    HS1-alt holds and S-full runs under neither reading."""
    core(tmp_path, D={"y12": -0.09}, Sgist={"y12": -0.03})
    rc, res = run(tmp_path)
    assert rc == 0, res
    assert res["HD_D_minus_B"]["verdict"] == "CONTRADICTED"
    hs1 = res["HS1_Sgist_minus_B"]
    assert hs1["verdict"] == "EQUIVALENT" and hs1["HS1_alt"].startswith("holds")
    sc = res["s_full_condition"]
    assert not sc["s_gist_loses_HS1_SUPPORTED"] and not sc["section_6_text_HS1_CONTRADICTED"]


def test_floor_gates_contrasts(tmp_path):
    """W512 with twins under 0.35 is FLOOR: HT is not read; X with twin within 10 points of the question alone is
    FLOOR for every HX contrast; other contrasts are read."""
    core(tmp_path, W512={"twin": 0.30}, X={"noctx": 0.75})
    rc, res = run(tmp_path)
    assert rc == 0, res
    assert res["validity_gate"]["W512"] is False and res["validity_gate"]["X"] is False
    assert res["HT_B_minus_W512_Y12_gold"]["verdict"].startswith("FLOOR") and "W512" in \
        res["HT_B_minus_W512_Y12_gold"]["verdict"]
    for k in ("HX_gap_D_X", "HX_gap_correction_slope", "HX_main_X_minus_P16", "HX_H_gap", "HX_isoplot_residual"):
        assert res[k]["verdict"].startswith("FLOOR"), k
    assert res["HD_D_minus_B"]["verdict"] == "EQUIVALENT"


def test_extension_rule_and_seeds_4_5(tmp_path):
    """HX-main with one sign on all three seeds but not resolved: extend_to_5_seeds; with seeds 4-5 home for every
    core cell the contrasts read five seeds."""
    def x_y12(n):  # X - P16 = 0.025 + 5 n: 0.025, 0.045, 0.005 on seeds 1-3, all positive, t 2.2
        return 0.005 + 6 * n
    core(tmp_path, X={"y12": x_y12})
    rc, res = run(tmp_path)
    xm = res["HX_main_X_minus_P16"]
    assert xm["verdict"] == "NOT_RESOLVED" and xm["sign_agreement"] == "3/3" and xm["extend_to_5_seeds"] is True
    core(tmp_path, seeds=(4, 5), X={"y12": x_y12})
    rc, res = run(tmp_path)
    assert rc == 0, res
    xm = res["HX_main_X_minus_P16"]
    assert res["seeds"] == [1, 2, 3, 4, 5] and xm["n"] == 5 and xm["extend_to_5_seeds"] is False
    assert xm["verdict"] == "SIGNIFICANT_BELOW_SESOI"  # t 3.5 > 2.776 at df 4, mean 0.025 under the SESOI


def test_incomplete_refused_and_partial(tmp_path):
    core(tmp_path)
    (tmp_path / "c12_X_r3" / "post_c12_X_r3" / "summary.json").unlink()
    rc, err = run(tmp_path)
    assert rc != 0 and "incomplete" in err and "'X', 3" in err
    rc, res = run(tmp_path, "--partial")
    assert rc == 0 and res["label"] == "DESCRIPTIVE, PARTIAL"
    assert not any(k.startswith(("HT", "HD", "HS", "HX", "HR")) for k in res)
    assert res["cells"]["X"]["Y12"]["n"] == 2


def test_duplicate_summary_refused(tmp_path):
    core(tmp_path)
    put(tmp_path, summary("B", 1, -0.03), machine="elsewhere")
    rc, err = run(tmp_path)
    assert rc != 0 and "second summary" in err


def test_conditional_cells_s_full_and_r(tmp_path):
    core(tmp_path)
    for r in (1, 2, 3):
        put(tmp_path, summary("Sfull", r, -0.03 + NOISE[r]))
        put(tmp_path, summary("R", r, 0.03 + 1.5 * NOISE[r]))
    rc, res = run(tmp_path)
    assert rc == 0, res
    assert res["HS0_Sfull_minus_B"]["verdict"] == "EQUIVALENT"
    assert res["HS0_mask_cost_Sgist_minus_Sfull"]["verdict"] == "CONTRADICTED"  # S-gist 8 points below S-full
    assert res["HR_R_minus_B"]["verdict"].startswith("NOT READ")
    rc, res = run(tmp_path, "--r-sign", "1")
    assert res["HR_R_minus_B"]["verdict"] == "SUPPORTED" and res["HR_R_minus_B"]["sign_transferred"] is True
    rc, res = run(tmp_path, "--r-sign", "-1")
    assert res["HR_R_minus_B"]["verdict"] == "CONTRADICTED" and res["HR_R_minus_B"]["sign_transferred"] is False


def test_lstar_16_p16_is_b(tmp_path):
    """L* = 16: no P16 machine; X is read against B and the isoplot has no line (one plain cell)."""
    core(tmp_path)
    for r in (1, 2, 3):
        p = tmp_path / f"c12_P16_r{r}"
        for f in p.rglob("*"):
            if f.is_file():
                f.unlink()
    p = subprocess.run(TOOL + ["--art", str(tmp_path), "--lstar", "16"], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    res = json.loads(p.stdout)["carry12"]
    assert res["p16_cell"] == "B" and res["HX"]["note"].startswith("L* = 16")
    assert res["HX_isoplot_residual"]["verdict"] == "NOT READ"
    assert res["HX_main_X_minus_P16"]["n"] == 3


def test_w512_reproduction_against_e1_home(tmp_path):
    core(tmp_path / "art")
    home = tmp_path / "e1_60m_L8"
    for r in (1, 2, 3):
        d = home / f"post_e1_60m_L8_r{r}"
        d.mkdir(parents=True)
        (d / "summary.json").write_text(json.dumps({"sft_weights_sha256": "s" if r != 2 else "other",
                                                    "pt_weights_sha256": "p", "panels": {"fineweb": 3.5}}))
    rc, res = run(tmp_path / "art", "--e1-home", str(home))
    assert rc == 0, res
    w = res["w512_reproduces_e1"]
    assert w["1"]["sft_weights_identical"] and not w["2"]["sft_weights_identical"] and w["3"]["pt_weights_identical"]
    assert abs(w["1"]["panel_fineweb_diff"]) < 1e-12
