"""experiments/tiny_chat_2026-10/size/ (amendment 6, PROPOSED): the 150M shapes and micro-batching, the Stage 0
jobs files (CUDA gate, the floor pilot's PASS, the 35% data job, a probe and the four-rate screens per depth; the
plain replay of E1 at 60M's pt_e1_60m_L32_r2 in the L4 file only), A6.1's depth rule, and analyze_size.py's
size slope (HS1), the 60M -> 150M step (HS2), the validity gate and the refusal of incomplete cells, on synthetic
summaries."""
import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "tiny_chat_2026-10"
SIZE = EXP / "size"
COLAB = ROOT / "tools" / "colab_chat"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ms = _load("make_size_jobs", SIZE / "make_size_jobs.py")
asz = _load("analyze_size", SIZE / "analyze_size.py")
bitid = _load("bitid", EXP / "x5" / "bitid.py")
ms.use_design()


def test_shapes_hit_the_target_at_head_dim_64():
    for sid, s in ms.shapes().items():
        assert abs(s["nonemb"] - ms.TARGET) / ms.TARGET < 0.005, sid
        assert s["d"] % 64 == 0 and s["heads"] == s["d"] // 64
        b, a = ms.batching(s["layers"])
        assert b * a * ms.mj.T == ms.mj.TOKENS_PER_STEP


def test_stage0_files_are_the_generated_ones_and_complete():
    sh = ms.shapes()
    for sid, s in sh.items():
        name, doc, _ = ms.stage0_doc(sid, s, replay=(s["layers"] == 4))
        on_disk = json.loads((COLAB / name).read_text(encoding="utf-8"))
        assert on_disk == json.loads(json.dumps(doc)), f"{name} is stale: rerun make_size_jobs.py"
        jobs = {j["name"]: j for j in doc["jobs"]}
        assert doc["max_parallel"] == 1
        assert "cuda_gate" in jobs and "data" in jobs
        g = jobs["gate"]["cmd"]
        assert "--stage0" in g and g[g.index("--need-pilot") + 1] == "PASS"
        assert jobs["data"]["cmd"][jobs["data"]["cmd"].index("--shares") + 1] == "35"
        probe = jobs[f"probe_{sid}"]
        assert probe["cmd"][probe["cmd"].index("--spec") + 1] == ms.spec_path(sid)
        screens = [j for n, j in jobs.items() if n.startswith(f"screen_{sid}_lr")]
        assert sorted(float(j["cmd"][j["cmd"].index("--lr") + 1]) for j in screens) == sorted(ms.tcr.REC_SCREEN_LRS)
        for j in screens:
            c = j["cmd"]
            assert c[c.index("--seed") + 1] == "101" and c[c.index("--steps") + 1] == "1500"
            assert j["after"] == f"probe_{sid}"
        assert (ms.REPLAY["job"] in jobs) == (s["layers"] == 4)
        spec = json.loads((ROOT / ms.spec_path(sid)).read_text(encoding="utf-8"))
        assert (spec["train"]["batch"], spec["train"]["accum"]) == ms.batching(s["layers"])
        assert spec["arch"]["custom"] == {k: s[k] for k in ("d", "layers", "heads", "d_ff")}


def test_replay_is_e1_60m_L32_r2_but_for_steps():
    e1 = json.loads((COLAB / "tiny_chat_e1_60m.json").read_text(encoding="utf-8"))
    ref = next(j for j in e1["jobs"] if j["name"] == "pt_e1_60m_L32_r2")["spec"]
    mine = ms.replay_spec()
    assert mine["train"]["steps"] == 500 == mine["train"]["lr_schedule"]["warmup_steps"]
    assert bitid.bare(mine) == bitid.bare(ref)


def summary(cell, seed, y, nonemb, total, fw=3.8, twin=0.53, noctx=0.25):
    pool = {"self": twin + y, "twin": twin, "gold": twin + y, "noctx": noctx}
    ctx = {"twin": twin, "noctx": noctx}
    return {"cell": cell, "seed": seed, "primary": y, "primary_all_items": y,
            "carry5": {"pool": pool, "ctx": ctx, "chance": 0.25, "pools": {}},
            "panels": {"fineweb": fw, "dialogue": 3.2}, "sft_evals": [[1500, 3.0]],
            "shape": {"nonemb": nonemb, "total": total}}


def write(art, block, s):
    d = art / block / f"post_{block}_{s['cell']}_r{s['seed']}"
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps(s))


N = {"20m": (20_066_496, 23_212_224), "60m": (60_187_008, 66_478_464), "150m": (150_436_480, 159_873_920)}


def make_art(tmp, y_of, depth=32, seeds=(1, 2, 3), floor150=False, plain=True):
    art = tmp / "art"
    noise = {"20m": (0.002, -0.001, -0.001), "60m": (-0.001, 0.002, -0.001), "150m": (0.0, -0.001, 0.001)}
    for r in seeds:
        e = {z: noise[z][r - 1] for z in noise}
        b20 = ("e5", "s35") if depth == 8 else ("e1", f"L{depth}")
        write(art, b20[0], summary(b20[1], r, y_of["20m"] + e["20m"], *N["20m"], fw=3.92))
        write(art, "e1_60m", summary(f"L{depth}", r, y_of["60m"] + e["60m"], *N["60m"], fw=3.73))
        tw = 0.30 if floor150 else 0.53
        write(art, f"e1_150m_r{r}", summary(f"L{depth}", r, y_of["150m"] + e["150m"], *N["150m"], fw=3.60, twin=tw))
    if plain:
        for k, (y, fw) in enumerate([(-0.04, 3.76), (-0.06, 3.84), (-0.08, 3.94)]):
            write(art, "e1", summary(f"L{2 if depth != 2 else 4}", k + 1, y, 20_017_536, 34_697_600, fw=fw))
    return art


def test_size_slope_supported_when_y_rises_with_log_size(tmp_path):
    x = {z: math.log2(N[z][0]) for z in N}
    y = {z: -0.08 + 0.02 * (x[z] - x["20m"]) for z in N}   # 2 points per doubling
    res = asz.size_block(str(make_art(tmp_path, y)), 32)
    assert abs(sum(res["weights_log2_nonemb"])) < 1e-12
    assert res["HS1_size_slope_Y"]["verdict"] == "SUPPORTED"
    assert res["HS1_size_slope_Y"]["mean"] == pytest.approx(0.02, abs=2e-3)
    assert res["HS2_step_60_150_per_doubling"]["verdict"] == "SUPPORTED"
    assert res["isoplot"]["extrapolation"] is True


def test_flat_size_is_equivalent_and_floor_ends_the_contrast(tmp_path):
    flat = {z: -0.06 for z in N}
    res = asz.size_block(str(make_art(tmp_path / "a", flat)), 32)
    assert res["HS1_size_slope_Y"]["verdict"] == "EQUIVALENT"
    res = asz.size_block(str(make_art(tmp_path / "b", flat, floor150=True)), 32)
    assert res["HS1_size_slope_Y"]["verdict"].startswith("FLOOR")
    assert res["HS2_step_60_150_per_doubling"]["verdict"].startswith("FLOOR")


def test_l8_reads_e5_s35_and_incomplete_is_refused(tmp_path):
    y = {"20m": -0.08, "60m": -0.05, "150m": -0.03}
    art = make_art(tmp_path, y, depth=8, seeds=(1, 2))
    with pytest.raises(SystemExit):
        asz.size_block(str(art), 8)
    part = asz.size_block(str(art), 8, partial=True)
    assert part["label"] == "DESCRIPTIVE, PARTIAL" and ("150m", 3) in part["missing"]
    assert part["cells"]["20m"]["Y"]["n"] == 2


def test_depth_rule(tmp_path):
    def runs(means, floor=()):
        out = {}
        for L, m in means.items():
            for r in (1, 2, 3):
                tw = 0.30 if L in floor else 0.53
                out[(f"L{L}", r)] = summary(f"L{L}", r, m, 1, 1, twin=tw)
        return out
    assert ms.depth_rule(runs({4: -0.07, 8: -0.055, 16: -0.06, 32: -0.04}))["depth"] == 32
    # L8 and L32 tied within 1 point: the higher 60M probe rate (L8's) wins
    assert ms.depth_rule(runs({4: -0.07, 8: -0.045, 16: -0.06, 32: -0.04}))["depth"] == 8
    # a FLOOR cell is never chosen
    assert ms.depth_rule(runs({4: -0.07, 8: -0.05, 16: -0.06, 32: -0.01}, floor=(32,)))["depth"] == 8
    r = runs({4: -0.07, 8: -0.05, 16: -0.06, 32: -0.04})
    del r[("L16", 3)]
    with pytest.raises(SystemExit):
        ms.depth_rule(r)
