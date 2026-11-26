"""The crossed curriculum's jobs files (tools/colab_chat/cc_jobs.py) against A32's renderer and the registration (sections 2 and 8):
every model is S71's spec at D1 with only the start, seed, packet and (margin arms) the twin term changed; stage 2 waits on its own
stage-1 model; WS and SCR train last; the margin arms carry their seed's twin file and one lambda; the dev sweep is section 2.5's
five runs at seed 99. Claude Opus 5.5 (a fork of the coordinating session), 26 Nov 2026."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools", "colab_chat"))
import b2cue_jobs as B  # noqa: E402
import cc_jobs as J  # noqa: E402

U = "/content/chat/upload"


def by_name(jobs):
    return {j["name"]: j for j in jobs}


def s71_rest(spec):
    s = json.loads(json.dumps(spec))
    for k in ("init_from", "seed"):
        s["train"].pop(k)
    s["train"].pop("twin_loss", None)
    s["data"].pop("sft_examples")
    s["data"].pop("twin_examples", None)
    return s


REF = s71_rest(B.arm_spec("D1", "x", 1, "P"))


def test_files_match_rendering():
    for c, lam in J.variants():
        assert open(J.out_name(c, lam)).read() == J.text(c, lam)


def test_main_models_and_order():
    for c, seed in (("main1", 1), ("main2", 2)):
        for lam in J.LAMBDAS:
            jobs = J.render(c, lam)["jobs"]
            train = [j["name"] for j in jobs if "spec" in j]
            assert train == [f"{a}_N250_{seed}" for a in ("L", "W", "S", "SYM1", "FIL1", "SB", "SBC", "NB", "NBC", "FB", "FBC", "WS", "SCR")]
            assert [j["name"] for j in jobs][0] == "gate"
            assert [j["name"] for j in jobs if j["name"].startswith("score_")] == [f"score_{n}" for n in train]
            jb = by_name(jobs)
            for n in train:
                j, arm = jb[n], n.split("_")[0]
                sp = j["spec"]
                assert s71_rest(sp) == REF                          # S71's spec at D1 otherwise
                assert sp["train"]["seed"] == seed
                assert "snapshot_steps" not in j and "rung_of" not in j
                if arm in ("L", "W", "S", "SYM1", "FIL1", "WS", "SCR"):
                    assert sp["train"]["init_from"] == f"{U}/sft_tlN250.safetensors" and "init_job" not in j and j["after"] == "gate"
                    assert sp["data"]["sft_examples"] == f"{U}/S71{arm}.jsonl" and "twin_loss" not in sp["train"]
                else:
                    parent = {"SB": "SYM1", "SBC": "SYM1", "NB": "S", "NBC": "S", "FB": "FIL1", "FBC": "FIL1"}[arm]
                    pn = f"{parent}_N250_{seed}"
                    assert j["init_job"] == pn and j["after"] == ["gate", pn]
                    assert sp["train"]["init_from"] == f"/content/chat/{pn}/{pn}.safetensors"
                    assert sp["data"]["sft_examples"] == f"{U}/S71BR.jsonl"
                    if arm.endswith("C"):
                        assert sp["data"]["twin_examples"] == f"{U}/train_twins_s{seed}.jsonl"
                        assert sp["train"]["twin_loss"] == {"groups": 4, "margin": 1.0, "lambda": lam, "mu": lam}
                    else:
                        assert "twin_loss" not in sp["train"] and "twin_examples" not in sp["data"]
            # the plain and margin branches of one stage-1 model differ only in the twin term
            for p, m in (("SB", "SBC"), ("NB", "NBC"), ("FB", "FBC")):
                a, b = (json.loads(json.dumps(jb[f"{x}_N250_{seed}"]["spec"])) for x in (p, m))
                b["train"].pop("twin_loss")
                b["data"].pop("twin_examples")
                assert a == b


def test_dev_sweep():
    jobs = J.render("dev")["jobs"]
    jb = by_name(jobs)
    train = [j["name"] for j in jobs if "spec" in j]
    assert train == ["SYM1_N250_99", "D0_N250_99", "Dn_0p3_N250_99", "Dn_1p0_N250_99", "Dn_3p0_N250_99"]
    s1 = jb["SYM1_N250_99"]["spec"]
    assert s1["data"]["sft_examples"] == f"{U}/S71SYM1.jsonl" and s1["train"]["init_from"] == f"{U}/sft_tlN250.safetensors"
    for n, lam in (("D0_N250_99", None), ("Dn_0p3_N250_99", 0.3), ("Dn_1p0_N250_99", 1.0), ("Dn_3p0_N250_99", 3.0)):
        j = jb[n]
        assert j["init_job"] == "SYM1_N250_99" and j["spec"]["train"]["seed"] == 99
        assert s71_rest(j["spec"]) == REF and j["spec"]["data"]["sft_examples"] == f"{U}/S71BR.jsonl"
        if lam is None:
            assert "twin_loss" not in j["spec"]["train"]
        else:
            assert j["spec"]["train"]["twin_loss"] == {"groups": 4, "margin": 1.0, "lambda": lam, "mu": lam}
            assert j["spec"]["data"]["twin_examples"] == f"{U}/train_twins_s99.jsonl"
        assert f"score_{n}" in jb and f"devmargin_{n}" in jb
        assert f"{U}/cc_dev_req.jsonl" in jb[f"devmargin_{n}"]["cmd"][2]
    assert "score_SYM1_N250_99" in jb


def test_gate_and_names():
    for c, lam in J.variants():
        jobs = J.render(c, lam)["jobs"]
        g = jobs[0]
        assert g["name"] == "gate" and f"{U}/cc_gate_D1.json" in g["cmd"]
        names = [j["name"] for j in jobs]
        assert len(names) == len(set(names))
        assert not any(n.startswith(("TL_", "B2C", "P_", "N_", "COV_", "R2_")) for n in names)
    gs = J.gate_spec()
    assert gs["train"]["twin_loss"] == {"groups": 2, "margin": 1.0, "lambda": 1.0, "mu": 1.0}
    assert gs["data"]["twin_examples"] == f"{U}/gate_twins.jsonl" and gs["train"]["batch"] == 1 and gs["train"]["steps"] == 2
