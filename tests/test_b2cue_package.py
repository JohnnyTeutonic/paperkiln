"""A32's (b2cue) package checks without a trainer: the rendered jobs files equal their rendering; every spec departs from its
registered parent (A31's S71 arm spec, A's sft_b1_r1 SFT1500, base 2's D1 shape) only where the frozen design says; job order,
barrier and watch wiring; the selection rule's cases; the gate's answer-mask comparison; the watch and barrier modes.
Claude Opus 5.5, 15 Nov 2026. CPU only, seconds."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "colab_chat"))
import a31_jobs as A31  # noqa: E402
import b2cue_gate as G  # noqa: E402
import b2cue_jobs as J  # noqa: E402
import b2cue_select as S  # noqa: E402

FILES = sorted((ROOT / "tools" / "colab_chat").glob("b2cue_c*.json"))


def git_show(rev, path):
    r = subprocess.run(["git", "-C", str(ROOT), "show", f"{rev}:{path}"], capture_output=True, text=True)
    if r.returncode:
        pytest.skip(f"{rev}:{path} is not in this clone")
    return json.loads(r.stdout)


def flat(d, pre=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, f"{pre}{k}."))
        else:
            out[f"{pre}{k}"] = v
    return out


def diff(a, b):
    fa, fb = flat(a), flat(b)
    return {k for k in set(fa) | set(fb) if fa.get(k, "<absent>") != fb.get(k, "<absent>")}


def test_rendered_files_match_and_cover_every_variant():
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "colab_chat" / "b2cue_jobs.py"), "--all", "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert len(FILES) == len(J.variants()) == 45   # 21 before the drop variants (launch review round 1, finding 2)
    for p in FILES:
        assert json.loads(p.read_text())["home_markers"] is True


def heavy(p):
    return [j for j in json.loads(Path(p).read_text())["jobs"] if "spec" in j]


def test_arm_specs_depart_from_s71_only_where_registered():
    allowed = {"arch.custom.d", "arch.custom.layers", "arch.custom.heads", "arch.custom.d_ff", "data.sft_examples", "train.init_from"}
    for p in FILES:
        for j in heavy(p):
            n, s = j["name"], j["spec"]
            if n.startswith(("sft_", "repro_")):
                continue
            arm, seed = n.split("_")[0], int(n.split("_")[-1]) if n[0] != "D" else J.SWEEP_SEED
            tl = s["train"].get("twin_loss")
            ref = A31.spec(seed, tl["lambda"] if tl else None, tl["mu"] if tl else None, dev=n.startswith("D"))
            d = diff(s, ref) - allowed - ({"train.twin_loss.answer_mask"} if arm == "NA" else set())
            assert not d, (p.name, n, d)
            assert s["data"]["sft_examples"].endswith("/P0.jsonl" if arm == "P0" else "/S74.jsonl"), n
            assert s["arch"] == J.ARCH["A" if "Aref" in n else "D1"], n
            if arm in ("N", "NA", "Dn"):
                assert tl["lambda"] == tl["mu"] and tl["margin"] == 1.0 and tl["groups"] == 4, n
            if arm == "M":
                assert tl["mu"] == 0.0 and tl["lambda"] > 0, n
            if arm in ("P", "P0", "D0"):
                assert tl is None and "twin_examples" not in s["data"], n
            assert tl is None or (tl.get("answer_mask") is True) == (arm == "NA"), n


def test_sft1500_departs_from_sft_b1_r1_only_where_registered():
    ref = next(j for j in git_show("ab15db97", "tools/colab_chat/tc_b1_winner_sft_20261109.json")["jobs"] if j["name"] == "sft_b1_r1")["spec"]
    allowed = {"name", "arch.custom.d", "arch.custom.layers", "arch.custom.heads", "arch.custom.d_ff", "data.vocab", "data.sft_examples",
               "train.init_from", "train.eval_at"}
    for p in FILES:
        for j in heavy(p):
            if j["name"].startswith("sft_"):
                s = j["spec"]
                assert not diff(s, ref) - allowed, (p.name, j["name"], diff(s, ref) - allowed)
                assert s["train"]["eval_at"] == [188, 375, 750] and j["env"] == {"MICROTORCH_CHECKPOINT_TXN": "1"}
                assert s["data"]["sft_examples"].endswith("/sft_human.jsonl")


def test_d1_is_base_twos_shape():
    for rev, path in (("711160a2", "tools/colab_chat/base2_N.json"), ("58fe11ab", "tools/colab_chat/base2_N2.json")):
        pt = [j for j in git_show(rev, path)["jobs"] if "spec" in j and "arch" in j["spec"]]
        assert pt and all(j["spec"]["arch"] == J.D1_ARCH for j in pt), (rev, path)


def jobs_of(name):
    return json.loads((ROOT / "tools" / "colab_chat" / name).read_text())["jobs"]


def test_c0_wiring():
    js = jobs_of("b2cue_c0.json")
    names = [j["name"] for j in js]
    assert names[0] == "gate" and names[-1] == "select"
    hv = [j["name"] for j in js if "spec" in j]
    assert hv == ["sft_N250", "D0", "Dn_0p1", "Dn_0p3", "Dn_1p0"]
    by = {j["name"]: j for j in js}
    for n in hv[1:]:
        assert by[n]["spec"]["train"]["seed"] == 37 and by[n]["init_job"] == "sft_N250" and by[n]["spec"]["export"]["formats"] == ["safetensors"]
        assert "b2cue_selection_req.jsonl" in by[f"score_{n}"]["cmd"][2]
        for s in (128, 256, 512):
            e = by[f"early_{n}_{s}"]
            assert e["rung_of"] == {"job": n, "step": s} and "--score-timeout" in e["cmd"] and e["cmd"][e["cmd"].index("--ref") + 1] == "D0"
    assert set(by["select"]["after"]) == set(hv[1:]) | {f"score_{n}" for n in hv[1:]} | {f"early_{n}_{s}" for n in hv[1:] for s in (128, 256, 512)}
    assert by["select"]["after_terminal"] and by["watch_sft_N250"]["after_terminal"]
    assert not any("confirmation" in json.dumps(j) for j in js)   # C0 never reads the confirmation tier


@pytest.mark.parametrize("name", ["b2cue_c1a_0p3.json", "b2cue_c2_0p1.json"])
def test_extra_seeds_wait_on_the_barrier_and_run_last(name):
    js = jobs_of(name)
    names = [j["name"] for j in js]
    b = names.index("barrier")
    bar = js[b]
    assert bar["after"] == names[:b] and bar["after_terminal"]
    after = js[b + 1:]
    assert {j["name"] for j in after if "spec" in j} == {f"{a}_{base}_{s}" for a in ("P", "N") for s in (175, 213)
                                                         for base in (("N250",) if "c1a" in name else ("N2",))}
    for j in after:
        if "spec" in j:
            assert "barrier" in j["after"]
    assert all("spec" not in j for j in js[:b] if j["name"].endswith(("_175", "_213")))


def test_watch_waits_on_the_previous_heavy_job():
    js = jobs_of("b2cue_c1a_0p3.json")
    hv = [j["name"] for j in js if "spec" in j]
    w = next(j for j in js if j["name"] == "watch_sft_S")
    assert w["after"] == ["gate", hv[hv.index("sft_S") - 1]] and w["after_terminal"]


def test_heavy_order_puts_p_first_and_s_after_n250():
    hv = [j["name"] for j in heavy(ROOT / "tools" / "colab_chat" / "b2cue_c1a_0p3.json")]
    assert hv[:3] == ["P_N250_1", "P_N250_50", "P_N250_125"]
    assert hv.index("sft_S") > max(hv.index(n) for n in hv if "N250" in n and not n.endswith(("_175", "_213")))
    assert hv[-4:] == ["P_N250_175", "P_N250_213", "N_N250_175", "N_N250_213"]


def test_no_arm_files_train_only_p_and_p0():
    for p in FILES:
        if "noarm" in p.name:
            assert all(j["name"].split("_")[0] in ("P", "P0", "sft", "repro") for j in heavy(p)), p.name
            assert "barrier" not in [j["name"] for j in json.loads(p.read_text())["jobs"]]


def test_gate_spec():
    for sh in ("D1", "A"):
        g = J.gate_spec(sh)
        assert g["train"]["batch"] == 1 and g["train"]["steps"] == 2 and g["train"]["twin_loss"]["groups"] == 2
        assert g["train"]["twin_loss"]["lambda"] == g["train"]["twin_loss"]["mu"] == 1.0 and g["arch"] == J.ARCH[sh]
        assert g["data"]["twin_examples"].endswith("gate_twins.jsonl") and "answer_mask" not in g["train"]["twin_loss"]
        fb = G.full_batch_spec(g)
        assert fb["train"]["batch"] == 16 and fb["train"]["twin_loss"]["groups"] == 4


def test_staged_specs_match_what_the_jobs_read():
    for c in ("c0", "c1a", "c1b", "c2"):
        st = J.staged_specs(c)
        js = jobs_of(f"b2cue_{c}.json" if c == "c0" else f"b2cue_{c}_0p3.json")
        gate = next(j for j in js if j["name"] == "gate")["cmd"]
        assert {f"b2cue_gate_{sh}.json" for sh in J.GATE_SHAPES[c]} == {os.path.basename(x) for x in gate if x.endswith(".json")}
        g = st["b2cue_gate_D1.json"]
        assert g["train"]["init_from"].endswith("/" + J.GATE_INIT[c])
        # every model the jobs read from the uploads is one the campaign uploads: the gate's init is among them
        reads = {x for j in js for x in json.dumps(j).replace('"', " ").split() if x.startswith(J.U + "/") and x.endswith(".safetensors")}
        assert J.U + "/" + J.GATE_INIT[c] in reads, c
        for name, spec in st.items():
            if name.startswith(("start_", "a31_spec_")):
                job = next(j for j in js if j["name"] == spec["name"])          # the scoring job that links <name>.safetensors
                assert f"{J.U}/{name}" in job["cmd"][2]
    a = J.staged_specs("c1b")["a31_spec_N_1_0p3.json"]["train"]["twin_loss"]
    assert a["lambda"] == a["mu"] == 0.3
    assert J.staged_specs("c1b")["a31_spec_M_50_0p3.json"]["train"]["twin_loss"]["mu"] == 0.0
    assert "twin_loss" not in J.staged_specs("c1b")["a31_spec_P_1.json"]["train"]


def rec(val, dec, nui):
    return {"complete": True, "val_loss": val, "dec_d": dec, "nui": nui}


def test_select_rule_cases():
    D0 = rec(4.0, 0.0, 0.10)
    d = S.decide({"D0": D0, "Dn_0p1": rec(4.0, 0.20, 0.10), "Dn_0p3": rec(4.05, 0.30, 0.104), "Dn_1p0": rec(4.07, 0.50, 0.20)})
    assert d["chosen"] == 0.3 and d["runs"]["1.0"]["eligible"] and not d["runs"]["1.0"]["nuisance_ok"]
    d = S.decide({"D0": D0, "Dn_0p1": rec(4.0, 0.297, 0.10), "Dn_0p3": rec(4.0, 0.30, 0.10), "Dn_1p0": rec(4.1, 0.9, 0.1)})
    assert d["chosen"] == 0.1 and not d["runs"]["1.0"]["eligible"]          # within 0.005 of the best: the smaller lambda
    d = S.decide({"D0": D0, "Dn_0p1": rec(4.0, 0.2, 0.2), "Dn_0p3": rec(4.0, 0.3, 0.2), "Dn_1p0": {"complete": False, "why": "x"}})
    assert d["chosen"] == 0.1 and "smallest eligible" in d["why"]
    d = S.decide({"D0": D0, "Dn_0p1": rec(4.2, 0.2, 0.1), "Dn_0p3": {"complete": False, "why": "x"}, "Dn_1p0": rec(4.3, 0.3, 0.1)})
    assert d["outcome"] == "no_arm" and d["chosen"] is None
    d = S.decide({"D0": {"complete": False, "why": "status FAILED"}, "Dn_0p1": rec(4, 1, 0), "Dn_0p3": rec(4, 1, 0), "Dn_1p0": rec(4, 1, 0)})
    assert d["outcome"] == "void_selection"


def test_select_is_void_without_evidence(tmp_path):
    (tmp_path / "upload").mkdir()
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "colab_chat" / "b2cue_select.py"), "--chat", str(tmp_path), "--out", str(tmp_path / "o")],
                       capture_output=True, text=True)
    assert r.returncode == 0
    d = json.loads((tmp_path / "o" / "decisions.json").read_text())
    assert d["outcome"] == "void_selection" and d["chosen"] is None


def ev(loss=2.0, gn=3.0, tl=0.5, da=0.1, mask=True):
    t = {"loss": tl, "d_a": da, "d_b": 0.2, "d_n": 0.05}
    if mask:
        t["answer_mask"] = True
    return {"loss": loss, "grad_norm": gn, "twin": t}


def test_compare_am_cases():
    cp = {1: {"loss": 2.0}, 2: {"loss": 1.9}}
    ok = {1: ev(), 2: ev()}
    assert G.compare_am(ok, {1: ev(), 2: ev(gn=3.01)}, cp) == []
    assert G.compare_am(None, ok, cp)
    assert any("answer_mask" in f for f in G.compare_am({1: ev(mask=False), 2: ev()}, ok, cp))
    assert any("plain run" in f for f in G.compare_am({1: ev(loss=2.001), 2: ev()}, {1: ev(loss=2.001), 2: ev()}, cp))
    assert any("d_a" in f for f in G.compare_am(ok, {1: ev(da=0.13), 2: ev()}, cp))
    assert any("not finite" in f for f in G.compare_am({1: ev(tl=float("nan")), 2: ev()}, ok, cp))


def early_cli(*args, env=None):
    return subprocess.run([sys.executable, str(ROOT / "tools" / "colab_chat" / "b2cue_early.py"), *args], capture_output=True, text=True,
                          env={**os.environ, "B2CUE_POLL_S": "0.05", **(env or {})}, timeout=60)


def test_barrier_lifts_only_when_every_job_is_home(tmp_path):
    (tmp_path / "HOME_a").write_text("1")
    p = subprocess.Popen([sys.executable, str(ROOT / "tools" / "colab_chat" / "b2cue_early.py"), "--barrier", "--out", str(tmp_path / "b"),
                          "--chat", str(tmp_path), "--jobs", "a,b"], env={**os.environ, "B2CUE_POLL_S": "0.05"})
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            p.wait(1.0)
        assert not (tmp_path / "b" / "barrier.json").exists()
        (tmp_path / "HOME_b").write_text("1")
        assert p.wait(10) == 0
    finally:
        p.kill()
    assert json.loads((tmp_path / "b" / "barrier.json").read_text())["jobs"] == ["a", "b"]


def write_events(d, rows):
    d.mkdir(parents=True, exist_ok=True)
    (d / "events.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_reproduction_check(tmp_path):
    ref = [{"event": "step", "step": s, "loss": 3.0 - s / 100} for s in range(1, 51)]
    write_events(tmp_path / "repro_A", [dict(e, loss=e["loss"] * 1.005) for e in ref])
    (tmp_path / "ref.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ref))
    r = early_cli("--repro", "--job", "repro_A", "--ref", str(tmp_path / "ref.jsonl"), "--steps", "50", "--out", str(tmp_path / "o"), "--chat", str(tmp_path))
    assert r.returncode == 0 and json.loads((tmp_path / "o" / "repro.json").read_text())["pass"]
    write_events(tmp_path / "repro_A", [dict(e, loss=e["loss"] * 1.02) for e in ref[:49]])
    early_cli("--repro", "--job", "repro_A", "--ref", str(tmp_path / "ref.jsonl"), "--steps", "50", "--out", str(tmp_path / "o"), "--chat", str(tmp_path))
    o = json.loads((tmp_path / "o" / "repro.json").read_text())
    assert not o["pass"] and 50 in o["failed_steps"] and 1 in o["failed_steps"]
    js = jobs_of("b2cue_c1b_0p3.json")
    hv = [j["name"] for j in js if "spec" in j]
    assert hv[-1] == "repro_A"
    s = next(j for j in js if j["name"] == "repro_A")["spec"]
    assert s["train"]["steps"] == 50 and s["arch"] == J.A_ARCH and s["train"]["init_from"].endswith("/base_A.safetensors")


def test_watch_stops_on_a_non_finite_loss(tmp_path):
    write_events(tmp_path / "sft_X", [{"event": "step", "step": 1, "loss": 2.0}, {"event": "step", "step": 2, "loss": None}])
    r = early_cli("--watch", "--job", "sft_X", "--out", str(tmp_path / "w"), "--chat", str(tmp_path), "--at", "1", "--steps", "3")
    assert r.returncode == 1, r.stderr   # a stop exits 1
    w = json.loads((tmp_path / "w" / "watch.json").read_text())
    assert "non-finite" in w["health_stop"] and w["stopped_pid"] is None


def test_watch_records_readouts_and_missing_steps(tmp_path):
    rows = [{"event": "step", "step": s, "loss": 3.0 - s / 10} for s in (1, 2, 3)] + [{"event": "eval", "step": 2, "val_loss": 3.5}]
    write_events(tmp_path / "sft_X", rows)
    (tmp_path / "STATUS_sft_X").write_text("FINISHED\n")
    r = early_cli("--watch", "--job", "sft_X", "--out", str(tmp_path / "w"), "--chat", str(tmp_path), "--at", "2,3", "--steps", "3")
    assert r.returncode == 1, r.stderr   # a missing readout is unread: exit 1
    w = json.loads((tmp_path / "w" / "watch.json").read_text())
    assert w["at"] == {"2": {"sft_loss": 3.0 - 2 / 10, "val_loss": 3.5}} and w["readout_missing"] == ["3"] and w["complete"]


def test_a_corrupt_log_line_is_unread_never_dropped(tmp_path):
    (tmp_path / "sft_X").mkdir()
    rows = [json.dumps({"event": "step", "step": s, "loss": 2.0}) for s in (1, 2, 3)]
    (tmp_path / "sft_X" / "events.jsonl").write_text(rows[0] + "\n{broken\n" + rows[1] + "\n" + rows[2] + "\n")
    (tmp_path / "STATUS_sft_X").write_text("FINISHED")
    r = early_cli("--watch", "--job", "sft_X", "--out", str(tmp_path / "w"), "--chat", str(tmp_path), "--at", "2", "--steps", "3")
    assert r.returncode == 0
    w = json.loads((tmp_path / "w" / "watch.json").read_text())
    assert w["health"] == "unread" and "does not parse" in w["health_unread"]
    import b2cue_early as E
    (tmp_path / "log").write_text(rows[0] + "\n" + rows[1][:10])          # a line still being written: tolerated while running
    assert len(E.strict_events(str(tmp_path / "log"), ended=False)) == 1
    with pytest.raises(ValueError):
        E.strict_events(str(tmp_path / "log"), ended=True)


def test_watch_ok_needs_every_readout_and_a_finished_job(tmp_path):
    rows = [{"event": "step", "step": s, "loss": 3.0} for s in (1, 2, 3)] + [{"event": "eval", "step": s, "val_loss": 3.5} for s in (1, 2, 3)]
    write_events(tmp_path / "sft_X", rows)
    (tmp_path / "STATUS_sft_X").write_text("FINISHED")
    early_cli("--watch", "--job", "sft_X", "--out", str(tmp_path / "w"), "--chat", str(tmp_path), "--at", "1,2,3", "--steps", "3")
    assert json.loads((tmp_path / "w" / "watch.json").read_text())["health"] == "ok"
    (tmp_path / "STATUS_sft_X").write_text("FAILED")
    early_cli("--watch", "--job", "sft_X", "--out", str(tmp_path / "w2"), "--chat", str(tmp_path), "--at", "1,2,3", "--steps", "3")
    assert json.loads((tmp_path / "w2" / "watch.json").read_text())["health"] == "unread"


def test_rendered_specs_are_already_resolved(tmp_path, monkeypatch):
    """jobs_vm.spec_for changes nothing but name and out_dir in any training spec (launch round 2: the home replay and the scorer
    compare the resolved spec with the jobs file's plus those two fields)."""
    import importlib
    monkeypatch.setenv("JOBS_CHAT_DIR", str(tmp_path))
    import jobs_vm
    importlib.reload(jobs_vm)
    for p in FILES:
        jobs = {j["name"]: j for j in json.loads(p.read_text())["jobs"]}
        for n, j in jobs.items():
            if "spec" not in j:
                continue
            got = json.loads(Path(jobs_vm.spec_for(j, jobs)).read_text().replace(str(tmp_path), "/content/chat"))
            want = dict(j["spec"], name=j.get("model_name", n), out_dir=f"/content/chat/{n}")
            assert got == want, (p.name, n)


def test_early_records_certify_their_evidence(tmp_path):
    import b2cue_early as E
    rows = [{"event": "step", "step": s, "loss": 3.0} for s in range(1, 6)]
    d = E.events_digest(rows, 3)
    assert d == E.events_digest(rows + [{"event": "step", "step": 9, "loss": 1.0}], 3)       # later lines do not change it
    assert d != E.events_digest(rows[:2] + [{"event": "step", "step": 3, "loss": 2.0}], 3)   # a changed step does
