"""experiments/tiny_chat_2026-10/analyze_x5.py (amendment 5, arm X5) on synthetic summaries and per-item records:
HX5-main's item-level paired test and each of its five verdicts, the validity gate (FLOOR), the reproduction
check of the L16 re-runs against E1's home summaries, the refusal of incomplete or mismatched inputs, the
isoplot, and analyze.py's E1_x5_20m block."""
import json
from pathlib import Path
import random
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "tiny_chat_2026-10"
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(ROOT / "tools" / "colab_chat"))
sys.path.insert(0, str(EXP / "x5"))
import analyze_x5 as ax  # noqa: E402
import bitid  # noqa: E402
import tiny_chat_run as tcr  # noqa: E402

FAMS = [("recall", 432), ("correction", 216), ("topic", 180), ("knowledge", 60), ("instruction", 144)]


def items(seed, model, plus, minus, twin=1.0, noctx=0.0):
    """Carry-5's 1,032 item records for one model. Y's 888 items have twin = 1; self is 1 except on the items
    a 'paired difference' targets: for the X model, `plus` items get self 1 where the control has 0, `minus`
    items get 0 where the control has 1 (the control's self: 1 on the plus items' complement)."""
    rng = random.Random(1000 * seed + 7)
    ids = [f"{f}-{k}" for f, n in FAMS for k in range(n)]
    prim = [i for i in ids if not i.startswith("instruction")]
    order = prim[:]
    rng.shuffle(order)
    up, down = set(order[:plus]), set(order[plus:plus + minus])
    recs = []
    for i in ids:
        fam = i.split("-")[0]
        if model == "x":
            s = 0.0 if i in down else 1.0
        else:
            s = 0.0 if i in up else 1.0
        g = s
        recs.append({"id": i, "family": fam, "slot": "x", "distance": (sum(map(ord, i)) % 4) + 1 if fam != "knowledge" else None,
                     "interference": 0, "target_turn": 4,
                     "fc": {"self": {"acc": s, "acc_norm": s}, "twin": {"acc": twin, "acc_norm": twin},
                            "gold": {"acc": g, "acc_norm": g}, "noctx": {"acc": noctx, "acc_norm": noctx}}})
    return recs


def write_post(d, cell, seed, recs, fw=3.84, sha="abc", share=35):
    """A post job's directory as tiny_chat_run.py post writes it, its pools computed from RECS exactly as
    carry5_pools does."""
    m = d / "carry5" / "m"
    m.mkdir(parents=True)
    (m / "items.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    pools = tcr.carry5_pools(str(m / "items.jsonl"))
    allm = {k: sum(r["fc"][k]["acc"] for r in recs) / len(recs) for k in ("gold", "self", "twin", "noctx")}
    pool = {k: v["mean"] for k, v in pools["acc"]["primary"].items()}
    ctx = {k: v["mean"] for k, v in pools["acc"]["context"].items()}
    s = {"cell": cell, "seed": seed, "share": share, "primary": pool["self"] - pool["twin"],
         "primary_all_items": allm["self"] - allm["twin"], "sft_weights_sha256": sha,
         "carry5": {"acc": allm, "acc_norm": allm, "pool": pool, "ctx": ctx, "chance": 0.25, "pools": pools},
         "panels": {"fineweb": fw, "dialogue": 3.24}, "sft_evals": [[1500, 3.0]]}
    (d / "summary.json").write_text(json.dumps(s))
    return s


def art_for(tmp, plus, minus, twin=1.0, noctx=0.0, perturb_home=False, drop=None, plain=True):
    art = tmp / "art"
    for s in (1, 2, 3):
        p, n = (plus[s - 1], minus[s - 1]) if isinstance(plus, list) else (plus, minus)
        if drop != ("x", s):
            write_post(art / "e1_x5" / f"post_e1_x5_X5_r{s}", "X5", s, items(s, "x", p, n, twin, noctx), fw=3.80)
        c = items(s, "c", p, n, twin, noctx)
        if drop != ("ctl", s):
            write_post(art / "e1_x5_ctl" / f"post_e1_x5_ctl_L16_r{s}", "L16", s, c)
        home = write_post(art / "e1" / f"post_e1_L16_r{s}", "L16", s, c)
        if perturb_home and s == 2:
            home["sft_weights_sha256"] = "other"
            (art / "e1" / f"post_e1_L16_r{s}" / "summary.json").write_text(json.dumps(home))
    if plain:  # other plain 20M runs for the isoplot: Y falls 0.1 per nat of FineWeb loss
        for cell, fw in (("L4", 3.87), ("L32", 3.92)):
            for s in (1, 2, 3):
                r = items(s, "c", 0, 0)
                write_post(art / "e1" / f"post_e1_{cell}_r{s}", cell, s, r, fw=fw)
    return art


def run(art, *extra):
    p = subprocess.run([sys.executable, str(EXP / "analyze_x5.py"), "--art", str(art), *extra],
                       capture_output=True, text=True)
    return p.returncode, (json.loads(p.stdout) if p.returncode == 0 else p.stderr)


def test_families_match_the_runner():
    assert ax.PRIMARY_FAMILIES == tcr.PRIMARY_FAMILIES and ax.CONTEXT_FAMILIES == tcr.CONTEXT_FAMILIES


@pytest.mark.parametrize("plus,minus,want", [
    (60, 0, "SUPPORTED"),                        # D = +0.068 on 888 items
    (18, 0, "SIGNIFICANT_BELOW_SESOI"),          # D = +0.020, se 0.0027
    (0, 45, "CONTRADICTED"),                     # D = -0.051
    (40, 40, "EQUIVALENT"),                      # D = 0, se 0.0069: the 90% interval inside +-0.03
    (444, 426, "NOT_RESOLVED"),                  # D = +0.020, se 0.019
])
def test_verdicts(tmp_path, plus, minus, want):
    rc, res = run(art_for(tmp_path, plus, minus))
    assert rc == 0, res
    h = res["E1_x5_20m"] if "E1_x5_20m" in res else res
    hx = h["HX5_main"]
    assert hx["verdict"] == want, hx
    assert hx["n"] == 3 * 888
    assert abs(hx["mean"] - (plus - minus) / 888) < 1e-12
    for s in ("1", "2", "3"):  # the mean of the item differences is Y(X5) - Y(L16) in every seed
        r = hx["per_seed"][s]
        assert abs(r["mean_difference"] - (r["Y_X5"] - r["Y_L16_rerun"])) < 1e-12
        assert abs(r["Y_L16_home"] - r["Y_L16_rerun"]) < 1e-12
    assert all(v["exact"] for v in h["reproduction"].values())
    assert h["floor_ok"] == {"X5": True, "L16": True}
    assert set(h["descriptive_by_family"]) == {"recall", "correction", "topic", "knowledge"}


def test_floor(tmp_path):
    rc, res = run(art_for(tmp_path, 60, 0, twin=1.0, noctx=0.95))  # twin minus question alone 0.05 < 0.10
    assert rc == 0, res
    assert res["HX5_main"]["verdict"].startswith("FLOOR")


def test_reproduction_mismatch_is_reported_not_hidden(tmp_path):
    rc, res = run(art_for(tmp_path, 60, 0, perturb_home=True))
    assert rc == 0, res
    assert res["reproduction"]["2"]["exact"] is False
    assert "sft_weights_sha256" in res["reproduction"]["2"]["differences"]
    assert res["HX5_main"]["control_reproduces_home"] == {"1": True, "2": False, "3": True}
    assert res["HX5_main"]["verdict"] == "SUPPORTED"


def test_incomplete_refused_and_partial(tmp_path):
    art = art_for(tmp_path, 60, 0, drop=("ctl", 3))
    rc, err = run(art)
    assert rc != 0 and "incomplete" in err
    rc, res = run(art, "--partial")
    assert rc == 0 and res["label"] == "DESCRIPTIVE, PARTIAL" and "HX5_main" not in res


def test_items_must_belong_to_the_summary(tmp_path):
    art = art_for(tmp_path, 60, 0)
    p = art / "e1_x5" / "post_e1_x5_X5_r1" / "summary.json"
    s = json.loads(p.read_text())
    s["primary"] += 0.01
    p.write_text(json.dumps(s))
    rc, err = run(art)
    assert rc != 0 and "not the same run" in err


def test_isoplot(tmp_path):
    rc, res = run(art_for(tmp_path, 60, 0))
    assert rc == 0, res
    iso = res["isoplot_all_plain_20m"]
    assert iso["runs"] == 9 and iso["cells"] == ["e1_L16", "e1_L32", "e1_L4"]
    assert all(v["extrapolation"] for v in iso["X5"].values())  # 3.80 is below the plain runs' 3.84-3.92
    assert res["isoplot_s35_plain_20m"]["runs"] == 9


def test_analyze_runs_the_block(tmp_path):
    art = art_for(tmp_path, 60, 0)
    design = tmp_path / "design.json"
    design.write_text(json.dumps({"levels": [0, 17, 35, 70], "e1_share": 35, "mode": "passes", "final": True}))
    p = subprocess.run([sys.executable, str(EXP / "analyze.py"), "--art", str(art), "--design", str(design),
                        "--partial"], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    res = json.loads(p.stdout)
    assert res["E1_x5_20m"]["HX5_main"]["verdict"] == "SUPPORTED"


def _run_dir(root, name, steps, total, bump=None):
    d = root / name
    d.mkdir(parents=True)
    spec = {"name": "tc_e1_L16_r1", "arch": {"custom": {"d": 320}}, "train": {"steps": total, "lr": 0.001, "seed": 1},
            "out_dir": f"/content/chat/{name}"}
    (d / f"{name}_spec.json").write_text(json.dumps(spec))
    ev = [{"event": "start", "steps": total}, {"event": "model", "d": 320, "params": 25078080}]
    for k in range(1, steps + 1):
        loss = 9.78 - k * 1e-3 + (1e-7 if bump == k else 0.0)
        ev.append({"event": "step", "grad_norm": 1.8, "loss": loss, "lr": 2e-6 * k, "step": k})
        if k % 5 == 0:
            ev.append({"event": "eval", "step": k, "val_loss": 9.0 - k * 1e-3})
    (d / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ev))
    return d


@pytest.mark.parametrize("bump,ok", [(None, True), (7, False)])
def test_bitid(tmp_path, bump, ok):
    e1 = _run_dir(tmp_path, "pt_e1_L16_r1", 40, 40)
    rep = _run_dir(tmp_path, "replay_e1_L16_r1", 10, 10, bump=bump)
    out = tmp_path / "bitid.json"
    p = subprocess.run([sys.executable, str(EXP / "x5" / "bitid.py"), "--replay", str(rep), "--e1", str(e1),
                        "--commit", "abc", "--e1-commit", "dc29b5e6", "--out", str(out)], capture_output=True, text=True)
    r = json.loads(out.read_text())
    assert (p.returncode == 0) is ok and r["ok"] is ok
    assert r["steps_compared"] == 10 and r["evals_compared"] == [5, 10] and r["spec_equal_but_steps"]
    if not ok:
        assert r["differences"][0] == {"step": 7, "field": "loss", "replay": 9.78 - 7e-3 + 1e-7, "e1": 9.78 - 7e-3}
    assert r["receipt"].startswith("amendment 5:") and "model_event" not in r


def _set_model(run_dir, extra):
    """Rewrite the model event of a _run_dir folder with fields added (None removes a field)."""
    p = run_dir / "events.jsonl"
    ev = [json.loads(x) for x in p.read_text().splitlines()]
    for e in ev:
        if e["event"] == "model":
            for k, v in extra.items():
                if v is None:
                    e.pop(k, None)
                else:
                    e[k] = v
    p.write_text("".join(json.dumps(e) + "\n" for e in ev))


@pytest.mark.parametrize("extra,ok", [
    ({"final_norm": True, "norm_placement": "pre"}, True),       # c6365b88's fields at their fixed values
    ({"final_norm": True}, True),
    ({"final_norm": False, "norm_placement": "pre"}, False),     # not the fixed value
    ({"norm_placement": {"attn": "post", "ffn": "pre"}}, False),
    ({"window": 64}, False),                                     # not a field a later commit added
    ({"d": 384}, False),                                         # a shared field that differs
    ({"params": None}, False),                                   # a field in E1's event alone
])
def test_bitid_model_event_rule(tmp_path, extra, ok):
    """Jonathan's ruling, 27 Oct 2026 (PREREGISTRATION.md A6.7): the model events agree on the fields both commits
    log; a field only the replay logs passes only when bitid.LOGGED_LATER lists it at its fixed value."""
    e1 = _run_dir(tmp_path, "pt_e1_60m_L32_r2", 40, 40)
    rep = _run_dir(tmp_path, "replay_e1_60m_L32_r2", 10, 10)
    _set_model(rep, extra)
    out = tmp_path / "bitid.json"
    p = subprocess.run([sys.executable, str(EXP / "x5" / "bitid.py"), "--replay", str(rep), "--e1", str(e1),
                        "--commit", "abc", "--e1-commit", "a90ec0e2", "--out", str(out)], capture_output=True, text=True)
    r = json.loads(out.read_text())
    assert (p.returncode == 0) is ok and r["ok"] is ok and r["model_event_equal"] is ok
    assert r["receipt"].startswith("amendment 6:") and r["n_differences"] == 0
    m = r["model_event"]
    if "params" in extra:
        assert m["e1_only"] == {"params": 25078080}
    elif "d" in extra:
        assert m["shared_differences"] == {"d": {"replay": 384, "e1": 320}}
    else:
        assert set(m["replay_only"]) == set(extra)
        assert all(v["ok"] is (k in bitid.LOGGED_LATER and v["value"] == bitid.LOGGED_LATER[k])
                   for k, v in m["replay_only"].items())


COLAB = ROOT / "tools" / "colab_chat"


def _jobs(name):
    return {j["name"]: j for j in json.loads((COLAB / name).read_text())["jobs"]}


def test_x5_jobs_match_e1_L16_but_for_attn_res():
    """Amendment 5: X5's runs are E1's L16 runs with arch.custom.attn_res added: the same steps, tokens, data,
    lr (E1 L16's chosen rate), seeds, SFT and post command (but its spec path and job names)."""
    e1, x5 = _jobs("tiny_chat_e1.json"), _jobs("tiny_chat_e1_x5.json")
    choice = json.loads((EXP / "stage0" / "lr_choice.json").read_text())["lr"]
    xchoice = json.loads((EXP / "stage0" / "lr_choice_x5.json").read_text())["lr"]
    assert xchoice == {"20m_X5": choice["20m_L16"]}
    for s in (1, 2, 3):
        a, b = e1[f"pt_e1_L16_r{s}"], x5[f"pt_e1_x5_X5_r{s}"]
        assert b["spec"]["arch"]["custom"].pop("attn_res") == {"form": "full", "scale": "sources"}
        assert {**b["spec"], "name": None} == {**a["spec"], "name": None}
        assert b["spec"]["train"]["lr"] == choice["20m_L16"]
        assert {k: v for k, v in b.items() if k not in ("name", "model_name", "spec")} == \
               {k: v for k, v in a.items() if k not in ("name", "model_name", "spec")}
        pa, pb = e1[f"post_e1_L16_r{s}"]["cmd"], x5[f"post_e1_x5_X5_r{s}"]["cmd"]
        if "--ember-commit" in pb:
            pb = pb[:pb.index("--ember-commit")]
        norm = lambda c: [x.replace("e1_x5_X5", "e1_L16").replace("x5/specs/shape_20m_X5", "specs/shape_20m_L16")
                          .replace("tc_e1_x5_X5", "tc_e1_L16") for x in c]  # noqa: E731
        assert norm(pb) == [x.replace("--cell", "--cell") for x in pa[:pa.index("--cell") + 1]] + ["X5"] + \
            pa[pa.index("--cell") + 2:]
        assert x5[f"post_e1_x5_X5_r{s}"]["outputs"][1:3] == ["carry5/m/items.jsonl", "carry5/m/gen.jsonl"]
    g = x5["gate"]["cmd"]
    for need in ("x5/cuda_gate_x5.json", "x5/plain_bitid.json", "x5/ember_gate.json"):
        assert any(x.endswith(need) for x in g), need
    ctl = _jobs("tiny_chat_e1_x5_ctl.json")
    for s in (1, 2, 3):
        c, a = ctl[f"post_e1_x5_ctl_L16_r{s}"]["cmd"], e1[f"post_e1_L16_r{s}"]["cmd"]
        i, j = c.index("--pt-weights"), a.index("--pt-weights")
        assert c[i + 1] == f"/content/chat/upload/tc_e1_L16_r{s}.safetensors"
        if "--ember-commit" in c:  # A5.4's pin, the same for X5's posts
            k = c.index("--ember-commit")
            assert x5[f"post_e1_x5_X5_r{s}"]["cmd"][-2:] == c[k:k + 2]
            c = c[:k] + c[k + 2:]
        assert c[:i] == a[:j] and c[i + 4:] == a[j + 4:]  # the same post command but the weights' location
    s0 = _jobs("tiny_chat_s0_x5.json")
    rep = s0["replay_e1_L16_r1"]["spec"]
    assert rep["train"]["steps"] == rep["train"]["lr_schedule"]["warmup_steps"] == 500
    assert {**rep, "train": {**rep["train"], "steps": 10000}} == {**e1["pt_e1_L16_r1"]["spec"]}


def test_gate_needs_the_receipts(tmp_path):
    ok, bad = tmp_path / "ok.json", tmp_path / "bad.json"
    ok.write_text(json.dumps({"ok": True}))
    bad.write_text(json.dumps({"ok": False}))
    base = [sys.executable, str(COLAB / "tiny_chat_run.py"), "gate", "--out", str(tmp_path / "g"), "--jobs",
            "tools/colab_chat/tiny_chat_e1_x5.json", "--choice", "experiments/tiny_chat_2026-10/stage0/lr_choice_x5.json",
            "--pilot-verdict", "experiments/tiny_chat_2026-10/pilot/verdict.json", "--need-pilot", "PASS"]
    p = subprocess.run(base + ["--need-ok", str(ok)], cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    for f in (bad, tmp_path / "missing.json"):
        p = subprocess.run(base + ["--need-ok", str(ok), "--need-ok", str(f)], cwd=ROOT, capture_output=True, text=True)
        assert p.returncode == 1 and "ok is" in p.stdout


def test_home_pick_over_split_seeds(tmp_path):
    """The seeds run on machines of their own, so pick_e1_x5_X5 runs at home: tiny_chat_run.py pick over the
    home post folders (their outputs carry sft_spec.json and the SFT model) chooses by the registered rule and
    copies the best seed's model, checked against its summary's sha256."""
    import hashlib
    runs = []
    for s, y in ((1, -0.05), (2, -0.02), (3, -0.02)):
        d = tmp_path / f"e1_x5_r{s}" / f"post_e1_x5_X5_r{s}"
        (d / "sft").mkdir(parents=True)
        w = d / "sft" / f"tc_sft_X5_r{s}.safetensors"
        w.write_bytes(f"weights {s}".encode())
        (d / "sft" / f"tc_sft_X5_r{s}.gguf").write_bytes(f"gguf {s}".encode())
        (d / "sft_spec.json").write_text(json.dumps({"name": f"tc_sft_X5_r{s}"}))
        (d / "summary.json").write_text(json.dumps({"seed": s, "primary": y,
                                                    "sft_weights_sha256": hashlib.sha256(w.read_bytes()).hexdigest()}))
        runs += ["--run", str(d)]
    out = tmp_path / "e1_x5" / "pick_e1_x5_X5"
    p = subprocess.run([sys.executable, str(COLAB / "tiny_chat_run.py"), "pick", "--out", str(out), *runs],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    best = json.loads((out / "best.json").read_text())
    assert best["best"]["seed"] == 2 and best["weights_match_summary"]  # the tie goes to the lower seed
    assert (out / "model.safetensors").read_bytes() == b"weights 2" and (out / "model.gguf").read_bytes() == b"gguf 2"
    x5_post = _jobs("tiny_chat_e1_x5.json")["post_e1_x5_X5_r2"]["outputs"]
    assert {"sft_spec.json", "sft/tc_sft_X5_r2.safetensors", "sft/tc_sft_X5_r2.gguf"} <= set(x5_post)


def test_ember_pin_is_the_gated_commit():
    """A5.4: X5's and the re-run's post jobs generate at the ember commit ember's gate receipt names; until that
    receipt is home neither carries a pin (and the X5 gate refuses without the receipt)."""
    rec = EXP / "x5" / "ember_gate.json"
    pins = set()
    for name, prefix in (("tiny_chat_e1_x5.json", "post_e1_x5_X5_r"), ("tiny_chat_e1_x5_ctl.json", "post_e1_x5_ctl_L16_r")):
        for n, j in _jobs(name).items():
            if n.startswith(prefix):
                c = j["cmd"]
                pins.add(c[c.index("--ember-commit") + 1] if "--ember-commit" in c else None)
    assert len(pins) == 1
    pin = pins.pop()
    if rec.exists():
        r = json.loads(rec.read_text())
        assert r["ok"] is True and r["ember_commit"] == pin and len(pin) == 40
    else:
        assert pin is None
