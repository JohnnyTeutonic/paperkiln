"""mtstudio's own-context floor (lane amendment A34: `train.twin_loss.floor` with `data.twin_floor_ref`) end to end on CPU.
Claude Opus 5.5, 15 Nov 2026. Wiring checks on test_mtstudio_twin.py's two-variable lookup, not a research claim:
- floor 0, or no floor key, is A31's run bit for bit (every step's SFT loss, twin record and gradient norm); with A31_BIN set to
  a build of paperkiln 290b6580 (A31's launch commit), this build's run without the floor equals that build's, step for step;
- the scorer and the trainer agree: references scored by `mtstudio score` on the start weights give a step-1 gap of about 0;
- the references are read at the weights the step's forward uses (step - 1), interpolated linearly between knots (with the
  learning rate 0 the logged gap is the reference offset itself);
- a binding floor lifts the own-context terms above those of the same margin run without it, and its gap closes;
- a resumed floor run reproduces the uninterrupted one;
- a reference file that does not fit the twins or the run is refused.
Needs MTSTUDIO_BIN (or ./build/mtstudio); skipped otherwise.
"""
# ruff: noqa: F811  (test_mtstudio_twin's `tmp` fixture is imported and then requested by name)
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_mtstudio_chat import BIN  # noqa: E402
from test_mtstudio_twin import EOT, TRIPLETS, run, spec, steps, tmp, twin_train  # noqa: E402,F401

needs_bin = pytest.mark.skipif(BIN is None, reason="set MTSTUDIO_BIN to a built mtstudio")
A31_BIN = os.environ.get("A31_BIN")
ENV = {"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin"}


def floor_train(floor, **kw):
    t = twin_train(**kw)
    t["twin_loss"]["floor"] = floor
    return t


def own_scores(tmp, target, name):
    """Per group, lp(yA|A)/n_A and lp(yB|B)/n_B and the reply lengths, scored by `mtstudio score` on `target`."""
    req = []
    for t in TRIPLETS:
        for side in ("A", "B"):
            req.append({"prompt": t[side][0], "continuation": t[side][1] + EOT})
    rp, out = tmp / f"{name}.req.jsonl", tmp / f"{name}.scores.jsonl"
    rp.write_text("".join(json.dumps(r) + "\n" for r in req), encoding="utf-8")
    r = subprocess.run([BIN, "score", str(target), "--in", str(rp), "--out", str(out)], capture_output=True, text=True,
                       timeout=600, env=ENV)
    assert r.returncode == 0, r.stderr
    rec = [json.loads(x) for x in out.read_text().splitlines() if x.strip()]
    res = []
    for g in range(len(TRIPLETS)):
        a, b = rec[2 * g], rec[2 * g + 1]
        res.append((a["sum"] / len(a["logprobs"]), b["sum"] / len(b["logprobs"]), len(a["logprobs"]), len(b["logprobs"])))
    return res


def write_ref(path, own, knots, offsets=None, groups=None, **override):
    """One line per group: the scored own terms (plus an offset per knot) at every knot."""
    offsets = offsets or [0.0] * len(knots)
    lines = []
    for g, (a, b, na, nb) in enumerate(own[:groups] if groups else own):
        r = {"g": g, "n_a": na, "n_b": nb, "steps": list(knots), "a": [a + o for o in offsets], "b": [b + o for o in offsets]}
        r.update(override)
        lines.append(json.dumps(r))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def start(tmp):
    """A start model (30 plain steps) and its own-context scores."""
    p = spec(tmp, "fstart")
    run(p)
    w = tmp / "fstart" / "fstart.safetensors"
    assert w.exists()
    return w, own_scores(tmp, p, "fstart")


def from_start(tmp, name, train, data, start_w):
    tr = dict(train)
    tr["init_from"] = str(start_w)
    return spec(tmp, name, tr, data)


@needs_bin
def test_floor_zero_is_a31(tmp):
    tw = {"twin_examples": str(tmp / "twins.jsonl")}
    a = steps(run(spec(tmp, "f_absent", twin_train(lam=0.5, mu=0.0), tw)))
    b = steps(run(spec(tmp, "f_zero", floor_train(0.0, lam=0.5, mu=0.0), tw)))
    assert len(a) == len(b) == 30
    for x, y in zip(a, b):
        assert (x["loss"], x["twin"], x["grad_norm"], x["sft_rows_hash"]) == (y["loss"], y["twin"], y["grad_norm"], y["sft_rows_hash"])
    assert all("floor" not in e["twin"] for e in b)


@needs_bin
@pytest.mark.skipif(not A31_BIN, reason="set A31_BIN to a build of paperkiln 290b6580 (A31's launch commit)")
def test_no_floor_equals_a31_build(tmp):
    tw = {"twin_examples": str(tmp / "twins.jsonl")}
    for mu in (0.0, 0.2):
        p = spec(tmp, f"x_new_{mu}", twin_train(lam=0.5, mu=mu), tw)
        mine = steps(run(p))
        q = spec(tmp, f"x_old_{mu}", twin_train(lam=0.5, mu=mu), tw)
        r = subprocess.run([A31_BIN, "run", str(q)], capture_output=True, text=True, timeout=1800, env=ENV)
        assert r.returncode == 0, r.stderr
        theirs = [json.loads(x) for x in (tmp / f"x_old_{mu}" / "events.jsonl").read_text().splitlines() if x.strip()]
        theirs = [e for e in theirs if e["event"] == "step"]
        assert len(mine) == len(theirs) == 30
        for x, y in zip(mine, theirs):
            assert (x["loss"], x["twin"], x["grad_norm"], x["sft_rows_hash"]) == (y["loss"], y["twin"], y["grad_norm"], y["sft_rows_hash"])
        assert (tmp / f"x_new_{mu}" / f"x_new_{mu}.safetensors").read_bytes() == (tmp / f"x_old_{mu}" / f"x_old_{mu}.safetensors").read_bytes()


@needs_bin
def test_scorer_and_trainer_agree_at_step_1(tmp, start):
    w, own = start
    ref = write_ref(tmp / "ref_parity.jsonl", own, [0, 29])
    ev = run(from_start(tmp, "f_parity", floor_train(1.0, lam=0.5, mu=0.0, steps=2),
                        {"twin_examples": str(tmp / "twins.jsonl"), "twin_floor_ref": str(ref)}, w))
    fl = next(e for e in ev if e["event"] == "twin_floor")
    assert fl["floor"] == 1.0 and fl["groups"] == len(TRIPLETS) and fl["knots_group0"] == [0, 29]
    s1 = steps(ev)[0]["twin"]["floor"]
    assert s1["gap_absmax"] < 1e-4, s1          # the start weights' own terms, scored and trained, agree


@needs_bin
def test_references_read_at_step_minus_one_and_interpolated(tmp, start):
    w, own = start
    knots, offs = [0, 10, 29], [0.0, 1.0, -1.0]
    ref = write_ref(tmp / "ref_interp.jsonl", own, knots, offs)
    ev = run(from_start(tmp, "f_interp", floor_train(1e-6, lam=1e-6, mu=0.0, lr=0.0, steps=30),
                        {"twin_examples": str(tmp / "twins.jsonl"), "twin_floor_ref": str(ref)}, w))
    for e in steps(ev):
        wgt = e["step"] - 1
        want = wgt / 10.0 if wgt <= 10 else 1.0 - 2.0 * (wgt - 10) / 19.0
        assert abs(e["twin"]["floor"]["gap"] - want) < 1e-4, (e["step"], e["twin"]["floor"], want)


@needs_bin
def test_floor_holds_own_context(tmp, start):
    w, own = start
    tw = {"twin_examples": str(tmp / "twins.jsonl")}
    # references 0.5 nats a token above the start's own terms at every knot: the floor binds, so it must lift the own-context
    # terms above those of the same margin run without it (the margin's own fall is A31's finding on the real model, not a toy's)
    ref = write_ref(tmp / "ref_hold.jsonl", own, [0, 99], [0.5, 0.5])
    common = dict(lam=1.0, mu=0.0, margin=3.0, groups=6, steps=100, lr=0.002)
    m = from_start(tmp, "f_margin", twin_train(**common), tw, w)
    f = from_start(tmp, "f_floor", floor_train(3.0, **common), {**tw, "twin_floor_ref": str(ref)}, w)
    run(m)
    evf = steps(run(f))
    om, of = own_scores(tmp, m, "f_margin"), own_scores(tmp, f, "f_floor")
    mean_m = sum(a + b for a, b, _, _ in om) / (2 * len(om))
    mean_f = sum(a + b for a, b, _, _ in of) / (2 * len(of))
    assert mean_f > mean_m + 0.05, (mean_m, mean_f)
    assert evf[0]["twin"]["floor"]["active"] == 1.0                      # every side starts 0.5 below its reference
    assert evf[-1]["twin"]["floor"]["gap"] < evf[0]["twin"]["floor"]["gap"]   # and the gap closes
    assert all(0.0 <= e["twin"]["floor"]["active"] <= 1.0 for e in evf)


@needs_bin
def test_resume_with_floor(tmp, start):
    w, own = start
    ref = write_ref(tmp / "ref_resume.jsonl", own, [0, 11], [0.0, 0.5])
    data = {"twin_examples": str(tmp / "twins.jsonl"), "twin_floor_ref": str(ref)}
    whole = steps(run(from_start(tmp, "fr_whole", floor_train(1.0, steps=12, checkpoint_every=6), data, w)))
    p = from_start(tmp, "fr_cut", floor_train(1.0, steps=6, checkpoint_every=6), data, w)
    run(p)
    s = json.loads(p.read_text())
    s["train"]["steps"] = 12
    p.write_text(json.dumps(s), encoding="utf-8")
    tail = [e for e in steps(run(p)) if e["step"] > 6]
    assert [e["step"] for e in tail] == list(range(7, 13))
    for a, b in zip(whole[6:], tail):
        assert a["loss"] == b["loss"] and a["twin"] == b["twin"], (a, b)


@needs_bin
def test_floor_refusals(tmp, start):
    w, own = start
    tw = {"twin_examples": str(tmp / "twins.jsonl")}

    def refused(name, train, data, needle):
        r = run(from_start(tmp, name, train, data, w), ok=False)
        assert r.returncode != 0 and needle in r.stderr, (needle, r.stderr)
    good = write_ref(tmp / "ref_good.jsonl", own, [0, 29])
    refused("fx_noref", floor_train(1.0), tw, "go together")
    refused("fx_nofloor", twin_train(), {**tw, "twin_floor_ref": str(good)}, "go together")
    refused("fx_zero_ref", floor_train(0.0), {**tw, "twin_floor_ref": str(good)}, "go together")
    refused("fx_neg", floor_train(-1.0), {**tw, "twin_floor_ref": str(good)}, "floor must be finite")
    refused("fx_str", floor_train("1"), {**tw, "twin_floor_ref": str(good)}, "must be a number")
    cases = {
        "fewer": (write_ref(tmp / "r1.jsonl", own, [0, 29], groups=len(own) - 1), "lines for"),
        "len": (write_ref(tmp / "r2.jsonl", own, [0, 29], n_a=999), "reply lengths differ"),
        "start": (write_ref(tmp / "r3.jsonl", own, [1, 29]), "start at 0"),
        "reach": (write_ref(tmp / "r4.jsonl", own, [0, 20]), "start at 0 and reach 29"),
        "order": (write_ref(tmp / "r5.jsonl", own, [0, 29, 29]), "rise strictly"),
        "g": (write_ref(tmp / "r6.jsonl", own, [0, 29], g=7), "group index"),
        "shape": (write_ref(tmp / "r7.jsonl", own, [0, 29], a=[0.0]), "one length"),
    }
    for k, (path, needle) in cases.items():
        refused(f"fx_{k}", floor_train(1.0), {**tw, "twin_floor_ref": str(path)}, needle)
    nan = tmp / "r8.jsonl"
    nan.write_text(good.read_text().replace('"a": [', '"a": [NaN, ', 1), encoding="utf-8")
    refused("fx_nan", floor_train(1.0), {**tw, "twin_floor_ref": str(nan)}, "r8.jsonl")
