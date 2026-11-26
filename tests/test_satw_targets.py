"""train.symbol_targets.recency_weight (tools/mtstudio.cpp): SATW, SAT with the natural pointer targets that the
recency rule gets right weighted W in the pointer loss. End to end on CPU with the real mtstudio, on test_sat_targets'
synthetic corpora (turn markers, so state ids are inserted). Claude Opus 5.5, 26 Nov 2026.

The claims: W = 1 is SAT's spec bit for bit (and, with MTSTUDIO_SAT_BIN, SAT's reviewed trainer); the nearest-right
split the trainer logs is the recency rule recomputed here from the logged targets and, independently, from the label
file and the replayed windows (sat_windows.expected_counts); the weight sums are W x right + wrong (+ twin targets at
1); the implementation check passes with W < 1; a resume inside the scaffold equals the uninterrupted run; from the
cut, SATW's steps are SAT's and the plain spec's; the held-out eval splits its accuracies by the rule. With
SATW_N_MAIN_IDS (a copy of N.main.ids), the real label file's window-counted nearest-right share is recomputed and
compared with the label build's receipt.
"""
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools" / "colab_chat"))
import sat_windows as W  # noqa: E402
import test_sat_targets as S  # noqa: E402

needs_bin = S.needs_bin
SAT_BIN = os.environ.get("MTSTUDIO_SAT_BIN")
WEIGHT = 0.307
NEW_STEP_FIELDS = ("natural_pointer_nearest_right", "natural_pointer_nearest_wrong", "twin_pointer_nearest_right",
                   "twin_pointer_nearest_wrong", "pointer_weight_sum", "natural_pointer_weight_sum")
NEW_EVAL_FIELDS = ("nearest_right_targets", "nearest_wrong_targets", "pointer_acc_nearest_right",
                   "pointer_acc_nearest_wrong", "recency_acc_nearest_right", "recency_acc_nearest_wrong")


def cfg(tmp, weight=None, **kw):
    c = S.sat_cfg(tmp, **kw)
    if weight is not None:
        c["recency_weight"] = {"nearest_right": weight}
    return c


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("satw")
    return tmp, S.setup(tmp)


def stripped(events, fields):
    out = []
    for e in events:
        e = json.loads(json.dumps(e))
        if "symbol_targets" in e and isinstance(e["symbol_targets"], dict):
            for f in fields:
                e["symbol_targets"].pop(f, None)
        for f in fields:
            e.pop(f, None)
        out.append(e)
    return out


def comparable(out):
    """Every event that a run with W = 1 must share with SAT's: steps, evals, windows, the held-out accuracies, the
    load and end events, without SATW's added fields."""
    kinds = ("step", "eval", "windows", "symbol_targets_eval", "symbol_targets_end", "symbol_targets")
    return {k: stripped(S.evs(out, k), NEW_STEP_FIELDS + NEW_EVAL_FIELDS + ("nearest_right_weight",)) for k in kinds}


@needs_bin
def test_weight_one_is_sat(data):
    tmp, _ = data
    sat = S.full(tmp, "sat_plain_key", cfg(tmp))
    one = S.full(tmp, "satw_w1", cfg(tmp, 1.0))
    assert comparable(one) == comparable(sat)
    for f in ("model.safetensors", "m.safetensors"):
        assert (one / f).read_bytes() == (sat / f).read_bytes()
    assert S.evs(one, "symbol_targets")[0]["nearest_right_weight"] == 1.0
    # With W = 1 every weight is 1: the sums are the target counts.
    for e in S.evs(one, "step"):
        st = e.get("symbol_targets")
        if st and "pointer_weight_sum" in st:
            assert st["pointer_weight_sum"] == st["pointer_targets"]
            assert st["natural_pointer_weight_sum"] == st["natural_pointer_targets"]
    if SAT_BIN:
        # SAT's reviewed trainer on SAT's spec: the same run, field for field (it logs none of SATW's fields).
        import sat_build_provenance as P
        assert P.check(SAT_BIN, str(ROOT), "fbbcdb75") == []
        ref = tmp / "sat_reviewed"
        if not (ref / "result.json").exists():
            r = S.run(S.spec(tmp, ref, cfg(tmp)), binary=SAT_BIN)
            assert r.returncode == 0, r.stderr[-1500:]
        assert comparable(ref) == comparable(sat)
        assert (ref / "model.safetensors").read_bytes() == (sat / "model.safetensors").read_bytes()


def nearest_from_logged(w):
    """The recency rule recomputed from a windows event's logged targets: per pointer [q, gold], the nearest earlier
    mention-bit index in q's row; split by the row's source."""
    men = sorted(i for i, _role, _lab, m in w["labels"] if m)
    out = {0: [0, 0], 1: [0, 0]}  # source -> [right, wrong]
    for q, gold in w["pointers"]:
        row0 = q // S.T * S.T
        before = [j for j in men if row0 <= j < q]
        src = w["twin"][q // S.T]
        out[src][0 if before and before[-1] == gold else 1] += 1
    return out


@needs_bin
def test_nearest_right_split_and_weights(data):
    """The trainer's per-step split equals the rule recomputed from its logged targets, and, for natural rows, the
    label file and the replay predict it exactly; the weight sums are W x natural right + natural wrong + twin."""
    tmp, _ = data
    out = S.full(tmp, "satw_w", cfg(tmp, WEIGHT))
    by_step = {}
    for w in S.evs(out, "windows"):
        if w["step"] >= S.CUT:
            continue
        got = nearest_from_logged(w)
        acc = by_step.setdefault(w["step"], {0: [0, 0], 1: [0, 0]})
        for src in (0, 1):
            for i in (0, 1):
                acc[src][i] += got[src][i]
    steps = {e["step"]: e["symbol_targets"] for e in S.evs(out, "step") if e["step"] < S.CUT}
    for step, acc in by_step.items():
        st = steps[step]
        assert (st["natural_pointer_nearest_right"], st["natural_pointer_nearest_wrong"]) == tuple(acc[0]), step
        assert (st["twin_pointer_nearest_right"], st["twin_pointer_nearest_wrong"]) == tuple(acc[1]), step
        assert st["natural_pointer_nearest_right"] + st["natural_pointer_nearest_wrong"] == st["natural_pointer_targets"]
        assert st["twin_pointer_nearest_right"] + st["twin_pointer_nearest_wrong"] == st["twin_pointer_targets"]
        nat = WEIGHT * st["natural_pointer_nearest_right"] + st["natural_pointer_nearest_wrong"]
        assert st["natural_pointer_weight_sum"] == pytest.approx(nat, rel=1e-12)
        assert st["pointer_weight_sum"] == pytest.approx(nat + st["twin_pointer_targets"], rel=1e-12)
    tot_r = sum(s["natural_pointer_nearest_right"] for s in steps.values())
    tot_w = sum(s["natural_pointer_nearest_wrong"] for s in steps.values())
    assert tot_r > 0 and tot_w > 0
    # Independently: the label file and the replayed windows.
    raw = np.array(S.SL_read_ids(tmp / "main.ids"))
    imap = W.insert_map(raw, S.K, S.EOT, S.USER, S.ASST)
    wins, _ = W.replay(len(raw), imap, S.T, 3, S.BATCH, S.ACCUM, S.CUT, S.EVERY)
    _, spans = S.SL.read(tmp / "A.satl")
    want = W.expected_counts(W.SpanIndex(spans), imap, wins, S.T, nearest=True)
    assert (want["natural_pointer_nearest_right"], want["natural_pointer_nearest_wrong"]) == (tot_r, tot_w)
    assert want["natural_pointer_nearest_right"] + want["natural_pointer_nearest_wrong"] == want["natural_pointer_targets"]
    # The weight acts: W < 1 trains other heads and trunk than W = 1, from the same windows.
    one = S.full(tmp, "satw_w1", cfg(tmp, 1.0))
    assert (out / "model.safetensors").read_bytes() != (one / "model.safetensors").read_bytes()
    assert [w["raw"] for w in S.evs(out, "windows")] == [w["raw"] for w in S.evs(one, "windows")]
    assert [e["loss"] for e in S.evs(out, "step") if e["step"] == 1] == [e["loss"] for e in S.evs(one, "step") if e["step"] == 1]


@needs_bin
def test_eval_split_by_the_recency_rule(data):
    tmp, _ = data
    out = S.full(tmp, "satw_w", cfg(tmp, WEIGHT))
    ev = S.evs(out, "symbol_targets_eval")
    assert ev
    for e in ev:
        n_r, n_w = e["nearest_right_targets"], e["nearest_wrong_targets"]
        assert n_r + n_w == e["pointer_targets"] and n_r > 0 and n_w > 0
        assert e["recency_acc_nearest_right"] == 1.0 and e["recency_acc_nearest_wrong"] == 0.0
        assert e["recency_acc"] == pytest.approx(n_r / (n_r + n_w))
        right = e["pointer_acc_nearest_right"] * n_r + e["pointer_acc_nearest_wrong"] * n_w
        assert right == pytest.approx(e["pointer_acc"] * (n_r + n_w))


@needs_bin
def test_implementation_check_with_weight(data):
    tmp, _ = data
    out = tmp / "satw_check"
    shutil.rmtree(out, ignore_errors=True)
    r = S.run(S.spec(tmp, out, cfg(tmp, WEIGHT)), MICROTORCH_SAT_CHECK="8")
    assert r.returncode == 0, r.stderr[-800:]
    res = S.evs(out, "sat_check_result")[-1]
    assert res["pass"] is True and res["equal"] is True
    S.check_result_can_fail(res)


@needs_bin
@pytest.mark.parametrize("stage,at,crash_at", [("prune", 4, 4), ("mkdir", 8, 4)])
def test_resume_inside_the_scaffold_with_weight(data, stage, at, crash_at):
    tmp, _ = data
    whole = S.full(tmp, "satw_w", cfg(tmp, WEIGHT))
    cut = tmp / f"satw_resume_{stage}_{at}"
    shutil.rmtree(cut, ignore_errors=True)
    p = S.spec(tmp, cut, cfg(tmp, WEIGHT))
    assert S.run(p, MICROTORCH_CKPT_CRASH=f"{stage}:{at}").returncode != 0
    r = S.run(p)
    assert r.returncode == 0, r.stderr[-1500:]
    assert S.evs(cut, "symbol_targets_resume")[0]["step"] == crash_at
    for kind in ("step", "eval", "windows", "symbol_targets_eval"):
        a = {k: v for k, v in S.last_by_step(whole, kind).items() if k[0] > crash_at}
        b = {k: v for k, v in S.last_by_step(cut, kind).items() if k[0] > crash_at}
        assert a == b and a, kind
    assert S.evs(whole, "symbol_targets_end") == S.evs(cut, "symbol_targets_end")
    assert (whole / "model.safetensors").read_bytes() == (cut / "model.safetensors").read_bytes()


@needs_bin
def test_from_the_cut_the_weight_does_nothing(data):
    """A SATW run published at cut_step, resumed under SATW's spec, SAT's and the plain one: identical from cut + 1."""
    tmp, _ = data
    base = tmp / "satw_to_cut"
    shutil.rmtree(base, ignore_errors=True)
    assert S.run(S.spec(tmp, base, cfg(tmp, WEIGHT)), MICROTORCH_CKPT_CRASH=f"prune:{S.CUT}").returncode != 0
    outs = {}
    for name, sat in (("satw", cfg(tmp, WEIGHT)), ("sat", cfg(tmp)), ("plain", None)):
        out = tmp / f"satw_cont_{name}"
        shutil.rmtree(out, ignore_errors=True)
        shutil.copytree(base, out, symlinks=True)
        r = S.run(S.spec(tmp, out, sat))
        assert r.returncode == 0, r.stderr[-1500:]
        outs[name] = out
    after = {n: {k: v for k, v in S.last_by_step(o, "step").items() if k[0] > S.CUT} for n, o in outs.items()}
    assert after["satw"] == after["sat"] == after["plain"] and len(after["satw"]) == S.STEPS - S.CUT
    weights = {n: (o / "model.safetensors").read_bytes() for n, o in outs.items()}
    assert weights["satw"] == weights["sat"] == weights["plain"]


@needs_bin
@pytest.mark.parametrize("rw,why", [
    ({"nearest_right": 0}, "must be in (0, 1]"),
    ({"nearest_right": -0.5}, "must be in (0, 1]"),
    ({"nearest_right": 1.5}, "must be in (0, 1]"),
    ({"nearest_right": "x"}, "must be {\"nearest_right\": W}"),
    ({"nearest_right": 0.5, "other": 1}, "must be {\"nearest_right\": W}"),
    (0.5, "must be {\"nearest_right\": W}"),
])
def test_weight_refused(data, rw, why):
    tmp, _ = data
    c = cfg(tmp)
    c["recency_weight"] = rw
    r = S.run(S.spec(tmp, tmp / "satw_refused", c))
    assert r.returncode != 0 and why in (r.stderr + r.stdout), (r.stderr + r.stdout)[-600:]


# ---------------------------------------------------------------- the real label file (needs N.main.ids)

REAL = Path("/mnt/c/ml_artifacts/sat/corpus")
N_MAIN = os.environ.get("SATW_N_MAIN_IDS")


@pytest.mark.skipif(not (N_MAIN and (REAL / "sat.labels").exists()), reason="set SATW_N_MAIN_IDS to a copy of N.main.ids")
def test_real_label_file_share_is_the_receipts():
    """The window-counted nearest-right share of the real label file over N1B's replayed windows (seed 1, 8 x 2 rows
    of 1,024, twins one row in ten, cut 13,288, state ids k 4), recomputed here, equals the label build's receipt
    count for count (met_by_windows recency_right / pointer_targets: 0.7647). Hours of CPU: run through
    scripts/cpu_budget.py."""
    import hashlib
    import sat_labels as SL
    rec = json.loads((REAL / "receipt.json").read_text())
    met = rec["files"]["sat.labels"]["met_by_windows"]
    h = hashlib.sha256()
    with open(N_MAIN, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    assert h.hexdigest() == rec["corpus_sha256"]
    raw = W.read_ids(N_MAIN)
    sv = json.loads((REAL / "labels_plan.json").read_text())["spec"]
    special = os.environ.get("SATW_SPECIAL_IDS")  # "eot,user,assistant" ids of base 2's vocabulary
    if not special:
        pytest.skip("set SATW_SPECIAL_IDS to the <|endoftext|>,<|user|>,<|assistant|> ids of base 2's vocabulary")
    eot, user, assistant = (int(x) for x in special.split(","))
    imap = W.insert_map(raw, sv["state_k"], eot, user, assistant)
    plan = json.loads((REAL / "labels_plan.json").read_text())
    assert imap.inserted == plan["inserted_ids"] and imap.n_post == plan["replay"]["tokens_after"]
    wins, _ = W.replay(len(raw), imap, sv["T"], sv["seed"], sv["batch"], sv["accum"], sv["cut_step"], sv["twin_every"])
    _, spans = SL.read(REAL / "sat.labels")
    got = W.expected_counts(W.SpanIndex(spans), imap, wins, sv["T"], nearest=True)
    assert got["natural_pointer_targets"] == met["pointer_targets"]
    assert got["natural_pointer_nearest_right"] == met["recency_right"]
