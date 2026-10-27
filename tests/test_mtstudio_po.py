"""mtstudio `train.objective: "provenance_po"` end to end on CPU (charter 0001,
Phase 5). Wiring checks on a ten-fact lookup task, not a research claim:
- dataset weights follow the frozen evidence table (exact 1.0, judge 0.3,
  unattributed 0), with Kish ESS and per-kind sums in the data event;
- at step 1 the policy IS the reference, so every pair's margin is zero;
- the loss falls, and held-out preference accuracy is reported;
- uniform mode weights every pair 1; the shuffled falsifier keeps the weight
  sum and ESS exactly while moving weights between pairs.
Needs MTSTUDIO_BIN (or ./build/mtstudio); skipped otherwise.
"""
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_mtstudio_chat import BIN, write_vocab_gguf  # noqa: E402
from test_mtstudio_grpo import run, spec  # noqa: E402
from test_mtstudio_sft import EOT, FACTS, WORDS, qa  # noqa: E402

needs_bin = pytest.mark.skipif(BIN is None, reason="set MTSTUDIO_BIN to a built mtstudio")


def receipt(kind, uri="t/x"):
    return {"evidence_kind": kind, "evidence_uri": uri, "audited": False, "reward": 1.0, "rationale": "r"}


def pairs():
    out = []
    kinds = ["exact_match", "judge", "exact_match", "judge", None]  # None: unattributed
    for i, (a, b) in enumerate(FACTS * 3):
        k = kinds[i % len(kinds)]
        out.append({"prompt": f"what is {a} plus {b} ?", "chosen": f"the answer is {a + b} .",
                    "rejected": f"the answer is {(a + b + 1) % 20} .",
                    "chosen_evidence": receipt(k) if k else {}, "rejected_evidence": receipt(k) if k else {}})
    return out


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("po")
    write_vocab_gguf(tmp / "v.gguf", ["<unk>", EOT] + WORDS, eos_id=1)
    ex = [qa(a, b) for a, b in FACTS] * 3 + [qa(0, 0)]
    (tmp / "ex.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ex), encoding="utf-8")
    run(spec(tmp, "base", {"steps": 150, "eval_every": 50, "objective": "sft"}, {"sft_examples": str(tmp / "ex.jsonl")}))
    (tmp / "pairs.jsonl").write_text("".join(json.dumps(p) + "\n" for p in pairs()), encoding="utf-8")
    return tmp


def po_spec(tmp, name, mode="provenance", steps=40):
    return spec(tmp, name, {"steps": steps, "eval_every": 20, "objective": "provenance_po", "lr": 0.003,
                            "init_from": str(tmp / "base" / "base.safetensors"),
                            "po": {"beta": 0.5, "weight_mode": mode}},
                {"pref_pairs": str(tmp / "pairs.jsonl")})


def expected(n_train):
    kinds = ["exact_match", "judge", "exact_match", "judge", None]
    w = [{"exact_match": 1.0, "judge": 0.3, None: 0.0}[kinds[i % 5]] for i in range(n_train)]
    s, s2 = sum(w), sum(x * x for x in w)
    return w, s, s * s / s2


@needs_bin
def test_weights_ess_and_zero_margin_at_start(base):
    ev = run(po_spec(base, "po_prov"))
    data = next(e for e in ev if e["event"] == "data")
    n = data["pairs"] - data["val_pairs"]
    w, s, ess = expected(n)
    assert data["objective"] == "provenance_po" and data["pairs"] == 30 and data["val_pairs"] == 1
    assert data["pairs_with_weight"] == sum(x > 0 for x in w)
    assert abs(data["weight_sum"] - s) < 1e-5 and abs(data["ess_kish"] - ess) < 1e-4
    assert set(data["pairs_by_kind"]) == {"exact_match", "judge", "unattributed"}
    steps = [e for e in ev if e["event"] == "step"]
    for k in ("weight_sum", "ess_kish", "margin", "pref_acc"):
        assert all(k in e for e in steps), k
    assert abs(steps[0]["margin"]) < 1e-4          # policy == reference before the first update
    first, last = steps[0]["loss"], sum(e["loss"] for e in steps[-5:]) / 5
    assert last < first
    evals = [e for e in ev if e["event"] == "po_eval"]
    assert [e["step"] for e in evals] == [20, 40] and all(0.0 <= e["pref_acc"] <= 1.0 for e in evals)


@needs_bin
def test_uniform_and_shuffled_modes(base):
    uni = next(e for e in run(po_spec(base, "po_uni", "uniform", steps=2)) if e["event"] == "data")
    assert uni["pairs_with_weight"] == uni["pairs"] - uni["val_pairs"]
    assert abs(uni["ess_kish"] - (uni["pairs"] - uni["val_pairs"])) < 1e-6
    prov = next(e for e in run(po_spec(base, "po_prov2", steps=2)) if e["event"] == "data")
    shuf = next(e for e in run(po_spec(base, "po_shuf", "shuffled", steps=2)) if e["event"] == "data")
    assert abs(shuf["weight_sum"] - prov["weight_sum"]) < 1e-6 and abs(shuf["ess_kish"] - prov["ess_kish"]) < 1e-6
    assert shuf["weight_mode"] == "shuffled"


@needs_bin
def test_po_refuses_without_reference(base):
    p = spec(base, "po_bad", {"steps": 2, "eval_every": 2, "objective": "provenance_po"},
             {"pref_pairs": str(base / "pairs.jsonl")})
    r = subprocess.run([BIN, "run", str(p)], capture_output=True, text=True, timeout=120)
    assert r.returncode != 0 and "init_from" in r.stderr


@needs_bin
def test_rpo_nll_term(base):
    p = spec(base, "po_nll", {"steps": 6, "eval_every": 6, "objective": "provenance_po", "lr": 0.003,
                              "init_from": str(base / "base" / "base.safetensors"),
                              "po": {"beta": 0.5, "nll": 1.0}}, {"pref_pairs": str(base / "pairs.jsonl")})
    ev = run(p)
    assert next(e for e in ev if e["event"] == "data")["nll"] == 1.0
    steps = [e for e in ev if e["event"] == "step"]
    assert all(e["chosen_nll"] > 0 for e in steps)
    # With nll = 0 the term is absent (plain DPO): chosen_nll is reported as 0.
    plain = [e for e in run(po_spec(base, "po_plain", steps=2)) if e["event"] == "step"]
    assert all(e["chosen_nll"] == 0 for e in plain)


@needs_bin
def test_stop_fields_decide_the_end_of_text(base):
    # chosen == rejected text: with both ending on end-of-text the pair carries no signal (margin 0
    # at every step); a chosen reply the budget cut (chosen_stop "length") is the prefix without the
    # end-of-text, so the two sequences differ by that token and the margin moves.
    same = [{"prompt": f"what is {a} plus {b} ?", "chosen": f"the answer is {a + b} .",
             "rejected": f"the answer is {a + b} ."} for a, b in FACTS * 3]
    (base / "same.jsonl").write_text("".join(json.dumps(p) + "\n" for p in same), encoding="utf-8")
    cut = [dict(p, chosen_stop="length", rejected_stop="eos") for p in same]
    (base / "cut.jsonl").write_text("".join(json.dumps(p) + "\n" for p in cut), encoding="utf-8")

    def go(name, pairs_file):
        p = spec(base, name, {"steps": 4, "eval_every": 4, "objective": "provenance_po", "lr": 0.003,
                              "init_from": str(base / "base" / "base.safetensors"),
                              "po": {"beta": 0.5, "weight_mode": "uniform"}}, {"pref_pairs": str(base / pairs_file)})
        return run(p)

    ev = go("po_same", "same.jsonl")
    data = next(e for e in ev if e["event"] == "data")
    assert data["chosen_open"] == 0 and data["rejected_open"] == 0
    assert all(abs(e["margin"]) < 1e-6 for e in ev if e["event"] == "step")
    ev = go("po_cut", "cut.jsonl")
    data = next(e for e in ev if e["event"] == "data")
    assert data["chosen_open"] == data["pairs"] and data["rejected_open"] == 0
    steps = [e for e in ev if e["event"] == "step"]
    assert abs(steps[0]["margin"]) < 1e-4 and any(abs(e["margin"]) > 1e-3 for e in steps[1:])
