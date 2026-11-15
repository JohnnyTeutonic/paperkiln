"""mtstudio for lane amendment A32 (b2cue) end to end on CPU: the answer-token margin (train.twin_loss.answer_mask, arm NA) and
train.eval_at (SFT1500's added held-out evaluations). Claude Opus 5.5, 15 Nov 2026. Wiring checks on test_mtstudio_twin.py's
two-variable lookup, not a research claim:
- an answer-mask run trains, logs answer_mask on every step's twin record, and writes the targets and masks it read
  (twin_masks.jsonl), equal to the twin file's masks;
- an all-ones mask is A31's whole-reply margin: every step's SFT and twin values equal the unmasked run's;
- a mask that does not cover its reply's targets, an empty mask, a missing mask or a weight other than 0 and 1 is refused;
- train.eval_at adds held-out evaluations at its steps and changes no training step: every step's loss and sft_rows_hash equal
  the run without it (evaluation reads no generator).
Needs MTSTUDIO_BIN (or ./build/mtstudio); skipped otherwise.
"""
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_mtstudio_twin import BIN, TRIPLETS, needs_bin, run, spec, steps, twin_train, tmp  # noqa: E402,F401


def with_masks(trips, ma, mb):
    return [dict(t, mask_A=ma, mask_B=mb) for t in trips]


def write(tmp, name, rows):
    p = tmp / name
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(p)


# each reply is "<digit> ." plus the close: three targets, the digit first
DIGIT_ONLY, ALL = [1, 0, 0], [1, 1, 1]


def am_train(**kw):
    t = twin_train(**kw)
    t["twin_loss"]["answer_mask"] = True
    return t


@needs_bin
def test_answer_mask_runs_and_records_its_masks(tmp):
    tw = write(tmp, "twins_masked.jsonl", with_masks(TRIPLETS, DIGIT_ONLY, DIGIT_ONLY))
    p = spec(tmp, "am", am_train(steps=4), {"twin_examples": tw})
    ev = run(p)
    st = steps(ev)
    assert len(st) == 4
    for e in st:
        assert e["twin"].get("answer_mask") is True
        assert 0.0 <= e["twin"]["active"] <= 1.0
    dump = [json.loads(x) for x in (tmp / "am" / "twin_masks.jsonl").read_text().splitlines() if x.strip()]
    assert len(dump) == len(TRIPLETS)
    for d in dump:
        assert d["mask_A"] == DIGIT_ONLY and d["mask_B"] == DIGIT_ONLY
        assert len(d["ids_A"]) == 3 and len(d["ids_B"]) == 3


@needs_bin
def test_all_ones_mask_is_the_whole_reply_margin(tmp):
    plain = steps(run(spec(tmp, "whole", twin_train(steps=6), {"twin_examples": str(tmp / "twins.jsonl")})))
    tw = write(tmp, "twins_ones.jsonl", with_masks(TRIPLETS, ALL, ALL))
    ones = steps(run(spec(tmp, "ones", am_train(steps=6), {"twin_examples": tw})))
    for a, b in zip(plain, ones):
        assert a["loss"] == b["loss"] and a["grad_norm"] == b["grad_norm"]
        for k in ("loss", "d_a", "d_b", "d_n", "active"):
            assert a["twin"][k] == b["twin"][k], (k, a["twin"][k], b["twin"][k])


@needs_bin
def test_answer_mask_changes_the_margin(tmp):
    plain = steps(run(spec(tmp, "whole2", twin_train(steps=3), {"twin_examples": str(tmp / "twins.jsonl")})))
    tw = write(tmp, "twins_digit.jsonl", with_masks(TRIPLETS, DIGIT_ONLY, DIGIT_ONLY))
    am = steps(run(spec(tmp, "digit", am_train(steps=3), {"twin_examples": tw})))
    assert plain[0]["loss"] == am[0]["loss"]                 # the SFT batch is untouched
    assert plain[0]["twin"]["loss"] != am[0]["twin"]["loss"]   # the margin reads fewer tokens
    assert plain[0]["twin"]["d_n"] == am[0]["twin"]["d_n"]     # dN is the whole reply either way


@needs_bin
def test_answer_mask_refusals(tmp):
    def refused(name, rows, needle):
        tw = write(tmp, name + ".jsonl", rows)
        r = run(spec(tmp, name, am_train(steps=2), {"twin_examples": tw}), ok=False)
        assert r.returncode != 0 and needle in r.stderr, r.stderr
    refused("am_short", with_masks(TRIPLETS[:4], [1, 0], DIGIT_ONLY), "mask_A has")
    refused("am_empty", with_masks(TRIPLETS[:4], [0, 0, 0], DIGIT_ONLY), "empty answer mask")
    refused("am_missing", TRIPLETS[:4], "mask_A must be")
    refused("am_half", with_masks(TRIPLETS[:4], [1, 0, 2], DIGIT_ONLY), "0 or 1")
    r = run(spec(tmp, "am_flag", {"twin_loss": {"groups": 2, "margin": 1, "lambda": 1, "mu": 0, "answer_mask": 1}},
                 {"twin_examples": write(tmp, "am_flag.jsonl", with_masks(TRIPLETS[:4], DIGIT_ONLY, DIGIT_ONLY))}), ok=False)
    assert r.returncode != 0 and "answer_mask must be true or false" in r.stderr


@needs_bin
def test_eval_at_changes_no_training_step(tmp):
    base = run(spec(tmp, "ev_plain", {"steps": 12}))
    extra = run(spec(tmp, "ev_at", {"steps": 12, "eval_at": [3, 5, 9]}))
    a, b = steps(base), steps(extra)
    assert [e["loss"] for e in a] == [e["loss"] for e in b]
    assert [e["grad_norm"] for e in a] == [e["grad_norm"] for e in b]
    assert [e["sft_rows_hash"] for e in a] == [e["sft_rows_hash"] for e in b]
    evals = [e["step"] for e in extra if e["event"] == "eval"]
    assert [3, 5, 9] == [s for s in evals if s in (3, 5, 9)]
    assert not [e for e in base if e["event"] == "eval" and e["step"] in (3, 5, 9)]
    r = run(spec(tmp, "ev_bad", {"steps": 2, "eval_at": [0]}), ok=False)
    assert r.returncode != 0 and "eval_at" in r.stderr


@needs_bin
@pytest.mark.skipif(os.environ.get("B2CUE_SLOW") != "1", reason="1,500 steps twice, about 50 minutes on 4 CPU threads: "
                    "B2CUE_SLOW=1 runs it")
def test_eval_at_on_the_sft1500_recipe_past_step_188(tmp):
    """Section 8 (launch round 4): the SFT1500 recipe's training block as C0 launches it (b2cue_c0.json's sft_N250: all 1,500 steps,
    lr, batch, accumulation, warm-up and cosine, clip, AdamW, seed, eval_every, eval_batch, checkpoint_every and eval_at [188, 375,
    750]; RoPE on all heads), with the init_from left out, at the toy model, vocabulary and T of these tests, run on the CPU with and
    without eval_at: every step's loss, gradient norm and SFT rows equal bit for bit, so the added evaluations consume no random state,
    and the held-out evaluations fall at exactly 188, 375 and 750 beyond eval_every's. The state tokens are left out: they need the
    chat template's specials, which the toy vocabulary lacks, and on single-turn SFT rows they insert no block (D1's vocabulary at
    its cap ran at about 15 s a step here, too slow for a test)."""
    jobs = json.loads((ROOT / "tools" / "colab_chat" / "b2cue_c0.json").read_text())
    rec = next(j for j in jobs["jobs"] if j["name"] == "sft_N250")["spec"]
    train = {k: v for k, v in rec["train"].items() if k != "init_from"}
    assert train["steps"] == 1500 and train["eval_at"] == [188, 375, 750]
    arch = {"rope_heads": rec["arch"]["rope_heads"]}
    plain = {k: v for k, v in train.items() if k != "eval_at"}
    a_ev = run(spec(tmp, "rec_plain", plain, arch=arch))
    b_ev = run(spec(tmp, "rec_at", train, arch=arch))
    a, b = steps(a_ev), steps(b_ev)
    assert len(a) == len(b) == 1500
    assert [e["loss"] for e in a] == [e["loss"] for e in b]
    assert [e["grad_norm"] for e in a] == [e["grad_norm"] for e in b]
    assert [e["sft_rows_hash"] for e in a] == [e["sft_rows_hash"] for e in b]
    ev_a = {e["step"] for e in a_ev if e["event"] == "eval"}
    ev_b = {e["step"] for e in b_ev if e["event"] == "eval"}
    assert ev_b - ev_a == {188, 375, 750}
