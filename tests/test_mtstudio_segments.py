"""mtstudio features the tiny-chat experiments rely on
(experiments/tiny_chat_2026-10/PREREGISTRATION.md), end to end on CPU:

- SFT examples in the "segments" form (a whole multi-turn dialogue in one
  row, every assistant turn trained): a one-turn segments record trains
  bit for bit like the {prompt, completion} record it spells out; a
  two-turn record counts both turns' targets and is learnt; malformed
  records are refused.
- train.eval_windows: the LM validation pass reads that many held-out
  windows (default 8, unchanged); 0 is refused.
- The zero-rate pass the experiments use to score a fixed text panel: a
  1-step run at lr 0 from init_from leaves the weights byte-identical and
  reports the panel's validation loss.

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
from test_mtstudio_moe import events as lm_events, setup as lm_setup, steps_of  # noqa: E402

needs_bin = pytest.mark.skipif(BIN is None, reason="set MTSTUDIO_BIN to a built mtstudio")

EOT = "<|endoftext|>"
WORDS = ["what", "is", "plus", "?", "the", "answer", "."] + [str(i) for i in range(20)]


def sft_setup(tmp, examples, steps=200, T=32, name="sft"):
    tmp.mkdir(parents=True, exist_ok=True)
    write_vocab_gguf(tmp / "v.gguf", ["<unk>", EOT] + WORDS, eos_id=1)
    (tmp / "ex.jsonl").write_text("".join(json.dumps(e) + "\n" for e in examples), encoding="utf-8")
    spec = {"name": name, "arch": {"preset": "llama-tiny"},
            "data": {"vocab": str(tmp / "v.gguf"), "vocab_cap": 64, "T": T, "sft_examples": str(tmp / "ex.jsonl")},
            "train": {"steps": steps, "lr": 0.003, "eval_every": 50, "checkpoint_every": 1000, "batch": 4,
                      "seed": 3, "gradmap_every": 0, "objective": "sft"},
            "export": {"formats": ["safetensors"]}, "out_dir": str(tmp / "run")}
    (tmp / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    return tmp / "spec.json"


def run(spec):
    return subprocess.run([BIN, "run", str(spec)], capture_output=True, text=True, timeout=2400)


def events(tmp):
    return [json.loads(x) for x in (tmp / "run" / "events.jsonl").read_text().splitlines() if x.strip()]


def q(a, b):
    return f"what is {a} plus {b} ?"


def ans(a, b):
    return f"the answer is {a + b} ."


def seg(*turns):
    out = []
    for a, b in turns:
        out += [{"text": q(a, b), "loss": False}, {"text": ans(a, b), "loss": True}]
    return {"segments": out}


FACTS = [(a, b) for a in range(5) for b in range(2)]


@needs_bin
def test_one_turn_segments_equal_prompt_completion(tmp_path):
    pc = [{"prompt": q(a, b), "completion": ans(a, b)} for a, b in FACTS] * 2
    sg = [seg((a, b)) for a, b in FACTS] * 2
    assert run(sft_setup(tmp_path / "pc", pc, steps=40)).returncode == 0
    p = run(sft_setup(tmp_path / "sg", sg, steps=40))
    assert p.returncode == 0, p.stderr
    key = lambda ev: [(e["step"], e["loss"], e["grad_norm"]) for e in ev if e["event"] == "step"]
    a, b = events(tmp_path / "pc"), events(tmp_path / "sg")
    assert key(a) == key(b)
    da, db = (next(e for e in ev if e["event"] == "data") for ev in (a, b))
    assert da["tokens"] == db["tokens"]
    assert [e["val_loss"] for e in a if e["event"] == "eval"] == [e["val_loss"] for e in b if e["event"] == "eval"]


@needs_bin
def test_two_turn_segments_train_both_turns(tmp_path):
    ex = [seg((a, b), (b, a)) for a, b in FACTS] * 3 + [seg((0, 0), (0, 0))]
    p = run(sft_setup(tmp_path, ex, steps=300))
    assert p.returncode == 0, p.stderr
    ev = events(tmp_path)
    data = next(e for e in ev if e["event"] == "data")
    # Both assistant turns count: 6 targets each ("the answer is N ." + end of text).
    assert data["tokens"] == 12 * (len(ex) - 1) and data["completion_unk"] == 0
    losses = [e["loss"] for e in ev if e["event"] == "step"]
    median = lambda xs: sorted(xs)[len(xs) // 2]
    assert median(losses[-40:]) < 0.3 * median(losses[:20])


@needs_bin
@pytest.mark.parametrize("bad,msg", [
    ({"segments": [{"text": ans(1, 1), "loss": True}]}, "first segment"),
    ({"segments": [{"text": q(1, 1), "loss": False}]}, "no loss segment"),
    ({"segments": [{"text": q(1, 1)}]}, "text (string) and loss (bool)"),
    ({"segments": []}, "non-empty array"),
    (seg((1, 1), (1, 2), (2, 1)), "data.T"),
])
def test_bad_segments_are_refused(tmp_path, bad, msg):
    p = run(sft_setup(tmp_path, [bad, seg((1, 1))], steps=5, T=20))
    assert p.returncode != 0 and msg in p.stderr, p.stderr[-500:]


@needs_bin
def test_eval_windows(tmp_path):
    base = lm_setup(tmp_path / "a", "w8", 20, train_extra={"eval_every": 10})
    more = lm_setup(tmp_path / "b", "w32", 20, train_extra={"eval_every": 10, "eval_windows": 32})
    for s in (base, more):
        assert subprocess.run([BIN, "run", str(s)], capture_output=True, text=True, timeout=1800).returncode == 0
    ea, eb = lm_events(base), lm_events(more)
    # Training is untouched; only the validation estimate changes.
    assert [(e["loss"], e["grad_norm"]) for e in steps_of(ea)] == [(e["loss"], e["grad_norm"]) for e in steps_of(eb)]
    va = [e["val_loss"] for e in ea if e["event"] == "eval"]
    vb = [e["val_loss"] for e in eb if e["event"] == "eval"]
    assert len(va) == len(vb) == 2 and va != vb
    bad = lm_setup(tmp_path / "c", "w0", 5, train_extra={"eval_windows": 0})
    p = subprocess.run([BIN, "run", str(bad)], capture_output=True, text=True, timeout=600)
    assert p.returncode != 0 and "eval_windows" in p.stderr


@needs_bin
def test_zero_rate_pass_scores_a_panel_without_moving_weights(tmp_path):
    trained = lm_setup(tmp_path / "t", "t", 30)
    assert subprocess.run([BIN, "run", str(trained)], capture_output=True, text=True, timeout=1800).returncode == 0
    ck = tmp_path / "t" / "t" / "t.safetensors"
    panel = lm_setup(tmp_path / "p", "p", 1, train_extra={"lr": 0.0, "eval_every": 1, "eval_windows": 16,
                                                          "init_from": str(ck)})
    p = subprocess.run([BIN, "run", str(panel)], capture_output=True, text=True, timeout=1800)
    assert p.returncode == 0, p.stderr[-1000:]
    ev = lm_events(panel)
    assert [e["step"] for e in ev if e["event"] == "eval"] == [1]
    assert (tmp_path / "p" / "p" / "p.safetensors").read_bytes() == ck.read_bytes()


@needs_bin
def test_corpus_too_short_for_a_held_out_window_is_refused(tmp_path):
    # 5% of this corpus is shorter than T + 2: before the guard the window draw wrapped round (a crash).
    spec = lm_setup(tmp_path, "short", 5)
    (tmp_path / "c.txt").write_text("the cat sat . " * 40, encoding="utf-8")
    p = subprocess.run([BIN, "run", str(spec)], capture_output=True, text=True, timeout=600)
    assert p.returncode != 0 and "held-out 5%" in p.stderr, p.stderr[-500:]
