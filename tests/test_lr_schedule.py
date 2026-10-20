"""mtstudio train.lr_schedule: linear warmup, then constant or cosine decay.

The schedule is a pure function of the step, so it is checked three ways: a
spec without it trains exactly as before (no "lr" on step events; with
MTSTUDIO_BASE_BIN set to a binary built before the schedule, the events are
compared against it), the logged rate follows the formula at every step,
and a 40-step run killed after its step-20 checkpoint and resumed matches
the uninterrupted run bit for bit (losses, grad norms, rates). Needs
MTSTUDIO_BIN (or ./build/mtstudio); skipped otherwise.
"""
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_mtstudio_chat import BIN  # noqa: E402
from test_mtstudio_moe import events, run, setup, steps_of  # noqa: E402

needs_bin = pytest.mark.skipif(BIN is None, reason="set MTSTUDIO_BIN to a built mtstudio")
BASE = os.environ.get("MTSTUDIO_BASE_BIN")

LR = 0.003
SCHED = {"warmup_steps": 10, "decay": "cosine", "min_lr_frac": 0.1}


def expected_lr(step, steps=40, warmup=10, min_frac=0.1):
    if step <= warmup:
        return LR * step / warmup
    p = (step - warmup) / (steps - warmup)
    return LR * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * p)))


@needs_bin
def test_no_schedule_is_unchanged(tmp_path):
    spec = setup(tmp_path / "a", "plain", 20)
    run(spec)
    ev = events(spec)
    st = steps_of(ev)
    assert len(st) == 20 and all("lr" not in e for e in st)
    assert "lr_schedule" not in next(e for e in ev if e["event"] == "model")
    if BASE:
        ref = setup(tmp_path / "b", "plain", 20)
        run(ref, binary=BASE)
        key = [(e["step"], e["loss"], e["grad_norm"]) for e in st]
        assert key == [(e["step"], e["loss"], e["grad_norm"]) for e in steps_of(events(ref))]


@needs_bin
def test_warmup_then_cosine_follows_the_formula(tmp_path):
    spec = setup(tmp_path, "sched", 40, train_extra={"lr_schedule": SCHED})
    run(spec)
    ev = events(spec)
    assert next(e for e in ev if e["event"] == "model")["lr_schedule"]["decay"] == "cosine"
    st = steps_of(ev)
    assert [e["step"] for e in st] == list(range(1, 41))
    for e in st:
        assert e["lr"] == pytest.approx(expected_lr(e["step"]), rel=1e-5), e["step"]
    # A few points by hand: warmup is linear from lr/10, the peak is the
    # base rate, halfway through the decay is (1 + 0.1) / 2 of it, and the
    # last step reaches min_lr_frac.
    lr = {e["step"]: e["lr"] for e in st}
    assert lr[1] == pytest.approx(0.0003, rel=1e-6)
    assert lr[5] == pytest.approx(0.0015, rel=1e-6)
    assert lr[10] == pytest.approx(0.003, rel=1e-6)
    assert lr[25] == pytest.approx(0.00165, rel=1e-6)
    assert lr[40] == pytest.approx(0.0003, rel=1e-6)
    # The schedule changes the trajectory once the rate differs from the base.
    plain = setup(tmp_path / "p", "plain", 40)
    run(plain)
    assert steps_of(events(plain))[1]["loss"] != st[1]["loss"]


def run_until_killed(spec, after_step):
    """Run spec, SIGKILL it once a step past `after_step` is logged."""
    p = subprocess.Popen([BIN, "run", str(spec)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                         bufsize=1)
    for line in p.stdout:
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("event") == "step" and e["step"] > after_step:
            os.kill(p.pid, signal.SIGKILL)
            break
    p.wait()
    return p.returncode


@needs_bin
def test_resume_follows_the_schedule(tmp_path):
    extra = {"lr_schedule": SCHED, "checkpoint_every": 20, "eval_every": 20}
    full = setup(tmp_path / "a", "s", 40, train_extra=extra)
    run(full)
    part = setup(tmp_path / "b", "s", 40, train_extra=extra)
    assert run_until_killed(part, 20) == -signal.SIGKILL, "the run finished before it could be interrupted"
    run(part)
    eb = events(part)
    assert any(e["event"] == "resume" and e["step"] == 20 for e in eb)
    key = lambda ev: [(e["step"], e["loss"], e["grad_norm"], e["lr"]) for e in steps_of(ev)]  # noqa: E731
    kb = key(eb)
    assert [k[0] for k in kb] == list(range(1, 41))
    assert key(events(full)) == kb, "resumed run diverged from the uninterrupted one"
