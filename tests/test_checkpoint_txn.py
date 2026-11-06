"""Transactional checkpoints (include/microtorch/checkpoint_txn.hpp), end to end on CPU with the real mtstudio.

A tiny model with state tokens and an annealing tail (the code path of the stronger base) trains uninterrupted as the
control. Then the same run is ended at every publication boundary, by MICROTORCH_CKPT_CRASH and by SIGKILL mid-write, restarted,
and must end with the control's model, optimiser, step and exported weights, byte for byte; the step and RNG follow from the
trainer's replay, so equal bytes after the crash window prove them. Damage to a generation must fall back to the previous one or
refuse, never resume from it. With the variable unset the trainer must equal MTSTUDIO_BASE_BIN (a build of the source below
this patch), step for step and byte for byte. Needs MTSTUDIO_BIN; skipped otherwise. Everything is written to tmp_path.
"""
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_mtstudio_chat import BIN, write_vocab_gguf  # noqa: E402

needs_bin = pytest.mark.skipif(BIN is None, reason="set MTSTUDIO_BIN to a built mtstudio")
BASE = os.environ.get("MTSTUDIO_BASE_BIN")
WORDS = ["the", "cat", "dog", "sat", "ran", "on", "mat", "log", "a", "big", "small", "red", "blue", "."]
STAGES = ["mkdir", "model", "optim", "state", "manifest", "dirsync", "rename", "commit", "links", "prune"]
STEPS, EVERY = 24, 6


def corpus(n, seed):
    r = random.Random(seed)
    return " ".join(f"the {r.choice(['big', 'small', 'red', 'blue'])} {r.choice(['cat', 'dog'])} sat on a mat ." for _ in range(n))


def chat_corpus(n, seed):
    """The corpus text state tokens read: <|user|>/<|assistant|> turns, as mix_pretrain's dialogue documents."""
    r = random.Random(seed)
    return "".join(f"<|user|>the {r.choice(['cat', 'dog'])} sat\n<|assistant|>on a mat .\n<|endoftext|>\n" for _ in range(n))


def setup(tmp, state_tokens=False):
    tmp.mkdir(parents=True, exist_ok=True)
    specials = ["<|endoftext|>", "<|user|>", "<|assistant|>"] if state_tokens else []
    write_vocab_gguf(tmp / "v.gguf", ["<unk>"] + specials + WORDS)
    make = chat_corpus if state_tokens else corpus
    (tmp / "main.txt").write_text(make(1500, 0), encoding="utf-8")
    (tmp / "tail.txt").write_text(make(400, 1), encoding="utf-8")


def spec(tmp, out, steps=STEPS, every=EVERY, state_tokens=False):
    arch = {"preset": "llama-tiny", "custom": {"d": 32, "layers": 2, "heads": 2}}
    if state_tokens:
        arch["state_tokens"] = {"k": 2, "mask": "causal"}
    s = {"name": "m", "arch": arch,
         "data": {"corpus": str(tmp / "main.txt"), "vocab": str(tmp / "v.gguf"), "T": 32, "max_tokens": 0,
                  "anneal": {"corpus": str(tmp / "tail.txt"), "from_fraction": 0.5}},
         "train": {"steps": steps, "lr": 0.003, "batch": 2, "accum": 1, "eval_every": every, "eval_windows": 2,
                   "checkpoint_every": every, "seed": 3, "optimizer": "adamw", "gradmap_every": 0},
         "export": {"formats": ["safetensors"]}, "out_dir": str(out)}
    p = Path(str(out) + ".spec.json")
    p.write_text(json.dumps(s), encoding="utf-8")
    return p


def env(extra=None):
    e = {**os.environ, "MICROTORCH_CHECKPOINT_TXN": "1", "OMP_NUM_THREADS": "1"}
    e.pop("MICROTORCH_CKPT_CRASH", None)
    return {**e, **(extra or {})}


def run(spec_path, extra=None, bin_=None, txn=True):
    e = env(extra)
    if not txn:
        e.pop("MICROTORCH_CHECKPOINT_TXN")
    return subprocess.run([bin_ or BIN, "run", str(spec_path)], env=e, capture_output=True, text=True, timeout=300)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def finals(out):
    return {f: sha(out / f) for f in ("model.safetensors", "optim.safetensors", "state.txt", "m.safetensors")}


def steps_of(out):
    rows = {}
    for line in (out / "events.jsonl").read_text().splitlines():
        e = json.loads(line)
        if e.get("event") == "step":
            rows[e["step"]] = (e["loss"], e["grad_norm"])
    return rows


def events(out, kind):
    return [e for e in map(json.loads, (out / "events.jsonl").read_text().splitlines()) if e.get("event") == kind]


@pytest.fixture(scope="module")
def control(tmp_path_factory):
    if BIN is None:
        pytest.skip("no mtstudio")
    tmp = tmp_path_factory.mktemp("txn")
    setup(tmp, state_tokens=True)
    out = tmp / "control"
    r = run(spec(tmp, out, state_tokens=True))
    assert r.returncode == 0, r.stderr[-800:]
    return tmp, out, finals(out), steps_of(out)


@needs_bin
def test_layout_and_flat_names(control):
    tmp, out, _, _ = control
    assert os.readlink(out / "ckpt_gen" / "live") == f"g{STEPS}"
    for f in ("model.safetensors", "optim.safetensors", "state.txt"):
        assert os.readlink(out / f) == f"ckpt_gen/live/{f}"
    gens = sorted(p.name for p in (out / "ckpt_gen").iterdir() if p.name.startswith("g"))
    assert gens == [f"g{STEPS - EVERY}", f"g{STEPS}"]  # the previous generation is kept
    m = json.loads((out / "ckpt_gen" / f"g{STEPS}" / "MANIFEST.json").read_text())
    for f in ("model.safetensors", "optim.safetensors", "state.txt"):
        assert m["files"][f]["sha256"] == sha(out / f)
    assert (out / "state.txt").read_text().splitlines()[0] == str(STEPS)
    saves = events(out, "checkpoint")
    assert [e["step"] for e in saves] == list(range(EVERY, STEPS + 1, EVERY))
    assert all(e["seconds"] >= 0 and e["bytes"] > 0 for e in saves)


@needs_bin
@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("step", [EVERY, 3 * EVERY])
def test_crash_at_every_boundary_resumes_identical(control, stage, step):
    tmp, ctl, want, want_steps = control
    out = tmp / f"crash_{stage}_{step}"
    sp = spec(tmp, out, state_tokens=True)
    r = run(sp, {"MICROTORCH_CKPT_CRASH": f"{stage}:{step}"})
    assert r.returncode == 137, r.stderr[-400:]
    r = run(sp)
    assert r.returncode == 0, r.stderr[-800:]
    rec = events(out, "checkpoint_recover")[-1]
    committed = stage in ("commit", "links", "prune")
    expect = step if committed else step - EVERY
    assert rec["recovered_step"] == (expect if expect else -1), rec
    assert not rec["fallback"] and not rec["rejected"]
    assert finals(out) == want
    got = steps_of(out)
    assert {k: got[k] for k in want_steps} == want_steps  # the replayed steps match, RNG and step included


@needs_bin
def test_sigkill_mid_write_resumes_identical(control):
    tmp, ctl, want, want_steps = control
    out = tmp / "sigkill"
    sp = spec(tmp, out, state_tokens=True)
    target = out / "ckpt_gen" / f"g{2 * EVERY}.tmp" / "model.safetensors"
    p = subprocess.Popen([BIN, "run", str(sp)], env=env(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    while p.poll() is None and not target.exists() and time.time() - t0 < 120:
        time.sleep(0.001)
    if p.poll() is None:
        p.send_signal(signal.SIGKILL)
    p.wait()
    r = run(sp)
    assert r.returncode == 0, r.stderr[-800:]
    assert finals(out) == want


@needs_bin
def test_truncated_or_flipped_live_generation_falls_back_then_refuses(control):
    tmp, ctl, want, _ = control
    out = tmp / "damaged"
    sp = spec(tmp, out, state_tokens=True)
    r = run(sp, {"MICROTORCH_CKPT_CRASH": f"prune:{2 * EVERY}"})
    assert r.returncode == 137
    live = out / "ckpt_gen" / f"g{2 * EVERY}"
    with open(live / "optim.safetensors", "r+b") as f:
        f.truncate(os.path.getsize(live / "optim.safetensors") - 7)
    r = run(sp)
    assert r.returncode == 0, r.stderr[-800:]
    rec = events(out, "checkpoint_recover")[-1]
    assert rec["fallback"] is True and rec["live"] == f"g{EVERY}" and rec["recovered_step"] == EVERY
    assert any("size differs" in x for x in rec["rejected"])
    assert (out / "ckpt_gen" / f"rejected_g{2 * EVERY}").is_dir()   # kept, not deleted
    assert finals(out) == want
    out2 = tmp / "damaged_all"
    sp2 = spec(tmp, out2, state_tokens=True)
    assert run(sp2, {"MICROTORCH_CKPT_CRASH": f"prune:{2 * EVERY}"}).returncode == 137
    before = sorted(p.name for p in (out2 / "ckpt_gen").iterdir())
    for g in (out2 / "ckpt_gen").iterdir():
        if g.is_dir() and not g.is_symlink():
            with open(g / "model.safetensors", "r+b") as f:
                f.seek(64)
                f.write(b"\xff" * 8)
    r = run(sp2)
    assert r.returncode != 0
    assert "no whole generation" in (r.stderr + r.stdout)
    assert sorted(p.name for p in (out2 / "ckpt_gen").iterdir()) != before  # rejected ones renamed, none deleted
    assert len(list((out2 / "ckpt_gen").glob("rejected_g*"))) == 2


@needs_bin
def test_flat_checkpoint_of_an_old_run_resumes_and_converts(control):
    tmp, ctl, want, _ = control
    out = tmp / "legacy"
    sp = spec(tmp, out, steps=STEPS, state_tokens=True)
    sp_stop = spec(tmp, tmp / "legacy_tmp", steps=STEPS, state_tokens=True)
    r = run(sp_stop, {"MICROTORCH_CKPT_CRASH": f"links:{2 * EVERY}"})
    assert r.returncode == 137
    out.mkdir()
    src = tmp / "legacy_tmp" / "ckpt_gen" / f"g{2 * EVERY}"
    for f in ("model.safetensors", "optim.safetensors", "state.txt"):
        (out / f).write_bytes((src / f).read_bytes())
    # What a relay brings back: the events too, which hold the run's RoPE record (recorded_rope_heads).
    (out / "events.jsonl").write_bytes((tmp / "legacy_tmp" / "events.jsonl").read_bytes())
    assert run(sp).returncode == 0
    assert (out / "model.safetensors").is_symlink()
    assert finals(out) == want


@needs_bin
@pytest.mark.skipif(not BASE, reason="set MTSTUDIO_BASE_BIN to a build of the pinned source")
@pytest.mark.parametrize("state_tokens", [False, True])
def test_default_path_is_the_base_source_byte_for_byte(tmp_path, state_tokens):
    setup(tmp_path, state_tokens=state_tokens)
    a, b = tmp_path / "base", tmp_path / "new"
    assert run(spec(tmp_path, a, state_tokens=state_tokens), bin_=BASE, txn=False).returncode == 0
    r = run(spec(tmp_path, b, state_tokens=state_tokens), txn=False)
    assert r.returncode == 0, r.stderr[-800:]
    assert finals(a) == finals(b)
    assert steps_of(a) == steps_of(b)
    assert not (b / "ckpt_gen").exists()


@needs_bin
@pytest.mark.parametrize("state_tokens", [False, True])
def test_transactional_run_equals_plain_run(tmp_path, state_tokens):
    setup(tmp_path, state_tokens=state_tokens)
    a, b = tmp_path / "plain", tmp_path / "txn"
    assert run(spec(tmp_path, a, state_tokens=state_tokens), txn=False).returncode == 0
    r = run(spec(tmp_path, b, state_tokens=state_tokens))
    assert r.returncode == 0, r.stderr[-800:]
    assert finals(a) == finals(b)
    assert steps_of(a) == steps_of(b)
