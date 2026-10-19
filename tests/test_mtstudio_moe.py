"""mtstudio with a mixture-of-experts FFN (arch.moe), end to end on CPU.

A tiny llama-family MoE trains on a small synthetic corpus; checks that the
loss falls with the auxiliary losses on and logged, that checkpoint/resume
continues the uninterrupted run step for step, that sparse upcycling from a
dense checkpoint starts at the dense model's validation loss, and that the
GGUF export follows the gate convention. With MTSTUDIO_BASE_BIN set to a
binary built from master, a dense spec's events are compared against it
(steps, losses, model event: identical). Needs MTSTUDIO_BIN (or
./build/mtstudio); skipped otherwise. Everything is written to tmp_path.
"""
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_mtstudio_chat import BIN, write_vocab_gguf  # noqa: E402

needs_bin = pytest.mark.skipif(BIN is None, reason="set MTSTUDIO_BIN to a built mtstudio")
BASE = os.environ.get("MTSTUDIO_BASE_BIN")

WORDS = ["the", "cat", "dog", "sat", "ran", "on", "mat", "log", "a", "big", "small", "red", "blue", "."]


def corpus(n_sent=1500, seed=0):
    rng = random.Random(seed)
    out = []
    for _ in range(n_sent):
        adj = rng.choice(["big", "small", "red", "blue"])
        noun = rng.choice(["cat", "dog"])
        verb = "sat" if noun == "cat" else "ran"
        place = "mat" if adj in ("big", "red") else "log"
        out.append(f"the {adj} {noun} {verb} on a {place} .")
    return " ".join(out)


def setup(tmp, name, steps, moe=None, train_extra=None, formats=("safetensors",)):
    tmp.mkdir(parents=True, exist_ok=True)
    write_vocab_gguf(tmp / "v.gguf", ["<unk>"] + WORDS)
    (tmp / "c.txt").write_text(corpus(), encoding="utf-8")
    arch = {"preset": "llama-tiny", "custom": {"d": 32, "layers": 2, "heads": 2}}
    if moe is not None:
        arch["moe"] = moe
    train = {"steps": steps, "lr": 0.003, "eval_every": 10, "checkpoint_every": 10, "batch": 2, "seed": 3,
             "gradmap_every": 0}
    train.update(train_extra or {})
    spec = {"name": name, "arch": arch,
            "data": {"vocab": str(tmp / "v.gguf"), "corpus": str(tmp / "c.txt"), "T": 16, "max_tokens": 0},
            "train": train, "export": {"formats": list(formats)}, "out_dir": str(tmp / name)}
    p = tmp / f"{name}.json"
    p.write_text(json.dumps(spec), encoding="utf-8")
    return p


def run(spec, binary=None):
    p = subprocess.run([binary or BIN, "run", str(spec)], capture_output=True, text=True, timeout=1800)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    return p


def events(spec):
    out_dir = json.loads(Path(spec).read_text())["out_dir"]
    return [json.loads(x) for x in (Path(out_dir) / "events.jsonl").read_text().splitlines() if x.strip()]


def steps_of(ev):
    return [e for e in ev if e["event"] == "step"]


MOE = {"experts": 4, "top_k": 2, "aux_loss": 0.01, "z_loss": 0.001}


@needs_bin
def test_moe_trains_and_logs_routing(tmp_path):
    spec = setup(tmp_path, "moe", 30, moe=MOE)
    run(spec)
    ev = events(spec)
    model = next(e for e in ev if e["event"] == "model")
    m = model["moe"]
    assert (m["experts"], m["top_k"], m["gate"]) == (4, 2, "renorm")  # gate defaults to renorm for k >= 2
    assert m["aux_loss"] == pytest.approx(0.01) and m["z_loss"] == pytest.approx(0.001)  # float32, like lr
    st = steps_of(ev)
    assert len(st) == 30
    first, last = sum(e["loss"] for e in st[:3]) / 3, sum(e["loss"] for e in st[-3:]) / 3
    print(f"loss {first:.4f} -> {last:.4f}; entropy {st[-1]['moe_entropy']:.3f}; aux_lb {st[-1]['aux_lb']:.4f}")
    assert last < first - 0.3
    for e in st:
        assert len(e["moe_frac"]) == 2 and all(len(r) == 4 for r in e["moe_frac"])
        assert all(abs(sum(r) - 1.0) < 1e-3 for r in e["moe_frac"])
        assert 0.0 <= e["moe_entropy"] <= 1.0 + 1e-9
        assert e["aux_lb"] > 0
        assert e["aux_z"] > 0
    keys = json.loads((Path(tmp_path / "moe" / "spec.json")).read_text())["arch"]["moe"]
    assert keys["experts"] == 4 and keys["gate"] == "renorm"  # the resolved default is recorded


@needs_bin
def test_moe_resume_continues_the_run(tmp_path):
    full = setup(tmp_path / "a", "moe", 30, moe=MOE)
    run(full)
    part = setup(tmp_path / "b", "moe", 20, moe=MOE)
    run(part)
    # Same spec, longer: resumes from the step-20 checkpoint.
    spec = json.loads(part.read_text())
    spec["train"]["steps"] = 30
    part.write_text(json.dumps(spec))
    run(part)
    eb = events(part)
    assert any(e["event"] == "resume" and e["step"] == 20 and e["optimizer"] == "restored" for e in eb)
    la = [(e["step"], e["loss"], e["moe_frac"]) for e in steps_of(events(full))]
    lb = [(e["step"], e["loss"], e["moe_frac"]) for e in steps_of(eb)]
    assert [s for s, _, _ in lb] == list(range(1, 31))
    assert la == lb, "resumed MoE run diverged from the uninterrupted one"


@needs_bin
def test_upcycled_model_starts_at_the_dense_loss(tmp_path):
    dense = setup(tmp_path, "dense", 20)
    run(dense)
    ckpt = tmp_path / "dense" / "dense.safetensors"
    assert ckpt.is_file()
    # One zero-learning-rate step, evaluated: the model is the initial one.
    zero = {"lr": 0.0, "eval_every": 1, "steps": 1}
    d0 = setup(tmp_path, "d0", 1, train_extra=dict(zero, init_from=str(ckpt)))
    run(d0)
    u0 = setup(tmp_path, "u0", 1, moe=MOE, train_extra=dict(zero, upcycle_from=str(ckpt)))
    run(u0)
    ev_u = events(u0)
    up = next(e for e in ev_u if e["event"] == "upcycle")
    assert up["experts"] == 4 and up["routers_fresh"] == 2
    vd = next(e for e in events(d0) if e["event"] == "eval")["val_loss"]
    vu = next(e for e in ev_u if e["event"] == "eval")["val_loss"]
    sd = steps_of(events(d0))[0]["loss"]
    su = steps_of(ev_u)[0]["loss"]
    print(f"dense val {vd:.7f} upcycled val {vu:.7f}; step-1 loss {sd:.7f} vs {su:.7f}; "
          f"routing {steps_of(ev_u)[0]['moe_frac']}")
    assert abs(vd - vu) < 1e-5 and abs(sd - su) < 1e-5
    # The raw (Switch) gate with k = 1 does not start at the dense function.
    r0 = setup(tmp_path, "r0", 1, moe={"experts": 4, "top_k": 1},
               train_extra=dict(zero, upcycle_from=str(ckpt)))
    run(r0)
    vr = next(e for e in events(r0) if e["event"] == "eval")["val_loss"]
    assert abs(vr - vd) > 1e-3


@needs_bin
def test_gguf_export_follows_the_gate(tmp_path):
    spec = setup(tmp_path, "g", 2, moe=MOE, formats=("safetensors", "gguf"))
    run(spec)
    ex = [e for e in events(spec) if e["event"] == "export" and e["format"] == "gguf"]
    assert ex and Path(ex[0]["path"]).is_file()
    raw = setup(tmp_path, "r", 2, moe={"experts": 2, "top_k": 1, "gate": "raw"}, formats=("safetensors", "gguf"))
    run(raw)
    assert any(e["event"] == "export_skipped" and e["format"] == "gguf" for e in events(raw))


def test_spec_validation(tmp_path):
    if BIN is None:
        pytest.skip("set MTSTUDIO_BIN")
    bad = [
        ({"experts": 0}, None, "experts must be"),
        ({"experts": 2, "top_k": 3}, None, "top_k must be"),
        ({"experts": 2, "gate": "softmax"}, None, "gate must be"),
        (MOE, {"checkpoint_activations": True}, "checkpoint_activations"),
        (None, {"upcycle_from": "x.safetensors"}, "needs arch.moe"),
        (MOE, {"upcycle_from": "x.safetensors", "init_from": "y.safetensors"}, "exclusive"),
    ]
    for i, (moe, extra, msg) in enumerate(bad):
        spec = setup(tmp_path / str(i), "bad", 1, moe=moe, train_extra=extra)
        p = subprocess.run([BIN, "plan", str(spec)], capture_output=True, text=True, timeout=60)
        assert p.returncode != 0 and msg in p.stderr, (moe, extra, p.stderr)


@needs_bin
@pytest.mark.skipif(not BASE, reason="set MTSTUDIO_BASE_BIN to a master build")
def test_dense_spec_unchanged_against_master(tmp_path):
    a = setup(tmp_path / "new", "dense", 25)
    b = setup(tmp_path / "old", "dense", 25)
    run(a)
    run(b, BASE)

    def strip(ev):
        out = []
        for e in ev:
            if e["event"] in ("done",):
                continue
            e = {k: v for k, v in e.items() if k not in ("path", "wall_seconds")}
            out.append(e)
        return out

    ea, eb = strip(events(a)), strip(events(b))
    # out_dir differs only through paths, which are stripped above.
    assert [e["loss"] for e in steps_of(ea)] == [e["loss"] for e in steps_of(eb)]
    assert ea == eb
    sa = (tmp_path / "new" / "dense" / "dense.safetensors").read_bytes()
    sb = (tmp_path / "old" / "dense" / "dense.safetensors").read_bytes()
    assert sa == sb
