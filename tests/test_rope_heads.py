"""arch.rope_heads end to end: new runs rotate every head, legacy runs load
and resume as trained (first head only), and the sweep/replay tooling pins
the legacy value where registered findings need it.

The tooling checks run anywhere. The mtstudio checks need a built binary
(MTSTUDIO_BIN, or ./build/mtstudio) and train a 1-layer, 4-head llama for a
few dozen steps (seconds on CPU). MTSTUDIO_OLD_BIN, a binary built from a
commit before arch.rope_heads (e.g. 0151ba3), adds the bit-for-bit check:
the same spec trained by the old binary and by this one with rope_heads =
"first" must log identical losses and export identical weights.
Everything is written to pytest's tmp_path (never the OneDrive tree).
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(HERE))

import export_web_chat  # noqa: E402
import mtsweep  # noqa: E402
import reproduce  # noqa: E402
from get_tinystories_data import EOS, tokenize_like_mtstudio  # noqa: E402
from test_mtstudio_chat import BIN, DIALOGUE, write_vocab_gguf  # noqa: E402

OLD_BIN = os.environ.get("MTSTUDIO_OLD_BIN")
needs_bin = pytest.mark.skipif(BIN is None, reason="set MTSTUDIO_BIN to a built mtstudio")


# ---- tooling (no binary) ----

def _write_run(root, name, model_event):
    d = root / "runs" / name
    d.mkdir(parents=True)
    lines = [{"event": "start", "name": name}, dict(model_event, event="model")]
    (d / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in lines))


def test_sweep_resume_pins_first_for_legacy_runs(tmp_path):
    sweep = {"base": {"arch": {"preset": "llama-tiny"}}, "factors": {"train.lr": [1e-3, 3e-3]},
             "out_root": str(tmp_path)}
    assert not mtsweep.rope_heads_stated(sweep)
    assert mtsweep.any_llama(sweep)
    assert mtsweep.legacy_rope_runs(str(tmp_path)) == []
    _write_run(tmp_path, "run_000_c00_s1", {"family": "llama", "heads": 4})
    _write_run(tmp_path, "run_001_c01_s1", {"family": "llama", "heads": 4, "rope_heads": "all"})
    _write_run(tmp_path, "run_002_c00_s2", {"family": "gpt2", "heads": 4})
    assert mtsweep.legacy_rope_runs(str(tmp_path)) == ["run_000_c00_s1"]
    pinned = mtsweep.pin_rope_heads(sweep, "first")
    assert pinned["base"]["arch"] == {"preset": "llama-tiny", "rope_heads": "first"}
    assert "rope_heads" not in sweep["base"]["arch"]  # the input is not mutated
    assert mtsweep.rope_heads_stated(pinned)
    assert mtsweep.rope_heads_stated({"base": {}, "factors": {"arch.rope_heads": ["all", "first"]}})


def test_reproduce_replays_registered_llama_findings_with_first():
    stage3 = json.loads((ROOT / "experiments/atlas_stage3/sweep.json").read_text())
    old = {"id": "S3-x", "date": "2026-08-02"}
    new = {"id": "S9-x", "date": "2026-10-07"}
    assert reproduce.replay_rope_heads(old, stage3, "registered") == "first"
    assert reproduce.replay_rope_heads(new, stage3, "registered") is None  # mtstudio default: all
    assert reproduce.replay_rope_heads(old, stage3, "all") == "all"
    stated = mtsweep.pin_rope_heads(stage3, "all")
    assert reproduce.replay_rope_heads(old, stated, "registered") is None  # manifest wins
    gpt2 = {"base": {"arch": {"preset": "gpt2-nano"}}, "factors": {}}
    assert reproduce.replay_rope_heads(old, gpt2, "registered") is None


def test_web_export_reads_the_runs_record():
    assert export_web_chat.rope_heads_of({"event": "model"}, {}) == "first"
    assert export_web_chat.rope_heads_of({"event": "model", "rope_heads": "all"}, {}) == "all"
    assert export_web_chat.rope_heads_of(None, {"arch": {"rope_heads": "all"}}) == "all"
    assert export_web_chat.rope_heads_of(None, {}) == "first"


# ---- mtstudio end to end ----

def _vocab_and_corpus(tmp):
    corpus = tmp / "corpus.txt"
    text = "".join(f"user: {u} assistant: {a}{EOS}\n" for _ in range(60) for u, a in DIALOGUE)
    corpus.write_text(text)
    words = sorted({t for t in tokenize_like_mtstudio(text) if t != EOS})
    tokens = ["<unk>"] + words + [EOS]
    vocab = tmp / "vocab.gguf"
    write_vocab_gguf(vocab, tokens, len(tokens) - 1)
    return corpus, vocab


def _spec(tmp, out, steps=40, rope=None, name="ropey"):
    corpus, vocab = _vocab_and_corpus(tmp)
    arch = {"preset": "llama-tiny", "custom": {"d": 32, "layers": 1, "heads": 4}}
    if rope:
        arch["rope_heads"] = rope
    spec = {"name": name, "arch": arch,
            "data": {"corpus": str(corpus), "vocab": str(vocab), "vocab_cap": 0, "T": 16},
            "train": {"steps": steps, "batch": 2, "lr": 1e-2, "eval_every": 20,
                      "checkpoint_every": 20},
            "export": {"formats": ["safetensors", "gguf"]},
            "out_dir": str(out)}
    return spec


def _write(path, obj):
    path.write_text(json.dumps(obj))
    return path


def _mt(*args, binary=None, check=True):
    r = subprocess.run([binary or BIN, *map(str, args)], capture_output=True, text=True,
                       timeout=600, env=dict(os.environ, OMP_NUM_THREADS="1"))
    if check:
        assert r.returncode == 0, r.stderr + r.stdout
    return r


def _events(out):
    return [json.loads(x) for x in (out / "events.jsonl").read_text().splitlines() if x.strip()]


def _sample(spec_path):
    r = _mt("sample", spec_path, "--prompt", "user: hello there assistant:", "--tokens", "12",
            "--temp", "0", "--topk", "1")
    return r.stdout.strip().splitlines()[-1]


@needs_bin
def test_new_run_rotates_every_head_and_records_it(tmp_path):
    out = tmp_path / "run"
    sp = _write(tmp_path / "spec.json", _spec(tmp_path, out))
    _mt("run", sp)
    model = [e for e in _events(out) if e["event"] == "model"]
    assert model and all(e["rope_heads"] == "all" for e in model)
    assert json.loads((out / "spec.json").read_text())["arch"]["rope_heads"] == "all"
    gguf = next(e for e in _events(out) if e["event"] == "export" and e["format"] == "gguf")
    assert gguf["rope_heads"] == "all" and "warning" not in gguf


@needs_bin
def test_legacy_run_loads_and_resumes_as_first(tmp_path):
    out = tmp_path / "run"
    sp_first = _write(tmp_path / "spec_first.json", _spec(tmp_path, out, rope="first"))
    _mt("run", sp_first)
    want = _sample(sp_first)
    gguf = next(e for e in _events(out) if e["event"] == "export" and e["format"] == "gguf")
    assert gguf["rope_heads"] == "first" and "warning" in gguf

    # Make it a run from before the field existed: no rope_heads anywhere.
    events = _events(out)
    for e in events:
        e.pop("rope_heads", None)
    (out / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    rec = json.loads((out / "spec.json").read_text())
    del rec["arch"]["rope_heads"]
    _write(out / "spec.json", rec)
    sp_plain = _write(tmp_path / "spec_plain.json", _spec(tmp_path, out))

    # Loads as trained: the unstated spec samples exactly what "first" sampled.
    assert _sample(sp_plain) == want
    assert _sample(out / "spec.json") == want
    # Stating the other value against the record is refused, not obeyed.
    sp_all = _write(tmp_path / "spec_all.json", _spec(tmp_path, out, rope="all"))
    r = _mt("sample", sp_all, "--tokens", "3", check=False)
    assert r.returncode != 0 and "rope_heads" in r.stderr

    # Resume past the checkpoint with the unstated spec: still "first".
    sp_more = _write(tmp_path / "spec_more.json", _spec(tmp_path, out, steps=60))
    _mt("run", sp_more)
    model = [e for e in _events(out) if e["event"] == "model"]
    assert len(model) == 2 and "rope_heads" not in model[0] and model[1]["rope_heads"] == "first"
    assert json.loads((out / "spec.json").read_text())["arch"]["rope_heads"] == "first"
    r = _mt("run", sp_all, check=False)
    assert r.returncode != 0 and "rope_heads" in r.stderr


@needs_bin
def test_first_and_all_are_different_models(tmp_path):
    outs = {}
    for rope in ("first", "all"):
        out = tmp_path / rope
        sp = _write(tmp_path / f"{rope}.json", _spec(tmp_path, out, rope=rope))
        _mt("run", sp)
        outs[rope] = [e["loss"] for e in _events(out) if e["event"] == "step"]
    assert outs["first"] != outs["all"]  # same init and data; only the rotation differs


@pytest.mark.skipif(BIN is None or not OLD_BIN, reason="set MTSTUDIO_OLD_BIN (a pre-rope_heads build)")
def test_first_is_bit_for_bit_the_pre_fix_engine(tmp_path):
    runs = {}
    for tag, binary, rope in (("old", OLD_BIN, None), ("new", BIN, "first")):
        out = tmp_path / tag
        sp = _write(tmp_path / f"{tag}.json", _spec(tmp_path, out, rope=rope, name="ropey"))
        _mt("run", sp, binary=binary)
        ev = _events(out)
        runs[tag] = ([(e["step"], e["loss"], e["grad_norm"]) for e in ev if e["event"] == "step"],
                     [e["val_loss"] for e in ev if e["event"] == "eval"],
                     (out / "ropey.safetensors").read_bytes())
    assert runs["old"][0] == runs["new"][0]
    assert runs["old"][1] == runs["new"][1]
    assert runs["old"][2] == runs["new"][2]
