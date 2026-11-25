"""data.warmup (tools/mtstudio.cpp), end to end on CPU with the real mtstudio: SYMW's one trainer change
(AI_ML/chat_screens_claude/base2/token_ladder/symbolic_warmup/REGISTRATION.md in the author's research tree).
Claude Opus 5.5, 26 Nov 2026.

The claim: a spec with data.warmup {corpus X, steps W} draws, row for row, the same raw values of the window generator as
the same spec without it; its windows of steps 1 to W are taken from X (offset = draw mod X's limit) and every window from
step W + 1 on, the tail's included, is the one the spec without it takes. MICROTORCH_LOG_WINDOWS=1 makes the trainer log
each micro-step's draws, offsets and corpus. Needs MTSTUDIO_BIN; skipped otherwise. Everything is written to tmp_path.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_mtstudio_chat import BIN, write_vocab_gguf  # noqa: E402
from test_checkpoint_txn import WORDS, chat_corpus, corpus  # noqa: E402

needs_bin = pytest.mark.skipif(BIN is None, reason="set MTSTUDIO_BIN to a built mtstudio")
STEPS, W = 20, 6


def setup(tmp, state_tokens):
    specials = ["<|endoftext|>", "<|user|>", "<|assistant|>"] if state_tokens else []
    write_vocab_gguf(tmp / "v.gguf", ["<unk>"] + specials + WORDS)
    make = chat_corpus if state_tokens else corpus
    (tmp / "main.txt").write_text(make(1500, 0), encoding="utf-8")
    (tmp / "tail.txt").write_text(make(400, 1), encoding="utf-8")
    (tmp / "warm.txt").write_text(make(300, 2), encoding="utf-8")


def spec(tmp, out, warm=None, state_tokens=False, steps=STEPS, anneal=0.5):
    arch = {"preset": "llama-tiny", "custom": {"d": 32, "layers": 2, "heads": 2}}
    if state_tokens:
        arch["state_tokens"] = {"k": 2, "mask": "causal"}
    data = {"corpus": str(tmp / "main.txt"), "vocab": str(tmp / "v.gguf"), "T": 32, "max_tokens": 0,
            "anneal": {"corpus": str(tmp / "tail.txt"), "from_fraction": anneal}}
    if warm is not None:
        data["warmup"] = warm
    s = {"name": "m", "arch": arch, "data": data,
         "train": {"steps": steps, "lr": 0.003, "batch": 2, "accum": 2, "eval_every": 10, "eval_windows": 2,
                   "checkpoint_every": 10, "seed": 3, "optimizer": "adamw", "gradmap_every": 0},
         "export": {"formats": []}, "out_dir": str(out)}
    p = Path(str(out) + ".spec.json")
    p.write_text(json.dumps(s), encoding="utf-8")
    return p


def run(p):
    e = {**os.environ, "OMP_NUM_THREADS": "1", "MICROTORCH_LOG_WINDOWS": "1", "CUDA_VISIBLE_DEVICES": ""}
    return subprocess.run([BIN, "run", str(p)], env=e, capture_output=True, text=True, timeout=300)


def evs(out, kind):
    return [e for e in map(json.loads, (out / "events.jsonl").read_text().splitlines()) if e.get("event") == kind]


@needs_bin
@pytest.mark.parametrize("state_tokens", [False, True])
def test_warmup_replaces_first_windows_only(tmp_path, state_tokens):
    setup(tmp_path, state_tokens)
    a, b = tmp_path / "with", tmp_path / "without"
    ra = run(spec(tmp_path, a, {"corpus": str(tmp_path / "warm.txt"), "steps": W}, state_tokens))
    rb = run(spec(tmp_path, b, None, state_tokens))
    assert ra.returncode == 0, ra.stderr[-800:]
    assert rb.returncode == 0, rb.stderr[-800:]
    wa, wb = evs(a, "windows"), evs(b, "windows")
    assert len(wa) == len(wb) == STEPS * 2
    wd = evs(a, "warmup_data")
    assert len(wd) == 1 and wd[0]["through_step"] == W and not evs(b, "warmup_data")
    warm_lim = wd[0]["tokens"] - 32 - 1
    tail_from = int(0.5 * STEPS)
    for x, y in zip(wa, wb):
        assert (x["step"], x["micro"]) == (y["step"], y["micro"])
        assert x["raw"] == y["raw"]                                   # the generator is consumed identically
        if x["step"] <= W:
            assert x["src"] == "warmup" and y["src"] == "main"
            assert x["lim"] == warm_lim
            assert x["at"] == [r % warm_lim for r in x["raw"]]
        else:
            assert x["src"] == y["src"] == ("anneal" if x["step"] > tail_from else "main")
            assert x["at"] == y["at"] and x["lim"] == y["lim"]         # N's own windows from step W + 1
    # The eval windows are data.corpus's held-out part in both runs.
    ea, eb = evs(a, "eval"), evs(b, "eval")
    assert [e["step"] for e in ea] == [e["step"] for e in eb]


@needs_bin
@pytest.mark.parametrize("warm,why", [
    ({"corpus": "", "steps": 3}, "needs a corpus"),
    ({"steps": 3}, "needs a corpus"),
    ({"corpus": "WARM", "steps": 0}, "steps > 0"),
    ({"corpus": "WARM", "steps": 10}, "must end before data.anneal"),   # the tail starts at step 11: a warm-up must end before step 10
    ({"corpus": "WARM", "steps": 25}, "must end before"),
])
def test_warmup_refused(tmp_path, warm, why):
    setup(tmp_path, False)
    w = dict(warm)
    if w.get("corpus") == "WARM":
        w["corpus"] = str(tmp_path / "warm.txt")
    r = run(spec(tmp_path, tmp_path / "bad", w))
    assert r.returncode != 0 and why in (r.stderr + r.stdout)


@needs_bin
def test_warmup_up_to_the_last_main_step(tmp_path):
    setup(tmp_path, False)
    out = tmp_path / "edge"
    r = run(spec(tmp_path, out, {"corpus": str(tmp_path / "warm.txt"), "steps": 9}))
    assert r.returncode == 0, r.stderr[-800:]
    srcs = {e["step"]: e["src"] for e in evs(out, "windows")}
    assert [srcs[s] for s in (9, 10, 11)] == ["warmup", "main", "anneal"]
