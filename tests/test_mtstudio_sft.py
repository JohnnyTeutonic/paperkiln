"""mtstudio post-training paths end to end on CPU (charter 0001, Phases 2-3):
`train.objective: "sft"` and `"rejection_sft"`, `train.init_from`, and the
`sample-set` sampler. A tiny llama-family model learns a ten-fact lookup
task in a few seconds. Needs MTSTUDIO_BIN (or ./build/mtstudio); skipped
otherwise. Everything is written to pytest's tmp_path.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_mtstudio_chat import BIN, write_vocab_gguf  # noqa: E402

needs_bin = pytest.mark.skipif(BIN is None, reason="set MTSTUDIO_BIN to a built mtstudio")

EOT = "<|endoftext|>"
WORDS = ["what", "is", "plus", "?", "the", "answer", "."] + [str(i) for i in range(20)]
FACTS = [(a, b) for a in range(5) for b in range(2)]


def setup(tmp, examples, objective="sft", steps=200, extra_train=None):
    vocab = ["<unk>", EOT] + WORDS
    write_vocab_gguf(tmp / "v.gguf", vocab, eos_id=1)
    (tmp / "ex.jsonl").write_text("".join(json.dumps(e) + "\n" for e in examples), encoding="utf-8")
    train = {"steps": steps, "lr": 0.003, "eval_every": 50, "checkpoint_every": 1000, "batch": 4,
             "seed": 3, "gradmap_every": 0, "objective": objective}
    train.update(extra_train or {})
    spec = {"name": "sft", "arch": {"preset": "llama-tiny"},
            "data": {"vocab": str(tmp / "v.gguf"), "vocab_cap": 64, "T": 16,
                     "sft_examples": str(tmp / "ex.jsonl")},
            "train": train, "export": {"formats": ["safetensors"]}, "out_dir": str(tmp / "run")}
    (tmp / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    return tmp / "spec.json"


def run(spec):
    return subprocess.run([BIN, "run", str(spec)], capture_output=True, text=True, timeout=2400)


def events(tmp):
    return [json.loads(x) for x in (tmp / "run" / "events.jsonl").read_text().splitlines() if x.strip()]


def qa(a, b):
    return {"prompt": f"what is {a} plus {b} ?", "completion": f"the answer is {a + b} ."}


@needs_bin
def test_sft_learns_completions_and_reports_its_objective(tmp_path):
    ex = [qa(a, b) for a, b in FACTS] * 3 + [qa(0, 0)]  # the last is the held-out tail
    spec = setup(tmp_path, ex, steps=300)
    p = run(spec)
    assert p.returncode == 0, p.stderr
    ev = events(tmp_path)
    data = next(e for e in ev if e["event"] == "data")
    model = next(e for e in ev if e["event"] == "model")
    assert data["objective"] == "sft" and data["examples"] == len(ex) and data["val_examples"] == 1
    assert data["completion_unk"] == 0 and model["objective"] == "sft"
    # Only completion targets count: 6 tokens ("the answer is N ." + end of text) per example.
    assert data["tokens"] == 6 * (len(ex) - 1)
    losses = [e["loss"] for e in ev if e["event"] == "step"]
    median = lambda xs: sorted(xs)[len(xs) // 2]
    assert median(losses[-40:]) < 0.3 * median(losses[:20])
    vals = [e["val_loss"] for e in ev if e["event"] == "eval"]
    assert vals[-1] < vals[0]


@needs_bin
def test_rejection_sft_needs_receipts_and_refuses_human_labels(tmp_path):
    def r(a, b, acc=True, kind="exact_match", **drop):
        rec = dict(qa(a, b), reward=1.0, rationale="matches", evidence_kind=kind, evidence_uri=f"t/{a}-{b}",
                   accepted=acc, audited=False)
        for k in drop:
            rec.pop(k)
        return rec
    good = [r(a, b) for a, b in FACTS] + [r(1, 1, acc=False), r(2, 1, acc=False)]
    spec = setup(tmp_path, good, objective="rejection_sft", steps=20)
    p = run(spec)
    assert p.returncode == 0, p.stderr
    data = next(e for e in events(tmp_path) if e["event"] == "data")
    assert data["examples"] == len(FACTS)
    assert data["provenance"] == {"evidence_kinds": {"exact_match": len(FACTS)}, "not_accepted_skipped": 2}

    no_receipt = [r(a, b) for a, b in FACTS[:3]] + [r(0, 1, rationale=None)]
    (tmp_path / "a").mkdir()
    p = run(setup(tmp_path / "a", no_receipt, objective="rejection_sft", steps=5))
    assert p.returncode != 0 and "rationale" in p.stderr

    human = [r(a, b) for a, b in FACTS[:3]] + [r(0, 1, kind="human")]
    (tmp_path / "h").mkdir()
    p = run(setup(tmp_path / "h", human, objective="rejection_sft", steps=5))
    assert p.returncode != 0 and "allow_human_labels" in p.stderr
    (tmp_path / "h2").mkdir()
    p = run(setup(tmp_path / "h2", human, objective="rejection_sft", steps=5,
                  extra_train={"allow_human_labels": True}))
    assert p.returncode == 0, p.stderr


@needs_bin
def test_example_longer_than_T_is_refused(tmp_path):
    long = {"prompt": "what is " * 10 + "?", "completion": "the answer is 1 ."}
    p = run(setup(tmp_path, [long, qa(1, 1)], steps=5))
    assert p.returncode != 0 and "data.T" in p.stderr


@needs_bin
def test_init_from_then_pure_sampler_replays(tmp_path):
    ex = [qa(a, b) for a, b in FACTS] * 3 + [qa(0, 0)]
    base = tmp_path / "base"
    base.mkdir()
    assert run(setup(base, ex, steps=300)).returncode == 0
    ck = base / "run" / "sft.safetensors"
    tuned = tmp_path / "tuned"
    tuned.mkdir()
    p = run(setup(tuned, ex, steps=10, extra_train={"init_from": str(ck)}))
    assert p.returncode == 0, p.stderr
    ev = events(tuned)
    assert any(e["event"] == "init" and e["from"] == str(ck) for e in ev)
    # Starting from the trained base, the first loss is already low.
    first = next(e["loss"] for e in ev if e["event"] == "step")
    base_first = next(e["loss"] for e in events(base) if e["event"] == "step")
    assert first < 0.5 * base_first
    # The sampler: pure temperature, deterministic per (seed, prompt, sample).
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text("".join(json.dumps({"prompt": e["prompt"], "uri": f"q/{i}"}) + "\n"
                               for i, e in enumerate(ex[:4])), encoding="utf-8")
    outs = []
    for k in range(2):
        out = tmp_path / f"s{k}.jsonl"
        p = subprocess.run([BIN, "sample-set", str(base / "spec.json"), "--prompts", str(prompts), "--out", str(out),
                            "--n", "3", "--temp", "1.0", "--seed", "5", "--max-new", "8"],
                           capture_output=True, text=True, timeout=300)
        assert p.returncode == 0, p.stderr
        outs.append(out.read_bytes())
    assert outs[0] == outs[1]
    recs = [json.loads(x) for x in outs[0].decode().splitlines()]
    assert len(recs) == 12 and {r["sampler"] for r in recs} == {"pure-temperature-v1"}
    assert all(r["logp"] <= 0 and r["prompt_uri"].startswith("q/") for r in recs)
    # The sampler's job is to report the policy, not to be right: every
    # completion has the trained template and ends at end-of-text, and a
    # completion sampled from a confident distribution has logp near 0.
    import re
    assert all(re.fullmatch(r"the answer is [0-9]+ \.", r["completion"]) and r["stop"] == "eos" for r in recs)
    assert min(r["logp"] for r in recs) > -3.0 and max(r["logp"] for r in recs) > -0.1
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run([BIN, "sample-set", str(base / "spec.json"), "--prompts", str(prompts),
                        "--out", str(tmp_path / "x.jsonl"), "--temp", "0"], check=True, capture_output=True)
