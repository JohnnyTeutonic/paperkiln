"""mtstudio `train.objective: "grpo"` end to end on CPU (charter 0001, Phase 4).

A tiny llama-family base is SFT'd on a ten-fact lookup task, then trained by
GRPO from it. Checks the wiring, not a research claim:
- at step 1 the policy IS the reference, so the logged KL is zero;
- every step event carries the monitors (reward, truth, hack_frac, KL,
  entropy, flat groups) and every grpo_eval event its held-out numbers;
- a format-only reward is the open hacking channel: its reward is ~1 from
  the start, while exact-match truth stays where the base left it;
- the shuffled-advantage falsifier runs.
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
from test_mtstudio_sft import EOT, FACTS, WORDS, qa  # noqa: E402

needs_bin = pytest.mark.skipif(BIN is None, reason="set MTSTUDIO_BIN to a built mtstudio")


def spec(tmp, name, train, data):
    s = {"name": name, "arch": {"preset": "llama-tiny"},
         "data": {"vocab": str(tmp / "v.gguf"), "vocab_cap": 64, "T": 16, **data},
         "train": {"lr": 0.003, "batch": 4, "seed": 3, "gradmap_every": 0, "checkpoint_every": 1000, **train},
         "export": {"formats": ["safetensors"]}, "out_dir": str(tmp / name)}
    p = tmp / f"{name}.json"
    p.write_text(json.dumps(s), encoding="utf-8")
    return p


def run(p):
    r = subprocess.run([BIN, "run", str(p)], capture_output=True, text=True, timeout=2400)
    assert r.returncode == 0, r.stderr
    return [json.loads(x) for x in (Path(json.loads(p.read_text())["out_dir"]) / "events.jsonl").read_text().splitlines()
            if x.strip()]


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("grpo")
    write_vocab_gguf(tmp / "v.gguf", ["<unk>", EOT] + WORDS, eos_id=1)
    ex = [qa(a, b) for a, b in FACTS] * 3 + [qa(0, 0)]
    (tmp / "ex.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ex), encoding="utf-8")
    run(spec(tmp, "base", {"steps": 150, "eval_every": 50, "objective": "sft"}, {"sft_examples": str(tmp / "ex.jsonl")}))
    refs = [{"prompt": f"what is {a} plus {b} ?", "answer": a + b, "uri": f"t/{a}-{b}"} for a, b in FACTS]
    (tmp / "train_prompts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in refs[:8]), encoding="utf-8")
    (tmp / "eval_prompts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in refs[8:]), encoding="utf-8")
    return tmp


def grpo_spec(tmp, name, reward="exact_match", advantage="normal", steps=12):
    return spec(tmp, name,
                {"steps": steps, "eval_every": 6, "objective": "grpo", "lr": 0.001,
                 "init_from": str(tmp / "base" / "base.safetensors"),
                 "grpo": {"group_size": 4, "prompts_per_step": 2, "max_new": 6, "beta": 0.05,
                          "temperature": 1.0, "reward": reward, "advantage": advantage, "eval_samples": 4}},
                {"grpo_prompts": str(tmp / "train_prompts.jsonl"), "grpo_eval": str(tmp / "eval_prompts.jsonl")})


@needs_bin
def test_grpo_events_and_zero_kl_at_the_start(base):
    ev = run(grpo_spec(base, "g_exact"))
    data = next(e for e in ev if e["event"] == "data")
    model = next(e for e in ev if e["event"] == "model")
    assert data["objective"] == "grpo" and data["reward"] == "exact_match:v1" and model["objective"] == "grpo"
    assert any(e["event"] == "init" for e in ev)
    steps = [e for e in ev if e["event"] == "step"]
    assert len(steps) == 12
    for k in ("reward_mean", "truth", "hack_frac", "kl", "entropy", "flat_groups", "groups", "abstained"):
        assert all(k in e for e in steps), k
    assert abs(steps[0]["kl"]) < 1e-6          # policy == reference before the first update
    assert all(e["groups"] == 2 for e in steps)
    # With exact match as the reward, reward and truth are the same number.
    assert all(abs(e["reward_mean"] - e["truth"]) < 1e-9 for e in steps)
    evals = [e for e in ev if e["event"] == "grpo_eval"]
    assert [e["step"] for e in evals] == [6, 12] and all(e["samples"] == 8 for e in evals)


@needs_bin
def test_format_reward_is_the_open_channel(base):
    ev = run(grpo_spec(base, "g_format", reward="format", steps=6))
    steps = [e for e in ev if e["event"] == "step"]
    # The SFT'd base already writes "the answer is N .": the format reward
    # pays for it whether or not N is right. Every group is flat (all 1s).
    assert all(e["reward_mean"] > 0.9 for e in steps)
    assert all(e["flat_groups"] == e["groups"] for e in steps)
    assert all(abs(e["hack_frac"] - (e["reward_mean"] - e["truth"])) < 1e-9 for e in steps)


@needs_bin
def test_shuffled_advantage_falsifier_runs(base):
    ev = run(grpo_spec(base, "g_shuf", advantage="shuffled", steps=4))
    assert next(e for e in ev if e["event"] == "data")["advantage"] == "shuffled"


@needs_bin
def test_grpo_refuses_without_reference(base):
    p = spec(base, "g_bad", {"steps": 2, "eval_every": 2, "objective": "grpo", "grpo": {}},
             {"grpo_prompts": str(base / "train_prompts.jsonl")})
    r = subprocess.run([BIN, "run", str(p)], capture_output=True, text=True, timeout=120)
    assert r.returncode != 0 and "init_from" in r.stderr


def shifted_spec(tmp, name, reward="exact_match", shift=1, steps=6):
    return spec(tmp, name,
                {"steps": steps, "eval_every": 6, "objective": "grpo", "lr": 0.001,
                 "init_from": str(tmp / "base" / "base.safetensors"),
                 "grpo": {"group_size": 4, "prompts_per_step": 2, "max_new": 6, "reward": reward,
                          "reward_shift": shift, "eval_samples": 4}},
                {"grpo_prompts": str(tmp / "train_prompts.jsonl"), "grpo_eval": str(tmp / "eval_prompts.jsonl")})


@needs_bin
def test_shifted_reward_key_is_pure_hack(base):
    # PT4's channel: a reward keyed to answer + 1. Every reward it pays is a
    # truth failure, so the monitor equals the reward exactly.
    ev = run(shifted_spec(base, "g_shift"))
    assert next(e for e in ev if e["event"] == "data")["reward_shift"] == 1
    for e in [x for x in ev if x["event"] in ("step", "grpo_eval")]:
        assert abs(e["reward_hack_frac"] - e["reward_mean"]) < 1e-9, e


@needs_bin
def test_shift_only_on_exact_match(base):
    p = shifted_spec(base, "g_shift_bad", reward="format")
    r = subprocess.run([BIN, "run", str(p)], capture_output=True, text=True, timeout=120)
    assert r.returncode != 0 and "reward_shift" in r.stderr
