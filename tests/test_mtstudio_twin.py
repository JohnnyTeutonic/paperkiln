"""mtstudio `train.twin_loss` (lane amendment A31, include/microtorch/posttrain/twin.hpp) end to end on CPU.
Claude Opus 5.5, 13 Nov 2026. Wiring checks on a two-variable lookup, not a research claim:
- the SFT batches are the plain spec's: step 1's SFT loss is bit-identical with and without the twin loss, and
  later steps differ (the twin gradient moved the weights);
- every step logs the twin loss, the per-token decisive shifts dA and dB, the nuisance shift dN and the share of
  active hinges; the data event counts groups and rows;
- the margin raises the decisive shift over training;
- resuming a twin run reproduces the uninterrupted run (the groups are a function of the seed and the step);
- with state tokens the rows load through the same path;
- the spec and the data are refused when they do not fit together (no silent drop).
Needs MTSTUDIO_BIN (or ./build/mtstudio); skipped otherwise.
"""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_mtstudio_chat import BIN, write_vocab_gguf  # noqa: E402

needs_bin = pytest.mark.skipif(BIN is None, reason="set MTSTUDIO_BIN to a built mtstudio")
EOT = "<|endoftext|>"
BPE_VOCAB = Path("/mnt/c/ml_artifacts/capacity/vocab/vocab_llama3_bpe.gguf")  # A31's vocabulary
WORDS = ["x", "y", "is", "what", "?", "."] + [str(i) for i in range(10)]


def triplet(a, b, b2):
    """A asks for x, B for y; N changes y's value, which A's question does not read."""
    told = f"x is {a} . y is {b} ."
    return {"A": [f"{told} what is x ?", f"{a} ."],
            "B": [f"{told} what is y ?", f"{b} ."],
            "N": [f"x is {a} . y is {b2} . what is x ?", f"{a} ."]}


TRIPLETS = [triplet(a, b, (b + 3) % 10) for a in range(10) for b in range(10) if a != b][:60]


def sft_rows():
    rows = []
    for t in TRIPLETS:
        for k in ("A", "B", "N"):
            rows.append({"prompt": t[k][0], "completion": t[k][1]})
    return rows + rows[:4]  # the tail is the held-out split


def spec(tmp, name, train=None, data=None, arch=None, T=24):
    s = {"name": name, "arch": {"preset": "llama-tiny", **(arch or {})},
         "data": {"vocab": str(tmp / "v.gguf"), "vocab_cap": 64, "T": T, "sft_examples": str(tmp / "sft.jsonl"),
                  **(data or {})},
         "train": {"objective": "sft", "steps": 30, "lr": 0.003, "batch": 4, "seed": 3, "gradmap_every": 0,
                   "eval_every": 1000, "checkpoint_every": 1000, **(train or {})},
         "export": {"formats": ["safetensors"]}, "out_dir": str(tmp / name)}
    p = tmp / f"{name}.json"
    p.write_text(json.dumps(s), encoding="utf-8")
    return p


def run(p, ok=True):
    r = subprocess.run([BIN, "run", str(p)], capture_output=True, text=True, timeout=1800,
                       env={"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin"})
    if not ok:
        return r
    assert r.returncode == 0, r.stderr
    return [json.loads(x) for x in (Path(json.loads(p.read_text())["out_dir"]) / "events.jsonl").read_text().splitlines()
            if x.strip()]


def twin_train(lam=0.5, mu=0.2, margin=1.0, groups=3, **kw):
    return {"twin_loss": {"groups": groups, "margin": margin, "lambda": lam, "mu": mu}, **kw}


@pytest.fixture(scope="module")
def tmp(tmp_path_factory):
    t = tmp_path_factory.mktemp("twin")
    write_vocab_gguf(t / "v.gguf", ["<unk>", EOT] + WORDS, eos_id=1)
    (t / "sft.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sft_rows()), encoding="utf-8")
    (t / "twins.jsonl").write_text("".join(json.dumps(x) + "\n" for x in TRIPLETS), encoding="utf-8")
    return t


def steps(ev):
    return [e for e in ev if e["event"] == "step"]


@needs_bin
def test_sft_batches_are_the_plain_specs(tmp):
    plain = steps(run(spec(tmp, "plain")))
    ev = run(spec(tmp, "twin", twin_train(), {"twin_examples": str(tmp / "twins.jsonl")}))
    twin = steps(ev)
    assert plain[0]["loss"] == twin[0]["loss"]          # same weights, same SFT batch at step 1
    assert plain[1]["loss"] != twin[1]["loss"]          # the twin gradient moved the weights
    # every step draws the same SFT rows with and without the twin term (A31 launch review, 13 Nov 2026)
    assert len(plain) == len(twin) == 30
    assert [e["sft_rows_hash"] for e in plain] == [e["sft_rows_hash"] for e in twin]
    assert len({e["sft_rows_hash"] for e in plain}) > 1
    data = next(e for e in ev if e["event"] == "twin_data")
    assert data["groups"] == len(TRIPLETS) and data["rows_per_group"] == 5
    assert data["twin_loss"] == {"groups_per_step": 3, "margin": 1.0, "lambda": 0.5, "mu": pytest.approx(0.2)}
    for e in twin:
        assert set(e["twin"]) == {"loss", "d_a", "d_b", "d_n", "active", "len"}
        assert 0.0 <= e["twin"]["active"] <= 1.0 and e["twin"]["d_n"] >= 0.0
    assert all("twin" not in e for e in plain)


@needs_bin
def test_mu_zero_drops_the_nuisance_row(tmp):
    ev = run(spec(tmp, "margin_only", twin_train(mu=0.0, steps=3), {"twin_examples": str(tmp / "twins.jsonl")}))
    assert next(e for e in ev if e["event"] == "twin_data")["rows_per_group"] == 4
    assert all("d_n" not in e["twin"] for e in steps(ev))


def decisive_shift(tmp, p):
    """Mean per-token decisive shift over every twin group, scored by `mtstudio score` on the trained model."""
    req = []
    for t in TRIPLETS:
        for ctx, rep in (("A", "A"), ("B", "A"), ("B", "B"), ("A", "B")):
            req.append({"prompt": t[ctx][0], "continuation": t[rep][1] + EOT})
    (tmp / "req.jsonl").write_text("".join(json.dumps(r) + "\n" for r in req), encoding="utf-8")
    out = tmp / f"{p.stem}.scores.jsonl"
    r = subprocess.run([BIN, "score", str(p), "--in", str(tmp / "req.jsonl"), "--out", str(out)], capture_output=True,
                       text=True, timeout=600, env={"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin"})
    assert r.returncode == 0, r.stderr
    rec = [json.loads(x) for x in out.read_text().splitlines() if x.strip()]
    d = []
    for g in range(len(TRIPLETS)):
        aa, ba, bb, ab = rec[4 * g:4 * g + 4]
        d.append((aa["sum"] - ba["sum"]) / len(aa["logprobs"]))
        d.append((bb["sum"] - ab["sum"]) / len(bb["logprobs"]))
    return sum(d) / len(d)


@needs_bin
def test_margin_raises_the_decisive_shift(tmp):
    tw = {"twin_examples": str(tmp / "twins.jsonl")}
    plain = spec(tmp, "long_plain", {"steps": 200, "lr": 0.002})
    twin = spec(tmp, "long_twin", twin_train(lam=1.0, mu=0.0, margin=1.0, groups=6, steps=200, lr=0.002), tw)
    run(plain)
    ev = steps(run(twin))
    dp, dt = decisive_shift(tmp, plain), decisive_shift(tmp, twin)
    assert dt > dp + 0.1, (dp, dt)
    assert ev[-1]["twin"]["active"] < 1.0


@needs_bin
def test_resume_draws_the_same_groups(tmp):
    whole = steps(run(spec(tmp, "whole", twin_train(steps=12, checkpoint_every=6), {"twin_examples": str(tmp / "twins.jsonl")})))
    p = spec(tmp, "cut", twin_train(steps=6, checkpoint_every=6), {"twin_examples": str(tmp / "twins.jsonl")})
    run(p)
    s = json.loads(p.read_text())
    s["train"]["steps"] = 12
    p.write_text(json.dumps(s), encoding="utf-8")
    resumed = steps(run(p))
    tail = [e for e in resumed if e["step"] > 6]
    assert [e["step"] for e in tail] == list(range(7, 13))
    for a, b in zip(whole[6:], tail):
        assert a["loss"] == b["loss"] and a["twin"] == b["twin"], (a, b)


@needs_bin
def test_state_tokens_rows_load(tmp):
    # state ids are defined over the paperkiln/1 template, so a BPE vocabulary with its specials: A31 vocabulary
    if not BPE_VOCAB.exists():
        pytest.skip(f"no BPE vocabulary at {BPE_VOCAB}")
    arch = {"state_tokens": {"k": 4, "mask": "causal"}}
    chat = dict(TRIPLETS[0])
    chat["A"] = ["<|user|>x is 1 .<|assistant|>noted .<|endoftext|><|user|>what is x ?<|assistant|>", "1 ."]
    chat["B"] = ["<|user|>x is 2 .<|assistant|>noted .<|endoftext|><|user|>what is x ?<|assistant|>", "2 ."]
    chat["N"] = ["<|user|>x is 1 .<|assistant|>noted , then .<|endoftext|><|user|>what is x ?<|assistant|>", "1 ."]
    (tmp / "twins_chat.jsonl").write_text("".join(json.dumps(x) + "\n" for x in [chat] + TRIPLETS[:5]), encoding="utf-8")
    ev = run(spec(tmp, "state", twin_train(steps=3), {"twin_examples": str(tmp / "twins_chat.jsonl"),
                                                      "vocab": str(BPE_VOCAB), "vocab_cap": 16384}, arch=arch, T=64))
    assert next(e for e in ev if e["event"] == "twin_data")["groups"] == 6
    assert len(steps(ev)) == 3
    # the chat group's five rows each hold one state block, after the first reply's close
    assert next(e for e in ev if e["event"] == "twin_data")["state_blocks"] == 5


@needs_bin
def test_refusals(tmp):
    tw = {"twin_examples": str(tmp / "twins.jsonl")}

    def refused(name, train=None, data=None, T=24, needle=""):
        r = run(spec(tmp, name, train, data, T=T), ok=False)
        assert r.returncode != 0 and needle in r.stderr, r.stderr
    refused("no_file", twin_train(), None, needle="go together")
    refused("no_loss", None, tw, needle="go together")
    refused("zero", twin_train(lam=0.0, mu=0.0), tw, needle="not both 0")
    refused("lm", twin_train(objective="lm"), {**tw, "corpus": str(tmp / "sft.jsonl")}, needle="sft objective only")
    refused("unknown", {"twin_loss": {"groups": 2, "margin": 1, "lambda": 1, "mu": 0, "beta": 1}}, tw, needle="unknown key")
    short = [{"prompt": f"x is {i} .", "completion": f"{i} ."} for i in range(10)]
    (tmp / "short.jsonl").write_text("".join(json.dumps(r) + "\n" for r in short), encoding="utf-8")
    refused("over_T", twin_train(), {**tw, "sft_examples": str(tmp / "short.jsonl")}, T=10, needle="data.twin_examples")
    bad = dict(TRIPLETS[0])
    bad["N"] = [bad["N"][0], "9 ."]
    (tmp / "bad.jsonl").write_text(json.dumps(bad) + "\n" + json.dumps(TRIPLETS[1]) + "\n", encoding="utf-8")
    refused("bad_n", twin_train(), {"twin_examples": str(tmp / "bad.jsonl")}, needle="keep A's reply")
    same = dict(TRIPLETS[0])
    same["B"] = [same["B"][0], same["A"][1]]
    (tmp / "same.jsonl").write_text(json.dumps(same) + "\n", encoding="utf-8")
    refused("same_reply", twin_train(), {"twin_examples": str(tmp / "same.jsonl")}, needle="same reply")
    # mu = 0 does not need N
    no_n = [{k: v for k, v in t.items() if k != "N"} for t in TRIPLETS[:4]]
    (tmp / "no_n.jsonl").write_text("".join(json.dumps(t) + "\n" for t in no_n), encoding="utf-8")
    assert run(spec(tmp, "no_n", twin_train(mu=0.0, steps=2), {"twin_examples": str(tmp / "no_n.jsonl")}))
    refused("no_n_mu", twin_train(steps=2), {"twin_examples": str(tmp / "no_n.jsonl")}, needle="N must be")
    shutil.rmtree(tmp / "no_n", ignore_errors=True)
