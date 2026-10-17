"""mtstudio on a byte-level BPE vocabulary (tools/bpe.py): training reads the
corpus through the C++ encoder with the same ids as Python, sampling decodes
bytes back to text, and a BPE vocabulary is never capped below its size.
Needs MTSTUDIO_BIN (or ./build/mtstudio); skipped otherwise.
"""
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))
import bpe  # noqa: E402
from test_mtstudio_chat import BIN  # noqa: E402

needs_bin = pytest.mark.skipif(BIN is None, reason="set MTSTUDIO_BIN to a built mtstudio")

CORPUS = ("Once upon a time, a little girl named Lily found a shiny red ball in the park. "
          "She played with it all day and was very happy.<|endoftext|>\n"
          "Tom liked to read books about the sea. One day, he saw a big whale!<|endoftext|>\n") * 60


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("bpe")
    (tmp / "corpus.txt").write_text(CORPUS, encoding="utf-8")
    tok = bpe.train(CORPUS, 600)
    bpe.write_gguf(str(tmp / "v.gguf"), tok)
    return tmp, tok


def spec(tmp, name, cap, steps=40):
    s = {"name": name, "arch": {"preset": "llama-tiny"},
         "data": {"corpus": str(tmp / "corpus.txt"), "vocab": str(tmp / "v.gguf"), "vocab_cap": cap, "T": 32},
         "train": {"steps": steps, "lr": 0.003, "batch": 4, "eval_every": 20, "checkpoint_every": 1000,
                   "gradmap_every": 0, "seed": 1},
         "export": {"formats": ["safetensors"]}, "out_dir": str(tmp / name)}
    p = tmp / f"{name}.json"
    p.write_text(json.dumps(s), encoding="utf-8")
    return p


@needs_bin
def test_trains_on_bpe_ids_identical_to_python(setup):
    tmp, tok = setup
    r = subprocess.run([BIN, "run", str(spec(tmp, "m", cap=4096))], capture_output=True, text=True, timeout=1200)
    assert r.returncode == 0, r.stderr
    ev = [json.loads(x) for x in (tmp / "m" / "events.jsonl").read_text().splitlines() if x.strip()]
    data = next(e for e in ev if e["event"] == "data")
    assert data["vocab"] == len(tok.tokens)
    assert data["tokens"] == len(tok.encode(CORPUS))       # the C++ encoder inside the trainer
    losses = [e["loss"] for e in ev if e["event"] == "step"]
    assert sum(losses[-10:]) / 10 < sum(losses[:10]) / 10
    # Sampling decodes bytes, not space-joined pieces.
    prompts = tmp / "p.jsonl"
    prompts.write_text(json.dumps({"prompt": "Once upon a time,"}) + "\n", encoding="utf-8")
    out = tmp / "s.jsonl"
    r = subprocess.run([BIN, "sample-set", str(tmp / "m.json"), "--prompts", str(prompts), "--out", str(out),
                        "--n", "4", "--temp", "0.7", "--seed", "3", "--max-new", "24"],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    for rec in (json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()):
        assert rec["completion"] == tok.decode(rec["completion_ids"])
        assert rec["prompt_ids"] == tok.encode("Once upon a time,")


@needs_bin
def test_bpe_vocab_is_never_capped(setup):
    tmp, _ = setup
    r = subprocess.run([BIN, "run", str(spec(tmp, "capped", cap=300, steps=2))], capture_output=True, text=True,
                       timeout=300)
    assert r.returncode != 0 and "vocab_cap" in r.stderr
