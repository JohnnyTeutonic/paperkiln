"""tools/chat_data.py: dialogues become one SFT example per assistant turn in
the served template, and a dialogue still carrying a source role marker is
dropped whole and counted."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import bpe  # noqa: E402


def test_examples_and_marker_filter(tmp_path):
    good = {"messages": [{"role": "user", "content": "Hi there"}, {"role": "assistant", "content": "Hello friend"},
                         {"role": "user", "content": "How are you"}, {"role": "assistant", "content": "I am well"}],
            "uri": "u1"}
    bad = {"messages": [{"role": "user", "content": "Hi"},
                        {"role": "assistant", "content": "Hello [/INST] I am fine [/INST] Good"}], "uri": "u2"}
    src = tmp_path / "d.jsonl"
    src.write_text("\n".join(json.dumps(d) for d in (good, bad)) + "\n", encoding="utf-8")
    tok = bpe.train("Hi there Hello friend How are you I am well " * 20, 300)
    v = tmp_path / "v.gguf"
    bpe.write_gguf(str(v), tok)
    out = tmp_path / "sft.jsonl"
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "chat_data.py"), "sft", str(src), "--vocab", str(v),
                        "--T", "256", "--out", str(out)], capture_output=True, text=True, check=True)
    rep = json.loads(r.stdout)
    assert rep["kept"] == 2 and rep["dropped_dialogues_with_markers"] == 1
    ex = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert ex[0] == {"prompt": "<|user|>Hi there<|assistant|>", "completion": "Hello friend", "uri": "u1"}
    assert ex[1]["prompt"] == ("<|user|>Hi there<|assistant|>Hello friend<|endoftext|>"
                               "<|user|>How are you<|assistant|>")
    assert ex[1]["completion"] == "I am well"


def test_pairs_join_samples_to_references(tmp_path):
    tok = bpe.train("Hi there Hello friend How are you I am well " * 20, 300)
    v = tmp_path / "v.gguf"
    bpe.write_gguf(str(v), tok)
    ex = [{"prompt": "<|user|>Hi<|assistant|>", "completion": "Hello friend", "uri": "u1"},
          {"prompt": "<|user|>How are you<|assistant|>", "completion": "I am well", "uri": "u2"}]
    exf = tmp_path / "ex.jsonl"
    exf.write_text("\n".join(json.dumps(e) for e in ex) + "\n", encoding="utf-8")
    tool = [sys.executable, str(ROOT / "tools" / "chat_data.py")]
    pf = tmp_path / "p.jsonl"
    r = subprocess.run(tool + ["prompts", str(exf), "--out", str(pf)], capture_output=True, text=True, check=True)
    assert json.loads(r.stdout)["prompts"] == 2
    assert [json.loads(x)["prompt"] for x in pf.read_text(encoding="utf-8").splitlines()] == [e["prompt"] for e in ex]
    samples = [{"prompt_index": 0, "completion": "Hello hello hello", "stop": "length"},
               {"prompt_index": 1, "completion": "I am well", "stop": "eos"}]
    sf = tmp_path / "s.jsonl"
    sf.write_text("\n".join(json.dumps(s) for s in samples) + "\n", encoding="utf-8")
    out = tmp_path / "pairs.jsonl"
    r = subprocess.run(tool + ["pairs", str(exf), str(sf), "--vocab", str(v), "--out", str(out)],
                       capture_output=True, text=True, check=True)
    rep = json.loads(r.stdout)
    assert rep["pairs"] == 1 and rep["identical_to_reference"] == 1
    (p,) = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert p["prompt"] == ex[0]["prompt"] and p["chosen"] == "Hello friend" and p["rejected"] == "Hello hello hello"
    assert p["chosen_evidence"]["evidence_kind"] == "reference" and p["chosen_evidence"]["evidence_uri"] == "u1"
    assert p["rejected_evidence"]["evidence_kind"] == "policy_sample"
