"""G2 (LOCAL_LLM_STACK.md section 3, compatibility gates): mtstudio's SFT rows from conversation
rows ({"messages"}, rendered by microtorch/chat_template.hpp under train.sft_loss) are bit-identical
to the rendered rows they replace: "all" to chat_data.py sft --all-turns' segments rows, "each" to
its per-turn {prompt, completion} rows, with and without per-turn state tokens (k 4), at two T
(one that cuts dialogues, one that does not), under both paperkiln vocabularies. "final" keeps
"all"'s inputs and trains the last kept reply only. Also the export: a BPE model's GGUF declares
paperkiln/1 (record and Jinja) and resolves to it, with a Modelfile beside it.

Needs MTSTUDIO_BIN (a built mtstudio). Pinned dialogues: corpus/data/human_dailydialog_v1 and
human_oasst_v1; vocabularies as tests/test_chat_template.py finds them.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))
import chat_template as CT  # noqa: E402
from test_chat_template import VOCAB8, VOCAB16  # noqa: E402

BIN = os.environ.get("MTSTUDIO_BIN")
pytestmark = pytest.mark.skipif(not BIN, reason="set MTSTUDIO_BIN to a built mtstudio")
N_PER_CORPUS = int(os.environ.get("PK_G2_N", "300"))


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def dialogues():
    import chat_data
    out = []
    for cid in ("human_dailydialog_v1", "human_oasst_v1"):
        p = ROOT / "corpus" / "data" / cid / "dialogues.jsonl"
        if not p.exists():
            continue
        rows = [d for d in read_jsonl(p) if not any(chat_data.MARKER.search(m["content"]) for m in d["messages"])]
        out += rows[:N_PER_CORPUS]
    return out


def vocabs():
    return [v for v in (VOCAB8, VOCAB16) if v.exists()]


def dump(tmp, name, vocab, T, rows, state_k=0, turns=None):
    data = tmp / f"{name}.jsonl"
    data.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    arch = {"preset": "llama-tiny"}
    if state_k:
        arch["state_tokens"] = {"k": state_k, "mask": "turn_state"}
    train = {"steps": 1, "objective": "sft", "seed": 1}
    if turns:
        train["sft_loss"] = {"turns": turns}
    spec = {"name": name, "arch": arch, "data": {"vocab": str(vocab), "vocab_cap": 0, "T": T, "sft_examples": str(data)},
            "train": train, "out_dir": str(tmp / ("run_" + name))}
    sp = tmp / f"{name}.spec.json"
    sp.write_text(json.dumps(spec), encoding="utf-8")
    out = tmp / f"{name}.rows.jsonl"
    r = subprocess.run([BIN, "sft-dump", str(sp), "--out", str(out)], capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stderr
    return read_jsonl(out), json.loads(r.stdout.strip().splitlines()[-1])


def legacy(tmp, vocab, T, all_turns):
    src = tmp / "dialogues.jsonl"
    out = tmp / f"legacy_{'all' if all_turns else 'each'}_{T}.jsonl"
    cmd = [sys.executable, str(ROOT / "tools" / "chat_data.py"), "sft", str(src), "--vocab", str(vocab), "--T", str(T),
           "--out", str(out)] + (["--all-turns"] if all_turns else [])
    rep = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
    assert rep["dropped_dialogues_with_markers"] == 0
    return [{k: v for k, v in r.items() if k != "uri"} for r in read_jsonl(out)], rep


@pytest.mark.parametrize("vi", [0, 1])
def test_g2_conversation_rows_equal_rendered_rows(tmp_path, vi):
    vs = vocabs()
    if vi >= len(vs):
        pytest.skip("vocabulary not on disk")
    vocab = vs[vi]
    ds = dialogues()
    if not ds:
        pytest.skip("no pinned dialogues on disk")
    (tmp_path / "dialogues.jsonl").write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in ds),
                                             encoding="utf-8")
    conv = [{"messages": d["messages"], "uri": d.get("uri", "")} for d in ds]
    report = {}
    for T in (96, 512):
        seg, seg_rep = legacy(tmp_path, vocab, T, True)
        pc, pc_rep = legacy(tmp_path, vocab, T, False)
        for k in (0, 4):
            a_seg, _ = dump(tmp_path, f"seg{T}k{k}", vocab, T, seg, k)
            a_conv, s_conv = dump(tmp_path, f"conv{T}k{k}", vocab, T, conv, k, "all")
            assert a_conv == a_seg, (T, k)
            e_pc, _ = dump(tmp_path, f"pc{T}k{k}", vocab, T, pc, k)
            e_conv, _ = dump(tmp_path, f"each{T}k{k}", vocab, T, conv, k, "each")
            assert e_conv == e_pc, (T, k)
            f_conv, _ = dump(tmp_path, f"final{T}k{k}", vocab, T, conv, k, "final")
            # "final" (no state tokens: a state cut can remove a final reply and drop its row): the
            # same rows and inputs as "all", trained on the last trained span of each.
            if k == 0:
                assert len(f_conv) == len(a_conv) and [r["x"] for r in f_conv] == [r["x"] for r in a_conv]
            for ra, rf in zip(a_conv, f_conv) if k == 0 else ():
                m = ra["m"]
                last_end = max(i for i, v in enumerate(m) if v) + 1
                start = last_end
                while start > 0 and m[start - 1]:
                    start -= 1
                assert rf["m"] == [1 if start <= i < last_end else 0 for i in range(last_end)]
            report[f"T{T}_k{k}"] = {"all_rows": len(a_conv), "each_rows": len(e_conv),
                                    "dropped_over_T": s_conv["dropped_over_T"],
                                    "exchanges_cut": s_conv["exchanges_cut_over_T"]}
        assert s_conv["conversations"] == len(conv)
        assert seg_rep["kept"] + seg_rep["dropped_over_T"] == len(conv)
    print(json.dumps({"vocab": vocab.name, "dialogues": len(conv), **report}))
    assert report["T96_k0"]["exchanges_cut"] > 0  # the short context exercises the cut


def test_sft_loss_refuses_rendered_rows_and_bad_conversations(tmp_path):
    vs = vocabs()
    if not vs:
        pytest.skip("no vocabulary on disk")
    rows = [CT.to_segments(CT.from_pairs([("hi", "hello")]))] * 3
    data = tmp_path / "seg.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    spec = {"name": "x", "arch": {"preset": "llama-tiny"}, "data": {"vocab": str(vs[0]), "vocab_cap": 0, "T": 64,
                                                                   "sft_examples": str(data)},
            "train": {"steps": 1, "objective": "sft", "sft_loss": {"turns": "all"}}, "out_dir": str(tmp_path / "r")}
    (tmp_path / "s.json").write_text(json.dumps(spec), encoding="utf-8")
    r = subprocess.run([BIN, "sft-dump", str(tmp_path / "s.json"), "--out", str(tmp_path / "o.jsonl")],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "rendered one" in r.stderr
    bad = [{"messages": [{"role": "user", "content": "say <|user|>"}, {"role": "assistant", "content": "no"}]}] * 3
    data.write_text("".join(json.dumps(x) + "\n" for x in bad), encoding="utf-8")
    spec["train"].pop("sft_loss")
    (tmp_path / "s.json").write_text(json.dumps(spec), encoding="utf-8")
    r = subprocess.run([BIN, "sft-dump", str(tmp_path / "s.json"), "--out", str(tmp_path / "o.jsonl")],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "reserved text" in r.stderr
    spec["train"]["sft_loss"] = {"turns": "last"}
    (tmp_path / "s.json").write_text(json.dumps(spec), encoding="utf-8")
    r = subprocess.run([BIN, "sft-dump", str(tmp_path / "s.json"), "--out", str(tmp_path / "o.jsonl")],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "all, final or each" in r.stderr


def test_system_message_and_loss_override_rows(tmp_path):
    """A system message folds into the first user turn (never trained); a reply with "loss": false
    stays as context only."""
    vs = vocabs()
    if not vs:
        pytest.skip("no vocabulary on disk")
    import llama3_bpe
    import bpe
    v = vs[-1]
    enc = (lambda s: llama3_bpe.from_gguf(str(v)).encode(s, add_special_tokens=False).ids) if v == VOCAB16 else \
        bpe.read_gguf(str(v)).encode
    m = [{"role": "system", "content": "Be brief."}] + CT.from_pairs([("Hi", "Hello."), ("Name?", "Kiln.")])
    m[2]["loss"] = False
    rows, st = dump(tmp_path, "sys", v, 64, [{"messages": m}] * 3, 0, "all")
    want = enc(CT.render(m))
    r = rows[0]
    assert r["x"] + [r["y"][-1]] == want[:len(r["x"]) + 1]
    trained = [t for t, k in zip(r["y"], r["m"]) if k]
    assert trained == enc("Kiln.") + [1]
    assert st["rows"] == 3


def test_export_declares_the_template(tmp_path):
    """A BPE model's F32 GGUF carries paperkiln.chat_template (the record) and tokenizer.chat_template
    (its Jinja), resolves to paperkiln/1 as declared, and gets a Modelfile beside it."""
    vs = vocabs()
    if not vs:
        pytest.skip("no vocabulary on disk")
    import llama3_bpe
    data = tmp_path / "ex.jsonl"
    data.write_text("".join(json.dumps({"messages": CT.from_pairs([("hi", "hello")])}) + "\n" for _ in range(4)),
                    encoding="utf-8")
    spec = {"name": "decl", "arch": {"preset": "llama-tiny"}, "data": {"vocab": str(vs[0]), "vocab_cap": 0, "T": 32,
                                                                      "sft_examples": str(data)},
            "train": {"steps": 2, "objective": "sft", "eval_every": 1000, "checkpoint_every": 1000,
                      "gradmap_every": 0, "batch": 2},
            "export": {"formats": ["safetensors", "gguf"]}, "out_dir": str(tmp_path / "run")}
    (tmp_path / "s.json").write_text(json.dumps(spec), encoding="utf-8")
    r = subprocess.run([BIN, "run", str(tmp_path / "s.json")], capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stderr[-2000:]
    g = tmp_path / "run" / "decl.gguf"
    meta = llama3_bpe.read_gguf_kv(str(g), {"paperkiln.chat_template", "tokenizer.chat_template",
                                            "tokenizer.ggml.tokens", "tokenizer.ggml.model"})
    rec = json.loads(meta["paperkiln.chat_template"])
    assert CT.rid(rec) == CT.V1 and rec["system"]["mode"] == CT.SYSTEM_MODE_V1
    assert meta["tokenizer.chat_template"] == CT.jinja(CT.V1)
    got, why = CT.resolve(meta)
    assert CT.rid(got) == CT.V1 and why == "declared"
    assert (tmp_path / "run" / "decl.Modelfile").read_text(encoding="utf-8") == CT.modelfile("decl.gguf")
