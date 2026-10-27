"""tools/publish_hf.py without the network: the token loader reads only the token lines and prints nothing,
sha256 lists are parsed and enforced before anything is staged, a stage holds the data files (hard links where
the filesystem allows), README.md and SHA256SUMS, a hand-written card with a metadata header is used as
written, and a run directory's README names the GGUF's own tokenizer."""
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("publish_hf", ROOT / "tools" / "publish_hf.py")
P = importlib.util.module_from_spec(spec)
sys.modules["publish_hf"] = P
spec.loader.exec_module(P)


def _s(b):
    return struct.pack("<Q", len(b)) + b


def write_gguf(path, kv):
    """A metadata-only GGUF v3 with string, u32 and string-array values."""
    out = struct.pack("<4sIQQ", b"GGUF", 3, 0, len(kv))
    for k, v in kv.items():
        out += _s(k.encode())
        if isinstance(v, str):
            out += struct.pack("<I", 8) + _s(v.encode())
        elif isinstance(v, int):
            out += struct.pack("<II", 4, v)
        else:
            out += struct.pack("<IIQ", 9, 8, len(v)) + b"".join(_s(x.encode()) for x in v)
    Path(path).write_bytes(out)


def test_env_file_loads_only_the_token(tmp_path, monkeypatch, capsys):
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "OTHER_SECRET"):
        monkeypatch.delenv(k, raising=False)
    env = tmp_path / ".env"
    env.write_text("# c\nOTHER_SECRET=nope\nexport HF_TOKEN=\"hf_test_value\"\nBROKEN\n", encoding="utf-8")
    P.load_env_file(str(env))
    assert os.environ["HF_TOKEN"] == "hf_test_value"
    assert "OTHER_SECRET" not in os.environ
    assert capsys.readouterr().out == ""


def test_sha256_list(tmp_path):
    a = "a" * 64
    f = tmp_path / "s.txt"
    f.write_text(f"{a}  dir/x.gguf\n{'b' * 64} *y.bin\n\n# note\n", encoding="utf-8")
    assert P.read_sha256_list(f) == {"x.gguf": a, "y.bin": "b" * 64}
    f.write_text("abc  x\n", encoding="utf-8")
    with pytest.raises(ValueError):
        P.read_sha256_list(f)


def test_gguf_string_kv(tmp_path):
    g = tmp_path / "m.gguf"
    write_gguf(g, {"general.alignment": 32, "tokenizer.ggml.tokens": ["<unk>", "a"],
                   "tokenizer.ggml.model": "gpt2"})
    assert P.gguf_string_kv(g, "tokenizer.ggml.model") == "gpt2"
    assert P.gguf_string_kv(g, "absent") is None
    (tmp_path / "x.bin").write_bytes(b"not a gguf at all, just bytes")
    assert P.gguf_string_kv(tmp_path / "x.bin", "tokenizer.ggml.model") is None


def _release(tmp_path):
    w = tmp_path / "w.gguf"
    w.write_bytes(b"weights")
    s = tmp_path / "spec.json"
    s.write_text("{}", encoding="utf-8")
    card = tmp_path / "card.md"
    card.write_text("---\nlicense: cc-by-nc-4.0\n---\n# m\n", encoding="utf-8")
    sums = tmp_path / "expected.sha256"
    sums.write_text(f"{P.sha256_file(w)}  w.gguf\n{P.sha256_file(s)}  spec.json\n", encoding="utf-8")
    return w, s, card, sums


def test_dry_run_stages_with_hashes(tmp_path, capsys):
    w, s, card, sums = _release(tmp_path)
    stage = tmp_path / "stage"
    rc = P.main(["--repo", "me/m", "--file", str(w), "--file", str(s), "--card", str(card),
                 "--sha256", str(sums), "--stage", str(stage), "--dry-run"])
    assert rc == 0
    assert "nothing uploaded" in capsys.readouterr().out
    assert sorted(p.name for p in stage.iterdir()) == ["README.md", "SHA256SUMS", "spec.json", "w.gguf"]
    assert (stage / "README.md").read_text(encoding="utf-8") == card.read_text(encoding="utf-8")
    assert P.read_sha256_list(stage / "SHA256SUMS") == P.read_sha256_list(sums)
    assert (stage / "w.gguf").read_bytes() == b"weights"
    # Rerunning onto the same stage (a resumed upload) is allowed.
    assert P.main(["--repo", "me/m", "--file", str(w), "--file", str(s), "--card", str(card),
                   "--stage", str(stage), "--dry-run"]) == 0


def test_hash_mismatch_refuses_before_staging(tmp_path, capsys):
    w, s, card, sums = _release(tmp_path)
    sums.write_text(f"{'0' * 64}  w.gguf\n{P.sha256_file(s)}  spec.json\n", encoding="utf-8")
    stage = tmp_path / "stage"
    rc = P.main(["--repo", "me/m", "--file", str(w), "--file", str(s), "--card", str(card),
                 "--sha256", str(sums), "--stage", str(stage), "--dry-run"])
    assert rc == 1
    assert "sha256 mismatch: w.gguf" in capsys.readouterr().out
    assert not stage.exists()


def test_unlisted_file_refuses(tmp_path, capsys):
    w, s, card, sums = _release(tmp_path)
    sums.write_text(f"{P.sha256_file(w)}  w.gguf\n", encoding="utf-8")
    assert P.main(["--repo", "me/m", "--file", str(w), "--file", str(s), "--card", str(card),
                   "--sha256", str(sums), "--dry-run"]) == 1
    assert "(not listed)" in capsys.readouterr().out


def test_stage_refuses_foreign_files(tmp_path):
    w, s, card, _ = _release(tmp_path)
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "other.txt").write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit):
        P.main(["--repo", "me/m", "--file", str(w), "--card", str(card), "--stage", str(stage), "--dry-run"])


def test_card_without_header_gets_one(tmp_path):
    c = tmp_path / "c.md"
    c.write_text("# m\n", encoding="utf-8")
    r = P.card_readme(str(c), "cc-by-nc-4.0")
    assert r.startswith("---\nlicense: cc-by-nc-4.0\n") and r.endswith("# m\n")


def test_run_dir_readme_names_the_tokenizer(tmp_path, capsys):
    run = tmp_path / "run"
    run.mkdir()
    (run / "spec.json").write_text(json.dumps({"name": "m"}), encoding="utf-8")
    (run / "m.safetensors").write_bytes(b"st")
    write_gguf(run / "m.gguf", {"tokenizer.ggml.model": "gpt2"})
    r = P.hub_readme(run, "m", "me/m", "mit", run / "m.gguf")
    assert "word-level" not in r and "tokenizer.ggml.model = gpt2" in r
    write_gguf(run / "m.gguf", {"tokenizer.ggml.model": "word"})
    assert "word-level" in P.hub_readme(run, "m", "me/m", "mit", run / "m.gguf")
    assert P.main([str(run), "--repo", "me/m", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "m.gguf" in out and "m.safetensors" in out
