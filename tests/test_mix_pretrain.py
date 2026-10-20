"""corpus/mix_pretrain.py: document indexing across chunk boundaries, ratios per part, no document
used twice, document-level interleaving, the separate tail file, determinism and the ledger."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "corpus"))
import mix_pretrain as M  # noqa: E402

EOT = "<|endoftext|>\n"


def _source(path, name, n, size):
    docs = [f"{name}-{i:04d} " + "x" * size for i in range(n)]
    path.write_bytes("".join(d + EOT for d in docs).encode("utf-8"))   # no newline translation
    return docs


def _docs(path):
    return [d for d in path.read_bytes().decode("utf-8").split(EOT) if d]


def test_index_handles_separators_split_across_chunks(tmp_path):
    p = tmp_path / "s.txt"
    docs = _source(p, "a", 50, 7)
    starts, ends = M.index_documents(str(p), chunk=5)   # smaller than the separator
    assert len(starts) == 50 and ends[-1] == p.stat().st_size
    raw = p.read_bytes()
    assert [raw[b:e].decode() for b, e in zip(starts, ends)] == [d + EOT for d in docs]


def test_mix_ratios_parts_and_no_reuse(tmp_path):
    fw, dl = tmp_path / "fw.txt", tmp_path / "dl.txt"
    _source(fw, "fw", 2000, 40)
    _source(dl, "dl", 2000, 40)
    names = {"fw": str(fw), "dl": str(dl)}
    main = M.parse_ratios("fw=0.7,dl=0.3", names)
    tail = M.parse_ratios("fw=0.5,dl=0.5", names)
    led = M.mix(str(tmp_path / "mix"), names, main, tail, 0.2, 100_000, seed=3)
    m, t = _docs(tmp_path / "mix.main.txt"), _docs(tmp_path / "mix.tail.txt")
    assert len(set(m + t)) == len(m) + len(t)                       # no document twice
    share = lambda docs, s: sum(len(d) for d in docs if d.startswith(s)) / sum(len(d) for d in docs)  # noqa: E731
    assert abs(share(m, "fw") - 0.7) < 0.02 and abs(share(t, "fw") - 0.5) < 0.03
    runs = sum(1 for x, y in zip(m, m[1:]) if x[:2] != y[:2])
    assert runs > len(m) / 4                                        # interleaved, not blocks
    assert led["parts"]["tail"]["fraction"] == 0.2 and led["sources"]["fw"]["documents"] == 2000
    led2 = M.mix(str(tmp_path / "mix2"), names, main, tail, 0.2, 100_000, seed=3)
    assert led2["parts"]["main"]["sha256"] == led["parts"]["main"]["sha256"]
    assert json.load(open(tmp_path / "mix.ledger.json"))["seed"] == 3


def test_max_total_and_shortfall(tmp_path):
    fw, dl = tmp_path / "fw.txt", tmp_path / "dl.txt"
    _source(fw, "fw", 500, 40)
    _source(dl, "dl", 50, 40)
    names = {"fw": str(fw), "dl": str(dl)}
    r = M.parse_ratios("fw=0.5,dl=0.5", names)
    led = M.mix(str(tmp_path / "m"), names, r, r, 0.1, "max", seed=1)
    assert led["total_bytes_target"] <= 2 * led["sources"]["dl"]["bytes"]
    try:
        M.mix(str(tmp_path / "m2"), names, r, r, 0.1, 10_000_000, seed=1)
    except SystemExit as e:
        assert "out of documents" in str(e)
    else:
        raise AssertionError("expected a shortfall")
