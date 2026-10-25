"""tools/bench_eval.py's request keys (26 Oct 2026, experiments/tiny_chat_2026-10/SCORE_AUDIT_2026-10-26.md):
every key names one request, and `compare` refuses two runs whose shared keys name different requests.
Offline: a fake tokeniser, no model."""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import bench_eval as be  # noqa: E402


class CharTok:
    """One id per character (and the end-of-text id 0)."""
    eot = 0
    ctx = 512

    def encode(self, s):
        return [ord(ch) for ch in s]


def mc(iid, context, choices, gold=0):
    return {"id": iid, "set": "hellaswag", "kind": "mc", "context": context, "choices": choices, "gold": gold}


def test_repeated_ids_become_distinct_and_first_items_keep_theirs():
    items = [mc("hellaswag/17234", "A man lifts weights. He", [" drops them.", " walks off."]),
             mc("hellaswag/5", "x", [" a", " b"]),
             mc("hellaswag/17234", "Choose jeans.", [" For a dressy look, avoid baggy jeans.", " Fit."])]
    out = be.unique_ids(items)
    assert [it["id"] for it in out] == ["hellaswag/17234", "hellaswag/5", "hellaswag/17234#2"]
    assert out[0] is items[0] and out[2]["context"] == "Choose jeans."
    reqs, _ = be.requests(out, CharTok())
    keys = [k for k, _, _ in reqs]
    assert len(keys) == len(set(keys)) == 6
    # Each item is scored by its own requests: the first item's keys carry its own continuations.
    by = {k: c for k, _, c in reqs}
    assert by["hellaswag/17234|0"] == CharTok().encode(" drops them.")
    assert by["hellaswag/17234#2|0"] == CharTok().encode(" For a dressy look, avoid baggy jeans.")


def test_requests_refuse_a_key_naming_two_requests():
    items = [mc("hellaswag/1", "a", [" b"]), mc("hellaswag/1", "c", [" d"])]
    with pytest.raises(SystemExit, match="name more than one request"):
        be.requests(items, CharTok())


def test_bench_v1_ids_are_unique_after_unique_ids():
    path = os.environ.get("BENCH_ITEMS", "C:/ml_artifacts/benchmarks/bench_v1.jsonl")
    if not os.path.exists(path):
        pytest.skip("bench_v1.jsonl not on this machine (BENCH_ITEMS)")
    raw = be.read_jsonl(path)
    ids = [it["id"] for it in be.unique_ids(raw)]
    assert len(ids) == len(set(ids)) == len(raw)
    assert len(raw) - len({it["id"] for it in raw}) == 433  # HellaSwag validation's repeated `ind`


def write_run(d, results):
    os.makedirs(os.path.join(d, "score"))
    with open(os.path.join(d, "score", "chunk00000.json"), "w", encoding="utf-8") as f:
        json.dump({"sig": "x", "results": results}, f)


def row(lps):
    return {"over_ctx": False, "logprobs": lps, "argmax": [1] * len(lps), "sum": sum(lps)}


def test_compare_refuses_shared_keys_naming_different_requests(tmp_path, capsys):
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    write_run(a, {"h/1|0": row([-1.0, -2.0]), "h/1|1": row([-3.0])})
    write_run(b, {"h/1|0": row([-1.0, -2.0, -0.5, -4.0]), "h/1|1": row([-3.0])})
    with pytest.raises(SystemExit, match="different lengths"):
        be.main(["compare", a, b])
    # The same requests compare as before.
    c = str(tmp_path / "c")
    write_run(c, {"h/1|0": row([-1.0, -2.0 + 1e-6]), "h/1|1": row([-3.0])})
    be.main(["compare", a, c])
    out = json.loads(capsys.readouterr().out)
    assert out["requests"] == 2 and out["max_abs_logprob_diff"] < 1e-5 and out["choice_agree"] == "1/1"
