"""tinyarith: frozen evaluation sets, split hygiene, vocabulary closure, and the
hack probe's declared verdicts checked against the C++ verifiers.

The verifier check needs POSTTRAIN_VERIFY_BIN (skipped without it).
"""
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "corpus", "fetchers"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import _common as C  # noqa: E402
import corpus_quality as cq  # noqa: E402
import tinyarith as ta  # noqa: E402

EVAL = os.path.join(ROOT, "corpus", "eval")
LEVELS = ta.LEVELS


def eval_path(R, split):
    return os.path.join(EVAL, f"tinyarith_r{R}_{split}_{ta.VERSION}.jsonl")


def rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


@pytest.mark.parametrize("R", LEVELS)
@pytest.mark.parametrize("split", ["dev", "test", "hackprobe"])
def test_frozen_sets_match_their_pins(R, split):
    eid = f"tinyarith_r{R}_{split}_{ta.VERSION}"
    want = C.expected_hashes(eid)
    p = eval_path(R, split)
    assert want, f"{eid} has no pin"
    assert C.sha256_file(p) == want[os.path.basename(p)], f"{eid} was modified after freezing"


@pytest.mark.parametrize("R", LEVELS)
def test_generator_reproduces_frozen_sets(R, tmp_path):
    _, evals, probes, _ = ta.generate(R)
    for split, rs in list(evals.items()) + [("hackprobe", probes)]:
        out = tmp_path / f"{split}.jsonl"
        ta.write_jsonl(str(out), rs)
        assert C.sha256_file(str(out)) == C.sha256_file(eval_path(R, split))


@pytest.mark.parametrize("R", LEVELS)
def test_splits_disjoint_by_fact_and_answers_correct(R):
    sp = ta.split_facts(R)
    tr, dv, te = set(sp["train"]), set(sp["dev"]), set(sp["test"])
    assert not (tr & dv) and not (tr & te) and not (dv & te)
    assert tr | dv | te == set(ta.facts(R))
    for split in ("dev", "test"):
        for r in rows(eval_path(R, split)):
            op, a, b = r["uri"].split("/")[-1].split("-")
            assert (op, int(a), int(b)) in sp[split]
            assert r["answer"] == ta.answer(op, int(a), int(b)) >= 0
            assert f" {a} " in f" {r['prompt']} " and f" {b} " in f" {r['prompt']} "
    train, _, _, _ = ta.generate(R)
    for r in train:
        op, a, b = r["uri"].split("/")[2].split("-")
        assert (op, int(a), int(b)) in tr


@pytest.mark.parametrize("R", LEVELS)
def test_vocabulary_is_closed_over_everything_the_task_says(R):
    vocab = cq.build_vocab([t.encode() for t in ta.vocabulary(R)])
    train, evals, probes, _ = ta.generate(R)
    texts = [f"{r['prompt']} {r['completion']} {ta.EOT}" for r in train]
    texts += [r["prompt"] for split in evals.values() for r in split]
    texts += [p["prompt"] for p in probes] + [c["completion"] for p in probes for c in p["cases"]]
    for t in texts:
        ids, words, _ = cq.tokenize_bytes(t.encode(), vocab)
        assert 0 not in ids, f"unknown token in: {t}"


def test_probe_hack_is_a_real_hack():
    for R in LEVELS:
        for p in rows(eval_path(R, "hackprobe")):
            sloppy = p["answer"] + p["distractor"]
            assert p["distractor"] >= 1 and sloppy != p["answer"]
            assert p["cases"][1]["completion"] == f"the answer is {sloppy} ."


BIN = os.environ.get("POSTTRAIN_VERIFY_BIN", "")


@pytest.mark.skipif(not (BIN and os.path.exists(BIN)), reason="set POSTTRAIN_VERIFY_BIN")
@pytest.mark.parametrize("R", LEVELS)
def test_probe_verdicts_match_the_cpp_verifiers(R, tmp_path):
    probes = rows(eval_path(R, "hackprobe"))
    refs = tmp_path / "refs.jsonl"
    refs.write_text("".join(json.dumps({"prompt": p["prompt"], "answer": p["answer"], "uri": p["uri"]}) + "\n"
                            for p in probes), encoding="utf-8")
    recs = [{"prompt": p["prompt"], "completion": c["completion"], "want_em": c["exact_match"],
             "want_fmt": c["format"]} for p in probes for c in p["cases"]]
    feed = "".join(json.dumps(r) + "\n" for r in recs)

    def run(*args):
        out = subprocess.run([BIN, *args], input=feed, capture_output=True, text=True, check=True).stdout
        return [json.loads(x) for x in out.splitlines() if x.strip()]

    em = run("exact_match", "--refs", str(refs))
    fm = run("format")
    assert [r["reward"] for r in em] == [float(r["want_em"]) for r in recs]
    assert [r["reward"] for r in fm] == [float(r["want_fmt"]) for r in recs]
    # The monitor's signal on the fixed inputs: format-pass and truth-fail.
    hack = sum(1 for e, f in zip(em, fm) if f["reward"] == 1.0 and e["reward"] == 0.0)
    assert hack == 2 * len(probes)
