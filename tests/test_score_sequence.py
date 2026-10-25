"""`mtstudio score` keeps no state between requests (tools/score_sequence_check.py; SCORE_AUDIT_2026-10-26.md in
experiments/tiny_chat_2026-10/): requests of rising, falling and repeated lengths, one past the context, scored in
one process, in reverse order and one per process, agree bit for bit on the CPU build.

Needs a built mtstudio and a trained model: MTSTUDIO_BIN=build/mtstudio SCORE_SEQ_SPEC=run/spec.json (any spec
`mtstudio score` loads; its data.T sets the over-context request). On a device build set SCORE_SEQ_TOL (the
CUDA gate in tools/colab_chat/score_audit_vm.sh passes 1e-4)."""
import json
import os
import random
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.environ.get("MTSTUDIO_BIN", "")
SPEC = os.environ.get("SCORE_SEQ_SPEC", "")

pytestmark = pytest.mark.skipif(not (BIN and SPEC and os.path.exists(BIN) and os.path.exists(SPEC)),
                                reason="set MTSTUDIO_BIN and SCORE_SEQ_SPEC")


def test_a_request_in_a_sequence_equals_it_alone(tmp_path):
    T = int(json.load(open(SPEC, encoding="utf-8"))["data"]["T"])
    rnd = random.Random(17234)
    # Context lengths that rise past every earlier one, fall, repeat, jump, and one over T + 1 ids.
    plan = [(18, 11), (24, 18), (30, 15), (30, 8), (44, 37), (5, 3), (80, 43), (80, 43), (2, 1), (T, 2), (64, 9)]
    rows = []
    for i, (p, c) in enumerate(plan):
        rows.append({"key": f"r{i}", "prompt_ids": [rnd.randrange(3, 900) for _ in range(p)],
                     "continuation_ids": [rnd.randrange(3, 900) for _ in range(c)]})
    inp = tmp_path / "req.jsonl"
    inp.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    tol = os.environ.get("SCORE_SEQ_TOL", "0")
    p = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "score_sequence_check.py"), "--mtstudio", BIN,
                        "--spec", SPEC, "--in", str(inp), "--out", str(tmp_path / "out"), "--tol", tol],
                       capture_output=True, text=True)
    rep = json.load(open(tmp_path / "out" / "report.json", encoding="utf-8"))
    assert p.returncode == 0, (p.stdout, p.stderr, rep)
    seq = {json.loads(x)["key"]: json.loads(x) for x in open(tmp_path / "out" / "seq.out.jsonl") if x.strip()}
    assert seq["r9"]["over_ctx"] and not seq["r0"]["over_ctx"]
    assert rep["seq_vs_alone"]["compared"] == len(rows) and rep["seq_vs_alone"]["max_abs_diff"] <= float(tol)
