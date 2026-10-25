"""Every stored IR document and gate GGUF of paperkiln and ember.cpp passes keramos's validator
(spec/tools/ir_validate.py, IR_SPEC section 10.2): keramos's own documents plus the stores that
tools/ir_corpus.py globs (both repositories' experiments, ML_ARTIFACTS where present). Needs numpy only."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import ir_corpus  # noqa: E402  (puts spec/tools on the path)
import ir_validate as v  # noqa: E402


def test_stored_corpus_passes():
    paths = v.corpus(ir_corpus.patterns())
    assert len(paths) > len(v.corpus())  # paperkiln's experiments hold stored documents
    bad = {p: v.file_errors(p) for p in paths}
    bad = {p: e for p, e in bad.items() if e}
    assert not bad, json.dumps(bad, indent=1)[:3000]
