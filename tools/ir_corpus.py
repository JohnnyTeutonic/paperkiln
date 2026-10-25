"""Every stored IR document and gate GGUF of paperkiln and ember.cpp through keramos's validator
(spec/tools/ir_validate.py; IR_SPEC section 10.2): keramos's own documents, the experiment receipts of
both repositories, and ML_ARTIFACTS (default C:/ml_artifacts: documents and the *gate* GGUFs).

    python tools/ir_corpus.py [--quiet]
"""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "spec" / "tools"))
import ir_validate  # noqa: E402


def patterns():
    art = os.environ.get("ML_ARTIFACTS", "C:/ml_artifacts" if os.name == "nt" else "/mnt/c/ml_artifacts")
    return [str(ROOT / "experiments" / "**" / "*.ir.json"),
            str(ROOT.parent / "ember.cpp" / "experiments" / "**" / "*.ir.json"),
            art + "/**/*.ir.json", art + "/*gate*/**/*.gguf"]


def main(argv):
    args = ["--corpus"]
    for p in patterns():
        args += ["--corpus-glob", p]
    return ir_validate.main(args + list(argv))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
