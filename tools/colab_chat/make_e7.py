#!/usr/bin/env python3
"""E7 (shape against corpus): one pretraining arm, E7A, and its rung spec.

    python tools/colab_chat/make_e7.py            write e7_E7A.json and base2_rung_A.json beside this file
    python tools/colab_chat/make_e7.py --check    exit 1 unless both files equal what this builder writes

E7A is the token ladder's arm N1B (make_ladder.arm, unchanged) with ONE thing changed: the model's shape. Its training
job's `arch.custom` is base A's (8 layers, d 768, 12 heads, d_ff 2,240: tc_b1_r1's spec, and base 2's arm C's) in place of
base 2's D1 (18 x 512, 8 heads, d_ff 1,536); its rung jobs read the model through base2_rung_A.json, which is
base2_rung_D1.json with the same arch swapped; and its names (job, model, store branch, rung cells, output folder) carry
E7A in place of N1B. Everything else is N1B's: corpus N (main and tail files, pins entry arm-N), the 60,000 steps,
batch 8 with 2 accumulation steps of 1,024 positions, peak rate 5e-4, warm-up 4,000 steps, cosine to a tenth, the tail
from 85%, bf16 with the flash kernel, seed 1, the marks (10,000, 58,500, 60,000), panels, Carry-5 items, the store's
caps and the release turn 0. tests/test_e7_jobs.py holds every leaf equal to N1B's but for these substitutions.

Nothing here is admitted: E7's launch gate is its registration, two reviews and Jonathan's dated line, through
e7_launch.py after the token ladder has ended.

Claude Opus 5.5 (a fork of the coordinating session), 19 Nov 2026.
"""
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_base2 as M   # noqa: E402  (base 2's builder, unchanged)
import make_ladder as L  # noqa: E402  (the token ladder's builder, unchanged)

NAME, REF = "E7A", "N1B"
# Base A's shape: tc_b1_r1's spec (C:/ml_artifacts/tiny_chat/b1_r1/pt_b1_r1/pt_b1_r1_spec.json) and base2_C.json's
A_CUSTOM = {"d": 768, "layers": 8, "heads": 12, "d_ff": 2240}
A_PARAMS = 72761088          # base2_C.json's _base2.params for this shape
D1_CUSTOM = {"d": 512, "layers": 18, "heads": 8, "d_ff": 1536}
RUNG_D1, RUNG_A = "tools/colab_chat/base2_rung_D1.json", "tools/colab_chat/base2_rung_A.json"


def substitute(x):
    if isinstance(x, dict):
        return {k: substitute(v) for k, v in x.items()}
    if isinstance(x, list):
        return [substitute(v) for v in x]
    if isinstance(x, str):
        return RUNG_A if x == RUNG_D1 else x.replace(REF, NAME)
    return x


def rung_spec():
    src = json.load(open(os.path.join(HERE, "base2_rung_D1.json"), encoding="utf-8"))
    assert src["arch"]["custom"] == D1_CUSTOM, src["arch"]
    out = copy.deepcopy(src)
    out["name"] = "base2_rung_A"
    out["arch"]["custom"] = dict(A_CUSTOM)
    out["_shape"] = {"note": src["_shape"]["note"], "params": A_PARAMS}
    return out


def arm():
    src = L.arm(REF)
    meta = src.pop("_base2")
    src.pop("_comment")
    out = substitute(src)
    pt = [j for j in out["jobs"] if j["name"] == f"pt_base2_{NAME}"]
    assert len(pt) == 1, [j["name"] for j in out["jobs"]]
    assert pt[0]["spec"]["arch"]["custom"] == D1_CUSTOM, pt[0]["spec"]["arch"]
    pt[0]["spec"]["arch"]["custom"] = dict(A_CUSTOM)
    meta = dict(meta, arm=NAME, shape="A", params=A_PARAMS, slot=None, ladder=False, e7_ref_arm=REF,
                tokens_per_parameter=round(meta["positions"] / A_PARAMS, 2))
    head = {"_comment": (f"E7, arm {NAME}: the token ladder's arm {REF} with base A's shape (8 x 768) in place of D1; "
                         "nothing else is changed. Not admitted: E7's launch gate is its registration, two reviews and "
                         "Jonathan's dated line."),
            "_base2": meta}
    head.update(out)
    return head


def files():
    return {f"e7_{NAME}.json": arm(), "base2_rung_A.json": rung_spec()}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    check = "--check" in argv
    bad = []
    for name, obj in files().items():
        path = os.path.join(HERE, name)
        if check:
            try:
                same = open(path, encoding="utf-8").read() == M.text(obj)
            except OSError:
                same = False
            if not same:
                bad.append(name)
        else:
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(M.text(obj))
    if check:
        print(json.dumps({"check": "equal" if not bad else "different", "files": sorted(files()), "different": bad}))
        return 1 if bad else 0
    print(json.dumps({"written": sorted(files())}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
