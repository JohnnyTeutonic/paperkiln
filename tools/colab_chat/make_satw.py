#!/usr/bin/env python3
"""The jobs file of SATW, SAT with the recency-weighted pointer loss (the author's research tree, the SATW registration
beside AI_ML/chat_screens_claude/base2/token_ladder/symbols_as_targets/). Claude Opus 5.5, 26 Nov 2026.

    python3 tools/colab_chat/make_satw.py            # writes satw_SATW.json
    python3 tools/colab_chat/make_satw.py --check    # compares it with what is on disk; non-zero on any difference

SATW is SAT (make_sat.py, imported and not edited) with one added key, train.symbol_targets.recency_weight
{"nearest_right": W} (tools/mtstudio.cpp): a natural pointer target whose nearest earlier mention is its gold weighs W
in the pointer loss, every other one 1. Everything else is SAT's: the four SAT files and their pins, sat_pin, sat_check
(run on SATW's own spec, so the check exercises the weighted loss), the heads, schedule and cut, N1B's shape, seed,
windows, marks, panels and the store's switches. The names are SATW's (pt_base2_SATW, ckpt-SATW, rung_SATW_*).

The release turn is 40 minutes (the ladder's slot 3): on the store's 60-minute grid SAT releases at 20 (as SYMW) and
W169 at 0, so SATW shares a turn with neither.

RECENCY_WEIGHT is the registration's W, 0.3077: the equal-mass weight, the natural pointer targets the recency rule
gets wrong over those it gets right in the replayed windows (the label build's receipt, met_by_windows:
2,982,760 / 9,692,780 = 0.30773), so nearest-right and nearest-wrong targets carry about equal total weight.
"""
import copy
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_base2 as M  # noqa: E402  (base 2's builder, unchanged)
import make_ladder as L  # noqa: E402  (the token ladder's builder, unchanged)
import make_sat as S  # noqa: E402  (SAT's builder, unchanged)

NAME = "SATW"
OFFSET_MIN = 40
OUT_FILE = "satw_SATW.json"
# ---- the weight of a natural nearest-right pointer target, as the registration fixes it. ----
RECENCY_WEIGHT = 0.3077
# --------------------------------------------------------------------------------------------------


def rename(obj):
    """SAT's names as SATW's: every SAT not followed by a letter (pt_base2_SAT, ckpt-SAT, rung_SAT_*, BASE2_SAT)."""
    return json.loads(re.sub(r"SAT(?![A-Za-z])", NAME, json.dumps(obj)))


def symbol_targets():
    return {**S.symbol_targets(), "recency_weight": {"nearest_right": RECENCY_WEIGHT}}


def arm():
    sat = S.arm()
    jobs = []
    for j in sat["jobs"]:
        j = copy.deepcopy(j)
        if j["name"] == "pt_base2_SAT":
            j["spec"]["train"]["symbol_targets"] = symbol_targets()
        jobs.append(rename(j) if j["name"] != "sat_check" else j)
    train = next(j for j in jobs if j["name"] == f"pt_base2_{NAME}")
    # sat_check runs SATW's own training spec (renamed, with the weight).
    i = next(k for k, j in enumerate(jobs) if j["name"] == "sat_check")
    chk = S.check_job(train["spec"])
    argv = chk["cmd"]
    spec = json.loads(argv[argv.index("--spec-json") + 1])
    spec["name"] = f"base2_{NAME}_check"
    argv[argv.index("--spec-json") + 1] = json.dumps(spec, sort_keys=True)
    jobs[i] = chk
    out = {k: copy.deepcopy(v) for k, v in sat.items() if k != "jobs"}
    out["jobs"] = jobs
    b = out["_base2"]
    b.update(arm=NAME, symbol_targets={**b["symbol_targets"], "recency_weight": {"nearest_right": RECENCY_WEIGHT}})
    out["_comment"] = (f"SATW: SAT with train.symbol_targets.recency_weight (natural pointer targets the recency rule "
                       f"gets right weigh {RECENCY_WEIGHT} in the pointer loss); nothing else is changed. Not admitted: "
                       "the launch gate is two reviews and the lead's ruling.")
    return L._with_offset(out, OFFSET_MIN)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    path = os.path.join(HERE, OUT_FILE)
    text = M.text(arm())
    if "--check" in argv:
        try:
            same = open(path, encoding="utf-8").read() == text
        except OSError:
            same = False
        print(json.dumps({"check": "equal" if same else "different", "file": OUT_FILE}))
        return 0 if same else 1
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(json.dumps({"written": OUT_FILE}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
