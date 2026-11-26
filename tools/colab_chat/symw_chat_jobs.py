#!/usr/bin/env python3
"""SYMW's chat campaign (the research tree's AI_ML/chat_screens_claude/base2/token_ladder/symbolic_warmup/REGISTRATION.md,
section 7): the token ladder's standard chat campaign with the base changed. Claude Fable 5.2, 26 Nov 2026.

    python3 tools/colab_chat/symw_chat_jobs.py [--check]        writes symw_chat_SYMW.json

SYMW is the ladder's arm N1B with a symbolic warm-up (same D1 shape, same steps), so its campaign is N1B's jobs file
(ladder_chat_jobs.py, imported and not edited) with the base's name changed wherever it occurs and nothing else: the
uploaded base base_tlSYMW.safetensors, the jobs sft_tlSYMW and TL_SYMW_<seed>, and the comment. The name N1B occurs in
N1B's file only inside those names (checked here and in tests/test_symw_chat_jobs.py).
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ladder_chat_jobs as J  # noqa: E402  (the ladder's renderer, unchanged)

BASE, LIKE = "SYMW", "N1B"
OUT = os.path.join(HERE, f"symw_chat_{BASE}.json")
NAME_FORMS = (f"tl{LIKE}", f"TL_{LIKE}_")       # the only strings that may carry the ladder base's name


def render():
    d = J.render(LIKE)
    d.pop("_comment")
    t = json.dumps(d)
    stray = re.sub("|".join(re.escape(x) for x in NAME_FORMS), "", t)
    if LIKE in stray or BASE in t:
        raise ValueError(f"{LIKE} occurs outside the job and file names, or {BASE} occurs already: no rendering")
    out = {"_comment": f"SYMW's chat campaign (lane base2/token_ladder/symbolic_warmup/REGISTRATION.md section 7): the token "
                       f"ladder's campaign on base {LIKE} with the base's name changed, rendered by "
                       "tools/colab_chat/symw_chat_jobs.py; edit that file, not this one."}
    out.update(json.loads(t.replace(LIKE, BASE)))
    return out


def text():
    return json.dumps(render(), indent=1) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    if a.check:
        if not os.path.exists(OUT) or open(OUT).read() != text():
            sys.exit("symw_chat_SYMW.json differs from its rendering")
        print("1 jobs file matches")
        return
    with open(OUT, "w", newline="\n") as f:
        f.write(text())
    print("1 jobs file written")


if __name__ == "__main__":
    main()
