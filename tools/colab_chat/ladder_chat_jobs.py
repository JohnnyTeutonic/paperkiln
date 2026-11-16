#!/usr/bin/env python3
"""The token ladder's chat campaigns (the research tree's AI_ML/chat_screens_claude/base2/token_ladder/REGISTRATION.md): A35's
jobs file with the base changed, one campaign a base. Claude Fable 5.2, 17 Nov 2026.

    python3 tools/colab_chat/ladder_chat_jobs.py [--check]        writes ladder_chat_<BASE>.json for every base below

A35 (a35_jobs.py, imported and not edited) is S71's chat recipe on base C: 50 reproduction steps of A's SFT1500 from A's base
and their check, A's SFT1500 from the base, then S71's spec on S71's packet at S71's two seeds, with the early health readouts
and the binding score job. A ladder campaign is that file with three things changed and nothing else:
  the base     base_tl<BASE>.safetensors, uploaded by the lane's launch script and bound there by hash;
  the shape    D1 (18 layers, width 512), b2cue_jobs' own D1 architecture, where C had A's;
  the names    sft_tl<BASE> and TL_<BASE>_<seed>, so that no job, folder or model shares a name with A32's or A35's.
The bases: base 2's finished arms N and S at 250,000 steps (N250, S250: the ladder's 4.1 B rung, which exists), and the six
ladder arms when each is home (N1B, S1B, N2B, S2B, NT, ST).
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import a35_jobs as A35  # noqa: E402  (A35's renderer, unchanged)
import b2cue_jobs as B  # noqa: E402  (A32's renderer, unchanged)

BASES = ("N1B", "S1B", "N2B", "S2B", "N250", "S250", "NT", "ST")
SHAPE = "D1"
SEEDS = A35.SEEDS


def tag(base):
    return f"tl{base}"


def arm_name(base, seed):
    return f"TL_{base}_{seed}"


def out_path(base):
    return os.path.join(HERE, f"ladder_chat_{base}.json")


def render(base):
    if base not in BASES:
        raise ValueError(base)
    t = tag(base)
    repro = B.repro_jobs()
    first = {k: v for k, v in repro[0].items() if k != "after"}
    gate = dict(repro[1], name="gate")
    jobs = [first, gate]
    jobs += B.sft_jobs(t, SHAPE, f"base_{t}.safetensors")
    scores = []
    for sd in SEEDS:
        n = arm_name(base, sd)
        spec = B.arm_spec(SHAPE, f"{B.C}/sft_{t}/sft_{t}.safetensors", sd, "P", packet=A35.PACKET)
        jobs += [B.train_job(n, spec, after=["gate", f"sft_{t}"], init_job=f"sft_{t}")] + B.early_jobs(n, n, scored=False)
        scores.append(B.score_job(n, A35.EARLY_REQ, n))
    jobs += scores
    return {"_comment": f"Token ladder, chat campaign on base {base} (lane base2/token_ladder/REGISTRATION.md), rendered by "
                        "tools/colab_chat/ladder_chat_jobs.py; edit that file, not this one.",
            "max_parallel": 1, "max_parallel_light": 2, "home_markers": True, "jobs": jobs}


def text(base):
    return json.dumps(render(base), indent=1) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    bad = []
    for b in BASES:
        if a.check:
            if not os.path.exists(out_path(b)) or open(out_path(b)).read() != text(b):
                bad.append(b)
        else:
            with open(out_path(b), "w", newline="\n") as f:
                f.write(text(b))
    if bad:
        sys.exit(f"differs from its rendering: {bad}")
    print(f"{len(BASES)} jobs files {'match' if a.check else 'written'}")


if __name__ == "__main__":
    main()
