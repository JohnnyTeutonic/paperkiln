#!/usr/bin/env python3
"""The jobs file of SYMW, the symbolic warm-up rung (the author's research tree,
AI_ML/chat_screens_claude/base2/token_ladder/symbolic_warmup/REGISTRATION.md). Claude Opus 5.5, 26 Nov 2026.

    python3 tools/colab_chat/make_symw.py            # writes symw_SYMW.json
    python3 tools/colab_chat/make_symw.py --check    # compares it with what is on disk; non-zero on any difference

SYMW is the token ladder's rung N1B (make_ladder.py, imported and not edited) with one change: the training windows of
steps 1 to WARMUP_STEPS come from the symbolic binding text (data.warmup, tools/mtstudio.cpp), so those 2,000 steps'
windows of N1B are replaced and nothing else. The window generator is drawn once a row in both runs, so from step 2,001 on
every window, the tail's included, is N1B's (tests/test_symw_warmup.py). Shape, seed, schedule (4,000 warm-up steps of the
rate, cosine to a tenth, the tail from step 51,001), 60,000 steps, marks, panels, Carry-5, pins and the store's switches are
N1B's. Added: the symbolic corpus among the store's inputs, a light job that checks its sha256 before training starts,
and the names. The release turn is 20 minutes (the ladder's slot 2), so SYMW never shares a turn with an arm on turn 0.
"""
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_base2 as M  # noqa: E402  (base 2's builder, unchanged)
import make_ladder as L  # noqa: E402  (the token ladder's builder, unchanged)

NAME = "SYMW"
REF = "N1B"
WARMUP_STEPS = 2000
SYMW_IDS = "symw.ids"
# sha256 of the symbolic corpus's ids (C:/ml_artifacts/symw/corpus/receipt.json); the pin job refuses any other bytes.
SYMW_IDS_SHA256 = "a33ab72837e63e2ecbd6109610821afec87443e93edba8061cdc0c315acc7ae2"
SYMW_SET = {"branch": "main", "prefix": "corpus/symw", "to": "chat/data", "files": [SYMW_IDS]}
OFFSET_MIN = 20
OUT_FILE = "symw_SYMW.json"


def pin_job():
    path = f"{M.DATA}/{SYMW_IDS}"
    return {"name": "symw_pin", "light": True, "halt_on_failure": True,
            "cmd": ["sh", "-c", f"sha256sum {path} | tee {{out}}/symw_pin.txt | grep -q '^{SYMW_IDS_SHA256} '"],
            "outputs": ["symw_pin.txt"]}


def arm():
    ref = L.arm(REF)
    out = copy.deepcopy(ref)
    rjob, job = f"pt_base2_{REF}", f"pt_base2_{NAME}"
    jobs = [M.pins_job("arm-N"), pin_job()]
    for j in ref["jobs"]:
        if j["name"] == "pins":
            continue
        j = copy.deepcopy(j)
        if j["name"] == rjob:
            j.update(name=job, model_name=f"base2_{NAME}", store_branch=f"ckpt-{NAME}", after=["pins", "symw_pin"])
            j["spec"]["name"] = f"base2_{NAME}"
            j["spec"]["out_dir"] = f"/content/chat/{job}"
            j["spec"]["data"]["warmup"] = {"corpus": f"{M.DATA}/{SYMW_IDS}", "steps": WARMUP_STEPS}
        else:
            j = json.loads(json.dumps(j).replace(REF, NAME))      # every name of a rung job (rung_N1B_*, BASE2_N1B, ...)
        jobs.append(j)
    out["jobs"] = jobs
    out["store_inputs"] = ref["store_inputs"] + [SYMW_SET]
    b = out["_base2"]
    b.update(arm=NAME, rung="1B", slot=None, warmup={"corpus": SYMW_IDS, "steps": WARMUP_STEPS,
                                                      "positions": WARMUP_STEPS * M.ROWS * M.T,
                                                      "replaces": f"{REF}'s windows of steps 1 to {WARMUP_STEPS}",
                                                      "sha256": SYMW_IDS_SHA256})
    b["corpus"] = "N after the symbolic warm-up"
    out["_comment"] = (f"SYMW: the token ladder's {REF} with its windows of steps 1 to {WARMUP_STEPS:,} taken from the symbolic "
                       "binding text (data.warmup); nothing else is changed. Not admitted: the launch gate is two reviews "
                       "and the lead's ruling.")
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
