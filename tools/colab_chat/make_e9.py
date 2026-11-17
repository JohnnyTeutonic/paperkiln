#!/usr/bin/env python3
"""E9 (scenario diversity): the D arms' pretraining jobs files.

    python tools/colab_chat/make_e9.py            write e9_<arm>.json beside this file
    python tools/colab_chat/make_e9.py --check    exit 1 unless every file equals what this builder writes

Each D arm is the token ladder's arm S1B (make_ladder.arm, unchanged) with only the scenario stream swapped: its
corpus files are the level's (`D2.main.ids`, `D2.tail.ids`, store prefix `corpus/D2`, pins entry `arm-D2`) in place
of S's, and its names (job, model, store branch, rung cells) carry the arm's name in place of S1B's. Steps, marks,
schedule, seed, shape, batch, panels, instruments and every other key are S1B's. A second-seed arm (<arm>s50) is
S1Bs50 with the same swap. tests/test_e9_jobs.py holds every leaf of every file equal to S1B's or S1Bs50's but for
these substitutions.

    arm     level    stream                                             corpus   release turn
    D2      L2       surface: ten renderings, more frames               D2       0 (slot 1)
    D3      L3       content: L2 plus four entity domains               D3       20 (slot 2)
    D4      L4       operations: L3 plus six operation families         D4       40 (slot 3)
    D4ops   L4-ops   L4's operations on L1's surface and content        D4ops    0 (after the first wave)
    Dks50   seed 50 of D2, D3 or D4 (the onset level, or D4)            as Dk    40 (beside the ladder's
                                                                                 N1Bs50 at 0 and S1Bs50 at 20)

The level streams come from E9's generator (research repo, cue_selective/diversity/generate_level.py) at the
scenario seed and shard size base 2 used for S; the corpus build for a D arm is S's stage 3 with that generator
and E9's validator. Until a D corpus is built and pinned (e9_pins.json holds its two id hashes, adopted from its stage receipt; base2_pins.json is never edited), the arm's pins job
fails and nothing trains. Nothing here is admitted: E9's launch gate is its frozen registration, two reviews
and Jonathan's dated line.

Claude Opus 5.5 (a fork of the coordinating session), 18 Nov 2026.
"""
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_base2 as M   # noqa: E402  (base 2's builder, unchanged)
import make_ladder as L  # noqa: E402  (the token ladder's builder, unchanged)

LEVEL = {"D2": "L2", "D3": "L3", "D4": "L4", "D4ops": "L4-ops"}
OFFSET_MIN = {"D2": 0, "D3": 20, "D4": 40, "D4ops": 0}
SECOND_OFFSET_MIN = 40
REF = "S1B"


def arms():
    out = {}
    for base in LEVEL:
        out[base] = {"base": base, "seed": 1, "ref": REF, "offset_min": OFFSET_MIN[base]}
        if base != "D4ops":
            out[base + L.SECOND] = {"base": base, "seed": M.SECOND_SEED, "ref": REF + L.SECOND,
                                    "offset_min": SECOND_OFFSET_MIN}
    return out


ARMS = arms()


def corpus_subs(c):
    """Exact string values naming S's corpus, and what they become for corpus c."""
    return {f"{M.DATA}/S.main.ids": f"{M.DATA}/{c}.main.ids", f"{M.DATA}/S.tail.ids": f"{M.DATA}/{c}.tail.ids",
            "S.main.ids": f"{c}.main.ids", "S.tail.ids": f"{c}.tail.ids", "corpus/S": f"corpus/{c}", "arm-S": f"arm-{c}"}


def substitute(x, ref, name, c):
    """x with every string leaf's `ref` replaced by `name`, and S's corpus values by corpus c's."""
    subs = corpus_subs(c)
    if isinstance(x, dict):
        return {k: substitute(v, ref, name, c) for k, v in x.items()}
    if isinstance(x, list):
        return [substitute(v, ref, name, c) for v in x]
    if isinstance(x, str):
        return subs.get(x, x.replace(ref, name))
    return x


def arm(name):
    a = ARMS[name]
    base, ref = a["base"], a["ref"]
    src = L.arm(ref)
    meta = src.pop("_base2")
    src.pop("_comment")
    out = substitute(src, ref, name, base)
    out = L._with_offset(out, a["offset_min"])
    pins = out["jobs"][0]["cmd"]                 # E9's own pins tool (A4, B4): base2_pins.py is never edited
    assert pins[1] == "tools/colab_chat/base2_pins.py", pins
    pins[1] = "tools/colab_chat/e9_pins.py"
    meta = dict(meta, arm=name, corpus=base, slot=None, e9_level=LEVEL[base], e9_ref_arm=ref)
    head = {"_comment": (f"E9 scenario diversity, arm {name}: the token ladder's arm {ref} with the scenario stream of "
                         f"level {LEVEL[base]} (corpus {base}) in place of S's; nothing else is changed. Not admitted: "
                         "E9's launch gate is its frozen registration, two reviews and Jonathan's dated line."),
            "_base2": meta}
    head.update(out)
    return head


DATA_HOURS = 6        # a corpus build's hour limit on one L4 (base 2's took about 3.5 hours for a larger tokenisation)


def data_jobs(base):
    """The corpus build of corpus `base` (D2, D3, D4, D4ops): base 2's data jobs file with E9's stage 3 alone and the
    E9 seed sets pulled beside the engine."""
    src = json.load(open(os.path.join(HERE, "base2_data.json"), encoding="utf-8"))
    vocab = "/content/chat/upload/vocab_llama3_bpe.gguf"
    return {"_comment": (f"E9 corpus build of {base} (level {LEVEL[base]}): base 2's stage 3 with the level's stream "
                         f"(e9_data.sh), on one L4 session, at most {DATA_HOURS} hours. Not admitted: E9's launch gate."),
            "store_inputs": src["store_inputs"] + [
                {"branch": "main", "prefix": "seed/e9", "to": "chat/upload/e9/diversity"},
                {"branch": "main", "prefix": "seed/e9_receipts", "to": "chat/upload/e9/diversity/results"}],
            "max_parallel": 1, "max_parallel_light": 1,
            "jobs": [{"name": "data", "light": True, "halt_on_failure": True,
                      "cmd": ["env", f"E9_ARM={base}", f"E9_LEVEL={LEVEL[base]}", "bash", "tools/colab_chat/e9_data.sh",
                              "{out}", vocab, "stage3"],
                      "outputs": [f"e9_{base}_stage3.json", f"e9_manifest_{base}.json"]}]}


def files():
    out = {f"e9_{n}.json": arm(n) for n in ARMS}
    out.update({f"e9_data_{b}.json": data_jobs(b) for b in LEVEL})
    return out


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
