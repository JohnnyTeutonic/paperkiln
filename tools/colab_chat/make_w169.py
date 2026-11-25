#!/usr/bin/env python3
"""The jobs file of the 169M base W169 (the author's research tree, AI_ML/chat_screens_claude/base2/w169/REGISTRATION.md).
Claude Opus 5.5 (a fork of the coordinating session), 26 Nov 2026.

    python3 tools/colab_chat/make_w169.py            # writes w169_W169.json and base2_rung_W169.json
    python3 tools/colab_chat/make_w169.py --check    # compares them with what is on disk; non-zero on any difference

Base 2's arm N (make_base2.py, imported and not edited) with the model's shape changed, and what the shape forces:

    shape         12 x 1024, 16 heads of 64, d_ff 2,752, SwiGLU, 4 state tokens: 168,588,288 parameters (N: 69,751,296)
    peak rate     0.0003 (N: 0.0005); the registration gives the reason
    names         base2_W169, pt_base2_W169, ckpt-W169, rung_W169_s<step>, BASE2_W169
    rung limits   a held rung's time limit 10,800 s (N: 3,600), since a rung of a model 2.4 times N's takes longer
    store halts   the stall halt at 240 minutes (N: 120) and the no-progress halt at 420 minutes (N: 300), for the same reason

Everything else is N's: the corpus files (N.main.ids, N.tail.ids) and the window sampler, 250,000 steps of 16 rows of 1,024
(8 x 2), 4.096 billion positions, the 4,000 warm-up steps, the cosine to a tenth of the peak, the tail from 85% of the
steps, bf16 with the flash kernel, the seed (1), the save every 500 steps, the marks (10,000, 60,000, 122,000, 244,000,
250,000), the panels and Carry-5 at each mark, the pins job, the store's inputs, push cap (200 GB) and release turn (0).
"""
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_base2 as M  # noqa: E402  (base 2's builder, unchanged)

NAME = "W169"
RECIPE = "N"
D, LAYERS, HEADS, D_FF = 1024, 12, 16, 2752
PARAMS = M.params(D, LAYERS, D_FF)
assert PARAMS == 168588288
LR = 0.0003
RUNG_TIMEOUT_S = 10800
STORE = {"store_stall_min": 240, "store_no_progress_min": 420}
SHAPE_FILE = "tools/colab_chat/base2_rung_W169.json"


def arch():
    a = M.arch_d1()
    a["custom"] = {"d": D, "layers": LAYERS, "heads": HEADS, "d_ff": D_FF}
    return a


def spec():
    s = M.spec(RECIPE)
    s["name"] = f"base2_{NAME}"
    s["arch"] = arch()
    s["train"]["lr"] = LR
    s["out_dir"] = f"/content/chat/pt_base2_{NAME}"
    return s


def rung_shape():
    s = spec()
    s.pop("out_dir")
    s["name"] = f"base2_rung_{NAME}"
    s["data"].pop("anneal")
    s["_shape"] = {"note": "the rung job's shape: arch and T only reach it", "params": PARAMS}
    return s


def _rung(job, step, of=True, after=None):
    panels = [(p, f"{M.DATA}/panel_{p}.ids") for p in M.PANELS]
    j = M.rung(job, step, NAME, SHAPE_FILE, panels, of=of, after=after, seed=1)
    if "timeout_s" in j:
        j["timeout_s"] = RUNG_TIMEOUT_S
    cmd = j["cmd"]
    if cmd[0] == "timeout":
        cmd[3] = str(RUNG_TIMEOUT_S)
    return j


def arm():
    ref = M.arm(RECIPE, store=STORE)
    job = f"pt_base2_{NAME}"
    marks = M.MARKS[RECIPE]
    ref_train = next(j for j in ref["jobs"] if j["name"] == f"pt_base2_{RECIPE}")
    train = dict(copy.deepcopy(ref_train), name=job, model_name=f"base2_{NAME}", store_branch=f"ckpt-{NAME}", spec=spec())
    jobs = [M.pins_job(f"arm-{RECIPE}"), train]
    jobs += [_rung(job, m) for m in marks[:-1]]
    jobs += [_rung(job, marks[-1], of=False, after=[job])]
    out = copy.deepcopy(ref)
    steps = train["spec"]["train"]["steps"]
    out["_comment"] = (f"Base 2, arm {NAME}: arm N's recipe at the 12 x 1024 shape ({PARAMS:,} parameters), peak rate {LR}. "
                       "Not admitted: the launch gate is two reviews and Jonathan's dated line.")
    out["_base2"] = {"arm": NAME, "recipe_of": RECIPE, "seed": 1, "shape": "W169", "params": PARAMS,
                     "positions_per_step": M.ROWS * M.T, "steps": steps, "positions": steps * M.ROWS * M.T,
                     "tokens_per_parameter": round(steps * M.ROWS * M.T / PARAMS, 2),
                     "last_main_step": int(M.TAIL_FROM * steps), "first_tail_step": int(M.TAIL_FROM * steps) + 1,
                     "marks": marks, "mark_positions": {str(m): m * M.ROWS * M.T for m in marks}, "corpus": RECIPE}
    out["jobs"] = jobs
    return out


def files():
    return {f"w169_{NAME}.json": arm(), "base2_rung_W169.json": rung_shape()}


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
