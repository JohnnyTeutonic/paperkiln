#!/usr/bin/env python3
"""The jobs files of the token ladder (the author's research tree, AI_ML/chat_screens_claude/base2/token_ladder/REGISTRATION.md).
Claude Fable 5.2, 17 Nov 2026.

    python3 tools/colab_chat/make_ladder.py            # writes ladder_<arm>.json for every arm below
    python3 tools/colab_chat/make_ladder.py --check    # compares them with what is on disk; non-zero on any difference

One variable is changed against base 2's arms N and S (make_base2.py, imported and not edited): the number of training
steps, and with it the positions read. Everything else of an arm is base 2's: the D1 shape, the corpus files and the rule
that draws one window a row from one generator, the batch (8 x 2 rows of 1,024), the peak rate, the 4,000 warm-up steps, the
cosine to a tenth of the peak, the tail part from 85% of the steps, the precision, the save interval, the pins job, the rung
job at each mark, the store's inputs and switches. The schedule's shape is held and its length follows the steps; the
warm-up is base 2's 4,000 steps at every rung (a choice, stated in the registration).

    arm    recipe  steps     positions   tokens a parameter   marks
    N1B    N        60,000   0.98 B      14                   10,000  58,500  60,000
    S1B    S        60,000   0.98 B      14                   the same
    N2B    N       122,000   2.00 B      29                   10,000  119,000  122,000
    S2B    S       122,000   2.00 B      29                   the same
    NT     N       375,000   6.14 B      88                   10,000  366,000  375,000
    ST     S       375,000   6.14 B      88                   the same

60,000 and 122,000 are base 2's own marks, so each low rung has read exactly what N or S had read at that mark, with a
schedule that ends there. Each rung's late mark is the last multiple of the save interval at or under 97.6% of its steps
(base 2's 244,000 of 250,000). The second pretraining seed (make_base2.SECOND_SEED, 50) of each arm is defined here from the
start, as <arm>s50, so that adding it later changes nothing; no second-seed arm is launched by this registration.

"slot" is the Colab L4 an arm takes in the registered order (three at once at the most): an arm starts when the arm before
it in its slot has ended. An arm's turn on the store's release grid is its slot's (0, 20 or 40 minutes), so the arms that
are live together never share a turn, as base 2's three arms did not.
"""
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_base2 as M  # noqa: E402  (base 2's builder, unchanged)

LATE_FRACTION = 0.976
FIRST_MARK = 10000
STEPS = {"1B": 60000, "2B": 122000, "T": 375000}
# The registered order: slot -> the arms that take it, one after another.
SLOTS = {1: ["N1B", "N2B", "S2B"], 2: ["S1B", "ST"], 3: ["NT"]}
SLOT_OFFSET_MIN = {1: 0, 2: 20, 3: 40}
SECOND = "s50"


def _rungs():
    r = {}
    for slot, names in SLOTS.items():
        for n in names:
            recipe, rung = n[0], n[1:]
            for seed, suffix in ((1, ""), (M.SECOND_SEED, SECOND)):
                r[n + suffix] = {"recipe": recipe, "rung": rung, "steps": STEPS[rung], "seed": seed, "slot": slot,
                                 "launched_by_this_registration": seed == 1}
    return r


RUNGS = _rungs()
FIRST_SEED_ARMS = [n for n, r in RUNGS.items() if r["seed"] == 1]


def marks(steps):
    late = int(LATE_FRACTION * steps) // M.CHECKPOINT_EVERY * M.CHECKPOINT_EVERY
    out = [FIRST_MARK, late, steps]
    if not (0 < FIRST_MARK < late < steps) or any(m % M.CHECKPOINT_EVERY for m in out):
        raise ValueError(f"marks {out} are not increasing saves of a {steps}-step run")
    return out


def spec(name):
    r = RUNGS[name]
    s = M.spec(r["recipe"])
    s["name"] = f"base2_{name}"
    s["out_dir"] = f"/content/chat/pt_base2_{name}"
    s["train"]["steps"] = r["steps"]
    s["train"]["seed"] = r["seed"]
    return s


def arm(name):
    """Arm `name`'s jobs file: make_base2.arm(recipe) with the steps, the marks, the seed and the names changed, and the
    release turn of its slot. Every other key is the recipe arm's own."""
    r = RUNGS[name]
    recipe, steps, seed = r["recipe"], r["steps"], r["seed"]
    ref = M.arm(recipe)
    job = f"pt_base2_{name}"
    s = spec(name)
    mk = marks(steps)
    panels = [(p, f"{M.DATA}/panel_{p}.ids") for p in M.PANELS]
    ref_train = next(j for j in ref["jobs"] if j["name"] == f"pt_base2_{recipe}")
    train = dict(copy.deepcopy(ref_train), name=job, model_name=f"base2_{name}", store_branch=f"ckpt-{name}",
                 snapshot_steps=mk, spec=s)
    jobs = [M.pins_job(f"arm-{recipe}"), train]
    jobs += [M.rung(job, m, name, M.SHAPE_D1, panels, seed=seed) for m in mk[:-1]]
    jobs += [M.rung(job, mk[-1], name, M.SHAPE_D1, panels, of=False, after=[job], seed=seed)]
    out = copy.deepcopy(ref)
    out["_comment"] = (f"Token ladder, arm {name}: base 2's arm {recipe} at {steps:,} steps and seed {seed}; nothing else is "
                       "changed. Not admitted: the launch gate is two reviews and Jonathan's dated line.")
    out["_base2"] = {"arm": name, "ladder": True, "recipe_of": recipe, "rung": r["rung"], "seed": seed, "slot": r["slot"],
                     "shape": "D1", "params": M.PARAMS["D1"], "positions_per_step": M.ROWS * M.T, "steps": steps,
                     "positions": steps * M.ROWS * M.T,
                     "tokens_per_parameter": round(steps * M.ROWS * M.T / M.PARAMS["D1"], 2),
                     "last_main_step": int(M.TAIL_FROM * steps), "first_tail_step": int(M.TAIL_FROM * steps) + 1,
                     "marks": mk, "mark_positions": {str(m): m * M.ROWS * M.T for m in mk}, "corpus": recipe}
    out = _with_offset(out, SLOT_OFFSET_MIN[r["slot"]])
    out["jobs"] = jobs
    return out


def _with_offset(d, off):
    """d with the release turn `off`, its switches written as make_base2.arm writes them: in RUNNER_STORE_DEFAULTS' order,
    before store_inputs, and only where a value is not the runner's own default."""
    sw = {k: d.get(k, v) for k, v in M.RUNNER_STORE_DEFAULTS.items()}
    sw["store_release_offset_min"] = off
    out = {}
    for k, v in d.items():
        if k in M.RUNNER_STORE_DEFAULTS:
            continue
        if k == "store_inputs":
            out.update({s: sw[s] for s in M.RUNNER_STORE_DEFAULTS if sw[s] != M.RUNNER_STORE_DEFAULTS[s]})
        out[k] = v
    return out


def files():
    return {f"ladder_{n}.json": arm(n) for n in RUNGS}


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
