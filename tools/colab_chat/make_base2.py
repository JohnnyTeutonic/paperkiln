#!/usr/bin/env python3
"""The jobs files of base 2 (AI_ML/chat_screens_claude/base2/BASE2_PLAN.md sections 2 to 5, in the author's research tree).
Claude Fable 5.2, 11 Nov 2026.

    python3 tools/colab_chat/make_base2.py            # writes the files below
    python3 tools/colab_chat/make_base2.py --check    # compares them with what is on disk; non-zero on any difference

    base2_data.json        the corpus build (base2_data.sh, three stages) and a 20-step load of the real ids
    base2_N.json           arm N: D1 from scratch on the natural corpus
    base2_S.json           arm S: D1 from scratch, the same recipe with 15% scenario tokens
    base2_C.json           arm C: base A continued on S's corpus, for equal compute
    base2_rehearsal.json   the rehearsal of the store path: a forced stop before the tail switch, and its control
    base2_rehearsal_b.json the same run with the forced stop after the tail switch, and its control
    base2_rung_D1.json     the D1 shape for the rung job (arch and T only reach it)
    base2_load_probe.json  the load measurement's spec

What C shares with S. Both read S's corpus with seed 1, and the trainer draws one window a row from one generator, row
after row, whatever the split of a step's 16 rows into micro-batches (tools/mtstudio.cpp, the `rng() % lim` in the
micro-step loop; tests/test_base2_jobs.py runs 8 x 2 against 16 x 1 on a CPU build and compares the losses). So C at
16 x 1 reads the windows S reads at 8 x 2 for as long as both read the main part: steps 1 to 212,500. S reads its tail
from step 212,501 and C from step 280,501 (`step > int(from_fraction * steps)`), so from 212,501 on they read different
text. C's marks at 60,000 and 122,000 are at S's windows; its marks from 244,000 on are not.

Every file runs under boot_base2.sh with a checkpoint store (runner_multi.py --ckpt-store hf): "store_inputs" names the
sets the boot pulls and checks, a training job's "store_branch" is where the courier keeps its generations, and its
"snapshot_steps" are the marks the courier pushes to the store's main branch. Each arm's first job hashes its ids, panels,
vocabulary (and, for C, base A's weights) against base2_pins.json and training waits on it, on every VM.
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
UP = "/content/chat/upload"
DATA = "/content/chat/data"
V = f"{UP}/vocab_llama3_bpe.gguf"
TXN = {"MICROTORCH_CHECKPOINT_TXN": "1"}
SHAPE_A = "experiments/tiny_chat_long_2026-10/specs/cell_Sfull.json"   # base A's shape, as base A's rungs read it
SHAPE_D1 = "tools/colab_chat/base2_rung_D1.json"
A_WEIGHTS = "tc_b1_r1.safetensors"
T, ROWS = 1024, 16                  # positions a step: 16 rows of 1,024, as 8 x 2 or 16 x 1
STEPS = {"N": 250000, "S": 250000, "C": 330000}
# Section 3.2. The last mark of each arm is its end.
MARKS = {"N": [10000, 60000, 122000, 244000, 250000], "S": [10000, 60000, 122000, 244000, 250000],
         "C": [10000, 60000, 122000, 244000, 305000, 330000]}
TAIL_FROM = 0.85
CHECKPOINT_EVERY = 500
RUNG_TIMEOUT_S = 3600
# Base A's rung settings (make_tc_b1.py): Carry-5 in forwards of at most 4,096 ids, a 1,024 MiB pool beside the held trainer.
RUNG_SCORE_BATCH = 4096
RUNG_SCORE_CHUNK = 4096
RUNG_POOL_MB = 1024
PANELS = ["fineweb", "dialogue", "scenario"]
SEED_VOCAB = {"branch": "main", "prefix": "seed/vocab", "to": "chat/upload"}
SEED_ENGINE = {"branch": "main", "prefix": "seed/scenario_engine", "to": "chat/upload/scenario_engine"}
SEED_WEIGHTS = {"branch": "main", "prefix": "seed/weights", "to": "chat/upload"}
PANEL_SETS = [{"branch": "main", "prefix": "corpus/panels", "to": "chat/data"},
              {"branch": "main", "prefix": "corpus/panel_scenario", "to": "chat/data", "files": ["panel_scenario.ids"]}]


def corpus_set(arm):
    return {"branch": "main", "prefix": f"corpus/{arm}", "to": "chat/data", "files": [f"{arm}.main.ids", f"{arm}.tail.ids"]}


def params(d, layers, d_ff, vocab=16384, k=4):
    """Token embedding (shared with the head), state embedding, per layer four attention projections, three SwiGLU matrices
    and two RMSNorm gains, and the final gain."""
    return vocab * d + k * d + layers * (4 * d * d + 3 * d * d_ff + 2 * d) + d


def arch_d1():
    return {"preset": "llama-tiny", "rope_heads": "all", "custom": {"d": 512, "layers": 18, "heads": 8, "d_ff": 1536},
            "state_tokens": {"k": 4, "mask": "causal"}}


def arch_a():
    return {"preset": "llama-tiny", "rope_heads": "all", "custom": {"d": 768, "layers": 8, "heads": 12, "d_ff": 2240},
            "state_tokens": {"k": 4, "mask": "causal"}}


PARAMS = {"D1": params(512, 18, 1536), "A": params(768, 8, 2240)}
assert PARAMS == {"D1": 69751296, "A": 72761088}


def spec(arm):
    """Section 3.1. N and S differ in the corpus alone; C is base A's shape on S's corpus at 16 x 1."""
    corpus = "N" if arm == "N" else "S"
    d1 = arm != "C"
    train = {"steps": STEPS[arm], "lr": 0.0005 if d1 else 0.00025, "batch": 8 if d1 else 16, "accum": 2 if d1 else 1,
             "clip": 1.0, "eval_every": 1000, "eval_windows": 32, "checkpoint_every": CHECKPOINT_EVERY, "gradmap_every": 0,
             "seed": 1, "optimizer": "adamw", "precision": "bf16", "attention_kernel": "flash",
             "lr_schedule": {"warmup_steps": 4000 if d1 else 2000, "decay": "cosine", "min_lr_frac": 0.1}}
    return {"name": f"base2_{arm}", "arch": arch_d1() if d1 else arch_a(),
            "data": {"corpus": f"{DATA}/{corpus}.main.ids", "vocab": V, "vocab_cap": 16384, "T": T, "max_tokens": 0,
                     "anneal": {"corpus": f"{DATA}/{corpus}.tail.ids", "from_fraction": TAIL_FROM}},
            "train": train, "export": {"formats": ["safetensors", "gguf"]}, "out_dir": f"/content/chat/pt_base2_{arm}"}


def rung_d1():
    s = spec("N")
    s.pop("out_dir")
    s["name"] = "base2_rung_D1"
    s["data"].pop("anneal")
    s["_shape"] = {"note": "the rung job's shape: arch and T only reach it", "params": PARAMS["D1"]}
    return s


def load_probe():
    """Twenty steps of D1 on S's real ids, no eval and no save: the trainer's start-up time and peak memory with 3.8
    billion ids held (the plan's risk 6)."""
    s = spec("S")
    s.pop("out_dir")
    s["name"] = "base2_load_probe"
    s["train"].update(steps=20, eval_every=1000, checkpoint_every=1000)
    s["train"]["lr_schedule"]["warmup_steps"] = 10
    s["export"] = {"formats": []}
    return s


def rung(job, step, tag, shape, panels, of=True, after=None, held=True):
    """The zero-rate panels and Carry-5 on the weights at a mark, with training held. of: the mark is a snapshot
    (rung_<step>/); otherwise the run's own last checkpoint, once it has ended. held False: beside training, not
    holding it. Every rung has a time limit: jobs_vm.py's own for a rung that holds training ("timeout_s"), and
    coreutils' timeout around the command for one that does not, which jobs_vm.py does not time."""
    weights = f"/content/chat/{job}/rung_{step}/model.safetensors" if of else f"/content/chat/{job}/model.safetensors"
    exclusive = of and held
    limit = [] if exclusive else ["timeout", "-k", "60", str(RUNG_TIMEOUT_S)]
    j = {"name": f"rung_{tag}_s{step}", "after": after or "pins", "light": True,
         "env": {"MICROTORCH_POOL_MB": RUNG_POOL_MB},
         "cmd": [*limit, "python3", "tools/colab_chat/tiny_chat_run.py", "rung", "--out", "{out}", "--spec", shape, "--export-gguf",
                 "--weights", weights, "--pt-dir", f"/content/chat/{job}", "--step", str(step), "--job", job,
                 "--cell", f"BASE2_{tag}", "--seed", "1", "--share", "28",
                 *[x for name, path in panels for x in ("--panel", f"{name}={path}")],
                 "--carry5-items", "corpus/eval/carry5_v1.jsonl",
                 "--score-batch-tokens", str(RUNG_SCORE_BATCH), "--score-chunk", str(RUNG_SCORE_CHUNK)],
         "outputs": ["rung.json"]}
    if of:
        j["rung_of"] = {"job": job, "step": step}
    if exclusive:
        j["exclusive"], j["timeout_s"] = True, RUNG_TIMEOUT_S
    return j


def pins_job(mode):
    return {"name": "pins", "light": True, "halt_on_failure": True,
            "cmd": ["python3", "tools/colab_chat/base2_pins.py", DATA, mode, "--upload", UP, "--out", "{out}/pins.json"],
            "outputs": ["pins.json"]}


def arm(a, store=None):
    """Arm a's jobs file. The store's switches come from ARM_STORE, then ARM_STORE_BY_ARM[a], then `store` (a test's):
    the registration sets them there, so that --check stays equal to what is on disk. A switch that is the
    runner's own default is not written, so the files do not change until a value does."""
    sw = {**ARM_STORE, **ARM_STORE_BY_ARM.get(a, {}), **(store or {})}
    if set(sw) != set(ARM_STORE):
        raise ValueError(f"unknown store switch: {sorted(set(sw) - set(ARM_STORE))}")
    job = f"pt_base2_{a}"
    corpus = "N" if a == "N" else "S"
    s = spec(a)
    marks = MARKS[a]
    panels = [(p, f"{DATA}/panel_{p}.ids") for p in PANELS]
    shape = SHAPE_A if a == "C" else SHAPE_D1
    train = {"name": job, "model_name": f"base2_{a}", "store_branch": f"ckpt-{a}", "after": ["pins"],
             "snapshot_steps": marks, "halt_on_failure": True, "env": TXN, "spec": s}
    if a == "C":
        train["init_file"] = A_WEIGHTS
    jobs = [pins_job(f"arm-{a}"), train]
    jobs += [rung(job, m, a, shape, panels) for m in marks[:-1]]
    jobs += [rung(job, marks[-1], a, shape, panels, of=False, after=[job])]
    steps = s["train"]["steps"]
    what = {"N": "D1 (18 x 512) from scratch on the natural corpus",
            "S": "D1 (18 x 512) from scratch on the scenario corpus: N's recipe with 15% scenario tokens",
            "C": "base A (8 x 768) continued from its final weights on the scenario corpus, for equal compute"}[a]
    return {"_comment": f"Base 2, arm {a}: {what}. Not admitted: the launch gate is two reviews and Jonathan's dated line.",
            "_base2": {"arm": a, "shape": "A" if a == "C" else "D1", "params": PARAMS["A" if a == "C" else "D1"],
                       "positions_per_step": ROWS * T, "steps": steps, "positions": steps * ROWS * T,
                       "last_main_step": int(TAIL_FROM * steps), "first_tail_step": int(TAIL_FROM * steps) + 1,
                       "marks": marks, "mark_positions": {str(m): m * ROWS * T for m in marks}, "corpus": corpus,
                       **({"same_windows_as_S_through_step": int(TAIL_FROM * STEPS["S"]),
                           "note": "C reads S's windows while both read the main part, steps 1 to 212,500; S reads its "
                                   "tail from 212,501 and C from 280,501"} if a == "C" else {})},
            # The courier's storage guard (runner_multi.py --store-push-cap-gb reads this when the flag is absent):
            # GB counted for the arm (what the store held for it at the last release from home, and what it has
            # pushed since) at which the courier stops pushing and the runner halts the arm. 0 is off. The
            # registered caps (12 Nov 2026) are in ARM_STORE_BY_ARM: 200, 200 and 250 GB.
            "store_push_cap_gb": sw["store_push_cap_gb"],
            # Minutes between an arm's runner's storage releases (--store-release-every-min reads this when the flag
            # is absent). Each release is two commits of the repository's hourly limit, which the three arms share.
            "store_release_every_min": sw["store_release_every_min"],
            # The other switches of the registration (runner_multi.py reads each when its flag is absent), written
            # only when they are not the runner's own defaults: the release's grace, the stall halt and its
            # minutes, the minutes after which pushes with no verified check halt the arm.
            **{k: sw[k] for k in RUNNER_STORE_DEFAULTS if sw[k] != RUNNER_STORE_DEFAULTS[k]},
            "store_inputs": [SEED_VOCAB, *([SEED_WEIGHTS] if a == "C" else []), corpus_set(corpus), *PANEL_SETS],
            "max_parallel": 1, "max_parallel_light": 1, "jobs": jobs}


def data():
    return {"_comment": "Base 2's corpus build on one L4 session taken for its processors and memory: three stages, each "
                        "pushed to the store, so a lost session repeats one at most; then a 20-step load of S's ids.",
            "store_inputs": [SEED_VOCAB, SEED_ENGINE],
            "max_parallel": 1, "max_parallel_light": 1,
            "jobs": [
                {"name": "data", "light": True, "halt_on_failure": True,
                 "cmd": ["bash", "tools/colab_chat/base2_data.sh", "{out}", V, "all"],
                 "outputs": ["base2_stage1.json", "base2_stage2.json", "base2_stage3.json", "data.ledger.json",
                             "filter.ledger.json", *[f"fineweb_edu_shard{n}_v1.sha256" for n in (2, 3, 4, 5)]]},
                # S's ids are pulled and checked against the store's manifest first (base2_data.sh loadprobe): a VM on
                # which the three stages were skipped holds their receipts and no ids. "store_inputs" cannot name
                # corpus/S here, because on the build's first VM that set does not exist yet and the boot would fail.
                {"name": "load_probe", "after": "data", "halt_on_failure": True,
                 "cmd": ["bash", "tools/colab_chat/base2_data.sh", DATA, V, "loadprobe", "{out}"],
                 "outputs": ["loadprobe.json", "run.log", "loadprobe_pins.json"]},
            ]}


STEP_S = 1.053               # D1 at 8 x 2, mixed precision, on an L4 (PROBE_RESULT.md)
STEP_S_A = 0.783             # base A's shape at 16 x 1, mixed precision, on an L4 (PROBE_RESULT.md): arm C
TICK_S = 240                 # runner_multi.TICK
# The hub's limit on commit creation, per repository, for every writer together (HTTP 429 in the rehearsal of
# 12 Nov 2026: "128 per hour"), and what the three arms, which share one repo, may use of it in steady state.
HUB_COMMITS_PER_HOUR = 128
ARMS_COMMITS_PER_HOUR_MAX = 40
ARM_RELEASE_EVERY_MIN = 60   # an arm's runner releases storage every hour (registered 12 Nov 2026): two commits each time
# A TEST THRESHOLD for commits_per_hour_bad_model (tests/test_base2_jobs.py), three quarters of the hub's limit. No
# code enforces it, and it was 64 before that model counted home's retries and the end's release. What the code
# does enforce is each writer's own ceiling, below: they sum to 108 for three arms, and an essential commit passes
# its ceiling after 15 minutes. Only the hub's own 429 bounds the repository.
ARMS_BAD_HOUR_TEST_THRESHOLD = 96
# Each writer's own ceiling (per writer, not per repository). Three arms: 3 x 24 + 3 x 12 = 108, under the hub's 128.
# Three runners keep three budget files (one a state folder) and three couriers one each: six counts, tied by nothing.
COURIER_COMMIT_BUDGET = 24   # runner_multi.py --store-commit-budget-per-hour (a VM's ceiling, not its steady state)
# runner_multi.py --store-home-commit-budget-per-hour. Home commits a release's lock and unlock and nothing else. A
# quarter is kept back from the timed release (nine of the twelve: four timed releases an hour at the most); the
# release at the campaign's end may use all twelve and waits for room. It was 8 until the rerun of 12 Nov 2026
# (PROTOCOL section 47.2), where three timed releases in the hour left the last one no room.
HOME_COMMIT_BUDGET = 12
UNLOCK_TRIES = 2                 # hf_store.UNLOCK_TRIES: an unlock is sent at most twice (it was four until 12 Nov 2026)
HOME_RELEASE_COMMITS_WORST = 1 + UNLOCK_TRIES   # one release at its worst: the lock, and the unlock sent twice
# The generations a branch holds. The rule: THREE (two verified and the newest, between two saves) PLUS ONE FOR EACH
# SESSION LOSS BETWEEN TWO VERIFIED SAVES. A new VM has verified only the generation it restored and removes nothing
# until it has verified one of its own; if it is lost before that, the next VM starts the same way with one more
# generation on the branch. One loss: four, and the next save removes two in one commit. Two in a row: five. This
# is not a bound on what a branch can hold: while checks stay unread or not whole nothing is removed at all, and
# only the push cap (or runner_multi.py --store-unverified-halt-min) ends that. The storage sums take
# ARM_SESSION_LOSSES losses in a row.
GENS_KEPT = 3
ARM_SESSION_LOSSES = 2
ARM_RELEASE_GRACE_S = 7200   # runner_multi.py --store-release-grace-s, its default: what the arms run with
PUSH_CAP_HEADROOM = 0.8      # the sum is held under this share of the cap (as the rehearsal's 28 GB was sized)
RELEASE_SKEW_S = 120         # hf_store.RELEASE_SKEW_S: a release adds it to the grace
ARM_MISSED_RELEASES = 2      # timed releases in a row that do not run, which the cap's sum survives
FINAL_LEDGER_ROUNDS = 2      # runner_multi.STORE_FINAL_LEDGER_ROUNDS: rounds of the end's release that meet an unread ledger
# The store's switches the registration sets, and where: ARM_STORE for all three arms, ARM_STORE_BY_ARM for one
# (a push cap differs by arm). arm() writes them into the jobs files, which runner_multi.py reads when the flag is
# absent; `--check` then compares the files on disk with these values. The values the lead registered (12 Nov
# 2026): a release every 60 minutes with the runner's default grace of 7,200 s, the arms' turns set off by 0, 20
# and 40 minutes; push caps of 200, 200 and 250 GB; the stall halt on at 120 minutes; the unverified halt at 90; the
# no-progress halt (runner_multi.store_no_progress, after the fifth launch review: two counters that count only
# evidence of billing, both reset by any new generation in the store) at 260 minutes in which Colab lists the
# session (a held rung's hour, a boot and a save are well under it) and at the 4th launch without progress. The
# rehearsal's files set their own.
RUNNER_STORE_DEFAULTS = {"store_release_grace_s": 7200, "store_stall_halt": False, "store_stall_min": 60,
                         "store_unverified_halt_min": 0,   # runner_multi.py's own, when neither flag nor key is given
                         "store_release_offset_min": 0,    # minutes an arm's release turns are set off on the UTC grid
                         "store_no_progress_min": 0,       # 0: the runner's own (stall minutes + boot limit + 15, with the stall halt)
                         "store_no_progress_launches": 0}  # 0: the runner's own (4, with the stall halt)
ARM_STORE = {**RUNNER_STORE_DEFAULTS, "store_push_cap_gb": 0, "store_release_every_min": ARM_RELEASE_EVERY_MIN,
             "store_release_grace_s": ARM_RELEASE_GRACE_S, "store_stall_halt": True, "store_stall_min": 120,
             "store_unverified_halt_min": 90, "store_no_progress_min": 260, "store_no_progress_launches": 4}
# The caps (the lead's decision after the second launch review, 12 Nov 2026): 200, 200 and 250 GB, not the sums'
# 45, 45 and 60. Only a release from home lowers a courier's count, so with home down every save adds to it: at the
# sums' caps an outage of about 4 to 5 hours capped every arm. At these, about 31 hours for N and S and 27 for C
# (home_outage_hours, worst case). 650 GB in all, against the account's 1 TB.
ACCOUNT_STORAGE_GB = 1000
ARM_STORE_BY_ARM = {"N": {"store_push_cap_gb": 200, "store_release_offset_min": 0},
                    "S": {"store_push_cap_gb": 200, "store_release_offset_min": 20},
                    "C": {"store_push_cap_gb": 250, "store_release_offset_min": 40}}


def gens_kept_max(losses=ARM_SESSION_LOSSES):
    """Generations on a branch at the most with `losses` session losses between two verified saves: 3 + losses."""
    return GENS_KEPT + losses


def home_commits_worst(release_every_min):
    """Home's part of the modelled bad hour: every timed release that fits in an hour and the release at the
    campaign's end, each at HOME_RELEASE_COMMITS_WORST. 6 at the arms' two hours; inside HOME_COMMIT_BUDGET."""
    timed = -(-60 // release_every_min) if release_every_min else 0
    return HOME_RELEASE_COMMITS_WORST * (int(timed) + 1)


HOME_BUDGET_RESERVE = max(2, HOME_COMMIT_BUDGET // 4)   # hf_store.CommitBudget's reserve, kept from every timed lock: 3


def home_timed_cap(unlock_tries=UNLOCK_TRIES):
    """The most commits one runner's TIMED releases can send in a rolling hour, by the budget's own rule and not by
    counting turns (a late round, its early retry, the next turn and that turn's retry can share an hour, so a
    count of turns understates it). A timed lock is admitted, counted with its unlock (two), only while that keeps
    the hour's count within the budget less its reserve: the count before it is at most 12 - 3 - 2 = 7. The round
    it starts then sends at most a lock and `unlock_tries` unlocks. 7 + 3 = 10 with two unlock sends; 9 with one."""
    return HOME_COMMIT_BUDGET - HOME_BUDGET_RESERVE - 2 + 1 + unlock_tries


def home_commits_rolling(release_every_min=None, unlock_tries=UNLOCK_TRIES):
    """One runner's worst rolling hour at home, in whole commits, as the BUDGET allows it (whatever the interval
    and the offsets): its timed releases at home_timed_cap, and the end's FINAL_LEDGER_ROUNDS rounds, which are
    all the end's release makes across every start of the state folder (each is counted before it is made) and
    are essential, so the reserve does not hold them back. A round is a lock and an unlock sent `unlock_tries`
    times. 10 + 2 x 3 = 16."""
    return home_timed_cap(unlock_tries) + FINAL_LEDGER_ROUNDS * (1 + unlock_tries)


def timed_turns_in_an_hour(release_every_min, offsets_min):
    """The most regular release turns of the arms that fall in one rolling hour: each arm's turns are on the UTC
    grid (runner_multi.store_release_turn), the multiples of `release_every_min` since the epoch plus the arm's
    offset, whenever its runner was started.
    Three arms with no offset: three turns an hour at an interval of an hour or more. With 0, a third and two
    thirds of the interval: two at two hours, three at one hour (every arm still has a turn in every hour)."""
    turns = sorted((off % release_every_min) + k * release_every_min for off in offsets_min for k in range(-2, 4))
    return max(sum(1 for t in turns if s <= t < s + 60) for s in turns)


def commits_rolling_three_arms(per_arm, release_every_min, offsets_min, unlock_tries=UNLOCK_TRIES):
    """The rolling hour for the three arms together: every arm's saves, mark and branch as in commits_rolling_hour,
    and each runner's home at what its budget allows (home_commits_rolling). The offsets do not enter: they keep
    the arms' rounds off each other's lock (timed_turns_in_an_hour says how many regular turns share an hour), and
    the budget's cap on a runner's timed commits holds with them or without."""
    return sum(p["sent_twice"] + p["branch"] for p in per_arm) + 3 * home_commits_rolling(release_every_min, unlock_tries)


def commits_per_hour(doc, step_s, release_every_min):
    """A jobs file's steady state against the hub's limit: one commit a save for each training job that runs at a
    time (max_parallel is 1: one), one a mark, and two (the lock and its removal) for each release its runner makes
    from home. {"generations", "marks", "release", "total"}, commits an hour.

    An average of the planned work, not a bound. Left out: a commit sent again after a refusal (a moved head, a
    429, a time-out), whose effect on the hub's count is not known; the creation of a branch (once a branch for a
    VM, so once more after each lost session); an unlock tried again; the release at the campaign's end; the data
    build (eight commits and a branch, up to 32 with every retry) and the seed (three), which come before the
    arms; anything done by hand; and the time saves, evaluations and rungs add to the probe's step times, which
    makes saves rarer. commits_per_hour_bad_model() and commits_rolling_hour() put numbers on the first four."""
    out = {"generations": 0.0, "marks": 0.0, "release": 2 * 60.0 / release_every_min if release_every_min else 0.0}
    for j in doc["jobs"]:
        if "spec" not in j:
            continue
        t = j["spec"]["train"]
        hours = t["steps"] * step_s / 3600
        out["generations"] = max(out["generations"], 3600 / (t["checkpoint_every"] * step_s))
        out["marks"] = max(out["marks"], len(j.get("snapshot_steps", [])) / hours)
    out["total"] = out["generations"] + out["marks"] + out["release"]
    return out


def commits_per_hour_bad_model(doc, step_s, release_every_min):
    """A MODELLED BAD HOUR FROM AVERAGE RATES, not a worst rolling hour: the average saves and marks an hour, each
    sent twice (one retry each, and the hub counts both); the session lost once in the hour, so the new VM creates
    its branch again (one call, tried twice too); and home's worst hour (home_commits_worst: a timed release and
    the release at the campaign's end, each a lock and an unlock sent twice). A real hour holds whole saves
    and may hold a whole mark: commits_rolling_hour(). More retries than one a commit are in neither."""
    p = commits_per_hour(doc, step_s, release_every_min)
    return 2 * (p["generations"] + p["marks"]) + 2 + home_commits_worst(release_every_min)


def commits_rolling_hour(doc, step_s, release_every_min):
    """One ROLLING HOUR of the same jobs file with everything in it at once, in whole commits: the saves that fit
    in an hour, rounded up; one mark; each of those sent twice; the branch created again after a lost session,
    tried twice; and at home home_commits_rolling(): the timed releases at what the runner's budget allows them,
    and the two rounds the end's release has at the most, each a lock and an unlock sent twice. Still a model:
    it takes one retry a commit, and the hub's own count is the only thing that bounds the repository.
    {"saves", "marks", "sent_twice", "branch", "home" (= "home_timed_cap_by_the_budget" + "home_final_rounds", both
    printed), "total"}, and "total_if_unlocks_land_at_once": the same
    hour with every unlock taken at its first try (home's rounds at two commits each)."""
    saves = max(math.ceil(3600 / (j["spec"]["train"]["checkpoint_every"] * step_s)) for j in doc["jobs"] if "spec" in j)
    out = {"saves": saves, "marks": 1, "sent_twice": 2 * (saves + 1), "branch": 2,
           "home": home_commits_rolling(release_every_min),
           "home_timed_cap_by_the_budget": home_timed_cap(),       # 12 - 3 - 2 + 1 + 2: what the budget lets timed rounds send
           "home_final_rounds": FINAL_LEDGER_ROUNDS * (1 + UNLOCK_TRIES)}
    out["total"] = out["sent_twice"] + out["branch"] + out["home"]
    out["total_if_unlocks_land_at_once"] = out["sent_twice"] + out["branch"] + home_commits_rolling(release_every_min, unlock_tries=1)
    return out


def gen_gb(shape):
    """GB of one generation of a shape: weights and two AdamW moments, 4 bytes a parameter."""
    return 3 * PARAMS[shape] * 4 / 1e9


def push_cap_gb(shape, checkpoint_every, step_s, marks, release_every_min, grace_s, losses=ARM_SESSION_LOSSES,
                missed_releases=ARM_MISSED_RELEASES):
    """The most the storage guard counts for ONE campaign on one branch while saves, checks and prunes work, and
    the cap that leaves it PUSH_CAP_HEADROOM. The guard's count is what the last release that RAN measured for the
    campaign plus everything pushed since; only a release that ran sets it back. In generations:
        kept     on the branch: 3 + losses (gens_kept_max);
        removed  removed and still stored when that release measured: it deletes what every ledger older than the
                 grace plus RELEASE_SKEW_S names, so what is younger stays: int((grace + skew) / save) + 1;
        since    pushed since that release, through `missed_releases` turns in a row that did not run (held, failed,
                 an unread ledger; the runner tries such a turn again within minutes, so this is slack), one of
                 them counted twice: int((1 + missed) * release / save) + 2;
    and every mark's weights (a third of a generation each; marks are never removed). Sizing assumptions, not
    bounds: a release measures within one save, saves come at the probe's step time, and the count is this
    campaign's share, each object once, not the account's usage (the caps reserve nothing there). A run whose
    checks stay unread or not whole removes nothing, grows by a generation a save, and is stopped by the cap after
    about (cap - counted) / generation saves, or sooner by runner_multi.py --store-unverified-halt-min."""
    g, save_s, release_s = gen_gb(shape), checkpoint_every * step_s, release_every_min * 60
    out = {"gen_gb": round(g, 3), "kept": gens_kept_max(losses), "removed": int((grace_s + RELEASE_SKEW_S) / save_s) + 1,
           "since": int((1 + missed_releases) * release_s / save_s) + 2, "marks": marks}
    out["counted_gb"] = round((out["kept"] + out["removed"] + out["since"]) * g + marks * g / 3, 1)
    out["suggested_cap_gb"] = math.ceil(out["counted_gb"] / PUSH_CAP_HEADROOM)
    return out


def home_outage_hours(shape, checkpoint_every, step_s, marks, release_every_min, grace_s, cap_gb):
    """How long an arm's courier pushes with home down before it reaches its cap. Only a release from home lowers
    the courier's count, so with home down it grows by every save. Just after a release that ran, the count is
    what the store holds for the campaign: GENS_KEPT generations on the branch, those removed and not yet
    deletable within the grace (as in push_cap_gb), and the marks. It grows by a generation a save. Hours from that
    release to the cap; the outage can begin up to one release interval after it, hence "worst"."""
    g, save_s = gen_gb(shape), checkpoint_every * step_s
    base = (GENS_KEPT + int((grace_s + RELEASE_SKEW_S) / save_s) + 1) * g + marks * g / 3
    rate = 3600 / save_s * g
    h = (cap_gb - base) / rate
    return {"at_release_gb": round(base, 1), "gb_an_hour": round(rate, 2), "hours_to_cap": round(h, 1),
            "hours_to_cap_worst": round(h - release_every_min / 60, 1)}


def rates(release_every_min=None, grace_s=None, losses=ARM_SESSION_LOSSES, missed_releases=ARM_MISSED_RELEASES,
          offsets_min=None):
    """The arms' commit figures and suggested push caps, from the jobs as this file makes them and the probe's
    step times, at a release interval, a grace and the three arms' release offsets (default: the arms' own, as
    arm() writes them). `make_base2.py --rates [--release-every-min M] [--grace-s S] [--release-offsets-min A,B,C]
    [--missed-releases N] [--session-losses N]` prints it; nothing is written. The rolling hour is given with the
    offsets and without. The caps are a suggestion for the lead."""
    step_s = {"N": STEP_S, "S": STEP_S, "C": STEP_S_A}
    per, caps, used, outage, registered = {}, {}, {}, {}, {}
    for k, a in enumerate(("N", "S", "C")):
        doc = files()[f"base2_{a}.json"]
        j = next(j for j in doc["jobs"] if "spec" in j)
        every = release_every_min or doc["store_release_every_min"]
        grace = grace_s or doc.get("store_release_grace_s", ARM_RELEASE_GRACE_S)
        used[a] = {"release_every_min": every, "release_grace_s": grace,
                   "release_offset_min": offsets_min[k] if offsets_min else doc.get("store_release_offset_min", 0)}
        per[a] = {"steady": commits_per_hour(doc, step_s[a], every), "bad_model": commits_per_hour_bad_model(doc, step_s[a], every),
                  "rolling": commits_rolling_hour(doc, step_s[a], every)}
        caps[a] = push_cap_gb(doc["_base2"]["shape"], j["spec"]["train"]["checkpoint_every"], step_s[a],
                              len(j.get("snapshot_steps", [])), every, grace, losses, missed_releases)
        registered[a] = doc.get("store_push_cap_gb") or 0
        outage[a] = home_outage_hours(doc["_base2"]["shape"], j["spec"]["train"]["checkpoint_every"], step_s[a],
                                      len(j.get("snapshot_steps", [])), every, grace, registered[a])
    rolls = [p["rolling"] for p in per.values()]
    every = max(u["release_every_min"] for u in used.values())
    offs = [u["release_offset_min"] for u in used.values()]
    return {"settings": used,
            "commits_an_hour_three_arms": {
                "steady_state_average": round(sum(p["steady"]["total"] for p in per.values()), 1),
                "modelled_bad_hour_from_average_rates": round(sum(p["bad_model"] for p in per.values()), 1),
                "rolling_hour_whole_commits_everything_at_once": commits_rolling_three_arms(rolls, every, offs),
                "rolling_hour_without_the_offsets": commits_rolling_three_arms(rolls, every, [0, 0, 0]),
                "rolling_hour_if_every_unlock_lands_at_once": commits_rolling_three_arms(rolls, every, offs, unlock_tries=1),
                "timed_release_turns_in_one_hour": timed_turns_in_an_hour(every, offs),
                "per_writer_ceilings_summed": 3 * COURIER_COMMIT_BUDGET + 3 * HOME_COMMIT_BUDGET,
                "hub_limit": HUB_COMMITS_PER_HOUR},
            "per_arm": per,
            "generations_kept": {"rule": "three plus one for each session loss between two verified saves",
                                 "losses_taken": losses, "kept": gens_kept_max(losses)},
            "missed_releases_taken": missed_releases,
            "suggested_store_push_cap_gb": {a: c["suggested_cap_gb"] for a, c in caps.items()},
            "counted_gb_three_arms": round(sum(c["counted_gb"] for c in caps.values()), 1),
            "caps_gb_three_arms": sum(c["suggested_cap_gb"] for c in caps.values()),
            "push_cap_sum": caps, "headroom": PUSH_CAP_HEADROOM,
            # The caps the jobs files carry (the registration's, set for home outages), against the account, and
            # how long each arm's courier pushes with home down before it reaches its cap.
            "registered_store_push_cap_gb": registered,
            "registered_caps_gb_three_arms": sum(registered.values()),
            "account_storage_gb": ACCOUNT_STORAGE_GB,
            "home_outage_hours_to_cap": outage}


REH_STEPS = 2000
REH_TAIL_FROM = 0.5          # the main part through step 1,000, the tail from step 1,001
# A save every 200 steps: 3.5 minutes at the probe's 1.053 s a step, 0.84 GB a generation, one commit. The interval is
# set by the storage guard, not by the drills. The guard counts what the store holds for the campaign at each release
# from home and what is pushed after it. A release (every REH_RELEASE_EVERY_MIN minutes) leaves what a ledger younger
# than its grace period names (600 s in the rehearsal), so a removed generation stays stored, and counted, for up to
# the grace and one release interval; on top come the generations each branch keeps (gens_kept_max: three, and one
# more for each session loss between two verified saves; the drill is one), the mark, and what is
# pushed between two releases. At a save every 200 steps that is under REH_PUSH_CAP_GB; at every 100 steps it is not
# (tests/test_base2_jobs.py does the sum).
REH_EVERY = 200
REH_PUSH_CAP_GB = 28         # the courier stops pushing at this many GB counted for the campaign
REH_RELEASE_GRACE_S = 600    # the launch's --store-release-grace-s: the time a restore has to read a removed generation
REH_RELEASE_EVERY_MIN = 20   # the runner's release from home: two commits each, six an hour, of the nine home's budget lets a timed lock have
# One forced stop a runner state (runner_multi.py keeps a single "pod_loss_drill"), so two files. The runner stops the
# VM at its first tick after the store holds the generation of the drill step. A push takes about a minute and a tick
# is 240 s, so the trainer is then between about 50 and 300 steps past the drill step: "a" (drill 400) stops between
# steps 450 and 700, before the switch at 1,000, and "b" (drill 1,200) between 1,250 and 1,500, after it. Where it
# really stopped is read from the VM at the stop and written to the drill's record, with `side_missed` when it is not
# the side meant. Each then resumes from the generation of its drill step or the one after, on its own side.
REH = {"a": {"job": "pt_reh", "drill": 400, "mark": 200, "held": False, "branch": "ckpt-R", "tag": "R"},
       "b": {"job": "pt_reh_b", "drill": 1200, "mark": 1800, "held": True, "branch": "ckpt-Rb", "tag": "Rb"}}
WATCH_LIMIT_S = 7200         # the watcher ends itself by then; `timeout` ends it five minutes later whatever it is doing


def rehearsal(which="a"):
    """Section 5: D1, mixed precision, 8 x 2, T 1,024, 2,000 steps, a save every 200, on a main and a tail corpus of
    different sizes cut from the 50 MB FineWeb-Edu sample, the tail from step 1,001; one forced stop; a mark and its
    rung; then the same 2,000 steps without interruption, not before the rung has ended.
    "a": the stop before the tail switch, and a mark at step 200 whose rung runs beside training, so the next VM
    scores the mark from the store's copy when the stop finds the rung unfinished. "b": the stop after the switch, and
    a mark at step 1,800 whose rung holds training, as the arms' rungs do. A watcher records the trainer's training
    step against the courier's pushed step every 30 seconds, in a file on the VM that comes home with its outputs
    (it commits nothing: in part a's first run its commits used the repository's whole hourly limit).
    The two parts run strictly one after the other, "a" to its end before "b" starts. Storage release runs from
    home with a grace of 600 s; it deletes only what a commit removed and recorded, so it is safe beside the earlier
    attempts' branches on the repo, whose objects it never touches. They share one thing in the store, the corpus
    main:corpus/rehearsal: "a" builds and publishes it, and "b" finds it there and pulls it. Their jobs, branches and
    marks are apart."""
    r = REH[which]
    job = r["job"]
    s = spec("N")
    s["name"] = f"base2_rehearsal_{which}"
    s["out_dir"] = f"/content/chat/{job}"
    s["data"] = {"corpus": f"{DATA}/rehearsal.main.ids", "vocab": V, "vocab_cap": 16384, "T": T, "max_tokens": 0,
                 "anneal": {"corpus": f"{DATA}/rehearsal.tail.ids", "from_fraction": REH_TAIL_FROM}}
    s["train"].update(steps=REH_STEPS, checkpoint_every=REH_EVERY, eval_every=REH_EVERY)  # a validation loss at each mark
    s["train"]["lr_schedule"]["warmup_steps"] = 100
    s["export"] = {"formats": ["safetensors"]}
    c = json.loads(json.dumps(s))
    c["name"], c["out_dir"] = f"base2_rehearsal_{which}_control", f"/content/chat/{job}_control"
    rg = rung(job, r["mark"], r["tag"], SHAPE_D1, [("main", f"{DATA}/rehearsal.main.ids"), ("tail", f"{DATA}/rehearsal.tail.ids")],
              after="data", held=r["held"])
    last_main = int(REH_TAIL_FROM * REH_STEPS)
    return {"_comment": f"Base 2's rehearsal of the store path, part {which} (BASE2_PLAN.md section 5): a forced stop "
                        f"{'before' if r['drill'] < last_main else 'after'} the tail switch. The control starts when the "
                        "interrupted run and its rung have both ended.",
            "_base2": {"rehearsal": which, "steps": REH_STEPS, "last_main_step": last_main, "first_tail_step": last_main + 1,
                       "checkpoint_every": REH_EVERY,
                       "pod_loss_drill_step": r["drill"], "mark": r["mark"], "rung_holds_training": r["held"],
                       "watch": f"ART/watch/watch.jsonl: the watch job's own file on the VM, home with its outputs ({job})",
                       "runs": "alone on the store: part a to its end, then part b",
                       "after": None if which == "a" else "base2_rehearsal.json",
                       "shared_with_the_other_part": ["main:corpus/rehearsal"],
                       "store_release_grace_s": REH_RELEASE_GRACE_S},
            "store_push_cap_gb": REH_PUSH_CAP_GB,
            "store_release_every_min": REH_RELEASE_EVERY_MIN,
            "store_inputs": [SEED_VOCAB],
            "max_parallel": 1, "max_parallel_light": 2,   # the watcher and a rung
            "jobs": [
                {"name": "data", "light": True, "halt_on_failure": True,
                 "cmd": ["bash", "tools/colab_chat/base2_data.sh", "{out}", V, "rehearsal"],
                 "outputs": ["rehearsal.json", "base2_pins_observed.json"]},
                # The watcher writes on the VM only and never opens the store. It ends itself after WATCH_LIMIT_S, and
                # coreutils' timeout ends it five minutes after that: a watcher cannot keep the campaign open.
                {"name": "watch", "light": True, "after": "data",
                 "cmd": ["timeout", "-k", "30", str(WATCH_LIMIT_S + 300),
                         "python3", "tools/colab_chat/base2_corpus.py", "drill-watch",
                         "--chat", "/content/chat", "--job", job, "--out", "{out}", "--interval", "30",
                         "--limit-s", str(WATCH_LIMIT_S)],
                 "outputs": ["watch_summary.json", "watch.jsonl"]},
                {"name": job, "model_name": s["name"], "store_branch": r["branch"], "after": ["data"],
                 "snapshot_steps": [r["mark"]], "pod_loss_drill_step": r["drill"], "halt_on_failure": True, "env": TXN,
                 "spec": s},
                rg,
                {"name": f"{job}_control", "model_name": c["name"], "store_branch": r["branch"] + "c",
                 "after": [job, rg["name"]], "after_terminal": True, "halt_on_failure": True, "env": TXN, "spec": c},
            ]}


def files():
    return {"base2_data.json": data(), "base2_N.json": arm("N"), "base2_S.json": arm("S"), "base2_C.json": arm("C"),
            "base2_rehearsal.json": rehearsal("a"), "base2_rehearsal_b.json": rehearsal("b"),
            "base2_rung_D1.json": rung_d1(), "base2_load_probe.json": load_probe()}


def text(obj):
    return json.dumps(obj, indent=1) + "\n"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--rates" in argv:  # the arms' commit figures and the suggested push caps, at given settings; nothing is written
        def opt(name, kind):
            return kind(argv[argv.index(name) + 1]) if name in argv else None
        kw = {"release_every_min": opt("--release-every-min", float), "grace_s": opt("--grace-s", float),
              "offsets_min": opt("--release-offsets-min", lambda s: [float(x) for x in s.split(",")])}
        for name, key in (("--missed-releases", "missed_releases"), ("--session-losses", "losses")):
            if opt(name, int) is not None:
                kw[key] = opt(name, int)
        print(json.dumps(rates(**kw), indent=1))
        return 0
    check = "--check" in argv
    bad = []
    for name, obj in files().items():
        path = os.path.join(HERE, name)
        if check:
            try:
                same = open(path, encoding="utf-8").read() == text(obj)
            except OSError:
                same = False
            if not same:
                bad.append(name)
        else:
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(text(obj))
    if check:
        print(json.dumps({"check": "equal" if not bad else "different", "files": sorted(files()), "different": bad}))
        return 1 if bad else 0
    print(json.dumps({"written": sorted(files())}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
