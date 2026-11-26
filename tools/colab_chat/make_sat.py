#!/usr/bin/env python3
"""The jobs file of SAT, "symbols as targets" (the author's research tree,
AI_ML/chat_screens_claude/base2/token_ladder/symbols_as_targets/REGISTRATION.md). Claude Opus 5.5, 26 Nov 2026.

    python3 tools/colab_chat/make_sat.py            # writes sat_SAT.json
    python3 tools/colab_chat/make_sat.py --check    # compares it with what is on disk; non-zero on any difference
    python3 tools/colab_chat/make_sat.py --smoke [--check]   # the CUDA smoke's jobs file, sat_SAT_smoke.json

SAT is the token ladder's rung N1B (make_ladder.py, imported and not edited) with one added training key,
train.symbol_targets (tools/mtstudio.cpp): a role head and a binding-pointer head on the final hidden state, trained on
silver labels with weight w0 * 2^(-step / half_life) to a hard zero at cut_step, and, before cut_step, one row in
`every` read from the owner-flip twin text (the twin rule of the brief of 26 Nov 2026). Shape, seed, schedule, steps,
marks, panels, Carry-5, pins and the store's switches are N1B's. Added: the four SAT files among the store's inputs, a
light job that checks their sha256 (sat_pin) and one that runs the registered implementation check on the VM's CUDA
build before training (sat_check: the forward on 8 fixed windows with labels A, B and none gives bit-identical
logits, MICROTORCH_SAT_CHECK=8), both halting the session on failure and both awaited by training; and the names.
The release turn is 20 minutes, SYMW's choice (the ladder's slot 2), so SAT never shares a turn with an arm on turn 0.

SAT_SHA256 holds the four files' sha256, built at home on 26 Nov 2026 in /mnt/c/ml_artifacts/sat/corpus: sat.labels
and sat_eval.labels from sat_build.sh (its receipt.json), sat_twins.ids and sat_twins.labels from build_twins.sh's
version 3 twins (twins_build.json frames_version 3, audit_passed, its ids_sha256 and labels_sha256). sat_pin fails closed on any
other bytes.
"""
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_base2 as M  # noqa: E402  (base 2's builder, unchanged)
import make_ladder as L  # noqa: E402  (the token ladder's builder, unchanged)

NAME = "SAT"
REF = "N1B"
OFFSET_MIN = 20
OUT_FILE = "sat_SAT.json"
CHECK_WINDOWS = 8

LABELS, EVAL_LABELS, TWIN_IDS, TWIN_LABELS = "sat.labels", "sat_eval.labels", "sat_twins.ids", "sat_twins.labels"
FILES = (LABELS, EVAL_LABELS, TWIN_IDS, TWIN_LABELS)

# ---- the built files' sha256 (sat_build.sh's receipt.json; build_twins.sh's twins_build.json, frames_version 3). ----
PLACEHOLDER = "0" * 64
SAT_SHA256 = {LABELS: "d1585800702b26f8b7fb8f8f62c0f12a905673d8074708e14033e4010f002e72",
              EVAL_LABELS: "9717d3783c9ec3826b2063586f2cd085526447f9f50e2502904348e11cbb55eb",
              TWIN_IDS: "a132ab0323c22bbc97c65f12e1c3dac335c6965bd50acce05b25244ade33ecdc",
              TWIN_LABELS: "e5c3c8f9da051b231eb7ff9d21c1df9d51458fd7eb30007b61d18888580d85ee"}
# ------------------------------------------------------------------------------------------------------------------

SAT_SETS = [
    {"branch": "main", "prefix": "corpus/sat_labels", "to": "chat/data", "files": [LABELS]},
    {"branch": "main", "prefix": "corpus/sat_eval", "to": "chat/data", "files": [EVAL_LABELS]},
    {"branch": "main", "prefix": "corpus/sat_twins", "to": "chat/data", "files": [TWIN_IDS, TWIN_LABELS]},
]


def symbol_targets():
    return {"labels": f"{M.DATA}/{LABELS}", "eval_labels": f"{M.DATA}/{EVAL_LABELS}", "w0": 1.0, "half_life": 2000,
            "cut_step": 13288, "role_classes": 8, "pointer_dim": 64,
            "twins": {"corpus": f"{M.DATA}/{TWIN_IDS}", "labels": f"{M.DATA}/{TWIN_LABELS}", "every": 10}}


def pin_job():
    paths = " ".join(f"{M.DATA}/{f}" for f in FILES)
    greps = " && ".join(f"grep -q '^{SAT_SHA256[f]}  {M.DATA}/{f}$' {{out}}/sat_pin.txt" for f in FILES)
    return {"name": "sat_pin", "light": True, "halt_on_failure": True,
            "cmd": ["sh", "-c", f"sha256sum {paths} | tee {{out}}/sat_pin.txt && {greps}"],
            "outputs": ["sat_pin.txt"]}


def check_job(train_spec):
    spec = copy.deepcopy(train_spec)
    spec["name"] = f"base2_{NAME}_check"
    return {"name": "sat_check", "light": True, "halt_on_failure": True, "after": ["pins", "sat_pin"],
            "cmd": ["python3", "tools/colab_chat/sat_check_vm.py", "--out", "{out}", "--windows", str(CHECK_WINDOWS),
                    "--spec-json", json.dumps(spec, sort_keys=True)],
            "outputs": ["sat_check.json"]}


def arm():
    ref = L.arm(REF)
    out = copy.deepcopy(ref)
    rjob, job = f"pt_base2_{REF}", f"pt_base2_{NAME}"
    jobs = [M.pins_job("arm-N"), pin_job()]
    train = None
    for j in ref["jobs"]:
        if j["name"] == "pins":
            continue
        j = copy.deepcopy(j)
        if j["name"] == rjob:
            j.update(name=job, model_name=f"base2_{NAME}", store_branch=f"ckpt-{NAME}",
                     after=["pins", "sat_pin", "sat_check"])
            j["spec"]["name"] = f"base2_{NAME}"
            j["spec"]["out_dir"] = f"/content/chat/{job}"
            j["spec"]["train"]["symbol_targets"] = symbol_targets()
            train = j
        else:
            j = json.loads(json.dumps(j).replace(REF, NAME))      # every name of a rung job (rung_N1B_*, BASE2_N1B, ...)
        jobs.append(j)
    jobs.insert(2, check_job(train["spec"]))
    out["jobs"] = jobs
    out["store_inputs"] = ref["store_inputs"] + SAT_SETS
    b = out["_base2"]
    b.update(arm=NAME, rung="1B", slot=None,
             symbol_targets={**symbol_targets(), "sha256": dict(SAT_SHA256),
                             "hashes_are_placeholders": any(v == PLACEHOLDER for v in SAT_SHA256.values()),
                             "check_windows": CHECK_WINDOWS})
    b["corpus"] = "N, with SAT's twin rows before the cut"
    out["_comment"] = (f"SAT: the token ladder's {REF} with train.symbol_targets (role and binding-pointer heads, "
                       "w0 1, half-life 2,000, cut at 13,288; twin rows one in ten before the cut); nothing else is "
                       "changed. Not admitted: the launch gate is two reviews and the lead's ruling.")
    return L._with_offset(out, OFFSET_MIN)


# ---- The CUDA smoke (the author's research tree, symbols_as_targets/SMOKE.md): SAT's real spec and inputs, short. ----
SMOKE_FILE = "sat_SAT_smoke.json"
SMOKE_NAME = "SATsmoke"
SMOKE_NAME_W = "SATWsmoke"     # the weight-on path's names
SMOKE = {"steps": 300, "cut_step": 200, "eval_every": 50, "checkpoint_every": 50, "drill_step": 100}


def smoke_spec(spec):
    """SAT's training spec with the smoke's length, cut, eval and checkpoint cadence, and its own names. A checkpoint
    every 50 (not 100) so that the heads, which travel with checkpoints before the cut only, are saved at 150, after
    the forced session loss at 100 and before the cut at 200."""
    s = copy.deepcopy(spec)
    s["name"] = f"base2_{SMOKE_NAME}"
    s["out_dir"] = f"/content/chat/pt_base2_{SMOKE_NAME}"
    t = s["train"]
    t.update(steps=SMOKE["steps"], eval_every=SMOKE["eval_every"], checkpoint_every=SMOKE["checkpoint_every"])
    t["symbol_targets"]["cut_step"] = SMOKE["cut_step"]
    return s


def smoke():
    """The smoke's jobs file, on the trainer SAT and SATW share (REGISTRATION.md section 17): both paths on one VM.
    pins and sat_pin; sat_check on SAT's smoke spec (weight off) and on SATW's (weight on, make_satw.RECENCY_WEIGHT); a
    watcher of the training job's generations; the training job on SATW's smoke spec (300 steps, cut at 200, the
    runner's forced session loss once the store holds step 100); then sat_smoke_vm.py checks on the same VM: the
    uninterrupted weight-on reference, SAT's spec uninterrupted (weight off), the comparator build at the ladder's
    launch commit and the identity runs after the cut for both specs. No rung jobs."""
    import make_satw as SW   # here, not at the top: make_satw imports this module
    full = arm()
    train = next(j for j in full["jobs"] if j["name"] == f"pt_base2_{NAME}")
    off = smoke_spec(train["spec"])
    on = copy.deepcopy(off)
    on["name"], on["out_dir"] = f"base2_{SMOKE_NAME_W}", f"/content/chat/pt_base2_{SMOKE_NAME_W}"
    on["train"]["symbol_targets"]["recency_weight"] = {"nearest_right": SW.RECENCY_WEIGHT}
    job = f"pt_base2_{SMOKE_NAME_W}"
    jobs = [j for j in full["jobs"] if j["name"] in ("pins", "sat_pin")]
    jobs.append(check_job(off))
    chk_w = check_job(on)
    chk_w["name"] = "sat_check_w"
    chk_w["after"] = ["pins", "sat_pin", "sat_check"]
    jobs.append(chk_w)
    watch = {"name": "smoke_watch", "light": True, "after": ["sat_check", "sat_check_w"],
             "cmd": ["timeout", "-k", "30", "7000", "python3", "tools/colab_chat/sat_smoke_vm.py", "watch",
                     "--job-dir", on["out_dir"], "--status", f"/content/chat/STATUS_{job}", "--out", "{out}",
                     "--interval", "5", "--limit-s", "6900"],
             "outputs": ["gens.jsonl"]}
    tj = copy.deepcopy(train)
    tj.update(name=job, model_name=on["name"], store_branch=f"ckpt-{SMOKE_NAME_W}", spec=on, snapshot_steps=[],
              pod_loss_drill_step=SMOKE["drill_step"], after=["pins", "sat_pin", "sat_check", "sat_check_w"])
    tj["env"] = {**tj.get("env", {}), "MICROTORCH_STEP_TIMING": "1"}
    checks = {"name": "smoke_checks", "light": True, "after": [job, "smoke_watch"],
              "env": {"MICROTORCH_CHECKPOINT_TXN": "1"},
              "cmd": ["timeout", "-k", "60", "3600", "python3", "tools/colab_chat/sat_smoke_vm.py", "checks",
                      "--main-dir", on["out_dir"], "--spec-json", json.dumps(on, sort_keys=True),
                      "--weight-off-spec-json", json.dumps(off, sort_keys=True), "--out", "{out}",
                      "--watch", "/content/chat/smoke_watch/gens.jsonl"],
              "outputs": ["smoke_vm.json", "events_ref.jsonl", "events_ref_off.jsonl", "events_A.jsonl", "events_A_off.jsonl",
                          "events_B.jsonl", "base_build.log", "ref.log", "ref_off.log", "A.log", "A_off.log", "B.log"]}
    jobs += [watch, tj, checks]
    out = {k: v for k, v in full.items() if k != "jobs"}
    out["jobs"] = jobs
    out["store_push_cap_gb"] = 20
    out["max_parallel_light"] = 2
    out["_base2"] = {**full["_base2"], "smoke": {**SMOKE, "recency_weight": SW.RECENCY_WEIGHT}, "marks": []}
    out["_comment"] = ("The CUDA smoke of SAT and SATW on their shared trainer: SAT's inputs, 300 steps, cut at 200, eval and "
                       "checkpoint every 50; sat_check on both specs; SATW's spec (weight on) trained with a forced "
                       "session loss once the store holds step 100; SAT's spec (weight off), the references, the "
                       "comparator and the identity runs of both specs after the cut on the same VM (sat_smoke_vm.py).")
    return out

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    name = SMOKE_FILE if "--smoke" in argv else OUT_FILE
    path = os.path.join(HERE, name)
    text = M.text(smoke() if "--smoke" in argv else arm())
    if "--check" in argv:
        try:
            same = open(path, encoding="utf-8").read() == text
        except OSError:
            same = False
        print(json.dumps({"check": "equal" if same else "different", "file": name}))
        return 0 if same else 1
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(json.dumps({"written": name}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
