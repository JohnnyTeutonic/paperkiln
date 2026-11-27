#!/usr/bin/env python3
"""The jobs files of the transfer programme's Stage 3(a): six 70M bases, each the token ladder's rung N1B (seed 1) or N1Bs50
(seed 50) with its corpus replaced by base 2's corpus N mixed with that run's five-task packet (the author's research tree,
AI_ML/transfer_programme/stage3a/REGISTRATION.md). Claude Opus 5.5, 28 Nov 2026.

    python3 tools/colab_chat/make_tp.py            # writes tp_<run>_data.json and tp_<run>.json for the runs pinned below
    python3 tools/colab_chat/make_tp.py --check    # compares them with what is on disk; non-zero on any difference

Two campaigns a run:
- tp_<run>_data.json, the first build (data only): the ladder's pins job and the data job tp_data in build mode. tp_data
  checks the uploaded bundle (tp_bundle_<sha256[:12]>.tar.gz) against DATA_BUNDLES, then runs the bundle's tp_data_vm.sh,
  which generates the run's packet on the VM with the programme's instruments, refuses it unless it equals the packet built
  at home (pins/<run>.json), tokenises it with tools/corpus_ids.py, mixes it into N.main.ids and N.tail.ids (from the store,
  checked by the pins job) and brings the corpus hashes home. No training.
- tp_<run>.json, training: the same data job in train mode from TRAIN_BUNDLES's bundle, which also holds the first build's
  corpus hashes (pins/<run>_corpus.json): on every VM the corpus must equal them byte for byte before the training job can
  start. The training job reads TP.main.ids and TP.tail.ids and keeps 15 marks (12 log-spaced from step 1,000, the ladder's
  10,000 and 58,500, and 51,000 where the tail begins); one end rung (panels and Carry-5, as the ladder's); the ladder's
  mid-run rungs are not run. The store's release turn is 40 minutes (W169 holds turn 0, SYMW's and SAT's runs turn 20).
Shape, seed, steps (60,000), schedule (4,000 warm-up steps, cosine to a tenth of 5e-4, the tail from step 51,001), batch,
precision, save interval, panels, Carry-5, pins, the store's inputs and switches are the reference rung's (make_ladder.py
and make_base2.py, imported and not edited).
"""
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_base2 as M  # noqa: E402  (base 2's builder, unchanged)
import make_ladder as L  # noqa: E402  (the token ladder's builder, unchanged)

# run -> sha256 of the bundle each campaign runs from (the research tree's AI_ML/transfer_programme/stage3a/bundles/);
# tp_data refuses any other bytes. A campaign is rendered only once its bundle exists.
DATA_BUNDLES = {"R1A": "07851737417669cbd2a3b94604c8b608dff911c73897d35f58610fd0a4e2ff1b"}
TRAIN_BUNDLES = {}
# run -> the reference rung (its trainer seed); the rectangles and packets are in the bundle's design.json and specs/
RUNS = {"R1A": "N1B", "R1B": "N1B", "R1C": "N1B", "R2A": "N1Bs50", "R2B": "N1Bs50", "R2C": "N1Bs50"}
# 12 log-spaced (the simulation's) plus the ladder's 10,000 and 58,500 and the tail's start, 51,000
MARKS = [1000, 1500, 2500, 4000, 6000, 9000, 10000, 13500, 20000, 27500, 37500, 49000, 51000, 58500, 60000]
OFFSET_MIN = 40
DATA_TIMEOUT_S = 5400


def bundle_name(sha):
    return f"tp_bundle_{sha[:12]}.tar.gz"


def data_job(run, sha, mode):
    up = "/content/chat/upload"
    b = bundle_name(sha)
    script = (f"sha256sum {up}/{b} | tee {{out}}/tp_bundle_pin.txt | grep -q '^{sha} ' && "
              f"rm -rf /content/chat/tp_unpack && mkdir -p /content/chat/tp_unpack && "
              f"tar -xzf {up}/{b} -C /content/chat/tp_unpack stage3a/tp_data_vm.sh && "
              f"timeout -k 60 {DATA_TIMEOUT_S} bash /content/chat/tp_unpack/stage3a/tp_data_vm.sh {run} {{out}} {b} {mode}")
    return {"name": "tp_data", "after": ["pins"], "halt_on_failure": True, "cmd": ["sh", "-c", script],
            "outputs": ["tp_bundle_pin.txt", "tp_data.json", "tp_mix.json", "tp_packet.json", "tp_data.log"]}


def _base(run, sha, what):
    ref_name = RUNS[run]
    out = copy.deepcopy(L.arm(ref_name))
    b = out["_base2"]
    b.update(arm=f"TP_{run}", ladder=False, recipe_of=ref_name, rung="1B", slot=None,
             corpus="TP: base 2's N with the run's five-task packet at 15% of each part's training tokens (tp_data)",
             transfer_programme={"stage": "3(a)", "run": run, "campaign": what, "bundle": bundle_name(sha),
                                 "bundle_sha256": sha})
    return out


def data_arm(run):
    sha = DATA_BUNDLES[run]
    out = _base(run, sha, "first build, data only")
    out["jobs"] = [M.pins_job("arm-N"), data_job(run, sha, "build")]
    out["_base2"].update(marks=[], mark_positions={})
    out["_comment"] = (f"Transfer programme Stage 3(a), run {run}, first build: the pins job and the data job (build mode) "
                       "only; no training. Not admitted: the launch gate is two reviews and the coordinator's ruling.")
    return L._with_offset(out, OFFSET_MIN)


def arm(run):
    sha = TRAIN_BUNDLES[run]
    ref_name = RUNS[run]
    out = _base(run, sha, "training")
    seed = L.RUNGS[ref_name]["seed"]
    rjob, job = f"pt_base2_{ref_name}", f"pt_tp_{run}"
    rtrain = next(j for j in out["jobs"] if j["name"] == rjob)
    train = copy.deepcopy(rtrain)
    train.update(name=job, model_name=f"tp_{run}", store_branch=f"ckpt-tp-{run}", after=["pins", "tp_data"],
                 snapshot_steps=list(MARKS))
    s = train["spec"]
    s["name"] = f"tp_{run}"
    s["out_dir"] = f"/content/chat/{job}"
    s["data"]["corpus"] = f"{M.DATA}/TP.main.ids"
    s["data"]["anneal"]["corpus"] = f"{M.DATA}/TP.tail.ids"
    panels = [(p, f"{M.DATA}/panel_{p}.ids") for p in M.PANELS]
    end = M.rung(job, MARKS[-1], f"TP_{run}", M.SHAPE_D1, panels, of=False, after=[job], seed=seed)
    out["jobs"] = [M.pins_job("arm-N"), data_job(run, sha, "train"), train, end]
    out["_base2"].update(marks=list(MARKS), mark_positions={str(m): m * M.ROWS * M.T for m in MARKS})
    out["_comment"] = (f"Transfer programme Stage 3(a), run {run}: the token ladder's {ref_name} with its corpus replaced by "
                       f"N mixed with the run's five-task packet (tp_data, train mode: the first build's bytes), "
                       f"{len(MARKS)} marks and one end rung. Not admitted: the launch gate is two reviews and the "
                       "coordinator's ruling.")
    return L._with_offset(out, OFFSET_MIN)


def files():
    out = {f"tp_{r}_data.json": data_arm(r) for r in RUNS if r in DATA_BUNDLES}
    out.update({f"tp_{r}.json": arm(r) for r in RUNS if r in TRAIN_BUNDLES})
    return out


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    check = "--check" in argv
    for r, h in list(DATA_BUNDLES.items()) + list(TRAIN_BUNDLES.items()):
        if r not in RUNS or len(h) != 64 or any(c not in "0123456789abcdef" for c in h):
            raise SystemExit(f"{r}: the bundle pin is not a sha256")
    if set(TRAIN_BUNDLES) - set(DATA_BUNDLES):
        raise SystemExit("a run's training campaign needs its first build first")
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
