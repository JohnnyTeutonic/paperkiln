#!/usr/bin/env python3
"""The jobs files of the transfer programme's Stage 3(a): six 70M bases, each the token ladder's rung N1B (seed 1) or N1Bs50
(seed 50) with its corpus replaced by base 2's corpus N mixed with that run's five-task packet (the author's research tree,
AI_ML/transfer_programme/stage3a/REGISTRATION.md). Claude Opus 5.5, 28 Nov 2026.

    python3 tools/colab_chat/make_tp.py            # writes tp_<run>.json for the six runs
    python3 tools/colab_chat/make_tp.py --check    # compares them with what is on disk; non-zero on any difference

What changes against the reference rung (make_ladder.py and make_base2.py, imported and not edited):
- a data job, tp_data, after the pins job: it checks the uploaded bundle (tp_bundle_<sha256[:12]>.tar.gz) against the
  run's pin in BUNDLES, then runs the bundle's tp_data_vm.sh, which generates the run's packet on the VM with the
  programme's instruments, refuses it unless it equals the packet built at home (the bundle's pins/<run>.json), tokenises it
  with tools/corpus_ids.py and mixes it into N.main.ids and N.tail.ids (both from the store, checked by the pins job) as
  /content/chat/data/TP.main.ids and TP.tail.ids. The runner reruns it on any replacement VM.
- the training job reads TP.main.ids and TP.tail.ids, waits for tp_data, and keeps 15 marks (12 log-spaced from step 1,000,
  the ladder's 10,000 and 58,500, and 51,000 where the tail begins); its names are the run's.
- one rung job, at the end (panels and Carry-5, as the ladder's end rung); the ladder's two mid-run rungs are not run.
- the store's release turn is 40 minutes (W169 holds turn 0 and SYMW's and SAT's runs turn 20).
Shape, seed, steps (60,000), schedule (4,000 warm-up steps, cosine to a tenth of 5e-4, the tail from step 51,001), batch,
precision, save interval, panels, Carry-5, pins, the store's inputs and switches are the reference rung's.
"""
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_base2 as M  # noqa: E402  (base 2's builder, unchanged)
import make_ladder as L  # noqa: E402  (the token ladder's builder, unchanged)

# run -> sha256 of the bundle it runs from (the research tree's AI_ML/transfer_programme/stage3a/bundle.json, which also
# holds the run's home-built packet pins); tp_data refuses any other bytes. A run is rendered only once its bundle exists.
BUNDLES = {"R1A": "96604bfb25755acc1025f01523948bb7a9688d4b880d37bb09bd848ff0a4614a"}
# run -> the reference rung (its trainer seed); the rectangles and packets are in the bundle's design.json and specs/
RUNS = {"R1A": "N1B", "R1B": "N1B", "R1C": "N1B", "R2A": "N1Bs50", "R2B": "N1Bs50", "R2C": "N1Bs50"}
# 12 log-spaced (the simulation's) plus the ladder's 10,000 and 58,500 and the tail's start, 51,000
MARKS = [1000, 1500, 2500, 4000, 6000, 9000, 10000, 13500, 20000, 27500, 37500, 49000, 51000, 58500, 60000]
OFFSET_MIN = 40
DATA_TIMEOUT_S = 5400


def bundle_name(run):
    return f"tp_bundle_{BUNDLES[run][:12]}.tar.gz"


def data_job(run):
    up = "/content/chat/upload"
    b, sha = bundle_name(run), BUNDLES[run]
    script = (f"sha256sum {up}/{b} | tee {{out}}/tp_bundle_pin.txt | grep -q '^{sha} ' && "
              f"rm -rf /content/chat/tp_unpack && mkdir -p /content/chat/tp_unpack && "
              f"tar -xzf {up}/{b} -C /content/chat/tp_unpack stage3a/tp_data_vm.sh && "
              f"timeout -k 60 {DATA_TIMEOUT_S} bash /content/chat/tp_unpack/stage3a/tp_data_vm.sh {run} {{out}} {b}")
    return {"name": "tp_data", "after": ["pins"], "halt_on_failure": True, "cmd": ["sh", "-c", script],
            "outputs": ["tp_bundle_pin.txt", "tp_data.json", "tp_mix.json", "tp_packet.json", "tp_data.log"]}


def arm(run):
    ref_name = RUNS[run]
    ref = L.arm(ref_name)
    seed = L.RUNGS[ref_name]["seed"]
    rjob, job = f"pt_base2_{ref_name}", f"pt_tp_{run}"
    rtrain = next(j for j in ref["jobs"] if j["name"] == rjob)
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
    out = copy.deepcopy(ref)
    out["jobs"] = [M.pins_job("arm-N"), data_job(run), train, end]
    b = out["_base2"]
    b.update(arm=f"TP_{run}", ladder=False, recipe_of=ref_name, rung="1B", slot=None, marks=list(MARKS),
             mark_positions={str(m): m * M.ROWS * M.T for m in MARKS},
             corpus="TP: base 2's N with the run's five-task packet at 15% of each part's training tokens (tp_data)",
             transfer_programme={"stage": "3(a)", "run": run, "bundle": bundle_name(run), "bundle_sha256": BUNDLES[run]})
    out["_comment"] = (f"Transfer programme Stage 3(a), run {run}: the token ladder's {ref_name} with its corpus replaced by "
                       f"N mixed with the run's five-task packet (tp_data), {len(MARKS)} marks and one end rung. Not "
                       "admitted: the launch gate is two reviews and the coordinator's ruling.")
    return L._with_offset(out, OFFSET_MIN)


def files():
    return {f"tp_{r}.json": arm(r) for r in RUNS if r in BUNDLES}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    check = "--check" in argv
    for r, h in BUNDLES.items():
        if r not in RUNS or len(h) != 64 or any(c not in "0123456789abcdef" for c in h):
            raise SystemExit(f"{r}: the bundle pin is not a sha256")
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
