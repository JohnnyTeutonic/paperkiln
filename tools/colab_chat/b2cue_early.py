#!/usr/bin/env python3
"""A32's early readouts, health stop, SFT1500 watch and extra-seed barrier (b2cue; REGISTRATION_DRAFT.md section 8, frozen at eef513ad).
Claude Opus 5.5, 15 Nov 2026. Three modes, one file:

  (default)   a rung job at step 128, 256 or 512 of a 1,024-step sweep or arm job: A31's health rule (a31_early.health, unchanged:
              fail-closed; a non-finite SFT or twin loss stops; at 25% a held-out SFT loss more than 10% above the reference's at step
              256 stops; a reference never compares with itself; an unread check is never a pass), the decision written before any kill.
              Then the trainer's metrics at the cut-off (a31_early.train_metrics) and the held-out SFT loss. In C0's sweep only
              (--score-timeout), also the twin-scored readout on the early-readout set (all 500 of S_in and the first 300 held-out-family
              triplets of S_hf_train): A31's readout measures per set. With --no-score (C1 and C2), no scoring: the departure the frozen
              design declares. P0 is its own reference (the non-finite leg only).
  --watch     an SFT1500 job's watch, from the gate to the job's end: every step's SFT loss and every held-out loss finite, or the
              job is stopped at once (SIGTERM) and the stop recorded; the logged SFT loss and held-out SFT loss at steps 188, 375, 750
              relayed as the job's early readout (watch.json, and one WATCH line each). A missing step is recorded, never a pass.
  --barrier   the extra seeds' barrier (section 7): exits 0 when every listed job has ended and the runner has marked its files home
              (/content/chat/HOME_<job>, runner_multi.py's home markers), so a cap stop during the extra seeds loses no three-seed file.

    python3 tools/colab_chat/b2cue_early.py --job J --step S --ref REF --out DIR [--no-score | --score-timeout S] [--chat C] [--bin B]
    python3 tools/colab_chat/b2cue_early.py --watch --job sft_X --out DIR --at 188,375,750 --steps 1500 [--chat C]
    python3 tools/colab_chat/b2cue_early.py --barrier --out DIR --jobs a,b,c [--chat C]
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import a31_early as A  # noqa: E402  (health, train_metrics, val_at, events, live_pid, finite: A31's, unchanged)

WAIT_S = 600
POLL_S = float(os.environ.get("B2CUE_POLL_S", "20"))   # tests poll faster


def write_json(path, obj):
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def early(a):
    up = f"{a.chat}/upload"
    self_ref = a.ref == a.job
    out = {"job": a.job, "step": a.step, "producer": "b2cue_early.py", "early": {}, "scored": not a.no_score}

    def write():
        try:
            out["summary"] = A.summary_line(a.job, a.step, out)
        except Exception as e:  # noqa: BLE001
            out["summary"] = f"{a.job} step {a.step}; summary failed: {type(e).__name__}"
        write_json(f"{a.out}/early.json", out)

    try:
        twin = "twin_loss" in json.load(open(f"{a.chat}/{a.job}_spec.json")).get("train", {})
        t0 = time.time()
        while True:
            ev = A.events(f"{a.chat}/{a.job}/events.jsonl")
            ref_ev = ev if self_ref else A.events(f"{a.chat}/{a.ref}/events.jsonl")
            verdict, why = A.health(ev, ref_ev, a.step, self_ref, twin)
            if verdict != "unread" or time.time() - t0 > WAIT_S:
                break
            time.sleep(POLL_S)
        out["train"] = A.train_metrics(ev, a.step)
        out["val_loss"] = A.val_at(ev, a.step)
        out["ref"] = {"job": a.ref, "val_loss": A.val_at(ref_ev, a.step)}
    except Exception as e:  # noqa: BLE001  (fail-closed)
        verdict, why = "unread", f"the health check failed: {type(e).__name__}: {e}"
    if verdict == "stop":
        out["health_stop"] = why
    elif verdict == "unread":
        out["health_unread"] = why
    write()
    if verdict == "stop":
        try:
            pid = A.live_pid(a.chat, a.job)
            out["stopped_pid"] = pid
            if pid:
                os.kill(pid, signal.SIGTERM)
                out["kill"] = "SIGTERM sent"
            else:
                out["kill"] = "no live training process (it had ended)"
        except OSError as e:
            out["kill"] = f"not sent: {type(e).__name__}: {e}"
        write()
    if not a.no_score:
        try:
            scored_readout(a, up, out)
        except Exception as e:  # noqa: BLE001
            out["readout_failed"] = f"{type(e).__name__}: {e}"
        write()
    print(out["summary"], flush=True)
    return 0


def scored_readout(a, up, out):
    sys.path.insert(0, up)
    import b2cue_readout as R   # A31's readout.py, uploaded and hash-pinned
    m = f"{a.out}/m"
    os.makedirs(m, exist_ok=True)
    link = f"{m}/{a.job}.safetensors"
    if os.path.lexists(link):
        os.remove(link)
    os.symlink(f"{a.chat}/{a.job}/rung_{a.step}/model.safetensors", link)
    spec = json.load(open(f"{a.chat}/{a.job}_spec.json"))
    spec["out_dir"] = m
    json.dump(spec, open(f"{m}/spec.json", "w"), indent=1)
    try:
        rc = subprocess.run([a.bin, "score", m, "--in", f"{up}/b2cue_early_req.jsonl", "--out", f"{a.out}/scores.jsonl",
                             "--batch-tokens", "8192"], timeout=a.score_timeout).returncode
    except subprocess.TimeoutExpired:
        rc = "timeout"
    out["score_rc"] = rc
    if rc == 0:
        man = json.load(open(f"{up}/b2cue_early_req.manifest.json"))
        trips = {n: R.read_jsonl(f"{up}/b2cue_{n}.jsonl") for n in man["sets"]}
        try:
            rows = R.validated_sets(man, trips, R.read_jsonl(f"{a.out}/scores.jsonl"))
            out["early"] = {n: R.summarise(rows[n].values()) for n in man["sets"]}
        except ValueError as e:
            out["score_invalid"] = str(e)


def watch(a):
    """An SFT1500 job's watch: non-finite stop at once; relayed losses at the --at steps."""
    at = [int(x) for x in a.at.split(",") if x]
    out = {"job": a.job, "producer": "b2cue_early.py --watch", "at": {}, "steps": a.steps}
    path = f"{a.out}/watch.json"
    seen = 0
    while True:
        ev = A.events(f"{a.chat}/{a.job}/events.jsonl")
        st = [e for e in ev if e.get("event") == "step"]
        bad = [e.get("step") for e in st if not A.finite(e.get("loss"))]
        badv = [e.get("step") for e in ev if e.get("event") == "eval" and not A.finite(e.get("val_loss"))]
        for s in at:
            if str(s) not in out["at"]:
                v = A.val_at(ev, s)
                sl = [e.get("loss") for e in st if e.get("step") == s]
                if v is not None or sl:
                    if v is not None and sl:
                        out["at"][str(s)] = {"sft_loss": sl[-1], "val_loss": v}
                        print(f"WATCH {a.job} step {s}: sft {sl[-1]} val {v}", flush=True)
        if bad or badv:
            out["health_stop"] = f"non-finite SFT loss at steps {bad[:5]} or held-out loss at {badv[:5]}"
            write_json(path, out)
            pid = A.live_pid(a.chat, a.job)
            out["stopped_pid"] = pid
            if pid:
                try:
                    os.kill(pid, signal.SIGTERM)
                    out["kill"] = "SIGTERM sent"
                except OSError as e:
                    out["kill"] = f"not sent: {e}"
            write_json(path, out)
            return 0
        status = open(f"{a.chat}/STATUS_{a.job}").read().strip() if os.path.exists(f"{a.chat}/STATUS_{a.job}") else "NONE"
        if status in ("FINISHED", "FAILED"):
            steps = {e.get("step") for e in st}
            out["status"] = status
            out["complete"] = steps == set(range(1, a.steps + 1))
            missing = [str(s) for s in at if str(s) not in out["at"]]
            if missing:
                out["readout_missing"] = missing
            write_json(path, out)
            return 0
        if len(st) != seen:
            seen = len(st)
            write_json(path, out)
        time.sleep(POLL_S)


def barrier(a):
    names = [n for n in a.jobs.split(",") if n]
    t0 = time.time()
    while True:
        missing = [n for n in names if not os.path.exists(f"{a.chat}/HOME_{n}")]
        if not missing:
            write_json(f"{a.out}/barrier.json", {"producer": "b2cue_early.py --barrier", "jobs": names, "lifted_after_s": round(time.time() - t0)})
            print(f"barrier lifted: {len(names)} jobs home", flush=True)
            return 0
        time.sleep(POLL_S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job")
    ap.add_argument("--step", type=int)
    ap.add_argument("--ref")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chat", default="/content/chat")
    ap.add_argument("--bin", default="/content/pk/build_cuda/mtstudio")
    ap.add_argument("--no-score", action="store_true")
    ap.add_argument("--score-timeout", type=int, default=2400)
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--at", default="188,375,750")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--barrier", action="store_true")
    ap.add_argument("--jobs", default="")
    a = ap.parse_args()
    if a.barrier:
        return barrier(a)
    if a.watch:
        return watch(a)
    if not (a.job and a.step and a.ref):
        sys.exit("--job, --step and --ref are needed")
    return early(a)


if __name__ == "__main__":
    sys.exit(main())
