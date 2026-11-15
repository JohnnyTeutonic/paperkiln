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
import hashlib
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


def strict_events(path, ended):
    """The job's events, failing closed (launch review round 1, finding 15): a line that does not parse is tolerated only as the last
    line of a log still being written; any other unparseable line, or a read error, raises ValueError (the check is then unread)."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")
    out = []
    for k, x in enumerate(lines):
        if not x.strip():
            continue
        try:
            out.append(json.loads(x))
        except ValueError:
            last = all(not y.strip() for y in lines[k + 1:])
            if ended or not last:
                raise ValueError(f"{path}: line {k + 1} does not parse")
    return out


def events_digest(ev, step):
    """The identity of the evidence a health record certifies (launch round 2): sha256 of the step and eval records with step <= step,
    a record written twice counting once (its last), in step order, as canonical JSON. score_b2cue.events_digest is the same."""
    by = {}
    for e in ev:
        if e.get("event") in ("step", "eval") and isinstance(e.get("step"), int) and e["step"] <= step:
            by[(e["step"], e["event"])] = e
    text = json.dumps([by[k] for k in sorted(by)], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def file_sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def job_ended(chat, job):
    p = f"{chat}/STATUS_{job}"
    return os.path.exists(p) and open(p).read().strip() in ("FINISHED", "FAILED")


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
            ev = strict_events(f"{a.chat}/{a.job}/events.jsonl", job_ended(a.chat, a.job))
            ref_ev = ev if self_ref else strict_events(f"{a.chat}/{a.ref}/events.jsonl", job_ended(a.chat, a.ref))
            verdict, why = A.health(ev, ref_ev, a.step, self_ref, twin)
            if verdict != "unread" or time.time() - t0 > WAIT_S:
                break
            time.sleep(POLL_S)
        out["train"] = A.train_metrics(ev, a.step)
        out["val_loss"] = A.val_at(ev, a.step)
        out["ref"] = {"job": a.ref, "val_loss": A.val_at(ref_ev, a.step)}
    except Exception as e:  # noqa: BLE001  (fail-closed)
        verdict, why = "unread", f"the health check failed: {type(e).__name__}: {e}"
    # a positive record (launch review round 1, finding 14): consumers require health == "ok" for this job and step, and the spec and
    # event prefix it certifies (round 2)
    out["health"] = verdict if verdict in ("ok", "stop") else "unread"
    try:
        out["spec_sha256"] = file_sha(f"{a.chat}/{a.job}_spec.json")
        out["events_digest"] = events_digest(ev, a.step)
    except Exception as e:  # noqa: BLE001
        out["health"], out["health_unread"] = "unread", f"the evidence identity cannot be read: {type(e).__name__}"
    if verdict == "stop":
        out["health_stop"] = why
    elif verdict != "ok":
        out["health_unread"] = why or "no verdict"
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
    """An SFT1500 job's watch: non-finite stop at once; relayed losses at the --at steps. Fail-closed (launch review round 1): any
    error reading the log is published as health "unread" before exiting; "ok" only when the job FINISHED with every step 1 .. steps
    finite, every held-out evaluation finite and all three readouts present and finite."""
    out = {"job": a.job, "producer": "b2cue_early.py --watch", "at": {}, "steps": a.steps, "health": "running"}
    path = f"{a.out}/watch.json"
    try:
        return watch_loop(a, out, path)
    except Exception as e:  # noqa: BLE001
        out["health"] = "unread"
        out["health_unread"] = f"the watch failed: {type(e).__name__}: {e}"
        write_json(path, out)
        return 0


def watch_loop(a, out, path):
    at = [int(x) for x in a.at.split(",") if x]
    seen = 0
    while True:
        ev = strict_events(f"{a.chat}/{a.job}/events.jsonl", job_ended(a.chat, a.job))
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
            out["health"] = "stop"
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
            return 1   # a stop is not a healthy run (launch round 2)
        status = open(f"{a.chat}/STATUS_{a.job}").read().strip() if os.path.exists(f"{a.chat}/STATUS_{a.job}") else "NONE"
        if status in ("FINISHED", "FAILED"):
            steps = {e.get("step") for e in st}
            out["status"] = status
            out["complete"] = steps == set(range(1, a.steps + 1))
            missing = [str(s) for s in at if str(s) not in out["at"]]
            if missing:
                out["readout_missing"] = missing
            fin = all(A.finite(v) for r in out["at"].values() for v in r.values())
            out["health"] = "ok" if (status == "FINISHED" and out["complete"] and not missing and fin) else "unread"
            if out["health"] != "ok":
                out["health_unread"] = f"status {status}, complete {out['complete']}, missing readouts {missing}, finite readouts {fin}"
            out["spec_sha256"] = file_sha(f"{a.chat}/{a.job}_spec.json") if os.path.exists(f"{a.chat}/{a.job}_spec.json") else None
            out["events_digest"] = events_digest(ev, a.steps)
            write_json(path, out)
            return 0 if out["health"] == "ok" else 1
        if len(st) != seen:
            seen = len(st)
            write_json(path, out)
        time.sleep(POLL_S)


def repro(a):
    """Section 2's reproduction check (reported): the job's step losses at steps 1 .. --steps against the reference events' (the
    last record of a step counts), each within 1% relative; a missing or non-finite step is recorded, never a pass."""
    def losses(path):
        by = {}
        for e in A.events(path):
            if e.get("event") == "step" and isinstance(e.get("step"), int):
                by[e["step"]] = e.get("loss")
        return by
    mine, ref = losses(f"{a.chat}/{a.job}/events.jsonl"), losses(a.ref)
    rows, bad = {}, []
    for s in range(1, a.steps + 1):
        x, y = mine.get(s), ref.get(s)
        if not (A.finite(x) and A.finite(y)):
            bad.append(s)
            continue
        rows[str(s)] = abs(x - y) / max(abs(y), 1e-12)
        if rows[str(s)] > 0.01:
            bad.append(s)
    out = {"producer": "b2cue_early.py --repro", "job": a.job, "steps": a.steps, "tolerance": 0.01, "max_rel": max(rows.values(), default=None),
           "failed_steps": bad, "pass": not bad, "reported_only": True}
    write_json(f"{a.out}/repro.json", out)
    print(f"REPRO {a.job}: {'pass' if not bad else 'FAIL'} (max relative {out['max_rel']})", flush=True)
    return 0


def barrier(a):
    """Lifts when every listed job is marked home; after --timeout-h hours it records "barrier not lifting" and exits 1, so the extra
    seeds never start (their jobs are skipped) and the lead closes the campaign with cause "barrier" (launch round 2)."""
    names = [n for n in a.jobs.split(",") if n]
    t0 = time.time()
    while True:
        missing = [n for n in names if not os.path.exists(f"{a.chat}/HOME_{n}")]
        if missing and time.time() - t0 > a.timeout_h * 3600:
            write_json(f"{a.out}/barrier.json", {"producer": "b2cue_early.py --barrier", "jobs": names, "lifted": False,
                                                 "not_home": missing, "why": "barrier not lifting"})
            print(f"BARRIER NOT LIFTING: {len(missing)} jobs not home", flush=True)
            return 1
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
    ap.add_argument("--repro", action="store_true")
    ap.add_argument("--timeout-h", type=float, default=3.0, help="barrier: hours before it records not lifting")
    ap.add_argument("--jobs", default="")
    a = ap.parse_args()
    if a.barrier:
        return barrier(a)
    if a.repro:
        return repro(a)
    if a.watch:
        return watch(a)
    if not (a.job and a.step and a.ref):
        sys.exit("--job, --step and --ref are needed")
    return early(a)


if __name__ == "__main__":
    sys.exit(main())
