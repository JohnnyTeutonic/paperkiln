#!/usr/bin/env python3
"""A31's early readouts and health stop (lane PROTOCOL.md section 50), a rung job on the pod. Claude Opus 5.5, 13 Nov 2026; revised
after the launch reviews of 13 Nov 2026 (health read first and fail-closed; NaN written as null counts as non-finite; timeouts).

At each cut-off (steps 128, 256 and 512 of 1,024: 12.5%, 25% and 50%) jobs_vm.py snapshots a training job's weights; this job
  1. reads the health rule first (registered before launch): a non-finite SFT or twin loss in the job's log up to the cut-off
     (nlohmann writes NaN and infinity as null, which counts as non-finite) stops the job; at the 25% cut-off, the job's held-out
     SFT loss more than 10% above the reference run's at step 256 (D0 for the sweep, P at the same seed for M and N) stops it. A
     reference is never compared with itself (D0 and P have only the non-finite leg). A held-out loss the rule needs that is not in
     the logs within 10 minutes is recorded as `health_unread`, which the verdict treats as a failed run. A stop ends the job's
     training process (SIGTERM) and is recorded as `health_stop`; the verdict treats it as a failed run whether or not the kill
     landed (the job may have finished first).
  2. then scores the snapshot on the early requests only (the dev twins and a held-out-family dev set kept apart from every final
     test set), with a 15-minute timeout, and writes early.json: both right per token, the decisive, nuisance and selectivity shifts
     per token on each early set (a31_readout.py's measures, validated), the training-step metrics up to the cut-off and the
     held-out SFT loss at the cut-off. Reported, not judged.

    python3 tools/colab_chat/a31_early.py --job J --step S --ref REF --out DIR [--chat /content/chat] [--bin MTSTUDIO]
"""
import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time

HEALTH_REL = 0.10
HEALTH_STEP = 256
WAIT_S = 600
SCORE_TIMEOUT_S = 900


def events(path):
    if not os.path.exists(path):
        return []
    out = []
    for x in open(path):
        try:
            out.append(json.loads(x))
        except ValueError:   # a line being written
            pass
    return out


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def val_at(ev, step):
    """The held-out SFT loss at a step: a number, None if absent, or the string 'nonfinite'."""
    for e in ev:
        if e.get("event") == "eval" and e.get("step") == step:
            return e["val_loss"] if finite(e.get("val_loss")) else "nonfinite"
    return None


def train_metrics(ev, step):
    st = [e for e in ev if e.get("event") == "step" and finite(e.get("step")) and e["step"] <= step]
    if not st:
        return {}
    nonfinite = any(not finite(e.get("loss")) or ("twin" in e and not finite((e.get("twin") or {}).get("loss"))) for e in st)
    window = [e for e in st[-32:] if finite(e.get("loss"))]
    out = {"step": st[-1]["step"], "nonfinite": nonfinite}
    if window:
        out["sft_loss"] = window[-1]["loss"]
        out["sft_loss_mean32"] = sum(e["loss"] for e in window) / len(window)
    tw = [e["twin"] for e in window if isinstance(e.get("twin"), dict)]
    for k in ("loss", "d_a", "d_b", "d_n", "active"):
        vals = [t[k] for t in tw if finite(t.get(k))]
        if vals:
            out[f"twin_{k}_mean32"] = sum(vals) / len(vals)
    return out


def health(ev, ref_ev, step, self_ref, twin=False):
    """('stop' | 'unread' | 'ok', reason) by the registered rule. Fail-closed (round-2 reviews, 13 Nov 2026): the log must hold
    every step 1 .. step (a step a resumed run wrote twice counts once), each with an SFT loss and, for a twin run, a twin record;
    anything less is unread."""
    tm = train_metrics(ev, step)
    if tm.get("nonfinite"):
        return "stop", "non-finite loss"
    st = [e for e in ev if e.get("event") == "step"]
    steps = {e.get("step") for e in st if finite(e.get("step")) and e["step"] <= step}
    if steps != set(range(1, step + 1)):
        return "unread", f"the log does not hold every step 1 to {step}"
    if any("loss" not in e for e in st if e.get("step", 0) <= step):
        return "unread", "a step without its SFT loss"
    if twin and not all(isinstance(e.get("twin"), dict) and "loss" in e["twin"] for e in st if e.get("step", 0) <= step):
        return "unread", "a twin run's step without its twin record"
    if step != HEALTH_STEP or self_ref:
        return "ok", ""
    v, r = val_at(ev, step), val_at(ref_ev, step)
    if v == "nonfinite":
        return "stop", f"non-finite held-out SFT loss at step {step}"
    if v is None or r is None or r == "nonfinite":
        return "unread", f"held-out SFT loss at step {step}: job {v!r}, reference {r!r}"
    if v > (1 + HEALTH_REL) * r:
        return "stop", f"held-out SFT loss {v:.4f} more than {HEALTH_REL:.0%} above the reference's {r:.4f} at step {step}"
    return "ok", ""


def live_pid(chat, job):
    want = f"{chat}/{job}_spec.json"
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            cmd = open(f"/proc/{p}/cmdline", "rb").read().split(b"\0")
        except OSError:
            continue
        # `run SPEC` anywhere on the command line: mtstudio itself, or a wrapper or interpreter in front of it
        if any(cmd[i] == b"run" and cmd[i + 1].decode(errors="replace") == want for i in range(len(cmd) - 1)):
            return int(p)
    return None


def num(x, fmt):
    """A metric formatted, or '?' when it is absent or not a finite number: a summary never fails on partial metrics."""
    return format(x, fmt) if finite(x) else "?"


def summary_line(job, step, out):
    parts = [f"{job} step {step}"]
    for name, m in (out.get("early") or {}).items():
        if isinstance(m, dict) and m.get("n"):
            parts.append(f"{name}: both/tok {num(m.get('both_tok'), '.3f')} dec {num(m.get('dec'), '+.3f')} "
                         f"nui {num(m.get('nui'), '.3f')} sel {num(m.get('sel'), '+.3f')}")
    tm = out.get("train") or {}
    if "sft_loss_mean32" in tm:
        parts.append(f"sft {num(tm.get('sft_loss_mean32'), '.3f')}")
    if any(k.startswith("twin_") for k in tm):
        parts.append(f"dA {num(tm.get('twin_d_a_mean32'), '+.3f')} dB {num(tm.get('twin_d_b_mean32'), '+.3f')} "
                     f"active {num(tm.get('twin_active_mean32'), '.2f')}")
    if finite(out.get("val_loss")):
        parts.append(f"val {out['val_loss']:.3f}")
    if out.get("health_stop"):
        parts.append("HEALTH STOP: " + out["health_stop"])
    if out.get("health_unread"):
        parts.append("HEALTH UNREAD: " + out["health_unread"])
    return "; ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--ref", required=True, help="the reference run for the health stop (the job itself for a reference)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chat", default="/content/chat")
    ap.add_argument("--bin", default="/content/pk/build_cuda/mtstudio")
    a = ap.parse_args()
    up = f"{a.chat}/upload"
    self_ref = a.ref == a.job
    out = {"job": a.job, "step": a.step, "producer": "a31_early.py", "early": {}}

    def write():
        try:
            out["summary"] = summary_line(a.job, a.step, out)
        except Exception as e:  # noqa: BLE001  (the record is written whatever the summary does)
            out["summary"] = f"{a.job} step {a.step}; summary failed: {type(e).__name__}"
        tmp = f"{a.out}/early.json.tmp"
        os.makedirs(a.out, exist_ok=True)
        with open(tmp, "w") as f:
            json.dump(out, f, indent=1)
        os.replace(tmp, f"{a.out}/early.json")

    # 1. health first, written before anything else can fail; any error reading it is an unread check, never a pass
    try:
        twin = "twin_loss" in json.load(open(f"{a.chat}/{a.job}_spec.json")).get("train", {})
        t0 = time.time()
        while True:
            ev = events(f"{a.chat}/{a.job}/events.jsonl")
            ref_ev = ev if self_ref else events(f"{a.chat}/{a.ref}/events.jsonl")
            verdict, why = health(ev, ref_ev, a.step, self_ref, twin)
            if verdict != "unread" or time.time() - t0 > WAIT_S:
                break
            time.sleep(20)
        out["train"] = train_metrics(ev, a.step)
        out["val_loss"] = val_at(ev, a.step)
        out["ref"] = {"job": a.ref, "val_loss": val_at(ref_ev, a.step)}
    except Exception as e:  # noqa: BLE001  (fail-closed: whatever went wrong is recorded as unread)
        verdict, why = "unread", f"the health check failed: {type(e).__name__}: {e}"
    if verdict == "stop":
        out["health_stop"] = why
    elif verdict == "unread":
        out["health_unread"] = why
    write()   # the decision is on disk before any kill is attempted (launch reviews, 13 Nov 2026)
    if verdict == "stop":
        try:
            pid = live_pid(a.chat, a.job)
            out["stopped_pid"] = pid
            if pid:
                os.kill(pid, signal.SIGTERM)
                out["kill"] = "SIGTERM sent"
            else:
                out["kill"] = "no live training process (it had ended)"
        except (ProcessLookupError, PermissionError, OSError) as e:
            out["kill"] = f"not sent: {type(e).__name__}: {e}"
        write()

    # 2. then the early readout, on the early requests only; a failure here is recorded beside the health result
    try:
        early_readout(a, up, out)
    except Exception as e:  # noqa: BLE001
        out["readout_failed"] = f"{type(e).__name__}: {e}"
    write()
    print(out["summary"], flush=True)
    return 0


def early_readout(a, up, out):
    sys.path.insert(0, up)
    import a31_readout as R   # the lane's readout.py, uploaded and hash-pinned: the one measurement code
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
        rc = subprocess.run([a.bin, "score", m, "--in", f"{up}/a31_early_req.jsonl", "--out", f"{a.out}/scores.jsonl",
                             "--batch-tokens", "8192"], timeout=SCORE_TIMEOUT_S).returncode
    except subprocess.TimeoutExpired:
        rc = "timeout"
    out["score_rc"] = rc
    if rc == 0:
        man = json.load(open(f"{up}/a31_early_req.manifest.json"))
        trips = {n: R.read_jsonl(f"{up}/a31_{n}_triplets.jsonl") for n in man["sets"]}
        try:
            rows = R.validated_sets(man, trips, R.read_jsonl(f"{a.out}/scores.jsonl"))
            out["early"] = {n: R.summarise(rows[n].values()) for n in man["sets"]}
        except ValueError as e:
            out["score_invalid"] = str(e)


if __name__ == "__main__":
    sys.exit(main())
