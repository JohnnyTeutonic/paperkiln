#!/usr/bin/env python3
"""A31's early readouts and health stop (lane PROTOCOL.md section 50), a rung job on the pod. Claude Opus 5.5, 13 Nov 2026.

At each cut-off (steps 128, 256 and 512 of 1,024: 12.5%, 25% and 50%) jobs_vm.py snapshots a training job's weights; this job scores
the snapshot on the early requests only (the dev twins and a held-out-family dev set kept apart from every final test set; the
final test sets are read at the end only) and writes early.json: twin accuracy (both right, per token), the decisive, nuisance and
selectivity shifts per token on each early set, the training-step metrics logged up to the cut-off (the last step's SFT loss, twin
loss, d_a, d_b, d_n and active-hinge share), and the held-out SFT loss at the cut-off. Reported, not judged.

The one action it may take, registered before launch (the health stop): at any cut-off, a non-finite SFT or twin loss in the job's log;
at the 25% cut-off, a held-out SFT loss more than 10% above the reference run's at the same step (D0 for the dev sweep, P at the same
seed for the arms). Then it ends the job's training process (SIGTERM) and records HEALTH_STOP in early.json; jobs_vm.py marks the job
FAILED. Nothing else here changes an arm, a weight or a stop.

    python3 tools/colab_chat/a31_early.py --job J --step S --ref REF --out DIR [--chat /content/chat] [--bin MTSTUDIO]
"""
import argparse
import json
import math
import os
import signal
import subprocess
import sys

HEALTH_REL = 0.10
HEALTH_STEP = 256


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


def val_at(ev, step):
    for e in ev:
        if e.get("event") == "eval" and e.get("step") == step:
            return e["val_loss"]
    return None


def train_metrics(ev, step):
    st = [e for e in ev if e.get("event") == "step" and e.get("step", 0) <= step]
    if not st:
        return {}
    last = st[-1]
    window = st[-32:]
    out = {"step": last["step"], "sft_loss": last["loss"], "sft_loss_mean32": sum(e["loss"] for e in window) / len(window)}
    tw = [e["twin"] for e in window if "twin" in e]
    if tw:
        for k in ("loss", "d_a", "d_b", "d_n", "active"):
            if k in tw[0]:
                out[f"twin_{k}_mean32"] = sum(t[k] for t in tw) / len(tw)
    out["nonfinite"] = any(not math.isfinite(e["loss"]) or ("twin" in e and not math.isfinite(e["twin"]["loss"])) for e in st)
    return out


def health(ev, ref_ev, step):
    """(stop, reason) by the registered rule."""
    tm = train_metrics(ev, step)
    if tm.get("nonfinite"):
        return True, "non-finite loss"
    if step == HEALTH_STEP:
        v, r = val_at(ev, step), val_at(ref_ev, step)
        if v is not None and r is not None and v > (1 + HEALTH_REL) * r:
            return True, f"held-out SFT loss {v:.4f} more than {HEALTH_REL:.0%} above the reference's {r:.4f} at step {step}"
    return False, ""


def measures(recs, n):
    """Five score records per triplet (readout.py's order) -> means of the early measures."""
    both = dec = nui = 0.0
    k = 0
    for i in range(n):
        s = recs[5 * i:5 * i + 5]
        if len(s) < 5 or any(r.get("over_ctx") for r in s):
            continue
        aa, ba, bb, ab, na = (r["sum"] for r in s)
        a, b = len(s[0]["logprobs"]), len(s[2]["logprobs"])
        both += float(aa / a > ab / b and bb / b > ba / a)
        d = ((aa - ba) / a + (bb - ab) / b) / 2
        dec += d
        nui += abs(aa - na) / a
        k += 1
    return {"n": k, "both_tok": both / k, "dec": dec / k, "nui": nui / k, "sel": (dec - nui) / k} if k else {"n": 0}


def live_pid(chat, job):
    want = f"{chat}/{job}_spec.json"
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            cmd = open(f"/proc/{p}/cmdline", "rb").read().split(b"\0")
        except OSError:
            continue
        if len(cmd) >= 3 and cmd[1] == b"run" and cmd[2].decode(errors="replace") == want:
            return int(p)
    return None


def summary_line(job, step, out):
    parts = [f"{job} step {step}"]
    for name, m in out.get("early", {}).items():
        if m.get("n"):
            parts.append(f"{name}: both/tok {m['both_tok']:.3f} dec {m['dec']:+.3f} nui {m['nui']:.3f}")
    tm = out.get("train", {})
    if tm:
        parts.append(f"sft {tm.get('sft_loss_mean32', float('nan')):.3f}")
        if "twin_d_a_mean32" in tm:
            parts.append(f"dA {tm['twin_d_a_mean32']:+.3f} dB {tm['twin_d_b_mean32']:+.3f} active {tm['twin_active_mean32']:.2f}")
    if out.get("val_loss") is not None:
        parts.append(f"val {out['val_loss']:.3f}")
    if out.get("health_stop"):
        parts.append("HEALTH STOP: " + out["health_stop"])
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
    man = json.load(open(f"{up}/a31_early_req.manifest.json"))
    m = f"{a.out}/m"
    os.makedirs(m, exist_ok=True)
    snap = f"{a.chat}/{a.job}/rung_{a.step}/model.safetensors"
    link = f"{m}/{a.job}.safetensors"
    if os.path.lexists(link):
        os.remove(link)
    os.symlink(snap, link)
    spec = json.load(open(f"{a.chat}/{a.job}_spec.json"))
    spec["out_dir"] = m
    json.dump(spec, open(f"{m}/spec.json", "w"), indent=1)
    rc = subprocess.run([a.bin, "score", m, "--in", f"{up}/a31_early_req.jsonl", "--out", f"{a.out}/scores.jsonl", "--batch-tokens", "8192"]).returncode
    out = {"job": a.job, "step": a.step, "producer": "a31_early.py", "score_rc": rc, "early": {}}
    if rc == 0:
        recs = [json.loads(x) for x in open(f"{a.out}/scores.jsonl") if x.strip()]
        for name, info in man["sets"].items():
            f = info["first_request"]
            out["early"][name] = measures(recs[f:f + 5 * info["triplets"]], info["triplets"])
    ev = events(f"{a.chat}/{a.job}/events.jsonl")
    out["train"] = train_metrics(ev, a.step)
    out["val_loss"] = val_at(ev, a.step)
    ref_ev = ev if a.ref == a.job else events(f"{a.chat}/{a.ref}/events.jsonl")
    out["ref"] = {"job": a.ref, "val_loss": val_at(ref_ev, a.step)}
    stop, why = health(ev, ref_ev, a.step)
    if stop and (a.ref != a.job or why == "non-finite loss"):   # a reference is never compared with itself
        out["health_stop"] = why
        pid = live_pid(a.chat, a.job)
        out["stopped_pid"] = pid
        if pid:
            os.kill(pid, signal.SIGTERM)
    out["summary"] = summary_line(a.job, a.step, out)
    tmp = f"{a.out}/early.json.tmp"
    json.dump(out, open(tmp, "w"), indent=1)
    os.replace(tmp, f"{a.out}/early.json")
    print(out["summary"], flush=True)
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
