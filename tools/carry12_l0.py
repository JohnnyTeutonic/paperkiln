#!/usr/bin/env python3
"""Stage L0 of the long-carry programme: an SFT model that is home, scored on Carry-12 v1, no training.

    python tools/carry12_l0.py run --model-dir DIR --out ROOT [--tag TAG] [--gen none|ember]
        [--mtstudio BIN] [--ember-bin BIN|auto] [--vocab V.gguf] [--gpu-layers 0] [--every N] [--smoke]
    python tools/carry12_l0.py table ROOT [--out FILE.md]
    python tools/carry12_l0.py colab-jobs --model TAG=LOCAL_DIR [...] --jobs-out FILE.json

Registration: experiments/tiny_chat_long_2026-10/STAGE_L0.md (frozen before any L0 scoring). DIR is a
tiny-chat pick folder (model.safetensors, spec.json and usually model.gguf; tiny_chat_run.py pick). `run`
writes a local spec (name "model", out_dir DIR, the vocabulary remapped to --vocab), exports the F32
GGUF when DIR has none, and runs tools/carry12_eval.py on the frozen set at the model's own context:
forced choice with gold, near, twin and noctx history on every item; with --gen ember also self and
near_self history played to the target (--max-new 24, ember's server after a spot-check against
mtstudio chat, 20 items in the Colab jobs, as the tiny-chat post job generates). ROOT/TAG/l0.json then holds only what the
registration reads: the validity gate and HL0 (D = gold minus near over the context families, item-level
paired interval, verdict against the 3-point SESOI), with the descriptive quantities beside them.
--every N > 1 or --smoke marks the receipt SMOKE: a test of the tooling, never an L0 result.

`colab-jobs` writes a jobs file (tools/colab_chat/runner_multi.py) with one job per model on the VM's GPU:
ember built by tiny_chat_run.ensure_ember, generation on, the model read from the runner's flat uploads
/content/chat/upload/TAG.model.safetensors and TAG.spec.json (`run --weights --spec-json`; the GGUF is
exported on the VM), and prints the runner's --upload arguments. It never launches anything.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import carry12_build as c12  # noqa: E402

REGISTRATION = os.path.join(ROOT, "experiments", "tiny_chat_long_2026-10", "STAGE_L0.md")
CONTEXT_FAMILIES = ("recall", "correction", "topic", "load")
PRIMARY_FAMILIES = CONTEXT_FAMILIES + ("knowledge",)
SESOI = 0.03
LAPTOP_VOCAB = "C:/ml_artifacts/capacity/vocab/vocab_llama3_bpe.gguf"
VM_VOCAB = "/content/chat/upload/vocab_llama3_bpe.gguf"
VOCABS = (LAPTOP_VOCAB, "/mnt/c/ml_artifacts/capacity/vocab/vocab_llama3_bpe.gguf", VM_VOCAB)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def jsave(p, obj):
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, p)


def local_spec(model_dir, vocab, out):
    """A spec that names DIR/model.safetensors (and DIR/model.gguf): the pick's spec with its VM paths
    replaced."""
    spec = jload(os.path.join(model_dir, "spec.json"))
    spec = {**spec, "name": "model", "out_dir": os.path.abspath(model_dir)}
    spec["data"] = {**spec["data"], "vocab": vocab}
    spec["train"] = {**spec.get("train", {})}
    spec["train"].pop("init_from", None)  # a VM path; scoring and chat read out_dir/name.safetensors
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, "spec_local.json")
    jsave(p, spec)
    return p, spec


def paired(diffs, z95=1.959964, z90=1.644854):
    n = len(diffs)
    m = sum(diffs) / n
    sd = math.sqrt(sum((d - m) ** 2 for d in diffs) / (n - 1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n else None
    return {"n": n, "mean": round(m, 5), "se": round(se, 5), "ci95": [round(m - z95 * se, 5), round(m + z95 * se, 5)],
            "ci90": [round(m - z90 * se, 5), round(m + z90 * se, 5)]}


def verdict(st, sesoi=SESOI):
    """HL0 predicts D < 0 beyond the SESOI (STAGE_L0.md section 3)."""
    lo95, hi95 = st["ci95"]
    lo90, hi90 = st["ci90"]
    if hi95 < 0:
        return "SUPPORTED" if st["mean"] <= -sesoi else "SIGNIFICANT_BELOW_SESOI"
    if lo95 > 0:
        return "CONTRADICTED"
    if -sesoi < lo90 and hi90 < sesoi:
        return "EQUIVALENT"
    return "NOT_RESOLVED"


def l0_receipt(items_path, summary, meta):
    """The registered quantities from carry12_eval's per-item records and summary."""
    with open(items_path, encoding="utf-8") as f:
        recs = [json.loads(x) for x in f if x.strip()]

    def pool(mode, fams):
        xs = [r["fc"][mode]["acc"] for r in recs if r["family"] in fams and mode in r["fc"]]
        return {"mean": round(sum(xs) / len(xs), 5) if xs else None, "n": len(xs)}
    twin_p, twin_c, noctx_c = pool("twin", PRIMARY_FAMILIES), pool("twin", CONTEXT_FAMILIES), pool("noctx", CONTEXT_FAMILIES)
    gate = {"twin_primary": twin_p["mean"], "twin_minus_noctx_context":
            None if twin_c["mean"] is None else round(twin_c["mean"] - noctx_c["mean"], 5)}
    gate["pass"] = gate["twin_primary"] is not None and gate["twin_primary"] >= 0.35 and \
        gate["twin_minus_noctx_context"] >= 0.10
    diffs = [r["fc"]["gold"]["acc"] - r["fc"]["near"]["acc"] for r in recs
             if r["family"] in CONTEXT_FAMILIES and "gold" in r["fc"] and "near" in r["fc"]]
    st = paired(diffs) if len(diffs) > 1 else None
    hl0 = {"definition": "D = acc(gold) - acc(near), raw log-prob forced choice, item-level over the context "
                         "families (recall, correction, topic, load); prediction D < 0 beyond the SESOI",
           "sesoi": SESOI, "acc_gold": pool("gold", CONTEXT_FAMILIES), "acc_near": pool("near", CONTEXT_FAMILIES),
           "D": st, "verdict": ("FLOOR" if not gate["pass"] else verdict(st)) if st else None,
           "verdict_if_no_floor": verdict(st) if st else None}
    m = summary
    desc = {"slope_gold_on_distance": m["hl0"]["slope_gold_on_distance"],
            "slope_gold_minus_near_on_distance": m["hl0"]["slope_gold_minus_near_on_distance"],
            "acc_by_mode_all": {k: v["all"]["mean"] for k, v in m["fc"]["acc"].items() if v["all"]["n"]},
            "pools_acc": m["pools"]["acc"], "derived": m["derived"],
            "gold_by_distance": {k: v["mean"] for k, v in m["fc"]["acc"]["gold"]["distance"].items()},
            "near_by_distance": {k: v["mean"] for k, v in m["fc"]["acc"]["near"]["distance"].items()},
            "family_distance_gold": m["fc"]["family_distance"]["gold"], "over_ctx": m["fc"]["over_ctx"],
            "gen_engine": m.get("gen_engine"), "gate_replies": m.get("gate"), "decay": m.get("decay")}
    return {"stage": "L0", **meta, "validity_gate": gate, "hl0": hl0, "descriptive": desc}


def cmd_run(a):
    if not a.model_dir and not (a.weights and a.spec_json):
        sys.exit("run: --model-dir DIR, or --weights W.safetensors and --spec-json S.json (the Colab uploads)")
    tag = a.tag or os.path.basename(os.path.normpath(a.model_dir))
    out = os.path.join(a.out, tag)
    os.makedirs(out, exist_ok=True)
    if not a.model_dir:  # the runner's flat uploads: a pick folder assembled from them
        a.model_dir = os.path.join(out, "model")
        os.makedirs(a.model_dir, exist_ok=True)
        for src, name in ((a.weights, "model.safetensors"), (a.spec_json, "spec.json")):
            dst = os.path.join(a.model_dir, name)
            if not os.path.exists(dst):
                os.symlink(os.path.abspath(src), dst)
    vocab = a.vocab or next((v for v in VOCABS if os.path.exists(v)), VM_VOCAB)
    spec_path, spec = local_spec(a.model_dir, vocab, out)
    w = os.path.join(a.model_dir, "model.safetensors")
    g = os.path.join(a.model_dir, "model.gguf")
    if not os.path.exists(g):
        p = subprocess.run([a.mtstudio, "export", spec_path, "--from", w, "--formats", "gguf"], capture_output=True,
                           text=True)
        if p.returncode != 0 or not os.path.exists(g):
            sys.exit(f"mtstudio export --formats gguf failed (rc {p.returncode}): {p.stderr[-500:]}")
    ember = a.ember_bin
    build = None
    if a.gen == "ember" and ember == "auto":
        sys.path.insert(0, os.path.join(HERE, "colab_chat"))
        import tiny_chat_run
        build = tiny_chat_run.ensure_ember(os.path.join(out, "ember_build.log"))
        if "error" in build:
            sys.exit(f"ember build failed: {build['error']}")
        ember = build["bin"]
    smoke = a.smoke or a.every > 1
    argv = [sys.executable, os.path.join(HERE, "carry12_eval.py"), "run", "--model", f"{tag}={spec_path}",
            "--out", os.path.join(out, "eval"), "--set", c12.OUT, "--scorer", "mtstudio", "--mtstudio", a.mtstudio,
            "--gen", a.gen, "--self-scope", "target", "--max-new", str(a.max_new), "--every", str(a.every),
            "--port", str(a.port)]
    if a.gen == "ember":
        argv += ["--ember-bin", ember, "--gpu-layers", str(a.gpu_layers), "--spot", str(a.spot)]
    p = subprocess.run(argv, capture_output=True, text=True)
    with open(os.path.join(out, "eval.log"), "w", encoding="utf-8") as f:
        f.write(p.stdout[-20000:] + p.stderr[-20000:])
    if p.returncode != 0:
        sys.exit(f"carry12_eval failed (rc {p.returncode}): {p.stderr[-800:]}")
    s = jload(os.path.join(out, "eval", "summary.json"))
    meta = {"smoke": smoke, "tag": tag,
            "registration": os.path.relpath(REGISTRATION, ROOT).replace(os.sep, "/"),
            "registration_sha256": sha256(REGISTRATION) if os.path.exists(REGISTRATION) else None,
            "model": {"dir": os.path.abspath(a.model_dir), "safetensors_sha256": sha256(w), "gguf_sha256": sha256(g),
                      "spec_name": jload(os.path.join(a.model_dir, "spec.json")).get("name"),
                      "T": spec["data"].get("T"), "arch": spec.get("arch")},
            "set": {"path": s["meta"]["set"], "sha256": s["meta"]["sha256"],
                    "pinned": s["meta"]["sha256"] == c12.pinned()},
            "eval": {"argv": argv[1:], "gen": a.gen, "every": a.every, "ember_build": build}}
    rec = l0_receipt(os.path.join(out, "eval", tag, "items.jsonl"), s["models"][tag], meta)
    jsave(os.path.join(out, "l0.json"), rec)
    print(json.dumps({k: rec[k] for k in ("tag", "smoke", "validity_gate")} | {"hl0": rec["hl0"]["D"],
                                                                                 "verdict": rec["hl0"]["verdict"]}))
    return 0


def fmt(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def cmd_table(a):
    rows = []
    for tag in sorted(os.listdir(a.root)):
        p = os.path.join(a.root, tag, "l0.json")
        if os.path.exists(p):
            rows.append(jload(p))
    L = ["# Stage L0 (Carry-12 v1)", "", "Registration: experiments/tiny_chat_long_2026-10/STAGE_L0.md. D = gold - near "
         "over the context families (HL0 predicts D < -0.03).", "",
         "| model | smoke | T | twin (primary) | twin - noctx (context) | gate | gold | near | D | 95% interval | verdict |",
         "|---|---|---:|---:|---:|---|---:|---:|---:|---|---|"]
    for r in rows:
        g, h = r["validity_gate"], r["hl0"]
        d = h["D"] or {}
        L.append(f"| {r['tag']} | {'SMOKE' if r['smoke'] else ''} | {r['model']['T']} | {fmt(g['twin_primary'])} | "
                 f"{fmt(g['twin_minus_noctx_context'])} | {'pass' if g['pass'] else 'FLOOR'} | "
                 f"{fmt(h['acc_gold']['mean'])} | {fmt(h['acc_near']['mean'])} | {fmt(d.get('mean'))} | "
                 f"[{fmt((d.get('ci95') or [None])[0])}, {fmt((d.get('ci95') or [None, None])[1])}] | {h['verdict']} |")
    text = "\n".join(L) + "\n"
    if a.out:
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
    print(text)
    return 0


def cmd_colab_jobs(a):
    jobs, uploads = [], []
    for m in a.model:
        tag, d = m.split("=", 1)
        for f in ("model.safetensors", "spec.json"):
            if not os.path.exists(os.path.join(d, f)) and not a.allow_missing:
                sys.exit(f"{d}/{f} missing (--allow-missing prepares the file before the picks are home)")
            # flat keys (the runner reassembles uploads in /content/chat/upload itself); WSL paths
            uploads.append(f"{tag}.{f}={d.rstrip('/')}/{f}")
        jobs.append({"name": f"l0_{tag}",
                     "cmd": ["python3", "tools/carry12_l0.py", "run",
                             "--weights", f"/content/chat/upload/{tag}.model.safetensors",
                             "--spec-json", f"/content/chat/upload/{tag}.spec.json", "--tag", tag, "--out", "{out}", "--gen", "ember", "--ember-bin", "auto",
                             "--mtstudio", "/content/pk/build_cuda/mtstudio", "--vocab", VM_VOCAB,
                             "--gpu-layers", "-1", "--spot", "20"],
                     "outputs": [f"{tag}/l0.json", f"{tag}/eval/summary.json", f"{tag}/eval/summary.md",
                                 f"{tag}/eval/{tag}/items.jsonl", f"{tag}/eval/{tag}/gen.jsonl",
                                 f"{tag}/eval/{tag}/gen_engine.json", f"{tag}/eval.log"]})
    uploads.append(f"vocab_llama3_bpe.gguf={LAPTOP_VOCAB.replace('C:/', '/mnt/c/')}")
    doc = {"_comment": "Stage L0 of the long-carry programme (experiments/tiny_chat_long_2026-10/STAGE_L0.md, "
                       "frozen before any L0 scoring): each home tiny-chat SFT model scored on Carry-12 v1 "
                       "(tools/carry12_l0.py run; forced choice gold/near/twin/noctx plus self and near_self "
                       "history to the target through ember's CUDA server after the 20-item spot-check against "
                       "mtstudio chat, EMBER_GEMM_TF32=0; forced choice by mtstudio score). One GPU job at a time. "
                       "Inputs: the runner's --upload of each model's safetensors and spec.json (the F32 GGUF is "
                       "exported on the VM) and vocab_llama3_bpe.gguf. PREPARED, NOT LAUNCHED: the main session "
                       "launches Colab work. --upload arguments: " + " ".join(f"--upload {u}" for u in uploads),
           "max_parallel": 1, "jobs": jobs}
    with open(a.jobs_out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    print(" ".join(f"--upload {u}" for u in uploads))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--model-dir", default="")
    p.add_argument("--weights", default="", help="with --spec-json, in place of --model-dir (Colab uploads)")
    p.add_argument("--spec-json", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--tag", default="")
    p.add_argument("--gen", default="none", choices=["none", "ember"])
    p.add_argument("--mtstudio", required=True)
    p.add_argument("--ember-bin", default="")
    p.add_argument("--vocab", default="")
    p.add_argument("--gpu-layers", default="0")
    p.add_argument("--spot", type=int, default=20)
    p.add_argument("--max-new", type=int, default=24)
    p.add_argument("--port", type=int, default=8094)
    p.add_argument("--every", type=int, default=1)
    p.add_argument("--smoke", action="store_true")
    p = sub.add_parser("table")
    p.add_argument("root")
    p.add_argument("--out", default="")
    p = sub.add_parser("colab-jobs")
    p.add_argument("--model", action="append", required=True)
    p.add_argument("--jobs-out", required=True)
    p.add_argument("--allow-missing", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "run" and a.gen == "ember" and not a.ember_bin:
        ap.error("--gen ember needs --ember-bin (a binary, or auto: build at tiny_chat_run's pinned commit)")
    return {"run": cmd_run, "table": cmd_table, "colab-jobs": cmd_colab_jobs}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
