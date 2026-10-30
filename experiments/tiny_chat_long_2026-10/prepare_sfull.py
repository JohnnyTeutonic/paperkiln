"""Build a review-only conditional S-full packet from exact frozen training code; never launch."""

from __future__ import annotations
import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REL = "experiments/tiny_chat_long_2026-10"
PIN = "3dc818b2476b0957b1337b4c9af22474b76350b5"
TRIGGER_SHA = "6868c6374d6ca637b91ec106d762c79c193b9cd0de9c784fb43311b9588da04a"
REFERENCE_SHA = "acd00d74a93299aa0614f98233f837ef3dc30d38ce9120330e9f2f5c2b982cf4"
RECEIPTS = f"{REL}/receipts/sfull_prospective_20261031"
TRIGGER = ROOT / REL / "receipts/hs1_five_seed_20261030/analysis_hs1.json"
REFERENCE_ART = Path("/mnt/c/ml_artifacts/tiny_chat/c12_Sgist_r1/data/data.ledger.json")
SEEDS = (1, 2, 3)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def frozen(p):
    return subprocess.check_output(["git", "-C", str(ROOT), "show", f"{PIN}:{p}"])


def trigger(data):
    if sha(data) != TRIGGER_SHA:
        raise ValueError("exact audited five-seed trigger bytes required")
    r = json.loads(data)
    h = r["HS1_Sgist_minus_B"]
    if (
        r.get("partial") is not False
        or r.get("seeds") != [1, 2, 3, 4, 5]
        or r.get("cells") != ["B", "Sgist"]
        or r.get("missing") != []
        or not all(r["validity_gate"].values())
        or h.get("verdict") != "SUPPORTED"
        or h.get("n") != 5
        or len(h.get("per_seed", [])) != 5
        or not all(x < 0 for x in h["per_seed"])
        or r.get("s_full_condition", {}).get("run_s_full") is not True
    ):
        raise ValueError("registered S-full trigger not established")
    return copy.deepcopy(h)


def build(seed):
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("initial S-full seeds exactly1,2,3")
    source = f"tools/colab_chat/carry12_Sgist_r{seed}.json"
    old = json.loads(frozen(source))
    doc = copy.deepcopy(old)
    # Do not call make_jobs.cell_spec('Sfull'): that unsupported cell falls back to L16.
    doc = json.loads(json.dumps(doc).replace("Sgist", "Sfull"))
    jobpath = f"tools/colab_chat/carry12_sfull_r{seed}.json"
    specpath = f"{REL}/specs/cell_Sfull.json"
    pt = f"pt_c12_Sfull_r{seed}"
    post = f"post_c12_Sfull_r{seed}"
    names = {j["name"] for j in doc["jobs"]}
    if names != {"cuda_gate", "data", "gate", pt, post}:
        raise ValueError("unexpected frozen source jobs")
    for j in doc["jobs"]:
        if j["name"] == pt:
            s = j["spec"]
            assert s["arch"]["custom"] == {
                "d": 768,
                "layers": 8,
                "heads": 12,
                "d_ff": 2240,
            }
            assert s["arch"]["state_tokens"] == {"k": 4, "mask": "turn_state"}
            s["arch"]["state_tokens"]["mask"] = "causal"
            j["after"] = ["data", "gate", "cuda_gate", "sfull_inputs"]
        elif j["name"] == "gate":
            j["cmd"][j["cmd"].index("--jobs") + 1] = jobpath
        elif j["name"] == post:
            j["cmd"][j["cmd"].index("--spec") + 1] = specpath
    verify = {
        "name": "sfull_inputs",
        "light": True,
        "after": "data",
        "cmd": [
            "python3",
            "tools/colab_chat/sfull_verify_inputs.py",
            "--data",
            "/content/chat/data",
            "--reference",
            f"{RECEIPTS}/data_reference.json",
            "--out",
            "{out}",
        ],
        "outputs": ["verified_inputs.json"],
    }
    doc["jobs"].insert(3, verify)
    doc["_comment"] = (
        f"REVIEW ONLY; NOT AUTHORIZED TO LAUNCH. Conditional Sfull seed{seed}, exact parent{PIN}, SgistL8 clone with causal mask only. Original CUDA/data/scientific gates remain. Additional fail-closed input identity gate. RunPodL4 cap$7 cumulative; max23.5h; proposed serial queue. No baseline retraining or unrelated jobs."
    )
    return jobpath, doc, source


def write(out):
    out = Path(out)
    files = {}
    proof = trigger(TRIGGER.read_bytes())
    ref = REFERENCE_ART.read_bytes()
    if sha(ref) != REFERENCE_SHA:
        raise ValueError("original Sgist1 reference changed")

    def add(path, data):
        p = out / path
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists() and p.read_bytes() != data:
            raise ValueError("refuse overwrite:" + str(p))
        p.write_bytes(data)
        files[path] = sha(data)

    for seed in SEEDS:
        path, doc, _ = build(seed)
        add(path, (json.dumps(doc, indent=1) + "\n").encode())
    spec = json.loads(frozen(f"{REL}/specs/cell_Sgist.json"))
    spec["arch"]["state_tokens"]["mask"] = "causal"
    add(f"{REL}/specs/cell_Sfull.json", (json.dumps(spec, indent=1) + "\n").encode())
    add(f"{RECEIPTS}/data_reference.json", ref)
    add(f"{RECEIPTS}/trigger_hs1_five_seed.json", TRIGGER.read_bytes())
    add(
        f"{RECEIPTS}/historical_history_parity.json",
        (TRIGGER.parent / "history_argmax_audit.json").read_bytes(),
    )
    add(f"{RECEIPTS}/.gitattributes", b"* -text\n")
    add(
        "tools/colab_chat/.gitattributes",
        b"carry12_sfull_*.json -text\nsfull_verify_inputs.py text eol=lf\n",
    )
    add(f"{REL}/specs/.gitattributes", b"cell_Sfull.json -text\n")
    add(
        "tools/colab_chat/sfull_verify_inputs.py",
        (ROOT / "tools/colab_chat/sfull_verify_inputs.py").read_bytes(),
    )
    boot = frozen("tools/colab_chat/boot_jobs.sh")
    old = b"pip -q install huggingface_hub pandas pyarrow"
    new = old + b" tokenizers==0.23.3"
    assert boot.count(old) == 1
    add("tools/colab_chat/boot_sfull.sh", boot.replace(old, new))
    baseline = {}
    art = REFERENCE_ART.parents[2]
    for cell in ["B", "Sgist"]:
        for seed in SEEDS:
            for suffix in [
                f"post_c12_{cell}_r{seed}/summary.json",
                "data/data.ledger.json",
                "gate/gate.json",
            ]:
                path = f"c12_{cell}_r{seed}/" + suffix
                baseline[path] = sha((art / path).read_bytes())
    manifest = {
        "baseline_artifact_sha256": baseline,
        "status": "READY_FOR_FUNDING_REVIEW_WITH_RUNTIME_DATA_GATE",
        "ready_for_launch": False,
        "archive_commit": None,
        "engineering_remaining": [
            "independent funding/parity/readiness review before any enable"
        ],
        "parent": PIN,
        "trigger_report_sha256": TRIGGER_SHA,
        "trigger": proof,
        "cell": "Sfull",
        "seeds": list(SEEDS),
        "comparison_seeds": [1, 2, 3],
        "reuse_baselines": ["B1-3", "Sgist1-3"],
        "data_reference_sha256": REFERENCE_SHA,
        "data_seed": 1,
        "weights_reused": False,
        "scientific_delta_from_Sgist": "arch.state_tokens.mask turn_state -> causal only; k4 and L8 retained",
        "bootstrap_delta": "frozen boot dependencies plus tokenizers==0.23.3; training/data code unchanged",
        "artifacts_sha256": files,
        "per_seed_cap_usd": 7,
        "total_cap_usd": 21,
        "expected_gpu_hours": [18, 27],
        "expected_gpu_usd": [11, 16],
        "proposed_queue": "serial; no watchdog entry enabled; root funding review required",
        "review_blockers": [
            "separate long-run and funding review; $40 top-up offered but uncredited; notify below $20, not a mandated reserve",
            "13 raw human corpora and 5 products unavailable locally; exact rebuild required; hash mismatch blocks PT, no fallback or automatic rerun",
            "six historical full-history argmax mismatches; no all-pass claim",
        ],
        "no_paid_launch": True,
    }
    p = out / f"{RECEIPTS}/manifest.json"
    p.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    print(json.dumps(write(a.out), indent=2))
