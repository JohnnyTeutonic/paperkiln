"""Post-hoc input/provenance audit; does not score models or amend the registration.

Run before resuming the registered size extension or launching its 150M Stage L0.
A readiness receipt never substitutes for the missing experiments or their analysis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
EXTENSIONS = (
    ("20m", "dc29b5e6", "ae9ed69f", "tiny_chat_e5_s35", "e5_s35", "post_e5_s35"),
    ("60m", "a90ec0e2", "163f73ad", "tiny_chat_e1_60m_L8", "e1_60m_L8", "post_e1_60m_L8"),
    ("150m", "3abfbc07", "528201fe", "tiny_chat_e1_150m", "e1_150m", "post_e1_150m_L8"),
)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def same_source_bytes(a, b):
    # Git on this Windows checkout writes CRLF. Only that checkout conversion
    # is ignored; JSON reformatting or any other source change still fails.
    return a.replace(b"\r\n", b"\n") == b.replace(b"\r\n", b"\n")


def registered_winner(rows):
    """The frozen pick stays over seeds 1-3; extension seeds cannot enter it."""
    initial = [r for r in rows if r["seed"] in (1, 2, 3)]
    if sorted(r["seed"] for r in initial) != [1, 2, 3]:
        raise ValueError("pick needs exactly one summary for each initial seed")
    if any(r.get("primary") is None for r in initial):
        raise ValueError("pick has an incomplete initial seed")
    return max(initial, key=lambda r: (r["primary"], -r["seed"]))


def git(*args):
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()


def audit(art):
    result = {"kind": "POST_HOC_INPUT_AUDIT_NOT_A_SCIENTIFIC_RESULT", "errors": [],
              "auditor_sha256": sha(Path(__file__)),
              "registration_sha256": sha(HERE.parent / "PREREGISTRATION.md"),
              "analysis_sha256": sha(HERE / "analysis_size_2026-10-30.json")}
    errors = result["errors"]
    # The registration's 28 Oct implementation note fixes both pins and the only
    # changes allowed on each launch branch: its seed-4 and seed-5 jobs files.
    extension = []
    for size, base, pin, stem, folder, post in EXTENSIONS:
        expected = [f"tools/colab_chat/{stem}_r{r}.json" for r in (4, 5)]
        changed = git("diff", "--name-only", base, pin).splitlines()
        valid = sorted(changed) == sorted(expected)
        if not valid:
            errors.append(f"{size}: extension pin differs beyond its two jobs files")
        for seed, job in zip((4, 5), expected):
            pinned = subprocess.check_output(["git", "-C", str(REPO), "show", f"{pin}:{job}"])
            working = REPO / job
            same = same_source_bytes(pinned, working.read_bytes())
            if not same:
                errors.append(f"{job}: working jobs file differs from pin after CRLF normalization")
            summary = art / f"{folder}_r{seed}" / f"{post}_r{seed}" / "summary.json"
            extension.append({"size": size, "seed": seed, "pin": git("rev-parse", pin),
                              "only_registered_jobs_added": valid, "jobs_file": job,
                              "jobs_file_sha256": hashlib.sha256(pinned).hexdigest(),
                              "working_jobs_equal_pin_after_CRLF_normalization": same,
                              "working_jobs_raw_sha256": sha(working), "summary": str(summary),
                              "summary_home": summary.is_file()})
    result["extension"] = extension
    result["missing_extension_runs"] = [f"{r['size']}/seed{r['seed']}" for r in extension if not r["summary_home"]]
    result["extension_complete"] = not result["missing_extension_runs"]
    analysis = json.loads((HERE / "analysis_size_2026-10-30.json").read_text())
    result["registered_extension_trigger"] = {
        key: {"verdict": analysis[key]["verdict"],
              "extend_to_5_seeds": analysis[key]["extend_to_5_seeds"]}
        for key in ("HS1_size_slope_Y", "HS2_step_60_150_per_doubling")}
    result["extension_required_by_registered_rule"] = any(
        v["extend_to_5_seeds"] for v in result["registered_extension_trigger"].values())

    rows = []
    for seed in (1, 2, 3, 4, 5):
        p = art / f"e1_150m_r{seed}" / f"post_e1_150m_L8_r{seed}" / "summary.json"
        s = json.loads(p.read_text())
        rows.append({"seed": seed, "primary": s.get("primary"), "summary_sha256": sha(p),
                     "sft_weights_sha256": s.get("sft_weights_sha256")})
    winner = registered_winner(rows)
    pick = art / "e1_150m" / "pick_e1_150m_L8"
    best = json.loads((pick / "best.json").read_text())
    spec = json.loads((pick / "spec.json").read_text())
    files = {n: {"bytes": (pick / n).stat().st_size, "sha256": sha(pick / n)}
             for n in ("model.safetensors", "model.gguf", "spec.json", "best.json")}
    with (pick / "model.safetensors").open("rb") as f:
        length = struct.unpack("<Q", f.read(8))[0]
        if length > 16 * 1024 * 1024:
            raise ValueError("safetensors header exceeds audit bound")
        header = json.loads(f.read(length))
    weights = [k for k in header if k != "__metadata__"]
    if not weights or any(k.startswith("adamw.") for k in weights):
        errors.append("pick safetensors is not a pure model weights file")
    for label, actual in (("pick receipt", best["best"]["seed"]), ("SFT spec", spec["train"]["seed"])):
        if actual != winner["seed"]:
            errors.append(f"{label} disagrees with registered three-seed pick")
    weight_hash = files["model.safetensors"]["sha256"]
    if weight_hash != winner["sft_weights_sha256"] or weight_hash != best["weights_sha256"]:
        errors.append("pick weights hash does not match selected post summary and receipt")
    with (pick / "model.gguf").open("rb") as f:
        if f.read(4) != b"GGUF":
            errors.append("pick GGUF magic is invalid")
    result["pick"] = {"selection_seeds": [1, 2, 3], "all_home_scores": rows,
                      "registered_winner": winner, "files": files, "model_tensor_count": len(weights),
                      "note": "Seed 4/5 scores do not change the registered model pick."}
    result["stage_l0_inputs_verified"] = not errors
    result["stage_l0_scoring_complete"] = (art / "l0" / "pick_e1_150m_L8" / "l0.json").is_file()
    result["note"] = "Input audit only; no Stage L0 scoring, extension launch, or scientific decision was performed."
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--art", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        ap.error("receipt exists; choose a new filename")
    result = audit(a.art)
    with a.out.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=1)
        f.write("\n")
    print(json.dumps({k: result[k] for k in ("errors", "missing_extension_runs", "stage_l0_inputs_verified")}, indent=1))
    return bool(result["errors"])


if __name__ == "__main__":
    raise SystemExit(main())
