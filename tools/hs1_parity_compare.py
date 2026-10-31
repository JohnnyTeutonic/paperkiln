"""Join audited new reconstructions without changing any historical parity decision."""

import argparse
import hashlib
import json
from pathlib import Path


def compare(cpu, fp64, gpu):
    if gpu.get("status") == "SCORER_COMPONENT_AUDITED_OVERALL_ERROR":
        if gpu.get("overall_run_status") != "ERROR" or gpu.get("generation") != []:
            raise ValueError(
                "partial component must preserve overall ERROR and absent generation"
            )
    elif gpu.get("status") != "RECONSTRUCTION_AUDITED":
        raise ValueError(
            "GPU reconstruction must first pass independent artifact audit"
        )
    cpu_rows = {r["key"]: r for r in cpu["rows"]}
    gpu_rows = {r["key"]: r for r in gpu["rows"]}
    fp64_rows = {r["key"]: r for r in fp64["rows"]}
    if (
        len(cpu_rows) != 6
        or len(gpu_rows) != 6
        or cpu_rows.keys() != gpu_rows.keys()
        or len(fp64_rows) != 4
    ):
        raise ValueError("six-reply/four-B diagnostic identity differs")
    rows = []
    for key, c in cpu_rows.items():
        g = gpu_rows[key]
        f = fp64_rows.get(key)
        if len(c["positions"]) != len(g["positions"]) or (
            f and len(f["positions"]) != len(c["positions"])
        ):
            raise ValueError("position inventory differs")
        positions = []
        for i, (cp, gp) in enumerate(zip(c["positions"], g["positions"])):
            if (
                cp["position_zero_based"] != i
                or gp["position_zero_based"] != i
                or cp["saved_chosen_id"] != gp["saved_generation_id"]
            ):
                raise ValueError("position/chosen identity differs")
            positions.append(
                {
                    "position_zero_based": i,
                    "saved_generation_id": cp["saved_chosen_id"],
                    "original_failed_position_unknown": True,
                    "new_cpu_full": cp["full"],
                    "new_cpu_kv": cp["kv"],
                    "new_gpu_original_geometry": gp,
                    "new_fp64_math": f["positions"][i] if f else None,
                }
            )
        rows.append(
            {
                "run": c["run"],
                "key": key,
                "positions": positions,
                "new_cpu_full_disagreements": [
                    p["position_zero_based"]
                    for p in positions
                    if not p["new_cpu_full"]["saved_chosen_matches"]
                ],
                "new_gpu_disagreements": [
                    p["position_zero_based"]
                    for p in positions
                    if not p["new_gpu_original_geometry"]["matches_saved_generation"]
                ],
            }
        )
    if sum(len(r["positions"]) for r in rows) != 89:
        raise ValueError("expected89 positions")
    return {
        "scope": "New CPU/GPU/FP64 reconstructions only. Original failed positions/logits remain unavailable; historical strict failures remain unchanged. Generation engine fresh/repeated calls do not recreate original preceding cache trajectory.",
        "gpu_audit_status": gpu["status"],
        "overall_run_status": gpu.get("overall_run_status", "COMPLETE"),
        "overall_run_errors": gpu.get("overall_run_errors", []),
        "positions": 89,
        "rows": rows,
        "generation": gpu["generation"],
        "generation_skip_reason": gpu["generation_skip_reason"],
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("cpu", "fp64", "gpu", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    result = compare(
        *(json.loads(getattr(a, k).read_text()) for k in ("cpu", "fp64", "gpu"))
    )
    result["inputs_sha256"] = {
        k: hashlib.sha256(getattr(a, k).read_bytes()).hexdigest()
        for k in ("cpu", "fp64", "gpu")
    }
    result["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with a.output.open("x") as f:
        f.write(json.dumps(result, indent=2) + "\n")
