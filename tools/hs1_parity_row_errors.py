"""Descriptive full-row arithmetic differences; not a historical or golden oracle."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run(receipt, archive, component):
    manifest = json.loads((receipt / "cpu_manifest.json").read_text())
    for n, h in manifest["files_sha256"].items():
        if n.endswith(".f32.gz") or n.endswith("_full.scores.jsonl"):
            if sha(receipt / n) != h:
                raise ValueError("CPU raw binding differs")
    c = json.loads(component.read_text())
    if c["status"] != "SCORER_COMPONENT_AUDITED_OVERALL_ERROR" or c[
        "archive_sha256"
    ] != sha(archive):
        raise ValueError("component binding differs")
    cpu = {}
    for model in ("c12_B_r1", "c12_B_r3", "c12_Sgist_r1", "c12_Sgist_r4"):
        a = np.frombuffer(
            gzip.decompress((receipt / (model + "_full.logits.f32.gz")).read_bytes()),
            dtype="<f4",
        ).reshape(-1, 16384)
        offset = 0
        for row in map(
            json.loads,
            (receipt / (model + "_full.scores.jsonl")).read_text().splitlines(),
        ):
            n = len(row["continuation_ids"])
            cpu[row["key"]] = a[offset : offset + n]
            offset += n
        if offset != len(a):
            raise ValueError("CPU raw row count differs")
    fp = json.loads((receipt / "fp64_report.json").read_text())
    fp64 = {}
    for row in fp["rows"]:
        p = receipt / row["raw_file"]
        if sha(p) != row["raw_sha256"]:
            raise ValueError("FP64 raw binding differs")
        fp64[row["key"]] = np.frombuffer(
            gzip.decompress(p.read_bytes()), dtype="<f8"
        ).reshape(-1, 16384)
    positions = []
    with zipfile.ZipFile(archive) as z:
        m = json.loads(z.read("input_manifest.json"))
        for model in m["models"].values():
            for chunk in model["chunks"]:
                prefix = (
                    next(k for k, v in m["models"].items() if v is model)
                    + "_chunk"
                    + str(chunk["index"])
                )
                a = np.frombuffer(z.read(prefix + ".f32"), dtype="<f4").reshape(
                    -1, 16384
                )
                offset = 0
                for row in map(
                    json.loads, z.read(prefix + "_original.jsonl").decode().splitlines()
                ):
                    if row["key"] not in chunk["target_keys"]:
                        continue
                    key = row["key"]
                    n = len(row["continuation_ids"])
                    gpu = a[offset : offset + n]
                    offset += n
                    if gpu.shape != cpu[key].shape:
                        raise ValueError("CPU/GPU shape differs")
                    for i in range(n):
                        g = gpu[i].copy()
                        g[0] = -np.inf
                        if int(np.argmax(g)) != row["argmax"][i]:
                            raise ValueError("GPU argmax differs from audited original")
                        positions.append(
                            {
                                "key": key,
                                "position_zero_based": i,
                                "gpu_cpu_full_max_abs": float(
                                    np.max(
                                        np.abs(
                                            gpu[i].astype("f8")
                                            - cpu[key][i].astype("f8")
                                        )
                                    )
                                ),
                                "gpu_fp64_max_abs": float(
                                    np.max(np.abs(gpu[i].astype("f8") - fp64[key][i]))
                                )
                                if key in fp64
                                else None,
                            }
                        )
                if offset != len(a):
                    raise ValueError("GPU raw row count differs")
    if len(positions) != 89:
        raise ValueError("expected89positions")
    return {
        "scope": __doc__,
        "overall_run_status": "ERROR",
        "positions": positions,
        "max_gpu_cpu_full_abs": max(p["gpu_cpu_full_max_abs"] for p in positions),
        "max_gpu_fp64_abs": max(
            p["gpu_fp64_max_abs"]
            for p in positions
            if p["gpu_fp64_max_abs"] is not None
        ),
        "inputs_sha256": {
            "archive": sha(archive),
            "component": sha(component),
            "cpu_manifest": sha(receipt / "cpu_manifest.json"),
            "fp64_report": sha(receipt / "fp64_report.json"),
        },
        "source_sha256": sha(Path(__file__)),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("receipt", "archive", "component", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    a = p.parse_args()
    r = run(a.receipt, a.archive, a.component)
    with a.output.open("x") as f:
        f.write(json.dumps(r, indent=2) + "\n")
