"""C2 CUDA leg: coalfire's CUDA-build logits for the C2 ids, against the frozen C2 referee (the independent float64
forward of the HF export in docs/receipts/c2_coalfire_parity_20261103/compare.py, imported unchanged).

  python3 compare_cuda.py HF_DIR IDS CUDA.f32 CUDA_REPEAT.f32 COALFIRE_CPU.f32 PAPERKILN.f32 OUT.json

Gate (exit 1 on failure): the CUDA logits have T x V finite values, every row within the frozen bound of the
referee (max |a - ref| / max(1, max |ref row|) <= 1e-4) and the referee's argmax in every row; and the referee, run
here, still passes the two CPU legs of the C2 receipt (their files are hash-checked inputs). Reported, not gated:
whether the repeat is bit-identical, and CUDA against coalfire's CPU build and against paperkiln.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

FROZEN = Path(__file__).resolve().parents[2] / "docs" / "receipts" / "c2_coalfire_parity_20261103" / "compare.py"
_spec = importlib.util.spec_from_file_location("c2_compare", FROZEN)
c2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c2)


def load(path, rows, vocab):
    a = np.fromfile(path, dtype="<f4")
    if a.size != rows * vocab:
        return None, f"{a.size} values, expected {rows * vocab}"
    return a.astype(np.float64).reshape(rows, vocab), ""


def score(a, ref):
    e = c2.rows_err(a, ref)
    argmax = bool(np.all(a.argmax(1) == ref.argmax(1)))
    finite = bool(np.isfinite(a).all())
    return {"pass": finite and argmax and float(e.max()) <= c2.BOUND, "max_rel_vs_float64": float(e.max()),
            "median_rel_vs_float64": float(np.median(e)), "argmax_identical": argmax, "finite": finite}


def main():
    hf_dir, ids_path, cuda_path, repeat_path, cpu_path, pk_path, out_path = sys.argv[1:8]
    ids = [int(t) for t in Path(ids_path).read_text().split()]
    ref, cfg = c2.referee(hf_dir, ids)
    T, V = ref.shape
    report = {"ids": T, "vocab": V, "bound": c2.BOUND, "referee": str(FROZEN.relative_to(FROZEN.parents[3])),
              "legs": {}}
    arrays = {}
    for name, p in (("coalfire_cuda", cuda_path), ("coalfire_cuda_repeat", repeat_path),
                    ("coalfire_cpu", cpu_path), ("paperkiln_cpu", pk_path)):
        a, why = load(p, T, V)
        if a is None:
            report["legs"][name] = {"pass": False, "reason": why}
            continue
        arrays[name] = a
        report["legs"][name] = score(a, ref)
    gated = ("coalfire_cuda", "coalfire_cpu", "paperkiln_cpu")
    ok = all(report["legs"].get(n, {}).get("pass") is True for n in gated)
    if "coalfire_cuda" in arrays:
        cu = arrays["coalfire_cuda"]
        if "coalfire_cuda_repeat" in arrays:
            report["repeat_bit_identical"] = bool(np.array_equal(cu, arrays["coalfire_cuda_repeat"]))
        for other in ("coalfire_cpu", "paperkiln_cpu"):
            if other in arrays:
                report[f"cuda_vs_{other}"] = {
                    "max_rel": float(c2.rows_err(cu, arrays[other]).max()),
                    "argmax_identical": bool(np.all(cu.argmax(1) == arrays[other].argmax(1)))}
    report["gated"] = list(gated)
    report["pass"] = ok
    Path(out_path).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
