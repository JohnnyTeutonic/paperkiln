"""Extract exact production kernels for standalone CUDA compilation."""

import argparse
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
r = Path(__file__).resolve().parents[1]
s = (r / "src/cuda_resident.cu").read_text()
start = s.index("// Transpose-flag GEMM:")
end = s.index("\n}  // namespace", start)
a.out.mkdir(parents=True, exist_ok=True)
(a.out / "gemm_candidate.cuh").write_text(s[start:end])

old = (r / "tests/fixtures/cuda_resident_87ccf28cf.cu").read_text()
start = old.index("// Transpose-flag GEMM:")
end = old.index("\n}  // namespace", start)
expected = (
    "// Exact kernel region from public revision 87ccf28cf src/cuda_resident.cu; names only changed for test isolation.\n"
    + old[start:end]
    .replace("gemm_tiled_ops", "old_gemm_tiled_ops")
    .replace("gemm_tiled_acc", "old_gemm_tiled_acc")
)
if (r / "tests/fixtures/gemm_transpose_old.cuh").read_text() != expected:
    raise ValueError("Frozen baseline region differs from preserved Git blob")
