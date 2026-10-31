# NT-only tiled GEMM load change

The final production change coalesces physical B rows only for NT. NN, TN and TT retain the original loads and shared layout. Accumulator initialization and FMA order remain unchanged.

The earlier exploratory change coalesced both transpose flags. It passed correctness, but its TN paired median old/candidate ratio was 0.866: about a 15% slowdown. That version was not adopted. Its raw results remain in `../gemm_transpose_cuda_20261031/`.

Before the narrowed run, the V2 engineering gate required NT median paired old/candidate time at least 1.05 with at least five wins in seven alternating pairs, and each other mode at least 0.90. This is a scoped engineering decision, not an inferential or whole-model benchmark.

Actual L4, CUDA 12.4, 256 cubed:

| Mode | Median paired old/candidate time |
|---|---:|
| NN | 1.00091 |
| NT | 1.25304 |
| TN | 1.00009 |
| TT | 1.00008 |

All 96 correctness cases match the preserved old kernel bits and pass independent full-array NumPy replay at the unchanged normalized bound 3e-5. Maximum observed error was 8.42113e-7. Cases cover four transpose flags, eight shapes including empty K and edge tiles, and assign/existing-C/split-K modes. The full production translation unit also compiles with local NVCC; the actual device harness extracts its exact production kernel region.

Raw logs, the prospective input manifest, parser and independent parent replay are retained here. No dispatch, cuBLAS, BF16, full-model throughput or model-quality claim follows. Owned paid resources are gone.
