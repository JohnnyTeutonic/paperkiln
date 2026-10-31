# Independent host-helper review: original prospective packet

Reviewed commit 5c329a897 without launching CUDA. The 39 optimized parser tests pass. An additional independent binary64 central finite-difference check of every element in all 14 gradient tensors passes, with maximum normalized error approximately 3.45e-8. The accompanying source snapshot reproduces that check without importing production operators.

One CUDA harness blocker remains in this original packet. At tests/test_host_helper_gate.cpp the loss is mean(mul(y, make_var(weights))). All nine helpers leave y host-fresh, including the materialized inplace outputs, while weights are created host-fresh. Therefore ops::mul selects its host forward/backward branch. mean also constructs a host gradient. Variable::accumulate(Matrix&&) moves that host buffer into y->grad; it does not create a deferred CUDA copy. The interposed callback consequently fails poison(self->grad)'s CHECK(host_stale) before invoking the helper backward. This does not exercise the intended stale upstream-gradient materialization.

A prospective harness-only correction can make the weight operand device-fresh before mul, which selects the existing device mul branch and creates a deferred upstream gradient. That corrected path requires a new review and offline controls; this review does not attest it. The fresh matmul inplace scope is otherwise appropriately limited to an output with no prior consumers and no DevState slot. No numeric reference or full-gradient inventory blocker was found for the nine registered nonboundary fixtures.

No CUDA result exists. No production modification, GPU launch, old-kernel failure, performance, broad audit closure, or full-residency claim is made.

Reproduce the additional check with NumPy installed:

    python independent_fd_check.py.txt /absolute/path/to/repository
