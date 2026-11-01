# Independent E9 matched v2 packet review

Disposition: APPROVE for bounded serial admission, not a scientific pass or a launch record.

All six manifest payloads match size and SHA-256, and all eight uploads are regular non-symlink files. The ember archive exactly matches source 3ada969788f4d5ea65f5b81b29c22c30b7f25c97 with tar.umask=0022. Staged checker, wrapper and VM script match working copies. Bash syntax, eight checker tests and 27 failure-audit cases pass; failure coverage includes duplicate keys/hash lines, missing rows, nonfinite metrics, CPU execution and wrong source pins.

Freeze v2 explicitly; original README and verify_stage defaults still point to v1. The runner must enforce $2 / two hours / 30 GB / one L4. Use fresh state, artifact and volume. The wrapper bounds execution at 90 minutes plus kill grace and intentionally returns zero for completed diagnostic negatives. Require both gate_summary.pass and execution.gate_pass exactly true, and replay locally with the frozen checker. Original strict-bound and A1 negatives remain unchanged.

Root guard testing passes one accepted fixture and six refusals (overspend, excessive hours, wrong volume size, wrong GPU, changed pin, already-home state). GPU calls: zero. The prepared plan has no dispatch deadline or launch authorization. DPO retains sole GPU ownership.
