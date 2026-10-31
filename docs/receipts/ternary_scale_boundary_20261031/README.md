# Public ternary scale host boundary

Status: exact-body CPU old-failure/new-pass proof complete; full-object build
and actual CUDA execution pending.

The public `ops::ternary_scale(const Matrix&)` scalar helper reads `get_data()`
without materializing a deferred device value. Its declaration has no host-fresh
precondition. A device-produced Matrix can therefore yield a scale from stale
host bytes. Add the existing materialization and debug host-read boundary before
the unchanged scalar loop. This does not change the mean/epsilon/reduction order.

The internal `ternary_ste` CPU fallback already materializes its input. This is
not evidence of a default QAT training failure or invalid historical experiment.
The public direct helper is the concrete affected interface.

The new test includes fresh-host and all-zero controls. Linux CPU builds
also use a test-only linker materialization seam: the old helper reads zeroed
synthetic stale storage, whereas the fixed helper asks the seam for the original
fixture. This proves the missing boundary without pretending to execute CUDA.
The independent fixture has exact mean absolute value 11/16 and scale 16/11.
The old direct helper returned 100000 with no materialization; the fixed body
returned 1.4545455 after one materialization. Fresh-host controls pass both.
`exact_body_cpu_proof.json` records commands, output and source/archive hashes;
`reproduce_exact_body_cpu.py` preserves the executed script. Only the function
symbol is renamed by the preprocessor to avoid duplicate archive definitions;
the complete function body is extracted verbatim from old/fixed source.
An initially drafted zero-dimensional control was removed because Matrix
constructors correctly reject zero dimensions; it is not part of this API gate.

`test_ternary_scale_boundary --cuda` requires an actual CUDA/deferred producer
(`slice_cols`), proves the input host is stale before and after poisoning only
non-authoritative host bytes, and requires the scalar helper to recover both
the scale and every original value. A prospective old/fixed actual-device pair
remains required before claiming device validation. CPU emulation cannot close
that gate. No paid launch is authorised by this receipt.

## Additional bounded source inspection

The source inventory binds this reading; these are source inspections, not new
CUDA tests and not an exhaustive audit:

- Cerebellum SelectiveGate residual reads consume the fresh host output of
  `ops::sub`, whose input materialization makes the host break explicit.
- MoE top-k logits and Llama recursion routing logits are materialized before
  host sorting/routing reads.
- SRD permutation materializes its input; its residual uses host-only `sub`.
- Recurrence gather/sum host fallbacks materialize their source in forward and
  the upstream gradient in backward.
- Depth attention host fallbacks materialize sources and query in forward,
  and upstream, sources, query and saved alpha/rinv/logit caches in backward.
- Flash-attention host fallback materializes qkv in forward and qkv, output,
  upstream and logsumexp in backward.
- RoPE/repeat-heads fallbacks materialize input and upstream; host RMS row
  forward materializes input and backward materializes upstream. Saved values
  remain host-coherent under the existing immutable-tape contract.

No new defect was established in those paths. This does not cover every helper,
formula consumer, external caller or mutation outside the documented contract.
