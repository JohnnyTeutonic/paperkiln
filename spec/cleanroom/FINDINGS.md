# Clean-room implementation: findings

A third implementation of the IR, written in Rust (rustc 1.98.1, no crates, about 5,500 lines) from the
specification text alone: `IR_SPEC.md`, `ir.schema.json`, `README.md`, `conformance/` cases and the adapter
contract, and `examples/` as data. Every file opened is listed in `READ_LOG.md`; no file under `tools/` or
`tests/`, and nothing of paperkiln or ember, was read. The conformance runner was used as a black box.

**Result at keramos v1.1.4 (e3b77ec):** 276 pass, 0 fail, 8 skip (`receipt.json`). By kind: model 21/0/0,
trace 37/0/0, op 54/0/0, refuse 74/0/0, digest 80/0/0, dequant 10/0/0, grad 0/0/8 (no training). Worst
log-probability error 1.3e-6 (tp_gemma3; tolerance 1e-5). The first complete run passed 275 of 276 runnable
cases; the one failure was finding 1.

Independent checks of the spec's own claims, all bit for bit: prefill = token-by-token decode = 4/3/4
chunking on all 20 suite models (§4.5); eviction on = off on pk_state (§6.8.3); turn_state without state ids
= plain causal (§6.8.2); the rewritten documents of pk_kimi, pk_srd and pk_mamba equal their originals
(§6.10.6). tp_mamba1 differs by 1.7e-6 (the spec reports 3e-7 for numpy); its trace passes.

Worked from the text alone, first time: the EXACT orders of operations, turn_state codes and eviction counts,
the rewind into an evicted segment, MoR routing and its per-application caches, the recurrence grammar and
fold, dequantisation (§4.7 was complete without consulting ggml), and the adapter's methods.

## Findings, ranked by how much each would trip up the next implementer

1. **§3.10, number serialisation on ties.** "Shortest round-trip digits" has two answers when a value sits
   exactly between two equally short forms; JavaScript (ECMAScript Number::toString) picks the even digit,
   Rust's standard formatter the upper one (…206.25 gives `…206.3`, not `…206.2`). Caught by
   `rfc8785_number_43143ff3c1cb0959`; after the fix the output equals Node's V8 on 220,000 doubles.
   *Fix:* state ECMAScript's tie-to-even rule, or cite its algorithm by section.
2. **§3.10 and §10.1, duplicate member names.** A document with duplicates "stays valid", but which member
   counts is unsaid; parsers keep the last, the first, or reject, so engines can read different graphs from
   one document. Chosen: last wins; no case settles it. *Fix:* refuse duplicates at load.
3. **§3.1 and §3.7, GGUF is not specified** (versions, the 32-byte alignment default, the origin of offsets).
   `pk_state.gguf` and `refuse/state_*.gguf` hold token strings that are not valid UTF-8, which breaks a
   strict reader. *Fix:* a short normative paragraph, including that GGUF strings are byte strings.
4. **§11.2 and §11.3, `run_node` for stateful ops.** Attention gets no kv slot declaration (rule 11 would
   refuse it) and no positions; recurrent ops get no slot declaration; an embedding with `state_weight` gets
   no vocab. Chosen: a kv slot from the node's own attributes, positions 0..T-1, vocab = the weight's rows
   (settled by `op/attention/*` and `op/embedding/state_weight`).
5. **§11.3, the shape of `run_node`'s returned `state`.** Chosen `{slot: {tensor: tensor in declared
   shape}}`, matching the fixtures' `expect_state` (13 cases settled it). *Fix:* say so.
6. **§3.1 and §9.3, documents for third-party GGUFs** are defined only by `tools/gguf_ir.py`; a spec-only
   engine cannot build them (the suite never tests it, since every `tp_` model embeds its document).
   Refused here as a capability.
7. **§5.4-§5.6 and rule 34, "refused at load" checks that need value widths**, with no rule for inferring
   widths at load. Checked at run time here; no case tests load-time detection.
8. **§3.5 and §10.1, reads of unwritten values.** No refusal rule for reading a value no node has written,
   or for an embed section that never writes `x`; that values persist across sections and applications is
   also unstated.
9. **§7.4(b), Mixture-of-Recursions** says how `x` is cut to the active rows, not other values (the
   injection `e`, block values the head reads). Chosen: gather every [T, w] value to the active rows;
   pk_mor reads only `x`, so nothing tests it.
10. **§5.18, `depth_attend` eps.** The formula says "eps 1e-5" and the node has an `eps` attribute. The
    attribute was used; every case uses 1e-5.
11. **§5.19, the lexical rules of `expr`** (names, where a number such as "2e" ends, whitespace) and how
    depth is counted for the 200-deep rule are only in `tools/ir_expr.py`. A first parser counted two
    levels per parenthesis and would have refused 101 nested parentheses.
12. **Refusal rule numbers.** Rule 1 overlaps rules 4, 5 and 8; `refuse/name_with_i_outside_block.json`
    cites rule 9 (should be 17a); `refuse/state_weight_wrong_name.json` cites 28 where 25 applies first;
    `refuse/recurrence_index_size_zero.json` is caught by the schema before rule 31.
13. **No "must accept" check cases.** A validator that refuses everything passes every refusal case. This
    one accepts all 21 suite models, the 4 rewritten documents and the 9 examples, but the suite does not
    prove it.
14. **§3.10, the I-JSON integer range: lexical form or value?** (`9007199254740993.0`, `1e300`). Judged by
    lexical form.
15. **§6.9, `truncate` on a model with recurrent state:** re-run from the start, or refuse? Re-run chosen;
    no trace tests it.
16. **Rule 20, depth_route steps** such as `[0, -1, 1]`: is -1 between two steps allowed? Accepted.
17. **§5.14, rope_heads "first"** is valid but has no reference result (the reference refuses it), and
    whether it applies to k is unsaid. Head 0 of q and k rotated.
18. **§6.10, a recurrence read with an index of another size** (`W[c,c]` read as `W[c,j]`): refuse or
    zero-pad? Zero-pad chosen.
19. **§5.19, folding a constant row reduction:** `rowsum(2)` as w additions or 2 x w? w additions chosen.
20. **JSON numbers to f32:** the conversion of attributes (round to nearest, once) is never stated.
21. **The chat profile (§8.1) has no adapter method and no cases**, so conformance to it cannot be shown.
22. **The adapter contract:** paths arrive in the runner's platform form (Windows paths to a Linux engine),
    so the runner must run on the engine's side; session state after an `error` reply is unspecified
    (pre-call state restored here); also unspecified: `truncate` past the end, reusing a `snapshot` tag,
    the `depth` report without depth_route, an empty list in `run_node` inputs.
23. **For implementers in other languages:** Rust has no f32 `erf`; one was written.

Not implemented: gradients, building documents for GGUFs that do not embed one (finding 6), sidecar
documents (rule 29).
