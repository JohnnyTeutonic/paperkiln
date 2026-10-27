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

## Answers

Revision 1.2 (keramos v1.2.0) settles every finding: IR_SPEC section 12.3 gives each answer, by the
numbers above. With the clean-room phase over, the Rust code was brought to 1.2 where the answers
require it (duplicate member names refused, rule 36, steps in one run, `depth_attend` eps optional,
`check` of a document alone); its receipt at v1.2.0 is `conformance/receipts/cleanroom.json`.

## Revision 1.3 and 1.4 pass (27 October 2026)

The engine brought from revision 1.2 (keramos 1d1fd377) to revision 1.4 (v1.4.1), from the text alone
(`READ_LOG.md`, section of the same name). Added: `attention.position_bias` (T5, 5.15, rule 39) and
`attention.relative_kv` (Shaw, 5.15, rule 40) in the kernel, `run_node` and the validator; both structural
(3.8); their tensor names under rules 25 and 17a and read through m_i under `weight_index` (3.7, 7.1, 7.2);
rule 38's label part. Nothing else in 1.3 and 1.4 needed code: the windowed rewind (6.9) holds because the
engine keeps every row; rule 35 under substitution, rules 17a and 36 under depth_route, `tensor_index`,
the hd16 models, the chunks traces and Primer's `pk_flex_mdha` already passed.

**Result at v1.4.1 (a29ce6f, tree at 9b6d5c6), WSL ext4, rustc 1.98.1:** 420 pass, 0 fail, 43 skip, 4 XFAIL
(the runner reports 400 pass, 20 XPASS, 4 XFAIL, since `expected_fail.json` still lists all 24). By kind:
model 27/0/0, trace 81/0/0, op 68/0/0, refuse 101/0/0 (+4 XFAIL), digest 90/0/0, dequant 10/0/0, accept
43/0/0, grad 0/0/11, chat 0/0/32. Every one of the 20 passed on the first run of the new code. Worst errors:
op cases 1.27e-7 (`relative_kv_keys_only`), `pk_flex_shaw` 1.01e-7, `pk_flex_t5` 9.5e-8 (tolerance 1e-5).
Each refusal of rules 39 and 40 cites the rule its case names. Still failing: `architecture_kimi_as_gpt2`,
`_swa_as_gpt2`, `_metadata_disagrees` and `_substituted_document` (finding R1).

Ranked as before, by how much each would trip up the next implementer:

R1. **Rule 38's comparison is not in the text (3.1, 9.3, rule 38).** 3.1 makes "the document section 9.3's
    builder (`tools/gguf_ir.py`, whose graphs are normative for this purpose only) builds from the same
    file's metadata and tensor shapes" the test, and 9.3 says its own graphs are descriptions, not
    normative. A spec-only engine therefore cannot say what graph a gpt2 or llama label means, nor when
    "the metadata builds no graph". Stopped there: only the label part is implemented (below), and the four
    cases above wait. The informative prose would give two necessary conditions (9.2: gpt2's graph has no
    window; kimi under gpt2 computes softmax attention), enough to refuse three of the four cases; they were
    not implemented, since a refusal should be the rule's, not a heuristic's. *Fix:* the text (or a normative
    appendix holding each architecture's document template as data, with metadata placeholders) would have
    to state, for llama, mistral, qwen2, qwen3, gpt2, gptneox, gemma3, mamba and llama with experts:
    (a) the metadata keys read, their types and the value when a key is absent, or that its absence builds
    no graph (embedding_length, block_count, context_length, vocab size or the token count,
    attention.head_count and head_count_kv, key_length, layer_norm_epsilon and layer_norm_rms_epsilon,
    rope.freq_base, rope.dimension_count, rope scaling, attention.sliding_window, expert_count and
    expert_used_count, use_parallel_residual, the ssm keys); (b) the embed, block and head node lists in order,
    each node's op, inputs, attributes as functions of the metadata (with the arithmetic and rounding of
    derived numbers such as the attention scale) and tensor names, and the graph when an optional tensor is
    absent (qwen2's biases, `output.weight` against the tied embedding, gemma3's local and global layers);
    (c) the slots (fields, max_len) and `model` fields; (d) the conditions under which the metadata builds no
    graph (missing keys, tensor shapes that disagree with the metadata); (e) see R3.
R2. **The labels of rule 38 are not written as strings (3.1, 9.3).** Rule 38 allows "an architecture of
    section 9.3"; the section's bullets name llama, mistral, qwen2, qwen3, gpt2, gptneox, gemma3 and mamba,
    while its text says llama.cpp has no architecture "mistral" (the suite labels `tp_mistral` so). Chosen:
    those eight strings and "paperkiln" (`architecture_without_graph`, "phi3", is refused by it). *Fix:* list
    the allowed `general.architecture` strings in rule 38.
R3. **What rule 38's comparison ignores is not complete (3.1).** Even with the builder's graphs, the "up to"
    list does not say whether node order (two independent nodes swapped), the order of `block.state`, or a
    `per_layer` list whose entries are all equal against the same constant are differences. *Fix:* state
    each, one way or the other.
R4. **A file with an embedded document and no `general.architecture` (3.1, rule 38).** Unstated. Chosen:
    refused (an absent label is neither "paperkiln" nor a 9.3 architecture). *Fix:* say so.
R5. **Comparing ne with unit dimensions (rules 26, 28, 39, 40).** "ne {w, K + 1}" with K = 0, "ne {heads, B}"
    with B = 1 or a router with n_rec 1: whether a 1-D tensor ne {w} (or a 3-D {w, 1, 1}) is of ne {w, 1} is
    unstated (3.1 gives GGUF's dimension count, not this). Chosen: trailing dimensions of 1 are ignored, as
    ggml treats them. No case tests it. *Fix:* one sentence in 3.1.
R6. **The members of the two attribute objects (5.15, rule 39, rule 40).** Rule 40 refuses "any other
    member" and 5.15's rule 39 list does not (10.1's "is not {weight, buckets}" implies it); neither says
    that `_` members inside them are prose (3.2 says so at any depth, and the schema allows them). Chosen:
    `_` members ignored, any other member refused, for both. *Fix:* the same words in both lists.
R7. **3.5's defaults omit `attention.relative_kv` (none).** 5.15 gives "absent = none" for both attributes;
    3.5 lists `position_bias` only. *Fix:* add it to 3.5.

Implementation notes (no finding): the T5 bias and Shaw's terms use the positions the kv slot stores (so
p - j is a distance in positions under depth_route, as 5.15 states); `relative_kv` reads key and value
widths independently (head_dim or heads * head_dim each); both together on one node are refused at load
(rule 40) and again at run time. Not a finding of the text but of its informative parts: Appendix C's
revision 1.4 paragraph and the clean-room row still describe this engine at revision 1.2, and
`expected_fail.json` lists 20 cases it now passes.

## Phase 2 (v1.4.2 text, 28 October 2026)

Implemented from the v1.4.2 text: rule 38 in full (`src/arch.rs`: D.1 to D.6's graphs and D.7's normal
form; the eight labels, an absent `general.architecture` refused), ne compared as stored with its dimension
count (3.1; R5's earlier "ignore trailing 1s" withdrawn), `render_chat` (8.1.1, 8.1.3, 8.1.6 with `tools`)
and `parse_reply` (8.1.6, 11.3) in `src/chat.rs`; `hello` advertises both and the profile `chat`. Q16 needed
no code: a recursion step's attention already runs on its active rows at their absolute positions, and the
kv slot stores positions, so the window is counted in positions and a cut keeps the rows of positions < n.

**Result (524 cases; the clone at 9b6d5c6 with the v1.4.2 text and fixtures, WSL ext4, rustc 1.98.1
48a229cea):** 513 pass, 0 fail, 11 skip (the gradient cases), 0 XFAIL. By kind: model 27, trace 83, op 68,
refuse 105, digest 90, dequant 10, accept 44, chat 86 (render 74, parse_reply 12) pass; grad 11 skip. Every
new case passed on its first run. Rule 38 was implemented from Appendix D alone, before any rule 38 case
was run: all 44 acceptance cases (every `tp_` model and the gpt2- and llama-labelled `pk_gpt2`, `pk_llama`,
`pk_moe`) equal their built graphs, and the four comparison refusals differ where their names say:
`architecture_kimi_as_gpt2` and `_substituted_document` at the slot's kind (recurrent against kv),
`_swa_as_gpt2` at the slot's window (4 against 0), `_metadata_disagrees` at the rope node's theta (10000
against `llama.rope.freq_base` 500000). Each chat refusal names the fault its case describes.

Ranked as before. None blocked a case; each is a reading the text leaves to the implementer:

P1. **D.1, the type of each key.** "A key of the wrong type" builds no graph, but no key's type is stated.
    Chosen: counts and sizes any GGUF integer type; epsilons, RoPE bases and the scaling factor any float or
    integer type; `rope.scaling.type` a string; truth values a bool or a number. *Fix:* give each key's
    allowed GGUF types (or "any integer", "any number").
P2. **D.7 step 3, a tied `linear` whose weight holds `{i}`.** "When the file has no tensor named by its
    weight" does not say how a block name is tested (the literal name, or with `{i}` replaced per
    application). Chosen: the literal name; only the head ties in Appendix D, so no case meets it. *Fix:*
    state the substitution, or restrict the step to names without `{i}`.
P3. **D.5, gemma3's 27B scale.** "1/sqrt(d / H)" beside D.1's "the integer part of d / H" for hd: whether
    d / H here is the binary64 quotient or the integer part is unstated. Chosen: the binary64 quotient (D.1's
    "computed in binary64 from the metadata values"). *Fix:* say which.
P4. **D.3, "a llama file" with experts.** Read as `general.architecture` "llama" only (a mistral file with
    `expert_count` gets the SwiGLU graph). *Fix:* name the label.
P5. **D.1, zero or negative sizes.** Only "a head count of 0" builds no graph; a block_count, embedding
    length or kernel of 0, or a negative value in a signed key, builds a graph here (the comparison or other
    rules then decide). *Fix:* state whether such values build a graph.
P6. **8.1.3, the record's members.** Which fields are required, and whether `system`, `user`, `assistant`
    and `stop` refuse members of their own beyond those listed, is not stated (8.1.6 states it for `tool`,
    `tool_call` and `tools`). Chosen: every field without "optional" required; any other member of those
    objects refused as an unimplemented field.
P7. **8.1.1 and 8.1.6, a message's members.** A message without `content` (other than an assistant message
    with calls), a non-string content, a non-boolean `loss` and members the text does not name are
    unstated. Chosen: content a required string, absent read as "" only on an assistant message with calls;
    `loss` a boolean; other members ignored (as 8.1.6 says of a call's).
P8. **11.3, `thinking` false under a record without `generation_prompt_nothink`.** Unstated. Chosen: the
    plain `generation_prompt`.
P9. **8.1.3, `trim` with system mode `turn`.** "Under fold the system text is not trimmed" implies it is
    trimmed under turn, including a `default_system`; no record combines them. Chosen so.
P10. **8.1.3 and 8.1.6, reserved strings and `strip_think`.** Whether a past reply's reasoning that
    `strip_think` would drop may hold a reserved string is unstated. Chosen: the content is checked as
    given, before stripping.

Engine limitation (no finding): a tool definition holding an integer of magnitude 9e15 or more is refused
(this engine keeps numbers as binary64).

### Answers (v1.4.2 text)

The v1.4.2 text settles P1 to P10, and the engine follows it (513 pass, 0 fail, 11 skip at 524 cases):
P1, the keys' types (D.1); P2, a tied weight's name tested as written (D.7 step 3); P3, the binary64
quotient d / H (D.5); P4, the label "llama" only (D.3); P5, only H = 0 builds no graph, so a kv head count
of 0 builds one (D.1); P6, the required fields and their defaults, and which members are refused: those of
`user`, `assistant`, `tool`, `tool_call` and `tools` beyond their own, never those of `system` or `stop`
(8.1.3, "Required fields"); P7, every message has a string `content`, an assistant message with calls
included, and members other than `role`, `content`, `loss` and (under 8.1.6) `tool_calls` and
`tool_call_id` are ignored (8.1.1); P8, the plain generation prompt (8.1.3); P9, the system text of mode
`turn` trimmed, `default_system` included (8.1.3); P10, the reserved check on each content as rendered,
after `trim` and before `strip_think` (8.1.3).
Under any record, v1 included, a `tool_calls` member (even null) is refused without the field `tool` or on a
message that is not an assistant message, and a `tool_call_id` member on a message that is not a tool
message (8.1.1).
