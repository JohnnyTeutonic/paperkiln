# The keramos IR, version 1

*Normative specification of the keramos IR: a versioned JSON graph of a language model that an
engine interprets, rather than recognising a fixed architecture. IR version 1, specification revision
1.6 (28 October 2026; v1.6.0).*

The format's GGUF keys keep their `paperkiln.` prefix (`paperkiln.ir`, `paperkiln.ir_version`,
`paperkiln.ir_sha256` and the rest), as does the architecture label "paperkiln" (rule 38): they are
names in the format, unchanged by the specification's name.

- Repository: keramos (`github.com/JohnnyTeutonic/keramos`), which holds this text, its schema, the
  reference tools, the worked documents and the conformance suite. Paths in this document are
  relative to it, except where the text names them as paperkiln's or ember.cpp's: `src/`,
  `include/microtorch/` and the `tools/` and `tests/` files called paperkiln's are in paperkiln's
  repository (`github.com/JohnnyTeutonic/paperkiln`).
- Schema: `ir.schema.json` (JSON Schema 2020-12, the document's structure).
- Validator: `tools/ir_validate.py` (the schema plus every refusal rule in section 10; its `digest`
  subcommand gives a document's digest, section 3.10).
- Reference interpreter: `tools/ir_numpy.py`, with `tools/ir_expr.py` for `expr`,
  `tools/ir_recurrence.py` for `recurrence` (section 6.10) and `tools/ir_quant.py` for stored tensors
  (section 4.7); `tools/ir_chat.py` renders chat templates (section 8.1.3).
- Conformance suite: `conformance/`, run by `tools/ir_conformance.py` (section 11).
- Worked documents: `examples/*.ir.json`.
- Implementations: paperkiln (exporter, trainer and `mtstudio`; CPU and CUDA) and ember.cpp
  (`graph_ir.cpp`, `graph_ir_cuda.cu`; CPU, CUDA, WebAssembly).
- Contracts between paperkiln and ember.cpp (outside this specification) and the log of gates:
  `LOCAL_LLM_STACK.md`, the master document of the workspace that holds both engines (sections 3 to
  5 and 10).
- Revisions and the repository's tags: section 2.6.

The C++ API that writes documents is paperkiln's `include/microtorch/ir.hpp`. That header documents functions,
not semantics: when the header and this document differ, this document holds.

## Contents

0. Status
1. Conventions
2. Versioning, profiles and change control
3. The document (core)
4. Numerics (core)
5. Core ops
6. The state profile
7. The recursion profile
8. The chat profile
9. Families (informative)
10. Validation
11. Conformance
12. Questions: settled and open
- Appendix A. The `-ffast-math` audit
- Appendix B. Changelog
- Appendix C. Engine capabilities
- Appendix D. The third-party graphs

## 0. Status

This document is version 1 of the IR, revision 1.6. It states the semantics that
`include/microtorch/ir.hpp`'s header comment stated under the number 0, reorganised into profiles,
with the numerics made consistent with what the reference build computes (section 4 and Appendix A).
Revision 1.1 adds the digest (section 3.10), the values of stored tensors (section 4.7) and the
`recurrence` profile (section 6.10), and settles the questions that were open in 1.0 (section 12).
Revision 1.2 states what a third implementation, written in Rust from this text alone (`cleanroom/`),
found the text left to the reference: the container (section 3.1), duplicate member names and reads of
values nothing writes (refused, rules 35 and 36), what is checked at load and what when the rows arrive
(section 10.1.1), the lexical rules of `expr`, the defaults of the adapter's single-node runs, and the
adapter's edge cases; its suite adds the documents every consumer must accept and the chat profile's
rendering (section 11). Revision 1.3 states what the first engine that reads no document, llama.cpp
(`adapters/llamacpp/`, which builds its graph from a file's architecture and metadata), showed the text
left open: a file's architecture must be the graph its document is, so that such a program refuses an
IR-only model instead of computing another one (section 3.1, rule 38); a rewind into a sliding window
(section 6.9); llama.cpp's handling of Mistral (section 9.3). It adds T5's relative position bias, an
optional attribute of `attention` (section 5.15, rule 39). Revision 1.4 adds the first primitive written
under the workspace's path for new primitives (`LOCAL_LLM_STACK.md` section 5.4): Shaw-style relative key and
value vectors, an optional attribute of `attention` (section 5.15, rule 40), with gradient cases for engines
that train (section 11.2); it states Primer's multi-dconv-head attention as `conv1d_causal` on the fused
projection beside a kv slot, with no new op (sections 6.9 and 9.2). Its patch v1.4.2 states what a second
pass of the clean-room implementation found the text left to the tools (rule 38's graphs, Appendix D),
settles windows under Mixture-of-Recursions (Q16: positions, section 5.15) and states the chat profile's
schema v2, `paperkiln/2`, as built (section 8.1.6). Revision 1.5 adds what the later
third-party families need (`LOCAL_LLM_WORK.md` E4; section 12.7): Gemma 2's attention soft-capping, an
optional attribute of `attention` (section 5.15); LongRoPE's per-pair frequency divisors, with a short and a
long set and the table's attention factor, optional attributes of `rope` (section 5.14, rule 41); a
per-group `split` (section 5.5); and DeepSeek-V2's multi-head latent attention, a new op over a kv slot whose
row is the compressed latent and the shared rotary key (section 5.20, rule 42). Section 9.3 gains phi3,
gemma2, qwen2moe, qwen3moe and deepseek2, whose mixture-of-experts layers are wiring of `moe_ffn`. Revision 1.6
lets real DeepSeek checkpoints build (section 12.8): a node attribute `apply` that runs a block node in some
applications only (section 3.8; Q17, leading dense layers), DeepSeek-V3's router as `moe_ffn` attributes
(sigmoid scores, a selection bias, group-limited routing; section 5.16), and YaRN's per-pair divisors as
`freq_factors` given by value (section 5.14). The IR documents
that paperkiln commits before 5349ee19 exported carry `"ir_version": 0` and are version 1 documents
(section 2.1); later exports carry 1 (section 2.3).

ember.cpp reviewed this text against its engine (25 October 2026). Where the engine and this document
differ, this document states the intended semantics and the engine changes. Section 12 records each
question that review settled, with its answer, and the questions still open.

## 1. Conventions

### 1.1 Terms

- **Document**: one IR JSON object (section 3).
- **Engine**: a program that runs documents (paperkiln's `mtstudio`, ember.cpp, the numpy reference).
- **Value**: a named intermediate result of the graph: a tensor of shape [T, width] or a list of
  such tensors (section 3.6).
- **Row**: one token's entry of a value. T is the number of rows a node runs on.
- **Application**: one run of the block's nodes. Application i is block index i,
  i = 0 .. block.repeat - 1.
- **Slot**: a named piece of decode state declared in `block.state` (section 6).
- **f32**: IEEE 754 binary32 with round-to-nearest-even. **Rounded** means rounded to f32 after that
  operation, with no fused multiply-add.
- **Fused multiply-add** (FMA, contraction): computing a * b + c with one rounding.

### 1.2 Normative language

Statements in sections 2 to 8, 10 and 11 are normative unless marked *informative*. "Is refused"
means a consumer must reject the document at load, naming the rule (section 10). Section 9 and
Appendices A to C are informative. Appendix D is normative where section 3.1 cites it: for rule 38, and
for an engine that builds the document of a GGUF without one.

## 2. Versioning, profiles and change control

### 2.1 ir_version

`ir_version` is an integer, the major version of the document format. This specification defines
version 1.

Version 0 is the number written before this specification existed. A version 0 document is a
version 1 document: no document that was valid under version 0 means anything different under
version 1. The validator checks this claim against every stored document (section 10.2).
Consumers treat 0 and 1 as the same format.

In a GGUF the version is also stored as the uint32 key `paperkiln.ir_version`, equal to the
document's `ir_version` (section 3.1).

### 2.2 What changes the version

- **A revision** (1.0, 1.1, ...) adds without changing meaning: a new op, a new optional attribute
  whose absence keeps the old behaviour, a new enum value, a new profile, or a clarification.
  It does not change `ir_version`. A document that uses an addition is refused by an older engine
  through the closed-set rules (section 10.1: unknown ops, attributes and values are refused by
  name), so no revision number is needed inside the document. Revisions are listed in Appendix B.
  A revision may also refuse a document that has no defined meaning: one whose meaning depends on
  which of two equal member names a parser keeps (rule 35), or one that reads a value no node writes
  (rule 36). No meaning changes, since there was none to keep, and no stored document is of either kind.
  A revision may also refuse a container whose metadata states another model than its embedded
  document (rule 38, revision 1.3): the document's meaning is unchanged, the same document in a file
  labelled as section 3.1 requires (or read in place of the embedded one) stays valid, and a file
  written before the rule is relabelled without touching its document or tensors
  (`tools/gguf_relabel.py`).
- **A major version** (`ir_version` + 1) is needed for any change that alters the meaning of a
  document that was valid before: a changed default, a changed evaluation order of an EXACT op, a
  changed tie rule, a removed op or attribute, or a changed name rule.

### 2.3 Unknown versions

An engine refuses a document whose `ir_version` it does not implement, at load, naming the
version and the versions it reads. An engine that implements version 1 reads 0 and 1.

An exporter writes a version number only once every engine it targets reads it.
- ember.cpp reads 0 and 1 as the same document and refuses any other version from ce41ffb on;
  earlier ember commits read 0 only (`IR version N is not supported (ember reads 0)`).
- paperkiln's exporters (`ir::kVersion`) and `tools/gguf_ir.py` (`IR_VERSION`) write 1 from
  5349ee19 on; earlier commits write 0. ember's C++ builder follows.
- Campaigns pinned to earlier paperkiln commits keep exporting 0 with their own code, so nothing
  running changes. Tiny-chat post jobs from b5048dd3 on generate with ember aa50a6112dfd, which
  reads 1.
- A pinned deployment runs its own pinned exports. A campaign or Space pinned to an ember commit
  that reads 0 only (69d7ca2 or earlier) serves the files that its pinned paperkiln commit exported,
  which carry 0, so a later exporter writing 1 does not reach it. A file written with 1 must not be
  served by such an ember.

### 2.4 Profiles

Every op and key belongs to one profile. An engine states the profiles it implements; within a
profile it may still lack a kernel for an op on some backend, and then refuses the document at load
naming the op and the backend.

| Profile | Contents | Sections |
|---|---|---|
| `core` | the document, tensors, per-layer attributes (from revision 1.6 the node attribute `apply`), prefill and decode, numerics and classes, the ops of section 5 except those marked otherwise, the kv slot with window and sinks | 3, 4, 5, 6.1, 6.2 |
| `state` | recurrent slots and ops (`linear_attention`, `ssm_scan`), per-turn state tokens (`state_weight`, `mask: turn_state`, `evict: turn_state`), prefix reuse, rewind and snapshots | 6 |
| `recursion` | `weight_index`, `tensor_index`, `depth_route`, `kv_rows: active`, `kv_from` (reserved) | 7 |
| `chat` | the template record and its resolution; the state-token template contract | 8 |
| `thirdparty` | the attributes and ops added for graphs built from llama.cpp metadata: grouped-query attention, `rope` `neox` / `n_rot` / `freq_scale` / `kv_heads`, `rmsnorm` `group` / `weight_offset`, `conv1d_causal`, `selective_scan`; from revision 1.5 `rope` `freq_factors` / `attn_factor`, `attention` `softcap`, `split` `groups`, `mla_attention`; from revision 1.6 `moe_ffn` `scoring` / `selection_bias` / `groups` and `freq_factors` given by value | 5.5, 5.6, 5.14, 5.15, 5.16, 5.20, 6.6, 6.7, 9.3 |
| `recurrence` | the `recurrence` op: a per-token state update declared in the document (init, let, update and readout bodies), over a recurrent slot of the state profile | 6.10 |

Profiles are not declared inside the document (an engine infers them from the ops and keys it
meets). Each engine's adapter for the conformance suite states its profiles (section 11.3).

### 2.5 Change control

A change reaches this document in this order: draft text reviewed by the other engine; the numpy
reference; both engines passing the change's gates; then this document and Appendix B, in the
commit that lands it. A draft is never normative.

### 2.6 Revisions and tags

Each revision of this document is a tag of the keramos repository, `vR.S.P`. R.S is the revision
(section 2.2; R is the `ir_version` it defines), so `v1.1.0` is revision 1.1 of IR version 1. P
counts later commits that change no valid document's meaning, no part of `ir.schema.json` and no
expected value of an existing case: wording, clarifications that state what the text already
requires, a fix that brings the reference into line with the text, new cases for such statements,
paths, the tools' interfaces, tests and receipts. A new revision or major version starts at P = 0. An engine names the tag and the full commit it conforms
to, in its receipt (`conformance/receipts/`); an engine that vendors this repository pins one tag
and records its commit.

## 3. The document (core)

### 3.1 Container

- In a GGUF, the compact JSON dump of the document is the string key `paperkiln.ir` and the version
  is the uint32 key `paperkiln.ir_version`, both written after every other metadata key. The string
  key `paperkiln.ir_sha256`, when present, follows them and holds the embedded document's digest
  (section 3.10): 64 lowercase hex digits. It is optional; a file whose digest key is not the digest
  of its embedded document is refused (rule 30). paperkiln's exporters write it.
- **The architecture** (revision 1.3). `general.architecture` of a file with an embedded document is
  the string `"paperkiln"` or one of the eight architectures of section 9.3, `"llama"`, `"mistral"`,
  `"qwen2"`, `"qwen3"`, `"gpt2"`, `"gptneox"`, `"gemma3"` and `"mamba"` (a file with an embedded document
  and no `general.architecture` key is refused), and it is one of the eight only when the embedded
  document is that architecture's graph: the document Appendix D builds from the same file's metadata
  and tensor names (normative for this purpose; `tools/gguf_ir.py` implements it), up to the names of
  values and slots, `_` members, the informative keys (`family_hint`, `tokenizer`, `inputs`, `outputs`,
  `modes`, `ir_version`) and attributes (`attention.softmax`, `moe_ffn.routing`), optional attributes
  written at their defaults (section 3.5), the order of the two operands of `add` and of `mul`, the
  spelling of numbers (each the f32 it rounds to) and whether a `linear` reaches the tensor it reads
  through `tied_to` (section 5.3). Names of values are compared by the write that makes each value:
  every write is a new value, and a read names the latest write before it in the order embed, the
  block's nodes once, head; so a value the block carries from one application to the next (the stream
  `x`) is named, where the block reads it before writing it, by the write that application 0 reads (the
  embed's), in both documents alike. `rmsnorm.group` is compared as written, absent against absent (its
  default, the row width, would need the widths section 10.1.1 does not infer). Nothing else is
  ignored: the nodes of each section and the slots of `block.state` are compared in order (two
  independent nodes written the other way round differ), and a `per_layer` object is never equal to a
  number, even when every entry is that number. Appendix D.7 states the comparison step by step. Every
  other file has the architecture `"paperkiln"`, whatever tensors it shares with a llama.cpp
  architecture. No llama.cpp release builds a graph for `"paperkiln"`, so a program that builds its
  graph from the architecture and its metadata and never reads the document (llama.cpp and the front
  ends built on it; transformers through its GGUF reader or any converter to an HF checkpoint) refuses
  such a file instead of computing another model; and where a file does
  name a llama.cpp architecture, its metadata and its document describe one model, so such a program
  computes the document's model, up to its own deviations from that architecture's graph (Appendix C).
  A file that breaks this is refused (rule 38). Rule 38 judges the file's embedded document, whether or
  not another document is read in place of it, and never the document read in place: that one is not in
  the file such a program reads, while the embedded one still is (refusal
  `architecture_substituted_document`: a valid kimi document read in place of the kimi document of a
  gpt2-labelled file is refused). `tools/gguf_relabel.py` relabels a file written before the rule: general.architecture
  becomes `"paperkiln"` and the old architecture's keys are renamed `paperkiln.*`; the document, its
  digest and the tensors are unchanged.
- A sidecar `<stem>.ir.json` beside the GGUF may hold the same document. When present, it must be
  the embedded document; a consumer that reads sidecars refuses one that differs.
- **The GGUF file.** The container is GGUF version 3, little-endian (version 2 has the same layout, and
  a reader may accept it): the magic `GGUF`, the uint32 version, the uint64 tensor count and the
  uint64 metadata count; each metadata key (a string), its uint32 value type (0 to 12, as ggml's
  `gguf_type`) and value; then each tensor's name, uint32 dimension count, uint64 dimensions (ne0
  first), uint32 ggml type and uint64 offset. A string is a uint64 byte length and that many bytes:
  a byte string, which need not be UTF-8 (tokenizer vocabularies hold invalid sequences, as
  `pk_state.gguf` does); only `paperkiln.ir` must be UTF-8, being a JSON text. The alignment is the
  uint32 key `general.alignment`, or 32 when it is absent. The tensor data section starts at the first
  multiple of the alignment at or after the end of the tensor infos, and a tensor's offset counts
  from the start of that section (each a multiple of the alignment). A tensor's values are section
  4.7's for its type. A tensor's ne is its stored dimension list, the dimension count included: a rule
  that states an ne (rules 26, 28, 39 and 40) compares the list as stored, so a stored ne {w} is not
  ne {w, 1}. An exporter writes every dimension of such a tensor, those of size 1 included (paperkiln's
  do), even where ggml's `ggml_n_dims` would count fewer.
- **A GGUF without `paperkiln.ir`** (a third-party file) holds no document. Building one from its
  `general.architecture` and metadata is a capability (section 10.3): an engine may refuse such a
  file at load, naming the capability. What an engine that builds one runs is the built document, as
  it would run any document. The document it builds is Appendix D's for the file's architecture
  (`tools/gguf_ir.py` is the reference builder; section 9.3 summarises the graphs). The conformance
  suite tests the graphs through rule 38: every `tp_` model embeds the document Appendix D builds from
  its own metadata, and rule 38's refusal cases embed documents it does not.

### 3.2 Top-level keys

A document is a JSON text (RFC 8259) of one object, in UTF-8. No object in it, at any depth and `_`
members included, holds a member name twice: a document with a duplicate member name is refused
(rule 35), whatever the two members hold and whichever of them a parser would keep, so every engine
reads one graph from one text. Rule 35 judges the document read: when a valid document is read in place
of a file's embedded one (ember's `EMBER_IR`, the suite's `ir`), a duplicate member name in the embedded
document is not a refusal (accept case `substituted_over_duplicate_embedded`); the container rules that
read the embedded document (24 and 38) parse it as JSON keeping the last of two equal member names. The
object has these keys:

| Key | Required | Meaning |
|---|---|---|
| `ir_version` | yes | integer (section 2.1) |
| `family_hint` | yes | a label naming the family (section 9.1); informative, never semantics |
| `model` | yes | section 3.3 |
| `tokenizer` | no | section 3.4 |
| `inputs`, `outputs`, `modes` | no | section 3.4 |
| `embed` | yes | `{"nodes": [...]}` |
| `block` | yes | `{"repeat", "state", "nodes"}` and optionally `"weight_index"` (section 7.1) |
| `head` | yes | `{"nodes": [...]}` |

Keys starting with `_`, at any depth, are prose for humans and carry no semantics. Every other key
is closed: a key this specification does not define is refused (section 10.1).

### 3.3 model

`model` is `{vocab, d, n_layers, ctx, dtype}`:
- `vocab`: the number of token ids the head produces logits for (ids 0 .. vocab - 1).
- `d`: the width of the residual stream.
- `n_layers`: equals `block.repeat`, the number of applications.
- `ctx`: the context length. Under the context contract (`LOCAL_LLM_STACK.md` section 3 point 5),
  generation stops at the limit, so no position reaches ctx.
- `dtype`: the compute dtype, `"f32"`.

An engine loading a document that is not a llama graph needs `model.d`, `model.vocab`, `model.ctx`,
`model.n_layers`, `block.repeat` and `block.state` and nothing else from outside the graph.

### 3.4 tokenizer, inputs, outputs, modes

- `tokenizer` describes the vocabulary the file carries: `type` (a label), `source` (`"gguf"`: the
  file's tokenizer keys), `bos` (an id or null) and `specials` (role names to ids, e.g. `unk`,
  `eot`, `user`, `assistant`). Only `specials` has semantics (section 8.2).
- `inputs` names the graph's inputs: `tokens` (the token ids) and `pos` (absolute positions:
  prefill 0 .. T-1, decode = tokens seen so far). Either a list of names or a list of
  `{name, kind, note}` objects. The block reads positions as the value `pos`.
- `outputs`: `logits` (a list of names or of `{name, kind}`). The head must write the value
  `logits`, [T, vocab] (rule 36).
- `modes`: prose describing prefill and decode; informative.

### 3.5 Sections and nodes

The graph is three node lists: `embed` (run once), `block` (run `block.repeat` times) and `head`
(run once). A node is an object:
- `op`: the op name (sections 5 to 7). An op this specification does not define is refused.
- `in`: the names of the values it reads, in order.
- `out`: the name it writes (a list of names for `split` and `rope`).
- the op's attributes. A node in the block names its slot, if it has one, in `state`.

Nodes run in order. A later write to a name replaces the earlier value. The embed section reads
`tokens` and must write `x`; the applications carry the residual stream (`x`, or under attention
residuals a list value, section 9.2); the head writes `logits`.

- **Reads.** A value stays readable, by later nodes, later applications and later sections, until a
  node writes its name again: the head reads the last application's values. Every value a node reads
  is written before it (or is `tokens` or `pos`): the sections are checked in order, embed, block and
  head, with the block as application 0, which reads nothing an application before it wrote. A node
  that reads a value no earlier node writes, an embed section that writes no `x` and a head that
  writes no `logits` are refused at load (rule 36). Under `depth_route` the scope is narrower
  (section 7.4(b)).

- **Required attributes.** The attributes each op's section names without a default are required,
  and `ir.schema.json` lists them (`required` of each `op_*` definition). They have no default:
  a node that lacks one is refused, whatever value an engine would assume.
- **Optional attributes and their defaults:** `linear.bias` and `scale.bias` null; `linear.tied_to`
  none; `embedding.state_weight` none; `rmsnorm.group` the row width and `rmsnorm.weight_offset` 0;
  `rope.kv_heads` = heads, `rope.n_rot` = head_dim and `rope.freq_scale` 1; `attention.mask`
  `"causal"`, `attention.position_bias` none and `attention.relative_kv` none; `depth_attend.eps` 1e-5; `conv1d_causal.bias` none; `selective_scan.d` none and
  `selective_scan.dt_softplus` true; a kv slot's `dtype` `"f32"`, `evict` and `kv_rows` absent.
- **Numbers.** A float attribute (`eps`, `theta`, `scale`, `shift`, `freq_scale`, `den_floor`,
  `weight_offset`, ...) is the JSON number's binary64 value (its nearest double, as JSON readers
  parse it) rounded once to the nearest f32, ties to even; an integer attribute is that integer.
  The rounding from decimal through binary64 to f32 can differ from rounding the decimal to f32
  directly in rare halfway cases; the two-step value is the attribute's (paperkiln's exporters
  write the shortest decimal of each f32, `ir::f32`, which both readings return exactly). A number
  inside `expr` and recurrence bodies is read differently: directly as the nearest f32 (section 5.19).
- **The graph inputs.** `tokens` is read by `embedding` nodes only, and `pos` only as `rope`'s third
  input and `pos_embedding`'s second; no node writes either name. Positions are always the absolute
  positions of section 3.4.
- **Names.** A tensor name holds `{i}` at most once (section 3.7).

### 3.6 Values

Tensors flow as [T, width] row-major matrices of f32: row t is token t. A `list` value (section
5.17) holds tensors in order; it is a value like a tensor (named by `out`, read by `in`), never a
GGUF tensor and never state.

### 3.7 Tensors and names

- `weight` (and `bias`, `router`, `experts`, `state_weight`, the `weight` of `attention.position_bias`,
  the `key` and `value` of `attention.relative_kv`, `moe_ffn.selection_bias`, `rope.freq_factors` (its name, or its `short` and
  `long`), `mla_attention`'s `k_b` and `v_b`, and the tensor attributes of `ssm_scan`,
  `conv1d_causal` and `selective_scan`) name GGUF tensors; `"{i}"` in any of them follows the rule below.
- Layout: a matrix is [out, in] row-major, as GGUF and llama.cpp store it (ggml `ne0` = in).
- Storage: each tensor's storage type is its GGUF type (f32, f16, q8_0, ...), and its values are the
  f32 values section 4.7 defines for that type. Engines compute in `model.dtype`.
- `"{i}"` in a name in the block is replaced by the block index i (by m_i under `weight_index`,
  section 7.1; by the application index under `tensor_index`, section 7.2). A name holds `{i}` at
  most once, and only in the block: outside the block it has no value, and a name holding it there
  is refused.
- Every tensor a document names must be a tensor of the file, except a `linear` weight with
  `tied_to` (section 5.3).

### 3.8 Per-layer attributes

Any attribute of a block node, and any field of a block state slot, may be
`{"per_layer": [v_0, ..., v_{L-1}]}` with L = `block.repeat`: application i reads v_i, except the
structural keys: a node's `op`, `in`, `out`, `state`, `tensor_index`, `sizes`, `groups`, `expr`, `position_bias`,
`relative_kv`, `freq_factors`, `moe_ffn`'s `groups`, a
recurrence's `index`, `inputs`, `weights`, `init`, `let`, `update` and `readout`, and its
tensor names (`weight`, `bias`, `router`, `experts`, `state_weight`, `tied_to`, `a`, `b`, `c`, `d`, `k_b`, `v_b`,
`selection_bias`),
and a slot's `name`, `kind`, `op`, `dtype`, `tensors`, `evict` and `kv_rows` (`kv_from`, reserved, is
per-layer by definition). An attribute not given per layer is the same in every application.
(Examples: gemma3's attention window and kv window, 512 on local layers and 0 on global ones; RoPE theta 10000 / 1e6; `depth_route`'s `step`; the injection
`affine`'s `scale`.) A `per_layer` list whose length is not `block.repeat` is refused, as is a
`per_layer` value outside the block.

**apply** (revision 1.6; Q17, section 12.8). A block node may carry `apply`, true or false, or `{"per_layer":
[b_0, ..., b_{L-1}]}` of trues and falses (default true). In an application whose value is false the node does
not run: it reads nothing, names no tensor (section 3.7 requires its tensors only in the applications that run
it) and writes nothing, so its outputs keep whatever value an earlier write left (section 3.5's persistence).
This states a block whose applications differ in their nodes, as a model whose first layers are dense and the
rest mixtures of experts (DeepSeek-V2-Lite's first layer, DeepSeek-V3's first three): both branches are listed,
each applied where it belongs, and both write the value the next node reads. The reads are checked per
application (rule 36): in every application each value a running node reads is written before it, in that
application or an earlier one, or by the embed section. Stateful nodes (`attention`, `mla_attention` and the
recurrent ops of section 6) and `depth_route` do not carry `apply` (rule 4: their slots run in every
application), and a block with `depth_route` has no `apply` (rule 43). Refused (rule 43): `apply` outside the
block, a per-layer entry that is not true or false, `apply` in a block with `depth_route`.

### 3.9 Prefill and decode

Every op has the same semantics on T rows (prefill) and on one row (decode, T = 1). In decode,
attention also reads and appends to its kv slot (section 6.2), and a recurrent op reads and replaces
its recurrent slot (section 6.3).

### 3.10 The digest

A document's digest identifies it exactly, so a paper can cite the document it used. It is the
SHA-256 of the document's canonical form:

1. The whole document, its `_` members included, is parsed as I-JSON (RFC 7493): no duplicate member
   names, every number a finite IEEE binary64 value, every integer within -2^53 .. 2^53, every
   string and member name well-formed Unicode, and no negative zero in floating form (below). A
   document that fails any of these checks has no digest, whatever its `_` members hold. (A
   document with a duplicate member name is refused as a document as well, rule 35.) The integer
   range is judged by the number's lexical form: an integer is a number token written without a
   fraction or an exponent (`-7`, `9007199254740993`), and it must lie within -2^53 .. 2^53. A token
   with a fraction or an exponent is its nearest binary64 value and need only be finite, so
   `9007199254740993.0` is the double 9007199254740992 and has a digest, as does `1e300` (vectors
   `fraction_form_beyond_2p53`, `exponent_form_1e300`).
2. Every member whose name starts with `_`, at any depth, is removed (prose, section 3.2).
3. What remains is serialised by the JSON Canonicalization Scheme, RFC 8785: no whitespace, member
   names sorted by their UTF-16 code units, strings as ECMAScript's `JSON.stringify` writes them
   (only `"`, `\` and the controls U+0000 .. U+001F escaped), numbers as ECMAScript's
   `Number.prototype.toString` writes the binary64 value (shortest round-trip digits; `1e-05` is
   written `0.00001`, `1e21` as `1e+21`). The digits are those of ECMA-262 (15th edition, 2024)
   section 6.1.6.1.20, Number::toString, with the alternative step 5 of its Note 2, which V8 and
   every engine RFC 8785 tests follow: the fewest digits k whose value rounds to the double; among
   those, the digit string s whose value s x 10^(n-k) is closest to the double; and when two are
   equally close, the one whose s is even. A double that lies exactly halfway between two shortest
   forms is therefore written with the even last digit, where a formatter that rounds halves up
   writes the odd one (vector `rfc8785_number_43143ff3c1cb0959`: the double 1424953923781206.25 is
   written `1424953923781206.2`, not `1424953923781206.3`).
4. The digest is the SHA-256 (FIPS 180-4) of that text's UTF-8 bytes, written as 64 lowercase hex
   digits. A citation may prefix it with `sha256:`.

- **What it identifies.** Two documents have the same digest exactly when they are equal as I-JSON
  values after step 2: member order, whitespace, number spelling (`1e-05`, `0.00001`, `1.0E-5`) and
  prose do not matter; array order does. `ir_version` is part of the document, so a version 0 file
  and its version 1 re-export (section 2.1) have different digests. A number is its binary64 value: two
  documents whose doubles differ but round to the same f32 attribute are different documents
  (paperkiln's exporters write each f32 attribute as the shortest decimal of the float, `ir::f32`).
- **Documents without a digest.** RFC 8785 writes a negative zero as `0`, in integer form (`-0`) or in
  floating form (`-0.0`, `-0e0`). The integer form is the integer zero: it serialises as `0` and the
  document has a digest (vector `minus_zero_integer_form`). In floating form, -0 and +0 are different
  values of an EXACT op (an `affine` shift of -0 keeps the sign of a zero product, +0 does not;
  section 4.2), so a document that holds a floating negative zero has no digest, nor does one outside
  I-JSON (step 1). Such a
  document stays valid; it is refused only as the subject of a digest (`DigestError`), and no
  paperkiln exporter writes one (every stored document has a digest).
- **Container.** `paperkiln.ir_sha256` (section 3.1) is the digest of the document embedded in the
  file. A document read in place of the embedded one (ember's `EMBER_IR`) has its own digest; the key
  is checked against the embedded document only.
- **Implementations.** `tools/ir_validate.py` (`canonical`, `digest`, and
  `python tools/ir_validate.py digest FILE`) is the reference; `ir::canonical` and `ir::digest` in
  `include/microtorch/ir.hpp` are paperkiln's C++, which the exporters use. The vectors are
  `conformance/digest/vectors.json` (section 11): RFC 8785's number table (Appendix B of the RFC),
  strings and member names that exercise the escapes and the UTF-16 order, every suite model's
  document, and texts that have no digest. The reference serialiser agrees with V8's own
  `JSON.stringify` and member sort on 23,216 numbers (random binary64 patterns, every decimal exponent
  from -330 to 308), on those strings and on all 24 documents then stored in `examples/` and paperkiln's `experiments/`
  (checked 25 October 2026), and on the 20 suite models' documents.

## 4. Numerics (core)

### 4.1 Arithmetic

Every op computes in f32. Where an op's text gives an order of operations and says "rounded", each
operation it names is rounded to f32 before the next. Where the text says "rounded" or "no fused
multiply-add", a * b + c is the product rounded and then the sum rounded; an engine must not
contract it (a C or C++ build pins this with `-ffp-contract=off` or an equivalent, whatever
`-march` the build adds).

### 4.2 Classes

Each op has a parity class:
- **EXACT**: bit-identical results, given bit-identical inputs, in the reference and in every
  conforming interpreter path.
- **ROUNDING**: within tolerance (section 4.3) of the reference.

An op is EXACT only where this document pins the evaluation order completely and the engines
follow it. The class of each op is stated with the op.

- **Signed zero.** EXACT includes the sign of zero: -0 and +0 are different results. IEEE 754 gives
  each primitive's sign (x + -0 = x, (-0) + (-0) = -0, (+0) + (-0) = +0). An accumulation stated
  "from 0" starts at +0, so a sum whose every term is -0 is +0.
- **NaN.** A NaN result matches any NaN: payload and sign bit are unspecified. Where an op's text
  defines its result on a NaN input (`relu`, `max`, `min`, `where` below), that result is EXACT.

The classes govern the IR's reference semantics and conforming interpreters. A backend or a tuned
path (a device kernel set, a fast path an engine picks by structural match) is classified separately
per engine, in the engine's capability table (Appendix C): a path that is ROUNDING by construction
(tiled or flash kernels, contraction by default on a device compiler) is judged at ROUNDING on every
op, EXACT ones included, and still conforms.

### 4.3 Tolerances

A ROUNDING result passes when, per tensor, |a - b| <= tol * max(1, max|ref|), with tol = 1e-5 on
CPU and 5e-5 on CUDA. A path that runs its matrix products in TF32 is looser and is judged on greedy
parity; which paths do is stated per engine in Appendix C.

- The whole-model gate is greedy parity (`LOCAL_LLM_STACK.md` section 3 point 1): the same greedy
  tokens on the frozen prompts.
- For official third-party checkpoints the gate is greedy N/N plus top-1 at every position, and
  logit closeness is judged against a float64 evaluation of the same document (`tools/ir_numpy.py`
  with float64 arithmetic), because float32 conditioning grows with depth.

### 4.4 Stated orders

An op's text may give an order of accumulation (for example "in ascending j"):
- for an EXACT op the order is normative, and part of what makes it EXACT;
- for a ROUNDING op it is the reference order: the order the numpy reference follows and an engine
  should follow where it can. An engine that sums in another order (a vectorised, tiled or blocked
  reduction) conforms when it stays within the ROUNDING tolerance.

### 4.5 Conditional exactness

Some statements are bit for bit only under a condition stated with them:
- **Same kernel.** A statement that two configurations give the same values bit for bit holds when
  the engine computes both with the same reduction order: eviction against keeping every row
  (section 6.8.3), `turn_state` on ids without a state id against plain causal attention (section
  6.8.2).
- **Same inputs.** A recurrent op's prefill and token-by-token decode leave the same state bit for
  bit given bit-identical inputs (section 6.3). The inputs come from earlier ops, often `linear`
  (ROUNDING). A whole model's prefill and token-by-token decode are bit-identical only in an engine
  whose every upstream reduction is independent of T; otherwise they agree within ROUNDING.

### 4.6 The reference build

The numpy reference (`tools/ir_numpy.py`) is the reference for values. paperkiln's C++ build links
transformer_core, compiled with `-ffast-math`, for its matrix product, `gelu` and row softmax.
Appendix A lists which ops reach those translation units and why no EXACT statement in this
document depends on them.

**The generating environment.** The stored expected values of the conformance suite (section 11)
are the reference run under CPython 3.13.14 and numpy 2.3.5 on Windows 11 x86-64
(`requirements-reference.txt`); `--regenerate` there reproduces every generated file (JSON by
value, the rest byte for byte: 242 files at revision 1.2, 277 at 1.3). Another numpy build can round a ROUNDING value differently: numpy
2.3.3 on Linux x86-64 (CPython 3.12.3, WSL) changes 6 of the 242, the `selective_scan` op fixture by
at most 2 ulp and the `tp_mamba1` model case and its four traces, which carry that difference
through the network, by at most 7.1e-7 of section 11.4's scale against their tolerance of 1e-5
(`conformance/receipts/reference_environment.json`). Such a build passes the suite, and its values
are not stored: expected values are regenerated only in the named environment, and a change of
environment that moves a stored value is a regeneration, made at a revision (section 2.6).

### 4.7 Stored tensors

A tensor stored in a GGUF type other than F32 has, element by element, the f32 values below; every op
reads those values (section 3.7). The arithmetic is ggml's scalar `dequantize_row_*`
(`ggml/src/ggml-quants.c`, block layouts in `ggml/src/ggml-common.h`, llama.cpp tag b6000, commit
4762ad7316dcdec20016ab5985fb46a27902204d:
https://github.com/ggml-org/llama.cpp/blob/4762ad7316dcdec20016ab5985fb46a27902204d/ggml/src/ggml-quants.c).
A block's fields are little-endian; `h(x)` is the fp16 field x converted to f32 (exact, subnormals
included); integers are converted to f32 exactly; the elements of a row are its blocks in order.

| Type (ggml id) | Block: elements, bytes | Element j of a block |
|---|---|---|
| F32 (0) | 1, 4 | the binary32 value |
| F16 (1) | 1, 2 | h(the value) |
| BF16 (30) | 1, 2 | the 16 bits as the high half of a binary32, low half zero |
| Q8_0 (8) | 32, 34: d (fp16), qs[32] (int8) | qs[j] * h(d) |
| Q4_0 (2) | 32, 18: d, qs[16] | ((qs[j] & 15) - 8) * h(d) for j < 16; ((qs[j - 16] >> 4) - 8) * h(d) for j >= 16 |
| Q5_0 (6) | 32, 22: d, qh (uint32), qs[16] | (q - 16) * h(d), q = the Q4_0 nibble of element j (unshifted) plus 16 times bit j of qh |
| Q5_1 (7) | 32, 24: d, m (fp16), qh, qs[16] | (q * h(d)) + h(m), q as Q5_0 |
| Q4_K (12) | 256, 144: d, dmin (fp16), scales[12], qs[128] | (d_s * q) - m_s, below |
| Q5_K (13) | 256, 176: d, dmin, scales[12], qh[32], qs[128] | (d_s * (q + 16 h)) - m_s, below |
| Q6_K (14) | 256, 210: ql[128], qh[64], scales[16] (int8), d | (h(d) * scales[s]) * (q - 32), below |

- **Q4_K, Q5_K.** The 256 elements are 8 sub-blocks of 32. Sub-block s (0 .. 7) has the 6-bit scale and
  minimum (sc_s, mn_s) of `get_scale_min_k4(s)`: for s < 4, sc_s = scales[s] & 63 and mn_s =
  scales[s + 4] & 63; for s >= 4, sc_s = (scales[s + 4] & 15) | ((scales[s - 4] >> 6) << 4) and mn_s =
  (scales[s + 4] >> 4) | ((scales[s] >> 6) << 4). d_s = h(d) * sc_s and m_s = h(dmin) * mn_s. Chunk c
  (0 .. 3) covers sub-blocks 2c and 2c + 1 with the 32 bytes qs[32c .. 32c + 31]: element l of sub-block
  2c has q = qs[32c + l] & 15, of sub-block 2c + 1 q = qs[32c + l] >> 4. In Q5_K, h is bit 2c (sub-block
  2c) or bit 2c + 1 (sub-block 2c + 1) of qh[l].
- **Q6_K.** Two halves of 128 elements; half n reads ql[64n ..], qh[32n ..] and scales[8n ..]. For
  l = 0 .. 31, with is = l / 16 (integer division): the elements 128n + l, + 32, + 64 and + 96 have
  q = (ql[l] & 15) | ((qh[l] & 3) << 4), (ql[l + 32] & 15) | (((qh[l] >> 2) & 3) << 4), (ql[l] >> 4) |
  (((qh[l] >> 4) & 3) << 4) and (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4), with the scales
  scales[is], scales[is + 2], scales[is + 4] and scales[is + 6] of the half.
- **Class: EXACT, on every backend.** Every product above is exact in f32: an fp16 value has at most 11
  significant bits and the integer factors at most 7 + 5 (Q6_K) or 6 + 5 (Q5_K) more, at most 23 in all.
  So Q8_0, Q4_0, Q5_0, Q6_K, F16 and BF16 values involve no rounding at all, and Q5_1, Q4_K and Q5_K
  exactly one (the final + h(m) or - m_s, of two exact operands). A fused multiply-add of the last step
  rounds the same exact value once, so contraction, vector width and evaluation order cannot change any
  value: every engine and backend gives the same bits. A non-finite fp16 field gives the IEEE result
  (a NaN matches any NaN, section 4.2).
- **Reference and fixtures.** `tools/ir_quant.py` is the numpy reference; `tests/test_ir_quant.py`
  checks it bit for bit against gguf-py 0.19.0's `dequantize` on random blocks of every type (fp16
  fields normal, subnormal, signed, zero and 65504) and against the exact value of every element
  computed in float64. `conformance/quant/blocks.json` holds fixtures for every type (section
  11).
- **What paperkiln writes.** `export.quant` (`include/microtorch/gguf.hpp`): `f32` writes F32 only;
  `q8_0`, `q8_0_out` and `q8_0_all` Q8_0 for the selected 2-D tensors and F32 for the rest; `q4_0`,
  `q4_k_m` and `q4_k_s` the types llama.cpp's quantiser chooses (llama_tensor_get_type): Q4_0, and the
  k-quants Q4_K, Q5_K and Q6_K where a tensor's ne0 is a multiple of 256, with llama.cpp's fallbacks
  Q5_0, Q5_1 and Q8_0 elsewhere, and the 1-D norms F32. Every type a paperkiln export can hold is in
  the table, as are those of the third-party files the gates load (F16, BF16, the k-quants).
- **Computing on quantised blocks.** The values above are what an op reads. An engine that multiplies
  quantised weights by activations it quantises too (ggml's `vec_dot_q4_K_q8_K` and its kin quantise
  the activations to Q8_K) computes a different function of the activations; such a path is ROUNDING
  by construction, judged on greedy parity, and its engine's capability table says so (Appendix C).

## 5. Core ops

Each op is what the named paperkiln CPU op computes, in its order of operations.

### 5.1 embedding

`{"op": "embedding", "in": ["tokens"], "weight": W, "out": "x"}`: rows of W [vocab, d] gathered by
token id. An id >= vocab is refused, unless the node carries `state_weight` (section 6.8.1).
EXACT.

### 5.2 pos_embedding

Learned absolute positions: `in [x, pos]`, out = x + weight[pos] row by row, weight [ctx, d]
(`position_embd`). pos must be < ctx (section 3.3). One float add per element. EXACT.

### 5.3 linear

x W^T (+ bias): weight [out, in], bias [out] or null. The bias is added once to the finished dot
product. `tied_to` names the tensor to read instead when the tensor `weight` names is absent from
the file. ROUNDING.

### 5.4 scale

Per-feature gain and bias: `{"op": "scale", "in": [x], "weight": w, "bias": b | null, "out": y}`,
w and b [d] tensors (`blk.{i}.` names in a block), d the width of x:

    y[t][j] = (x[t][j] * w[j]) + b[j]      (x[t][j] * w[j] with no bias)

The product rounded to f32, then the sum, with no fused multiply-add. Row by row (the head may run
it on the last row only). A weight or bias whose length is not x's width is refused, naming the
tensor (rule 37, checked when x's width is known: section 10.1.1). EXACT (against the same order of
operations).

### 5.5 split

`in [x]`, `sizes [s0, s1, ...]` summing to x's width, `out [o0, o1, ...]`: o_k = the s_k consecutive
columns of x after the first s0 + ... + s_{k-1} (a fused qkv projection read as q, k, v). Copies.
Sizes that do not sum to x's width are refused (rule 37). EXACT.

*thirdparty profile (revision 1.5):* `groups` g (a positive integer, default 1; structural): x's width is g
consecutive groups of s0 + s1 + ... columns, and o_k is the k-th part of every group, the groups in order
(o_k has width g s_k). A per-head split: MLA's query into its rotation-free and rotated parts (section 5.20). A
width other than g times the sizes' sum is refused (rule 37), a `groups` that is not a positive integer (rule
10). EXACT.

### 5.6 rmsnorm

x / sqrt(mean(x^2) + eps) * weight, over each row. ROUNDING.

*thirdparty profile:*
- `group` g (default the row width): each consecutive g columns are normalised on their own, with
  weight [g] shared by every group (per-head q/k norm: qwen3, gemma3). g must be positive (rule 10)
  and divide the width (rule 37).
- `weight_offset` c (default 0): the weight is read as w + c, rounded to f32 once at load (a
  load-time transform; Gemma's (1 + w), needed only when a file holds HF's raw w: llama.cpp's
  convert already stores 1 + w).

### 5.7 layernorm

(x - mean(x)) / sqrt(var(x) + eps) * weight + bias over each row, var the biased (divide by width)
mean of squared deviations; weight and bias [d]. ROUNDING.

### 5.8 gelu

`approx` `"tanh"`: 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))) (GPT-2's, and what paperkiln
trains with); `"none"`: the exact 0.5 x (1 + erf(x / sqrt(2))). An engine must not substitute one
for the other. Another value is refused. ROUNDING. (*Informative:* erf is the f32 libm `erff`;
a language whose standard library has none, Rust's for one, supplies its own, which is ROUNDING like
any libm's.)

### 5.9 silu, sigmoid, relu

- `silu`: x * sigmoid(x). ROUNDING.
- `sigmoid`: 1 / (1 + exp(-x)) (`ops::sigmoid`; highway gates, the SRD router). ROUNDING.
- `relu`: x where x > 0, else +0, so relu(-0) = +0 and relu(NaN) = +0 (`ops::relu`'s
  `std::max(0.0f, x)`, ember's, and `expr`'s `relu`). EXACT.

### 5.10 add, mul, sub

Elementwise over two inputs of one shape (two shapes are refused, rule 37: there is no
broadcasting); `sub` is `in [a, b]`: a - b. EXACT.

### 5.11 affine

(x * scale) + shift, float attributes `scale` and `shift`; the product is rounded, then the sum
(`ops::scale` then `ops::add_scalar`). EXACT.

### 5.12 mul_col

`in [x, c]`, x [T, w], c [T, 1]: x[t, j] * c[t]. EXACT.

### 5.13 rms_row

`in [x]` [T, w] -> [T, 1]: sqrt(ss / w + eps), ss the sum of x[t, j]^2 accumulated in ascending j
(`ops::rms_row`; the SRD router uses eps 1e-8). ROUNDING.

### 5.14 rope

Rotates q and k: `in [q, k, pos]`, `out [q', k']`. Attributes `heads`, `head_dim` (even), `theta`,
`rope_heads` and `pairing`, all required.
- `rope_heads`: `"all"` (every head) or `"first"` (legacy runs): head 0 of q and head 0 of k
  rotate (the first head_dim columns of each), and every other column of both passes through
  unchanged (paperkiln's `ops::apply_rope` with all_heads false; op case `rope/first_head`).
- `pairing` `"interleaved"`: adjacent pairs (x[2j], x[2j+1]) within a head.
- The angle of pair j at position pos is pos * theta^(-2j/head_dim); each rotated element is
  x0 * cos - x1 * sin (or x0 * sin + x1 * cos), both products rounded, then the sum or difference,
  with no fused multiply-add.

*thirdparty profile:*
- `kv_heads` (default `heads`): k's head count.
- `pairing` `"neox"`: the pair (x[j], x[j + n_rot/2]) within a head (llama.cpp NEOX, HF
  rotate_half).
- `n_rot` (default head_dim, even, 0 < n_rot <= head_dim): only a head's first n_rot dims rotate,
  the rest pass through (partial rotary, GPT-NeoX). The angle of pair j is
  (pos * freq_scale) * theta^(-2j/n_rot).
- `freq_scale` (default 1, positive): linear RoPE scaling, 1 / factor.
- `freq_factors` (revision 1.5; LongRoPE, Ding et al. 2024, arXiv 2402.13753, as Phi-3 and Phi-3.5 ship it,
  arXiv 2404.14219; one set is also Llama 3.1's `rope_freqs`): a tensor name, or `{"short": S, "long": L,
  "original_ctx": N}` with two tensor names and an integer N >= 1. Each tensor holds n_rot / 2 values f_0 ..
  f_{n_rot/2 - 1} (ne {n_rot / 2}), and pair j's angle is divided by f_j: ((pos * freq_scale) * freq_j) / f_j.
  With two sets, the sequence's length chooses: n, the session's length after the call (positions 0 .. n - 1),
  selects L when n > N and S otherwise, for every row of the call and every row the kv slot holds. A call that
  takes the length past N therefore changes the rotation of every key already in the slot: the engine
  recomputes the slot (it runs the whole sequence again, as Phi-3's own generation does at that point), and a
  truncate back to n <= N does the same with S (section 6.9). Prefill and decode then agree, both computing
  what a fresh run over the same ids computes. Structural (section 3.8). Refused (rule 41): a value of another
  form, a member other than these, `original_ctx` not an integer >= 1; a tensor not of ne {n_rot / 2} in an
  application that reads it.
  From revision 1.6 `freq_factors` may also be the list of the n_rot / 2 values themselves (positive numbers,
  each read as the f32 it rounds to, section 3.5): one set, for factors a builder computes from metadata rather
  than reads from a tensor. YaRN (Peng et al. 2023, arXiv 2309.00071, section 3.2) is such a set: pair j's
  inverse frequency theta^(-2j/n_rot) ((1 - e_j) / s + e_j), s the scaling factor and e_j = 1 - ramp(j) with
  the ramp linear between the correction dimensions of beta_fast = 32 and beta_slow = 1 rotations over the
  original context (floor and ceiling), so f_j = 1 / ((1 - e_j) / s + e_j) (Appendix D.11; YaRN's attention
  temperature is the attention node's `scale`, or `attn_factor`). A list of another length or with an entry
  that is not positive is refused (rule 41).
- `attn_factor` m (revision 1.5; default 1, positive): the table holds cos * m and sin * m (LongRoPE's and
  YaRN's attention scaling; for Phi-3, sqrt(1 + ln(s) / ln(N)) with s the context's extension ratio, which
  llama.cpp's converter writes as `rope.scaling.attn_factor`). Not positive: rule 10.

**The angle table**, in f32: for pair j, freq_j = 1 / powf(theta, f32(2j) / f32(n_rot)) (the
division rounded, then powf, then the reciprocal), and angle(pos, j) = (f32(pos) * freq_scale) *
freq_j, each product rounded; the table holds cosf(angle) and sinf(angle). This is what paperkiln
(`ops::apply_rope`, freq_scale 1) and ember (`graph_ir.cpp`) compute. The numpy reference computes
the angle in double and rounds cos and sin to f32, within ROUNDING of the table (Q3). With
`freq_factors` (revision 1.5) angle(pos, j) is that product divided by f_j, rounded; with `attn_factor` m the
table holds cosf(angle) * m and sinf(angle) * m, each product rounded. For a single node's run (section 11.2)
the sequence's length is the largest position plus one.

Class: the rotation is EXACT given the table. The table is ROUNDING across platforms (cosf, sinf and
powf differ by an ulp between libm implementations), so two builds agree bit for bit when their libm
agrees.

### 5.15 attention

softmax(q k^T * scale) v per head, over the kv slot the node names in `state` (section 6.2).
Attributes `heads`, `kv_heads`, `head_dim`, `scale`, `causal`, `window` (0 = unbounded), `sinks`,
`mask` (section 6.8.2; absent = `"causal"`), `position_bias`, `relative_kv` and `softcap` (below; absent = none). `causal` is true
(section 12, Q6). `in [q, k, v]`, q [T, heads * head_dim]; head h at columns h * head_dim ... ROUNDING.

- **Masking.** Masked keys contribute exactly 0, and the row maximum is taken over visible keys only.
- **Sliding window with sinks** (`ops::swa_attention`; window >= 1). With p the query's position and
  j a key's (both counted from the first token of the slot, 0-based), j is visible iff

      j <= p  and  (j < sinks  or  j >= p - window + 1)

  that is, the first `sinks` positions plus the last `window` positions up to and including p.
  Sinks are positions, not parameters (no tensor). The softmax runs over that union, each key once
  (the sink range is cut at the window's start when they meet). paperkiln sums sinks first, then the
  window, each in ascending j (the reference order, section 4.4). window >= p + 1 with sinks 0 is
  plain causal attention.
- **Windows under Mixture-of-Recursions** (v1.4.2, Q16). p and j are positions, never row indices,
  under `depth_route` too: a recursion step runs on its active rows at their absolute positions
  (section 7.4(b)), and its kv slot holds the rows of the tokens active at that application (section
  7.5). A key of that slot at position j is visible from a query at position p by the rule above, the
  window and sinks counted in positions; the rows of tokens that exited are absent, never replaced by
  earlier rows, so a query sees the sinks and the positions p - window + 1 .. p that the application
  holds, which may be fewer than `window` rows. The window's reach in the text is therefore the same at
  every depth. Counting the window in active rows (the last `window` rows the application holds)
  computes another function: on the suite's `pk_mor` under window 3 and sinks 1 the two readings'
  logits are 1.0e-2 of max(1, max |ref|) apart over 14 ids, against the ROUNDING tolerance 1e-5 (traces
  `pk_mor_window` and `pk_mor_window_rewind`, acceptance case `mor_window`).
- **Decode.** A query at position p reads the k/v rows of positions 0 .. sinks-1 and of the last
  `window` positions, p - window + 1 .. p, its own row included. So after a call ending at position p
  the slot needs the rows of the sinks and of positions p - window + 2 .. p (window - 1 of them), and
  during the next call the new token's rows as they are written: sinks + window - 1 rows per kv head
  suffice between calls (revision 1.3; the earlier wording, sinks + window, counted the next token's row
  too). Anything older is never read again. Positions stay absolute (decode pos = tokens seen, not
  rows kept); the sinks are the first tokens of the prompt. Under the context contract no position
  reaches model.ctx.
- **Consistency with the slot.** A kv slot declares the same window and sinks as the attention
  nodes that read it, and the same `kv_heads` and `head_dim`. A node whose window, sinks,
  `kv_heads` or `head_dim` differs from its slot's is refused.
- *thirdparty profile:* **grouped-query attention**, kv_heads < heads: query head h reads k/v head
  h / (heads / kv_heads); k and v are [T, kv_heads * head_dim]. heads % kv_heads == 0.
- **Relative position bias** (`position_bias`, optional; revision 1.3; T5's bucketed bias, Raffel et
  al. 2020 section 2.1; paperkiln's `ops::relpos_attention`). `{"weight": W, "buckets": [b_0, ...,
  b_{D-1}]}`: W names a tensor [B, heads] (ne {heads, B}: one row per bucket, one column per query
  head), and the buckets are D >= 1 integers in 0 .. B - 1. Head h adds W[b(p - j), h] to the scaled
  score of every key it sees, where p and j are the positions of the query and the key (section 3.4,
  absolute, as the window's) and b(n) = b_k with k = min(n, D - 1): the last bucket covers every larger
  distance.
  The score is s = (q . k) * scale, rounded, then s + W[b(p - j), h], rounded; masking is unchanged
  (causal, window and sinks, `turn_state`), and a masked key still contributes exactly 0. Under
  grouped-query attention the bias is the query head's. The bucketing is data, not a function the
  engine knows: paperkiln writes `ops::t5_position_buckets(B, max_distance)` (T5's causal buckets,
  exact below B / 2 and logarithmic up to max_distance), D = max_distance + 1. `position_bias` is
  structural (section 3.8): one table for every layer is one name, as T5 shares it; a table per layer
  names it with `{i}`. The bias depends on p - j alone, so decode needs nothing beyond the kv slot's
  rows and their positions, and prefill and decode agree as they do without it. Under `depth_route`
  (v1.3.1) p and j are the positions of the step's active rows (section 7.4), as everywhere: the
  distance is never counted in rows, and the reference computes it so (no case yet; ember refuses the
  combination by name, a capability refusal, section 10.3). Refused (rule 39):
  `buckets` empty or holding an integer below 0, any member other than `weight`, `buckets` and `_`
  members (prose, section 3.2), or W not of ne {heads, B} with B greater than every bucket.
- **Relative key and value vectors** (`relative_kv`, optional; revision 1.4; Shaw, Uszkoreit and Vaswani
  2018, arXiv 1803.02155, eqs. 3 and 4 with the clipping of their section 3.3; paperkiln's
  `ops::shaw_attention`). `{"key": RK, "value": RV, "max_distance": K}`: K an integer >= 0; RK names a
  tensor [K + 1, w] (ne {w, K + 1}: row n is the vector of distance n), and RV, when present and not null,
  a tensor [K + 1, w'] of its own; each width is `head_dim` (one set of vectors shared by every head, as
  in Shaw's big model) or `heads * head_dim` (one set per query head, head h's at columns h * head_dim
  ..., as in Shaw's base model). With p and j the positions of the query and the key (section 3.4,
  absolute, as the window's) and n = min(p - j, K) (Shaw's clip(j - i, k) for the keys a causal query
  sees: row n holds Shaw's w_{-n}), head h computes

      s(p, j) = ((q_p . k_j) + (q_p . rk[n])) * scale
      out_p   = (sum over the visible j of a(p, j) v_j) + (sum over the visible j of a(p, j) rv[n])

  each dot product summing the head's columns in ascending order, rounded, then the sum, then the
  product; a(p, j) is the softmax of s over the keys the query sees, with masking unchanged (causal,
  window and sinks, `turn_state`): a masked key contributes exactly 0 to both sums. Without RV the second
  sum is absent. Under grouped-query attention rk[n] and rv[n] are the query head's, k_j and v_j its kv
  head's. These are the reference's orders (section 4.4); the class is ROUNDING, so an engine may sum
  otherwise (gather q_p . rk[n] per distance first, fold rk into the keys, add up a by distance before
  multiplying rv) within the tolerance. `relative_kv` is structural (section 3.8): a set shared by every
  layer is one name, a set per layer (Shaw's) names it with `{i}`. The vectors depend on p - j alone, so
  decode needs nothing beyond the kv slot's rows and their positions, and prefill and decode agree as they
  do without it. Under `depth_route` p and j are positions, as for `position_bias`. Refused (rule 40):
  `key` missing, `max_distance` not an integer >= 0, `value` neither a name nor null, any member other
  than `key`, `value`, `max_distance` and `_` members (prose, section 3.2);
  a tensor not of ne {head_dim or heads * head_dim, K + 1} in an application that reads it; `relative_kv`
  and `position_bias` on one node (revision 1.4 defines each alone).
- **Soft-capping** (`softcap` c, optional; revision 1.5; Gemma 2, Gemma Team 2024, arXiv 2408.00118,
  section 2): a positive number. The scaled score s = (q . k) * scale (with `relative_kv`, its sum of two dot
  products times scale) becomes c * tanh(s / c): the division rounded, then tanh, then the product. A
  `position_bias` is added after it (llama.cpp's order; no released model combines them). Masking is
  unchanged. Per-layer values are allowed (section 3.8). Not positive: rule 10. The final-logit soft cap of
  the same models is an `expr` node in the head, `tanh(x / c) * c` (section 9.3).
- `softmax` (a string) is an informative attribute that paperkiln's exporter writes ("max-shifted,
  masked entries exactly zero"); it carries no semantics.

### 5.16 moe_ffn

A mixture of SwiGLU experts in llama.cpp's layout:
- `router` [E, d] tensor (`ffn_gate_inp`), logits = x R^T.
- `experts` `{gate, up, down}`: 3-D stacks (`ffn_*_exps`), expert e's [out, in] matrix at slice e.
- `n_experts` E, `top_k` k (1 <= k <= E): the top-k of the E logits per token, ties to the lower
  expert index.
- `gate_norm` `"renorm"`: gates = softmax over the k selected logits (the E-way softmax renormalised
  over them); `"raw"`: the E-way softmax's selected entries.
- *thirdparty profile (revision 1.6; DeepSeek-V3's router, DeepSeek-AI 2024, arXiv 2412.19437, eqs. 12-16, and
  auxiliary-loss-free load balancing, Wang et al. 2024, arXiv 2408.15664):* `scoring` `"softmax"` (default:
  the scores s_e are the E-way softmax of the logits) or `"sigmoid"` (s_e = sigmoid(logit_e)); `selection_bias`
  b, a tensor [E] (ne {E}), added to the scores for the choice of experts only: the selection scores are c_e =
  s_e + b_e (c = s without it); `groups` `{"n": G, "top": g}`: the E experts are G consecutive groups of E / G,
  each group's score the sum of its two largest selection scores (its one score when a group holds one
  expert), the g groups of the largest scores are kept (ties to the lower group index) and only their experts
  are candidates. The k selected experts are the k candidates of the largest selection scores, ties to the
  lower expert index; the gates are their scores s_e, divided by the sum of those k scores under `"renorm"` and
  as they are under `"raw"` (with softmax scores and no bias this is the statement above). The reference's
  orders: the scores and their sum in f32, ascending. Refused: `scoring` another value (rule 5); `groups` whose
  n does not divide E, whose top is not in 1 .. n, or that keeps fewer than k experts (rule 10); a selection bias
  not of ne {E} in an application that runs the node (rule 44). transformers' implementation keeps the experts of
  dropped groups at a selection score of 0 rather than removing them, which differs only when fewer than k
  candidates have a positive selection score. `selection_bias` and `groups` are structural (section 3.8).
- `activation` `"silu"`: expert_e(x) = down_e(silu(gate_e x) * up_e x).
- out = the sum over the selected e of gate_e * expert_e(x), accumulated in ascending expert index.
- `routing` (a string) is an informative attribute; it carries no semantics.

ROUNDING.

### 5.17 list, append

- `list`: `in [x0, x1, ...]`: a list value holding those tensors in order. EXACT.
- `append`: `in [L, x]`: the list L with x added at the end. EXACT.

### 5.18 depth_attend

Attention over depth (Attention Residuals, block form; `nn::AttnResStack::depth_attend` and
`ops::depth_attend`): `in [L, x1, ...]` -> [T, d], `weight` the pseudo-query [d], `eps` (optional,
default 1e-5: the attribute governs, and every stored document writes 1e-5). The
sources s_0 .. s_{S-1} are L's tensors in order, then x1, ... in order. S = 1: out = s_0 bit for
bit and `weight` is not read. S >= 2, each row t on its own (decode needs nothing but the current
token's sources).

S is structural: it is fixed at load by the list's construction (the entries `list` and `append`
put in it before the node), never by data, so a consumer checks it at load. With S = 1 the node's
output is s_0 itself, and the tensor `weight` names has no effect on any value: its contents are
never read. An exporter must write that tensor (llama_attnres and attnres write
`blk.0.attnres_attn_q` as zeros); a consumer may accept its absence at S = 1, and must refuse a
missing pseudo-query at S >= 2 at load.
Under `weight_index`, S at application i counts application i's entries, whichever stored layer i
binds. For S >= 2:

    r_i  = 1 / sqrt(ss_i / d + eps), ss_i = sum of s_i[t, j]^2 in ascending j
           (ops::rmsnorm with unit weight; eps the node's eps)
    l_i  = sum over j of (s_i[t, j] * r_i) * w[j], both products rounded; the sum is a
           matmul reduction
    a    = softmax over i of l_i: e_i = exp(l_i - max_i l_i), z = e_0 + e_1 + ... in
           ascending i, a_i = e_i / z
    out  = s_0 a_0 + s_1 a_1 + ..., per element, added in ascending i (each product
           rounded, then each sum)

ROUNDING (the orders are the reference orders, section 4.4).

### 5.19 expr

A runtime-defined op (a new activation or gate from a paper, added to a model without recompiling
either engine; `ops::expr`, `microtorch/expr.hpp`; numpy reference `tools/ir_expr.py`).

`in [v0, v1, ...]`, every input [T, w] with one w; out [T, w]. Attribute `expr`, a string in this
closed grammar (nothing else is accepted; spaces and tabs are ignored):

    expr    := term (('+' | '-') term)*
    term    := unary (('*' | '/') unary)*
    unary   := '-' unary | power
    power   := primary ('^' unary)?
    primary := number | name | pi | '(' expr ')'
               | fn '(' expr (',' expr)* ')'

So `^` binds tightest and is right-associative (2^3^2 = 2^(3^2), -x^2 = -(x^2), x^-1 is allowed).

- **Tokens.** The text is read left to right into tokens, each the longest match at its place;
  between tokens only spaces and tabs may stand, and they are skipped (any other character outside a
  token, a newline included, is an error naming its column):

      number := digits ('.' digits?)? exponent? | '.' digits exponent?
      exponent := ('e' | 'E') ('+' | '-')? digits
      name   := [A-Za-z_] [A-Za-z0-9_]*
      operator and punctuation := + - * / ^ ( ) ,

  digits are ASCII 0-9. An `e` or `E` directly after a number's digits begins its exponent, which
  must have digits: `2e`, `2e+` and `1.5E` are errors, never the number 2 followed by the name e.
  A number directly followed by a name (`2x`) is two tokens, which the grammar then refuses (no
  operator between them). Names are ASCII; a name is a function only when `(` follows it (a
  function name without `(` is an error, and any other name followed by `(` is an unknown function).
- **Numbers** are decimal or scientific (1, 0.5, .5, 2., 1e-5, 1.5E+3), read as the nearest float
  (strtof in the C locale: the decimal point is "."), and must be finite. `pi` is the nearest float
  to pi.
- **Names** are positional, not the IR values' names: the inputs in the order of `in` are x, then
  a, b, c, ... (x the first, a the second; at most 23 inputs, x and a .. v).
- **Functions**, each over the expression's [T, w] values:

      exp log sqrt tanh erf abs   elementwise (f32 libm)
      sigmoid(e)       1 / (1 + exp(-e)), as the sigmoid op
      relu(e)          e where e > 0, else +0, as the relu op
      max(a, b)        a where a >= b, else b   (so max(NaN, b) = b, max(a, NaN) = NaN)
      min(a, b)        a where a <= b, else b
      where(c, a, b)   a where c > 0, else b    (a NaN condition selects b)
      rowsum(e)        the sum of the row's w entries, added in ascending column order from 0
      rowmean(e)       rowsum(e) / w
      rowmax(e)        the row's largest entry

  The row reductions broadcast back to [T, w]. `+ - * /` are elementwise and `^` is pow (powf).
- **Numerics.** Every primitive is computed in f32 and rounded to f32 after it, with no fused
  multiply-add. A subexpression without inputs is folded in f32 with the same operations
  (`where(c, ...)` on constants included), and a result without inputs fills [T, w]. A row reduction
  of a constant keeps its own operations: `rowsum(c)` is c added w times in ascending column order from
  0, never c * w, `rowmean(c)` that sum divided by w, and `rowmax(c)` is c (op case
  `expr/rowsum_constant`: `rowsum(0.1)` over 2048 columns is 204.8033, not 204.8). The rows are
  independent, so decode (T = 1) is prefill's last row.
- **Gradients** (for training engines) come from the primitives: max, min and where pass the
  gradient to the chosen operand, none to where's condition; rowmax to the row's first largest
  entry; abs and relu give 0 at 0. An engine needs only the forward.
- An expression that does not parse is refused at load, naming the column. Every engine accepts
  expressions nested up to 200 deep (the reference parser's limit, `MAX_DEPTH` in
  `tools/ir_expr.py`, which ember's vendored parser shares); a deeper one may be refused. An engine
  that cannot run an expression refuses it at load, naming the function.
- **Depth** counts the grammar's rules as a parser enters them: the whole text is an `expr` at depth
  0, and each `term`, `unary`, `power`, `primary` or `expr` that a rule's right-hand side names is
  one deeper than that rule (the operands of a chain of `+` or `*` share one depth; a function's
  arguments and a parenthesised `expr` are one deeper than their `primary`; the operand of a unary
  minus and the exponent of `^` are one deeper than their rule). An expression is within the limit
  when no rule is entered deeper than 200. So a name alone reaches depth 4, each pair of parentheses
  around it adds 5, each unary minus 1: 39 nested parentheses around `-x` reach exactly 200 (op case
  `expr/depth_200`, which every engine must accept), 40 around `x` reach 204, and 196 minus signs
  before `x` reach 200. A recurrence body (section 6.10.2) counts the same way from its `expr`.

ROUNDING (libm transcendentals differ by an ulp between platforms).

### 5.20 mla_attention (thirdparty profile; revision 1.5)

Multi-head latent attention (DeepSeek-AI 2024, DeepSeek-V2, arXiv 2405.04434, section 2.1, equations 9 to
19): every head's keys and values are linear maps of one compressed latent per token, and the rotary part of
the key is one vector per token shared by every head, so the kv slot holds the latent and that vector, not
per-head keys and values.

`{"op": "mla_attention", "in": [qn, qr, c, kr], "state": slot, "heads": H, "kv_rank": r, "qk_nope": dn,
"qk_rope": dr, "v_head_dim": dv, "k_b": K, "v_b": V, "scale": s, "causal": true, "window": w, "sinks": n_s,
"out": a}`, every attribute required:
- qn [T, H * dn]: the rotation-free part of each head's query (head h at columns h * dn ...); qr [T, H * dr]:
  its rotated part, after RoPE; c [T, r]: the latent; kr [T, dr]: the shared rotary key, after RoPE. (A `split`
  with `groups` H separates qn and qr in the query projection's per-head layout, section 5.5; one `rope` node
  with heads H, kv_heads 1 and head_dim dr rotates qr and kr.)
- K names a tensor [H, r, dn] (ne {dn, r, H}): K[h] [r, dn] is head h's up-projection of keys, W^UK,
  transposed, the absorbed form llama.cpp's converter writes (`attn_k_b`). V names a tensor [H, dv, r] (ne {r,
  dv, H}): V[h] [dv, r] is head h's W^UV (`attn_v_b`).
- Head h, for a query at position p and each key position j it sees:

      qa_p      = K[h] qn_{p,h}                          (qa [r]: each entry a dot product over dn)
      s(p, j)   = ((qa_p . c_j) + (qr_{p,h} . kr_j)) * scale
      a(p, j)   = the softmax of s(p, .) over the keys p sees (section 5.15's masking)
      u_p       = sum over the visible j of a(p, j) c_j  (u [r])
      out_{p,h} = V[h] u_p                               (out [dv], at columns h * dv ... of a [T, H * dv])

  each dot product rounded, then the sum, then the product; these are the reference's orders. The class is
  ROUNDING, so an engine may decompress instead (k_{j,h} = [K[h]^T c_j | kr_j], v_{j,h} = V[h] c_j, then
  section 5.15's attention), the paper's statement and transformers' implementation, within the tolerance.
- **The slot.** The node reads a kv slot (section 6.2) that declares `kv_heads` 1, `head_dim` r + dr and the
  node's window and sinks: per position it holds c and kr, r + dr values, against 2 H (dn + dr) for the
  decompressed keys and values. Decode reads the visible rows and appends the new token's c and kr; masking,
  decode and positions follow section 5.15, rewind and snapshots section 6.9, as for `attention`. One node reads
  a slot (section 6.1).
- Refused: a slot with kv_heads other than 1, head_dim other than r + dr, or another window or sinks (rule 12);
  `causal` not true (rule 14); an input of another width (rule 37, when the rows arrive); K not of ne {dn, r,
  H} or V not of ne {r, dv, H} in an application that reads it (rule 42). `heads`, `kv_rank`, `qk_nope`,
  `qk_rope` and `v_head_dim` are positive integers (the schema, rule 1) and may be per-layer; `k_b` and `v_b`
  are structural (section 3.8).

## 6. The state profile

### 6.1 Slots

`block.state` lists the slots. There is one instance of each slot per application (per layer) and
per sequence; a node names its slot in `state`. A block may declare several slots (srd: one kv and
one recurrent). Every slot is read by at least one node, a slot name is declared once, and a node
that names a slot names a declared slot of the kind its op needs. A kv slot is read by exactly one
attention or `mla_attention` node (two readers would append twice per application, which is not defined), and a
recurrent slot of op `recurrence` by exactly one recurrence node (section 6.10); a block may still
declare several kv slots. Nodes outside the block have no state, so a stateful op outside
the block is refused.

### 6.2 kv (core)

`{name, kind "kv", heads, kv_heads, head_dim, max_len, window, sinks, dtype}`: the k and v rows of
the tokens seen so far, `dtype` `"f32"`. window 0 and sinks 0 keep every row up to `max_len`, the
slot's capacity in positions: a position >= max_len is refused at run time (exporters write
max_len = model.ctx; an engine may hold fewer positions by choice and then refuses earlier).
Optional keys: `evict` (section 6.8.3), `kv_rows` (section 7.5) and `kv_from` (section 7.6). The
attention nodes that read a slot declare its window, sinks, kv_heads and head_dim (section 5.15); an
`mla_attention` node's slot declares kv_heads 1 and head_dim kv_rank + qk_rope (section 5.20).

### 6.3 recurrent

`{name, kind "recurrent", op, dtype "f32", tensors: [{name, shape, init}, ...]}`: a fixed-size
state carried from token to token by the op named in `op`, with the tensors listed (shapes positive
integers), every tensor set to `init` (0) when a sequence starts.

- Decode applies the op's update for one token. Prefill applies it for t = 0 .. T-1 in order, so
  prefilling a prompt and decoding it token by token leave the same state bit for bit, given the
  same inputs (section 4.5). A chunked or parallel prefill sums in another order and is ROUNDING
  against the sequential one.
- The size does not grow with the sequence; there is no max_len or window, and positions never
  enter a recurrent op (model.ctx still bounds every sequence under the context contract).
- The ops and their tensors: `linear_attention` (num, den), `ssm_scan` (s), `conv1d_causal` (x),
  `selective_scan` (s), and `recurrence` (the tensors its update assigns, section 6.10, whose `init`
  is 0 or `"body"`). A slot whose op is not one of these, whose dtype is not f32, or whose tensors
  do not fit its op (sections 6.4 to 6.7 and 6.10) is refused.

### 6.4 linear_attention

Causal linear attention with the elu+1 feature map (`ops::kimi_attention`; paperkiln's "kimi"
attention, and the cheap path of SRD) over a recurrent slot. `in [q, k, v]`, each
[T, heads * head_dim], head h at columns h * head_dim ... as for attention. No scale, no softmax, no
positions, and every column c is independent of every other:

    phi(x)    = x + 1             for x > 0
              = (exp(x) - 1) + 1  for x <= 0, rounded after the subtraction and again
                after the addition
    num[c]    = num[c] + phi(k[t, c]) * v[t, c]   (product rounded, then the sum)
    den[c]    = den[c] + phi(k[t, c])
    out[t, c] = (phi(q[t, c]) * num[c]) / max(den[c], den_floor)

num and den are updated with token t BEFORE out[t] is formed (token t sees keys 0 .. t). State: a
recurrent slot with tensors num and den, each [heads, head_dim], f32, initial 0 (after the first
token num = phi(k_0) v_0 and den = phi(k_0) exactly). Attributes `heads`, `head_dim`, `feature_map`
`"elu_plus_one"`, `den_floor` 1e-8, `causal` true. ROUNDING (exp).

### 6.5 ssm_scan

paperkiln's state-space scan (`ops::ssm_scan`, as `mamba::S4Layer` uses it) over a recurrent slot.
`in [u]` [T, n]; tensors `a` [n, n] (row-major, A[i][j] at i*n + j), `b` [n], `c` [n], `d` [1];
attribute `n`. For t = 0 .. T-1:

    s_t[i]  = (A[i][0] s_{t-1}[0] + A[i][1] s_{t-1}[1] + ... + A[i][n-1] s_{t-1}[n-1])
              + b[i] u[t, i]      summed left to right from +0, each product rounded
    y[t, i] = c[i] s_t[i] + d u[t, i]

The whole of s_t is formed from s_{t-1} before any element of the state is overwritten. State: a
recurrent slot with one tensor s [n], f32, initial 0. ROUNDING.

### 6.6 conv1d_causal (thirdparty profile)

Depthwise causal convolution over time (Mamba): `in [x]` [T, C]; `weight` [C, K] (`ssm_conv1d`),
`bias` [C] optional; attributes `channels` C >= 1 and `kernel` K >= 2. State: a recurrent slot with
one tensor x [K - 1, C], the K - 1 previous inputs, oldest first, initial 0. With xx = the state's
rows then x's:

    out[t, c] = sum over k = 0 .. K-1, ascending, of w[c, k] * xx[t + k, c]
                (+ bias[c], added once to the finished sum)

then the state becomes the last K - 1 rows of xx. ROUNDING.

### 6.7 selective_scan (thirdparty profile)

Mamba-1's selective state-space scan (`ggml_ssm_scan`): `in [x, dt, B, C]`: x and dt [T, D], B and
C [T, N]; tensors `a` [D, N] (A as stored: llama.cpp writes -exp(A_log)) and `d` [D] optional (the
skip); attributes `channels` D, `d_state` N, `dt_softplus` (default true). State: a recurrent slot
with one tensor s [D, N], initial 0. For t = 0 .. T-1, every channel c:

    delta  = dt[t, c] <= 20 ? log1p(exp(dt[t, c])) : dt[t, c]
             (softplus; delta = dt when dt_softplus is false)
    xd     = x[t, c] * delta
    s[c,n] = s[c,n] * exp(delta * A[c,n]) + B[t,n] * xd
    y[t,c] = sum over n ascending of s[c,n] * C[t,n]   (+ d[c] * x[t, c] after the sum)

ROUNDING (exp, log1p).

### 6.8 Per-turn state tokens

The state ids S_0 .. S_{k-1} are the ids vocab .. vocab + k - 1, never produced by the tokeniser;
k = `paperkiln.state_tokens.k` (section 8.2). They take three keys.

#### 6.8.1 embedding state_weight

With `state_weight` (GGUF `state_embd.weight`, [k, d], ggml ne {d, k}), an id v with
vocab <= v < vocab + k reads row v - vocab of it instead of the token embedding:

    {"op": "embedding", "in": ["tokens"], "weight": "token_embd.weight",
     "state_weight": "state_embd.weight", "out": "x"}

k is the tensor's row count and equals the uint32 GGUF key `paperkiln.state_tokens.k`. Without
`state_weight` an id >= vocab is refused; with it, an id >= vocab + k is. A negative id is always
refused (at run time, when the ids arrive). The tied output head covers ids 0 .. vocab - 1 only:
state rows are never logits and never sampled. EXACT.

#### 6.8.2 attention mask turn_state

`"mask": "turn_state"` (absent = `"causal"`) is defined with window 0 and sinks 0 only, and the
embedding must carry `state_weight`.

Every position p (0-based from the slot's first token) has a code that depends on the ids up to p
alone:
- c(p) = 1 if id_p >= vocab (a state position), else 0;
- s(p) = the number of COMPLETE state blocks strictly before p, a complete block being k
  consecutive ids vocab, vocab + 1, .., vocab + k - 1 in that order: a block ends at position q
  iff ids q - k + 1 .. q are exactly S_0 .. S_{k-1} (complete blocks never overlap). This is
  `ops::turn_state_codes` and `tools/ir_numpy.py`'s `turn_state_codes` (a counter that a text id
  resets, an S_0 restarts at 1 and any other out-of-order state id resets to 0).

A state id outside a complete block (a training window that starts inside one) is a state position
of the current segment and completes nothing. Partial blocks arise only in training windows; an
engine serving prompts may refuse them under the raw-ids rule (section 8.2).

The key at j <= p is visible to the query at p iff

    s(j) == s(p)   or   c(j) == 1

(j <= p gives s(j) <= s(p)): a turn's text sees its own segment and every earlier state block; a
state block sees its turn's text, its own earlier positions and every earlier block. The positions
of block n have s = n (the block is not complete before its own positions), so they belong to their
turn's segment. Every state position, in a complete block or not, has c = 1 and stays visible to
every later query. Masked keys contribute exactly 0 and the row maximum is taken over visible keys
only; paperkiln sums the visible keys in ascending position. RoPE and positions are unchanged.
ROUNDING, as attention.

Ids without a state id form one segment, so the mask is the causal mask and the function is plain
causal attention. The values are bit for bit those of the causal path in an engine that computes
both with the same reduction order (paperkiln runs its causal kernel when no id is a state id;
ember's CPU turn_state and causal kernels do the same per-key arithmetic in ascending order);
otherwise they agree within ROUNDING (section 4.5).

#### 6.8.3 kv evict turn_state

`"evict": "turn_state"` (optional) on a kv slot requires every attention node reading the slot to
have mask `turn_state`. Once the last position of state block n is written, no later query sees a
text row of any segment <= n, so the engine MAY drop those rows. Evicting eagerly, after n complete
turns the slot holds n * k state rows plus the rows of the current segment; the lazy rule below also
allows segment n - 1's text rows until the next text position arrives. Positions stay absolute
(decode pos = tokens seen, not rows kept).

The timing is the engine's: it may drop segment n's text rows at the end of the call that completes
block n (eager) or later, but by the end of the call that writes the first text position after
block n (lazy, as ember does) at the latest. So after a call ending at length L, with m complete
blocks in 0 .. L-1 and block m - 1 ending at q, the slot holds the state rows, the text rows of
segment m, and, only while no text position follows q, possibly the text rows of segment m - 1; the
text rows of every earlier segment are gone. (A partial state block's positions are state rows,
c = 1, and are never dropped.) The bound is closed: an engine that holds rows longer is running with
eviction off, which keeps every row for the whole sequence and is the comparison run of the next
paragraph, not a third timing.

Against keeping every row, evicting is EXACT on a kernel that sums the visible keys sequentially in
ascending position (ember's CPU attention; paperkiln's KV cache, `LlamaBlock::forward_cached_rows`:
the same keys in the same order, when both runs feed the same chunks of ids), and ROUNDING on tiled
device kernels (ember's CUDA flash2: tile boundaries and running-maximum rescales move). Rewind
under eviction: section 6.9.

### 6.9 Prefix reuse, rewind and snapshots

Truncating a sequence to its first n tokens (prefix reuse, rewind) leaves the state a fresh run over
those n tokens would leave. Per slot:
- **kv, every row kept:** keep the rows of positions < n.
- **kv with evict turn_state:** let m be the segment of the cut: the number of complete state
  blocks among positions 0 .. n-1 (the segment position n would belong to). When the text rows of
  segment m are still held (always for the current segment; for the one before it, under lazy
  eviction, until a text position follows its block), keep the rows of positions < n. When they
  were dropped (an earlier message edited; a re-tokenised reply that differs from the generated
  ids), the engine drops every row of segment >= m, text and state, keeps the state rows of
  segments < m, and re-prefills positions start(m) .. n - 1 from the ids it holds, start(m) being
  the position after block m - 1 (or 0); the result is the rows a run that never evicted would hold,
  less the dropped text rows of segments < m. A cut at start(m) re-prefills nothing. Regenerating
  the latest reply cuts inside the reply's segment: an engine that evicts lazily still holds its
  text rows (no text position follows its state block yet) and keeps the rows before the cut; one
  that evicted them when the reply's block completed re-prefills that segment like any other.
  The normative result is the state a fresh run over the first n tokens would leave. A lower layer of
  an engine may refuse a forward that starts inside a segment whose rows it dropped; the engine's
  session (and its conformance adapter's `truncate` and `forward`) must then succeed by
  re-prefilling (Q12).
- **kv with a window** (window >= 1; revision 1.3): keep the rows of positions < n. The rows a window
  dropped are gone: a query at position p reads the sinks and positions p - window + 1 .. p, so a
  query at n reads rows of positions back to n - window + 1, and an engine that frees the rows its
  window no longer reads (a ring buffer, a cache of about sinks + window rows) may no longer hold them
  after a cut. When every row of positions max(sinks, n - window + 1) .. n - 1, and of the sinks below
  n, is still held, the engine keeps the rows of positions < n. When any of them is gone, it cannot
  rebuild them from the slot (each layer's row at a position depends on the rows of earlier positions
  in the layers below, which may be gone too), so it re-prefills: it runs the first n tokens again from
  the start, or restores a snapshot taken at n. Either way the slot holds what a fresh run over the
  first n tokens holds, so prefill and decode keep agreeing across a rewind. Or the engine refuses the
  truncate: an error naming the dropped rows, decided before anything runs, the session left as it
  was (section 11.3); its adapter replies `skip`, so the trace is skipped, never passed. Re-prefilling
  and refusing both conform; accepting the cut and computing the next token without those rows never
  does. A lower layer may refuse such a cut, as transformers' `Cache.crop` does and llama.cpp's
  `llama_memory_seq_rm` should (finding 4 of `adapters/llamacpp/FINDINGS.txt`, H3 of
  `adapters/hf/FINDINGS.txt`); the engine above it then re-prefills or refuses. Traces `pk_gpt2_swa_rewind_window` (window 4, sinks 2: 14 tokens, a cut to 7) and
  `tp_gemma3_rewind_window` (window 4 on its local layers: 48 tokens, more than llama.cpp's smallest
  windowed cache, a cut to 10) and `tp_mistral_hd16_rewind_window` (window 4 on every layer: 14
  tokens, a cut to 7).
- **kv under rope with two `freq_factors` sets** (revision 1.5): the set follows the sequence's length
  (section 5.14), so a truncate to n <= original_ctx after a longer sequence, like a forward that takes the
  length past original_ctx, leaves rows rotated with the other set: the engine recomputes them (runs the first
  n tokens again, or restores a snapshot taken at n); keeping them never conforms. The traces `tp_phi3_decode`,
  `tp_phi3_chunks` and `tp_phi3_truncate` (original_ctx 6) cross the boundary, the last in both directions.
- **kv with kv_rows active:** keep the rows whose positions are < n, per application, with each
  application's written-rows mask truncated with them (section 7.5). With a window too (v1.4.2, Q16),
  the window's rule above holds per application, over the rows that application holds: when every row
  it holds at positions max(sinks, n - window + 1) .. n - 1, and at the sinks below n, is still held, it
  keeps its rows of positions < n; otherwise the engine re-prefills (or restores a snapshot taken at n)
  or refuses the truncate (trace `pk_mor_window_rewind`: 14 tokens, a cut to 8, 3 new ids).
- **recurrent:** a recurrence cannot drop tokens. Prefix reuse keeps a snapshot of the slot's
  tensors taken after the prefix's last token; rewinding means restoring a snapshot or running again
  from the start. A snapshot is the slot's tensors, nothing else. A recurrence's slot is such a slot
  (section 6.10.5). Truncating a session whose model has recurrent state is defined like any other
  truncation, and an engine's session (and its adapter's `truncate`) does not refuse it: it restores a
  snapshot taken at n when it holds one, and otherwise runs the first n tokens again from the start
  (traces `pk_srd_truncate`, a kv and a recurrent slot, `tp_mamba1_truncate`, and, from revision 1.4,
  `pk_flex_mdha_truncate`, a `conv1d_causal` slot beside a kv slot: the conv slot's K - 1 rows are restored
  or recomputed with the kv rows, never kept from the longer sequence).

### 6.10 recurrence (recurrence profile)

A recurrent op defined in the document: a fixed-size state, its initial value, a per-token update
and a readout, each written as a body in a closed grammar (`expr`'s, section 5.19, with indexed
tensors and an ordered sum). A paper's new recurrent primitive, a state-space model or a linear
attention variant, travels as data and runs in both engines without engine code; the four
hand-written recurrent ops are recurrences too (section 6.10.6). It extends ONNX's `Scan` and Keras's
cell contract with decode-state semantics, numerics classes and a gradient check as conformance
(`docs/IR_PRIOR_ART.md` section 4c).

#### 6.10.1 The node

    {"op": "recurrence", "in": ["q", "k", "v"], "state": "lin",
     "index":   {"h": 2, "e": 8},
     "inputs":  ["q[h,e]", "k[h,e]", "v[h,e]"],
     "weights": {"A[h,e]": "blk.{i}.decay"},
     "init":    ["num[h,e] = 0"],
     "let":     ["pk[h,e] = ..."],
     "update":  ["num[h,e] = A[h,e] * num[h,e] + pk[h,e] * v[h,e]", "den[h,e] = den[h,e] + pk[h,e]"],
     "readout": "y[h,e] = ...",
     "out": "a"}

- `index` (required): index names to sizes (positive integers).
- `inputs` (required): one declaration `name[i1, ..., ik]` per value of `in`, in order. The value is
  [T, w] with w the product of the declared sizes; row t is the input's tensor at token t, row-major
  (the last index fastest). A value whose width is not w is refused (rule 34), at load by an engine
  that knows the widths and at the latest when the rows arrive.
- `weights` (optional): declarations to tensor names. The tensor (`{i}` replaced as in section 3.7,
  `tensor_index` allowed) holds as many elements as its declaration, read row-major in numpy order
  (section 3.7); its stored shape is otherwise free (a [1] tensor is declared `D[o]` with o of size 1).
- A declaration (`inputs`, `weights`) may repeat an index: `W[c,c]` declares a square tensor over c,
  of sizes [size(c), size(c)]. The declaration fixes the tensor's shape only: a read gives it any
  indices of the right number (`W[c,j]`), whatever their sizes, and a coordinate beyond a dimension
  reads +0 (section 6.10.3's zero padding), so `W[c,j]` with size(j) = 4 > size(c) = 3 reads +0 at
  j = 3 (op case `recurrence/read_beyond_declaration`). An assignment (`init`, `let`, `update`,
  `readout`) may not repeat an index (rule 32).
- `update` (required): one assignment per state tensor, `name[i1, ..., ik] = body`; the names are
  the state's tensors and their index sizes the tensors' shapes. `init` (optional): an assignment for
  a state tensor whose initial value is a body (a learned initial state, read from weights); every
  other state tensor starts at +0. `let` (optional): intermediate tensors, in order. `readout`
  (required): one assignment, the output row; the value `out` is [T, w], w the product of its sizes.
- `state` names a recurrent slot of op `recurrence` (section 6.3) whose tensors are the update's: one
  per state tensor, with its name and shape, and `init` `"body"` exactly when the node's `init`
  assigns it (otherwise 0). The slot is read by this node alone.
- Names (index, input, weight, state, let and the readout's) are identifiers
  ([A-Za-z_][A-Za-z0-9_]*), distinct across all of them, and none is a function name, `pi`, `sum`,
  `rowsum`, `rowmean`, `rowmax` or `per_layer`. They are local to the node: an input's name need not
  be its value's name in `in`.
- Every key of the node is structural (section 3.8): its declarations and bodies are the same in every
  application, and only its weights' tensors differ, through `{i}`.

#### 6.10.2 Bodies

A body is an `expr` (section 5.19) with these changes; nothing else is accepted:

    assignment := name '[' index (',' index)* ']' '=' expr
    primary    := number | 'pi' | name '[' ix (',' ix)* ']' | index
                  | fn '(' expr (',' expr)* ')' | 'sum' '(' index ',' expr ')' | '(' expr ')'
    ix         := index | index ('+' | '-') integer | integer
    fn         := exp log sqrt tanh erf abs sigmoid relu max min where

- `name[...]` reads a tensor (an input, a weight, a state tensor or a let), one `ix` per dimension. A
  bare index name is that index's value at the point: an integer, as f32 (exact). Any other bare name
  (an undeclared name, or a tensor written without its indices) is refused (rule 32).
- `sum(j, e)` adds e over j = 0 .. size(j) - 1. j must be declared and not already in scope.
- The functions and operators are `expr`'s, with the same definitions (`max`, `min`, `where`, `relu`,
  `sigmoid`, `^`). The row reductions `rowsum`, `rowmean` and `rowmax` are not part of this grammar
  (`sum` replaces them). Numbers are read as in `expr` (the nearest float, strtof).
- **Scope.** In `t[i1, ..., ik] = e`, the indices i1 .. ik are in scope, distinct and declared; `sum`
  adds its index inside its argument. Every index a body uses is in scope.
- **What a body reads.** `init`: weights only. `let`: inputs, weights, the state before the token and
  earlier lets. `update`: inputs, weights, the state before the token and the lets. `readout`: inputs,
  weights, the lets and the state after the token. A read has as many indices as its tensor has
  dimensions.
- Every body nests at most 200 deep (as `expr`).

*Why this extension.* `expr` covers stateless maps of [T, w] rows. A state with structure needs
indexed tensors (a matrix state, an outer product B[n] * x[c], a state shifted by one row) and a
contraction in a stated order (a matrix-vector product). Index offsets with zero padding give the
causal convolution's history. Those three additions express the four hand-written ops (section
6.10.6) and keep `expr`'s numerics rule for every primitive.

#### 6.10.3 Semantics

At the start of a sequence each state tensor takes its init body's value, or +0. Then for each
token t in order (prefill: t = 0 .. T-1; decode: the one token):

1. every input name reads row t of its value, as a tensor of its declared shape;
2. the lets are evaluated in order;
3. every state tensor's update is evaluated from the state before the token (the whole new state is
   formed before any of it replaces the old);
4. the readout is evaluated with the state after the update; row t of `out` is its tensor, row-major.

A body is evaluated at every point of its assignment's index space. At a point:

- **Reads.** Each ix is the index's value plus its offset, or the integer. A read whose coordinates
  lie outside the tensor's shape in any dimension is +0 (zero padding).
- **sum(j, e)** is ((+0 + e|j=0) + e|j=1) + ... + e|j=n-1, in ascending j, each addition rounded.
- **Everything else** is `expr`'s numerics: every primitive computed in f32 and rounded to f32 after
  it, with no fused multiply-add; constant subexpressions are folded in f32 with the same operations.

A recurrence runs over the rows of one sequence. Several sequences stacked in one call (a training
batch) are separate sequences, each from its own initial state.

#### 6.10.4 Class

A recurrence whose bodies use none of `exp`, `log`, `tanh`, `erf`, `sigmoid` and `^` is **EXACT**:
its arithmetic is IEEE `+ - * /`, `sqrt` and selections in a fully stated order (section 4.4). One
that uses them is **ROUNDING** (libm transcendentals differ by an ulp between platforms). Both are
stated against the sequential order of section 6.10.3; a chunked or parallel evaluation (a scan, the
SSD form, fla's `chunk`) is ROUNDING against it, as section 6.3 says of every recurrent op.

#### 6.10.5 Decode state and gradients

- **Decode state.** The slot's tensors are the state (section 6.3): prefilling a prompt and decoding
  it token by token leave the same state bit for bit given the same inputs (section 4.5); a snapshot
  is the slot's tensors, and rewinding restores one or runs again from the start (section 6.9). The
  conformance suite checks it (prefill against incremental decode, snapshot and restore).
- **Gradients** (for training engines): backpropagation through the sequential fold of section
  6.10.3, from the primitives' rules (`expr`'s: `max`, `min` and `where` pass the gradient to the
  chosen operand and none to `where`'s condition; `abs` and `relu` give 0 at 0); a read passes its
  gradient to the element it read (none for a read outside the tensor); `sum` passes it to every term;
  index values and numbers have none. Gradients reach the inputs and the weights (through any body,
  `init` included). An engine may derive the backward by autograd over the unrolled bodies (paperkiln
  does) or supply one for a body it recognises; either way it passes the suite's gradient cases,
  central finite differences of the reference evaluated in float64 (section 11).
- *Informative.* Under these rules an unselected branch still has its forward value: in
  `where(x, x + 1, exp(x))` a large x makes `exp(x)` infinite, and the gradient 0 it receives times
  that infinity is NaN. Bodies keep the unselected branch finite, as `exp(min(0, x))` does in section
  6.10.6 (`min(0, x)` is x wherever that branch is selected, NaN included).

#### 6.10.6 The hand-written recurrent ops as recurrences (informative)

`tools/ir_recurrence.py` (`rewrite`) replaces each hand-written recurrent node with these, keeping the
slot's name and its tensors:

- **linear_attention** (heads H, head_dim E, den_floor f): index {h: H, e: E}; inputs q, k, v [h,e];
  let `pk[h,e] = where(k[h,e], k[h,e] + 1, (exp(min(0, k[h,e])) - 1) + 1)`; update
  `num[h,e] = num[h,e] + pk[h,e] * v[h,e]`, `den[h,e] = den[h,e] + pk[h,e]`; readout
  `y[h,e] = (phi(q) * num[h,e]) / max(den[h,e], f)`, phi(q) the same `where` on q. ROUNDING.
- **ssm_scan** (n; tensors a, b, c, d): index {r: n, j: n, o: 1}; input u[r]; weights A[r,j], B[r],
  C[r], D[o]; update `s[r] = sum(j, A[r,j] * s[j]) + B[r] * u[r]`; readout
  `y[r] = C[r] * s[r] + D[0] * u[r]`. EXACT.
- **conv1d_causal** (C, K; weight, bias): index {c: C, k: K, r: K - 1}; input u[c]; weights W[c,k],
  Bias[c]; let `xx[k,c] = where(k - (K - 1.5), u[c], x[k,c])` (the K - 1 earlier inputs, then this
  token's; the read x[K-1, c] lies outside the state, is +0 and is not selected); update
  `x[r,c] = xx[r + 1,c]`; readout `y[c] = sum(k, W[c,k] * xx[k,c]) + Bias[c]`. EXACT.
- **selective_scan** (D, N; a, d, dt_softplus): index {c: D, n: N}; inputs x[c], dt[c], Bm[n], Cm[n];
  weights A[c,n], Dk[c]; lets `delta[c] = where(dt[c] - 20, dt[c], log(1 + exp(min(20, dt[c]))))`
  (dt itself when dt_softplus is false) and `xd[c] = x[c] * delta[c]`; update
  `s[c,n] = s[c,n] * exp(delta[c] * A[c,n]) + Bm[n] * xd[c]`; readout
  `y[c] = sum(n, s[c,n] * Cm[n]) + Dk[c] * x[c]`. ROUNDING (log(1 + e) for log1p).

Measured (25 October 2026): in the numpy reference, the linear_attention, ssm_scan and conv1d_causal
rewrites reproduce the hand-written ops' outputs and states bit for bit on the suite's op fixtures,
and selective_scan within 9.1e-8 (log1p, and the hand-written readout's BLAS order); the rewritten
pk_kimi, pk_srd and pk_mamba documents give bit-identical logits and tp_mamba1's within 3.0e-7. In
paperkiln's C++, `ops::recurrence` over the linear_attention body equals `ops::kimi_attention` bit
for bit, and over the ssm_scan body `ops::ssm_scan` (`tests/test_recurrence_op.cpp`). Section 6.4's
num starts at +0, so its first sum is +0 + phi(k_0) v_0, which the recurrence computes; the
hand-written ops store phi(k_0) v_0 itself, which differs only in the sign of a zero.

## 7. The recursion profile

### 7.1 weight_index

A key of the block itself, not of a node: `{"per_layer": [m_0, ..., m_{L-1}]}`, L = block.repeat,
integers m_i >= 0.

- Application i of the block reads every tensor name it holds (`weight`, `bias`, `router`,
  `experts`, ssm_scan's `a b c d`, `position_bias`'s weight, `relative_kv`'s key and value, ...) with
  `"{i}"` replaced by m_i, i.e. `blk.{m_i}.*`, so several
  applications share one stored layer's tensors (tied-block recursion: prelude layers, a shared block
  applied R times, coda layers; paperkiln family_hint `llama_recursive`).
- Everything else per application stays indexed by i: each application has its own instance of
  every state slot (its own kv cache: a shared layer sees different inputs on each pass), and
  per-layer attributes read v_i.
- Absent: m_i = i.
- The stored layers are 0 .. max m_i, each used at least once. Their count is the GGUF's
  `<arch>.block_count` (a file without that key is refused), while `model.n_layers` =
  `block.repeat` counts applications.
- A file whose applications share tensors has architecture `"paperkiln"` (llama.cpp's graphs apply
  each stored layer once).
- An engine that does not implement weight_index refuses the file at load.

### 7.2 tensor_index

`"tensor_index": "application"`, a key of a node in a block with a `weight_index`: in that node,
`"{i}"` in its tensor names means the application index i, not the stored layer m_i (the queries
`rec_app.{i}.*` of llama_attnres under recursion, section 9.2). This is the only second meaning of
`"{i}"`. Every untagged node keeps m_i.

- The key belongs to the ops that name tensors, as `ir.schema.json` lists it: `embedding`,
  `pos_embedding`, `linear`, `scale`, `rmsnorm`, `layernorm`, `moe_ffn`, `depth_attend`, `ssm_scan`,
  `conv1d_causal`, `selective_scan` and `recurrence`. On any other op it is an attribute the op does
  not define (rule 4). Its one value is `"application"` (rule 5). `attention` is not among them, so the
  names of `position_bias` and `relative_kv` always read m_i.
- It governs every tensor name of the node that holds `{i}` (`weight`, `bias`, `router`, `experts`,
  ssm_scan's `a b c d`, a recurrence's `weights`), and only that node: the tag on one node moves no
  other node's names. It is structural (section 3.8), so it holds in every application.
- i runs over every application 0 .. `block.repeat` - 1, so the file holds a tensor for each
  application that reads one (rule 25), whichever stored layer that application binds. These
  tensors lie outside the stored layers: `<arch>.block_count` counts the stored layers 0 .. max m_i
  alone (section 7.1, rule 27).
- State and per-layer attributes are already per application under `weight_index` (section 7.1);
  the tag changes tensor names only.
- Without `weight_index`, m_i = i and the tag would change nothing; it is refused there (rule 19),
  as it is outside the block, where `{i}` has no value (section 3.7).

The two readings are far apart on a trained model. On the suite's `pk_llama_attnres_rec`
(`weight_index` [0, 1, 1, 2], one pair of pseudo-queries per application, `rec_app.0` .. `rec_app.3`),
the document without its tags is valid (every `rec_app.{m_i}` is a tensor of the file, and nothing
forbids one pseudo-query serving several source counts) and computes another function: over the
untagged trace's 8 ids its logits are 5.2e-2 of max(1, max |ref|) from the tagged document's, and
the argmax differs at one of the 8 positions, against the ROUNDING tolerance 1e-5. Cases: the model case and its decode and chunks traces; the trace
`pk_llama_attnres_rec_untagged`, which runs the same file under the untagged document
(`accept/pk_llama_attnres_rec_untagged.ir.json`, also an acceptance case) and expects that
document's values, so an engine that reads the application index wherever a name is `rec_app.{i}`,
or in every node once one is tagged, fails it; and the refusals `tensor_index_without_weight_index`
and `tensor_index_in_head` (rule 19), `tensor_index_value_unknown` (rule 5),
`tensor_index_on_affine` (rule 4) and `tensor_index_absent_application_tensor` (rule 25: a tagged
`rmsnorm` names `blk.3.attn_norm.weight` at application 3, past the stored layers 0 .. 2, which an
engine that applies the tag to `depth_attend` alone accepts). There is no op case: `run_node` runs a
node as application 0 of a one-node block without `weight_index`, where the key is refused.

### 7.3 Embedding re-injection

paperkiln recursion inject `"add"` needs no new op: the embed section gathers the embedding a
second time into `e`, and every block starts with `affine(e; scale {"per_layer": [c_i]}, shift 0)
-> inj`, `add(x, inj) -> x`, c_i = 1 before the first application of repeats 1 .. R-1 and 0
elsewhere (x + 0 e is x up to the sign of zero).

### 7.4 depth_route

Per-token recursion depth (Mixture-of-Recursions, causal token choice; paperkiln recursion schedule
`"mor"`, family_hint `llama_mor`, tiny-chat amendment 4). A control node: it must be the block's
FIRST node (and occurs nowhere else), and it changes which rows the rest of the block runs on.

    {"op": "depth_route", "in": ["x"],
     "weight": "rec.0.router.weight", "route": "token_choice",
     "n_rec": R, "gate": "score" | "none",
     "step": {"per_layer": [s_0, ..., s_{L-1}]},
     "out": "rec_depth"}

- s_i = -1: application i is outside the recursion (prelude or coda): the node does nothing and the
  block runs on every row as usual. s_i = k in 0 .. R-1: application i belongs to recursion step k.
  The applications of step k are consecutive, the steps appear in ascending order 0 .. R-1, and
  every step has the same number of applications. The steps' applications form one run: -1 stands
  only before step 0 (the prelude) and after step R-1 (the coda), never between two steps
  (`[-1, 0, -1, 1, 2]` is refused, rule 20; refusal `mor_step_gap`). `n_rec` R >= 1.
- **(a) Routing**, at the first application of step 0: logits = x W^T, W = `weight` [R, d] (a matmul
  reduction, as linear), [T, R]. Each token's depth n_t = 1 + j*, j* the index of the largest logit,
  ties to the LOWER index; p = softmax over the row (e_j = exp(l_j - max), z summed in ascending j,
  p_j = e_j / z); its gate g_t = p_t[n_t - 1]. The depth and gate hold for the rest of the forward;
  `out` receives the depths [T, 1] (the depth trace, 1 .. R).
- **(b)** At the first application of step k the step's input x_in = x is kept, and the active
  tokens are those with n_t > k. Every application of step k runs the block's other nodes on the
  active rows only: the rows of active tokens, in order, with their absolute positions (pos), so for
  those nodes T is the number of active tokens. Rows of inactive tokens are not computed and stay as
  they are. **What a step sees.** A step's applications start from two values alone, x_in's active
  rows (as `x`) and their positions (`pos`); the values the step's nodes write are local to the step
  (they persist from one of its applications to the next) and only x leaves it, through (c). So the
  block's nodes after depth_route read only `x`, `pos` and values written before them in the same
  application, in every application, the prelude's and the coda's included, since all share one node
  list; and the head reads `x` and the embed section's values, not the block's other values. A
  document that reads anything else (an injection `e` from the embed section, say) is refused at
  load (rule 36; refusal `mor_reads_outside_value`).
- **(c)** After the last application of step k, for each active token: gate `"score"`:
  x = x_in + g (x - x_in), the difference rounded, then the product, then the sum (no fused
  multiply-add); gate `"none"`: x stays the step's output. Inactive tokens keep x_in.
- **(d)** A step with no active token is skipped: its applications run nothing and write no kv rows
  (in decode, the one token has exited).
- A token's depth is a function of its own prefix only (constraint 1, section 7.7), so prefill and
  token-by-token decode give the same depths and, to rounding, the same logits. Depths are nested: a
  token active at step k was active at every earlier step.
- The router tensor is written in full (constraint 4) and stays F32 in paperkiln's Q8_0 variants;
  its ggml ne is {d, R}.
- `out` is the depth report: no node reads it (before the first application of step 0 it holds no
  value).
- depth_route is not defined together with list values (`list`, `append`, `depth_attend`:
  attention residuals under Mixture-of-Recursions, an exploratory draft) or with mask `turn_state`
  (section 9.2): both are refused.
- Class: EXACT routing given the logits; ROUNDING otherwise (a token whose two largest logits differ
  by less than rounding may route differently in another engine: the gate's trace reports the
  smallest margin per prompt).

### 7.5 kv_rows active

A key of a kv slot: `"active"`, the only value (MoR's recursion-wise caching; required on every kv
slot under depth_route). At application i the slot holds exactly the k/v rows of the tokens active
at i (every token when s_i = -1), each written once, when the token passes application i, at its
absolute position with RoPE unchanged. A query at application i attends causally (key position <=
query position) to exactly those rows. A token that exits before application i writes no row there
and never gets one later. Engines keep each application's row positions; truncating the cache to n
tokens keeps the rows whose positions are < n (section 6.9).

So each application keeps its own compact cache: after L tokens, application i's instance of the
slot holds #{t < L : s_i = -1 or n_t > s_i} rows, those tokens' rows in ascending position, and an
application of step k holds only the rows of tokens whose depth exceeds k. The queries of
application i are the active rows at i, and each attends to the rows application i holds at
positions <= its own (the prefix's rows and the current call's earlier active rows). A window and
sinks on the slot and its attention node narrow that set by positions, as section 5.15 states (Q16):
never to the last `window` rows the application holds.

### 7.6 kv_from (reserved)

`"kv_from": {"per_layer": [s_0, ...]}` on a kv slot (MoR's recursive KV sharing): application i
would read, and never append to, the kv rows application s_i wrote for every token; s_i = i is the
default. Reserved and not implemented: a document carrying it is refused. Constraint 3 (section 7.7)
binds any future definition.

### 7.7 Binding constraints

Agreed with ember (23 October 2026), binding on any implementation:
1. Routing is CAUSAL token choice: a token's depth is a function of its own prefix only (a per-token
   threshold on its router score, or top-k over the token's own recursion choices; any capacity
   rule causal), never top-k across the sequence (expert choice), so prefill equals decode and a
   T = 1 step can decide. `route` values other than `"token_choice"` are refused.
2. Under `"kv_rows": "active"`, a query at application i attends to exactly the rows written at
   application i, positions absolute and RoPE unchanged; a token that exits before application i
   writes no row there and never gets one later (engines keep a written-rows mask per application,
   truncated with the cache on prefix reuse or rewind).
3. `kv_from` requires s_i <= i and s_i among the applications every token passes through (the
   prelude or the first recursion step); s_i = i reads and appends as now.
4. The router weight names are written out in full by the exporter (rec.0.router.weight,
   rec.1.router.weight, ...): `{k}` is not a substitution rule, and a router name holding `{` is
   refused.
5. The gate exports each prompt's per-token depth trace from mtstudio, so engines check routing
   decisions token by token as well as logits.

## 8. The chat profile

### 8.1 Conversations and the template record

The chat template travels in the model file as data: the GGUF string key `paperkiln.chat_template`
holds the record as JSON. One schema serves instruction tuning and chat alike. The implementations
are paperkiln's `tools/chat_template.py` and its C++ twin `include/microtorch/chat_template.hpp`,
which ember vendors.

*Why (informative).* This avoids four common failures. Base, instruct and chat checkpoints are not
three formats: an instruction is a one-exchange conversation. System prompts are not handled
differently in every engine or silently dropped: the template declares its system mode. Single-turn
SFT does not quietly erode multi-turn behaviour: the loss policy is declared and recorded. Templates
do not drift between a tokenizer config, per-engine code and Jinja: one record, rendered by every
engine and gated.

#### 8.1.1 Schema

A conversation is `{"messages": [{"role": "system" | "user" | "assistant", "content": str, "loss":
bool?}], "uri", ...}`.

- At most one system message, and only first. After it, user and assistant alternate, starting with
  the user.
- Every message has a `role` and a string `content` (a message without one is refused; an assistant
  message with tool calls whose text is empty has `""`). A `tool_calls` member (even null) under a
  record without `tool`, or on a message that is not an assistant message, is refused, as is a
  `tool_call_id` on a message that is not a tool message, under any record. Other members are ignored.
- A training conversation ends on an assistant message; a prompt conversation ends on a user message.
- An instruction example is a conversation with one exchange.
- `loss` (assistant messages only) overrides the loss policy for that message; `false` keeps the reply
  as context only.
- The `tool` role, an assistant message's `tool_calls` and a conversation's `tools` (tool definitions)
  are refused under every record without the field `tool`; `paperkiln/2` defines them, with the
  structure section 8.1.6 states.
- Every pinned `corpus/data/<id>/dialogues.jsonl` of paperkiln is already in this form, so nothing is
  converted and no hash changes. paperkiln's v1 fetchers drop or strip system messages
  (`_common.normalise_dialogue`), and that rule stays in every v1 corpus.
- `{"prompt", "completion"}`, `{"segments"}` and the pretraining text (`render_document`: the
  rendered exchanges plus "\n") are rendered v1 forms. mtstudio still reads them, and new data is
  stored as conversations. `chat_template.py parse` turns a rendered file back into conversations,
  refusing any row that does not round-trip, and writes a ledger with the source and output sha256.
- The evaluation sets keep their item schemas and convert at read time (`from_carry_item`,
  `from_prompt`).

#### 8.1.2 Loss policy

`train.sft_loss`: `{"turns": "all" | "final" | "each"}` applies to conversation rows; legacy rows
refuse it, because their rendering fixes their loss.

- `all` (the default, behind `DEFAULT_SFT_TURNS` / `kDefaultSftTurns`) trains every reply; `final`
  trains the last reply kept; `each` gives one row per reply.
- A reply's trained tokens are its content plus the template's assistant close (v1: the
  `<|endoftext|>` after it).
- Markers, user and system text, padding, state ids and state positions never carry loss. State
  blocks are inserted after rendering, as before.
- A row keeps the longest prefix of whole exchanges that fits T + 1 (`each`: a reply whose prefix
  does not fit is dropped); a reply is never cut.
- Runs log a `sft_conversations` event with the template, the policy and the counts.

#### 8.1.3 The template record

The GGUF string key `paperkiln.chat_template` holds the record as JSON.

- **Fields:** `name`, `version`, `bos`, `content` (`raw`, or `trim`: whitespace stripped from both
  ends of each message's own text, the whitespace of Python's `str.strip()` and HF's Jinja `trim`:
  U+0009 to U+000D, U+001C to U+001F, U+0020, U+0085, U+00A0, U+1680, U+2000 to U+200A, U+2028,
  U+2029, U+202F, U+205F and U+3000, and nothing else, so U+200B stays; under `fold` the system text
  is not trimmed), `system` (`fold` with `join`, `turn` with `open`/`close`, or `none`),
  `user` and `assistant` `{open, close}`, `generation_prompt`, optional `generation_prompt_nothink`,
  `stop` `{eos, turn}`, `past_assistant` (`keep` or `strip_think`), optional `default_system`, and
  `reserved`. `paperkiln/2` adds `tool`, `tool_call` and `tools` (section 8.1.6). Third-party records
  add `markers`, `match` and `architectures`.
- **Rendering:** a conversation renders as bos, then each message as open + content + close, then
  the generation prompt when a reply is wanted. The system message renders by its mode: `turn` as
  its own open + content + close; `fold` as no turn of its own, its text plus `join` opening the
  first user message's content; `none` refuses it. A record's `default_system` stands in when the
  conversation has no system message. Content holding a `reserved` string is refused, in
  both languages and by `parse` alike, so no text can forge a turn: the check reads each content as
  rendered, after `trim` and before `strip_think` (so a reserved string inside stripped reasoning is
  still refused). Under `trim` the system text of mode `turn` is trimmed like any message,
  `default_system` included (only `fold` keeps it untrimmed). `generation_prompt_nothink` replaces the
  generation prompt only when the request says `thinking: false` and the record has it; otherwise the
  plain `generation_prompt` is rendered.
- **Required fields:** `name`, `version`, `user`, `assistant`, `generation_prompt` and `stop`; the others
  may be absent (`bos` "", `content` `raw`, `system` mode `none`, `past_assistant` `keep`, `reserved`
  none). `user`, `assistant` and `tool` hold `open` and `close` only, `tool_call` `open` and `join`,
  `tools` `join` and `sep`, and another member in them is refused; the members of `system` other than
  `mode`, `open`, `close` and `join`, and of `stop` other than `eos` and `turn`, carry nothing a renderer
  reads and are not refused. `<unk>` is ordinary text. A field
  or field value the renderer does not implement, a conversation that breaks section 8.1.1's structure
  and a reserved role are refused, never rendered. `tools/ir_chat.py` is the reference renderer; the
  suite's chat cases (`conformance/chat/render.json`, the adapter method `render_chat`, section 11.3)
  check it on every record below.
- **paperkiln/1** is this section's template byte for byte:
  - `<|user|>U<|assistant|>A<|endoftext|>` per exchange, then `<|user|>U<|assistant|>`;
  - no BOS; ids 0-3 are `<unk>`, `<|endoftext|>`, `<|user|>` and `<|assistant|>`;
  - stops at end-of-text (`eos`) or a generated role marker (`turn`, marker dropped);
  - reserved: the three markers;
  - system `fold`: the system text plus "\n\n" opens the first user message in the rendered window.
    No token is spent and no vocabulary changes (Gemma's convention). `none` is implemented too;
    the mode is `SYSTEM_MODE_V1` / `kSystemModeV1`, `fold` for v1.
- **paperkiln/legacy-word** is the word-level `user: U assistant: A ` form, prompting with
  `assistant:`, which ember serves in chat mode for those models.
- **Third-party records (ember owns and gates them):** `chatml-qwen2.5/1` (default system message),
  `chatml-qwen3/1` and `gemma3/1`.
  - `chatml-qwen3/1` has `strip_think` and `thinking`: HF's `enable_thinking=False` appends
    `<think>\n\n</think>\n\n` to the generation prompt, while the default (thinking on) appends
    nothing.
  - `gemma3/1`: bos `<bos>`, trim, fold with "\n\n", user and model turns between `<start_of_turn>`
    and `<end_of_turn>\n`, stops `<end_of_turn>` and `<eos>`.
  - `strip_think` applies only to content containing `</think>`:
    `content.split("</think>")[-1].lstrip("\n")` for each assistant message before the last user
    message. Prompt conversations always end on a user message, so HF's re-wrapping of replies after
    it never arises and is not implemented.
  - ChatML's assistant close is `<|im_end|>\n`, and its stop is `<|im_end|>`. A close's trailing text
    is rendered in the next prompt and never generated, and both engines re-render the history every
    turn: no stop set holds "\n".
  - Engines never inject a system message that neither the client nor the record supplies.

#### 8.1.4 Resolution

`resolve()`, the same in every engine:

1. a declared `paperkiln.chat_template`; a field value the engine does not implement is refused by
   name, never guessed (`paperkiln/2` resolves by this step only, section 8.1.6);
2. ids 0-3 with exactly the v1 specials' texts gives `paperkiln/1` (the 8k `bpe` and the 16k
   `llama3_bpe` vocabularies both qualify, tested), so every file exported before the declaration
   resolves to v1 and is never rewritten;
3. `tokenizer.ggml.model` "word" gives `paperkiln/legacy-word`;
4. a third-party record, matched by the sha256 of the file's exact `tokenizer.chat_template`, else
   by the marker check (each of its `markers` a single token and `general.architecture` in its
   `architectures`; llama.cpp b6000's Gemma 3 GGUF carries no `tokenizer.chat_template`);
5. otherwise no chat mode, with the reason logged.

An engine older than the template work resolves any file with the v1 specials as v1 by step 2. That
is supported only without system messages.

#### 8.1.5 Export

Every GGUF of a model whose vocabulary has the v1 specials (or whose spec names `chat.template`)
carries the record and `tokenizer.chat_template`, after the tokenizer block, in every variant.
`tokenizer.chat_template` is the Jinja generated from the record, equal to the record's rendering,
for llama.cpp `--jinja`, LM Studio and Ollama's import. A `<name>.Modelfile` sits beside the F32
GGUF: Ollama's TEMPLATE in Go syntax, plus `PARAMETER stop` per stop string. Third-party tools stop
at `tokenizer.ggml.eos_token_id`; only paperkiln's and ember's engines also stop at a generated role
marker (GGUF has no key for a second stop set). Word-level vocabularies write no keys.

State tokens are defined over `paperkiln/1` only (section 8.2).

#### 8.1.6 v2: system and tool turns (`paperkiln/2`)

`paperkiln/2` (v1.4.2, as built in paperkiln 22f8fe6d) gives the system message a turn of its own behind
its own token, and adds tool calls, tool results and tool definitions. The tiny-chat programme and every
file without the declaration stay `paperkiln/1`.

- **Declaration.** The record is always declared (`paperkiln.chat_template`, `"name": "paperkiln",
  "version": 2`) and never inferred: a file without the key and with the v1 specials resolves to
  `paperkiln/1` (section 8.1.4, step 2), so every v1 file behaves as before. Engines act on the record's
  fields (`tool`, `tool_call`, `tools`), not on its version. A file that declares it has no state tokens
  (section 8.2, rule 28).
- **The record**, member for member:

      {"name": "paperkiln", "version": 2, "bos": "", "content": "raw",
       "system": {"mode": "turn", "open": "<|system|>", "close": ""},
       "user": {"open": "<|user|>", "close": ""},
       "assistant": {"open": "<|assistant|>", "close": "<|endoftext|>"},
       "tool": {"open": "<|tool|>", "close": ""},
       "tool_call": {"open": "<|tool_call|>", "join": "\n"},
       "tools": {"join": "\n\n", "sep": "\n"},
       "generation_prompt": "<|assistant|>",
       "stop": {"eos": ["<|endoftext|>"],
                "turn": ["<|user|>", "<|assistant|>", "<|system|>", "<|tool|>"]},
       "past_assistant": "keep",
       "reserved": ["<|endoftext|>", "<|user|>", "<|assistant|>", "<|system|>", "<|tool_call|>",
                    "<|tool|>", "<|spare_3|>", "<|spare_4|>", "<|spare_5|>", "<|spare_6|>",
                    "<|spare_7|>"]}

  (`"\n"` is one newline, as in JSON.) `<|tool_call|>` is not a stop: a reply goes on after a call
  token until a stop.
- **The three fields.** `tool` is `{open, close}`, a tool result's turn; `tool_call` is `{open, join}`, a
  call inside an assistant turn; `tools` is `{join, sep}`, the definitions inside the system turn. A
  record holding any other member in them, `tool` without `tool_call` or the reverse, `tools` without
  them, or `tool` with a system mode other than `turn` is refused by name (section 8.1.4, step 1).
  Under a record without `tool`, tool messages, `tool_calls` and a non-empty `tools` are refused (an
  empty `tools` list is no definitions under any record); under one without `tools`, a non-empty
  `tools` is refused.
- **Tokens.** paperkiln's 16k Llama 3-scheme vocabulary plus 8 control tokens (GGUF token type 3) at
  ids 16384 .. 16391, in this order: `<|system|>`, `<|tool_call|>`, `<|tool|>`, `<|spare_3|>` ..
  `<|spare_7|>`; 16,392 tokens, the merges and every text id unchanged, so every `.ids` corpus stays
  valid. They parse out of text by their type, as the v1 specials do. A model gains their rows with each
  new embedding row the mean of the old rows (the head is tied). A consumer refuses a file whose
  declared record has `tool` unless each of its `reserved` strings is a single token of the vocabulary
  of type 3 (control) (paperkiln's `check_control_tokens`). The 8k `bpe` vocabularies stay v1.
- **Structure** (under a record with `tool`, in place of section 8.1.1's alternation). Roles `system`,
  `user`, `assistant` and `tool`. At most one system message, and only first; then a user message;
  after a user message, an assistant message; an assistant message with k > 0 tool calls is followed by
  exactly k tool messages, the answers to its calls in order, then an assistant message; an assistant
  message without calls is followed by a user message. A prompt conversation (a reply wanted) ends on a
  user message or on the tool message that answers an assistant message's last call; a training
  conversation ends on an assistant message, with or without calls.
  - `tool_calls`, on assistant messages only: absent or `null` (no calls), or a non-empty list of
    `{"id"?, "type"?, "function": {"name", "arguments"?}}`, where `type`, when present, is
    `"function"`; `id`, when present, a string; `name` a non-empty string without "\n"; `arguments` a
    string, `""` when absent, rendered verbatim (it is the call's text, not parsed). Other members of a
    call are ignored.
  - `tool_call_id`, on tool messages only: when present, the id of the call the message answers, a
    call without an id counting as id null; so a string `tool_call_id` against a call without an id is
    refused, and a null one is accepted only there. It is checked, never rendered.
  - `loss` stays on assistant messages only. Rendering accepts empty contents; a stored conversation
    has non-empty content everywhere except an assistant message with calls.
- **Rendering** (section 8.1.3's, with these additions). The system turn is `system.open` + its content
  + `system.close` (`<|system|>S`, no close). With tool definitions, that content is the system text,
  then `tools.join` ("\n\n"), then the definitions' texts joined by `tools.sep` ("\n"); with no system
  message, or an empty one, it is the definitions alone, and the system turn is rendered for them. An
  assistant message renders as `assistant.open` + content, then for each call `tool_call.open` + name +
  `tool_call.join` + arguments, then `assistant.close`. A tool message renders as `tool.open` + content +
  `tool.close` (`<|tool|>R`: no close, and no id or name, as results follow call order). A content, a
  call's name or arguments, or a definition's text that holds a `reserved` string is refused. A
  conversation of user and assistant messages only renders exactly as under `paperkiln/1`.
- **A definition's text.** Each definition is a JSON object (anything else is refused), written as
  Python's `json.dumps(d, ensure_ascii=False)`, which is HF transformers' Jinja `tojson` (paperkiln's C++
  `detail::dump_py` writes the same): no whitespace except one space after each `,` and each `:`; members
  in the order received; `true`, `false`, `null`; an integer in decimal (`-0` as `0`); a string in double
  quotes, with `"` and `\` written `\"` and `\\`, U+0008, U+0009, U+000A, U+000C and U+000D written `\b`,
  `\t`, `\n`, `\f` and `\r`, every other code point below U+0020 as `\u` and four lowercase hex digits,
  and every other code point as itself (U+007F and non-ASCII text included). A definition holding a
  number written with a fraction or an exponent (`1.5`, `1.0`, `1e3`) is refused: two serialisers may
  write it differently.
- **Reply parsing** (`parse_reply`). A generated reply, its ids without the stop, is split at every id of
  `tool_call.open`'s control token, never on text, and each part is decoded (the tokenizer's decoding of
  the part's ids). Part 0 is the content. Every later part must hold `tool_call.join` ("\n") after at
  least one character: the name is the text before its first "\n", the arguments the rest (empty, or
  holding further newlines). The calls get the ids `call_0`, `call_1`, ... in order and the type
  `"function"`. When any later part does not parse (no "\n", or an empty name), the reply has no calls:
  the engine returns the whole decoded reply as the content, the call tokens decoded as their text.
  Ordinary BPE pieces that spell `<|tool_call|>` are content, as is a spare or a non-stop marker, which
  decodes as its text.
- **Loss** (section 8.1.2). Trained: an assistant message's content, its call pieces (the call tokens,
  names, joins and arguments) and its close. Never trained: the system turn (system text and
  definitions), user text, tool results and role markers. `train.sft_loss` applies per assistant message
  (`each`: one row per reply, cut after it); a row keeps the longest prefix of whole exchanges that fits
  T + 1, an exchange being a user message and every message up to the next user message. An engine that
  drops history for its context drops the oldest whole exchange and keeps the system turn.
- **Export** (section 8.1.5). The GGUF carries the record, its `tokenizer.chat_template` (the Jinja of the
  record, with `tools`, `tool_calls` and tool messages rendered as the record renders them, under HF's
  `tojson`) and the 16,392 tokens with their types; the Modelfile renders the turns, and its tool
  definitions and call arguments go through Ollama's Go JSON, so their text there is not the record's.
- **Gates.** The v1 parts of paperkiln's `tests/chat_template_cases.json` byte-identical (32 cases,
  records, Jinja, Modelfile, resolutions); 42 v2 conversations (18 rendered, 24 refused) and 12
  reply-parsing cases, C++ = Python, which are also this suite's (section 11.2); the v2 Jinja equal to
  the record on 1,000 random conversations; paperkiln's v2 SFT rows equal to an independent
  construction; ember against `mtstudio chat` 40/40 on every field over 40 v2 prompts (ember b49eb5c,
  `LOCAL_LLM_STACK.md` section 10, 27 October 2026).

#### 8.1.7 Gates

- **G1, v1 bit identity:** today's strings and ids on every frozen prompt set, Carry-5 and Carry-12
  item, and pinned corpus.
- **G2, SFT rows:** conversation rows equal the rendered rows, with and without state tokens.
- **G3:** state-token insertion unchanged.
- **G4, declared vs absent:** a v1 GGUF with and without the keys gives the same /chat prompt ids,
  replies and stop reasons, 40/40 in mtstudio and ember (ember 69d7ca2, the tiny-chat pin, included).
- **G5, template parity:**
  - ember vs mtstudio 40/40 on a system-prompt set;
  - the generated Jinja equals the record under jinja2;
  - third-party records equal HF's templates;
  - the hand-written cases in paperkiln's `tests/chat_template_cases.json` pass in both languages;
    the same cases are the conformance suite's chat cases (74 conversations, 32 of them v1, and 12
    reply-parsing cases), so any engine can show them.

### 8.2 The state-token template contract

`paperkiln.state_tokens.k` = k > 0 (GGUF metadata, uint32, absent = 0) declares the state ids S_0 .. S_{k-1} (section 6.8).

- They are defined over the chat template `paperkiln/1` only: a file that declares another
  `paperkiln.chat_template` with k > 0 is refused. A file with state tokens has architecture
  `"paperkiln"` (llama.cpp's graph has no state embedding), and its `tokenizer.specials` name the
  `eot`, `user` and `assistant` ids (below).
- A system message is folded into the first user turn by `paperkiln/1`, so it is that exchange's
  text: under kv evict turn_state it is evicted with that turn and carried on by its state block.
- Every exchange of the served template is

      <|user|> U <|assistant|> A <|endoftext|> S_0 .. S_{k-1}

  that is, over ids: an exchange opens at `<|user|>`, has a reply once `<|assistant|>` follows it,
  and closes at the next `<|endoftext|>`, which the engine follows with S_0 .. S_{k-1} (in prompts,
  and after a generated `<|endoftext|>` as a k-token prefill, no sampling). Every `<|endoftext|>`
  ends the exchange state, so documents without exchanges get no block.
- A reply that ends without `<|endoftext|>` (max_new_tokens, a stop string, a client abort) gets
  `<|endoftext|>` and S_0 .. S_{k-1} before the next `<|user|>`.
- **Raw ids** (id lists a client supplies): an id >= vocab is accepted only as part of a complete
  block S_0 .. S_{k-1} in order directly after an `<|endoftext|>` that closed an exchange; absent
  blocks are not inserted into raw ids. Text the engine encodes (a chat request, a text
  `/completion` prompt) gets the insertion rule above.
- `tokenizer.specials` names the same eot, user and assistant ids as the resolved template (in
  `paperkiln/1`, the ids of `<|endoftext|>`, `<|user|>` and `<|assistant|>`); a file whose specials
  differ from its vocabulary's marker ids is refused.
- *Informative:* training (paperkiln) inserts the same blocks in pretraining streams and SFT rows (a
  block that would end an SFT row is left out: no position reads it), with no loss where the target
  is a state id or the input is a state position. Reference: `nn::insert_state_ids`,
  `nn::state_ids_error`, `ops::turn_state_codes` (`llama.hpp`, `ops.hpp`); paperkiln's `tools/state_tokens.py`.

## 9. Families (informative)

### 9.1 family_hint

`family_hint` labels the graph; engines never derive semantics from it (an engine may use it, with a
structural match, to pick a tuned path that computes the same function). For paperkiln's own files
it equals the family below; for third-party files it equals `general.architecture`. The
architecture itself is normative (section 3.1, rule 38): a llama.cpp architecture only for a document
that is that architecture's graph (section 9.3), and "paperkiln" for every other file, including one
whose tensors are a llama.cpp architecture's set (gpt2_swa, kimi, recurrence, plain flex below). No
llama.cpp graph computes a "paperkiln" file, and llama.cpp refuses it. Its metadata keys are
`paperkiln.{context_length, embedding_length, block_count, vocab_size, attention.head_count,
feed_forward_length}` (a gpt2-tensor export also writes `paperkiln.attention.layer_norm_epsilon`, as a
file relabelled by `tools/gguf_relabel.py` keeps its old architecture's keys under `paperkiln.`).

### 9.2 paperkiln families

A tensor that is the same thing as a gpt2 tensor keeps llama.cpp's gpt2 name: `token_embd`,
`position_embd`, `blk.{i}.attn_norm`, `attn_qkv`, `attn_output`, `ffn_norm`, `ffn_up`, `ffn_down`,
`output_norm`, `output` (each `.weight`, and `.bias` where the module has one); new tensors are
`blk.{i}.<name>.weight` (and `.bias`). Worked documents: `examples/*.ir.json`.

- **gpt2, gpt2_swa**: as above (fused qkv split q | k | v, tanh GELU, learned positions);
  gpt2_swa adds a window and sinks. gpt2 has the architecture gpt2; gpt2_swa has gpt2's tensors and
  the architecture "paperkiln" (llama.cpp's gpt2 graph has no window, so under gpt2 it computed dense
  attention; paperkiln writes "paperkiln" from its revision 1.3 exporter on).
- **llama**: pre-RMSNorm (eps 1e-5), interleaved RoPE on every head, SwiGLU, no biases, tied output
  (`examples/chat_sftfinal.ir.json`); with MoE, `moe_ffn` in place of the SwiGLU (`examples/moe_tiny.ir.json`).
- **kimi**: gpt2 with `linear_attention` (slot "lin": num, den [heads, head_dim]) in place of
  attention; the gpt2 tensor set exactly, architecture "paperkiln" (under gpt2, as paperkiln exported
  it before revision 1.3, llama.cpp loaded it and computed softmax attention: rule 38 refuses such a
  file, and `tools/gguf_relabel.py` relabels it).
- **srd**: gpt2 whose attention blends two paths of one qkv per token: p =
  pred_down(gelu(pred_up(h))) on the normalised input h (`blk.{i}.srd_pred_up` [d/4, d],
  `srd_pred_down` [d, d/4], both biased), g = sigmoid(affine(rms_row(h - p), 4, -1)), gi =
  affine(g, -1, 1); out = mul_col(attention, g) + mul_col(linear_attention, gi), then attn_output.
  Slots kv and "lin" (paperkiln).
- **attnres**: gpt2 blocks whose residual stream is a list (Attention Residuals, block form, one
  list entry per transformer block): embed ends with list [x] -> "bank"; block i reads h =
  depth_attend(bank; blk.{i}.attnres_attn_q), a = attention sublayer of layernorm(h), h2 =
  depth_attend(bank, a; blk.{i}.attnres_ffn_q), f = ffn of layernorm(h2), then appends a + f to
  bank; the head reads depth_attend(bank; attnres_out_q). Pseudo-queries [d];
  blk.0.attnres_attn_q is written as zeros and never read (its bank has one entry). Slot kv only
  (paperkiln).
- **flex, flex_highway, flex_plain**: the flex decoder (residual, highway, plain): pre-norm
  layernorm or rmsnorm (attn_norm / ffn_norm / output_norm, no bias under rmsnorm); FFN gelu
  (ffn_up, ffn_down biased), relu (the same with relu), a runtime-defined activation `expr:<e>` (the
  same with an expr node over one input) or swiglu (ffn_gate, ffn_up, ffn_down, no biases:
  down(silu(gate h) * up h)); positions learned or sinusoidal (position_embd holds the fixed table,
  rounded to f32) or rope (no position_embd; a rope node, all heads, interleaved pairs, theta 10000,
  between split and attention, as in the llama block); attention exact or windowed. A
  runtime-defined norm `expr:<e>` (e over the one input x; the row reductions allowed) computes
  y = e(x) * g + b row by row, g and b learned [d] (initialised to 1 and 0, layernorm's affine): an
  expr node (in [x], out h), then a scale node on h with weight g [d] and bias b [d], named
  blk.{i}.attn_norm_gain / blk.{i}.ffn_norm_gain / output_norm_gain (.weight, .bias): (e(x) * g) +
  b, as paperkiln's mul_row then add_bias. Positions may instead be T5's relative bias
  (`position: "t5_relative"`, revision 1.3): no position table and no rope node; every attention node
  carries `position_bias` {weight `rel_pos_bias.weight` [rel_buckets, heads], one table read by every
  layer; buckets `ops::t5_position_buckets(rel_buckets, rel_max_distance)`} (section 5.15), and the
  attention is exact (no window). Or Shaw's relative vectors (`position: "shaw_relative"`, revision 1.4):
  no position table and no rope node; every attention node carries `relative_kv` {key
  `blk.{i}.attn_rel_k.weight`, value `blk.{i}.attn_rel_v.weight` or null, max_distance K} (section 5.15),
  each layer's own vectors, shared by the heads ([K + 1, head_dim]) or one set per head ([K + 1, d]), and
  the attention is exact. Independently of the positions, Primer's multi-dconv-head attention
  (`qkv_conv: K`, So et al. 2021, arXiv 2109.08668) puts a `conv1d_causal` node (channels 3d, kernel K, no
  bias, weight `blk.{i}.attn_qkv_conv.weight` [3d, K], recurrent slot `qkv_conv` with tensor x [K - 1, 3d])
  between the fused projection `qkv` and its split, before the rope node: a depthwise convolution is per
  column, so one node over q's, k's and v's columns is the paper's three. Each sublayer f combines with
  its input stream x as
  residual x + f, plain f, or highway x + sigmoid(W x + b) * (f - x) with W, b =
  blk.{i}.highway_attn / highway_ffn [d, d] + bias, nodes linear, sigmoid, sub, mul, add. Two options
  of 26 October 2026 use existing ops only. An expr activation, norm or embedding tail may read more
  inputs than its row: a gate projection (a `linear` of the sublayer's input, `blk.{i}.ffn_gate`), and
  learned scalars and per-channel vectors, each a `scale` node over a row of ones (an `expr` "1" on the
  row) with weight `blk.{i}.<slot>_p_<letter>.weight`, `embd_p_<letter>.weight` or
  `output_norm_p_<letter>.weight` ([1] repeated, or [d]); the expr reads them as a, b, .... And the
  norm of each sublayer has a placement: pre (the default), post (on the stream after the residual),
  sandwich (pre, and `attn_post_norm` / `ffn_post_norm` on the sublayer's output), branch_post (on the
  sublayer's output only) or none; with both sublayers post the head has no output_norm. The GPT-2
  flavour (layernorm, gelu, learned, residual) exports as gpt2 (dense; architecture gpt2) or gpt2_swa
  (windowed; gpt2's tensors, architecture "paperkiln"); plain with that flavour is gpt2's tensors with
  IR-only semantics (no gpt2 graph omits the skip connections), architecture "paperkiln"; the others
  are "paperkiln" with tensors of their own.
- **recurrence**: kimi's model with its linear attention replaced by a recurrence node (section 6.10) over
  q, k and v, slot "rec": the gpt2 tensor set exactly, architecture "paperkiln" (only the IR states
  the model; gpt2 before revision 1.3, as kimi). mtstudio's `arch.custom.attention` "recurrence" trains it, with `arch.custom.recurrence` the
  node's body (index, inputs q, k, v, let, update, readout; no weights of its own); without it the body
  is section 6.10.6's linear attention, and the model is kimi's bit for bit. Documents rewritten by
  `tools/ir_recurrence.py` keep their family's tensors and take the hint `<family>_recurrence`.
- **mamba** (paperkiln's, not Mamba-1): no positions; block: h = layernorm(x) (blk.{i}.ssm_norm),
  u = ssm_in(h) [n], y = ssm_out(ssm_scan(u)) [d], y = y * gelu(ssm_gate(h)) * gelu(ssm_out_gate(x))
  (multiplied in that order), x = x + y; projections biased; ssm_scan tensors blk.{i}.ssm_a [n, n],
  ssm_b, ssm_c [n], ssm_d [1]; slot "ssm": s [n]. Head: output_norm, then output with a bias
  (paperkiln).
- **llama_recursive**: the llama block (tensor names and nodes as llama) under a block
  weight_index: n_layers = prelude + block * repeats + coda applications of prelude + block + coda
  stored layers (`block._recursion` records the counts for humans); architecture "paperkiln", whose
  block_count is the stored count. With inject "add", embed adds a second embedding node (out e) and
  the block an affine and an add at its start (section 7.3).
- **llama_mor**: llama_recursive with per-token depth: the block's nodes are a depth_route node
  (router rec.0.router.weight [R, d], at the first application after the prelude) and then the llama
  block's; weight_index lists every application a token can take (prelude, the shared block R times,
  coda); the kv slot has `"kv_rows": "active"`. No injection. Architecture "paperkiln", block_count =
  the stored layers.
- **llama_attnres**: the llama block (tensor names and nodes as llama) whose residual stream is a
  list, one entry per sublayer: embed ends with list [x] -> bank. Block i: x = depth_attend(bank;
  blk.{i}.attnres_attn_q), then affine(x; scale {"per_layer": [1, 3, 5, ..]} (entry i = 2i + 1),
  shift 0); the llama attention sublayer of rmsnorm(x; attn_norm) gives o; append(bank, o). x =
  depth_attend(bank; blk.{i}.attnres_ffn_q), then affine(x; scale {"per_layer": [2, 4, 6, ..]}
  (entry i = 2i + 2), shift 0); the SwiGLU of rmsnorm(x; ffn_norm) gives f; append(bank, f). Head:
  depth_attend(bank; attnres_out_q), then affine(x; scale 2L + 1 (a plain scalar: the head sits
  outside the blocks), shift 0), output_norm, output. Keys and weights are depth_attend's own; the
  scale by the source count S is the only attnres-specific step. Every reader of the stream is an
  RMSNorm, so a mean of sources and their sum differ only through eps; multiplying by S makes the
  zero-query model exactly llama (Kimi's convex mix is llama with eps * S^2). Pseudo-queries are
  [d], F32 in every variant; blk.0.attnres_attn_q is written as zeros and never read (a loader may
  require it present; an exporter must not drop it). Under weight_index the affine attributes and
  the bank are per APPLICATION, not per stored block: application i's source count grows with i
  even when it binds blk.{m_i}; the queries are rec_app.{i}.* with `"tensor_index": "application"`
  (section 7.2). Architecture "paperkiln". Reference: paperkiln c48ca962. Cases: pk_llama_attnres
  (no recursion) and pk_llama_attnres_rec (fixed recursion, `tensor_index`; section 7.2's cases).
- **llama_state**: the llama block (tensor names and nodes as llama) with an embedding state_weight
  (state_embd.weight [k, d]) and the state-token template; the attention mask is turn_state with kv
  evict turn_state (S-gist), or absent (causal: S-full, the insertion alone). Architecture
  "paperkiln" (llama.cpp's graph has no state embedding). Under fixed recursion the hint stays
  llama_recursive and the same keys carry the rest; with Mixture-of-Recursions it is not defined
  (paperkiln refuses it, and a document combining depth_route with mask turn_state is refused).

### 9.3 Third-party families

An engine that has the capability (section 3.1) builds the document of a llama.cpp GGUF without
`paperkiln.ir` from `general.architecture` and its metadata: the graph Appendix D states (normative
there, for this and for rule 38; `tools/gguf_ir.py` is the reference builder and `tools/ir_numpy.py`
runs it). family_hint = general.architecture; llama.cpp's tensor names. The summaries below are
informative: a file with an embedded document names one of these architectures only when its
document is the graph Appendix D builds from the file (rule 38), and an engine without the capability
refuses a file without a document.

- **llama, mistral**: pre-RMSNorm, interleaved RoPE, grouped-query attention, SwiGLU;
  `<arch>.attention.sliding_window` w sets window w, sinks 0 (HF Mistral's mask). llama.cpp has no
  architecture "mistral" (b6000 and its later releases refuse such a file): its converter writes a
  Mistral checkpoint as "llama" without the window key, and its llama graph has no window, so llama.cpp
  computes a converted Mistral without its window (the suite's `tp_mistral` and `tp_mistral_hd16`
  test this builder, not llama.cpp). A llama file with `llama.expert_count` and `ffn_gate_inp` (Mixtral's layout, revision
  1.3): `moe_ffn` in place of the SwiGLU (router `ffn_gate_inp`, experts `ffn_{gate,up,down}_exps`,
  top_k `expert_used_count`, gate_norm renorm, activation silu), as llama.cpp's llama graph.
- **qwen2, qwen3**: llama with NEOX RoPE; qwen2 biased q/k/v, qwen3 per-head q/k rmsnorm (group
  head_dim) before RoPE.
- **gpt2**: as paperkiln's gpt2 (fused qkv split q | k | v, tanh GELU).
- **gptneox**: gpt2 without positions, NEOX partial RoPE (n_rot = rope.dimension_count), exact GELU,
  untied head; parallel residual x = (ffn(ln2 x) + x) + attn(ln1 x) when use_parallel_residual.
- **gemma3**: embedding then affine(scale sqrt(d)); sandwich rmsnorms (attn_norm,
  post_attention_norm, ffn_norm, post_ffw_norm), q/k rmsnorm, NEOX RoPE, GeGLU (gelu tanh of gate,
  times up), scale 1/sqrt(key_length); local layers (i % 6 < 5) have window =
  attention.sliding_window and theta 10000, global ones window 0 and rope.freq_base: per-layer
  attributes.
- **mamba** (Mamba-1; not paperkiln's mamba, which has architecture "paperkiln"): rmsnorm, ssm_in ->
  split x | z, conv1d_causal (slot "conv") + silu, ssm_x -> split dt | B | C, ssm_dt (biased),
  selective_scan (slot "ssm"; a = ssm_a, d = ssm_d), * silu(z), ssm_out, residual add.
- **phi3** (revision 1.5): pre-RMSNorm; fused `attn_qkv` split q | k | v (or separate q, k, v); NEOX RoPE over
  `rope.dimension_count` dims, with LongRoPE's `freq_factors` {short `rope_factors_short.weight`, long
  `rope_factors_long.weight`, original_ctx `rope.scaling.original_context_length`} and `attn_factor`
  `rope.scaling.attn_factor` when the file holds the factor tensors; scale 1/sqrt(head_dim);
  `attention.sliding_window` w > 0 is the window (transformers' Phi-3; llama.cpp b6000 disables Phi's window);
  SwiGLU from the fused `ffn_up` (gate | up, a `split`); `output` or the tied embedding. Other RoPE scaling and
  the phi3 MoE layout are refused by name.
- **gemma2** (revision 1.5): gemma3's block without q/k norms; one RoPE base (`rope.freq_base`, default 10000)
  on every layer; local layers i % 2 == 0 with window `attention.sliding_window` (default 4096), global ones
  window 0; attention `softcap` = `attn_logit_softcapping` (default 50); the head's logits through `expr`
  `tanh(x / c) * c`, c = `final_logit_softcapping` (default 30); scale 1/sqrt(key_length), 1/sqrt(d / heads)
  for the 27B (46 layers).
- **qwen2moe, qwen3moe** (revision 1.5; wiring, no new op): qwen2's and qwen3's attention; `moe_ffn` in
  llama.cpp's expert layout, gate_norm raw for qwen2moe (llama.cpp b6000's graph, which ignores transformers'
  `norm_topk_prob`) and renorm for qwen3moe; qwen2moe adds its shared expert, a SwiGLU over
  `ffn_{gate,up,down}_shexp` times sigmoid(`ffn_gate_inp_shexp` h) per token (a `linear` with one output,
  `sigmoid`, `mul_col`). Mixtral is the llama bullet's MoE form.
- **deepseek2** (revision 1.5; DeepSeek-V2 in llama.cpp's MLA layout): pre-RMSNorm; the query from `attn_q`
  (V2-Lite) or `attn_q_b`(rmsnorm(`attn_q_a` h)); `split` with groups heads into qn | qr; `attn_kv_a_mqa` h
  split into c | kr, c through `attn_kv_a_norm`; interleaved RoPE on qr (heads) and kr (one head);
  `mla_attention` with K `attn_k_b`, V `attn_v_b` and scale 1/sqrt(qk_nope + qk_rope); `attn_output`; then
  `moe_ffn` (renorm when `expert_weights_norm`, else raw) times `expert_weights_scale` (an `affine` when it is
  not 1) plus the shared experts (`ffn_*_shexp`, ungated). From revision 1.6 (DeepSeek-V2 and V3 as published,
  section 12.8): the first `leading_dense_block_count` applications run a dense SwiGLU in place of the mixture,
  the two branches selected by `apply`; `expert_gating_func` 2 is `moe_ffn` `scoring` sigmoid, `exp_probs_b` its
  `selection_bias`, and `expert_group_count` / `expert_group_used_count`, when the file holds them (later
  llama.cpp releases write them; b6000 neither writes nor reads them, so it computes V3's router without its
  group limit), its `groups`; YaRN (`rope.scaling.type` yarn) is `freq_factors` by value, and the attention scale
  is mscale^2 / sqrt(qk_nope + qk_rope) with mscale = 1 + `rope.scaling.yarn_log_multiplier` ln(factor)
  (DeepSeek's softmax temperature, as llama.cpp computes it). Refused by name: files converted before llama.cpp's
  MLA keys, other gating functions and other RoPE scaling. transformers' DeepseekV2 class (4.57) leaves the
  temperature out of its scale, which the published model and llama.cpp apply; its DeepseekV3 class applies
  it.

## 10. Validation

### 10.1 What a consumer refuses

A consumer refuses a document, naming the rule, when any of these holds: at load, except for the
rules section 10.1.1 lists as checked when the rows arrive. Each rule names its section;
`tools/ir_validate.py` reports every violation as `IR_SPEC <section> (rule <n>): <message>`, the
rule a single fault breaks under exactly one number (a fault can still break a second rule as a
consequence: a wrong `block.repeat` names tensors the file lacks). Rule numbers are never reused or
moved: a rule added by a later revision takes the next free number (35 to 37 in 1.2, 38 and 39 in
1.3, 40 in 1.4, 41 and 42 in 1.5, 43 and 44 in 1.6) and is listed with its section's rules.

Document (section 3):
1. The text is not a JSON object, or it does not match `ir.schema.json` in a respect that no other
   rule names: the types of values, the closed key sets of the document, its sections and its slots,
   and the required sections. (The version, the model's fields, an op, an attribute, an attribute's
   value, an arity, a required attribute, a slot's kind and fields, a recurrence's declarations, a
   position bias, relative vectors, frequency factors, MLA's tensors, `apply` and a selection bias are rules 2
   to 13, 31 and 39 to 44.)
35. An object of the text holds a member name twice, at any depth, `_` members included (3.2).
36. A node reads a value that no node before it writes (in the order of 3.5: embed, the block as
   application 0, head; `tokens` and `pos` excepted); the embed section writes no `x`; the head
   writes no `logits`; under `depth_route`, a block node after it reads anything but `x`, `pos` and
   values written before it in the same application, or the head reads a block value other than `x`
   (3.4, 3.5, 7.4). Under `apply` (revision 1.6) every application is checked with the nodes it runs, a
   value written in an earlier application staying readable (3.8).
2. `ir_version` is not a version the consumer reads (2.3).
3. `model.d`, `model.vocab`, `model.ctx` or `model.n_layers` is not a positive integer, or
   `block.repeat` != `model.n_layers` (3.3).
4. A node's `op` is not an op of this specification, or a node carries an attribute its op does
   not define (3.5; `_` keys excepted).
5. An attribute value is outside its op's closed set: `gelu.approx`, `rope.pairing`,
   `rope.rope_heads`, `attention.mask`, `moe_ffn.gate_norm`, `moe_ffn.activation`, `moe_ffn.scoring`,
   `linear_attention.feature_map`, `depth_route.route`, `depth_route.gate`, `kv.evict`,
   `kv.kv_rows`, `kv.dtype`, `tensor_index` (sections 5 to 7).
6. A `per_layer` list outside the block, on a structural key, or of a length other than
   `block.repeat` (3.8).
7. A node's arity is wrong: one output except `split` and `rope`; `sub`, `add`, `mul`,
   `mul_col`, `append` and `pos_embedding` read two values; `embedding`, `relu`, `sigmoid`,
   `silu`, `gelu`, `affine`, `rms_row`, `scale`, `rmsnorm`, `layernorm`, `linear`, `moe_ffn`,
   `ssm_scan`, `conv1d_causal` and `depth_route` read one; `attention` and `linear_attention`
   read [q, k, v]; `mla_attention` reads [qn, qr, c, kr]; `rope` reads [q, k, pos] and writes two; `selective_scan` reads four; `list`,
   `depth_attend`, `expr` and `recurrence` read at least one, `expr` at most 23 (sections 5 to 7).
8. A required attribute is missing: `scale.weight`, `affine.scale` and `affine.shift`,
   `rms_row.eps`, `expr.expr` (a string), and the attributes each op's section lists.
9. An `expr` does not parse (5.19).
10. `split.sizes` has a different count than `split.out` (5.5); `rmsnorm.group` is not positive
    (5.6); `rope.n_rot` is not even and in 1 .. head_dim, or `rope.freq_scale` is not positive
    (5.14); `attention.heads` is not a multiple of `kv_heads`, or window or sinks is negative (5.15);
    `moe_ffn.top_k` is not in 1 .. n_experts (5.16); `conv1d_causal` has channels < 1 or kernel < 2
    (6.6); `split.groups` is not a positive integer (5.5); `rope.attn_factor` or `attention.softcap` is
    not positive (5.14, 5.15); `moe_ffn.groups` does not divide `n_experts`, keeps a `top` outside 1 .. n, or
    keeps fewer than `top_k` experts (5.16).
39. `attention.position_bias` is not `{weight, buckets}` (with `_` members allowed, and no other) with
    `buckets` a non-empty list of integers >= 0; or (container) its weight's tensor is not of ne {heads, B} with B greater than every bucket,
    in an application that reads it (5.15).
40. `attention.relative_kv` is not `{key, max_distance}` with an optional `value` (with `_` members
    allowed, and no other), `key` a name,
    `value` a name or null and `max_distance` an integer >= 0; or the node also carries
    `position_bias`; or (container) its key's or value's tensor is not of ne {head_dim or heads *
    head_dim, max_distance + 1} in an application that reads it (5.15).
41. `rope.freq_factors` is not a name or `{short, long, original_ctx}` with two names and an integer >= 1
    (revision 1.5), or a list of n_rot / 2 positive numbers (revision 1.6); or (container) one of its tensors is
    not of ne {n_rot / 2} in an application that reads it (5.14).
42. (container) An `mla_attention` node's `k_b` is not of ne {qk_nope, kv_rank, heads}, or its `v_b` not of
    ne {kv_rank, v_head_dim, heads}, in an application that reads it (revision 1.5; 5.20).
43. `apply` (revision 1.6) outside the block, with a per-layer entry that is not true or false, or in a block
    with `depth_route` (3.8).
44. (container) A `moe_ffn` node's `selection_bias` tensor is not of ne {n_experts} in an application that runs
    the node (revision 1.6; 5.16).

State (section 6):
11. A slot's kind is not kv or recurrent; a slot is declared twice or read by no node; a stateful
    node names no declared slot of its kind, or a stateful op lies outside the block (6.1).
12. A kv slot lacks heads, kv_heads, head_dim, max_len, window or sinks; an attention node's
    window, sinks, kv_heads or head_dim differs from its slot's (5.15, 6.2).
13. A recurrent slot's op is not a recurrent op, or differs from the op reading it; its dtype is not
    f32; its tensors are empty, have a non-positive shape or an init other than 0 (`"body"` too, for
    op `recurrence`), or do not fit the op (6.3 to 6.7, 6.10).
14. `attention.causal` is not true; `linear_attention.causal` is not true; `mla_attention.causal` is not true
    (5.15, 6.4, 5.20).
15. `mask: turn_state` with a window or sinks (in any application), with a windowed kv slot, or
    without an embedding `state_weight` (6.8.2).
16. `evict: turn_state` with an attention node reading the slot whose mask is not turn_state
    (6.8.3).
17. `state_weight` outside the embed section, on more than one embedding, or with `{i}` in its name
    (6.8.1, Q8).
17a. A name holds `{i}` outside the block, or more than once (3.7); a node reads `tokens` or `pos`
    other than as section 3.5 allows, or writes either (3.5); a kv slot is read by more than one
    attention node (6.1); a node reads a `depth_route`'s `out` (7.4). Under `depth_route` a block node
    after it that reads `tokens` or the router's `out` breaks rule 17a; rule 36 follows as a consequence
    (neither is in the step's scope, 7.4(b)), so a consumer may report either and a refusal case cites
    17a (`mor_reads_tokens`, `mor_reads_route_out`). Rule 36 alone covers a step's read of any other
    value from outside the step (`mor_reads_outside_value`).

Recursion (section 7):
18. `weight_index` is not `{per_layer: [block.repeat integers >= 0]}`, or leaves a stored layer
    0 .. max unused (7.1).
19. `tensor_index` outside a block, or in a block without `weight_index` (7.2; its value is rule 5).
20. `depth_route` is not the block's first node, occurs more than once or outside the block; its
    `n_rec` is not an integer >= 1; its `step` is not a `per_layer` list of integers forming the
    consecutive, ascending, equal-length steps 0 .. R-1 in one run, with -1 only before and after
    them (the list's length is rule 6); its weight holds `{`; it has no `out` (7.4, 7.7; its `route`
    and `gate` values are rule 5).
21. `depth_route` with a kv slot whose `kv_rows` is not `"active"`, or with no kv slot (7.5).
22. A kv slot carries `kv_from` (7.6).
23. `depth_route` together with `mask: turn_state` (9.2, llama_state), or with `list`, `append` or
    `depth_attend` nodes (7.4).

Container (a GGUF with `paperkiln.ir`, sections 3.1, 6.8, 7, 8.2):
24. `paperkiln.ir_version` is missing, not a uint32, or differs from the `ir_version` of the
    document embedded in the file (a document read in place of the embedded one is governed by its
    own `ir_version`, rule 2).
25. A tensor the document names is not a tensor of the file (a `linear` weight passes through its
    `tied_to`). A missing pseudo-query that only S = 1 `depth_attend` nodes name is not refused:
    the validator reports it as a warning tagged "exporter rule" (exporters write it, 5.18).
26. The `depth_route` router's ne is not {d, n_rec}.
27. Under `weight_index`: the architecture is not `paperkiln`, or `<arch>.block_count` is missing or
    differs from the number of stored layers.
28. State tokens: `paperkiln.state_tokens.k` without an embedding `state_weight`, or the reverse;
    `state_weight`'s ne is not {d, k}; the architecture is not `paperkiln`; `tokenizer.specials`
    lacks eot, user or assistant, or names ids whose tokens are not the template's markers; a
    declared `paperkiln.chat_template` other than `paperkiln/1`.
29. A sidecar `<stem>.ir.json` that differs from the embedded document (for a consumer that reads
    sidecars).
30. `paperkiln.ir_sha256` is present but not a string equal to the digest of the embedded document,
    or the embedded document has no digest (3.10).
38. `general.architecture` is absent, or neither `"paperkiln"` nor one of `"llama"`, `"mistral"`,
    `"qwen2"`, `"qwen3"`, `"gpt2"`, `"gptneox"`, `"gemma3"` and `"mamba"`; or it is one of those eight and
    the embedded document is not the graph Appendix D builds from the file's metadata and tensor names,
    compared as Appendix D.7 states; or the metadata builds no graph (3.1, Appendix D). Checked on the
    embedded document always, also when another is read in place of it, and never on the document read
    in place.

Recurrence (section 6.10; `tools/ir_recurrence.py` reports these as "IR_SPEC 6.10 (rule N)"):
31. `index` is not a map of names to positive integers; `inputs` does not declare each value of `in`
    (at least one); `weights` is not a map of declarations to tensor names; `update` is missing or
    empty, `readout` is not one assignment; a declaration or a body does not parse (naming the body
    and the column), uses `rowsum`, `rowmean`, `rowmax` or an unknown function; a name is not an
    identifier, is reserved, or is declared twice.
32. A body reads a name its section cannot read (6.10.2); a read has the wrong number of indices; an
    index is not declared or not in scope; a bare name is not an index in scope; `sum` runs over an
    index already in scope; an assignment repeats an index (a declaration may, 6.10.1).
33. The slot of a recurrence is not of op `recurrence`, is read by another node, or its tensors are
    not the update's (names, shapes, and `init` `"body"` exactly for the tensors `init` assigns); an
    `init` assigns a name that is not a state tensor, assigns one twice, or with other sizes than its
    update.
34. A weight whose tensor's element count is not its declaration's (container), or an input whose
    width is not its declaration's (at load where the width is known, at the latest when the rows
    arrive).

Widths (sections 5 and 10.1.1):
37. A value's width does not fit the node that reads it: a `scale` weight or bias whose length is not
    the input's width (5.4); `split` sizes (times `groups`) that do not sum to it (5.5); an
    `mla_attention` input of another width than its attributes give (5.20); an `rmsnorm` group that does
    not divide it (5.6); `add`, `mul` or `sub` operands of two shapes (5.10); a `mul_col` column that
    is not [T, 1] (5.12); a `rope` input whose width is not its heads times head_dim (5.14);
    `depth_attend` sources of two shapes (5.18); `expr` inputs of two shapes (5.19).

### 10.1.1 At load and when the rows arrive

Every rule is checked at load, from the document and the file, except those that need a value's
width: rule 37 and rule 34's input widths. The IR does not declare widths (a value's width follows
from the tensors and the ops that make it), so this specification defines no width inference: an
engine that infers widths may make those checks at load, and every engine makes them at the latest
when the rows arrive, refusing the call (an adapter's `error`) and naming the rule, never computing a
value from a misfit. The validator infers no widths and makes neither check; the reference
interpreter makes both when it runs; the suite's refusal cases test only the load-time rules.

### 10.2 The schema and the validator

`ir.schema.json` is the structural half (rules 1, 4 to 8 in part). `tools/ir_validate.py`
runs it and every other rule:

    python tools/ir_validate.py doc.ir.json [model.gguf ...]    # documents and GGUFs
    python tools/ir_validate.py --corpus [--corpus-glob PAT]    # stored documents (below)
    python tools/ir_validate.py digest doc.ir.json|model.gguf   # the digest (section 3.10)

It is standard library only (it checks recurrence nodes with `tools/ir_recurrence.py`'s parser, which
is too). It evaluates the schema with its own evaluator for the keywords the
schema uses, and with `jsonschema` too when that package is installed (`tests/test_ir_validate.py`
checks they agree). `tools/ir_check.py` calls it for the semantic rules and keeps the container
checks. `--corpus` checks this repository's documents (`examples/`, the suite's models and rewritten
documents) and every file an engine's `--corpus-glob` patterns name (documents, and GGUFs with an
embedded document).

### 10.3 Engine capability refusals

An engine may refuse a valid document it cannot run: an op without a kernel on a backend, or an
attribute combination it does not implement. Such a refusal names the op or attribute and the
engine; it is not a validation failure, and the validator does not report it. Each engine's
capabilities are listed in Appendix C, outside the normative text.

## 11. Conformance

### 11.1 Layout

`conformance/` holds the suite; `conformance/manifest.json` lists every case with its
kind, profile, class and tolerance:

- `ops/`: per-op fixtures: inputs, attributes, expected outputs (f32 bit patterns), class and
  tolerance.
- `state/`: state traces: token-by-token decode against prefill; eviction on and off; rewind into an
  evicted turn and into a sliding window; MoR depth traces; recurrent state snapshots.
- `models/`: whole models: a small document with F32 weights in a GGUF, prompts, expected greedy ids
  and log-probabilities (and depth traces under depth_route).
- `refuse/`: documents (and GGUF metadata) every consumer must refuse, each with the rule it breaks
  (rule 38's cases are GGUFs whose architecture is not their document's graph).
- `accept/documents.json`: the documents every consumer must accept (every suite model, every
  rewritten document in place of its model's own, every worked document of `examples/`), so a
  validator that refuses everything fails the suite.
- `chat/render.json`: chat template records and conversations with their rendered text or refusal
  (sections 8.1.3 and 8.1.6); `chat/parse_reply.json`: generated replies split into content and tool
  calls (section 8.1.6).
- `digest/vectors.json`: digest vectors (section 3.10).
- `quant/blocks.json`: stored blocks of every storage type with their f32 values (section 4.7).
- `grad/recurrence.json`: gradient cases of recurrence nodes (section 6.10.5); `grad/attention.json`
  (revision 1.4): gradient cases of attention with `relative_kv` (section 5.15).
- `recurrence/`: suite models' documents with their hand-written recurrent ops rewritten as recurrences
  (section 6.10.6), which the traces `state/<model>_recurrence.json` run in place of each file's own
  document, against the original document's expected values.

Expected values come from the numpy reference (`tools/ir_numpy.py`); paperkiln's `mtstudio` is
checked against them wherever it has a path (section 11.4).

### 11.2 Fixture formats

All files are JSON (weights in GGUF). A tensor is `{"shape": [r, c], "f32": "<hex>"}`: the values in
row-major order, each as 8 hex digits of its little-endian IEEE binary32 bit pattern, big-endian
digit order (1.0 is `3f800000`). Integer arrays are plain JSON lists.

- **op case** (`ops/<op>.json`): `{"op", "cases": [{"name", "node", "inputs": {name: tensor},
  "weights": {name: tensor}, "expect": {name: tensor}, "expect_state"?: {slot: {tensor: tensor}},
  "class", "tol"}]}`. `node` is an IR node; the engine runs it alone as application 0 of a one-node
  block, recurrent slots starting at their initial value, and `expect_state` gives a recurrent
  slot's tensors after the last row. A `list` value is a JSON list of tensors; `tokens` and `pos`
  are integer lists. What a one-node block lacks takes these values, which the fixtures' expected
  values assume: an `attention` node reads a kv slot made from its own `heads`, `kv_heads`,
  `head_dim`, `window` and `sinks` (an `mla_attention` node: its `heads`, kv_heads 1, head_dim kv_rank +
  qk_rope, `window` and `sinks`), empty at the start, with room for the rows, and its rows sit at
  positions 0 .. T-1 (the mask of section 5.15 at those positions; mask `"causal"`, as no fixture
  has ids); a `rope` node whose `freq_factors` has two sets takes the sequence's length as its largest
  position plus one (section 5.14); a recurrent op reads a slot of its own op whose tensors are those sections 6.4 to 6.7 and
  6.10 give, at their initial values; an `embedding` with `state_weight` takes vocab as the row count
  of its `weight` (ids vocab .. vocab + k - 1 read the state rows); a node with `{i}` in a tensor name
  reads it with i = 0.
- **state trace** (`state/<name>.json`): `{"model": "models/<name>", "evict": null | false,
  "steps": [{"method": "forward" | "truncate" | "snapshot" | "restore", ...its params}],
  "expect": [one object per step], "score_form"?: {...}}`. Steps are the adapter methods of
  section 11.3 on one session; each expectation holds `len` and the reports the step checks
  (`logits`, `kv_rows` and `kv_positions` as lists of allowed values per application, `depth`,
  `recurrent`), at the trace's class and tolerance. `score_form` restates a forward-only trace as
  one scoring request (prompt, continuation, log-probabilities, argmax, depths, allowed final kv
  rows) for an engine that scores rather than reporting logits (mtstudio).
- **model case** (`models/<name>/case.json` beside `models/<name>/<name>.gguf`): `{"model",
  "prompts": [{"ids", "continuation", "logprobs", "argmax", "depth"?}], "banned", "class", "tol"}`:
  the continuation is the reference's greedy one (ids in `banned` never chosen), `logprobs` its
  per-token log-probabilities and `argmax` the reference's argmax at each continuation position.
  A paperkiln-trained model's directory also holds `<name>.safetensors` and `mtstudio.json` (its
  run's spec) so mtstudio can score it.
- **trace with a document** (`state/<model>_recurrence.json`): a state trace with `"ir": "<path in the
  suite>"`, the document the engine runs in place of the file's own (`open`'s `ir`).
- **digest vectors** (`digest/vectors.json`): `{"vectors": [{"name", "text", "prose"?, "canonical",
  "sha256"} | {"name", "text", "refused"}]}`. `text` is a JSON text; with `prose` true the `_` members
  are kept (a test of the serialiser alone), otherwise `sha256` is the digest. A `refused` text has no
  digest.
- **dequantisation** (`quant/blocks.json`): `{"cases": [{"name", "type", "ggml_type", "rows", "n",
  "bytes", "expect", "class"}]}`: `bytes` (hex) holds rows x n elements of the type, row-major; `expect`
  is their f32 values [rows, n]; the class is EXACT.
- **gradient** (`grad/recurrence.json`, `grad/attention.json`): `{"cases": [{"name", "node", "inputs",
  "weights", "out_grad", "expect", "tol"}]}`: `expect` maps each value of `in` and each weight tensor to
  d sum(out * out_grad) / d(it), central differences in float64 of the reference's recurrence
  (`tools/ir_recurrence.py`) or of section 5.15's formula (`tools/ir_conformance_gen.py`, `attention64`).
- **refusal** (`refuse/<name>.json`): `{"doc"?: <document>, "text"?: <JSON text>, "gguf": "<path in
  the suite>", "rule": <10.1 rule number>, "section": "<section>"}`. With `doc` or `text` (the
  document as a JSON text, written byte for byte, for a fault a parsed value cannot hold: a duplicate
  member name), the engine loads the GGUF (a suite model) with that document in place of its own (as
  ember's `EMBER_IR` does); without either, the GGUF (`refuse/<name>.gguf`) breaks a container rule.
  The version that governs is that of the document actually read, the one in place of the file's own;
  the file's `paperkiln.ir_version` must still match the document embedded in it (rule 24). `rule` is
  the rule the fault breaks (a consequence it also breaks is not cited). A consumer passes when it
  refuses; the reference adapter must also report a violation of `rule` under `section`
  (`IR_SPEC <section> (rule <rule>)`).
- **acceptance** (`accept/documents.json`): `{"cases": [{"name", "gguf"?, "ir"?}]}`, paths in the
  suite (`../examples/...` for the worked documents). The engine checks the GGUF with its own document,
  or with `ir` in its place, or `ir` alone when there is no `gguf` (the document rules only); it passes
  when it accepts.
- **chat** (`chat/render.json`): `{"records": {id: record}, "cases": [{"name", "record": id or a record,
  "messages", "generation_prompt", "thinking"?, "tools"?, "expect": str} | {..., "refused": true}]}`:
  the rendering of sections 8.1.3 and 8.1.6, or a refusal; `tools`, when present, is the conversation's
  list of tool definitions. The conversations and records are paperkiln's
  `tests/chat_template_cases.json` (gate G5); the expected texts come from `tools/ir_chat.py`.
- **reply parsing** (`chat/parse_reply.json`, kind `chat`): `{"cases": [{"name", "tokens": {id: text},
  "ids", "call_id", "expect": {"content", "tool_calls"} | null}]}`: the `parse_reply` of section 8.1.6
  over a toy vocabulary (`tokens` gives each id's decoded text; `call_id` is the call token's id);
  `expect` null where a part does not parse. From the same paperkiln file.

### 11.3 The adapter contract

`tools/ir_conformance.py` drives an engine through an adapter. The runner ships the `ir_numpy`
adapter (the reference). Any engine joins with a program that speaks this protocol; the runner
starts it once with `--adapter cmd:"<program and arguments>"`. An in-process Python adapter (a
subclass of the runner's `Adapter`) joins through `ir_conformance.main(argv, adapters={name:
factory})`; paperkiln's `mtstudio` adapter is one (paperkiln's `tools/ir_conformance_mtstudio.py`,
through `mtstudio score`, `expr_eval` and `ir_eval`).

**Transport.** JSON Lines: one request per line on the program's stdin, one reply per line on its
stdout, in request order; stderr is free for logs. A request is `{"id": int, "method": str,
"params": {...}}`. A reply is one of

    {"id": n, "result": {...}}     the method ran
    {"id": n, "error": "message"}  the method failed (the case fails)
    {"id": n, "skip": "reason"}    not implemented (the case is reported as skipped, never passed)

**Values.** A tensor is `{"shape": [...], "f32": "<hex>"}` of any rank (a recurrent `s` [n] is
1-D; section 11.2: 8 hex digits per value, the binary32 bit pattern, row-major). Ids and positions
are JSON integer lists.

**Paths** (`gguf`, `ir`) are absolute paths of the runner's own platform, passed as the runner has
them; the engine opens them as given. The runner and the engine therefore run on one platform: a
Linux engine is driven by a runner on Linux (WSL included), a Windows engine by one on Windows.

**Errors.** An `error` reply leaves the session as it was before the call (its ids, state and
snapshots), as a `skip` does; the next call sees the pre-call session.

**Skips.** A `skip` covers the whole call: there is no per-report skip. The engine decides it before
running anything, so a skipped call leaves the session as it was.

An engine's own refusal of a case it cannot build is a `skip`, not an `error`: the engine declines
the model before computing any of the case's values and says what it does not run (an architecture
it has no graph or builder for, a kernel's shape precondition, a capability refusal of section 10.3).
The adapter may find the refusal when it loads the model, by a throwaway run on a session of its own
before the case's first call, and then skips every call of the case; its reason quotes the
engine's message. An exception from a path the engine offers for the case (a builder that lists the
architecture and stops) is an `error`, so the case fails; so is a refusal after some of the case's
values have been computed. (ONNX Runtime GenAI's CPU kernel refusing head sizes its own builder
emits is a skip; its builder stopping on an architecture it lists is a failure:
`adapters/ortgenai/FINDINGS.txt`, findings 1 and 2.)

**Sessions.** A session is one sequence on one loaded model. Its length `len` is the number of
tokens seen; its state is whatever the engine keeps (kv rows, recurrent tensors, MoR row sets). A
session owns the ids it has seen, so it can re-prefill after a rewind (section 6.9).

**Methods.**

| Method | params | result |
|---|---|---|
| `hello` | `{}` | `{"engine", "version", "backend", "exact": bool, "profiles", "methods"}` |
| `check` | `{"gguf": path or null, "ir": path or null}` | `{"accepted": true}` or `{"accepted": false, "message": str}`: load only; a refusal is a result, not an error. `gguf` null checks the document `ir` alone (the document rules, no container rules) |
| `open` | `{"gguf": path, "ir": path or null, "evict": true / false / null}` | `{"session": str, "vocab": int, "applications": int, "state_k": int}` |
| `forward` | `{"session", "ids": [...], "report": [...]}` | `{"len": int, ...the reports}` |
| `truncate` | `{"session", "n": int, "report": [...]}` | `{"len": n, ...the reports}` |
| `snapshot` | `{"session", "tag": str}` | `{"len": int}` |
| `restore` | `{"session", "tag": str, "report": [...]}` | `{"len": int, ...the reports}` |
| `close` | `{"session"}` | `{}` |
| `run_node` | `{"node", "inputs", "weights", "state"}` | `{"outputs", "state"}` |
| `run_node_grad` | `{"node", "inputs", "weights", "out_grad"}` | `{"grads": {name: tensor}}` |
| `digest` | `{"text": str, "prose": bool}` | `{"sha256", "canonical"?}` or `{"refused": str}` |
| `dequantize` | `{"type": str, "n": int, "bytes": hex}` | `{"values": tensor [n]}` |
| `render_chat` | `{"record": object, "messages": [...], "generation_prompt": bool, "thinking": bool or null, "tools"?: [...]}` | `{"text": str}` or `{"refused": str}` |
| `parse_reply` | `{"tokens": {id: str}, "ids": [...], "call_id": int}` | `{"parsed": true, "content": str, "tool_calls": [...]}` or `{"parsed": false}` |

- `ir` (optional) is a document that replaces the file's embedded one (ember's `EMBER_IR`).
- `evict`: null runs the document as declared; false keeps every kv row even where the document
  declares `evict: turn_state` (the comparison run of section 6.8.3); true requires the declaration,
  and is an `error` on a document without it (a request the session cannot honour).
- A snapshot lives until its session is closed and can be restored any number of times. A
  `snapshot` under a tag already in use replaces the earlier one; `restore` of a tag the session has
  not snapshotted is an `error`.
- `forward` appends `ids` at positions len .. len + n - 1: on a fresh session it is a prefill, later
  an incremental step (n = 1 is token-by-token decode); any chunking is allowed. `ids` is not empty
  (an empty list is an `error`). `truncate` applies section 6.9 to the first n tokens, 0 <= n <= len:
  n = len changes nothing, and n > len (past the end) or n < 0 is an `error`. `restore` returns the
  session to the ids and state of a `snapshot`.
- `report` lists what to return after the call. `forward` may ask for any of them; `truncate` and
  `restore` only for `kv_rows`, `kv_positions` and `recurrent` (they run no rows, so `logits` and
  `depth` there are a runner error, and the suite never asks). An engine that cannot give a report
  replies `skip` to the call:

| Report | Shape | Meaning |
|---|---|---|
| `logits` | tensor [n, vocab] | the logits of the rows this call ran (`forward` only) |
| `kv_rows` | `{slot: [rows per application]}` | k/v rows each application's instance of each kv slot holds after the call (after any eviction the call allows); `{}` for a document without kv slots |
| `kv_positions` | `{slot: [[positions] per application]}` | the absolute positions of those rows, ascending |
| `depth` | `[int per id of this call]` | under `depth_route`: each new token's depth, 1 .. R; `[]` for a document without `depth_route` |
| `recurrent` | `{slot: [{tensor: tensor} per application]}` | every recurrent slot's tensors after the call; `{}` for a document without recurrent slots |

- `run_node` (op fixtures, optional): `node` is one IR node; `inputs` maps value names to tensors (a
  `list` value is a JSON list of tensors; `tokens` and `pos` are integer lists); `weights` maps
  tensor names to tensors; `state` is null or `{slot: {"decl": <slot>, "tensors": {name: tensor}}}`,
  the slot's declaration and its tensors before the call (the suite passes null: every slot at its
  initial value, with section 11.2's defaults for what the one-node block lacks). The node runs as
  application 0 of a one-node block. The result's `outputs` maps every name in the node's `out` to a
  tensor (a list value to a JSON list of tensors), and its `state` is `{slot: {tensor: tensor}}`: each
  recurrent slot's tensors after the last row, each in its declared shape (`expect_state`'s form; a kv
  slot is not reported, and a node without a recurrent slot gives `{}`). In `inputs`, `tokens` and
  `pos` are integer lists and every other JSON list is a list value; `[]` is a list value with no
  tensors, which no document makes (`list` reads at least one value), so the suite never passes one
  and an engine may reply `error` to it.
- `run_node_grad` (gradient cases, optional; for training engines): the node runs as in `run_node`
  from the initial state, and the result maps each value of `in` and each weight tensor's name to the
  gradient of sum(out * out_grad).
- `digest` (optional): the digest of the JSON text (section 3.10), and its canonical form when the
  engine has it; `refused` when the text has no digest. `dequantize` (optional): the f32 values of n
  elements of the storage type stored as the bytes (section 4.7).
- `render_chat` (optional, the chat profile): the text a conversation renders to under a template
  record (section 8.1.3), with the generation prompt when `generation_prompt` is true and, when
  `thinking` is false, the record's `generation_prompt_nothink`, and with `tools` (sent only when the
  case has them) as the conversation's tool definitions; `refused` where section 8.1 refuses (an
  unimplemented field or value, a conversation of the wrong structure, a reserved role or text, a
  definition that is not portable).
- `parse_reply` (optional, the chat profile, v1.4.2): the split of section 8.1.6 of the reply `ids`, each
  part decoded by concatenating its ids' texts from `tokens` (keys are the ids as decimal strings);
  `parsed` false where a part does not parse (the engine would then return the whole reply as content).
  `tool_calls` is a list of `{"id", "type": "function", "function": {"name", "arguments"}}`.

**Expectations the suite checks.**
- Logits within the case's class and tolerance; greedy ids and depth traces identical. An EXACT case
  is checked bit for bit when `hello` reports `"exact": true` (a conforming interpreter path), and at
  the backend's ROUNDING tolerance otherwise (section 4.2).
- `kv_rows` with eviction on (section 6.8.3 gives the legal counts): after a call ending at length
  L, with m the number of complete state blocks in 0 .. L-1 and q the last position of block m - 1,
  the eager count is #{p < L : c(p) = 1} + #{p < L : c(p) = 0 and s(p) = m}. When no text position
  follows q in 0 .. L-1, the count with segment m - 1's text rows still held is legal too; once a
  later text position is present, only the eager count is. The suite stores the allowed values as a
  list, and `kv_positions` likewise as a list of allowed position sets. With eviction off, or
  without state tokens, it holds L rows (min(L, sinks + window) is not
  required: an engine may keep more rows of a windowed slot than it reads, so `kv_rows` of a windowed
  slot is not checked).
- `kv_rows` under `kv_rows: active`: application i holds #{t < L : s_i = -1 or n_t > s_i} rows, the
  positions of those tokens.
- `recurrent`: within ROUNDING of the reference after prefill; equal bit for bit between a prefill
  and the token-by-token decode of the same ids in the same engine where section 4.5's condition
  holds, otherwise within ROUNDING.
- Digest vectors: the sha256 equal (and the canonical form, when reported); a refused text refused.
- Acceptance: `check` accepts. Chat: the rendered text equal, or a refusal where the case refuses;
  reply parsing: `content` and `tool_calls` equal, or `parsed` false where `expect` is null.
- Dequantisation: bit for bit on every backend (section 4.7).
- Gradients: per tensor |a - b| <= tol * max(1, max|ref|), tol the case's (1e-4).

**Environment.** The runner starts an engine's program with every `EMBER_*`, `MICROTORCH_*` and
`MTSTUDIO_*` variable removed except those named with `--keep-env NAME` (an engine setting such as
`EMBER_IR_CTX` changes what it runs, so a clean run carries none unless it is stated).

**Expected failures.** `expected_fail.json` maps an engine (the `engine` of `hello`, by prefix) to
the case ids it is known to fail, each with the reason and the fix it waits for. Such a case is
reported XFAIL, not hidden as a skip; if it passes it is reported XPASS, and the entry should go.

The in-process Python adapters implement the same methods with the same parameters and results
(tensors as float32 arrays instead of hex). The numpy reference has no cache: its adapter recomputes
the whole prefix and reports the row counts and positions the rules above define.

### 11.4 Running

    python tools/ir_conformance.py --adapter ir_numpy
    MTSTUDIO_BIN=build/mtstudio python tools/ir_conformance_mtstudio.py   # in paperkiln
    python tools/ir_conformance.py --adapter cmd:"path/to/engine --conformance"
    python tools/ir_conformance.py --regenerate     # rebuild expected values from the reference

A case passes when every value is bit-identical (class EXACT) or within its tolerance (ROUNDING:
|a - b| <= tol * max(1, max|ref|)), greedy ids and depth traces are identical, and kv rows held and
their positions are among the allowed values. The runner prints one line per case and a summary,
and exits 1 on any failure.

## 12. Questions: settled and open

### 12.1 Settled in revision 1.0

Each question below was open in the draft. ember's review proposed an answer; the answer adopted is
stated, with the reason where it differs from ember's.

- **Q1. Head ops row-wise.** Settled as a consequence: every op allowed outside the block is
  row-wise (stateful ops and depth_route are refused there), so the head can run on the last row
  only. ember's narrower head list is a capability refusal (Appendix C).
- **Q2. Informative attributes without `_`.** `attention.softmax` and `moe_ffn.routing` stay
  informative attributes (sections 5.15, 5.16). Renaming them would remove an attribute, a major
  change (section 2.2).
- **Q3. The RoPE angle table.** The f32 recipe both engines compute is normative (section 5.14); the
  table is ROUNDING across libms. The numpy reference keeps its double-precision table, within
  ROUNDING of the recipe, because the stored third-party gate receipts were measured against it.
- **Q4. Contraction.** Normative: every translation unit that computes an op whose text says
  "rounded" or "no fused multiply-add", EXACT or not, is built so that a * b + c is not contracted
  (section 4.1). ember sets `-ffp-contract=off` on `graph_ir.cpp` and `utils.cpp` too (85e269e), and
  paperkiln on every translation unit of its own (Q15, section 12.2).
- **Q5. Per-layer scope.** Settled (section 3.8): every attribute may be per-layer except the
  structural keys. ember proposed an allow-list of the numeric attributes it implements (attention
  window and sinks, rope theta and freq_scale, affine scale and shift, depth_route step, slot window
  and sinks). Not adopted as the specification: version 0's text allowed any attribute, and a
  version 1 document must not refuse what a version 0 document could say (section 2.1). ember's
  list is its capability (Appendix C).
- **Q6. Non-causal attention.** `causal` is a constant true (rule 14).
- **Q7. Attention and slot consistency.** Closed by rule 12, checked at load; ember moves its CPU
  shape check to load and compares window and sinks for unwindowed slots too.
- **Q8. state_weight placement.** Closed by rule 17, which matches ember.
- **Q9. `{i}` outside the block.** Refused (section 3.7, rule 17a), as ember does.
- **Q10. Tokenizer descriptor.** `tokenizer` is informative except `specials`; under state tokens the
  specials must name the template's marker ids (section 8.2). ember reads the template's ids, which
  then agree.
- **Q12. Rewind into an evicted segment.** The normative result is a fresh run's state over the
  first n tokens (section 6.9). A lower layer may refuse a forward into a dropped segment; the
  engine's session, and its adapter's `truncate` and `forward`, must succeed by re-prefilling.
- **Q13. The latest time to evict.** The bound is closed (section 6.8.3): eager or lazy; a longer
  hold is eviction off for the whole sequence.
- **Defaults** (ember's C1). Required attributes have no defaults; the optional attributes and their
  defaults are listed in section 3.5. ember refuses their absence (e358153).
- **One attention node per kv slot** (C2), section 6.1. Several kv slots per block stay defined;
  ember serves one (Appendix C).
- **max_len** (C3), section 6.2: the slot's capacity; positions >= max_len are refused.
- **`tokens` and `pos`** (C4), section 3.5.
- **depth_route's `out`** (C5) and **depth_route with list values** (C6), section 7.4.
- **Ids out of range** (C7), section 6.8.1.
- **Partial state blocks** (C8). Serving engines may refuse partial or out-of-order blocks in a
  forward (section 6.8.2); the suite's traces hold only complete blocks, though a block may be split
  across two calls (a block left open at the end of a call).
- **expr limits and locale** (C11), section 5.19.
- **NaN and signed zero** (C12), sections 4.2, 5.9, 5.19 and 6.5.
- **More than one `{i}`** (C13): refused (section 3.7).
- **kv dtype** (C14): only `"f32"` (the schema's enum).

### 12.2 Settled in revision 1.1

- **Q11. Digest.** Adopted as proposed (section 3.10): RFC 8785 serialisation of the document without
  its `_` members, then SHA-256. One concrete problem was found and is handled by restriction, not by
  departing from RFC 8785: the scheme writes -0 as `0`, while a negative zero is significant in an EXACT
  op, so a document that holds one has no digest (no stored document does). Numbers are their binary64
  values, so documents that spell a number differently are the same document, and documents whose
  doubles differ but round to one f32 attribute are not. Member names sort by UTF-16 code units and
  strings are not normalised (NFC), as names are matched byte for byte anyway. The digest is
  implemented in `tools/ir_validate.py` (the reference, with a `digest` subcommand) and in paperkiln's
  C++ (`ir::digest`), whose exporters write it as `paperkiln.ir_sha256`; container rule 30 refuses a
  key that does not name the embedded document. Vectors: `conformance/digest/`.
- **Q14. Quantised storage** (ember's C10). Settled by section 4.7: ggml's scalar `dequantize_row_*`
  (llama.cpp b6000) for F16, BF16, Q8_0, Q4_0, Q5_0, Q5_1, Q4_K, Q5_K and Q6_K, the types paperkiln's
  exports and the gated third-party files hold. Every dequantised value is EXACT on every backend:
  every product is exact in f32 and at most one rounding follows. `tools/ir_quant.py` is the
  reference (bit for bit with gguf-py 0.19.0 on random blocks of every type); fixtures:
  `conformance/quant/`. EXACT then applies to ops on those values; an engine that multiplies
  quantised blocks by quantised activations is ROUNDING by construction.
- **Q15. Contraction in paperkiln** (from Q4). Decided yes (Jonathan delegated the decision to the main
  session, 25 October 2026): paperkiln's own translation units are compiled with `-ffp-contract=off`
  (GCC and Clang, `CMakeLists.txt`, every target paperkiln defines: the library, mtstudio, the tools,
  tests and the Python extension); MSVC's default `/fp:precise` does not contract and paperkiln passes
  neither `/fp:contract` nor `/fp:fast`. The vendored transformer_core keeps its own flags
  (`-ffast-math`, Appendix A), and nvcc's device code its `--fmad=true`: both are ROUNDING already.
  What the flag guarantees: no expression in paperkiln's code is contracted into a fused multiply-add,
  whatever `-march` a build adds (arm64, where GCC and Clang contract by default; `-march=native`; the
  wheels' x86-64-v3), so every statement this document makes about paperkiln's EXACT paths holds on
  every build, not only on plain x86-64. On x86-64 without `-march` (every paperkiln build except the
  wheels' core) there is no FMA instruction to contract into, so the flag changes no instruction
  there. The gate (25 October 2026; paperkiln 86a978ca, GCC 13.3, x86-64 Release on WSL ext4): built with and
without the flag from the same tree, all 75 object files of paperkiln's own targets, the 56 linked
binaries (mtstudio, the tools and every test) and libmicrotorch.a are byte-identical, so the CPU
Release build trains, tests and exports exactly as before (the precision gate's off path included),
and transformer_core's compile flags carry no `-ffp-contract`. Builds that add `-march` with FMA, and
arm64 builds, now keep the stated orders that they would otherwise have contracted.

### 12.3 Settled in revision 1.2

The findings of the clean-room implementation (`cleanroom/FINDINGS.md`, numbered there), each with the
answer adopted. Where the reference already behaved one way, that behaviour is now the text.

- **F1. Ties in number serialisation.** ECMA-262 section 6.1.6.1.20's closest-then-even digits
  (section 3.10 step 3); vector `rfc8785_number_43143ff3c1cb0959` kept.
- **F2. Duplicate member names.** Refused at load, at any depth, `_` members included (section 3.2,
  rule 35); refusals `duplicate_member_top`, `_node`, `_prose` and `_embedded` (a GGUF).
- **F3. The GGUF container.** Specified (section 3.1): version 3, `general.alignment` or 32, offsets
  from the data section's start, strings as byte strings that need not be UTF-8.
- **F4, F5. Single-node runs.** The defaults the fixtures assume (section 11.2) and the shape of
  `run_node`'s `state` (section 11.3).
- **F6. Documents for third-party GGUFs.** A capability an engine may refuse (section 3.1);
  `tools/gguf_ir.py` the informative reference builder (section 9.3).
- **F7. Checks that need widths.** Rule 37 and rule 34's input widths are checked when the rows
  arrive at the latest (section 10.1.1); the IR defines no width inference.
- **F8. Reads of unwritten values.** Refused (rule 36), with persistence across nodes, applications
  and sections stated (section 3.5).
- **F9. Mixture-of-Recursions' active rows.** A step sees `x` and `pos` alone and only `x` leaves it,
  the reference's rule; reads of other values are refused (section 7.4(b), rule 36).
- **F10. depth_attend's eps.** The attribute governs; it is optional with default 1e-5 (section 5.18).
- **F11. The lexical rules and depth of `expr`.** In section 5.19, from `tools/ir_expr.py`.
- **F12. Rule numbers.** Each refusal case cites the rule its fault breaks; rule 1 no longer covers
  what rules 2 to 13 and 31 name, nor rules 19 and 20 what rule 5 names; the reference reports every
  refusal with its rule and the runner checks it (section 10.1).
- **F13. Must-accept cases.** `accept/documents.json` (section 11.2).
- **F14. The I-JSON integer range.** By lexical form (section 3.10 step 1).
- **F15. truncate with recurrent state.** Defined; a snapshot or a run from the start (section 6.9).
- **F16. depth_route steps with a gap.** Refused, as the reference did (section 7.4, rule 20).
- **F17. rope_heads "first".** Head 0 of q and of k; the reference implements it (section 5.14).
- **F18. A read with an index of another size.** Zero padding, as the reference and paperkiln do
  (section 6.10.1).
- **F19. A row reduction of a constant.** Its w additions (section 5.19).
- **F20. Attributes to f32.** The binary64 value rounded once (section 3.5).
- **F21. The chat profile's conformance.** The adapter method `render_chat` and 32 chat cases.
- **F22. The adapter's edge cases.** Paths, the post-error state and the edge cases (section 11.3).
- **F23. erf in other languages.** An informative note (section 5.8).

### 12.4 Settled in revision 1.3

llama.cpp b6000 run against the suite (`adapters/llamacpp/`, FINDINGS.txt points K1 to K5), and the
extractor's proposal of T5's bias (paperkiln, 26 October 2026). Each with the answer adopted.

- **K1. The architecture of an IR-only file** (finding 7). Files whose model only their document
  states carried a llama.cpp architecture (kimi, recurrence, gpt2_swa and plain flex as gpt2), which
  llama.cpp loads and computes as plain gpt2 without a warning, as does transformers through any GGUF
  to HF conversion (H4 of `adapters/hf/FINDINGS.txt`); section 9.2 contradicted 9.1. Settled
  as the safer reading: a llama.cpp architecture only where the document is that architecture's
  graph, "paperkiln" otherwise, and the check is mechanical (section 3.1, rule 38: the document against
  section 9.3's builder run on the file's own metadata, so a label whose metadata disagrees with the
  document, an epsilon or a RoPE base, is caught too). paperkiln's exporter writes "paperkiln" for
  those families; `tools/gguf_relabel.py` relabels earlier files without touching their documents.
  Refusals `architecture_kimi_as_gpt2`, `architecture_swa_as_gpt2`, `architecture_metadata_disagrees`,
  `architecture_without_graph`; the suite's pk_gpt2_swa, pk_kimi and pk_recurrence are relabelled (their
  documents, digests and expected values unchanged). To build section 9.3's graph for pk_moe the
  builder gains llama.cpp's MoE llama (Mixtral's layout).
- **K2. A rewind into a sliding window** (finding 4). Section 6.9 states it: the rows the window
  dropped are gone; an engine keeps the rows of positions < n when the window's rows behind the cut are
  still held, and otherwise re-prefills (from the start or a snapshot), so the state is a fresh run's
  and prefill = decode holds; the reference has no cache and computes exactly that. Traces
  `pk_gpt2_swa_rewind_window` and `tp_gemma3_rewind_window`. On llama.cpp the second separates the
  cache settings: after the cut its logits are within 4.6e-4 of the reference (its GELU table's error,
  as before the cut) with `swa_full` on (the library's default) and with `swa_full` off behind
  llama-server's guard, and 1.04 away with `swa_full` off and a 4-row ubatch (a 32-cell windowed cache)
  without the guard. transformers refuses the crop and its adapter re-prefills, within 2.3e-6 (H3 of
  `adapters/hf/FINDINGS.txt`); refusing at the engine's level, with the session unchanged, conforms
  too. H2 of the same file: 5.15's Decode paragraph now counts the window - 1 rows needed between calls.
- **K3. llama.cpp and Mistral.** Section 9.3 said llama.cpp b6000 ignores Mistral's window; it has no
  architecture "mistral" and refuses such a file, and its converter writes Mistral as llama without the
  window. The text says so; `tp_mistral` stays (it tests the builder).
- **K4. Runner messages.** A recurrent slot the engine does not report fails with "recurrent slot X not
  reported", and a trace an engine skips keeps the engine's own reason when its score fallback is not
  implemented either (`tools/ir_conformance.py`).
- **K5. llama.cpp's capabilities.** A row in Appendix C (informative).
- **T5's relative position bias** (LOCAL_LLM_STACK.md section 11 item 8). Adopted as proposed, an
  optional `attention` attribute `position_bias` {weight, buckets} (section 5.15), with these
  statements added: the distance is between positions, not rows (equal outside Mixture-of-Recursions);
  the table is [B, heads] with ne {heads, B}, HF's and llama.cpp's `attn_rel_b` layout; the last bucket
  covers every larger distance; the bias is added after the scaled product, each step rounded; it
  composes with windows, sinks, `turn_state` and grouped-query attention; it is structural; rule 39.
  The proposal's numpy lines matched `mtstudio score` on paperkiln's gate model within 2.62e-7 of
  max(1, |ref|), greedy 40/40, and the reference's lines give its values bit for bit on the three op
  cases. The suite has the three op cases, four refusals and, from v1.3.2, the whole-model case
  `pk_flex_t5` (paperkiln's gate recipe), a new case that moves no existing value (section 2.6).
  ember runs it on its CPU interpreter (1ab6a252; the op cases within 1.6e-7 at v1.3.0).

- **transformers' findings** (`adapters/hf/FINDINGS.txt`, H1 to H6, keramos f1b7b49). H1: a
  multi-id forward over an existing state is now a case of its own for every model (traces
  `<model>_chunks`), where transformers' Mamba bug showed only by chance. H2: 5.15's Decode paragraph
  counts window - 1 rows between calls. H3: a rewind into a window may be refused (section 6.9). H4:
  rule 38 covers HF-checkpoint engines as it does llama.cpp. H5 is K4. H6: a row in Appendix C.

- **ember's five questions on revision 1.3** (v1.3.1; ember conforms to v1.3.0 at 355 pass, 0 fail, 49
  skip, receipt 4d05e95, following the reference in each): rule 35 under substitution judges the document
  read (section 3.2); rule 38 judges the embedded document even under substitution (3.1); a read of
  `tokens` or of `depth_route`'s out under depth_route is a rule 17a fault with rule 36 as its consequence
  (10.1); rule 38's comparison names values by their writes, the block's carried values by the embed's,
  and compares `rmsnorm.group` as written (3.1); `position_bias` under depth_route uses positions (5.15).
  Cases: accept `substituted_over_duplicate_embedded`, refusals `architecture_substituted_document`,
  `mor_reads_tokens`, `mor_reads_route_out`.

### 12.5 Open

None. Q16 is settled in v1.4.2 (section 12.6), Q17 in revision 1.6 (section 12.8). The next question takes the
number Q18.

### 12.6 Settled in revision 1.4

The first primitive drafted under the workspace's path for new primitives (`LOCAL_LLM_STACK.md` section
5.4), and the wiring the arXiv chain's Primer refusal needed. Each with the answer adopted.

- **N1. Shaw-style relative positions** (Shaw, Uszkoreit and Vaswani 2018; published, not new here).
  Neither `position_bias` (a scalar per head and distance) nor `expr` (no op of it mixes tokens) states a
  vector added to the keys and values by distance, so the chain refused papers that use it. Adopted as an
  optional attribute of `attention`, `relative_kv` {key, value, max_distance} (section 5.15, rule 40), not a
  new op: the node keeps its slot, mask, window and decode rules, and an engine without the attribute
  refuses it by name (rule 1's closed key set). The tables are rows by clipped distance, so a causal model
  stores K + 1 vectors where Shaw's text has 2K + 1 (the K positive distances are never visible), and
  their width selects vectors shared by the heads or one set per head, both of Shaw's configurations.
  Cases: op `attention/relative_kv`, `_per_head`, `_keys_only`, `_window_sinks`, `_grouped_query`,
  `_distance_zero`; gradient `grad/attention.json` (3 cases); model `pk_flex_shaw` (K 4, one set per head
  and layer) with decode and chunks traces; refusals `relative_kv_*` (7, rules 40 and 6).
- **N2. Primer's multi-dconv-head attention** (So et al. 2021; published). Wiring, not an op: a depthwise
  causal convolution over time on each column of the q, k and v projections is `conv1d_causal` (section
  6.6) on the fused projection's 3d columns, before RoPE, over a recurrent slot beside the kv slot. The
  existing text settles the three questions ember asked: a block may declare several slots and each
  recurrent slot has one reader (6.1); a truncate restores or recomputes the conv slot's K - 1 rows with
  the kv rows (6.9, as `pk_srd_truncate`); the slot runs on every decode step beside the kv cache, as SRD's
  does. Cases: model `pk_flex_mdha` (K 3, RoPE, squared ReLU) with decode, snapshot, truncate and chunks
  traces. The clean-room engine, written from revision 1.2's text and unchanged since, passes all of them
  and the acceptance case: the wiring needed no new text.
- **N3. relative_kv's names under weight_index** (ember's question on 1.4.0; v1.4.1). Section 3.7 listed
  the tensor names without `relative_kv`'s, so whether its `{i}` follows `weight_index` was stated only by
  7.1's "every tensor name it holds". Stated in 3.7, 7.1 and 7.2 as the reference, ember and the validator
  already behave: m_i, never the application (attention carries no `tensor_index`). Case: the trace
  `pk_flex_shaw_weight_index` and the acceptance case `relative_kv_weight_index`, pk_flex_shaw's file under
  its document with `weight_index` [1, 0], whose logits are 1.4e-1 of max(1, max |ref|) from the unpermuted document's over 8 ids (the argmax agrees everywhere), against the ROUNDING tolerance 1e-5.
- **Q16. Windows under Mixture-of-Recursions** (v1.4.2). Section 5.15 defined a window over positions,
  while the reference counted it in a recursion step's active rows (`tools/ir_numpy.py` built the mask
  from row indices 0 .. T-1 of the step's rows). Settled as positions, the text's reading: the reference
  was brought into line with it (a patch, section 2.6: no stored document combines `depth_route` with a
  window, so no value moves). The evidence: section 7.4(b) and 7.5 place every active row at its absolute
  position, and `position_bias` and `relative_kv` under `depth_route` were already settled as positions
  (v1.3.1, revision 1.4), so a rows reading would make the window the one attribute that counts rows;
  a window is a span of text, and in rows its reach would grow with depth as tokens exit; section 6.9's
  rewind rule is stated in positions; and ember keeps each application's row positions under
  `kv_rows: active` (its compact caches report `kv_positions`) and already masks `turn_state` by
  positions, so the positions reading needs no state an engine lacks. Stated in sections 5.15, 6.9 and
  7.5. Cases: the traces `pk_mor_window` (a prefill of 4, a call of 3, then token-by-token decode to 14
  ids) and `pk_mor_window_rewind` (14 ids, a cut to 8, 3 new ids), and the acceptance case `mor_window`:
  `pk_mor`'s file under its document with window 3 and sinks 1 on the slot and the attention node
  (`accept/pk_mor_window.ir.json`), 1.0e-2 of max(1, max |ref|) from the rows reading over the trace's 14
  ids. ember refuses `depth_route` with a windowed slot (a capability refusal, section 10.3), so its
  traces skip until it runs the combination.
- **The clean-room engine's revision 1.3 and 1.4 pass** (v1.4.2; `cleanroom/FINDINGS.md`, R1 to R7). The
  Rust engine, written from revision 1.2's text, was brought to revision 1.4 from the text alone:
  `position_bias`, `relative_kv` and rule 38's labels needed no reading beyond the text, and their 20
  expected failures pass. Its findings, each settled by stating what the reference and the validator do:
  R1, rule 38 named `tools/gguf_ir.py`'s graphs as normative, so an engine that may not read the tools
  could not apply it: Appendix D now states every graph (metadata, defaults, the graph when an optional
  tensor is absent, when the metadata builds none) and D.7 the comparison; R2, the allowed labels are
  written as strings (3.1, rule 38); R3, node order, slot order and `per_layer` against a number are
  differences (3.1); R4, a file with an embedded document and no `general.architecture` is refused (rule
  38); R5, ne is compared as stored, its dimension count included (3.1); R6, `position_bias` and
  `relative_kv` allow `_` members and no others (5.15, rules 39 and 40); R7, 3.5 lists `relative_kv`'s
  default. The engine then implemented rule 38 from Appendix D alone, the chat profile from sections
  8.1 and 11.3 (`render_chat`, `parse_reply`) and Q16 (which needed no code: its steps already ran at
  their positions). That second pass found ten readings the text left open, P1 to P10, each settled
  as the reference behaves: the metadata keys' types and which zero values build no graph (D.1); a
  tied name tested as written (D.7); the 27B scale's d / H a binary64 quotient (D.5); experts on the
  label "llama" only (D.3); a record's required fields and which of its objects refuse other members
  (8.1.3); a message's required string content (8.1.1); `thinking: false` without
  `generation_prompt_nothink`, `trim` under a system turn, and the reserved check before `strip_think`
  (8.1.3). Its receipt at v1.4.2 has no expected failure.
- **Schema v2 of the chat profile** (v1.4.2). Section 8.1.6 held the design from before the build;
  paperkiln built `paperkiln/2` (22f8fe6d: `tools/chat_template.py` and the C++ twin
  `include/microtorch/chat_template.hpp`, which ember vendors from b49eb5c), and section 8.1.6 now states
  the record as built: its fields, tokens, rendering, structure, reply parsing, loss, export and
  resolution, with the reference renderer `tools/ir_chat.py` extended to it. A patch (section 2.6): the
  chat record is data beside the document, not part of it, and no document, schema entry or expected value
  moves; an engine without v2 refuses the record's new fields by name (section 8.1.4, step 1), so no file
  that resolved before resolves differently. Cases: the 42 v2 conversations of paperkiln's
  `tests/chat_template_cases.json` (18 rendered, 24 refused) in `chat/render.json` beside the 32 v1 cases,
  and its 12 reply-parsing cases in `chat/parse_reply.json` (adapter method `parse_reply`, section 11.3).

### 12.7 Settled in revision 1.5

The later third-party families (`LOCAL_LLM_WORK.md` E4), each taken through the triage of `LOCAL_LLM_STACK.md`
section 5.4, cheapest first. Each is a published mechanism; none is new here.

- **L1. Gemma 2's soft-capping** (Gemma Team 2024, arXiv 2408.00118). The final cap is row-wise, an `expr` in
  the head (triage b: no new text). The attention cap acts on the scores inside the softmax, which no node
  exposes, so it is an optional attribute of `attention`, `softcap` (triage c; section 5.15), after the scale
  and before any position bias (llama.cpp's order; transformers' Gemma 2 also caps the scaled scores). The
  local and global windows already existed as per-layer attributes. Cases: op `attention/softcap`,
  `_window_sinks`, `_position_bias`; model `tp_gemma2` (local layers windowed, both caps binding) with decode,
  chunks and rewind-into-window traces; refusals `softcap_not_positive` and `architecture_gemma2_softcap_dropped`
  (rule 38: the file's metadata states the cap its embedded document drops).
- **L2. LongRoPE** (Ding et al. 2024, arXiv 2402.13753; Phi-3, arXiv 2404.14219). Per-pair rescaling of the
  RoPE frequencies is an attribute of `rope` (triage c; section 5.14): `freq_factors`, one set or a short and a
  long set with the length that separates them, and `attn_factor`, the table's scaling. The set follows the
  sequence's length, as in transformers' Phi-3 (its generation recomputes the cache when a sequence first
  passes `original_max_position_embeddings`), not the context an engine is configured for (llama.cpp picks the
  set once from its context size, so configured above original_ctx it computes another function on shorter
  sequences); the decode and rewind rules follow (sections 5.14 and 6.9). One set is Llama 3.1's `rope_freqs`
  as well (section 9.3's llama does not build it yet). Cases: op `rope/freq_factors`, `_short`, `_long`,
  `rope/attn_factor`, `rope/freq_factors_partial_attn_factor`; model `tp_phi3` (original_ctx 6, partial
  rotary, attn_factor 1.25, tied head) with decode, chunks and truncate traces that cross the boundary;
  refusals `attn_factor_not_positive` and `freq_factors_*` (4: rules 41 and 6).
- **L3. Third-party mixtures of experts** (Mixtral; Qwen1.5-MoE and Qwen2-MoE; Qwen3-MoE). Wiring (triage a):
  `moe_ffn` already states llama.cpp's expert layout, top-k and both gate normalisations, and Qwen-MoE's shared
  expert with its sigmoid gate is `linear`, `sigmoid` and `mul_col`. Section 9.3 gains qwen2moe and qwen3moe;
  Mixtral is the llama builder's existing MoE form. Cases: models `tp_qwen2moe` and `tp_qwen3moe` with decode
  and chunks traces.
- **L4. Multi-head latent attention** (DeepSeek-V2, arXiv 2405.04434). No existing node computes it: its keys
  and values are per-head linear maps of a cached latent, its query's rotary part is a separate per-head slice,
  and its value width differs from its key width (triage d, a new op). Stated as `mla_attention` (section 5.20)
  in the absorbed form llama.cpp converts to, over a kv slot whose row is the latent and the shared rotary key,
  so every rule of the kv slot (window, sinks, decode, rewind, snapshots, `kv_rows`) applies unchanged and no
  new kind of slot is needed; the per-head split of the query is `split` with `groups` (section 5.5). The
  reference agrees with the decompressed float64 statement of the paper's equations within 2.1e-7 of max(1,
  |ref|) (`tests/test_ir_v15.py`, with the soft cap's 3.8e-8 and LongRoPE's 8.8e-8). Cases: op
  `mla_attention/two_heads`, `_window_sinks`, `_one_head`, `split/groups`; model `tp_deepseek2` (query through
  the low-rank q_a / q_b, latent 6, rope 4, nope 4, v 3; raw gates scaled by 2.5; one shared expert) with decode,
  chunks and truncate traces; refusals `split_groups_zero` and `mla_*` (7: rules 12, 42, 7, 8, 14).

### 12.8 Settled in revision 1.6

What real DeepSeek checkpoints need beyond revision 1.5's MLA, each taken through `LOCAL_LLM_STACK.md` section
5.4's triage, cheapest first. Each is a published mechanism.

- **Q17. Layers whose node lists differ** (opened in revision 1.5). DeepSeek-V2-Lite's first layer and V3's
  first three are dense, the rest mixtures of experts (DeepSeek-AI 2024, arXiv 2405.04434 and 2412.19437). Three
  statements were weighed: a per-layer choice of subgraph, a second block section of its own repeat before the
  first, and a per-application switch on single nodes. The last is the smallest: one optional attribute,
  `apply` (section 3.8), whose absence keeps every document's meaning; an interpreter skips a node in an
  application, and every other rule (per-layer values, `{i}` names, `weight_index`, state) is unchanged. A
  second section would add a document key, a second state list and a second repeat to every rule that speaks of
  the block; a subgraph choice would add nested node lists. Stateful nodes do not take it, so every slot still
  runs in every application. Cases: model `tp_deepseek3` (one dense layer, two mixture layers) with decode,
  chunks and truncate traces; refusals `apply_entries_not_boolean`, `apply_outside_block`,
  `apply_with_depth_route` (rule 43), `apply_on_attention` (rule 4), `apply_leaves_value_unwritten` (rule 36).
- **R1. DeepSeek-V3's router** (eqs. 12-16 of arXiv 2412.19437; the bias of Wang et al. 2024, arXiv
  2408.15664). Attributes of `moe_ffn` (triage c; section 5.16): `scoring` sigmoid, a `selection_bias` that
  enters only the choice of experts, and `groups`, node-limited routing (the top groups by the sum of their two
  largest biased scores). The gates come from the unbiased scores, renormalised over the k chosen; the routed
  scaling factor stays an `affine`. The reference agrees with a float64 statement of the equations within
  1.5e-7 of max(1, |ref|) (`tests/test_ir_v16.py`). Cases: op `moe_ffn/sigmoid_renorm`,
  `_sigmoid_bias_groups`, `_softmax_bias_raw`; refusals `moe_groups_not_dividing` (10), `moe_scoring_unknown`
  (5), `moe_selection_bias_shape` (44).
- **R2. YaRN** (Peng et al. 2023, arXiv 2309.00071, sections 3.2 and 3.4). Its NTK-by-parts interpolation is a
  fixed per-pair rescaling of the RoPE frequencies, which revision 1.5's `freq_factors` already states; the
  values come from the metadata (scale factor, original context, the betas), so `freq_factors` may now be given
  by value (triage c; section 5.14) and the builder computes them (Appendix D.11; within 7.5e-8 of the paper's
  ramp written out in float64, `tests/test_ir_v16.py`). Its attention temperature is DeepSeek's mscale^2 in the
  attention scale (or `attn_factor` for a model that scales the table). Cases: op `rope/freq_factors_values`;
  model `tp_deepseek3` (factor 4 over an original context of 8); refusals `freq_factors_values_length` and
  `freq_factors_values_not_positive` (41).

## Appendix A. The `-ffast-math` audit

*Informative. Audit of 25 October 2026 at paperkiln c3f5ef12, with revision 1.1's additions: the
contraction flag (Q15) and the translation units of `Matrix`'s elementwise operators. No computed value
changed.*

**A.1 Flags.** `third_party/transformer_cpp/CMakeLists.txt` (identical in the sibling
`coalfire.cpp`, which local Windows configures pick up first; clones, Colab, RunPod and ext4 builds
use the vendored copy) compiles the static library `transformer_core` with
`target_compile_options(transformer_core PRIVATE -O3 -march=native -ffast-math)` on GCC and Clang,
and `/O2 /arch:AVX2 /fp:fast` (C++ only) on MSVC; the Python wheel replaces `-march=native` with
`-march=x86-64-v3`. The options are PRIVATE: paperkiln's own library (`src/*.cpp`) and tools
(`tools/mtstudio.cpp`) are built with the Release defaults (GCC `-O3 -DNDEBUG`, no `-march`, no
`-ffast-math`) plus `-ffp-contract=off` (Q15, section 12.2), and no `-ffast-math` reaches a link line,
so no executable links GCC's `crtfastmath.o` (which would set flush-to-zero at start-up; `nm` finds no
`set_fast_math` in mtstudio, expr_eval or the tests). paperkiln's CUDA sources use nvcc's defaults
(`--fmad=true`).

**A.2 What paperkiln calls in those translation units.** `include/microtorch/primitives.hpp` is the
whitelist: `matmul_optimized` (`src/matmul_optimized.cpp`), `Matrix::apply_gelu` and
`Matrix::apply_softmax` (`src/components.cpp`), and `Matrix` storage and transpose
(`src/matrix.cpp`, copies only). `gguf_export.cpp` writes files and evaluates nothing.

**A.3 The reductions there.**
- `matmul_optimized`: recursive halving of M, N and K until all are <= 128, then 48 x 48 x 48
  blocks; each block's dot product accumulates 8 lanes with `_mm256_fmadd_ps` (FMA, by intrinsic,
  whatever the flags), sums the lanes, then a scalar tail; block results are added into C. The order
  over K depends on M, N and K, so one row can sum in a different order at T = 1 than at T > 1.
- `apply_softmax`: written as sequential loops (row max, exp and sum, divide), but under
  `-O3 -march=native -ffast-math` the compiler may vectorise the sum (reassociation), call a vector
  exp, and replace the division with a reciprocal multiply. Rows run under OpenMP.
- `apply_gelu`: elementwise; fast-math may reassociate 0.044715 x x x and contract the products.

**A.4 Which IR ops reach them in paperkiln's CPU build.**

| IR op or statement | paperkiln code | Translation unit | Verdict |
|---|---|---|---|
| `linear` (all families, the head, the MoR router, highway, SRD, Mamba projections, MoE experts) | `ops::matmul` -> `device::gemm` -> `matmul_optimized` | transformer_core, fast-math | ROUNDING, as stated |
| `attention`, prefill and training (`ops::fused_attention`, `fused_attention_blocked`) | q k^T and A v by `device::gemm`; softmax loops in `ops.cpp` | both | ROUNDING, as stated |
| `attention`, KV-cache decode (`LlamaBlock::forward_cached_rows`) and `mask: turn_state` (`ops::attention_heads_turn_state`) | scalar loops, ascending | `src/llama.cpp`, `src/ops.cpp` | sequential ascending sums |
| `gelu` | `Matrix::apply_gelu` | transformer_core, fast-math | ROUNDING, as stated |
| `depth_route` gate p, `moe_ffn` gates, gpt2 `attnres` `depth_attend` softmax (`nn::AttnResStack`) | `ops::softmax_row` -> `apply_softmax`; attnres logits by `ops::matmul` | transformer_core, fast-math | ROUNDING, as stated; their stated orders are reference orders (4.4) |
| `depth_attend` of `llama_attnres` (`ops::depth_attend`) | scalar loops | `src/depth_attend.cpp` | follows the stated order |
| `depth_route` routing (argmax, ties low) | loop in `Llama::forward` | `src/llama.cpp` | EXACT given the logits, as stated |
| `embedding`, `pos_embedding`, `split`, `list`, `append`, `mul`, `relu`, `affine`'s shift, `mul_col`, `scale`, `rope` rotation | `ops.cpp` (`Matrix::hadamard` is inline in the header) | paperkiln | EXACT, as stated (A.5) |
| `add`, `sub`, `affine`'s product (`ops::add`, `ops::sub`, `ops::scale` on the host) | `Matrix` `operator+=`, `-=`, `*=` (float) | transformer_core's `components.cpp`, fast-math | EXACT, as stated: one IEEE operation per element on two runtime operands (A.5) |
| `rmsnorm`, `layernorm`, `silu`, `sigmoid`, `rms_row`, `expr`, `linear_attention`, `ssm_scan` | `ops.cpp`, `expr.cpp`, `kimi_linear.cpp` | paperkiln | ROUNDING, as stated; orders followed |
| `recurrence` (`ops::recurrence`) | `src/recurrence.cpp` (reads, ordered sums) and the elementwise ops of the rows above | both | EXACT bodies EXACT, the rest ROUNDING, as stated (6.10.4) |

**A.5 Verdicts on the statements that claim bit identity.**
- The EXACT ops run in paperkiln's own translation units, compiled with `-ffp-contract=off` and
  without `-ffast-math`, except `add`, `sub` and `affine`'s product, which loop over the elements in
  transformer_core's `components.cpp`. There each element is one IEEE addition, subtraction or product
  of two runtime operands: `-ffast-math` has nothing to reassociate, contract or fold in it (it enables
  value-changing rewrites only across several operations or with constants), and with no
  flush-to-zero at start-up (A.1) a vector lane computes what a scalar would. They stay EXACT, on every
  build (Q15).
- `rope`: the rotation is EXACT given the table; the table is ROUNDING (section 5.14, Q3). The
  earlier wording ("computed in double") described the numpy reference only.
- Eviction (6.8.3): paperkiln's KV cache sums the visible keys in scalar ascending loops in
  `src/llama.cpp`; the q, k and v rows come from `linear`, whose order depends on the rows per call,
  so keep and evict agree bit for bit when both runs feed the same chunks (as
  `tests/test_state_tokens.cpp` gate 7 does). EXACT under that condition, now stated.
- No state id = plain causal (6.8.2): paperkiln's `Llama::forward` builds no turn_state codes when
  no id is a state id, so both run the causal kernel and agree bit for bit; an engine whose turn_state
  and causal kernels sum differently agrees within ROUNDING. Now stated (4.5).
- Recurrent prefill = decode (6.3): true of each recurrent op given identical inputs. paperkiln's
  inputs come from `linear`, whose order depends on T, so a whole paperkiln model's prefill and
  token-by-token decode agree within ROUNDING; ember's CPU build measured them bit-identical. Now
  stated (4.5).
- MoR (7.4): the depths are prefix-determined and the logits agree to rounding, as stated.

**A.6 ember.cpp** (read only; ember's statements and review of 25 October 2026). ember compiles none
of its interpreter with `-ffast-math`, and sets `-ffp-contract=off` on `graph_ir_ops.cpp`,
`graph_ir_expr.cpp` and `graph_ir_mamba.cpp` (`ember.cpp/CMakeLists.txt` lines 305-307; MSVC does
not contract without `/fp:contract`). Its EXACT results rest on sequential ascending sums and on
no contraction. The rope rotation (`graph_ir.cpp`, `utils.cpp`), `linear_attention` and `ssm_scan`
(`graph_ir.cpp`), whose text says "rounded", run outside those three files: safe on x86-64 (no
`-march`) and WebAssembly (no FMA), not on arm64 (Apple clang contracts by default; GCC aarch64 in
gnu++17 too), so ember sets the flag on `graph_ir.cpp` and `utils.cpp` too (ember 85e269e, which
changes no x86-64 instruction; Q4). ember's `linear`
(`dot_f32_as_f64`) sums in a fixed 8-lane double order independent of T and of threads, so its
whole-model prefill equals token-by-token decode bit for bit (section 4.5). Its CUDA paths are
ROUNDING by construction (tiles, flash2, nvcc's default `--fmad`); its interpreter's cuBLAS runs
FP32, while its tuned llama path runs TF32 by default (Appendix C).

Nothing in this audit changes a computed value: the tiny-chat precision gate (HEAD with precision off
equal to dc29b5e6 bit for bit) and the pinned commits of the RunPod campaigns are unaffected.

## Appendix B. Changelog

- **v1.6.0** (tag, 28 October 2026; revision 1.6). Real DeepSeek checkpoints (section 12.8): the node attribute
  `apply` (section 3.8, rule 43, rule 36 per application; Q17 settled), `moe_ffn` `scoring`, `selection_bias` and
  `groups` (section 5.16, rules 5, 10 and 44), `freq_factors` by value (section 5.14, rule 41); section 9.3 and
  Appendix D.11's deepseek2 build leading dense blocks, V3's router and YaRN. Schema: those attributes (`apply`
  on every op that is neither stateful nor `depth_route`). Tools: the reference computes them;
  `tools/gguf_ir.py`'s `yarn_factors`; `tools/ir_validate.py` checks rules 43 and 44 and the new clauses;
  `tests/test_ir_v16.py` checks the reference against float64 statements. Suite: model `tp_deepseek3` with
  decode, chunks and truncate traces, accept case and digest vector; op cases `moe_ffn/*` (3) and
  `rope/freq_factors_values`; refusals (10): 598 cases. Every earlier expected value unchanged; regenerated in
  the named environment (section 4.6). `conformance/expected_fail.json`: the clean-room engine's new cases.

- **v1.5.0** (tag, 28 October 2026; revision 1.5). The later third-party families (section 12.7):
  `attention.softcap` (section 5.15); `rope.freq_factors` and `rope.attn_factor` (section 5.14, rule 41, with
  the decode and rewind rule of the two sets, section 6.9); `split.groups` (section 5.5); the op `mla_attention`
  (section 5.20, rule 42; rules 7, 10, 12, 14 and 37 name it); section 9.3's phi3, gemma2, qwen2moe, qwen3moe
  and deepseek2; Q17 opened. Schema: those attributes and the op. Tools: the reference computes them;
  `tools/gguf_ir.py` builds the five families; `tools/ir_validate.py` checks rules 41 and 42 and the new
  clauses; `tests/test_ir_v15.py` checks the reference against float64 statements of the published formulas.
  Suite: models `tp_phi3`, `tp_gemma2`, `tp_qwen2moe`, `tp_qwen3moe` and `tp_deepseek2` with model cases, decode
  and chunks traces, truncate traces (`tp_phi3`, `tp_deepseek2`), a rewind into the window (`tp_gemma2`), accept
  cases and digest vectors; op cases `attention/softcap*` (3), `rope/freq_factors*` and `rope/attn_factor` (5),
  `split/groups`, `mla_attention/*` (3); refusals (14): 578 cases. The refusal `architecture_without_graph`
  relabels pk_llama as "falcon" (it used "phi3", which section 9.3 now defines); its rule is unchanged. Every
  earlier expected value unchanged; regenerated in the named environment (section 4.6).
  `conformance/expected_fail.json`: the clean-room engine's new cases.

- **v1.4.2** (tag, 28 October 2026; section 2.6). Three statements of what the reference and the
  implementations already do, none moving a valid document's meaning, the schema or an existing expected
  value (section 12.6). (1) Q16 settled: a window under `depth_route` counts positions (sections 5.15,
  6.9, 7.5); the reference, which counted a step's active rows, brought into line with the text (no
  stored document combined the two). (2) The clean-room engine's revision 1.3 and 1.4 pass
  (`cleanroom/FINDINGS.md`): Appendix D states the third-party graphs and rule 38's comparison, which 3.1
  had left to `tools/gguf_ir.py`; rule 38's labels as strings and a missing label refused; ne compared
  as stored; `_` members in `position_bias` and `relative_kv`; `relative_kv`'s default in 3.5; and the
  ten readings of its second pass, P1 to P10 (D.1, D.3, D.5, D.7, 8.1.1, 8.1.3). `cleanroom/rust`: the
  engine at revision 1.4, with rule 38 from Appendix D and the chat methods (`arch.rs`, `chat.rs`). (3) The
  chat profile's schema v2, `paperkiln/2`, as built in paperkiln 22f8fe6d (section 8.1.6, with 8.1.1,
  8.1.3, 8.1.4 and 8.1.7); `tools/ir_chat.py` renders it and parses replies; the adapter's `render_chat`
  takes `tools` and the new optional method `parse_reply` splits a reply (section 11.3). New cases, no
  existing case or value changes: traces `pk_mor_window` and `pk_mor_window_rewind`, acceptance case
  `mor_window` (`accept/pk_mor_window.ir.json`), 42 v2 conversations in `chat/render.json` (18 rendered,
  24 refused) and the 12 cases of `chat/parse_reply.json`: 524 cases; regenerated in the named
  environment (section 4.6). `expected_fail.json`: the clean-room engine's entries removed.
- **v1.4.1** (tag, 27 October 2026; section 2.6). `relative_kv`'s `key` and `value` named among the tensor
  names of section 3.7, of 7.1's `weight_index` and of 7.2's reach (ember's question; section 12.6 N3), as
  the reference already reads them. New cases, no existing case or value changes: the trace
  `pk_flex_shaw_weight_index` and the acceptance case `relative_kv_weight_index` (pk_flex_shaw under
  `weight_index` [1, 0], `accept/pk_flex_shaw_weight_index.ir.json`): 467 cases; regenerated in the named
  environment (section 4.6). `expected_fail.json`: the clean-room engine's two new cases.
- **v1.4.0** (tag, 27 October 2026; revision 1.4). Shaw-style relative key and value vectors, an optional
  attribute of `attention` (section 5.15, rule 40; section 12.6 N1), and Primer's multi-dconv-head
  attention stated as `conv1d_causal` wiring beside a kv slot (sections 6.9, 9.2; 12.6 N2). Schema:
  `attention.relative_kv` {key, value, max_distance}. Tools: the reference computes `relative_kv`;
  `tools/ir_validate.py` checks rule 40 (`structure`, `relative_kv_gguf_errors`); the gradient kind's
  second file, `grad/attention.json`, from `attention64`, a float64 statement of the formula. Suite: op
  cases `attention/relative_kv*` (6); gradient cases (3); models `pk_flex_shaw` and `pk_flex_mdha` (paperkiln's
  recipes in `tools/ir_conformance_mtstudio.py`, trained by mtstudio) with model case, decode, chunks (both),
  snapshot and truncate (`pk_flex_mdha`) traces, accept cases and digest vectors; refusals `relative_kv_*`
  (7): 465 cases. Every earlier expected value unchanged, regenerated in the named environment (section
  4.6). `conformance/expected_fail.json`: the clean-room engine's 10 `relative_kv` cases.
- **v1.3.5** (tag, 27 October 2026; section 2.6). `tensor_index` (section 7.2) stated in full from what
  the text and the schema already require: the ops that carry it, its reach over a node's names and no
  other node's, applications 0 .. repeat - 1 against `block_count`'s stored layers, rule 19 in the head;
  with the measured distance between the two readings on `pk_llama_attnres_rec` (5.2e-2 against the
  tolerance 1e-5). New cases, no existing case or value changes: the trace `pk_llama_attnres_rec_untagged`
  (the file under its document without the tags, `accept/pk_llama_attnres_rec_untagged.ir.json`), the
  acceptance case `untagged_application_queries`, and the refusals `tensor_index_in_head` (rule 19),
  `tensor_index_value_unknown` (rule 5), `tensor_index_on_affine` (rule 4) and
  `tensor_index_absent_application_tensor` (rule 25): 437 cases; regenerated in the named environment
  (section 4.6). Section 11.3: an engine's own refusal of a case it cannot build is a skip, an exception
  from a path it offers a failure (Jonathan's ruling, 27 October 2026; O4 of
  `adapters/ortgenai/FINDINGS.txt`). Appendix C: the ONNX Runtime GenAI row (its O5) and the engines'
  `tensor_index` paths. Section 9.2's llama_attnres names the suite's cases.
- **v1.3.4** (tag, 27 October 2026; section 2.6). The specification's name is the keramos IR (titles and
  prose); the format's keys and labels keep their `paperkiln` names (the front matter). Three third-party
  models with head_dim 16, the smallest head size onnxruntime's CPU GroupQueryAttention runs with fused
  rotary (finding 1 of `adapters/ortgenai/FINDINGS.txt`; every earlier attention model of the suite has
  head_dim 4 or 8): `tp_llama_hd16` (d 32, 2 heads, 2 kv heads), `tp_mistral_hd16` (d 32, 2 heads over 1 kv
  head, window 4) and `tp_qwen3_hd16` (d 32, 4 heads over 2, head_dim 16 apart from d / heads, as in the
  released Qwen3 checkpoints), 2 layers each, with their model cases, decode and chunks traces, truncate
  traces (llama, qwen3) and a rewind into the window (mistral), accept cases and digest vectors: 431 cases,
  no existing case or value changes; regenerated in the named environment (section 4.6) and reproduced.
- **v1.3.3** (tag, 27 October 2026; section 2.6). The runner: an expected failure (`expected_fail.json`) covers a case the engine fails with an error as well as one it fails on values, so the clean-room engine's refusal of `position_bias` at load reports XFAIL. No case or expected value changes.
- **v1.3.2** (tag, 27 October 2026; section 2.6). The whole-model case of T5's relative bias, `pk_flex_t5`
  (paperkiln's recipe in its `tools/ir_conformance_mtstudio.py`: flex, rmsnorm, relu, `position:
  "t5_relative"` with 8 buckets up to distance 12, so the 16-token contexts reach the last bucket; trained by
  mtstudio at paperkiln 5c2e5064), with its decode and chunks traces, its accept case and its digest
  vector: 413 cases, no existing value moves. Appendix C and 12.4: ember runs `position_bias` on its CPU
  interpreter (1ab6a252). The clean-room engine's 4 cases of the model are expected failures (its 1.2
  build refuses the attribute).
- **v1.3.1** (tag, 27 October 2026; section 2.6). ember's five questions on revision 1.3, settled by
  stating what the reference does (section 12.4): rules 35 and 38 under substitution (3.2, 3.1), rules 17a
  and 36 under depth_route (10.1), rule 38's comparison of carried values and `rmsnorm.group` (3.1),
  `position_bias` under depth_route (5.15). New cases (no existing value moves): accept
  `substituted_over_duplicate_embedded` (with `accept/pk_llama.ir.json`), refusals
  `architecture_substituted_document`, `mor_reads_tokens`, `mor_reads_route_out`; 408 cases. Known stored
  pre-1.3 files that rule 38 refuses, all gpt2-labelled IR-only exports kept as receipts hash them:
  `C:/ml_artifacts/kimi_tiny/kimi_tiny.gguf`, `swa_tiny/swa_tiny.gguf` and
  `capacity/flex_plain/flex_plain_tiny.gguf`
  (left as they are; `tools/gguf_relabel.py` relabels copies). The whole-model case `pk_flex_t5` and
  mtstudio's receipt follow in v1.3.2.
- **v1.3.0** (tag, 27 October 2026; revision 1.3). What llama.cpp b6000 against the suite showed
  (`adapters/llamacpp/FINDINGS.txt`), and T5's relative position bias (section 12.4). New rules: a
  file's architecture is its document's graph or "paperkiln" (section 3.1, rule 38; checked on the
  embedded document against section 9.3's builder run on the file's metadata), and
  `attention.position_bias` (section 5.15, rule 39). Schema: `attention.position_bias` {weight,
  buckets}. Stated: a rewind into a sliding window (section 6.9); llama.cpp's Mistral (section 9.3);
  the builder's MoE llama (section 9.3); the flex options of 26 October built from existing ops, more
  expr inputs and norm placements (section 9.2, informative); Q16 open (section 12.5). Tools: the reference computes
  `position_bias`; `tools/ir_validate.py` checks rules 38 and 39 (`architecture_errors`,
  `structure`); `tools/gguf_ir.py` builds MoE llama and needs only the standard library (its f32 equals
  numpy's shortest float32 decimal); `tools/gguf_relabel.py`; the runner's two messages (K4). Suite:
  pk_gpt2_swa, pk_kimi and pk_recurrence relabelled "paperkiln" (their documents, digests and every
  expected value unchanged; only the three GGUFs' metadata moved); op cases `attention/position_bias`, `_window_sinks`, `_grouped_query`; traces
  `pk_gpt2_swa_rewind_window`, `tp_gemma3_rewind_window`, and `<model>_chunks` for all 21 models
  (forwards of 3, 1 and 2 ids after a prefill of 4: H1 of `adapters/hf/FINDINGS.txt`); refusals
  `architecture_*` (4, rule 38) and
  `position_bias_*` (4, rules 39 and 6); every earlier expected value unchanged, regenerated in the
  named environment (section 4.6). The whole-model case of T5's bias, `pk_flex_t5`, and mtstudio's
  receipt follow in v1.3.2 (the build they need waits for CPU slots).
- **v1.2.1** (tag, 26 October 2026; documentation patch). Appendix C's ember WebGPU row: every interpreter
  document except recurrence runs on a WGSL interpreter of the document (ember 36c61eb, gated 17b3999 on a
  software adapter). No case, schema or expected value changes.
- **v1.2.0** (tag, 26 October 2026; revision 1.2). The clean-room implementation's findings, settled
  (section 12.3). New refusals: duplicate member names (rule 35) and reads of values no node writes,
  an embed section without `x` and a head without `logits` (rule 36), neither of which a stored
  document holds (section 2.2); the width checks numbered and placed when the rows arrive (rule 37,
  section 10.1.1). Schema: `depth_attend.eps` optional, default 1e-5. Stated from the reference: the
  GGUF container and third-party documents as a capability (3.1), ties in number serialisation and
  the integer range by lexical form (3.10), persistence of values (3.5), attribute numbers (3.5),
  `rope_heads` "first" (5.14, now in the reference), `expr`'s tokens, depth and constant row
  reductions (5.19), truncation with recurrent state (6.9), reads beyond a declaration (6.10.1),
  MoR's step scope and contiguous steps (7.4), trim's whitespace (8.1.3), the adapter's paths, error
  state, edge cases and single-node defaults (11.2, 11.3). Rule citations corrected: every refusal
  case cites the rule its fault breaks, and the reference reports `IR_SPEC <section> (rule <n>)`
  (the runner checks the rule). Suite: 83 refusals (9 new), an acceptance kind (34 documents), a
  chat kind (32 cases, adapter method `render_chat`), 5 op cases, 2 traces and 4 digest vectors added,
  370 cases in all; every earlier expected value unchanged; regenerated in the named environment
  (section 4.6), where the reference passes 362 and skips the 8 gradient cases.
- **v1.1.4** (tag, 26 October 2026; section 2.6). Documentation: the environment that generated the
  suite's stored expected values (section 4.6; `requirements-reference.txt`), with the receipt of
  its reproduction and of a Linux numpy 2.3.3 regeneration (`conformance/receipts/
  reference_environment.json`). No expected value changes.
- **v1.1.3** (tag, 25 October 2026; section 2.6). Documentation: ember's corrections to section 12.1
  (it refuses a missing required attribute, e358153) and Appendix C (its v1.1.2 conformance, the
  WebGPU row), and every engine's receipt naming v1.1.2.
- **v1.1.2** (tag, 25 October 2026; section 2.6). Documentation: the chat profile's normative text
  (conversations, loss policy, the template record, resolution, export, v2, gates) is section 8.1,
  where it had been included by reference from `LOCAL_LLM_STACK.md` section 3 point 3, with the same
  meaning; Appendix C states ember's revision 1.1 and its CUDA path for Mixture-of-Recursions (ember
  66c8b2c).
- **v1.1.1** (tag, 25 October 2026; section 2.6). Clarifications from ember.cpp's implementation of
  1.1, ruled by the specification's owner: the integer token `-0` serialises as `0` and has a digest
  (section 3.10); every I-JSON check runs on the whole document before the `_` members are removed,
  and the reference now does so (it checked only duplicate names before the removal); a declaration
  may repeat an index, an assignment may not (6.10.1); a bare name that is not an index in scope is
  rule 32 (6.10.2). New cases: six digest texts refused for what their `_` members hold, a
  recurrence over a square weight `W[c,c]` (op and gradient), and two refusals. No valid document's
  digest or meaning moves.
- **1.1** (25 October 2026). Additions, no change of meaning (section 2.2): the digest (section 3.10)
  and the GGUF key `paperkiln.ir_sha256` (section 3.1, rule 30); the values of stored tensors (section
  4.7); the `recurrence` profile (section 6.10, rules 31-34), with the recurrent slot op `recurrence`
  and its `init` `"body"`; the conformance kinds digest, dequantisation and gradient, traces with a
  document (section 11); Q11, Q14 and Q15 settled (section 12.2), with `-ffp-contract=off` on
  paperkiln's translation units. Appendix A states where `add`, `sub` and `affine`'s product run.
- **1.0** (25 October 2026). The normative text of `include/microtorch/ir.hpp`'s header comment
  (IR version 0) moved here and split into profiles; versioning (section 2), validation (section
  10), conformance (section 11) and the open questions (section 12) added. Clarifications: the
  numerics classes made consistent with the reference build (section 4, Appendix A): stated orders
  of ROUNDING ops are reference orders; conditional exactness for eviction, `turn_state` without
  state ids and recurrent prefill; the RoPE table's class. Version 0 documents are version 1
  documents. Settled with ember.cpp's review (section 12.1): required attributes have no defaults;
  NaN and signed zero in EXACT ops; the f32 RoPE table; per-layer scope; one attention node per kv
  slot; `tokens` and `pos`; one `{i}` per name; depth_route's `out` and its exclusion of list values;
  the eviction bound and the rewind result; raw ids; the specials under state tokens. After ember's
  conformance run: rule 24 compares the version key with the embedded document, and a missing S = 1
  pseudo-query is an exporter-rule warning, not a refusal.
- Version 0 history (informative; commits in paperkiln unless marked): llama in the IR (1147ca0);
  gpt2 ops (ember c843421); sliding window with sinks (ember aa145f5); MoE, highway, attention
  residuals, SRD, kimi linear attention, mamba scan; third-party ops (grouped-query attention, NEOX
  and partial RoPE, per-head RMSNorm, per-layer attributes, weight_offset, freq_scale,
  conv1d_causal, selective_scan); `expr` (1159d63e); `scale` (3c1912ed); `weight_index` (567a441d);
  `depth_route` and `kv_rows` (c4427679); per-turn state tokens; `llama_attnres` and `tensor_index`
  (c48ca962).

## Appendix C. Engine capabilities (informative)

What each engine runs, per path, as of 27 October 2026. These are statements about engines, not
semantics; `LOCAL_LLM_STACK.md` section 4's op table is the running record. Where a row lists a gap
the engine has agreed to close, the specification already states the intended behaviour.

| Engine and path | Profiles | Class met | Matrix products | Refuses (valid documents it does not run) |
|---|---|---|---|---|
| numpy reference (`tools/ir_numpy.py`) | core, state, recursion, chat, thirdparty, recurrence | the reference (prefill only: no cache; no backward) | float32 numpy (BLAS order) | none |
| clean-room Rust engine (`cleanroom/rust`, `kcr`; informative: a third implementation written from this text alone, not a product; revision 1.4 at v1.4.2, rule 38 from Appendix D) | core, state, recursion, chat, thirdparty, recurrence | EXACT where this document says so (CPU) | scalar f32, ascending | GGUFs without a document (section 3.1; it builds Appendix D's graphs only to check rule 38), sidecars, `run_node_grad` |
| llama.cpp b6000 (`adapters/llamacpp/`; informative: reads no document, builds its graph from `general.architecture` and its metadata) | none; the files whose architecture is one of its graphs (rule 38) | ROUNDING on the CPU path except: f32 GELU through an fp16 table (outside tol 1e-5); f16 attention under flash attention whatever the cache type; tanh GELU on gptneox, whose models use exact GELU; no window on llama | ggml CPU | "paperkiln" (every IR-only file) and "mistral" (no such architecture); a rewind into a window is safe only with `swa_full` or llama-server's guard (section 6.9) |
| Hugging Face transformers 5.20.2 (`adapters/hf/`; informative: reads no document, runs HF checkpoints exported from the suite's GGUFs) | none; the files whose architecture is one of its model classes | ROUNDING on CPU in float32 (sdpa and eager); Mistral and Gemma 3 windows and Gemma 3's per-layer RoPE as specified; Mamba wrong on a multi-id forward over a cache (its finding 1) | torch CPU | "paperkiln"; windowed and recurrent rewinds are refused by `Cache.crop`, so its adapter re-prefills (section 6.9); through its own GGUF reader, Gemma 3's RoPE scaling and attention scale are lost and llama-architecture MoE files do not load |
| ONNX Runtime GenAI 0.18.0, onnxruntime 1.31.1, CPU (`adapters/ortgenai/`; informative: reads no document; each GGUF exported to an HF checkpoint and built by its model builder, `-p fp32 -e cpu`) | none; the files whose architecture its builder takes (llama, qwen3, mistral, gemma3; its own GGUF loader for llama and mistral) | ROUNDING on CPU in float32 where it runs: Llama and Qwen3 at head_dim 16 on the stock build, Gemma 3 once its finding 3 is fixed, the GGUF loader included; Mistral's window dropped (since 0.15.2) | onnxruntime CPU (MLAS) | "paperkiln"; Mixtral, GPT-2, GPT-NeoX and Mamba (no builder); on CPU its GroupQueryAttention refuses head_size % 16 != 0 with fused rotary (head_size % 8 != 0 without), which the builder does not check; Gemma 3 does not build under 0.18.0 (KeyError 'rope_type'), and 0.15.2 builds it with its sliding layers' RoPE scaled by the global factor; `rewind_to` exact, into a window included (`adapters/ortgenai/FINDINGS.txt`) |
| paperkiln `mtstudio`, CPU | the families it trains (section 9.2), from its own runs; recurrence nodes, attention with `relative_kv` and their gradients through `ir_eval` | EXACT ops EXACT on every build (`-ffp-contract=off`); eviction EXACT at equal chunking (Appendix A) | `matmul_optimized` (FMA, shape-dependent order) | documents it did not export (it runs its modules, not documents); recurrence bodies with weights of their own in its training lane (the op takes them) |
| paperkiln `mtstudio`, CUDA | as CPU | ROUNDING | cuBLAS FP32; TF32 opt-in | as CPU |
| ember interpreter, CPU | core, state, recursion, chat, thirdparty, recurrence | EXACT where this document says so | `dot_f32_as_f64`, a fixed 8-lane double order independent of T and threads | see below |
| ember interpreter, CUDA | as CPU, ops with device kernels | ROUNDING by construction (tiles, flash2, nvcc's default `--fmad`) | cuBLAS FP32 (its own handle at default math) | documents with an op without a device kernel are served on the CPU automatically (documents with recurrence among them); `EMBER_IR_DEVICE=cpu` forces the CPU for every document. `mask: turn_state` and its lazy eviction run on the device, ROUNDING, from ember 210f3b8; depth_route with kv_rows: active runs on the device, ROUNDING, from ember 66c8b2c (below) |
| ember interpreter, WebAssembly | as CPU | as CPU (no FMA in the build) | as CPU | as CPU (4 GB heap for 0.5B models) |
| ember WebGPU (browser) | llama and llama.cpp-layout MoE GGUFs on the tuned path's kernels; every other document ember's interpreter runs, all profiles except recurrence, on a WGSL interpreter of the document (ember 36c61eb); F32, F16, BF16 and Q8_0 matrices kept as stored | ROUNDING (WGSL; f32 accumulation, one invocation per output row) | f32, ascending columns per row | recurrence, rope_heads "first", forked lists, k-quant and Q4_0/Q5_x matrices, kv contexts past 4,096 positions (these run in WebAssembly); gated on a software adapter (SwiftShader) against ember CPU: greedy 40/40 and top-1 at every position (17b3999) |
| ember tuned llama path, CPU and CUDA | llama | ROUNDING against the interpreter | CUDA: TF32 on by default (`EMBER_GEMM_TF32=0` gives FP32) | anything it is not selected for |

**Revision 1.6 in the engines.** The numpy reference computes all of it (section 12.8 for its float64
checks), and paperkiln's gates run it against transformers on synthetic DeepSeek-V2 (a dense first layer,
YaRN) and DeepSeek-V3 models (`LOCAL_LLM_STACK.md` section 5.3). mtstudio has no path for third-party documents.
The clean-room engine (revision 1.4) fails the 44 cases that need revision 1.5 or 1.6, which `expected_fail.json`
lists. ember's interpreter implements the revision in its own session.

**Revision 1.5 in the engines.** The numpy reference computes all of it; its float64 checks are in section
12.7, and its gates against transformers on synthetic models converted by llama.cpp's own converter (Phi-3
with LongRoPE and with a window, Gemma 2, Mixtral, Qwen2-MoE, Qwen3-MoE, DeepSeek-V2 with and without the
low-rank query) are paperkiln's (`LOCAL_LLM_STACK.md` section 5.3). paperkiln's mtstudio runs only its own
modules, so the five new models and the new op cases are skips for it. The clean-room engine (revision 1.4,
v1.4.2) has no graph for the five architectures and no path for the new attributes and op: the 35 cases
that use them are in `expected_fail.json`. ember's interpreter
implements the revision in its own session. llama.cpp, transformers and ONNX Runtime GenAI keep their v1.3.5
receipts.

**Revision 1.4 in the engines.** The numpy reference computes `relative_kv`. paperkiln's mtstudio runs
it through `ops::shaw_attention`, the op it trains with, in its `ir_eval` (op and gradient cases) and in
`pk_flex_shaw`'s model case; its window-and-sinks case is a skip, as for `position_bias`. The clean-room
engine ran `pk_flex_mdha` from revision 1.2's text; at v1.4.2, brought to revision 1.4 from the text alone,
it passes every case but the 11 gradient ones (section 12.6).
llama.cpp, transformers and ONNX Runtime GenAI read no document: the two new models are architecture
"paperkiln" (rule 38) and the other new cases need a document or a single node, so each of the 28 is a skip
for them by construction; their v1.3.5 receipts stand. ember's CPU interpreter (757e36b, written from
section 5.15's text) passes 413, fails 0 and skips 52 at v1.4.0: every new case but the three gradient
ones (at v1.4.1, ember b49eb5c, 415, 0 and 52 with the two `weight_index` cases; at v1.4.2, ember 1456c0c, 472, 0 and 52, the 57 new cases passing: the windows under `depth_route`, `parse_reply` and the v2 chat cases), with `relative_kv` in the stated order and `pk_flex_mdha` as written at ec3201b. Its CUDA builds
run `pk_flex_mdha` on the device and serve `relative_kv` on the CPU; WebAssembly is bit-equal to the CPU
and WebGPU (a software adapter) has a `relative_kv` kernel. On the arXiv chain's model from Shaw et al.
(1803.02155; flex d 64, 2 layers, per-head tables, max_distance 16) ember's replies equal mtstudio's
42/42 and its logits are within 5.1e-7 of the numpy reference (ember
`experiments/d2_primitives_2026-10-27`).

**ember's tuned-path selection.** ember selects its tuned llama path by structural match (ember
f2727c3): the node list against the canonical llama graph, with `family_hint` "llama" only as a
pre-filter (section 9.1).

**ember's CUDA builds and state tokens.** ember 210f3b8 or later serves state-token models on CUDA:
the turn_state mask and its lazy eviction run on the device (ember aadeed2; the fallback prefill
attention reduction pads head_dim to a power of two, with a barrier fix, kernel-tested at 12 head_dims,
606e1d4), ROUNDING. On state4 on a T4 against ember's CPU: greedy 40/40, top-1 9264/9264, prefill logits
within 6.2e-7; kv rows held equal the CPU's and mtstudio's on 40/40 prompts; rewinds into evicted turns
80/80.

**ember's CUDA builds and Mixture-of-Recursions.** ember 66c8b2c or later serves MoR documents on CUDA.
The router logits come from the device and each token's depth and gate from the CPU's routing code on
the host, so a depth is EXACT given its logits; each recursion step runs over its active tokens at
their absolute positions into each application's compact kv cache, and decode runs per launch.
ROUNDING. On a T4 against ember's CPU on mor3 and mor2p0: greedy 40/40, top-1 1299/1299, logits within
2.0e-7, depth traces 40/40 in prefill and decode, kv rows held per application equal to the CPU's,
rewinds 160/160. A pos_embedding or a window inside a MoR block keeps the document on the CPU (the CPU interpreter counts the window in positions over each application's rows, section 5.15, from ember 1456c0c).

**ember and revision 1.1.** ember 39dcce9 or later implements revision 1.1 on its CPU interpreter,
written from this text: it checks `paperkiln.ir_sha256` at load (rule 30), answers `digest` and
`dequantize` (one dequantisation routine, which its CUDA and WebAssembly paths read too) and runs
`recurrence` (rules 31-34). It does not answer `run_node_grad`, as it does not train.

**ember interpreter refusals** (capability, not validation; section 10.3): more than one kv slot per
block; per-layer values other than attention window and sinks, rope theta and freq_scale, affine
scale and shift, depth_route step, and the slot's window (per-layer sinks included); `rope_heads`
"first"; rope nodes with differing theta, freq_scale, n_rot or pairing; head ops outside its head
list (which excludes `split`, `moe_ffn`, `rope`, `pos_embedding` and `embedding`); `depth_route` with
recurrent state; `mask: turn_state` with `kv_rows: active`; eviction with recurrent
state; `expr` nested deeper than 200 (the CUDA expr kernel takes up to 96 instructions, a stack of 16
and 8 reductions, and serves larger expressions on the CPU); `attention.position_bias` per layer or under
`depth_route` (ember runs it on the CPU interpreter otherwise, from 1ab6a252).

**tensor_index** (section 7.2). The numpy reference and the clean-room engine pass every case of
section 7.2 at v1.3.5, as does ember's CPU interpreter (from aa50a61, which binds a tagged node's
names by application on every path and refuses the key nowhere at load; receipt at v1.3.5, 388 pass,
0 fail, 49 skip); paperkiln's mtstudio exports it for llama_attnres under fixed recursion and scores
its own models (the model case). On paperkiln's gate model ar_rec (llama_attnres under 1 + [2] x 2 + 1
recursion, weight_index [0, 1, 2, 1, 2, 3], F32 and Q8_0), at ember ec3201b, the CPU interpreter is
within 2.7e-7 of the numpy reference, WebAssembly is bit-equal to the CPU, CUDA is within 2.1e-7 of
the CPU and WebGPU on a software adapter is within 1.5e-6, each with greedy 40/40; on every path the
untagged document computes its own function, 2.1e-2 from the tagged reading (ember
`experiments/tensor_index_gate_2026-10-27`).

**ember's conformance.** At ember 39dcce9 its adapter passes keramos v1.1.2 (as v1.1.1) with 230 pass, 0
fail and 54 skip, the skips being the single-op fixtures of ops other than `recurrence` (its `run_node` runs
recurrence nodes) and the gradient cases; receipt `conformance/receipts/ember_cpu.json`. ember's
comparison of its load-time verdicts with `tools/ir_validate.py` finds no disagreement at v1.1.1.

**Differences between the validator and ember at load:**
- *Two kv slots in one block* (expected, remains). Valid (section 6.1); ember refuses them as a
  capability (one kv slot per block).
- *A missing S = 1 pseudo-query* (resolved). ember accepts it, as section 5.18 allows; the validator
  now accepts it too, with an exporter-rule warning.
- *The version key with a substituted document* (resolved). Rule 24 compares the key with the
  embedded document in both (section 11.2).

**paperkiln gaps:** its numpy reference keeps the double
RoPE table (Q3).

## Appendix D. The third-party graphs

*Normative for rule 38 (section 3.1) and for an engine that builds the document of a GGUF without
`paperkiln.ir` (a capability, section 3.1); v1.4.2. `tools/gguf_ir.py` implements it, `tools/ir_validate.py`
applies D.7, and section 9.3 summarises the graphs. Each graph follows llama.cpp b6000's graph of the
architecture, or the model's own Hugging Face semantics where section 9.3 says llama.cpp approximates.*

### D.1 Reading the file

- arch is the string `general.architecture`; D.3 to D.6 give a graph for `"llama"`, `"mistral"`,
  `"qwen2"`, `"qwen3"`, `"gpt2"`, `"gptneox"`, `"gemma3"` and `"mamba"`, and D.8 to D.11 (revision 1.5) for
  `"phi3"`, `"gemma2"`, `"qwen2moe"`, `"qwen3moe"` and `"deepseek2"`; no other value builds one.
- K(key) is the value of the metadata key `<arch>.<key>` (for a Mistral file, `mistral.embedding_length`
  and so on). Where a key is written "K(key), default v", an absent key reads v; every other key read
  is required, and a file without it builds no graph, as does a key of the wrong type. Types: a count
  or size (block_count, embedding_length, head counts, key_length, context_length, dimension_count,
  sliding_window, expert counts, the ssm sizes) is any GGUF integer type; an epsilon, `rope.freq_base`
  and `rope.scaling.factor` any GGUF integer or float type; `rope.scaling.type` and
  `general.architecture` strings; a truth value (`use_parallel_residual`, `ssm.dt_b_c_rms`) a bool or a
  number, true when it holds true or a non-zero number. A head count H of 0 builds no graph (d / H has
  no value); any other zero or negative count builds a graph, which the document rules refuse wherever
  it is embedded (`block.repeat` and the sizes are positive integers), so rule 38 needs no case of its
  own for it.
- "The file has T" tests a tensor name. The vocabulary size V is the number of entries of
  `tokenizer.ggml.tokens` when that key is present with at least one entry, else ne[1] of
  `token_embd.weight` (a file with neither builds no graph).
- Every number a graph writes into an attribute is computed in binary64 from the metadata values (an
  f32 key's value is exact in binary64) and rounded to f32 once, as section 3.5 reads any attribute: the
  attention scale 1/sqrt(hd) is the f32 nearest to the binary64 quotient of 1 and the binary64 square
  root.
- Common values (D.3 to D.5, D.8 to D.10): d = K(embedding_length), L = K(block_count), H = K(attention.head_count),
  KV = K(attention.head_count_kv), default H, hd = K(attention.key_length), default the integer part of
  d / H, ctx = K(context_length).
- **No RoPE scaling** (where D.3 to D.11 require it): K(rope.scaling.type) is absent or `"none"`, and the
  file has no `rope_freqs.weight`; otherwise the file builds no graph.

### D.2 Notation

A graph is the document

    {"ir_version": 1, "family_hint": arch,
     "model": {"vocab": V, "d": d, "n_layers": L, "ctx": ctx, "dtype": "f32"},
     "embed": {"nodes": E}, "block": {"repeat": L, "state": S, "nodes": N}, "head": {"nodes": HD}}

with the informative keys `tokenizer`, `inputs`, `outputs` and `modes` (section 3.2), which rule 38
ignores; the block has no other key. Nodes are written `op(in, ...) -> out {attributes}`: the node
`{"op": op, "in": [in, ...], attributes, "out": out}`, `out` a list where two names are given in
brackets. A node has exactly the attributes shown (rule 38's comparison fills absent optional attributes
with their defaults, D.7). B stands for the prefix `blk.{i}.`. Shorthands:

- `RMS(x, W) -> y` is `rmsnorm(x) -> y {weight: W, eps: eps}`; `RMSg(x, W) -> y` adds `group: hd`.
- `LN(x, P) -> y` is `layernorm(x) -> y {weight: P.weight, bias: P.bias, eps: eps}`.
- `LIN(x, W) -> y` is `linear(x) -> y {weight: W, bias: null}`; `LIN(x, W, C) -> y` has `bias: C`.
- `HEAD` is `LIN(h, output.weight) -> logits`, with `tied_to: "token_embd.weight"` added when the file
  has no `output.weight`.
- `ROPE(theta, pairing, n_rot)` is `rope(q, k, pos) -> [q, k] {heads: H, kv_heads: KV, head_dim: hd,
  n_rot: n_rot, theta: theta, rope_heads: "all", pairing: pairing}`, with `freq_scale` only where shown.
- `ATT(scale, w)` is `attention(q, k, v) -> a {state: "kv", heads: H, kv_heads: KV, head_dim: hd,
  causal: true, scale: scale, window: w, sinks: 0}` (`tools/gguf_ir.py` also writes the informative
  `softmax`).
- `KV(w)` is the slot `{"name": "kv", "kind": "kv", "heads": H, "kv_heads": KV, "head_dim": hd,
  "max_len": ctx, "window": w, "sinks": 0, "dtype": "f32"}`.
- `E0` is `[embedding(tokens) -> x {weight: token_embd.weight}]`.

### D.3 llama, mistral, qwen2, qwen3

eps = K(attention.layer_norm_rms_epsilon); theta = K(rope.freq_base), default 10000; n_rot =
K(rope.dimension_count), default hd; no RoPE scaling. pairing is `"neox"` for qwen2 and qwen3 and
`"interleaved"` for llama and mistral. w = K(attention.sliding_window), default 0, for llama and mistral;
w = 0 for qwen2 and qwen3. The projections carry biases when the file has `blk.0.attn_q.bias` (then q, k
and v read `B attn_q.bias`, `B attn_k.bias` and `B attn_v.bias`), and the graph has q/k norms when the
file has `blk.0.attn_q_norm.weight`, whatever the architecture.

    E  = E0
    S  = [KV(w)]
    N  = RMS(x, B attn_norm.weight) -> h
         LIN(h, B attn_q.weight[, B attn_q.bias]) -> q
         LIN(h, B attn_k.weight[, B attn_k.bias]) -> k
         LIN(h, B attn_v.weight[, B attn_v.bias]) -> v
         [RMSg(q, B attn_q_norm.weight) -> q;  RMSg(k, B attn_k_norm.weight) -> k]   (q/k norms only)
         ROPE(theta, pairing, n_rot)
         ATT(1/sqrt(hd), w)
         LIN(a, B attn_output.weight) -> o
         add(x, o) -> x
         RMS(x, B ffn_norm.weight) -> h2
         FFN
         add(x, f) -> x
    HD = RMS(x, output_norm.weight) -> h;  HEAD

FFN is, for a file labelled `"llama"` (not mistral, qwen2 or qwen3) with K(expert_count) present and
non-zero and the tensor
`blk.0.ffn_gate_inp.weight` (Mixtral's layout),

    moe_ffn(h2) -> f {router: B ffn_gate_inp.weight,
                      experts: {"gate": B ffn_gate_exps.weight, "up": B ffn_up_exps.weight,
                                "down": B ffn_down_exps.weight},
                      n_experts: K(expert_count), top_k: K(expert_used_count),
                      gate_norm: "renorm", activation: "silu"}

and otherwise (every other file of the four architectures)

    LIN(h2, B ffn_gate.weight) -> g;  LIN(h2, B ffn_up.weight) -> u;  silu(g) -> g;  mul(g, u) -> gu
    LIN(gu, B ffn_down.weight) -> f

### D.4 gpt2, gptneox

eps = K(attention.layer_norm_epsilon). For gptneox: no RoPE scaling, theta = K(rope.freq_base), default
10000, n_rot = K(rope.dimension_count), default hd, and P = K(use_parallel_residual), default true. gpt2
reads no RoPE key.

    E  = E0 + [pos_embedding(x, pos) -> x {weight: position_embd.weight}]     (gpt2)
    E  = E0                                                                    (gptneox)
    S  = [KV(0)]
    N  = LN(x, B attn_norm) -> h
         LIN(h, B attn_qkv.weight, B attn_qkv.bias) -> qkv
         split(qkv) -> [q, k, v] {sizes: [H hd, KV hd, KV hd]}
         ROPE(theta, "neox", n_rot)                                            (gptneox only)
         ATT(1/sqrt(hd), 0)
         LIN(a, B attn_output.weight, B attn_output.bias) -> o
         then, for gptneox with P true:  F;  add(f, x) -> x;  add(x, o) -> x
         and otherwise:                  add(x, o) -> x;  F;  add(x, f) -> x
    HD = LN(x, output_norm) -> h;  HEAD                                        (gpt2)
    HD = LN(x, output_norm) -> h;  LIN(h, output.weight) -> logits             (gptneox: never tied)

with F = `LN(x, B ffn_norm) -> h2; LIN(h2, B ffn_up.weight, B ffn_up.bias) -> u; gelu(u) -> u {approx:
A}; LIN(u, B ffn_down.weight, B ffn_down.bias) -> f`, A `"tanh"` for gpt2 and `"none"` for gptneox. In
the parallel form F reads the block's input x, before the attention output is added.

### D.5 gemma3

eps = K(attention.layer_norm_rms_epsilon); n_rot = K(rope.dimension_count), default hd; the file has no
`rope_freqs.weight`, and K(rope.scaling.type) is absent, `"none"` or `"linear"` (any other value builds no
graph); c = K(rope.scaling.factor), default 1 (0 reads 1), when the type is `"linear"`, else 1. sw =
K(attention.sliding_window), required, and g = K(rope.freq_base), required when some application is
global (L > 5). Application i is local when i mod
6 < 5 and global otherwise. Per-layer values, always written as `{"per_layer": [L entries]}` objects,
even when every entry is equal:

- windows: sw for a local application, 0 for a global one;
- thetas: 10000 local, g global;
- freq_scales, only when c != 1: 1 local, 1/c global (with c = 1 the rope node has no `freq_scale`).

scale = 1/sqrt(d / H) when L = 62 (the 27B model; d / H is the binary64 quotient, not its integer part),
else 1/sqrt(hd).

    E  = E0 + [affine(x) -> x {scale: sqrt(d), shift: 0}]
    S  = [KV(windows)]
    N  = RMS(x, B attn_norm.weight) -> h
         LIN(h, B attn_q.weight) -> q;  LIN(h, B attn_k.weight) -> k;  LIN(h, B attn_v.weight) -> v
         RMSg(q, B attn_q_norm.weight) -> q;  RMSg(k, B attn_k_norm.weight) -> k
         ROPE(thetas, "neox", n_rot), with freq_scale: freq_scales when c != 1
         ATT(scale, windows)
         LIN(a, B attn_output.weight) -> o
         RMS(o, B post_attention_norm.weight) -> o
         add(x, o) -> x
         RMS(x, B ffn_norm.weight) -> h2
         LIN(h2, B ffn_gate.weight) -> g;  LIN(h2, B ffn_up.weight) -> u
         gelu(g) -> g {approx: "tanh"};  mul(g, u) -> gu
         LIN(gu, B ffn_down.weight) -> f
         RMS(f, B post_ffw_norm.weight) -> f
         add(x, f) -> x
    HD = RMS(x, output_norm.weight) -> h;  HEAD

### D.6 mamba (Mamba-1)

kc = K(ssm.conv_kernel), di = K(ssm.inner_size), ns = K(ssm.state_size), r = K(ssm.time_step_rank), eps =
K(attention.layer_norm_rms_epsilon), L = K(block_count), d = K(embedding_length); ctx = K(context_length)
when present and non-zero, else 1048576. The file builds no graph when it lacks `blk.0.ssm_dt.weight` or
`blk.0.ssm_d`, has `blk.0.ssm_dt_norm.weight`, or K(ssm.dt_b_c_rms) is true (Falcon-Mamba). H, KV and hd
are not read.

    E  = E0
    S  = [{"name": "conv", "kind": "recurrent", "op": "conv1d_causal", "dtype": "f32",
           "tensors": [{"name": "x", "shape": [kc - 1, di], "init": 0}]},
          {"name": "ssm", "kind": "recurrent", "op": "selective_scan", "dtype": "f32",
           "tensors": [{"name": "s", "shape": [di, ns], "init": 0}]}]
    N  = RMS(x, B attn_norm.weight) -> h
         LIN(h, B ssm_in.weight) -> xz
         split(xz) -> [xs, z] {sizes: [di, di]}
         conv1d_causal(xs) -> xs {state: "conv", weight: B ssm_conv1d.weight, bias: B ssm_conv1d.bias,
                                  channels: di, kernel: kc}
         silu(xs) -> xs
         LIN(xs, B ssm_x.weight) -> xdb
         split(xdb) -> [dt, bm, cm] {sizes: [r, ns, ns]}
         LIN(dt, B ssm_dt.weight, B ssm_dt.bias) -> dt
         selective_scan(xs, dt, bm, cm) -> y {state: "ssm", a: B ssm_a, d: B ssm_d, channels: di,
                                              d_state: ns, dt_softplus: true}
         silu(z) -> z
         mul(y, z) -> y
         LIN(y, B ssm_out.weight) -> o
         add(x, o) -> x
    HD = RMS(x, output_norm.weight) -> h;  HEAD

### D.7 The comparison

Rule 38 compares the file's embedded document with the graph built from the same file, each first
brought to a normal form; the file passes when the two normal forms are equal. The normal form of a
document, given the file's tensor names:

1. Remove every `_` member at any depth; keep only the top-level keys `model`, `embed`, `block` and
   `head` (so `ir_version`, `family_hint`, `tokenizer`, `inputs`, `outputs` and `modes` are not compared);
   remove `attention.softmax` and `moe_ffn.routing`.
2. Fill each optional attribute of section 3.5 that a node lacks with its default, except
   `rmsnorm.group`: `linear.bias` and `linear.tied_to` null, `scale.bias` null, `rope.freq_scale` 1,
   `rope.kv_heads` the node's `heads`, `rope.n_rot` its `head_dim`, `attention.mask` `"causal"`,
   `rmsnorm.weight_offset` 0, `depth_attend.eps` 1e-5, `conv1d_causal.bias` null, `selective_scan.d` null
   and `selective_scan.dt_softplus` true, and from revision 1.5 `split.groups` 1 and `rope.attn_factor` 1;
   a kv slot without `dtype` gets `"f32"`. Attributes whose default is "none" (`position_bias`,
   `relative_kv`, `state_weight`, and from revision 1.5 `softcap` and `freq_factors`) stay absent. From
   revision 1.6 a block node without `apply` gets true, a `moe_ffn` without `scoring` gets `"softmax"`, and
   `selection_bias` and `groups` stay absent.
3. A `linear` with a non-null `tied_to`: when the file has no tensor named by its `weight`, `weight`
   becomes the `tied_to` name; then `tied_to` becomes null. The name is tested as written, `{i}`
   included (Appendix D ties only the head's linear, whose names hold no `{i}`).
4. Each slot of `block.state` is named by its index (the k-th slot `slotk`), and every `state` attribute
   follows its slot.
5. Values are named by their writes: walking the embed's nodes, the block's nodes once and the head's
   nodes in order, each name in a node's `out` is a new value (numbered in that order), and each name in
   its `in` is the latest write of that name before the node (`tokens` and `pos` are themselves; a name
   with no earlier write is "unwritten" with its name). The inputs of `add` and of `mul` are compared as
   an unordered pair.
6. Every number (integer or not, outside names) is the f32 it rounds to (section 3.5), so 5 and 5.0 are
   equal; true, false, null and strings are themselves; a `per_layer` object is an object, never equal to
   a number.

Two normal forms are equal when they have the same members with equal values, lists (nodes, slots,
inputs, sizes, `per_layer` entries) of the same length equal in order.

### D.8 phi3 (revision 1.5)

eps = K(attention.layer_norm_rms_epsilon); theta = K(rope.freq_base), default 10000; n_rot =
K(rope.dimension_count), default hd; ff = K(feed_forward_length); w = K(attention.sliding_window), default 0.
The file has no `rope_freqs.weight`, K(rope.scaling.type) is absent or `"none"`, and the file has no
`blk.0.ffn_gate_inp.weight`; otherwise it builds no graph. LongRoPE applies when the file has
`rope_factors_short.weight` or `rope_factors_long.weight`: it must have both (one alone builds no graph),
F = `{"short": "rope_factors_short.weight", "long": "rope_factors_long.weight", "original_ctx":
K(rope.scaling.original_context_length)}` and m = K(rope.scaling.attn_factor), default 1 (0 reads 1).

    E  = E0
    S  = [KV(w)]
    N  = RMS(x, B attn_norm.weight) -> h
         when the file has blk.0.attn_qkv.weight:
             LIN(h, B attn_qkv.weight[, B attn_qkv.bias]) -> qkv
             split(qkv) -> [q, k, v] {sizes: [H hd, KV hd, KV hd]}
         otherwise:
             LIN(h, B attn_q.weight[, B attn_q.bias]) -> q;  LIN(h, B attn_k.weight[, B attn_k.bias]) -> k
             LIN(h, B attn_v.weight[, B attn_v.bias]) -> v
         ROPE(theta, "neox", n_rot), with freq_factors: F and attn_factor: m under LongRoPE (attn_factor only
             when m != 1)
         ATT(1/sqrt(hd), w)
         LIN(a, B attn_output.weight[, B attn_output.bias]) -> o
         add(x, o) -> x
         RMS(x, B ffn_norm.weight) -> h2
         LIN(h2, B ffn_up.weight) -> gu2;  split(gu2) -> [g, u] {sizes: [ff, ff]}
         silu(g) -> g;  mul(g, u) -> gu;  LIN(gu, B ffn_down.weight) -> f
         add(x, f) -> x
    HD = RMS(x, output_norm.weight) -> h;  HEAD

Each bracketed bias is present when the file has that tensor of layer 0 (`blk.0.attn_qkv.bias` and so on).

### D.9 gemma2 (revision 1.5)

eps = K(attention.layer_norm_rms_epsilon); n_rot = K(rope.dimension_count), default hd; theta =
K(rope.freq_base), default 10000; no RoPE scaling. sw = K(attention.sliding_window), default 4096 (0 reads
4096); c = K(attn_logit_softcapping), default 50 (0 reads 50); fc = K(final_logit_softcapping), default 30
(0 reads 30). Application i is local when i mod 2 = 0 and global otherwise; windows, written as a
`{"per_layer": [L entries]}` object, are sw for a local application and 0 for a global one. scale =
1/sqrt(d / H) when L = 46 (the 27B model; the binary64 quotient), else 1/sqrt(hd). C is the shortest decimal
that reads back as the f32 nearest fc (Python's repr of that float, `30.0` for 30).

    E  = E0 + [affine(x) -> x {scale: sqrt(d), shift: 0}]
    S  = [KV(windows)]
    N  = RMS(x, B attn_norm.weight) -> h
         LIN(h, B attn_q.weight) -> q;  LIN(h, B attn_k.weight) -> k;  LIN(h, B attn_v.weight) -> v
         ROPE(theta, "neox", n_rot)
         ATT(scale, windows), with softcap: c
         LIN(a, B attn_output.weight) -> o
         RMS(o, B post_attention_norm.weight) -> o
         add(x, o) -> x
         RMS(x, B ffn_norm.weight) -> h2
         LIN(h2, B ffn_gate.weight) -> g;  LIN(h2, B ffn_up.weight) -> u
         gelu(g) -> g {approx: "tanh"};  mul(g, u) -> gu
         LIN(gu, B ffn_down.weight) -> f
         RMS(f, B post_ffw_norm.weight) -> f
         add(x, f) -> x
    HD = RMS(x, output_norm.weight) -> h;  HEAD written to z (its out is "z", not "logits");
         expr(z) -> logits {expr: "tanh(x / C) * C"}

### D.10 qwen2moe, qwen3moe (revision 1.5)

D.3's graph with qwen2moe read as qwen2 and qwen3moe as qwen3 (pairing `"neox"`, w = 0, biases and q/k norms
by the tensors the file has), with FFN replaced by

    moe_ffn(h2) -> f {router: B ffn_gate_inp.weight,
                      experts: {"gate": B ffn_gate_exps.weight, "up": B ffn_up_exps.weight,
                                "down": B ffn_down_exps.weight},
                      n_experts: K(expert_count), top_k: K(expert_used_count),
                      gate_norm: G, activation: "silu"}

G `"raw"` for qwen2moe and `"renorm"` for qwen3moe, followed, for qwen2moe when the file has
`blk.0.ffn_gate_inp_shexp.weight`, by

    LIN(h2, B ffn_gate_shexp.weight) -> sg;  LIN(h2, B ffn_up_shexp.weight) -> su
    silu(sg) -> sg;  mul(sg, su) -> sgu;  LIN(sgu, B ffn_down_shexp.weight) -> sf
    LIN(h2, B ffn_gate_inp_shexp.weight) -> sgate;  sigmoid(sgate) -> sgate
    mul_col(sf, sgate) -> sf;  add(f, sf) -> f

### D.11 deepseek2 (revision 1.5; leading dense blocks, V3's router and YaRN from revision 1.6)

d = K(embedding_length), L = K(block_count), H = K(attention.head_count), ctx = K(context_length), eps =
K(attention.layer_norm_rms_epsilon); dr = K(rope.dimension_count); dn = K(attention.key_length_mla) - dr;
dv = K(attention.value_length_mla); r = K(attention.kv_lora_rank); theta = K(rope.freq_base), default 10000.
The file builds no graph when it lacks `blk.0.attn_k_b.weight` (or the key `attention.key_length_mla`), when
K(expert_gating_func), default 1 (0 reads 1), is neither 1 nor 2, or when it has `rope_freqs.weight` or a
K(rope.scaling.type) other than absent, `"none"` or `"yarn"`. k = K(leading_dense_block_count), default 0, at
most L; M = `blk.{k}.` (the first mixture application). G is `"renorm"` when K(expert_weights_norm), default
false, is true, else `"raw"`; s = K(expert_weights_scale), default 1 (0 reads 1). Under YaRN: c =
K(rope.scaling.factor), o = K(rope.scaling.original_context_length), F = the list of dr / 2 values of
`tools/gguf_ir.py`'s `yarn_factors(dr, theta, c, o)` (section 5.14's formula with beta_fast 32 and beta_slow 1,
in binary64, each rounded to f32) and m = 1 + K(rope.scaling.yarn_log_multiplier), default 0, times ln(c);
without it no F and m = 1.

    E  = E0
    S  = [{"name": "kv", "kind": "kv", "heads": H, "kv_heads": 1, "head_dim": r + dr, "max_len": ctx,
           "window": 0, "sinks": 0, "dtype": "f32"}]
    N  = RMS(x, B attn_norm.weight) -> h
         when the file has blk.0.attn_q_a.weight:
             LIN(h, B attn_q_a.weight) -> qa;  RMS(qa, B attn_q_a_norm.weight) -> qa
             LIN(qa, B attn_q_b.weight) -> q
         otherwise:  LIN(h, B attn_q.weight) -> q
         split(q) -> [qn, qr] {sizes: [dn, dr], groups: H}
         LIN(h, B attn_kv_a_mqa.weight) -> ckr
         split(ckr) -> [c, kr] {sizes: [r, dr]}
         RMS(c, B attn_kv_a_norm.weight) -> c
         rope(qr, kr, pos) -> [qr, kr] {heads: H, kv_heads: 1, head_dim: dr, n_rot: dr, theta: theta,
                                        rope_heads: "all", pairing: "interleaved"[, freq_factors: F]}
         mla_attention(qn, qr, c, kr) -> a {state: "kv", heads: H, kv_rank: r, qk_nope: dn, qk_rope: dr,
                                            v_head_dim: dv, k_b: B attn_k_b.weight, v_b: B attn_v_b.weight,
                                            scale: m^2 / sqrt(dn + dr), causal: true, window: 0, sinks: 0}
         LIN(a, B attn_output.weight) -> o
         add(x, o) -> x
         RMS(x, B ffn_norm.weight) -> h2
         DENSE   (when k > 0)
         MOE     (when k < L)
         add(x, f) -> x
    HD = RMS(x, output_norm.weight) -> h;  HEAD

with DENSE = `LIN(h2, B ffn_gate.weight) -> g; LIN(h2, B ffn_up.weight) -> u; silu(g) -> g; mul(g, u) -> gu;
LIN(gu, B ffn_down.weight) -> f` and MOE =

    moe_ffn(h2) -> f {D.10's attributes, gate_norm: G[, scoring: "sigmoid"][, selection_bias: B exp_probs_b.bias]
                      [, groups: {"n": K(expert_group_count), "top": K(expert_group_used_count)}]}
    affine(f) -> f {scale: s, shift: 0}                                    (only when s != 1)
    when the file has M ffn_gate_shexp.weight:
        LIN(h2, B ffn_gate_shexp.weight) -> sg;  LIN(h2, B ffn_up_shexp.weight) -> su
        silu(sg) -> sg;  mul(sg, su) -> sgu;  LIN(sgu, B ffn_down_shexp.weight) -> sf
        add(f, sf) -> f

`scoring` is written when K(expert_gating_func) is 2; `selection_bias` when the file has M exp_probs_b.bias;
`groups` when K(expert_group_count), default 1, is above 1 (`top` K(expert_group_used_count), default 1). When
0 < k < L, every DENSE node carries `apply: {"per_layer": [i < k for i in 0 .. L - 1]}` and every MOE node
`apply: {"per_layer": [i >= k ...]}`; otherwise no node carries `apply`.
