# The paperkiln IR, version 1

*Normative specification of the paperkiln model IR: a versioned JSON graph of a language model that an
engine interprets, rather than recognising a fixed architecture. IR version 1, specification revision
1.0 (25 October 2026).*

- Schema: `spec/ir.schema.json` (JSON Schema 2020-12, the document's structure).
- Validator: `tools/ir_validate.py` (the schema plus every refusal rule in section 10).
- Reference interpreter: `tools/ir_numpy.py`, with `tools/ir_expr.py` for `expr`.
- Conformance suite: `spec/conformance/`, run by `tools/ir_conformance.py` (section 11).
- Implementations: paperkiln (exporter, trainer and `mtstudio`; CPU and CUDA) and ember.cpp
  (`graph_ir.cpp`, `graph_ir_cuda.cu`; CPU, CUDA, WebAssembly).
- Contracts between the repositories, the chat template record and the log of gates:
  `LOCAL_LLM_STACK.md` at the workspace root (sections 3 to 5 and 10).

The C++ API that writes documents is `include/microtorch/ir.hpp`. That header documents functions,
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

## 0. Status

This document is version 1 of the IR. It states the semantics that `include/microtorch/ir.hpp`'s
header comment stated under the number 0, reorganised into profiles, with the numerics made
consistent with what the reference build computes (section 4 and Appendix A). Every IR document
written so far carries `"ir_version": 0` and is a version 1 document (section 2.1). Exporters write 0
today and may write 1 under the rule of section 2.3.

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
means a consumer must reject the document at load, naming the rule (section 10). Section 9 and the
appendices are informative.

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
- **A major version** (`ir_version` + 1) is needed for any change that alters the meaning of a
  document that was valid before: a changed default, a changed evaluation order of an EXACT op, a
  changed tie rule, a removed op or attribute, or a changed name rule.

### 2.3 Unknown versions

An engine refuses a document whose `ir_version` it does not implement, at load, naming the
version and the versions it reads. An engine that implements version 1 reads 0 and 1.

An exporter writes a version number only once every engine it targets reads it.
- ember.cpp reads 0 and 1 as the same document and refuses any other version from ce41ffb on;
  earlier ember commits read 0 only (`IR version N is not supported (ember reads 0)`).
- paperkiln's exporters (`ir::kVersion`) and `tools/gguf_ir.py` (`IR_VERSION`) write 0 today. They
  may write 1 now that ember ce41ffb reads it; ember's C++ builder follows.
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
| `core` | the document, tensors, per-layer attributes, prefill and decode, numerics and classes, the ops of section 5 except those marked otherwise, the kv slot with window and sinks | 3, 4, 5, 6.1, 6.2 |
| `state` | recurrent slots and ops (`linear_attention`, `ssm_scan`), per-turn state tokens (`state_weight`, `mask: turn_state`, `evict: turn_state`), prefix reuse, rewind and snapshots | 6 |
| `recursion` | `weight_index`, `tensor_index`, `depth_route`, `kv_rows: active`, `kv_from` (reserved) | 7 |
| `chat` | the template record and its resolution; the state-token template contract | 8 |
| `thirdparty` | the attributes and ops added for graphs built from llama.cpp metadata: grouped-query attention, `rope` `neox` / `n_rot` / `freq_scale` / `kv_heads`, `rmsnorm` `group` / `weight_offset`, `conv1d_causal`, `selective_scan` | 5.6, 5.14, 5.15, 6.6, 6.7, 9.3 |

Profiles are not declared inside the document (an engine infers them from the ops and keys it
meets). Each engine's adapter for the conformance suite states its profiles (section 11.3).

### 2.5 Change control

A change reaches this document in this order: draft text reviewed by the other engine; the numpy
reference; both engines passing the change's gates; then this document and Appendix B, in the
commit that lands it. A draft is never normative.

## 3. The document (core)

### 3.1 Container

- In a GGUF, the compact JSON dump of the document is the string key `paperkiln.ir` and the version
  is the uint32 key `paperkiln.ir_version`, both written after every other metadata key.
- A sidecar `<stem>.ir.json` beside the GGUF may hold the same document. When present, it must be
  the embedded document; a consumer that reads sidecars refuses one that differs.
- A third-party GGUF without `paperkiln.ir` gets a document built from its `general.architecture`
  and metadata (section 9.3).

### 3.2 Top-level keys

A document is a JSON object with these keys:

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
  `logits`, [T, vocab].
- `modes`: prose describing prefill and decode; informative.

### 3.5 Sections and nodes

The graph is three node lists: `embed` (run once), `block` (run `block.repeat` times) and `head`
(run once). A node is an object:
- `op`: the op name (sections 5 to 7). An op this specification does not define is refused.
- `in`: the names of the values it reads, in order.
- `out`: the name it writes (a list of names for `split` and `rope`).
- the op's attributes. A node in the block names its slot, if it has one, in `state`.

Nodes run in order. A later write to a name replaces the earlier value. The embed section reads
`tokens` and must write `x`; each application reads and writes `x`; the head reads `x` and writes
`logits`.

- **Required attributes.** The attributes each op's section names without a default are required,
  and `spec/ir.schema.json` lists them (`required` of each `op_*` definition). They have no default:
  a node that lacks one is refused, whatever value an engine would assume.
- **Optional attributes and their defaults:** `linear.bias` and `scale.bias` null; `linear.tied_to`
  none; `embedding.state_weight` none; `rmsnorm.group` the row width and `rmsnorm.weight_offset` 0;
  `rope.kv_heads` = heads, `rope.n_rot` = head_dim and `rope.freq_scale` 1; `attention.mask`
  `"causal"`; `conv1d_causal.bias` none; `selective_scan.d` none and `selective_scan.dt_softplus`
  true; a kv slot's `dtype` `"f32"`, `evict` and `kv_rows` absent.
- **The graph inputs.** `tokens` is read by `embedding` nodes only, and `pos` only as `rope`'s third
  input and `pos_embedding`'s second; no node writes either name. Positions are always the absolute
  positions of section 3.4.
- **Names.** A tensor name holds `{i}` at most once (section 3.7).

### 3.6 Values

Tensors flow as [T, width] row-major matrices of f32: row t is token t. A `list` value (section
5.17) holds tensors in order; it is a value like a tensor (named by `out`, read by `in`), never a
GGUF tensor and never state.

### 3.7 Tensors and names

- `weight` (and `bias`, `router`, `experts`, `state_weight` and the tensor attributes of
  `ssm_scan`, `conv1d_causal` and `selective_scan`) name GGUF tensors.
- Layout: a matrix is [out, in] row-major, as GGUF and llama.cpp store it (ggml `ne0` = in).
- Storage: each tensor's storage type is its GGUF type (f32, f16, q8_0, ...). Engines compute in
  `model.dtype`.
- `"{i}"` in a name in the block is replaced by the block index i (by m_i under `weight_index`,
  section 7.1; by the application index under `tensor_index`, section 7.2). A name holds `{i}` at
  most once, and only in the block: outside the block it has no value, and a name holding it there
  is refused.
- Every tensor a document names must be a tensor of the file, except a `linear` weight with
  `tied_to` (section 5.3).

### 3.8 Per-layer attributes

Any attribute of a block node, and any field of a block state slot, may be
`{"per_layer": [v_0, ..., v_{L-1}]}` with L = `block.repeat`: application i reads v_i, except the
structural keys: a node's `op`, `in`, `out`, `state`, `tensor_index`, `sizes`, `expr` and its
tensor names (`weight`, `bias`, `router`, `experts`, `state_weight`, `tied_to`, `a`, `b`, `c`, `d`),
and a slot's `name`, `kind`, `op`, `dtype`, `tensors`, `evict` and `kv_rows` (`kv_from`, reserved, is
per-layer by definition). An attribute not given per layer is the same in every application. (Examples: gemma3's attention window and kv window, 512
on local layers and 0 on global ones; RoPE theta 10000 / 1e6; `depth_route`'s `step`; the injection
`affine`'s `scale`.) A `per_layer` list whose length is not `block.repeat` is refused, as is a
`per_layer` value outside the block.

### 3.9 Prefill and decode

Every op has the same semantics on T rows (prefill) and on one row (decode, T = 1). In decode,
attention also reads and appends to its kv slot (section 6.2), and a recurrent op reads and replaces
its recurrent slot (section 6.3).

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
it on the last row only). A weight or bias whose length is not x's width is refused at load,
naming the tensor. EXACT (against the same order of operations).

### 5.5 split

`in [x]`, `sizes [s0, s1, ...]` summing to x's width, `out [o0, o1, ...]`: o_k = the s_k consecutive
columns of x after the first s0 + ... + s_{k-1} (a fused qkv projection read as q, k, v). Copies.
EXACT.

### 5.6 rmsnorm

x / sqrt(mean(x^2) + eps) * weight, over each row. ROUNDING.

*thirdparty profile:*
- `group` g (default the row width): each consecutive g columns are normalised on their own, with
  weight [g] shared by every group (per-head q/k norm: qwen3, gemma3). g must be positive and divide
  the width.
- `weight_offset` c (default 0): the weight is read as w + c, rounded to f32 once at load (a
  load-time transform; Gemma's (1 + w), needed only when a file holds HF's raw w: llama.cpp's
  convert already stores 1 + w).

### 5.7 layernorm

(x - mean(x)) / sqrt(var(x) + eps) * weight + bias over each row, var the biased (divide by width)
mean of squared deviations; weight and bias [d]. ROUNDING.

### 5.8 gelu

`approx` `"tanh"`: 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))) (GPT-2's, and what paperkiln
trains with); `"none"`: the exact 0.5 x (1 + erf(x / sqrt(2))). An engine must not substitute one
for the other. Another value is refused. ROUNDING.

### 5.9 silu, sigmoid, relu

- `silu`: x * sigmoid(x). ROUNDING.
- `sigmoid`: 1 / (1 + exp(-x)) (`ops::sigmoid`; highway gates, the SRD router). ROUNDING.
- `relu`: x where x > 0, else +0, so relu(-0) = +0 and relu(NaN) = +0 (`ops::relu`'s
  `std::max(0.0f, x)`, ember's, and `expr`'s `relu`). EXACT.

### 5.10 add, mul, sub

Elementwise over two inputs of one shape; `sub` is `in [a, b]`: a - b. EXACT.

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
- `rope_heads`: `"all"` (every head) or `"first"` (head 0 only, legacy runs).
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

**The angle table**, in f32: for pair j, freq_j = 1 / powf(theta, f32(2j) / f32(n_rot)) (the
division rounded, then powf, then the reciprocal), and angle(pos, j) = (f32(pos) * freq_scale) *
freq_j, each product rounded; the table holds cosf(angle) and sinf(angle). This is what paperkiln
(`ops::apply_rope`, freq_scale 1) and ember (`graph_ir.cpp`) compute. The numpy reference computes
the angle in double and rounds cos and sin to f32, within ROUNDING of the table (Q3).

Class: the rotation is EXACT given the table. The table is ROUNDING across platforms (cosf, sinf and
powf differ by an ulp between libm implementations), so two builds agree bit for bit when their libm
agrees.

### 5.15 attention

softmax(q k^T * scale) v per head, over the kv slot the node names in `state` (section 6.2).
Attributes `heads`, `kv_heads`, `head_dim`, `scale`, `causal`, `window` (0 = unbounded), `sinks`,
`mask` (section 6.8.2; absent = `"causal"`). `causal` is true (section 12, Q6). `in [q, k, v]`, q
[T, heads * head_dim]; head h at columns h * head_dim ... ROUNDING.

- **Masking.** Masked keys contribute exactly 0, and the row maximum is taken over visible keys only.
- **Sliding window with sinks** (`ops::swa_attention`; window >= 1). With p the query's position and
  j a key's (both counted from the first token of the slot, 0-based), j is visible iff

      j <= p  and  (j < sinks  or  j >= p - window + 1)

  that is, the first `sinks` positions plus the last `window` positions up to and including p.
  Sinks are positions, not parameters (no tensor). The softmax runs over that union, each key once
  (the sink range is cut at the window's start when they meet). paperkiln sums sinks first, then the
  window, each in ascending j (the reference order, section 4.4). window >= p + 1 with sinks 0 is
  plain causal attention.
- **Decode.** The slot must keep the k/v rows of positions 0 .. sinks-1 and of the last `window`
  positions (p - window + 1 .. p, the new token included). Anything older is never read again, so
  sinks + window rows per kv head suffice. Positions stay absolute (decode pos = tokens seen, not
  rows kept); the sinks are the first tokens of the prompt. Under the context contract no position
  reaches model.ctx.
- **Consistency with the slot.** A kv slot declares the same window and sinks as the attention
  nodes that read it, and the same `kv_heads` and `head_dim`. A node whose window, sinks,
  `kv_heads` or `head_dim` differs from its slot's is refused.
- *thirdparty profile:* **grouped-query attention**, kv_heads < heads: query head h reads k/v head
  h / (heads / kv_heads); k and v are [T, kv_heads * head_dim]. heads % kv_heads == 0.
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
- `activation` `"silu"`: expert_e(x) = down_e(silu(gate_e x) * up_e x).
- out = the sum over the selected e of gate_e * expert_e(x), accumulated in ascending expert index.
- `routing` (a string) is an informative attribute; it carries no semantics.

ROUNDING.

### 5.17 list, append

- `list`: `in [x0, x1, ...]`: a list value holding those tensors in order. EXACT.
- `append`: `in [L, x]`: the list L with x added at the end. EXACT.

### 5.18 depth_attend

Attention over depth (Attention Residuals, block form; `nn::AttnResStack::depth_attend` and
`ops::depth_attend`): `in [L, x1, ...]` -> [T, d], `weight` the pseudo-query [d], `eps`. The
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
           (ops::rmsnorm with unit weight; eps 1e-5)
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
  (`where(c, ...)` on constants included), and a result without inputs fills [T, w]. The rows are
  independent, so decode (T = 1) is prefill's last row.
- **Gradients** (for training engines) come from the primitives: max, min and where pass the
  gradient to the chosen operand, none to where's condition; rowmax to the row's first largest
  entry; abs and relu give 0 at 0. An engine needs only the forward.
- An expression that does not parse is refused at load, naming the column. Every engine accepts
  expressions nested up to 200 deep (the reference parser's limit, `MAX_DEPTH` in
  `tools/ir_expr.py`, which ember's vendored parser shares); a deeper one may be refused. An engine
  that cannot run an expression refuses it at load, naming the function.

ROUNDING (libm transcendentals differ by an ulp between platforms).

## 6. The state profile

### 6.1 Slots

`block.state` lists the slots. There is one instance of each slot per application (per layer) and
per sequence; a node names its slot in `state`. A block may declare several slots (srd: one kv and
one recurrent). Every slot is read by at least one node, a slot name is declared once, and a node
that names a slot names a declared slot of the kind its op needs. A kv slot is read by exactly one
attention node (two readers would append twice per application, which is not defined); a block may
still declare several kv slots. Nodes outside the block have no state, so a stateful op outside
the block is refused.

### 6.2 kv (core)

`{name, kind "kv", heads, kv_heads, head_dim, max_len, window, sinks, dtype}`: the k and v rows of
the tokens seen so far, `dtype` `"f32"`. window 0 and sinks 0 keep every row up to `max_len`, the
slot's capacity in positions: a position >= max_len is refused at run time (exporters write
max_len = model.ctx; an engine may hold fewer positions by choice and then refuses earlier).
Optional keys: `evict` (section 6.8.3), `kv_rows` (section 7.5) and `kv_from` (section 7.6). The
attention nodes that read a slot declare its window, sinks, kv_heads and head_dim (section 5.15).

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
  `selective_scan` (s). A slot whose op is not one of these, whose dtype is not f32, or whose tensors
  do not fit its op (sections 6.4 to 6.7) is refused.

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
- **kv with kv_rows active:** keep the rows whose positions are < n, per application, with each
  application's written-rows mask truncated with them (section 7.5).
- **recurrent:** a recurrence cannot drop tokens. Prefix reuse keeps a snapshot of the slot's
  tensors taken after the prefix's last token; rewinding means restoring a snapshot or running again
  from the start. A snapshot is the slot's tensors, nothing else.

## 7. The recursion profile

### 7.1 weight_index

A key of the block itself, not of a node: `{"per_layer": [m_0, ..., m_{L-1}]}`, L = block.repeat,
integers m_i >= 0.

- Application i of the block reads every tensor name it holds (`weight`, `bias`, `router`,
  `experts`, ssm_scan's `a b c d`, ...) with `"{i}"` replaced by m_i, i.e. `blk.{m_i}.*`, so several
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
  every step has the same number of applications. `n_rec` R >= 1.
- **(a) Routing**, at the first application of step 0: logits = x W^T, W = `weight` [R, d] (a matmul
  reduction, as linear), [T, R]. Each token's depth n_t = 1 + j*, j* the index of the largest logit,
  ties to the LOWER index; p = softmax over the row (e_j = exp(l_j - max), z summed in ascending j,
  p_j = e_j / z); its gate g_t = p_t[n_t - 1]. The depth and gate hold for the rest of the forward;
  `out` receives the depths [T, 1] (the depth trace, 1 .. R).
- **(b)** At the first application of step k the step's input x_in = x is kept, and the active
  tokens are those with n_t > k. Every application of step k runs the block's other nodes on the
  active rows only: the rows of active tokens, in order, with their absolute positions (pos), so for
  those nodes T is the number of active tokens. Rows of inactive tokens are not computed and stay as
  they are.
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
positions <= its own (the prefix's rows and the current call's earlier active rows).

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

### 8.1 The template record and resolution

The chat template travels in the model file as data: the GGUF string key `paperkiln.chat_template`
holds the record as JSON. The record's fields, its rendering, `paperkiln/1`, the third-party
records, resolution (`resolve()`, the same in every engine), export and the gates are specified in
`LOCAL_LLM_STACK.md` section 3 point 3 (3a to 3h), with `tools/chat_template.py` and
`include/microtorch/chat_template.hpp` as the implementations. They are part of this profile by
reference and are not restated here.

### 8.2 The state-token template contract

`paperkiln.state_tokens.k` = k > 0 (GGUF metadata, uint32, absent = 0; `LOCAL_LLM_STACK.md` section
3 point 3f) declares the state ids S_0 .. S_{k-1} (section 6.8).

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
  `nn::state_ids_error`, `ops::turn_state_codes` (`llama.hpp`, `ops.hpp`); `tools/state_tokens.py`.

## 9. Families (informative)

### 9.1 family_hint

`family_hint` labels the graph; engines never derive semantics from it (an engine may use it, with a
structural match, to pick a tuned path that computes the same function). For paperkiln's own files
it equals the family below; for third-party files it equals `general.architecture`. "paperkiln" is
the GGUF architecture of a file no llama.cpp graph computes; its metadata keys are
`paperkiln.{context_length, embedding_length, block_count, vocab_size, attention.head_count,
feed_forward_length}`.

### 9.2 paperkiln families

A tensor that is the same thing as a gpt2 tensor keeps llama.cpp's gpt2 name: `token_embd`,
`position_embd`, `blk.{i}.attn_norm`, `attn_qkv`, `attn_output`, `ffn_norm`, `ffn_up`, `ffn_down`,
`output_norm`, `output` (each `.weight`, and `.bias` where the module has one); new tensors are
`blk.{i}.<name>.weight` (and `.bias`). Worked documents: `ir/*.ir.json`.

- **gpt2, gpt2_swa**: as above (fused qkv split q | k | v, tanh GELU, learned positions);
  gpt2_swa adds a window and sinks.
- **llama**: pre-RMSNorm (eps 1e-5), interleaved RoPE on every head, SwiGLU, no biases, tied output
  (`ir/chat_sftfinal.ir.json`); with MoE, `moe_ffn` in place of the SwiGLU (`ir/moe_tiny.ir.json`).
- **kimi**: gpt2 with `linear_attention` (slot "lin": num, den [heads, head_dim]) in place of
  attention; the gpt2 tensor set exactly (architecture gpt2: llama.cpp loads it but computes softmax
  attention, so only the IR reproduces the model).
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
  b, as paperkiln's mul_row then add_bias. Each sublayer f combines with its input stream x as
  residual x + f, plain f, or highway x + sigmoid(W x + b) * (f - x) with W, b =
  blk.{i}.highway_attn / highway_ffn [d, d] + bias, nodes linear, sigmoid, sub, mul, add. The GPT-2
  flavour (layernorm, gelu, learned, residual) exports as gpt2 / gpt2_swa; plain with that flavour
  is gpt2 tensors (gpt2 architecture, IR-only semantics); others paperkiln.
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
  (section 7.2). Architecture "paperkiln". Reference: paperkiln c48ca962; ember's ar3 gate (CPU),
  which has no recursion and so does not cover `tensor_index`.
- **llama_state**: the llama block (tensor names and nodes as llama) with an embedding state_weight
  (state_embd.weight [k, d]) and the state-token template; the attention mask is turn_state with kv
  evict turn_state (S-gist), or absent (causal: S-full, the insertion alone). Architecture
  "paperkiln" (llama.cpp's graph has no state embedding). Under fixed recursion the hint stays
  llama_recursive and the same keys carry the rest; with Mixture-of-Recursions it is not defined
  (paperkiln refuses it, and a document combining depth_route with mask turn_state is refused).

### 9.3 Third-party families

An engine builds the document of a llama.cpp GGUF without `paperkiln.ir` from
`general.architecture` and its metadata, as `tools/gguf_ir.py` does (the reference; its docstring
has each graph; `tools/ir_numpy.py` runs it). family_hint = general.architecture; llama.cpp's
tensor names.

- **llama, mistral**: pre-RMSNorm, interleaved RoPE, grouped-query attention, SwiGLU;
  `<arch>.attention.sliding_window` w sets window w, sinks 0 (HF Mistral's mask; llama.cpp b6000
  ignores it).
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

## 10. Validation

### 10.1 What a consumer refuses

A consumer refuses a document at load, naming the rule, when any of these holds. Each rule names
its section; `tools/ir_validate.py` reports every violation as `IR_SPEC <section>: <message>`.

Document (section 3):
1. It is not a JSON object matching `spec/ir.schema.json` (types, required keys, closed key sets).
2. `ir_version` is not a version the consumer reads (2.3).
3. `model.d`, `model.vocab`, `model.ctx` or `model.n_layers` is not a positive integer, or
   `block.repeat` != `model.n_layers` (3.3).
4. A node's `op` is not an op of this specification, or a node carries an attribute its op does
   not define (3.5; `_` keys excepted).
5. An attribute value is outside its op's closed set: `gelu.approx`, `rope.pairing`,
   `rope.rope_heads`, `attention.mask`, `moe_ffn.gate_norm`, `moe_ffn.activation`,
   `linear_attention.feature_map`, `depth_route.route`, `depth_route.gate`, `kv.evict`,
   `kv.kv_rows`, `tensor_index` (sections 5 to 7).
6. A `per_layer` list outside the block, on a structural key, or of a length other than
   `block.repeat` (3.8).
7. A node's arity is wrong: one output except `split` and `rope`; `sub`, `add`, `mul`,
   `mul_col`, `append` and `pos_embedding` read two values; `embedding`, `relu`, `sigmoid`,
   `silu`, `gelu`, `affine`, `rms_row`, `scale`, `rmsnorm`, `layernorm`, `linear`, `moe_ffn`,
   `ssm_scan`, `conv1d_causal` and `depth_route` read one; `attention` and `linear_attention`
   read [q, k, v]; `rope` reads [q, k, pos] and writes two; `selective_scan` reads four; `list`,
   `depth_attend` and `expr` read at least one, `expr` at most 23 (sections 5 to 7).
8. A required attribute is missing: `scale.weight`, `affine.scale` and `affine.shift`,
   `rms_row.eps`, `expr.expr` (a string), and the attributes each op's section lists.
9. An `expr` does not parse (5.19).
10. `split.sizes` has a different count than `split.out` (5.5); `rmsnorm.group` is not positive
    (5.6); `rope.n_rot` is not even and in 1 .. head_dim, or `rope.freq_scale` is not positive
    (5.14); `attention.heads` is not a multiple of `kv_heads`, or window or sinks is negative (5.15);
    `moe_ffn.top_k` is not in 1 .. n_experts (5.16); `conv1d_causal` has channels < 1 or kernel < 2
    (6.6).

State (section 6):
11. A slot's kind is not kv or recurrent; a slot is declared twice or read by no node; a stateful
    node names no declared slot of its kind, or a stateful op lies outside the block (6.1).
12. A kv slot lacks heads, kv_heads, head_dim, max_len, window or sinks; an attention node's
    window, sinks, kv_heads or head_dim differs from its slot's (5.15, 6.2).
13. A recurrent slot's op is not a recurrent op, or differs from the op reading it; its dtype is not
    f32; its tensors are empty, have a non-positive shape or an init other than 0, or do not fit the
    op (6.3 to 6.7).
14. `attention.causal` is not true; `linear_attention.causal` is not true (5.15, 6.4).
15. `mask: turn_state` with a window or sinks (in any application), with a windowed kv slot, or
    without an embedding `state_weight` (6.8.2).
16. `evict: turn_state` with an attention node reading the slot whose mask is not turn_state
    (6.8.3).
17. `state_weight` outside the embed section, on more than one embedding, or with `{i}` in its name
    (6.8.1, Q8).
17a. A name holds `{i}` outside the block, or more than once (3.7); a node reads `tokens` or `pos`
    other than as section 3.5 allows, or writes either (3.5); a kv slot is read by more than one
    attention node (6.1); a node reads a `depth_route`'s `out` (7.4).

Recursion (section 7):
18. `weight_index` is not `{per_layer: [block.repeat integers >= 0]}`, or leaves a stored layer
    0 .. max unused (7.1).
19. `tensor_index` in a block without `weight_index`, or with a value other than `"application"`
    (7.2).
20. `depth_route` is not the block's first node, occurs more than once or outside the block; its
    `n_rec` is not an integer >= 1; its `step` is not a `per_layer` list of block.repeat integers
    forming the consecutive, ascending, equal-length steps 0 .. R-1 with -1 elsewhere; its `route`
    is not token_choice, its gate not score or none; its weight holds `{`; it has no `out` (7.4,
    7.7).
21. `depth_route` with a kv slot whose `kv_rows` is not `"active"`, or with no kv slot (7.5).
22. A kv slot carries `kv_from` (7.6).
23. `depth_route` together with `mask: turn_state` (9.2, llama_state), or with `list`, `append` or
    `depth_attend` nodes (7.4).

Container (a GGUF with `paperkiln.ir`, sections 3.1, 6.8, 7, 8.2):
24. `paperkiln.ir_version` is missing, not a uint32, or differs from the document's `ir_version`.
25. A tensor the document names is not a tensor of the file (a `linear` weight passes through its
    `tied_to`). The validator checks exports, so it reports a missing S = 1 `depth_attend`
    pseudo-query too, which a consumer may accept (5.18).
26. The `depth_route` router's ne is not {d, n_rec}.
27. Under `weight_index`: the architecture is not `paperkiln`, or `<arch>.block_count` is missing or
    differs from the number of stored layers.
28. State tokens: `paperkiln.state_tokens.k` without an embedding `state_weight`, or the reverse;
    `state_weight`'s ne is not {d, k}; the architecture is not `paperkiln`; `tokenizer.specials`
    lacks eot, user or assistant, or names ids whose tokens are not the template's markers; a
    declared `paperkiln.chat_template` other than `paperkiln/1`.
29. A sidecar `<stem>.ir.json` that differs from the embedded document (for a consumer that reads
    sidecars).

### 10.2 The schema and the validator

`spec/ir.schema.json` is the structural half (rules 1, 4 to 8 in part). `tools/ir_validate.py`
runs it and every other rule:

    python tools/ir_validate.py doc.ir.json [model.gguf ...]    # documents and GGUFs
    python tools/ir_validate.py --corpus                        # every stored document

It is standard library only. It evaluates the schema with its own evaluator for the keywords the
schema uses, and with `jsonschema` too when that package is installed (`tests/test_ir_validate.py`
checks they agree). `tools/ir_check.py` calls it for the semantic rules and keeps the container
checks.

### 10.3 Engine capability refusals

An engine may refuse a valid document it cannot run: an op without a kernel on a backend, or an
attribute combination it does not implement. Such a refusal names the op or attribute and the
engine; it is not a validation failure, and the validator does not report it. Each engine's
capabilities are listed in Appendix C, outside the normative text.

## 11. Conformance

### 11.1 Layout

`spec/conformance/` holds the suite; `spec/conformance/manifest.json` lists every case with its
kind, profile, class and tolerance:

- `ops/`: per-op fixtures: inputs, attributes, expected outputs (f32 bit patterns), class and
  tolerance.
- `state/`: state traces: token-by-token decode against prefill; eviction on and off; rewind into an
  evicted turn; MoR depth traces; recurrent state snapshots.
- `models/`: whole models: a small document with F32 weights in a GGUF, prompts, expected greedy ids
  and log-probabilities (and depth traces under depth_route).
- `refuse/`: documents (and GGUF metadata) every consumer must refuse, each with the rule it breaks.

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
  are integer lists.
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
- **refusal** (`refuse/<name>.json`): `{"doc"?: <document>, "gguf": "<path in the suite>", "rule":
  <10.1 rule number>, "section": "<section>"}`. With `doc`, the engine loads the GGUF (a suite
  model) with that document in place of its own (as ember's `EMBER_IR` does); without it, the GGUF
  (`refuse/<name>.gguf`) breaks a container rule. A consumer passes when it refuses; the reference
  adapter must also report a violation under `section`.

### 11.3 The adapter contract

`tools/ir_conformance.py` drives an engine through an adapter. The adapters shipped are `ir_numpy`
(the reference) and `mtstudio` (paperkiln's C++ engine, through `mtstudio score` and `expr_eval`).
Any other engine joins with a program that speaks this protocol; the runner starts it once with
`--adapter cmd:"<program and arguments>"`.

**Transport.** JSON Lines: one request per line on the program's stdin, one reply per line on its
stdout, in request order; stderr is free for logs. A request is `{"id": int, "method": str,
"params": {...}}`. A reply is one of

    {"id": n, "result": {...}}     the method ran
    {"id": n, "error": "message"}  the method failed (the case fails)
    {"id": n, "skip": "reason"}    not implemented (the case is reported as skipped, never passed)

**Values.** A tensor is `{"shape": [...], "f32": "<hex>"}` of any rank (a recurrent `s` [n] is
1-D; section 11.2: 8 hex digits per value, the binary32 bit pattern, row-major). Ids and positions
are JSON integer lists.

**Skips.** A `skip` covers the whole call: there is no per-report skip. The engine decides it before
running anything, so a skipped call leaves the session as it was.

**Sessions.** A session is one sequence on one loaded model. Its length `len` is the number of
tokens seen; its state is whatever the engine keeps (kv rows, recurrent tensors, MoR row sets). A
session owns the ids it has seen, so it can re-prefill after a rewind (section 6.9).

**Methods.**

| Method | params | result |
|---|---|---|
| `hello` | `{}` | `{"engine", "version", "backend", "exact": bool, "profiles", "methods"}` |
| `check` | `{"gguf": path, "ir": path or null}` | `{"accepted": true}` or `{"accepted": false, "message": str}`: load only; a refusal is a result, not an error |
| `open` | `{"gguf": path, "ir": path or null, "evict": true / false / null}` | `{"session": str, "vocab": int, "applications": int, "state_k": int}` |
| `forward` | `{"session", "ids": [...], "report": [...]}` | `{"len": int, ...the reports}` |
| `truncate` | `{"session", "n": int, "report": [...]}` | `{"len": n, ...the reports}` |
| `snapshot` | `{"session", "tag": str}` | `{"len": int}` |
| `restore` | `{"session", "tag": str, "report": [...]}` | `{"len": int, ...the reports}` |
| `close` | `{"session"}` | `{}` |
| `run_node` | `{"node", "inputs", "weights", "state"}` | `{"outputs", "state"}` |

- `ir` (optional) is a document that replaces the file's embedded one (ember's `EMBER_IR`).
- `evict`: null runs the document as declared; false keeps every kv row even where the document
  declares `evict: turn_state` (the comparison run of section 6.8.3); true requires the declaration,
  and is an `error` on a document without it (a request the session cannot honour).
- A snapshot lives until its session is closed and can be restored any number of times.
- `forward` appends `ids` at positions len .. len + n - 1: on a fresh session it is a prefill, later
  an incremental step (n = 1 is token-by-token decode); any chunking is allowed. `truncate` applies
  section 6.9 to the first n tokens. `restore` returns the session to the ids and state of a
  `snapshot`.
- `report` lists what to return after the call. `forward` may ask for any of them; `truncate` and
  `restore` only for `kv_rows`, `kv_positions` and `recurrent` (they run no rows, so `logits` and
  `depth` there are a runner error, and the suite never asks). An engine that cannot give a report
  replies `skip` to the call:

| Report | Shape | Meaning |
|---|---|---|
| `logits` | tensor [n, vocab] | the logits of the rows this call ran (`forward` only) |
| `kv_rows` | `{slot: [rows per application]}` | k/v rows each application's instance of each kv slot holds after the call (after any eviction the call allows); `{}` for a document without kv slots |
| `kv_positions` | `{slot: [[positions] per application]}` | the absolute positions of those rows, ascending |
| `depth` | `[int per id of this call]` | under `depth_route`: each new token's depth, 1 .. R |
| `recurrent` | `{slot: [{tensor: tensor} per application]}` | every recurrent slot's tensors after the call |

- `run_node` (op fixtures, optional): `node` is one IR node; `inputs` maps value names to tensors (a
  `list` value is a JSON list of tensors; `tokens` and `pos` are integer lists); `weights` maps
  tensor names to tensors; `state` is null or `{slot: {"decl": <slot>, "tensors": {name: tensor}}}`,
  the slot's declaration and its tensors before the call. The node runs as application 0 of a
  one-node block. The result maps every name in the node's `out` to a tensor, and gives the slots'
  tensors after the call.

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
    MTSTUDIO_BIN=build/mtstudio python tools/ir_conformance.py --adapter mtstudio
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
  (section 4.1). ember extends `-ffp-contract=off` to `graph_ir.cpp` and `utils.cpp`. paperkiln's
  translation units do not pin the flag: on x86-64 without `-march` (every paperkiln build, the
  wheel's core excepted) there is no FMA instruction to contract into, so they conform there; on an
  FMA target they would not. Adding the flag is deferred to Jonathan (it is a compiler-flag change,
  and the registered tiny-chat arms rely on HEAD reproducing their pinned commits bit for bit; on
  x86-64 it changes no instruction). Listed in Appendix C.
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
  defaults are listed in section 3.5. ember defaults several required attributes today and refuses
  their absence from now on.
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

### 12.2 Open

- **Q11. Digest.** A canonical serialisation and hash, so a paper can cite the exact document, is not
  defined. Proposed: RFC 8785 canonical JSON of the document without `_` keys, then sha256.
- **Q14. Quantised storage** (ember's C10). The dequantisation arithmetic per GGUF type is
  unstated; the candidate is ggml's `dequantize_row_*` as the reference, with EXACT applying to ops
  on the dequantised f32 values. Both engines dequantise per row to the same values as the f32 path
  on the gated models; no gate compares dequantisation across engines bit for bit.
- **Q15. Contraction in paperkiln** (from Q4): whether paperkiln adds `-ffp-contract=off` to its
  translation units. Jonathan's decision.

## Appendix A. The `-ffast-math` audit

*Informative. Audit of 25 October 2026 at paperkiln c3f5ef12. No compiler flag and no numerics
changed.*

**A.1 Flags.** `third_party/transformer_cpp/CMakeLists.txt` (identical in the sibling
`coalfire.cpp`, which local Windows configures pick up first; clones, Colab, RunPod and ext4 builds
use the vendored copy) compiles the static library `transformer_core` with
`target_compile_options(transformer_core PRIVATE -O3 -march=native -ffast-math)` on GCC and Clang,
and `/O2 /arch:AVX2 /fp:fast` (C++ only) on MSVC; the Python wheel replaces `-march=native` with
`-march=x86-64-v3`. The options are PRIVATE: paperkiln's own library (`src/*.cpp`) and tools
(`tools/mtstudio.cpp`) are built with the Release defaults only (GCC `-O3 -DNDEBUG`, no `-march`, no
`-ffast-math`). paperkiln's CUDA sources use nvcc's defaults (`--fmad=true`).

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
| `embedding`, `pos_embedding`, `split`, `list`, `append`, `add`, `mul`, `sub`, `relu`, `affine`, `mul_col`, `scale`, `rope` rotation | `ops.cpp` | paperkiln | EXACT, as stated (A.5) |
| `rmsnorm`, `layernorm`, `silu`, `sigmoid`, `rms_row`, `expr`, `linear_attention`, `ssm_scan` | `ops.cpp`, `expr.cpp`, `kimi_linear.cpp` | paperkiln | ROUNDING, as stated; orders followed |

**A.5 Verdicts on the statements that claim bit identity.**
- The EXACT ops run in paperkiln's own translation units: `-ffast-math` does not reach them, and
  without `-march` an x86-64 build has no FMA instruction to contract into. They stay EXACT.
  Contraction on FMA targets is question Q4.
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
gnu++17 too), so ember extends the flag to `graph_ir.cpp` and `utils.cpp` (Q4). ember's `linear`
(`dot_f32_as_f64`) sums in a fixed 8-lane double order independent of T and of threads, so its
whole-model prefill equals token-by-token decode bit for bit (section 4.5). Its CUDA paths are
ROUNDING by construction (tiles, flash2, nvcc's default `--fmad`); its interpreter's cuBLAS runs
FP32, while its tuned llama path runs TF32 by default (Appendix C).

Nothing in this audit changes a computed value: the tiny-chat precision gate (HEAD with precision off
equal to dc29b5e6 bit for bit) and the pinned commits of the RunPod campaigns are unaffected.

## Appendix B. Changelog

- **1.0** (25 October 2026). The normative text of `include/microtorch/ir.hpp`'s header comment
  (IR version 0) moved here and split into profiles; versioning (section 2), validation (section
  10), conformance (section 11) and the open questions (section 12) added. Clarifications: the
  numerics classes made consistent with the reference build (section 4, Appendix A): stated orders
  of ROUNDING ops are reference orders; conditional exactness for eviction, `turn_state` without
  state ids and recurrent prefill; the RoPE table's class. Version 0 documents are version 1
  documents. Settled with ember.cpp's review (section 12.1): required attributes have no defaults;
  NaN and signed zero in EXACT ops; the f32 RoPE table; per-layer scope; one attention node per kv
  slot; `tokens` and `pos`; one `{i}` per name; depth_route's `out` and its exclusion of list values;
  the eviction bound and the rewind result; raw ids; the specials under state tokens.
- Version 0 history (informative; commits in paperkiln unless marked): llama in the IR (1147ca0);
  gpt2 ops (ember c843421); sliding window with sinks (ember aa145f5); MoE, highway, attention
  residuals, SRD, kimi linear attention, mamba scan; third-party ops (grouped-query attention, NEOX
  and partial RoPE, per-head RMSNorm, per-layer attributes, weight_offset, freq_scale,
  conv1d_causal, selective_scan); `expr` (1159d63e); `scale` (3c1912ed); `weight_index` (567a441d);
  `depth_route` and `kv_rows` (c4427679); per-turn state tokens; `llama_attnres` and `tensor_index`
  (c48ca962).

## Appendix C. Engine capabilities (informative)

What each engine runs, per path, as of 25 October 2026. These are statements about engines, not
semantics; `LOCAL_LLM_STACK.md` section 4's op table is the running record. Where a row lists a gap
the engine has agreed to close, the specification already states the intended behaviour.

| Engine and path | Profiles | Class met | Matrix products | Refuses (valid documents it does not run) |
|---|---|---|---|---|
| numpy reference (`tools/ir_numpy.py`) | core, state, recursion, chat, thirdparty | the reference (prefill only: no cache) | float32 numpy (BLAS order) | `rope_heads` "first" |
| paperkiln `mtstudio`, CPU | the families it trains (section 9.2), from its own runs | EXACT ops EXACT on x86-64; eviction EXACT at equal chunking (Appendix A) | `matmul_optimized` (FMA, shape-dependent order) | documents it did not export (it runs its modules, not documents) |
| paperkiln `mtstudio`, CUDA | as CPU | ROUNDING | cuBLAS FP32; TF32 opt-in | as CPU |
| ember interpreter, CPU | core, state, recursion (not `tensor_index` yet), chat, thirdparty | EXACT where this document says so | `dot_f32_as_f64`, a fixed 8-lane double order independent of T and threads | see below |
| ember interpreter, CUDA | as CPU, ops with device kernels | ROUNDING by construction (tiles, flash2, nvcc's default `--fmad`) | cuBLAS FP32 (its own handle at default math) | documents with an op without a device kernel are served on the CPU automatically (turn_state and MoR documents among them); `EMBER_IR_DEVICE=cpu` forces the CPU for every document |
| ember interpreter, WebAssembly | as CPU | as CPU (no FMA in the build) | as CPU | as CPU (4 GB heap for 0.5B models) |
| ember tuned llama path, CPU and CUDA | llama | ROUNDING against the interpreter | CUDA: TF32 on by default (`EMBER_GEMM_TF32=0` gives FP32) | anything it is not selected for |

**ember's tuned-path selection.** ember selects its tuned llama path today by `family_hint` "llama"
with no window and no state tokens. Section 9.1 forbids deriving semantics from `family_hint`;
ember is adding a structural check (the node list against the canonical llama graph) before
selecting the path.

**ember interpreter refusals** (capability, not validation; section 10.3): more than one kv slot per
block; per-layer values other than attention window and sinks, rope theta and freq_scale, affine
scale and shift, depth_route step, and the slot's window (per-layer sinks included); `rope_heads`
"first"; rope nodes with differing theta, freq_scale, n_rot or pairing; head ops outside its head
list (which excludes `split`, `moe_ffn`, `rope`, `pos_embedding` and `embedding`); `depth_route` with
recurrent state or a windowed slot; `mask: turn_state` with `kv_rows: active`; eviction with recurrent
state; `expr` nested deeper than 200 (the CUDA expr kernel takes up to 96 instructions, a stack of 16
and 8 reductions, and serves larger expressions on the CPU); `tensor_index` until it is implemented.

**Gaps ember is closing** (from its review): closed attribute and key sets (rules 1 and 4) and
`tensor_index` refused or implemented; the load-time refusals of rules 3, 10, 11, 12, 13, 15, 20, 24,
26, 27 and 28; depth_attend's S computed at load (a missing S >= 2 pseudo-query refused then);
`-ffp-contract=off` on `graph_ir.cpp` and `utils.cpp`; the structural check above; required
attributes without defaults (section 3.5).

**paperkiln gaps:** `-ffp-contract=off` is not pinned (Q15); its numpy reference keeps the double
RoPE table (Q3).
