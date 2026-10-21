#pragma once
// paperkiln IR export (LOCAL_LLM_STACK.md section 4): the network as a
// versioned JSON graph that travels inside the GGUF, so an engine can
// interpret any paperkiln model rather than recognise a fixed architecture.
//
// Exported per module, never traced from the tape: each module emits its
// own node spec (Embedding/Linear/LayerNorm/MLP/CausalSelfAttention/
// SlidingWindowAttention/MoEFFN/LlamaBlock/Llama::export_ir,
// nn::gpt2_block_ir; the other families through family_export() in
// tools/parity_model.hpp); the free functions below
// cover the ops that have no module of their own (rmsnorm, rope, attention,
// split, pos_embedding and the elementwise ops a block wires inline).
//
// Shape of a document (ir/chat_sftfinal.ir.json is the worked example):
//   ir_version, family_hint, model {vocab, d, n_layers, ctx, dtype},
//   tokenizer, inputs, embed {nodes}, block {repeat, state, nodes}, head
//   {nodes}, outputs, modes. ember loads a non-llama file from model.d,
//   model.vocab, model.ctx, model.n_layers, block.repeat and block.state
//   alone. Tensors flow as [T, width] row matrices; a
//   node reads the values named in "in" and writes "out" (a later write to
//   the same name replaces it). "weight" names a GGUF tensor, [out, in]
//   row-major; "{i}" in a block is the block index. dtype is the compute
//   dtype; each tensor's storage type is its GGUF type. Keys starting with
//   "_" are prose for humans and carry no semantics.
//
// Ops (v0):
//   embedding   rows of weight [vocab, d] gathered by token id. EXACT.
//   pos_embedding  learned absolute positions: in [x, pos], out = x +
//               weight[pos] row by row, weight [ctx, d] (position_embd).
//               pos must be < ctx: under the context contract
//               (LOCAL_LLM_STACK.md section 3 point 5) generation stops at
//               the limit, so no position reaches ctx. One float add per
//               element. EXACT.
//   linear      x W^T (+ bias), weight [out, in], bias [out] (the bias is
//               added once to the finished dot product); "tied_to" names
//               the tensor to read instead when "weight" is absent. ROUNDING.
//   split       in [x], sizes [s0, s1, ...] summing to x's width, out
//               [o0, o1, ...]: o_k = the s_k consecutive columns of x after
//               the first s0 + ... + s_{k-1} (a fused qkv projection read as
//               q, k, v). Copies. EXACT.
//   rmsnorm     x / sqrt(mean(x^2) + eps) * weight. ROUNDING.
//   layernorm   (x - mean(x)) / sqrt(var(x) + eps) * weight + bias over each
//               row, var the biased (divide by width) mean of squared
//               deviations; weight and bias [d]. ROUNDING.
//   gelu        approx "tanh": 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))
//               (GPT-2's, and what paperkiln trains with); "none": the exact
//               0.5 x (1 + erf(x / sqrt(2))). An engine must not substitute
//               one for the other. ROUNDING.
//   rope        rotates q and k in place; rope_heads "all" (every head) or
//               "first" (head 0 only, legacy runs); pairing "interleaved" =
//               adjacent pairs (x[2j], x[2j+1]) within a head, angle
//               pos * theta^(-2j/head_dim). EXACT.
//   attention   softmax(q k^T * scale) v per head over the kv state slot it
//               names; causal, window (0 = unbounded) and sinks. ROUNDING.
//               Sliding window with sinks (ops::swa_attention; window >= 1):
//               with p the query's position and j a key's (both counted from
//               the first token of the slot, 0-based), j is visible iff
//                 j <= p  and  (j < sinks  or  j >= p - window + 1),
//               i.e. the first `sinks` positions plus the last `window`
//               positions up to and including p. Sinks are positions, not
//               parameters (no tensor). The softmax runs over that union,
//               each key once (the sink range is cut at the window's start
//               when they meet); masked keys contribute exactly 0, and the
//               row max is taken over visible keys only. paperkiln sums
//               sinks first, then the window, each in ascending j. window
//               >= p + 1 with sinks 0 is plain causal attention.
//               Decode: the slot must keep the k/v rows of positions
//               0..sinks-1 and of the last `window` positions (p - window +
//               1 .. p, the new token included); anything older is never
//               read again, so sinks + window rows per kv head suffice.
//               Positions stay absolute (decode pos = tokens seen, not rows
//               kept); the sinks are the first tokens of the prompt. Under
//               the context contract no position reaches model.ctx.
//   add, mul    elementwise. EXACT.  silu  x * sigmoid(x). ROUNDING.
//   moe_ffn     a mixture of SwiGLU experts in llama.cpp's layout:
//                 router  [E, d] tensor (ffn_gate_inp), logits = x R^T
//                 experts {gate, up, down}: 3-D stacks (ffn_*_exps), expert
//                         e's [out, in] matrix at slice e
//                 n_experts E, top_k k: top-k of the E logits per token,
//                         ties to the lower expert index
//                 gate_norm "renorm": gates = softmax over the k selected
//                         logits (the E-way softmax renormalised over them);
//                         "raw": the E-way softmax's selected entries
//                 activation "silu": expert_e(x) = down_e(silu(gate_e x) * up_e x)
//                 out = sum over the selected e of gate_e * expert_e(x),
//                 accumulated in ascending expert index. ROUNDING.
//
// Ops for the gated, attention-residual, routed and recurrent families
// (flex highway/plain, attnres, srd, kimi, mamba). Each is what the named
// paperkiln CPU op computes, in its order of operations; "rounded" means
// rounded to f32 after that operation, with no fused multiply-add.
//   sub         in [a, b]: a - b elementwise. EXACT.
//   relu        max(0, x). EXACT.
//   sigmoid     1 / (1 + exp(-x)) (ops::sigmoid; highway gates, the SRD
//               router). ROUNDING.
//   affine      (x * scale) + shift, float attributes scale and shift; the
//               product is rounded, then the sum (ops::scale then
//               ops::add_scalar). EXACT.
//   mul_col     in [x, c], x [T, w], c [T, 1]: x[t, j] * c[t]. EXACT.
//   rms_row     in [x] [T, w] -> [T, 1]: sqrt(ss / w + eps), ss the sum of
//               x[t, j]^2 accumulated in ascending j (ops::rms_row; the SRD
//               router uses eps 1e-8). ROUNDING.
//   linear_attention  causal linear attention with the elu+1 feature map
//               (ops::kimi_attention; paperkiln's "kimi" attention, and the
//               cheap path of SRD) over a recurrent state slot. in [q, k, v],
//               each [T, heads * head_dim], head h at columns h*head_dim..
//               as for attention. No scale, no softmax, no positions, and
//               every column c is independent of every other:
//                 phi(x)    = x + 1             for x > 0
//                           = (exp(x) - 1) + 1  for x <= 0, rounded after the
//                             subtraction and again after the addition
//                 num[c]    = num[c] + phi(k[t, c]) * v[t, c]  (product
//                             rounded, then the sum)
//                 den[c]    = den[c] + phi(k[t, c])
//                 out[t, c] = (phi(q[t, c]) * num[c]) / max(den[c], den_floor)
//               num and den are updated with token t BEFORE out[t] is formed
//               (token t sees keys 0..t). State: a recurrent slot with
//               tensors num and den, each [heads, head_dim], f32, initial 0
//               (after the first token num = phi(k_0) v_0 and den = phi(k_0)
//               exactly). Attributes heads, head_dim, feature_map
//               "elu_plus_one", den_floor 1e-8, causal true. Prefill is the
//               decode update for t = 0..T-1 in order; a chunked or parallel
//               prefill sums in another order and is ROUNDING against it.
//               ROUNDING (exp).
//   list        in [x0, x1, ...]: a list value holding those tensors in
//               order. A list is a value like a tensor (named by "out", read
//               by "in"), never a GGUF tensor and never state. EXACT.
//   append      in [L, x]: the list L with x added at the end. EXACT.
//   depth_attend  attention over depth (Attention Residuals block form,
//               nn::AttnResStack::depth_attend): in [L, x1, ...] -> [T, d].
//               The sources s_0..s_{S-1} are L's tensors in order, then x1,
//               ... in order. S = 1: out = s_0 bit for bit and "weight" is not
//               read. S >= 2, each row t on its own (decode needs nothing but
//               the current token's sources):
//                 r_i  = 1 / sqrt(ss_i / d + eps), ss_i = sum of s_i[t, j]^2
//                        in ascending j (ops::rmsnorm with unit weight; eps
//                        1e-5)
//                 l_i  = sum over j of (s_i[t, j] * r_i) * w[j], w = "weight"
//                        [d] (the pseudo-query), both products rounded; the
//                        sum is a matmul reduction
//                 a    = softmax over i of l_i: e_i = exp(l_i - max_i l_i),
//                        z = e_0 + e_1 + ... in ascending i, a_i = e_i / z
//                 out  = s_0 a_0 + s_1 a_1 + ..., per element, added in
//                        ascending i (each product rounded, then each sum).
//               ROUNDING.
//   ssm_scan    paperkiln's state-space scan (ops::ssm_scan, as
//               mamba::S4Layer uses it) over a recurrent state slot. in [u]
//               [T, n]; weights "a" [n, n] (row-major, A[i][j] at i*n + j),
//               "b" [n], "c" [n], "d" [1]; attribute n. For t = 0..T-1:
//                 s_t[i]  = (A[i][0] s_{t-1}[0] + A[i][1] s_{t-1}[1] + ... +
//                            A[i][n-1] s_{t-1}[n-1]) + b[i] u[t, i]
//                           summed left to right from 0, each product rounded
//                 y[t, i] = c[i] s_t[i] + d u[t, i]
//               The whole of s_t is formed from s_{t-1} before any element of
//               the state is overwritten. State: a recurrent slot with tensor
//               s [n], f32, initial 0. Prefill is the sequential scan.
//               ROUNDING.
//
// Runtime-defined ops (a new activation or gate from a paper, added to a
// model without recompiling either runtime; ops::expr, microtorch/expr.hpp;
// numpy reference tools/ir_expr.py):
//   expr        in [v0, v1, ...], every input [T, w] with one w; out [T, w].
//               Attribute "expr", a string in this closed grammar (nothing
//               else is accepted; spaces and tabs are ignored):
//                 expr    := term (('+' | '-') term)*
//                 term    := unary (('*' | '/') unary)*
//                 unary   := '-' unary | power
//                 power   := primary ('^' unary)?
//                 primary := number | name | pi | '(' expr ')'
//                            | fn '(' expr (',' expr)* ')'
//               so ^ binds tightest and is right-associative (2^3^2 =
//               2^(3^2), -x^2 = -(x^2), x^-1 is allowed). A number is
//               decimal or scientific (1, 0.5, .5, 2., 1e-5, 1.5E+3), read
//               as the nearest float (strtof), and must be finite; pi is
//               the nearest float to pi. Names are positional, not the IR
//               values': the inputs in the order of "in" are x, then a, b,
//               c, ... (x the first, a the second; at most 23 inputs, x
//               and a .. v). Functions,
//               each over the expression's [T, w] values:
//                 exp log sqrt tanh erf abs   elementwise (f32 libm)
//                 sigmoid(e)       1 / (1 + exp(-e)), as the sigmoid op
//                 relu(e)          max(0, e), as the relu op
//                 max(a, b)        a where a >= b, else b
//                 min(a, b)        a where a <= b, else b
//                 where(c, a, b)   a where c > 0, else b
//                 rowsum(e)        the sum of the row's w entries, added in
//                                  ascending column order from 0
//                 rowmean(e)       rowsum(e) / w
//                 rowmax(e)        the row's largest entry
//               the row reductions broadcast back to [T, w]. + - * / are
//               elementwise and ^ is pow (powf). Every primitive is
//               computed in f32 and rounded to f32 after it, with no fused
//               multiply-add; a subexpression without inputs is folded in
//               f32 with the same operations (where(c, ...) on constants
//               included), and a result without inputs fills [T, w]. The
//               rows are independent, so decode (T = 1) is prefill's last
//               row. The training gradients come from the primitives
//               (max, min and where pass it to the chosen operand, none to
//               where's condition; rowmax to the row's first largest
//               entry; abs and relu 0 at 0); an engine needs only the
//               forward. An engine that cannot run an expression fails at
//               load naming the function. ROUNDING (libm transcendentals
//               differ by an ulp between platforms).
//
// Every op has the same prefill (T tokens) and decode (T = 1) semantics;
// attention additionally reads and appends to its kv state in decode, and
// a recurrent op reads and replaces its recurrent state.
// A kv state declares the same window and sinks as the attention node that
// reads it (0 and 0 = keep every row up to max_len).
//
// State slots (block.state; one instance per block, i.e. per layer, and per
// sequence; a node names its slot in "state"):
//   kv          {name, kind "kv", heads, kv_heads, head_dim, max_len, window,
//               sinks, dtype}: the k and v rows of the tokens seen so far.
//   recurrent   {name, kind "recurrent", op, dtype "f32", tensors: [{name,
//               shape, init}, ...]}: a fixed-size state carried from token to
//               token by the op named in "op" (linear_attention: num, den;
//               ssm_scan: s), shapes as listed, every tensor set to "init"
//               (0) when a sequence starts. Decode applies the op's update
//               for one token; prefill applies it for t = 0..T-1 in order, so
//               prefilling a prompt and decoding it token by token leave the
//               same state bit for bit. The size does not grow with the
//               sequence and there is no max_len or window; positions never
//               enter a recurrent op (model.ctx still bounds every sequence
//               under the context contract). Prefix reuse keeps a snapshot of
//               the tensors taken after the prefix's last token: a recurrence
//               cannot drop tokens, so rewinding means restoring a snapshot
//               or running again from the start.
// A block may declare several slots (srd: one kv and one recurrent).
//
// Third-party graphs (LOCAL_LLM_STACK.md section 7): an engine builds the IR
// of a llama.cpp GGUF without paperkiln.ir from general.architecture and its
// metadata, as tools/gguf_ir.py does (the reference; tools/ir_numpy.py runs
// it). Attributes and ops they add:
//   per-layer attributes  any attribute of a block node or of a block state
//               slot may be {"per_layer": [v_0, ..., v_{L-1}]}, L =
//               block.repeat: block i reads v_i (gemma3: attention window
//               and kv window 512 on local layers, 0 on global ones; rope
//               theta 10000 / 1e6). Every other attribute is the same in
//               every block.
//   attention   kv_heads < heads (grouped-query): query head h reads k/v
//               head h / (heads / kv_heads); k and v are [T, kv_heads *
//               head_dim]. heads % kv_heads == 0.
//   rope        kv_heads (default heads): k's head count. pairing "neox":
//               the pair (x[j], x[j + n_rot/2]) within a head (llama.cpp
//               NEOX, HF rotate_half). n_rot (default head_dim, even):
//               only a head's first n_rot dims rotate, the rest pass
//               through (partial rotary, GPT-NeoX); the angle of pair j is
//               (pos * freq_scale) * theta^(-2j/n_rot), freq_scale default 1
//               (linear RoPE scaling: 1 / factor). EXACT given the angle
//               table (computed in double, then rounded to f32 cos/sin).
//   rmsnorm     group g (default the row width): each consecutive g columns
//               normalised on their own, weight [g] shared by every group
//               (per-head q/k norm: qwen3, gemma3). weight_offset c (default
//               0): the weight is read as w + c, rounded to f32 once at load
//               (a load-time transform; Gemma's (1 + w), needed only when a
//               file holds HF's raw w: llama.cpp's convert already stores 1 + w).
//   conv1d_causal  depthwise causal convolution over time (Mamba): in [x]
//               [T, C]; weight [C, K] (ssm_conv1d), bias [C] optional;
//               attributes channels C, kernel K; state names a recurrent
//               slot with tensor x [K - 1, C], the K - 1 previous inputs,
//               oldest first, initial 0. With xx = the state's rows then x's:
//                 out[t, c] = sum over k = 0..K-1, ascending, of
//                             w[c, k] * xx[t + k, c]  (+ bias[c], added once
//                             to the finished sum)
//               then the state becomes the last K - 1 rows of xx. ROUNDING.
//   selective_scan  Mamba-1's selective state-space scan (ggml_ssm_scan):
//               in [x, dt, B, C]: x and dt [T, D], B and C [T, N]; weights
//               "a" [D, N] (A as stored: llama.cpp writes -exp(A_log)), "d"
//               [D] optional (the skip); attributes channels D, d_state N,
//               dt_softplus (true). State: a recurrent slot with tensor s
//               [D, N], initial 0. For t = 0..T-1, every channel c:
//                 delta  = dt[t, c] <= 20 ? log1p(exp(dt[t, c])) : dt[t, c]
//                          (softplus; delta = dt when dt_softplus is false)
//                 xd     = x[t, c] * delta
//                 s[c,n] = s[c,n] * exp(delta * A[c,n]) + B[t,n] * xd
//                 y[t,c] = sum over n ascending of s[c,n] * C[t,n]
//                          (+ d[c] * x[t, c] after the sum)
//               Prefill is the sequential scan. ROUNDING (exp, log1p).
// Third-party families (family_hint = general.architecture; llama.cpp's
// tensor names; tools/gguf_ir.py's docstring has each graph):
//   llama, mistral  pre-RMSNorm, interleaved RoPE, grouped-query attention,
//               SwiGLU; <arch>.attention.sliding_window w sets window w,
//               sinks 0 (HF Mistral's mask; llama.cpp b6000 ignores it).
//   qwen2, qwen3  llama with NEOX RoPE; qwen2 biased q/k/v, qwen3 per-head
//               q/k rmsnorm (group head_dim) before RoPE.
//   gpt2        as paperkiln's gpt2 (fused qkv split q | k | v, tanh GELU).
//   gptneox     gpt2 without positions, NEOX partial RoPE (n_rot =
//               rope.dimension_count), exact GELU, untied head; parallel
//               residual x = (ffn(ln2 x) + x) + attn(ln1 x) when
//               use_parallel_residual.
//   gemma3      embedding then affine(scale sqrt(d)); sandwich rmsnorms
//               (attn_norm, post_attention_norm, ffn_norm, post_ffw_norm),
//               q/k rmsnorm, NEOX RoPE, GeGLU (gelu tanh of gate, times
//               up), scale 1/sqrt(key_length); local layers (i % 6 < 5) have
//               window = attention.sliding_window and theta 10000, global
//               ones window 0 and rope.freq_base: per-layer attributes.
//   mamba       (Mamba-1; not paperkiln's "mamba" family, which has
//               architecture "paperkiln") rmsnorm, ssm_in -> split x | z,
//               conv1d_causal (state "conv") + silu, ssm_x -> split dt | B |
//               C, ssm_dt (biased), selective_scan (state "ssm"; a = ssm_a,
//               d = ssm_d), * silu(z), ssm_out, residual add.
//
// Families (family_hint; GGUF general.architecture). A tensor that is the
// same thing as a gpt2 tensor keeps llama.cpp's gpt2 name: token_embd,
// position_embd, blk.{i}.attn_norm, attn_qkv, attn_output, ffn_norm, ffn_up,
// ffn_down, output_norm, output (each .weight, and .bias where the module
// has one); new tensors are blk.{i}.<name>.weight (and .bias). "paperkiln" is the
// architecture of a file no llama.cpp graph computes; its metadata keys are
// paperkiln.{context_length, embedding_length, block_count, vocab_size,
// attention.head_count, feed_forward_length}.
//   gpt2, gpt2_swa  as above (gpt2).
//   kimi        gpt2 with linear_attention (state "lin": num, den [heads,
//               head_dim]) in place of attention; the gpt2 tensor set
//               exactly (gpt2: llama.cpp loads it but computes softmax
//               attention, so only the IR reproduces the model).
//   srd         gpt2 whose attention blends two paths of one qkv per token:
//               p = pred_down(gelu(pred_up(h))) on the normalised input h
//               (blk.{i}.srd_pred_up [d/4, d], srd_pred_down [d, d/4], both
//               biased), g = sigmoid(affine(rms_row(h - p), 4, -1)), gi =
//               affine(g, -1, 1); out = mul_col(attention, g) +
//               mul_col(linear_attention, gi), then attn_output. States kv
//               and "lin" (paperkiln).
//   attnres     gpt2 blocks whose residual stream is a list (Attention
//               Residuals, block form, one list entry per transformer
//               block): embed ends with list [x] -> "bank"; block i reads h =
//               depth_attend(bank; blk.{i}.attnres_attn_q), a = attention
//               sublayer of layernorm(h), h2 = depth_attend(bank, a;
//               blk.{i}.attnres_ffn_q), f = ffn of layernorm(h2), then append
//               a + f to bank; the head reads depth_attend(bank;
//               attnres_out_q). Pseudo-queries [d]; blk.0.attnres_attn_q is
//               written as zeros and never read (its bank has one entry).
//               State kv only (paperkiln).
//   flex, flex_highway, flex_plain  the flex decoder (residual, highway,
//               plain): pre-norm layernorm or rmsnorm (attn_norm / ffn_norm /
//               output_norm, no bias under rmsnorm), FFN gelu (ffn_up, ffn_down
//               biased), relu (the same with relu), a runtime-defined
//               activation "expr:<e>" (the same with an expr node over one
//               input) or swiglu (ffn_gate,
//               ffn_up, ffn_down, no biases: down(silu(gate h) * up h)),
//               positions learned or sinusoidal (position_embd holds the
//               fixed table, rounded to f32), attention exact or windowed.
//               Each sublayer f combines with its input stream x as
//               residual x + f, plain f, or highway x + sigmoid(W x + b) *
//               (f - x) with W, b = blk.{i}.highway_attn / highway_ffn
//               [d, d] + bias, nodes linear, sigmoid, sub, mul, add. The
//               GPT-2 flavour (layernorm, gelu, learned, residual) exports
//               as gpt2 / gpt2_swa; plain with that flavour is gpt2 tensors
//               (gpt2 architecture, IR-only semantics); others paperkiln.
//   mamba       no positions; block: h = layernorm(x) (blk.{i}.ssm_norm), u =
//               ssm_in(h) [n], y = ssm_out(ssm_scan(u)) [d], y = y *
//               gelu(ssm_gate(h)) * gelu(ssm_out_gate(x)) (multiplied in that
//               order), x = x + y; projections biased; ssm_scan weights
//               blk.{i}.ssm_a [n, n], ssm_b, ssm_c [n], ssm_d [1]; state
//               "ssm": s [n]. Head: output_norm, then output with a bias
//               (paperkiln).
//
// GGUF: the compact dump is the string key kGGUFKey and the version the
// uint32 key kGGUFVersionKey, written after every other metadata key.
#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace microtorch {
namespace ir {

// Insertion-ordered, so a node reads op, in, attributes, out.
using json = nlohmann::ordered_json;

constexpr uint32_t kVersion = 0;
constexpr const char* kGGUFKey = "paperkiln.ir";
constexpr const char* kGGUFVersionKey = "paperkiln.ir_version";

// A float32 attribute as the shortest decimal that reads back as the same
// float (1e-5f -> 1e-05, not 9.999999747378752e-06).
double f32(float v);

// Nodes for the ops that no module owns.
json rmsnorm(const std::string& in, const std::string& weight, float eps, const std::string& out);
json rope(const std::string& q, const std::string& k, const std::string& pos, size_t heads, size_t head_dim,
          float theta, bool all_heads);
json attention(const std::string& q, const std::string& k, const std::string& v, const std::string& state,
               size_t heads, size_t kv_heads, size_t head_dim, float scale, bool causal, size_t window,
               size_t sinks, const std::string& out);
json kv_state(const std::string& name, size_t heads, size_t kv_heads, size_t head_dim, size_t max_len,
              size_t window, size_t sinks);
json binary(const std::string& op, const std::string& a, const std::string& b, const std::string& out);
json unary(const std::string& op, const std::string& in, const std::string& out);
json layernorm(const std::string& in, const std::string& weight, const std::string& bias, float eps,
               const std::string& out);
// approx: "tanh" | "none" (anything else throws).
json gelu(const std::string& in, const std::string& approx, const std::string& out);
json split(const std::string& in, const std::vector<size_t>& sizes, const std::vector<std::string>& outs);
json pos_embedding(const std::string& x, const std::string& pos, const std::string& weight, const std::string& out);
// (x * scale) + shift.
json affine(const std::string& in, float scale, float shift, const std::string& out);
json rms_row(const std::string& in, float eps, const std::string& out);
// Causal elu+1 linear attention over the recurrent slot `state`.
json linear_attention(const std::string& q, const std::string& k, const std::string& v, const std::string& state,
                      size_t heads, size_t head_dim, const std::string& out);
// The recurrent slot linear_attention reads: num and den, [heads, head_dim].
json linear_attention_state(const std::string& name, size_t heads, size_t head_dim);
// ins[0] a list, ins[1..] tensors; weight = the pseudo-query [d].
json depth_attend(const std::vector<std::string>& ins, const std::string& weight, float eps, const std::string& out);
json list(const std::vector<std::string>& ins, const std::string& out);
json append(const std::string& list, const std::string& x, const std::string& out);
// prefix + "ssm_a.weight" etc. hold A [n, n], b, c [n], d [1].
json ssm_scan(const std::string& in, const std::string& state, const std::string& prefix, size_t n,
              const std::string& out);
json ssm_state(const std::string& name, size_t n);
// A runtime-defined op: text in the grammar of microtorch/expr.hpp.
json expr(const std::vector<std::string>& ins, const std::string& text, const std::string& out);

}  // namespace ir
}  // namespace microtorch
