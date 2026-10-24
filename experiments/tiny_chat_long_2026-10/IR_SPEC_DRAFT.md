# IR spec text for review (DRAFT, 24 Oct 2026): per-turn state tokens, llama attention residuals, and their recursion combinations

*Proposed wording for `include/microtorch/ir.hpp`, written for review by paperkiln and ember before any
implementation. Nothing here is normative and `ir.hpp` is not edited: the text becomes normative only
when the implementing commit copies it into `ir.hpp` and both runtimes pass the gates in section 5.
Context: `PREREGISTRATION_DRAFT.md` (this folder), cells S-full / S-gist (E4), X (cross-layer) and the
exploratory step-2 cells; `CROSSLAYER_LITCHECK.md`.*

Primary sources read for the state-token design (24 Oct 2026, arXiv PDFs):
- **Gisting** (Mu, Li and Goodman, arXiv 2304.08467v3, NeurIPS 2023): k copies of one gist token are inserted
  after the prompt; "input tokens after the gist tokens cannot attend to any of the prompt tokens before the
  gist tokens (but they can attend to the gist tokens)"; for decoder-only models the lower-left block of the
  causal triangle is masked (their Figure 2a). Trained by ordinary instruction finetuning with the modified
  mask, no extra loss. k in {1, 2, 5, 10}: "generally insensitive to the number of gist tokens", 10 sometimes
  worse. LLaMA-7B, one gist token: 99.2% of the positive control's ROUGE-L on seen prompts, 91.0% unseen,
  75.4% on the human split; failures on verbatim details ("Arts & Culture" compressed to "Culture") and
  runaway repetition. Positive control = the same token with the ordinary mask; negative control = no task.
  At inference the gist activations are cached and the prompt's rows dropped: 1 cached row in place of 26.
  The positions after the gist keep their absolute positions.
- **Recurrent Memory Transformer** (Bulatov, Kuratov and Burtsev, arXiv 2207.06881v2, NeurIPS 2022): m memory
  tokens at the start (read) and the end (write) of each segment; the write block's outputs become the next
  segment's read memory; in the decoder the causal mask applies to the sequence tokens, while memory tokens
  inside a read or write block see each other. Trained with backpropagation through time over 0-4 previous
  segments: deeper unrolls "drastically" improve perplexity but bring "instabilities and out-of-memory
  issues" at larger memory. Memory 1-50: a large gain from the first token, fading from 5 to 50; RMT with 10
  memory tokens matches Transformer-XL with a 75-state cache on WikiText-103 (16 layers, d 410).
- **Compressive Transformer** (Rae et al., arXiv 1911.05507): old memories compressed at rate c by pooling,
  1D or dilated convolution or a most-used rule; an attention-reconstruction loss (its gradient stopped from
  the main network) worked best (enwik8 0.973 bpc against 0.986-0.996 for pooling or BPTT-trained conv).
  Read for contrast only: it compresses per layer with a separate network, which neither the IR nor
  mtstudio has.
- Abstracts only, for orientation: AutoCompressors (2305.14788: summary vectors per segment as soft prompts,
  up to 30,720 tokens), Activation Beacon (2401.03462: per-layer key/value compression, 8x KV reduction).

The design below takes gisting's mask (a pure attention-mask change, trained end to end), RMT's chaining
(each turn's state block reads the previous blocks, so state can pass forward through attention, with full
backpropagation through the whole window instead of a truncated unroll), and gisting's inference rule (keep
the state rows, drop the turn's rows).

---

## 1. Per-turn state tokens (paperkiln `arch.state_tokens`, family flag on any attention family)

### 1.1 Inputs: state slots outside the vocabulary

    embedding (extended)  an id v with vocab <= v < vocab + k reads row v - vocab
                of "state_weight" [k, d] (GGUF state_embd.weight) instead of the
                token embedding: {"op": "embedding", "in": ["ids"], "weight":
                "token_embd.weight", "state_weight": "state_embd.weight",
                "out": "x"}. Without "state_weight", an id >= vocab is refused
                (as now). The tied output head covers ids 0 .. vocab - 1 only:
                state rows are never logits and never sampled. EXACT.

Why not new vocabulary tokens: the 16,384-entry vocabulary has four specials (`<unk>`, `<|endoftext|>`,
`<|user|>`, `<|assistant|>`) and no reserved ids, and every run's vocabulary must stay byte-identical (the
pre-registration's section 1). A state row is a learned vector, as RMT's memory tokens are; the tokeniser
never produces one.

### 1.2 The template contract (LOCAL_LLM_STACK.md section 3 point 3, extended)

    paperkiln.state_tokens.k  (GGUF metadata, u32; absent = 0). When k > 0, every
                exchange in the served template is
                  <|user|> U <|assistant|> A <|endoftext|> S_0 S_1 .. S_{k-1}
                with S_j the id vocab + j: the engine appends the k state ids
                after every <|endoftext|> that ends an assistant reply, in
                prompts and after generation alike (a generated <|endoftext|>
                is followed by a k-token prefill of the state ids, no
                sampling). A state block is exactly the ids vocab .. vocab+k-1
                in ascending order; any other use of an id >= vocab is refused.

### 1.3 The attention mask

    attention   "mask": "turn_state" (default "causal"). Every position p has a
                segment index s(p) and a kind c(p) in {text, state}, both
                functions of the ids up to p alone: c(p) = state iff id_p >=
                vocab; s(p) = the number of COMPLETE state blocks (k state
                positions each) strictly before p. A key at position j <= p is
                visible to the query at p iff
                  s(j) == s(p)                      (the same segment: a turn's
                                                     text and, after it, its own
                                                     state block, causally), or
                  c(j) == state and s(j) < s(p)     (an earlier turn's state).
                So the text of turn n sees its own earlier tokens and the state
                blocks of turns 0 .. n-1; the state block of turn n sees turn n's
                text, its own earlier state positions and every earlier state
                block (the chain through which state passes forward). Masked keys
                contribute exactly 0 and the row maximum is over visible keys
                only (the attention op's rule); the softmax sums the visible keys
                in ascending position. Positions stay absolute (RoPE unchanged).
                Only with window 0 and sinks 0 (refused otherwise). ROUNDING, as
                attention.

Gisting's decoder mask is the case of one segment followed by one gist block; this is that rule applied after
every turn, with RMT's read-from-previous-state chaining expressed by the second clause. A sequence with no
state ids is one segment, so `turn_state` on ordinary text is plain causal attention bit for bit.

### 1.4 The kv cache: dropping finished turns

    kv slot     "evict": "turn_state" (optional; requires every attention node
                reading the slot to have mask "turn_state"). When the last state
                position of block n has been written, no later query can see
                any text row with s = n (later queries have s > n, and the rule
                admits only state rows of earlier segments), so the engine MAY
                delete those rows. After n complete turns the slot holds n * k
                state rows plus the rows of the current turn. Positions stay
                absolute (decode pos = tokens seen, not rows kept), as for the
                sliding window. Evicting is EXACT against keeping every row: the
                deleted rows are masked for every later query, and the visible
                keys are summed in the same ascending order either way. Engines
                that keep every row compute the same values by the mask.
                Truncating the cache to n tokens (prefix reuse, rewind) is safe
                at any position at or after the last evicted row; rewinding into
                an evicted turn needs a re-prefill from the start of that turn's
                segment, whose rows the engine must then hold (engines keep the
                rows of the current segment, so regenerating the latest reply
                needs nothing more).

Size, from this folder's receipts (f32, 20m_L8: 2 x 448 x 4 bytes x 8 layers = 28 KiB per row): a 14-turn
Carry-12 conversation at 48-token replies is at most 870 tokens (`carry12_v0_budget.json`), 24 MiB of cache
in full; with k = 4 and eviction, 13 x 4 state rows plus at most ~110 rows of the current turn, about 4.5 MiB.
Attention per decoded token falls in the same ratio. The cache stops growing with the number of turns, which
is what ember's WASM Space cares about.

### 1.5 Training (mtstudio; for the record, not IR)

- State ids are inserted on the fly when `arch.state_tokens.k > 0`: after every `<|endoftext|>` that closes an
  exchange (an `<|endoftext|>` with an `<|assistant|>` after the latest `<|user|>` in the same document), in
  pretraining windows and SFT rows alike, before windowing. FineWeb-Edu documents have no exchanges and get
  none. The same `.ids` mixes then serve every cell, so the token stream and data order before insertion are
  the plain cell's.
- No loss at a position whose target is a state id, nor at the last state position (its target is the next
  `<|user|>`, always supplied by the template). State positions are never prediction targets.
- Full backpropagation through the whole window (the chain of state blocks inside a T = 1024 window is
  trained end to end; RMT's truncated unroll is not needed at this length).
- Initialisation: `state_embd` rows from the embedding initialiser. Parameters: k x d (1,792 at k = 4,
  d 448).
- The mask needs a kernel: today's `k_attn_masked_softmax` visits one contiguous key range per row; the
  `turn_state` row is the current segment's range plus a sparse set of earlier state positions. On the CPU
  path a per-row visible list; on CUDA a segment-and-kind array per sequence read by the scores, softmax and
  backward kernels.

## 2. `llama_attnres`: attention residuals on the llama block (cell X)

No new op. A family composed from existing nodes (`list`, `append`, `depth_attend`, `rmsnorm`, `rope`,
`attention`, the SwiGLU nodes), the full (sublayer) form of Attention Residuals, as `nn::AttnResStack` with
`block_size` 0 (`CROSSLAYER_LITCHECK.md` section 4):

    llama_attnres  the llama block (RMSNorm, interleaved RoPE on every head,
                SwiGLU, no biases, tied output) whose residual stream is a list
                with one entry per sublayer: embed ends with list [x] -> "bank";
                block i: h = depth_attend(bank; blk.{i}.attnres_attn_q); a =
                attn_output(attention(rope(q, k), v) of rmsnorm(h; attn_norm));
                bank = append(bank, a); h2 = depth_attend(bank;
                blk.{i}.attnres_ffn_q); f = ffn_down(silu(ffn_gate g) * ffn_up g),
                g = rmsnorm(h2; ffn_norm); bank = append(bank, f). Head: h =
                depth_attend(bank; attnres_out_q), output_norm, output (tied).
                Pseudo-queries [d], initialised to zero (uniform depth weights:
                under pre-norm the plain model's sublayer inputs up to eps).
                blk.0.attnres_attn_q is written as zeros and never read (its bank
                has one entry: depth_attend with S = 1 copies s_0). Sources per
                model: 2 L + 1. State kv only. Architecture "paperkiln".

- The difference from the existing `attnres` family is the block (llama instead of GPT-2) and the granularity
  (sublayer entries, not one `a + f` entry per block). ember runs `list`, `append` and `depth_attend` on CPU,
  CUDA and WASM (attnres tiny 40/40, ember ec53c26 and 1232431), so ember should need nothing new; that is
  unverified until the gate in section 5 passes on a trained tiny `llama_attnres`.
- mtstudio: `AttnResStack` around the llama blocks (today only `parity::AttnResLM`, GPT-2 blocks), a new
  `arch.custom.attn_res = {form: "full"}` and the exporter's family. At 20m_L16: 33 sources, 2 x 16 + 1
  queries of 320 = 10,560 parameters.
- Decode needs nothing beyond the current token's bank (depth attention is per row); prefill equals decode
  as for every row-wise op.

## 3. Attention residuals under recursion (step-2 cell, conditional on X's main effect)

Fixed recursion (`weight_index`) already composes with section 2: every application appends its two entries,
so the bank holds 2 x (applications) + 1 sources, and earlier passes of a shared block are sources for later
passes (Depth-Attention's and Hyperloop's loop-as-source arrangement, `CROSSLAYER_LITCHECK.md`). One question
the IR must answer is whose pseudo-queries a pass uses:

- **Shared with the stored layer** (today's rule: every tensor name in a block with "{i}" reads blk.{m_i}):
  no IR change. All passes of a shared block then ask the same depth question.
- **Per application** (proposed): a node key

      "weight_index": "application"   on a node of a block with a block
                weight_index: that node's tensor names substitute "{i}" with the
                application index i, not m_i (GGUF rec_app.{i}.attnres_attn_q,
                rec_app.{i}.attnres_ffn_q, written in full as constraint 4
                requires of router names). Every other node keeps m_i. Absent:
                m_i, as now.

  Cost d per query per application (at 20m_L4 recursed 4 times: 16 applications, 33 queries, 21K
  parameters). Recommended for the cell, because a pass that reads the same sources with the same query as
  the previous pass gains nothing from depth attention.

## 4. Attention residuals under Mixture-of-Recursions (EXPLORATORY: untested anywhere)

`CROSSLAYER_LITCHECK.md` found no combination of cross-layer connections with Mixture-of-Recursions. The idea
(Jonathan): MoR chooses how many passes a token takes; depth attention chooses what each pass reads. Under
`depth_route` a token that has exited does not run the later applications (step (b)), so the banks of
different tokens would hold different numbers of entries. Proposed wording:

    list (row presence)  a list value carries, per entry, a row-presence mask
                [T] (every row present by default). Under depth_route, an
                append at an application of step k writes the entry for the
                active rows only and marks the inactive rows absent (their
                entry is never computed; engines may store any value).
    depth_attend (rows with absent entries)  each row attends over its present
                entries only, in ascending index; S_row counts them (always >= 1:
                the embedding entry and the prelude's entries are present for
                every row). S_row = 1 copies the entry bit for bit, as S = 1 now.
    depth_route gate under llama_attnres  with gate "score", each entry an
                active token appends during step k is multiplied by its gate g
                (mul_col) before the append: the analogue of x_in + g (S(x) -
                x_in), since under attention residuals a step's contribution is
                the entries it adds. gate "none": entries appended unscaled.

- Causality holds: presence is a function of the token's depth, which is a function of its own prefix
  (constraint 1); depth attention is per row, so prefill equals decode.
- `kv_rows: "active"` is unchanged (it concerns the kv slot, not the bank).
- Gates before any run: section 5's, plus "every token forced to the largest depth with gate none equals
  recursion x AttnRes (section 3) bit for bit", MoR's gate (3) in this form.

## 5. Gates (as amendment 4, A4.6, for each item before a run uses it)

1. **Prefill equals token-by-token decode** (logits to rounding) on CPU, for every new mask, family and key.
2. **State tokens, eviction is exact:** greedy generation and the logits of every position equal between an
   engine run that keeps every kv row and one that evicts at `turn_state`, 40/40 on a 14-turn conversation
   set (Carry-12 items, gold history), CPU and CUDA; and with k > 0 but no state ids in the input, the model
   equals plain causal attention bit for bit.
3. **numpy IR** (`tools/ir_numpy.py`, the new mask, the extended embedding, `llama_attnres`, the
   per-application key, row presence) against `mtstudio score` within ROUNDING, greedy 40/40.
4. **Finite-difference gradcheck** in mtstudio through the `turn_state` mask (its backward kernel), through
   `llama_attnres` (as `test_attn_res`), and through the presence-masked depth attention.
5. **ember** 40/40 on CPU, CUDA and WASM for `llama_attnres` and the state-token model (eviction on and off);
   `tools/colab_cuda_validate.sh` legs for the new kernels on Colab.
