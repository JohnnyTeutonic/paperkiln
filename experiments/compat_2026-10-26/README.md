# pk_base_v1 and pk_sft_v1 in llama.cpp and Ollama, 26 Oct 2026

LOCAL_LLM_STACK.md section 11 item 6: the models on the 16k Llama 3-scheme vocabulary
(`tokenizer.ggml.model` "gpt2", pre "llama-bpe") run end to end in llama.cpp and Ollama as they run in
paperkiln and ember. Everything here ran on the CPU in WSL, 2 threads, under `scripts/cpu_budget.py`. Nothing
was pulled from or pushed to Ollama's registry or Hugging Face.

## Setup

- **Models** (C:/ml_artifacts/): `next_base/base/pk_base_v1.gguf` (F32, sha256 510d288c...),
  `next_base/base/pk_base_v1_q4_k_m.gguf` (78b3482f..., paperkiln's real k-quants, byte-identical to
  llama.cpp b6000's llama-quantize), `general_chat/sft/pk_sft_v1.gguf` (F32, 4d53c7ff...; it carries
  `paperkiln.chat_template` and the generated `tokenizer.chat_template`). The base F32 file was exported
  before the conversation schema landed (24 Oct), so it has no `tokenizer.chat_template`; the base model is
  driven by raw completion here, which needs none.
- **llama.cpp**: tag b6000 (4762ad7316dc), the tag the k-quant gates used, built in WSL
  (`~/llamacpp_b6000_build`, Release, GGML_NATIVE off, AVX2; `llamacpp/llamacpp_commit.txt`).
- **Ollama**: v0.40.11, the official Linux release archive (the one install.sh fetches) unpacked user-level
  into `~/ollama` with its CPU libraries only (no sudo, no systemd service), served on 127.0.0.1:11535 with
  a private `OLLAMA_MODELS` that is deleted at the end.
- **References** (no paperkiln or ember process rerun): ember's and mtstudio's greedy replies recorded on
  the 40 frozen prompts (`tinychat_hf_prompts_v1.jsonl`, 60 new tokens): pk_base_v1 F32 in
  `ember.cpp/experiments/pk_base_v1_2026-10-24/pk_base_v1_parity.jsonl` (ember = mtstudio 40/40), Q4_K_M in
  `ember.cpp/experiments/kquant_2026-10-25/q4_k_m_replies.jsonl`, pk_sft_v1 in
  `C:/ml_artifacts/general_chat/parity/parity_cpu.jsonl` (ember = mtstudio 40/40).
- **Harnesses**: `llamacpp_gate.py` (llama-server: /tokenize, /detokenize, /completion, /apply-template,
  /v1/chat/completions) and `ollama_gate.py` (/api/chat, /api/generate). Greedy everywhere: temperature 0,
  top_k 1, top_p 1, min_p 0, repeat penalty 1, context 512; llama.cpp also bans the `<unk>` id as paperkiln
  and ember do (Ollama's API has no logit bias; its replies equal ember's regardless).

## Results

| check | result |
|---|---|
| llama.cpp loads F32 and Q4_K_M pk_base_v1 and F32 pk_sft_v1 | all three (`llamacpp/load.json`: arch llama, n_vocab 16384, EOG `<|endoftext|>`, 110.12M parameters = paperkiln's 97.5M plus the separate 16384 x 768 output matrix) |
| tokenisation, llama.cpp /tokenize (specials parsed) vs paperkiln's `tools/llama3_bpe.py` | **794/794 records, 84,934 tokens identical**; the C++ `llama3_tokenizer.hpp` (tools/llama3_encode, built from this tree) also 794/794; /detokenize round trip 794/794 |
| greedy pk_base_v1 F32 (f32 KV cache) vs ember and mtstudio | **40/40** replies, token counts, stop reasons and prompt token counts |
| the same with llama.cpp's default f16 KV cache | **40/40** |
| greedy Q4_K_M vs ember's Q4_K_M | 29/40 replies (token counts 30, stops 35, prompt tokens 40) |
| the Q4_K_M file dequantised to F32 by llama-quantize, run as F32, vs ember's Q4_K_M | **40/40** |
| llama-server `--jinja` /apply-template on pk_sft_v1 vs `tools/chat_template.py` render() | **60/60** strings and ids (20 single-turn, 20 with 1-4 earlier exchanges, 20 with a system message) |
| llama-server /v1/chat/completions, pk_sft_v1, vs ember and mtstudio | **40/40** replies and prompt tokens; finish "stop" where ember stopped at end-of-text, 40/40; completion tokens equal less the end-of-text token llama.cpp counts, 40/40 |
| Ollama v0.40.11: paperkiln's Modelfile, /api/chat vs ember and mtstudio | **40/40** (section below) |

**The Q4_K_M differences are llama.cpp's activation quantisation, not the weights.** llama.cpp's CPU
kernels multiply Q4_K and Q6_K blocks by activations quantised to Q8_K (`ggml_vec_dot_q4_K_q8_K`);
ember multiplies the stored blocks' f32 values (ggml's `dequantize_row_*`, IR_SPEC 4.7). With the
activation quantisation taken out (the same file dequantised to F32 and run in llama.cpp), llama.cpp gives
ember's Q4_K_M replies 40/40. The 11 differing prompts (0, 5, 10, 11, 15, 23, 27, 28, 30, 34, 36) part after
a shared prefix of 8-88 characters, as quantised base models with small top-1 margins do; on 5 of them
(0, 23, 27, 28, 36) llama.cpp's Q4_K_M reply is the F32 model's own reply.

## Ollama

`ollama create pk-sft-v1 -f Modelfile` imports the F32 GGUF with paperkiln's Modelfile
(`ollama/pk_sft_v1.Modelfile`, tools/chat_template.py `modelfile()`, which mtstudio writes beside an F32
export); Ollama reports family llama, 110.12M parameters, runner llamacpp, and the three stop strings.
Ollama 0.40's debug log gives each prompt's length in bytes, not its text, so each of the 60 conversations
goes through `/api/chat` and, rendered by paperkiln, through `/api/generate` with `raw`.

| check | result |
|---|---|
| prompt bytes, /api/chat = raw = paperkiln's rendering | **60/60** (20 with a system message) |
| /api/chat prompt tokens = paperkiln's tokens of the rendering | **60/60** |
| 24-token greedy continuation, /api/chat = raw | 59/60 (conversation 24: a 1.2e-3-nat tie at token 13, `ollama/conversation24.md`) |
| /api/chat greedy on the 40 frozen prompts vs ember and mtstudio | **40/40** replies and prompt tokens; done "stop" 40/40; eval_count equal less the end-of-text token Ollama counts, 40/40 |

**Found and fixed: the Modelfile dropped the system message.** Ollama keeps a system message in `.Messages`
as well as in `.System`, so the Modelfile's fold, written for the first message (`eq $i 0`), never fired
once a system message was present (it sits at index 0, the first user message at 1): the first run gave
0/20 system conversations at paperkiln's length, each short by exactly the system text and its "

".
paperkiln 33c45899 folds into the first user message instead (a `$first` flag) and drops every
whitespace-trimming action, so the record's text is emitted exactly; the C++ twin in
`chat_template.hpp` is changed the same way (byte-identical, tests/test_chat_template.cpp), and the
rendering, the Jinja and the resolution are unchanged. The tables above are the fixed Modelfile's.
llama.cpp's `--jinja` path was never affected (the Jinja handles the system message itself, 60/60 above).

## Commands

From WSL, each under `python3 scripts/cpu_budget.py run --session paperkiln --cores 2 -- ...`:

    python3 llamacpp_gate.py --llamacpp ~/llamacpp_b6000_build/build/bin --out ~/cp_compat/out --threads 2
    python3 ollama_gate.py --ollama ~/ollama/bin/ollama --work ~/cp_ollama --out ~/cp_ollama/out --threads 2

Both resume by leg; receipts were copied into `llamacpp/` and `ollama/`.
