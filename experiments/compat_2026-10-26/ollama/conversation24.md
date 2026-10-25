# Conversation 24: /api/chat and the raw rendering part at a near tie

Both prompts are 266 bytes and 55 tokens (paperkiln's rendering: 266 bytes, 55 tokens). The
24-token greedy continuations agree for 12 tokens, then /api/chat continues " when you are feeling
down." and /api/generate (raw, sent second, so Ollama's prompt cache served its prefix) " with the sun
shining bri...". llama.cpp b6000 on the same F32 file, from position 0 (no cache, f32 KV), gives
"Yes, it is quite surprising to see you here, especially when you are feeling down.", the /api/chat
reply, and its top two at that step are " when" -1.40692 and " with" -1.40808: a margin of 1.2e-3
nats, which a different batch split of the same prefix can flip. (`conversation24_probe.py`:
llama-server /completion with n_probs 2 along its own greedy path; the only other step under
0.05 nats is step 2, " it" -0.92794 against " I" -0.96568.)
