# Specs gallery

A run is one file; these are known-good starting points. Copy one, point
`data.corpus` / `data.vocab` at your files, and:

```bash
./mtstudio plan specs/<spec>.json     # resolve + print, no execution
./mtstudio run  specs/<spec>.json     # the full lifecycle
./mtstudio serve <out_dir> 8080       # live dashboard on a running job
```

| Spec | What it demonstrates |
|---|---|
| [tinychat-quick.json](tinychat-quick.json) | The chat quickstart's Quick model: llama-tiny on TinyChat v2 (about 450,000 parameters, about 10 minutes on a laptop CPU) |
| [tinychat-better.json](tinychat-better.json) | The quickstart's Better model: llama family at d=192, 3 layers, 6 heads (several hours on a CPU) |
| [tinystories-nano.json](tinystories-nano.json) | The 5-minute smoke: gpt2-nano, 200 steps, safetensors out |
| [tinystories-llama.json](tinystories-llama.json) | The full loop: llama-tiny + early stopping + GGUF with embedded vocab, ready for tinyllama.cpp |
| [tinystories-llama-3k.json](tinystories-llama-3k.json) | The syntax run: 3,000 steps, batch+accum, val 4.64→3.68 (transcript in the README) |
| [tinystories-muon.json](tinystories-muon.json) | The Muon hybrid optimizer: per-head on qkv, full-matrix on hidden, AdamW on the rest |
| [sweep-lr-seeds.json](sweep-lr-seeds.json) | An mtsweep description, not a run spec: lr × seeds grid with per-cell seed statistics (`python tools/mtsweep.py specs/sweep-lr-seeds.json`) |

Spec-builder note: `studio/index.html` writes these interactively — the
form's output is exactly this format.

Field reference: the annotated block in the
[top-level README](../README.md#custom-configuration--the-spec-format).
