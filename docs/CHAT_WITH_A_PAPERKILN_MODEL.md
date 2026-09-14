# Chat with a paperkiln model

This folder holds a small language model trained by paperkiln
(https://github.com/JohnnyTeutonic/paperkiln), a C++ training stack
written from scratch, and two ways to talk to it. It was trained on a
CPU in 53 minutes: a two-layer Llama-style model, 128 wide, 0.95 million
parameters, 3,000 steps on a slice of TinyStories with a 4,096-word
vocabulary. Final validation loss 3.667 (`result.json` is the run's own
receipt; `spec.json` is the one file that described the whole run).

Expect children's-story English in lower case, a short attention span
and pronouns that wander. That is what a million parameters and an hour
of CPU buy. It is a real model, trained end to end by the stack, and
the point of the exercise is that it exists at all.

## Option A: Python only (recommended)

You need Python 3.9+ and two packages. Nothing to compile.

```bash
pip install transformers torch tokenizers
python hf_chat.py model
```

Type a prompt at `you>`; an empty line quits. One-shot instead:

```bash
python hf_chat.py model --prompt "once upon a time"
python hf_chat.py model --prompt "the dog saw a" --temp 0.8 --topk 40 --tokens 60
python hf_chat.py model --prompt "once upon a time" --temp 0     # greedy
```

Prompts are lower-cased for you (the vocabulary is lower case). Words
outside the 4,096-word vocabulary map to `<unk>`, so keep it simple:
`the cat`, `lily was sad because`, `tom and his mom went to`.

What is in `model/`: `config.json` (a standard Llama config),
`model.safetensors` (the weights, renamed to the Hugging Face layout)
and `tokenizer.json` (the run's word-level vocabulary). It opens with
`transformers.AutoModelForCausalLM.from_pretrained("model")` like any
Llama checkpoint, so you can ignore `hf_chat.py` entirely and write your
own loop. One thing to know if you do: the vocabulary has no real
end-of-sequence token (id 0 is `<unk>` and the config lists it as eos),
so `model.generate()` stops the moment it samples it. `hf_chat.py` bans
the special tokens at every step instead, which is the rule paperkiln's
own sampler uses; with `--temp 0` it reproduces that sampler token for
token (the repository's `tools/hf_export_verify.py` is the receipt for
that claim, and it passed on this model at its eight-token standard).

## Option B: the C++ engine, from the GGUF

`story3k.gguf` is the same model as a GGUF file with the vocabulary
embedded, exported by paperkiln itself. It runs in ember.cpp
(https://github.com/JohnnyTeutonic/ember.cpp), a separately written
inference engine:

```bash
git clone https://github.com/JohnnyTeutonic/ember.cpp && cd ember.cpp
mkdir build && cd build && cmake .. -DHAS_CUDA=OFF && make -j
./tinyllama ../../story3k.gguf ../../story3k.gguf 4 prompt "once upon a time" \
    --max-tokens 40 -ngl 0 --top-k 1 --raw-prompt
```

(The GGUF path is given twice: model and vocabulary are the same file.)
It needs CMake, a C++17 compiler, Boost and OpenMP; the engine's README
lists the packages per platform.

## Option C: train your own and chat with it

Clone paperkiln, build it (`cmake -S . -B build && cmake --build build`),
put a text corpus and a vocabulary GGUF where `specs/tinystories-llama-3k.json`
expects them, and run:

```bash
./build/mtstudio run specs/tinystories-llama-3k.json
python tools/hf_export.py <out_dir> --hf-dir <out_dir>/hf
python tools/hf_chat.py <out_dir>/hf
```

The run prints the ember.cpp command at the end, writes an `events.jsonl`
you can drop onto `studio/index.html` to watch the loss curve and the
per-layer gradient norms, and exports the same two artefacts you have
here.

## What the samples look like

Greedy (`--temp 0`), which is the deterministic reference:

    once upon a time there was a little girl named lily and he all the sun

Sampled (`--temp 0.8 --topk 40`, seed 2):

    tom and lily went to the market with their room in the room and said
    goodbye to the store like them and walks to the door and each other
    animals for my mom and he both enjoyed bubbles and smiles as he had
    the boy seen
