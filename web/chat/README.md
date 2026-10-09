# paperkiln web chat

A chat page for paperkiln's small word-level models (the TinyChat specs) that
needs no server and no install. The model runs in the visitor's browser, in
plain JavaScript (`paperkiln.js`: no WASM, no build step, no CDN, no
dependencies). The whole site is a static folder, so any static host will do.

- `index.html` is the chat page. It looks and behaves like `mtstudio chat`
  (`studio/chat.html`) and adds a download progress bar and a note that
  everything runs on the visitor's device.
- `paperkiln.js` is the inference engine: a safetensors reader, the word
  tokeniser, the llama, flex and gpt2 decoders with a KV cache, the sampler,
  and the chat loop. It loads in a browser as `window.Paperkiln` and in Node
  through `require("./paperkiln.js")`.
- `firebase.json` is a minimal Firebase Hosting configuration.

## Export a trained model

Run this from the directory you trained in, so the spec's relative paths
(such as `data.vocab`) resolve:

```bash
python3 tools/export_web_chat.py runs/tinychat-quick site/
```

The result is self-contained:

```
site/index.html  site/paperkiln.js  site/firebase.json  site/.nojekyll
site/model/manifest.json      architecture, file names, sizes, sha256
site/model/vocab.json         token list in id order
site/model/<name>.safetensors the exported weights, unchanged
site/model/card.json          the model card (tools/model_card.py), if the run has one
```

The vocabulary is the one `mtstudio` uses: the spec's vocab GGUF, capped
at `data.vocab_cap`, or the run's exported `<name>.gguf` if the spec's file
cannot be found. gpt2-family runs export no GGUF,
so run the exporter from the training directory or pass `--vocab
releases/tinychat_vocab.gguf`. The exporter fails, with a reason, when the
vocabulary does not match the weights or the attention type is not yet
supported in the browser. Exact attention is supported in the llama, flex
and gpt2 families; kimi, srd, attnres and sliding-window attention are not.

The page reads `model/manifest.json` and fetches the weights and
vocabulary it names. Sizes: tinychat-quick is about 1.8 MB in total; the
gpt2-small preset with the TinyChat vocabulary is about 13 MB.

## Try it locally

Browsers will not fetch the model from a `file://` page, so serve the
folder:

```bash
cd site && python3 -m http.server 8000
# open http://127.0.0.1:8000/
```

## Host it

**GitHub Pages.** Either commit the folder's contents as `docs/` on your
default branch and choose *Settings → Pages → Deploy from a branch →
main, /docs*, or publish it on its own branch:

```bash
cd site
git init -b gh-pages && git add -A && git commit -m "paperkiln web chat"
git remote add origin git@github.com:<you>/<repo>.git
git push -f origin gh-pages      # then Settings → Pages → gh-pages, / (root)
```

The `.nojekyll` file stops Pages from processing the folder. Each file must
stay under GitHub's 100 MB limit, which is far above these models.

**Firebase Hosting.** With the Firebase CLI installed and a project
created:

```bash
cd site
firebase deploy --only hosting --project <your-project-id>
```

`firebase.json` serves the folder as it is (`"public": "."`) and caches
`model/` for an hour.

**Anything else.** Upload the folder's contents to any static host or
object store that serves files over HTTP(S).

## Behaviour

The prompt is mtstudio's chat template, `user: <text> assistant:`, with up
to the last eight history entries before it in the same form. Replies are
greedy at temperature 0 (the default). Above 0, the page samples with
mtstudio chat's top-k 40 and top-p 0.9. A reply stops at the end-of-text
token, when the model opens a new turn (`user :` or another `assistant :`,
which is dropped), or at the reply-length limit. The text is joined the way
the engine joins it, with `.,!?;:)` attached to the word before. Context
beyond the model's window `T` slides exactly as it does in the engine.

## Parity

`tools/test_web_chat_parity.py` exports a run, then compares the
JavaScript engine under Node with `mtstudio sample --temp 0 --topk 1`
token for token, and with `mtstudio chat` replies at temperature 0:

```bash
python3 tools/test_web_chat_parity.py --run runs/tinychat-quick --mtstudio build/mtstudio
```

It needs `node` on `PATH`. If you change an op in the C++ engine, change
`paperkiln.js` to match and run this test again.
