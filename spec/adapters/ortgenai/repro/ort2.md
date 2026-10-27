**Describe the bug**

For Gemma 3 checkpoints with `rope_scaling` (4B, 12B and 27B: linear, factor 8, meant for the global layers), the model builder also applies the linear factor to the RoPE caches of the sliding-window (local) layers. transformers uses unscaled default RoPE with `rope_local_base_freq` on the local layers, so five of every six layers rotate at one eighth of their trained frequency at every position after the first.

`Gemma3Model.make_rotary_embedding_multi_cache` builds the global caches, then changes `rope_attrs["theta"]` to `rope_local_base_freq` and builds the local caches, leaving `rope_attrs["rescale_factors"]` (set from the linear factor in `make_rope_init`) in place:
- [builders/gemma.py#L134-L147](https://github.com/microsoft/onnxruntime-genai/blob/6b7ff3ca3efa15da219050fdb4ecf8bed67875af/src/python/py/models/builders/gemma.py#L134-L147) (main at 6b7ff3c; the same in 0.18.0 and 0.15.2)
- the factor: [builders/base.py#L730-L734](https://github.com/microsoft/onnxruntime-genai/blob/6b7ff3ca3efa15da219050fdb4ecf8bed67875af/src/python/py/models/builders/base.py#L730-L734); used in `make_rotary_embedding_caches_from_scratch`, [base.py#L3489-L3493](https://github.com/microsoft/onnxruntime-genai/blob/6b7ff3ca3efa15da219050fdb4ecf8bed67875af/src/python/py/models/builders/base.py#L3489-L3493)

This may be the cause of #2277 (Gemma 3 4B export diverging from transformers at layer 0, a sliding-attention layer), though I have not run that checkpoint.

**To Reproduce**

Gemma 3 builds only under transformers 4 at present (#2788: `KeyError: 'rope_type'` under transformers 5), so this uses transformers 4.57.6 with main's builder (or onnxruntime-genai 0.15.2's). A small random Gemma 3 text checkpoint (head_dim 16, 5 sliding + 1 global layer, linear factor 8), built for CPU fp32, run for 48 tokens and compared with transformers; the script also checks the `cos_cache_local` initialiser against the unscaled and scaled formulas.

```python
# usage: python repro.py [path/to/builder.py]   (default: python -m onnxruntime_genai.models.builder)
import os, subprocess, sys, tempfile
import numpy as np, onnx, torch
from onnx import numpy_helper
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import Gemma3ForCausalLM, Gemma3TextConfig, PreTrainedTokenizerFast
import onnxruntime_genai as og

V, F = 256, 8.0
torch.manual_seed(0)
cfg = Gemma3TextConfig(  # transformers 4 field names, as Google ships Gemma 3
    vocab_size=V, hidden_size=64, intermediate_size=128, num_hidden_layers=6, num_attention_heads=4,
    num_key_value_heads=2, head_dim=16, query_pre_attn_scalar=16, sliding_window=512, max_position_embeddings=1024,
    pad_token_id=0, eos_token_id=1, bos_token_id=2,
    rope_theta=1e6, rope_local_base_freq=10000.0, rope_scaling={"rope_type": "linear", "factor": F})
model = Gemma3ForCausalLM(cfg).eval()
for p in model.parameters():
    p.data.normal_(0, 0.2)
d = tempfile.mkdtemp()
model.save_pretrained(d)
vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 2, **{f"w{i}": i for i in range(3, V)}}
t = Tokenizer(models.WordLevel(vocab, unk_token="<pad>"))
t.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
PreTrainedTokenizerFast(tokenizer_object=t, pad_token="<pad>", eos_token="<eos>", bos_token="<bos>").save_pretrained(d)
out = os.path.join(d, "onnx")
builder = [sys.argv[1]] if len(sys.argv) > 1 else ["-m", "onnxruntime_genai.models.builder"]
subprocess.run([sys.executable, *builder, "-i", d, "-o", out, "-p", "fp32", "-e", "cpu", "-c", os.path.join(d, "c")],
               check=True, capture_output=True)

pos = np.arange(1024)[:, None]
for i in onnx.load(os.path.join(out, "model.onnx")).graph.initializer:
    if "cos_cache" in i.name:
        c = numpy_helper.to_array(i)
        inv = (1e4 if "local" in i.name else 1e6) ** (-np.arange(0, 16, 2) / 16)
        print(f"{i.name}: vs cos(p * inv_freq) {np.abs(c - np.cos(pos * inv)).max():.2e}, "
              f"vs cos(p * inv_freq / {F:g}) {np.abs(c - np.cos(pos * inv / F)).max():.2e}")

ids = list(range(3, 51))
c = og.Config(out)
c.clear_providers()
m = og.Model(c)
params = og.GeneratorParams(m)
params.set_search_options(max_length=1024)
g = og.Generator(m, params)
g.append_tokens(ids)
got = np.array(g.get_output("logits"))[0]
with torch.no_grad():
    want = model(torch.tensor([ids])).logits[0].numpy()
err = np.abs(got - want).max(axis=-1) / np.abs(want).max()
print(f"vs transformers: max |diff| / max |logit| {err.max():.3g} (position 0: {err[0]:.2e}), "
      f"argmax differs at {int((got.argmax(-1) != want.argmax(-1)).sum())} of {len(ids)} positions")
```

Output, transformers 4.57.6, main's builder (6b7ff3c), onnxruntime-genai 0.18.0 runtime:

```
cos_cache_global: vs cos(p * inv_freq) 2.00e+00, vs cos(p * inv_freq / 8) 1.12e-06
cos_cache_local: vs cos(p * inv_freq) 2.00e+00, vs cos(p * inv_freq / 8) 2.02e-06
vs transformers: max |diff| / max |logit| 0.519 (position 0: 2.23e-07), argmax differs at 16 of 48 positions
```

onnxruntime-genai 0.15.2 (its own builder and runtime) gives the same caches and the same 0.519.

With one line added to `make_rotary_embedding_multi_cache`, `self.rope_attrs["rescale_factors"] = 1` after the local theta is set:

```
cos_cache_global: vs cos(p * inv_freq) 2.00e+00, vs cos(p * inv_freq / 8) 1.12e-06
cos_cache_local: vs cos(p * inv_freq) 1.67e-05, vs cos(p * inv_freq / 8) 2.00e+00
vs transformers: max |diff| / max |logit| 5.45e-07 (position 0: 2.23e-07), argmax differs at 0 of 48 positions
```

**Expected behavior**

`cos_cache_local`/`sin_cache_local` built from `rope_local_base_freq` without the global layers' scaling, and logits within float32 rounding of transformers.

**Suggested fix**

Reset `rope_attrs["rescale_factors"] = 1` (and any other global-only scaling state) before the local caches are made, or build the local caches from the `sliding_attention` RoPE parameters (transformers 5's per-layer-type form; see #2788).

**Desktop**
 - OS: Ubuntu 24.04 (WSL2), x86_64, CPU execution provider
 - onnxruntime-genai main at 6b7ff3c (builder.py from the source tree) with the 0.18.0 runtime; also 0.15.2. onnxruntime 1.31.1
 - transformers 4.57.6, torch 2.14.1+cpu, onnx 1.23.2, Python 3.12.3

**Additional context**

Found by a conformance test suite. The analysis and this write-up were done with an AI assistant (Claude); the reproduction was run and its output checked.
