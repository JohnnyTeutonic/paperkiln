**Describe the bug**

With transformers 5.x, the model builder cannot build any Gemma 3 text model. `Model.make_rope_init` reads `config.rope_parameters["rope_type"]`, but transformers 5 stores Gemma 3's RoPE per layer type:

```
{"sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
 "full_attention":    {"rope_type": "linear", "factor": 8.0, "rope_theta": 1000000.0}}
```

so the build stops with `KeyError: 'rope_type'`. This happens both for a `config.json` written by transformers 5 and for the transformers 4 form Google ships (`rope_theta`, `rope_local_base_freq`, `rope_scaling`), which transformers 5 converts to the nested form on load.

Code (main at 6b7ff3c):
- `make_rope_init` indexes the flat key: [builders/base.py#L706](https://github.com/microsoft/onnxruntime-genai/blob/6b7ff3ca3efa15da219050fdb4ecf8bed67875af/src/python/py/models/builders/base.py#L706) (0.18.0: line 670).
- `Gemma3Model` has no `make_rope_init` of its own: [builders/gemma.py#L122-L127](https://github.com/microsoft/onnxruntime-genai/blob/6b7ff3ca3efa15da219050fdb4ecf8bed67875af/src/python/py/models/builders/gemma.py#L122-L127). `Gemma4Model.make_rope_init` already reads the nested form ([gemma.py#L188](https://github.com/microsoft/onnxruntime-genai/blob/6b7ff3ca3efa15da219050fdb4ecf8bed67875af/src/python/py/models/builders/gemma.py#L188)).

Under transformers 4.57.6 the 0.18.0 builder does not import (`cannot import name 'Qwen3_5ForConditionalGeneration' from 'transformers'`, builders/base.py line 29), and the wheel declares no transformers requirement, so with 0.18.0 there is no environment in which Gemma 3 builds. On main the import is lazy (`resolve_transformers_class`) and Gemma 3 builds under transformers 4.57.6. (Separately, on main under transformers 4.57.6 a plain Llama checkpoint fails in `make_config_init` with `TypeError: 'NoneType' object is not iterable`: transformers 4 sets `config.rope_scaling = None` and [base.py#L499-L502](https://github.com/microsoft/onnxruntime-genai/blob/6b7ff3ca3efa15da219050fdb4ecf8bed67875af/src/python/py/models/builders/base.py#L499-L502) iterates it.)

**To Reproduce**

A small random Gemma 3 text checkpoint (6 layers, 5 sliding and 1 global, head_dim 16, linear RoPE factor 8 on the global layers as in Gemma 3 4B/12B/27B), saved with a WordLevel tokenizer and passed to the builder:

```python
import os, subprocess, sys, tempfile
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import Gemma3ForCausalLM, Gemma3TextConfig, PreTrainedTokenizerFast

V = 256
cfg = Gemma3TextConfig(
    vocab_size=V, hidden_size=64, intermediate_size=128, num_hidden_layers=6, num_attention_heads=4,
    num_key_value_heads=2, head_dim=16, query_pre_attn_scalar=16, sliding_window=512, max_position_embeddings=1024,
    pad_token_id=0, eos_token_id=1, bos_token_id=2,
    rope_parameters={"sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                     "full_attention": {"rope_type": "linear", "factor": 8.0, "rope_theta": 1e6}})
d = tempfile.mkdtemp()
Gemma3ForCausalLM(cfg).save_pretrained(d)
vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 2, **{f"w{i}": i for i in range(3, V)}}
t = Tokenizer(models.WordLevel(vocab, unk_token="<pad>"))
t.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
PreTrainedTokenizerFast(tokenizer_object=t, pad_token="<pad>", eos_token="<eos>", bos_token="<bos>").save_pretrained(d)
subprocess.run([sys.executable, "-m", "onnxruntime_genai.models.builder", "-i", d, "-o", os.path.join(d, "onnx"),
                "-p", "fp32", "-e", "cpu", "-c", os.path.join(d, "cache")])
```

Output (onnxruntime-genai 0.18.0, transformers 5.20.2; the same with main's `builder.py`, at base.py line 706):

```
  File ".../onnxruntime_genai/models/builders/base.py", line 670, in make_rope_init
    if config.rope_parameters["rope_type"] == "yarn":
       ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^
KeyError: 'rope_type'
```

The same checkpoint with onnxruntime-genai 0.18.0 and transformers 4.57.6:

```
ImportError: cannot import name 'Qwen3_5ForConditionalGeneration' from 'transformers'
```

**Expected behavior**

A model, as for Gemma 4; or, if transformers 5 is not supported for Gemma 3, a clear error and a declared transformers requirement.

**Suggested fix**

Give `Gemma3Model` a `make_rope_init` that reads the per-layer-type entries, as `Gemma4Model` does: the `full_attention` entry for the global caches, and the `sliding_attention` entry (its own theta and `rope_type`, "default") for the local ones. Flattening `rope_parameters` to its `full_attention` entry would make the build run but would also carry the global layers' linear factor into the local caches; that is a separate bug, #2789.

**Desktop**
 - OS: Ubuntu 24.04 (WSL2), x86_64, CPU execution provider
 - onnxruntime-genai 0.18.0 (PyPI) and main at 6b7ff3c (builder.py run from the source tree), onnxruntime 1.31.1
 - transformers 5.20.2 and 4.57.6, torch 2.14.1+cpu, onnx 1.23.2, Python 3.12.3

**Additional context**

Found by a conformance test suite. The analysis and this write-up were done with an AI assistant (Claude); the reproduction was run and its output checked.
