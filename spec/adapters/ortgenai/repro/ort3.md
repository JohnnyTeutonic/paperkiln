**Describe the bug**

For Mistral checkpoints with `sliding_window` set (Mistral 7B v0.1 and models built on it: 4096), the model builder no longer applies the window, while genai_config still describes a window-sized KV cache. The result on CPU: logits past the window are those of a windowless model, and once the sequence passes the window-sized cache's capacity (24 positions here, with window 8) `append_tokens` fails; a 48-token prompt fails at once. onnxruntime-genai 0.15.2 wrote `local_window_size = sliding_window` on every Mistral layer and matches transformers.

The window is now kept only for layers whose `config.layer_types` entry is `"sliding_attention"`; `MistralConfig` has no `layer_types` (transformers 4.57.6 and 5.20.2), the builder's default is all `"full_attention"`, and so every layer gets `local_window_size = -1`:
- default layer types: [builders/base.py#L131-L135](https://github.com/microsoft/onnxruntime-genai/blob/6b7ff3ca3efa15da219050fdb4ecf8bed67875af/src/python/py/models/builders/base.py#L131-L135)
- `is_local`: [base.py#L1562-L1564](https://github.com/microsoft/onnxruntime-genai/blob/6b7ff3ca3efa15da219050fdb4ecf8bed67875af/src/python/py/models/builders/base.py#L1562-L1564)
- window reset per layer: [base.py#L4760-L4761](https://github.com/microsoft/onnxruntime-genai/blob/6b7ff3ca3efa15da219050fdb4ecf8bed67875af/src/python/py/models/builders/base.py#L4760-L4761)
- genai_config still gets a `sliding_window` section (with `"layers": []`) because `window_size > 0`: [base.py#L1451-L1463](https://github.com/microsoft/onnxruntime-genai/blob/6b7ff3ca3efa15da219050fdb4ecf8bed67875af/src/python/py/models/builders/base.py#L1451-L1463)

(main at 6b7ff3c; the same in 0.18.0.)

**To Reproduce**

A small random Mistral checkpoint with `sliding_window=8` and head_dim 16, built for CPU fp32, then run three ways and compared with transformers (which applies the window):

```python
# usage: python repro.py [path/to/builder.py]   (default: python -m onnxruntime_genai.models.builder)
import json, os, subprocess, sys, tempfile
import numpy as np, onnx, torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import MistralConfig, MistralForCausalLM, PreTrainedTokenizerFast
import onnxruntime_genai as og

V = 256
torch.manual_seed(0)
cfg = MistralConfig(vocab_size=V, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                    num_key_value_heads=2, head_dim=16, sliding_window=8, max_position_embeddings=1024,
                    pad_token_id=0, eos_token_id=1, bos_token_id=2)
model = MistralForCausalLM(cfg).eval()
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
for n in onnx.load(os.path.join(out, "model.onnx")).graph.node:
    if n.op_type == "GroupQueryAttention":
        print(n.name, "local_window_size", [a.i for a in n.attribute if a.name == "local_window_size"])
print("genai_config sliding_window:", json.load(open(os.path.join(out, "genai_config.json")))["model"]["decoder"].get("sliding_window"))

ids = list(range(3, 51))
with torch.no_grad():
    want = model(torch.tensor([ids])).logits[0].numpy()
c = og.Config(out)
c.clear_providers()
m = og.Model(c)

def run(label, chunks):
    params = og.GeneratorParams(m)
    params.set_search_options(max_length=1024)
    g = og.Generator(m, params)
    rows, error = [], None
    for ch in chunks:
        try:
            g.append_tokens(ch)
        except Exception as e:
            error = str(e).split("Status Message: ")[-1][:150]
            break
        rows.append(np.array(g.get_output("logits"))[0])
    if rows:
        got = np.concatenate(rows)
        n = len(got)
        err = np.abs(got - want[:n]).max(axis=-1) / np.abs(want[:n]).max()
        first = next((i for i, e in enumerate(err) if e > 1e-5), None)
        print(f"{label}: {n} positions, max |diff| / max |logit| {err.max():.3g}, first position past 1e-5: {first}, "
              f"argmax differs at {int((got.argmax(-1) != want[:n].argmax(-1)).sum())}")
    if error:
        print(f"{label}: append_tokens failed after {len(rows) and len(np.concatenate(rows))} positions: {error}")

run("prompt of 16", [ids[:16]])
run("prompt of 8, then one token at a time", [ids[:8]] + [[t] for t in ids[8:]])
run("prompt of 48", [ids])
```

Output, onnxruntime-genai 0.18.0, transformers 5.20.2 (main's builder.py gives the same):

```
/model/layers.0/attn/GroupQueryAttention local_window_size [-1]
/model/layers.1/attn/GroupQueryAttention local_window_size [-1]
genai_config sliding_window: {'window_size': 8, 'slide_key_value_cache': False, 'slide_inputs': False, 'layers': [], 'cache_slack': 0}
prompt of 16: 16 positions, max |diff| / max |logit| 0.556, first position past 1e-5: 8, argmax differs at 7
prompt of 8, then one token at a time: 24 positions, max |diff| / max |logit| 0.677, first position past 1e-5: 8, argmax differs at 14
prompt of 8, then one token at a time: append_tokens failed after 24 positions: /onnxruntime_src/onnxruntime/core/framework/op_kernel.cc:87 virtual OrtValue* onnxruntime::OpKernelContext::OutputMLValue(int, const onnxruntime::Tens
prompt of 48: append_tokens failed after 0 positions: total_sequence_length must equal past_sequence_length + kv_sequence_length for a dynamic cache, or be no greater than past_sequence_length for a stati
```

onnxruntime-genai 0.15.2, transformers 4.57.6, the same script:

```
/model/layers.0/attn/GroupQueryAttention local_window_size [8]
/model/layers.1/attn/GroupQueryAttention local_window_size [8]
genai_config sliding_window: None
prompt of 16: 16 positions, max |diff| / max |logit| 3.48e-07, first position past 1e-5: None, argmax differs at 0
prompt of 8, then one token at a time: 48 positions, max |diff| / max |logit| 3.68e-07, first position past 1e-5: None, argmax differs at 0
prompt of 48: 48 positions, max |diff| / max |logit| 3.43e-07, first position past 1e-5: None, argmax differs at 0
```

**Expected behavior**

`local_window_size = sliding_window` on every layer of a Mistral model (as 0.15.2 wrote), and logits within float32 rounding of transformers at every position.

**Suggested fix**

Where `config.layer_types` is absent and `sliding_window` is set (Mistral, and any configuration class like it), treat every layer as sliding, as before, so that the attention nodes and the genai_config `sliding_window.layers` list agree.

**Desktop**
 - OS: Ubuntu 24.04 (WSL2), x86_64, CPU execution provider
 - onnxruntime-genai 0.18.0 (PyPI) and main at 6b7ff3c (builder.py from the source tree); 0.15.2 for comparison. onnxruntime 1.31.1
 - transformers 5.20.2 and 4.57.6, torch 2.14.1+cpu, onnx 1.23.2, Python 3.12.3

**Additional context**

Found by a conformance test suite. The analysis and this write-up were done with an AI assistant (Claude); the reproduction was run and its output checked.
