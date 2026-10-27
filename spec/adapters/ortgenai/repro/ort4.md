**Describe the bug**

On CPU fp32 the model builder always emits `GroupQueryAttention` with `do_rotary=1` (`is_gqa_supported` includes `("cpu", FLOAT)`; `is_fused_rope_supported` is true for every EP but DML). onnxruntime's GQA rotary path requires `head_size % 16 == 0` (`CheckRotaryCaches`), so a model whose head size is a multiple of 8 but not of 16 (8, 24, 40, 56, 72, ...) builds and loads, then fails at the first `append_tokens` with `head_size shall be a multiple of 16`. The same kernel runs these head sizes with `do_rotary=0` and a separate `RotaryEmbedding` node.

- builder: [builders/base.py#L739-L750](https://github.com/microsoft/onnxruntime-genai/blob/6b7ff3ca3efa15da219050fdb4ecf8bed67875af/src/python/py/models/builders/base.py#L739-L750) and [base.py#L780-L782](https://github.com/microsoft/onnxruntime-genai/blob/6b7ff3ca3efa15da219050fdb4ecf8bed67875af/src/python/py/models/builders/base.py#L780-L782) (main at 6b7ff3c)
- kernel check: onnxruntime [group_query_attention_helper.h#L195-L199](https://github.com/microsoft/onnxruntime/blob/a6f085c3faeaaf43fa45a5153a3b416c492b7542/onnxruntime/contrib_ops/cpu/bert/group_query_attention_helper.h#L195-L199) (main at a6f085c; onnxruntime 1.31.1 behaves the same)

Common head sizes (64, 80, 96, 128, 256) are unaffected; small test checkpoints and any model with such a head size are.

**To Reproduce**

Small random Llama checkpoints with head_dim 8, 16, 24 and 32, each built for CPU fp32 and run on an 8-token prompt:

```python
# usage: python repro.py [path/to/builder.py]   (default: python -m onnxruntime_genai.models.builder)
import os, subprocess, sys, tempfile
import numpy as np, onnx, torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
import onnxruntime_genai as og

V = 256
torch.manual_seed(0)
builder = [sys.argv[1]] if len(sys.argv) > 1 else ["-m", "onnxruntime_genai.models.builder"]
for head_dim in (8, 16, 24, 32):
    cfg = LlamaConfig(vocab_size=V, hidden_size=4 * head_dim, intermediate_size=128, num_hidden_layers=1,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=head_dim, max_position_embeddings=256,
                      pad_token_id=0, eos_token_id=1, bos_token_id=2)
    model = LlamaForCausalLM(cfg).eval()
    d = tempfile.mkdtemp()
    model.save_pretrained(d)
    vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 2, **{f"w{i}": i for i in range(3, V)}}
    t = Tokenizer(models.WordLevel(vocab, unk_token="<pad>"))
    t.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    PreTrainedTokenizerFast(tokenizer_object=t, pad_token="<pad>", eos_token="<eos>", bos_token="<bos>").save_pretrained(d)
    out = os.path.join(d, "onnx")
    subprocess.run([sys.executable, *builder, "-i", d, "-o", out, "-p", "fp32", "-e", "cpu", "-c", os.path.join(d, "c")],
                   check=True, capture_output=True)
    rot = [a.i for n in onnx.load(os.path.join(out, "model.onnx")).graph.node if n.op_type == "GroupQueryAttention"
           for a in n.attribute if a.name == "do_rotary"]
    c = og.Config(out)
    c.clear_providers()
    m = og.Model(c)
    p = og.GeneratorParams(m)
    p.set_search_options(max_length=64)
    g = og.Generator(m, p)
    ids = list(range(3, 11))
    try:
        g.append_tokens(ids)
        with torch.no_grad():
            want = model(torch.tensor([ids])).logits[0].numpy()
        print(f"head_dim {head_dim}: do_rotary={rot}, runs, max |diff| vs transformers "
              f"{np.abs(np.array(g.get_output('logits'))[0] - want).max():.2e}")
    except Exception as e:
        print(f"head_dim {head_dim}: do_rotary={rot}, first append_tokens fails: {str(e).split('Status Message: ')[-1].strip()}")
```

Output, onnxruntime-genai 0.18.0, onnxruntime 1.31.1, transformers 5.20.2 (main's builder.py gives the same; so does 0.15.2 with transformers 4.57.6):

```
head_dim 8: do_rotary=[1], first append_tokens fails: head_size shall be a multiple of 16. Got head_size % 16 == 8
head_dim 16: do_rotary=[1], runs, max |diff| vs transformers 1.79e-07
head_dim 24: do_rotary=[1], first append_tokens fails: head_size shall be a multiple of 16. Got head_size % 16 == 8
head_dim 32: do_rotary=[1], runs, max |diff| vs transformers 3.58e-07
```

With the builder changed to use a separate `RotaryEmbedding` node when `head_size % 16 != 0` (`is_fused_rope_supported` returning false for those head sizes, plus the existing `fuse_qkv=false` option, since the packed Q/K/V path leaves the `RotaryEmbedding` inputs empty):

```
head_dim 8: do_rotary=[0], runs, max |diff| vs transformers 1.19e-07
head_dim 16: do_rotary=[1], runs, max |diff| vs transformers 1.79e-07
head_dim 24: do_rotary=[0], runs, max |diff| vs transformers 2.24e-07
head_dim 32: do_rotary=[1], runs, max |diff| vs transformers 3.58e-07
```

**Expected behavior**

A model that runs (separate `RotaryEmbedding` when the fused path cannot take the head size), or an error at build time.

**Suggested fix**

In the builder, on CPU, fall back to an unfused `RotaryEmbedding` with unpacked Q/K/V when `head_size % 16 != 0`, and refuse at build time when `head_size % 8 != 0` (the kernel's other check); or lift the restriction in the kernel.

**Desktop**
 - OS: Ubuntu 24.04 (WSL2), x86_64, CPU execution provider
 - onnxruntime-genai 0.18.0 (PyPI) and main at 6b7ff3c (builder.py from the source tree); also 0.15.2. onnxruntime 1.31.1
 - transformers 5.20.2 (and 4.57.6 with 0.15.2), torch 2.14.1+cpu, onnx 1.23.2, Python 3.12.3

**Additional context**

Found by a conformance test suite. The analysis and this write-up were done with an AI assistant (Claude); the reproduction was run and its output checked.
