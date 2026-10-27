# A small random Gemma 3 text checkpoint (head_dim 16, 5 sliding + 1 global layer, linear RoPE factor 8 on the
# global layers, as in Gemma 3 4B/12B/27B), built with the model builder (fp32, cpu) and run with onnxruntime-genai;
# logits compared with transformers.
# usage: python gemma3.py [BUILDER_PY]   (default: python -m onnxruntime_genai.models.builder)
import os
import subprocess
import sys
import tempfile
from importlib.metadata import version

import numpy as np
import torch
import transformers
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import Gemma3ForCausalLM, Gemma3TextConfig, PreTrainedTokenizerFast

V, FACTOR = 256, 8.0
torch.manual_seed(0)
torch.set_num_threads(2)
tf5 = int(transformers.__version__.split(".")[0]) >= 5
kw = dict(vocab_size=V, hidden_size=64, intermediate_size=128, num_hidden_layers=6, num_attention_heads=4,
          num_key_value_heads=2, head_dim=16, query_pre_attn_scalar=16, sliding_window=512, max_position_embeddings=1024,
          pad_token_id=0, eos_token_id=1, bos_token_id=2)
if tf5:
    kw["rope_parameters"] = {"sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                             "full_attention": {"rope_type": "linear", "factor": FACTOR, "rope_theta": 1e6}}
else:
    kw.update(rope_theta=1e6, rope_local_base_freq=10000.0, rope_scaling={"rope_type": "linear", "factor": FACTOR})
model = Gemma3ForCausalLM(Gemma3TextConfig(**kw)).eval()
for p in model.parameters():                       # larger weights than the default init, so differences show
    p.data.normal_(0, 0.2)

d = tempfile.mkdtemp()
model.save_pretrained(d)
vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 2, **{f"w{i}": i for i in range(3, V)}}
t = Tokenizer(models.WordLevel(vocab, unk_token="<pad>"))
t.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
PreTrainedTokenizerFast(tokenizer_object=t, pad_token="<pad>", eos_token="<eos>", bos_token="<bos>").save_pretrained(d)

out = os.path.join(d, "onnx")
builder = [sys.executable, sys.argv[1]] if len(sys.argv) > 1 else [sys.executable, "-m", "onnxruntime_genai.models.builder"]
r = subprocess.run(builder + ["-i", d, "-o", out, "-p", "fp32", "-e", "cpu", "-c", os.path.join(d, "cache")],
                   capture_output=True, text=True)
print("transformers", transformers.__version__, "onnxruntime-genai", version("onnxruntime-genai"),
      "onnxruntime", version("onnxruntime"), "builder:", " ".join(builder[1:]))
if r.returncode != 0:
    print("builder failed:", "\n".join(r.stderr.strip().splitlines()[-4:]))
    sys.exit(0)

import onnx                                        # noqa: E402
from onnx import numpy_helper                      # noqa: E402
inits = {i.name: numpy_helper.to_array(i) for i in onnx.load(os.path.join(out, "model.onnx")).graph.initializer
         if "cos_cache" in i.name}
pos = np.arange(1024)[:, None]
for name, c in sorted(inits.items()):
    inv = 10000.0 ** (-np.arange(0, 16, 2) / 16) if "local" in name else 1e6 ** (-np.arange(0, 16, 2) / 16)
    unscaled = np.abs(c - np.cos(pos[:len(c)] * inv)).max()
    scaled = np.abs(c - np.cos(pos[:len(c)] * inv / FACTOR)).max()
    print(f"{name}: max |c - cos(p * inv_freq)| {unscaled:.2e}, max |c - cos(p * inv_freq / {FACTOR:g})| {scaled:.2e}")

import onnxruntime_genai as og                     # noqa: E402
ids = list(range(3, 51))
cfg = og.Config(out)
cfg.clear_providers()
m = og.Model(cfg)
params = og.GeneratorParams(m)
params.set_search_options(max_length=1024)
g = og.Generator(m, params)
g.append_tokens(ids)
got = np.array(g.get_output("logits"))[0]
with torch.no_grad():
    want = model(torch.tensor([ids])).logits[0].numpy()
err = np.abs(got - want).max(axis=-1) / np.abs(want).max()
print(f"onnxruntime-genai vs transformers, {len(ids)} tokens: max |diff| / max |logit| {err.max():.3g} "
      f"(position 0: {err[0]:.2e}), argmax differs at {int((got.argmax(-1) != want.argmax(-1)).sum())} of {len(ids)}")
