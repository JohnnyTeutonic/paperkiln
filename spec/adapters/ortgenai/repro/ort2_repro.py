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
