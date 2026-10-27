# A small random Mistral checkpoint with sliding_window 8 (head_dim 16), built with the model builder (fp32, cpu)
# and run with onnxruntime-genai; logits compared with transformers.
# usage: python mistral.py [BUILDER_PY]   (default: python -m onnxruntime_genai.models.builder)
import json
import os
import subprocess
import sys
import tempfile
from importlib.metadata import version

import numpy as np
import onnx
import torch
import transformers
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import MistralConfig, MistralForCausalLM, PreTrainedTokenizerFast

V, WINDOW = 256, 8
torch.manual_seed(0)
torch.set_num_threads(2)
cfg = MistralConfig(vocab_size=V, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                    num_key_value_heads=2, head_dim=16, sliding_window=WINDOW, max_position_embeddings=1024,
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
builder = [sys.executable, sys.argv[1]] if len(sys.argv) > 1 else [sys.executable, "-m", "onnxruntime_genai.models.builder"]
r = subprocess.run(builder + ["-i", d, "-o", out, "-p", "fp32", "-e", "cpu", "-c", os.path.join(d, "cache")],
                   capture_output=True, text=True)
print("transformers", transformers.__version__, "onnxruntime-genai", version("onnxruntime-genai"),
      "onnxruntime", version("onnxruntime"), "builder:", " ".join(builder[1:]))
if r.returncode != 0:
    print("builder failed:", "\n".join(r.stderr.strip().splitlines()[-4:]))
    sys.exit(0)
print("config.sliding_window:", model.config.sliding_window, " config.layer_types:", getattr(model.config, "layer_types", None))
for n in onnx.load(os.path.join(out, "model.onnx")).graph.node:
    if n.op_type == "GroupQueryAttention":
        print(n.name, "local_window_size =", [a.i for a in n.attribute if a.name == "local_window_size"])
print("genai_config sliding_window:",
      json.load(open(os.path.join(out, "genai_config.json")))["model"]["decoder"].get("sliding_window"))

import onnxruntime_genai as og                     # noqa: E402
ids = list(range(3, 51))
with torch.no_grad():
    want = model(torch.tensor([ids])).logits[0].numpy()
c = og.Config(out)
c.clear_providers()
m = og.Model(c)


def run(chunks):
    params = og.GeneratorParams(m)
    params.set_search_options(max_length=1024)
    g = og.Generator(m, params)
    rows = []
    for ch in chunks:
        try:
            g.append_tokens(ch)
        except Exception as e:
            return np.concatenate(rows) if rows else None, str(e).split("Status Message: ")[-1][:160]
        rows.append(np.array(g.get_output("logits"))[0])
    return np.concatenate(rows), None


def report(label, chunks):
    got, error = run(chunks)
    n = 0 if got is None else len(got)
    if n:
        err = np.abs(got - want[:n]).max(axis=-1) / np.abs(want[:n]).max()
        first = next((i for i, e in enumerate(err) if e > 1e-5), None)
        print(f"{label}: {n} positions, max |diff| / max |logit| {err.max():.3g}, first position past 1e-5: {first}, "
              f"argmax differs at {int((got.argmax(-1) != want[:n].argmax(-1)).sum())} of {n}")
    if error:
        print(f"{label}: error after {n} positions: {error}")


report("prompt of 8, then 1 token at a time to 48", [ids[:8]] + [[t] for t in ids[8:]])
report("prompt of 16", [ids[:16]])
report("prompt of 48", [ids])
