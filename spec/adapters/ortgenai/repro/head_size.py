# Small random Llama checkpoints with head_dim 8, 16, 24, 32, built with the model builder (fp32, cpu) and run for
# one prompt with onnxruntime-genai.
# usage: python head_size.py [BUILDER_PY]   (default: python -m onnxruntime_genai.models.builder)
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
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

import onnxruntime_genai as og

V = 256
torch.manual_seed(0)
torch.set_num_threads(2)
builder = [sys.executable, sys.argv[1]] if len(sys.argv) > 1 else [sys.executable, "-m", "onnxruntime_genai.models.builder"]
print("transformers", transformers.__version__, "onnxruntime-genai", version("onnxruntime-genai"),
      "onnxruntime", version("onnxruntime"), "builder:", " ".join(builder[1:]))
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
    r = subprocess.run(builder + ["-i", d, "-o", out, "-p", "fp32", "-e", "cpu", "-c", os.path.join(d, "cache")],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"head_dim {head_dim}: build failed:", r.stderr.strip().splitlines()[-1])
        continue
    rot = [a.i for n in onnx.load(os.path.join(out, "model.onnx")).graph.node if n.op_type == "GroupQueryAttention"
           for a in n.attribute if a.name == "do_rotary"]
    c = og.Config(out)
    c.clear_providers()
    m = og.Model(c)
    p = og.GeneratorParams(m)
    p.set_search_options(max_length=64)
    g = og.Generator(m, p)
    try:
        g.append_tokens(list(range(3, 11)))
        got = np.array(g.get_output("logits"))[0]
        with torch.no_grad():
            want = model(torch.tensor([list(range(3, 11))])).logits[0].numpy()
        print(f"head_dim {head_dim}: builds (GroupQueryAttention do_rotary={rot}), runs; "
              f"max |diff| vs transformers {np.abs(got - want).max():.2e}")
    except Exception as e:
        print(f"head_dim {head_dim}: builds (GroupQueryAttention do_rotary={rot}), first append_tokens fails: "
              + str(e).split("Status Message: ")[-1].strip())
