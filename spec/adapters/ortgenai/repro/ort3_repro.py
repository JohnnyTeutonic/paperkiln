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
