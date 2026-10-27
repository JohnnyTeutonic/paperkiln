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
