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
