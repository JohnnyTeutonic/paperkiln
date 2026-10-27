# Three small GGUF config-mapping gaps: Qwen3 head_dim, Mistral detection, GPT-2 n_positions.
# Needs: pip install gguf
import os
import tempfile
from importlib.metadata import version

import gguf
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM

rng = np.random.default_rng(0)
w = lambda *s: (rng.standard_normal(s) * 0.2).astype(np.float32)
one = lambda n: np.ones(n, np.float32)
E, H, KV, FF, V = 64, 4, 2, 128, 256


def write(arch, meta, tensors, name=None):
    d = tempfile.mkdtemp()
    g = gguf.GGUFWriter(os.path.join(d, "m.gguf"), arch)
    if name:
        g.add_name(name)
    for f, *a in meta:
        getattr(g, f)(*a)
    g.add_vocab_size(V); g.add_tokenizer_model("no_vocab")
    for n, t in tensors:
        g.add_tensor(n, t)
    g.write_header_to_file(); g.write_kv_data_to_file(); g.write_tensors_to_file(); g.close()
    return d


def attn_llama(D):
    t = [("token_embd.weight", w(V, E))]
    for n, s in [("attn_norm", one(E)), ("attn_q", w(H * D, E)), ("attn_k", w(KV * D, E)), ("attn_v", w(KV * D, E)),
                 ("attn_output", w(E, H * D)), ("ffn_norm", one(E)), ("ffn_gate", w(FF, E)), ("ffn_up", w(FF, E)),
                 ("ffn_down", w(E, FF))]:
        t.append(("blk.0." + n + ".weight", s))
    return t + [("output_norm.weight", one(E)), ("output.weight", w(V, E))]


def load(d):
    return AutoModelForCausalLM.from_pretrained(d, gguf_file="m.gguf", dtype=torch.float32).eval()


def forward(m):
    with torch.no_grad():
        return m(torch.tensor([list(range(1, 9))])).logits


common = [("add_context_length", 4096), ("add_embedding_length", E), ("add_block_count", 1),
          ("add_feed_forward_length", FF), ("add_head_count", H), ("add_head_count_kv", KV),
          ("add_layer_norm_rms_eps", 1e-6), ("add_rope_freq_base", 1e6)]
print("transformers", transformers.__version__, "gguf", version("gguf"), "torch", torch.__version__)

# (a) Qwen3 with head_dim 32 (written as qwen3.attention.key_length, as convert_hf_to_gguf.py does)
D = 32
t = attn_llama(D) + [("blk.0.attn_q_norm.weight", one(D)), ("blk.0.attn_k_norm.weight", one(D))]
m = load(write("qwen3", common + [("add_key_length", D), ("add_value_length", D)], t))
print("(a) qwen3, key_length 32: config.head_dim =", m.config.head_dim)
try:
    forward(m)
    print("    forward ran")
except Exception as e:
    print("    forward:", type(e).__name__, str(e)[:120])

# (b) a Mistral as llama.cpp writes it: architecture "llama", general.name "Mistral-7B-v0.1"
llama_meta = common + [("add_rope_dimension_count", 16)]
m = load(write("llama", llama_meta, attn_llama(16), name="Mistral-7B-v0.1"))
print("(b) llama, general.name 'Mistral-7B-v0.1': loads as", type(m).__name__)
try:
    m = load(write("llama", llama_meta, attn_llama(16), name="mistral"))
    print("    general.name 'mistral': loads as", type(m).__name__)
except Exception as e:
    print("    general.name 'mistral':", type(e).__name__, str(e)[:120])

# (c) GPT-2 with context_length 2048
C = 2048
t = [("token_embd.weight", w(V, E)), ("position_embd.weight", w(C, E))]
for n, s in [("attn_norm.weight", one(E)), ("attn_norm.bias", 0 * one(E)), ("attn_qkv.weight", w(3 * E, E)),
             ("attn_qkv.bias", 0 * one(3 * E)), ("attn_output.weight", w(E, E)), ("attn_output.bias", 0 * one(E)),
             ("ffn_norm.weight", one(E)), ("ffn_norm.bias", 0 * one(E)), ("ffn_up.weight", w(4 * E, E)),
             ("ffn_up.bias", 0 * one(4 * E)), ("ffn_down.weight", w(E, 4 * E)), ("ffn_down.bias", 0 * one(E))]:
    t.append(("blk.0." + n, s))
t += [("output_norm.weight", one(E)), ("output_norm.bias", 0 * one(E)), ("output.weight", w(V, E))]
meta = [("add_context_length", C), ("add_embedding_length", E), ("add_block_count", 1),
        ("add_feed_forward_length", 4 * E), ("add_head_count", H), ("add_layer_norm_eps", 1e-5)]
try:
    m = load(write("gpt2", meta, t))
    print("(c) gpt2, context_length 2048: n_ctx", m.config.n_ctx, "n_positions", m.config.n_positions,
          "wpe", tuple(m.transformer.wpe.weight.shape))
except Exception as e:
    print("(c) gpt2, context_length 2048:", type(e).__name__, str(e)[:200])
