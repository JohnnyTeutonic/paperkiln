# A small random Gemma 3 GGUF (6 layers: 5 sliding-window, 1 global), f32, no tokenizer.
# usage: python make_gemma3.py OUT.gguf [SLIDING_WINDOW] [ROPE_LINEAR_FACTOR]
# needs gguf-py (pip install gguf, or PYTHONPATH=llama.cpp/gguf-py)
import sys

import numpy as np
import gguf

out = sys.argv[1]
window = int(sys.argv[2]) if len(sys.argv) > 2 else 512
factor = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
n_embd, n_head, n_kv, head_dim, n_ff, n_layer, n_vocab = 64, 4, 2, 16, 128, 6, 256

rng = np.random.default_rng(0)
def w(*shape, scale=0.2):
    return (rng.standard_normal(shape) * scale).astype(np.float32)
def norm(n):
    return (1.0 + 0.1 * rng.standard_normal(n)).astype(np.float32)

g = gguf.GGUFWriter(out, "gemma3")
g.add_context_length(4096)
g.add_embedding_length(n_embd)
g.add_block_count(n_layer)
g.add_feed_forward_length(n_ff)
g.add_head_count(n_head)
g.add_head_count_kv(n_kv)
g.add_key_length(head_dim)
g.add_value_length(head_dim)
g.add_layer_norm_rms_eps(1e-6)
g.add_rope_freq_base(1e6)
g.add_sliding_window(window)
if factor:
    g.add_rope_scaling_type(gguf.RopeScalingType.LINEAR)
    g.add_rope_scaling_factor(factor)
g.add_vocab_size(n_vocab)
g.add_tokenizer_model("no_vocab")

g.add_tensor("token_embd.weight", w(n_vocab, n_embd, scale=1.0))
for i in range(n_layer):
    p = f"blk.{i}."
    g.add_tensor(p + "attn_norm.weight", norm(n_embd))
    g.add_tensor(p + "attn_q.weight", w(n_head * head_dim, n_embd))
    g.add_tensor(p + "attn_k.weight", w(n_kv * head_dim, n_embd))
    g.add_tensor(p + "attn_v.weight", w(n_kv * head_dim, n_embd))
    g.add_tensor(p + "attn_q_norm.weight", norm(head_dim))
    g.add_tensor(p + "attn_k_norm.weight", norm(head_dim))
    g.add_tensor(p + "attn_output.weight", w(n_embd, n_head * head_dim))
    g.add_tensor(p + "post_attention_norm.weight", norm(n_embd))
    g.add_tensor(p + "ffn_norm.weight", norm(n_embd))
    g.add_tensor(p + "ffn_gate.weight", w(n_ff, n_embd))
    g.add_tensor(p + "ffn_up.weight", w(n_ff, n_embd))
    g.add_tensor(p + "ffn_down.weight", w(n_embd, n_ff))
    g.add_tensor(p + "post_ffw_norm.weight", norm(n_embd))
g.add_tensor("output_norm.weight", norm(n_embd))

g.write_header_to_file()
g.write_kv_data_to_file()
g.write_tensors_to_file()
g.close()
print("wrote", out)
