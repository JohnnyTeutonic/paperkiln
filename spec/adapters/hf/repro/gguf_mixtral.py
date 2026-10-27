# A small Mixtral-style GGUF as llama.cpp's convert_hf_to_gguf.py writes MixtralForCausalLM: general.architecture
# "llama", llama.expert_count / expert_used_count, merged blk.N.ffn_{gate,up,down}_exps tensors.
# Needs: pip install gguf
import os
import tempfile
from importlib.metadata import version

import gguf
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM

E, H, KV, D, FF, L, V, X, K = 64, 4, 2, 16, 128, 2, 256, 4, 2   # 4 experts, 2 per token
rng = np.random.default_rng(0)
w = lambda *s: (rng.standard_normal(s) * 0.2).astype(np.float32)
one = lambda n: np.ones(n, np.float32)

d = tempfile.mkdtemp()
g = gguf.GGUFWriter(os.path.join(d, "moe.gguf"), "llama")
g.add_name("Mixtral-tiny")
g.add_context_length(4096); g.add_embedding_length(E); g.add_block_count(L); g.add_feed_forward_length(FF)
g.add_head_count(H); g.add_head_count_kv(KV); g.add_rope_dimension_count(D); g.add_layer_norm_rms_eps(1e-5)
g.add_rope_freq_base(1e6); g.add_expert_count(X); g.add_expert_used_count(K)
g.add_vocab_size(V); g.add_tokenizer_model("no_vocab")
g.add_tensor("token_embd.weight", w(V, E))
for i in range(L):
    p = f"blk.{i}."
    for n, t in [("attn_norm", one(E)), ("attn_q", w(H * D, E)), ("attn_k", w(KV * D, E)), ("attn_v", w(KV * D, E)),
                 ("attn_output", w(E, H * D)), ("ffn_norm", one(E)), ("ffn_gate_inp", w(X, E)),
                 ("ffn_gate_exps", w(X, FF, E)), ("ffn_up_exps", w(X, FF, E)), ("ffn_down_exps", w(X, E, FF))]:
        g.add_tensor(p + n + ".weight", t)
g.add_tensor("output_norm.weight", one(E))
g.add_tensor("output.weight", w(V, E))
g.write_header_to_file(); g.write_kv_data_to_file(); g.write_tensors_to_file(); g.close()

print("transformers", transformers.__version__, "gguf", version("gguf"), "torch", torch.__version__)
ids = torch.tensor([list(range(1, 17))])
runs = []
for _ in range(2):
    m, info = AutoModelForCausalLM.from_pretrained(d, gguf_file="moe.gguf", dtype=torch.float32,
                                                   output_loading_info=True)
    with torch.no_grad():
        runs.append(m.eval()(ids).logits)
print("class:", type(m).__name__, " num_local_experts:", getattr(m.config, "num_local_experts", None))
print("missing_keys:", sorted(info["missing_keys"]))
print("two loads of the same file give the same logits:", torch.equal(runs[0], runs[1]))
