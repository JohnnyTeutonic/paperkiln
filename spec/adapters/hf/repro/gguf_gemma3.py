# A small Gemma 3 GGUF with the metadata llama.cpp's convert_hf_to_gguf.py writes for Gemma 3 4B/12B/27B
# (gemma3.rope.scaling.type linear, gemma3.rope.scaling.factor 8), loaded with from_pretrained(gguf_file=...).
# Needs: pip install gguf
import copy
import os
import tempfile

import gguf
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, Gemma3ForCausalLM

E, H, KV, D, FF, L, V = 64, 4, 2, 16, 128, 6, 256          # head_dim 16 (key_length), 5 sliding + 1 global layer
rng = np.random.default_rng(0)
w = lambda *s: (rng.standard_normal(s) * 0.2).astype(np.float32)
nrm = lambda n: (1 + 0.1 * rng.standard_normal(n)).astype(np.float32)   # llama.cpp stores Gemma norms as 1 + w

d = tempfile.mkdtemp()
g = gguf.GGUFWriter(os.path.join(d, "g.gguf"), "gemma3")
g.add_context_length(4096); g.add_embedding_length(E); g.add_block_count(L); g.add_feed_forward_length(FF)
g.add_head_count(H); g.add_head_count_kv(KV); g.add_key_length(D); g.add_value_length(D)
g.add_layer_norm_rms_eps(1e-6); g.add_rope_freq_base(1e6); g.add_sliding_window(512)
g.add_rope_scaling_type(gguf.RopeScalingType.LINEAR); g.add_rope_scaling_factor(8.0)
g.add_vocab_size(V); g.add_tokenizer_model("no_vocab")
g.add_tensor("token_embd.weight", w(V, E))
for i in range(L):
    p = f"blk.{i}."
    for n, t in [("attn_norm", nrm(E)), ("attn_q", w(H * D, E)), ("attn_k", w(KV * D, E)), ("attn_v", w(KV * D, E)),
                 ("attn_q_norm", nrm(D)), ("attn_k_norm", nrm(D)), ("attn_output", w(E, H * D)),
                 ("post_attention_norm", nrm(E)), ("ffn_norm", nrm(E)), ("ffn_gate", w(FF, E)), ("ffn_up", w(FF, E)),
                 ("ffn_down", w(E, FF)), ("post_ffw_norm", nrm(E))]:
        g.add_tensor(p + n + ".weight", t)
g.add_tensor("output_norm.weight", nrm(E))
g.write_header_to_file(); g.write_kv_data_to_file(); g.write_tensors_to_file(); g.close()

torch.set_num_threads(2)
m = AutoModelForCausalLM.from_pretrained(d, gguf_file="g.gguf", dtype=torch.float32).eval()
from importlib.metadata import version
print("transformers", transformers.__version__, "gguf", version("gguf"), "torch", torch.__version__)
print("rope_parameters:", m.config.rope_parameters)
print("head_dim:", m.config.head_dim, " query_pre_attn_scalar:", m.config.query_pre_attn_scalar)

# The same weights, with the configuration the file describes: linear RoPE factor 8 on the global layers,
# attention scale 1/sqrt(key_length) (what llama.cpp computes for this file); and each fix alone.
ids = torch.tensor([list(range(1, 65))])
with torch.no_grad():
    a = m(ids).logits
for label, rope, scale in [("both", True, True), ("RoPE scaling only", True, False), ("attention scale only", False, True)]:
    cfg = copy.deepcopy(m.config)
    if rope:
        cfg.rope_parameters["full_attention"] = {"rope_type": "linear", "factor": 8.0,
                                                 "rope_theta": m.config.rope_parameters["full_attention"]["rope_theta"]}
    if scale:
        cfg.query_pre_attn_scalar = D
    ref = Gemma3ForCausalLM(cfg).eval()
    ref.load_state_dict(m.state_dict())
    with torch.no_grad():
        b = ref(ids).logits
    print(f"GGUF model vs the same weights with {label} corrected: max |diff| / max |logit| "
          f"{((a - b).abs().max() / b.abs().max()).item():.3g}, argmax differs at "
          f"{int((a.argmax(-1) != b.argmax(-1)).sum())} of {ids.shape[1]} positions")
