#!/usr/bin/env python3
"""Synthetic third-party gate (LOCAL_LLM_STACK.md section 5.3), for graph features no small official
checkpoint exercises or that we cannot download: a tiny randomly initialised HF transformers model of
the family, converted by llama.cpp's own convert_hf_to_gguf.py (f32; a placeholder vocabulary, since
the graph is under test, not tokenisation), then spec/tools/gguf_ir.py's IR on spec/tools/ir_numpy.py against
HF's PyTorch forward of the same random weights (eager attention, f32) on random token sequences
longer than the attention window.

    python3 tools/thirdparty_synth.py --family F --llamacpp ~/llamacpp_b6000 \
        --work DIR --out RECEIPT.jsonl [--n 8] [--len 48] [--max-new 16] [--seed 0] \
        [--spec-tools DIR] [--export DIR]

Families: mistral_swa, gemma3 (22 Oct 2026); phi3_longrope, phi3_swa, gemma2, mixtral, qwen2moe, qwen3moe,
deepseek2, deepseek2_qlora (E4, 27 Oct 2026; keramos revision 1.5).

Per sequence: max |dlogit| over the range at every position (rel), argmax agreement at every
position, and greedy continuations (IR recomputed from position 0 each step vs the argmax of HF's
cached forward, thirdparty_parity.hf_greedy; for phi3_longrope HF's forward recomputed from position 0
each step as well, as HF's generate does when a sequence crosses original_max_position_embeddings, so
the factor set is the one the whole sequence's length selects).
Also, per family, a check that the feature under test matters: the same IR with it removed has to
disagree with HF (windows off; soft-capping off; LongRoPE's short and long factors swapped); and for
gemma3 the same check with the norm weights stored as HF's raw w and weight_offset 1 on every rmsnorm.
Writes RECEIPT.jsonl (one row per sequence), RECEIPT.summary.json and RECEIPT.ir.json.

--spec-tools  the keramos tools to import (default spec/tools, the pinned tag); a keramos working copy's
              tools/ while a revision is drafted.
--export DIR  also writes DIR/<family>.gguf, DIR/<family>_hf/ (the HF checkpoint) and
              DIR/<family>.cases.json (the sequences, HF's greedy continuations and HF's logits at each
              sequence's last position, f32 hex) for another engine's gate (ember).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _spec_tools(argv):
    """The keramos tools directory from --spec-tools (read before the imports that need it)."""
    for i, a in enumerate(argv):
        if a == "--spec-tools" and i + 1 < len(argv):
            return os.path.abspath(os.path.expanduser(argv[i + 1]))
        if a.startswith("--spec-tools="):
            return os.path.abspath(os.path.expanduser(a.split("=", 1)[1]))
    return os.path.join(os.path.dirname(HERE), "spec", "tools")


sys.path.insert(0, _spec_tools(sys.argv[1:]))  # keramos
import gguf_ir  # noqa: E402
import ir_numpy  # noqa: E402
import thirdparty_parity  # noqa: E402

VOCAB = 512
FAMILIES = ("mistral_swa", "gemma3", "phi3_longrope", "phi3_swa", "gemma2", "mixtral", "qwen2moe", "qwen3moe",
            "deepseek2", "deepseek2_qlora", "deepseek2_dense_yarn", "deepseek3")
# deepseek3: DeepSeek-V3's router groups (n_group 2, topk_group 1), which llama.cpp b6000's converter does not write;
# the gate writes them under the later llama.cpp key names, as it writes Mistral's window.
DS3_GROUPS = (2, 1)
# phi3_longrope: original_max_position_embeddings 32, so sequences of 12 (short factors throughout), 24
# (crossing to the long ones during the greedy continuation) and 48 tokens (long throughout).
LONGROPE_ORIG = 32
LONGROPE_LENS = (12, 24, 48)


def hf_model(family, seed):
    import torch
    torch.manual_seed(seed)
    if family == "mistral_swa":
        from transformers import MistralConfig, MistralForCausalLM
        cfg = MistralConfig(vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                            num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=256,
                            sliding_window=8, rms_norm_eps=1e-5, rope_theta=10000.0, tie_word_embeddings=False,
                            initializer_range=0.2, attn_implementation="eager")
        model = MistralForCausalLM(cfg)
    elif family == "gemma3":
        from transformers import Gemma3ForCausalLM, Gemma3TextConfig
        cfg = Gemma3TextConfig(vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=8,
                               num_attention_heads=2, num_key_value_heads=1, head_dim=32, query_pre_attn_scalar=32,
                               max_position_embeddings=256, sliding_window=8, rms_norm_eps=1e-6, rope_theta=1e6,
                               rope_local_base_freq=10000.0, rope_scaling=None, final_logit_softcapping=None,
                               attn_logit_softcapping=None, tie_word_embeddings=True, initializer_range=0.2,
                               attn_implementation="eager")
        model = Gemma3ForCausalLM(cfg)
    elif family in ("phi3_longrope", "phi3_swa"):
        from transformers import Phi3Config, Phi3ForCausalLM
        g = torch.Generator().manual_seed(seed + 31)
        long_ = family == "phi3_longrope"
        # head_dim 16, partial rotary 0.5 (Phi-4-mini rotates 0.75): 8 rotated dims, 4 factors per set.
        scaling = None
        if long_:
            scaling = {"type": "longrope",
                       "short_factor": (1.0 + 0.5 * torch.rand(4, generator=g)).tolist(),
                       "long_factor": (2.0 + 18.0 * torch.rand(4, generator=g)).tolist()}
        cfg = Phi3Config(vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                         original_max_position_embeddings=LONGROPE_ORIG if long_ else 128, rms_norm_eps=1e-5,
                         rope_theta=10000.0, rope_scaling=scaling, partial_rotary_factor=0.5 if long_ else 1.0,
                         sliding_window=None if long_ else 8, tie_word_embeddings=not long_, initializer_range=0.2,
                         pad_token_id=0, bos_token_id=1, eos_token_id=2, attn_implementation="eager")
        model = Phi3ForCausalLM(cfg)
    elif family == "gemma2":
        from transformers import Gemma2Config, Gemma2ForCausalLM
        # Soft caps small enough to bind on random weights (Gemma 2 ships 50 and 30).
        cfg = Gemma2Config(vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                           num_attention_heads=2, num_key_value_heads=1, head_dim=32, query_pre_attn_scalar=32,
                           max_position_embeddings=256, sliding_window=8, rms_norm_eps=1e-6, rope_theta=10000.0,
                           attn_logit_softcapping=4.0, final_logit_softcapping=2.0, tie_word_embeddings=True,
                           initializer_range=0.2, attn_implementation="eager")
        model = Gemma2ForCausalLM(cfg)
    elif family == "mixtral":
        from transformers import MixtralConfig, MixtralForCausalLM
        cfg = MixtralConfig(vocab_size=VOCAB, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
                            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                            rms_norm_eps=1e-5, rope_theta=1e6, sliding_window=None, num_local_experts=4,
                            num_experts_per_tok=2, tie_word_embeddings=False, initializer_range=0.2,
                            attn_implementation="eager")
        model = MixtralForCausalLM(cfg)
    elif family == "qwen2moe":
        from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM
        cfg = Qwen2MoeConfig(vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                             num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                             rms_norm_eps=1e-6, rope_theta=1e6, decoder_sparse_step=1, moe_intermediate_size=32,
                             shared_expert_intermediate_size=96, num_experts_per_tok=2, num_experts=6,
                             norm_topk_prob=False, mlp_only_layers=[], qkv_bias=True, tie_word_embeddings=False,
                             initializer_range=0.2, attn_implementation="eager")
        model = Qwen2MoeForCausalLM(cfg)
    elif family == "qwen3moe":
        from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
        cfg = Qwen3MoeConfig(vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                             num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                             rms_norm_eps=1e-6, rope_theta=1e6, moe_intermediate_size=32, num_experts=6,
                             num_experts_per_tok=2, norm_topk_prob=True, decoder_sparse_step=1, mlp_only_layers=[],
                             tie_word_embeddings=False, initializer_range=0.2, attn_implementation="eager")
        model = Qwen3MoeForCausalLM(cfg)
    elif family in ("deepseek2", "deepseek2_qlora"):
        from transformers import DeepseekV2Config, DeepseekV2ForCausalLM
        # MLA: latent rank 20, decoupled RoPE 8, nope 16, v 12 (unequal to every other width); every layer MoE
        # (first_k_dense_replace 0), softmax gates without renormalisation, scaled by 2.5, one shared expert.
        cfg = DeepseekV2Config(vocab_size=VOCAB, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
                               num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4,
                               n_shared_experts=1, n_routed_experts=6, routed_scaling_factor=2.5, kv_lora_rank=20,
                               q_lora_rank=24 if family == "deepseek2_qlora" else None, qk_rope_head_dim=8,
                               v_head_dim=12, qk_nope_head_dim=16, topk_method="greedy", n_group=1, topk_group=1,
                               num_experts_per_tok=2, first_k_dense_replace=0, norm_topk_prob=False,
                               max_position_embeddings=256, rms_norm_eps=1e-6, rope_theta=10000.0, rope_scaling=None,
                               attention_bias=False, tie_word_embeddings=False, initializer_range=0.2,
                               pad_token_id=0, bos_token_id=1, eos_token_id=2, attn_implementation="eager",
                               scoring_func="softmax")  # read by llama.cpp's convert (DeepSeek's own configs hold it)
        model = DeepseekV2ForCausalLM(cfg)
    elif family == "deepseek2_dense_yarn":
        from transformers import DeepseekV2Config, DeepseekV2ForCausalLM
        # keramos 1.6: the first layer dense (first_k_dense_replace 1, the apply attribute), YaRN factor 4 over an
        # original context of 16 (sequences of 48). mscale_all_dim 0, so llama.cpp's softmax temperature is 1, and
        # attention_factor 1, so transformers' DeepseekV2 scales cos and sin by 1 as llama.cpp does.
        cfg = DeepseekV2Config(vocab_size=VOCAB, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
                               num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4,
                               n_shared_experts=2, n_routed_experts=6, routed_scaling_factor=1.0, kv_lora_rank=20,
                               q_lora_rank=None, qk_rope_head_dim=8, v_head_dim=12, qk_nope_head_dim=16,
                               topk_method="greedy", n_group=1, topk_group=1, num_experts_per_tok=2,
                               first_k_dense_replace=1, norm_topk_prob=False, max_position_embeddings=256,
                               rms_norm_eps=1e-6, rope_theta=10000.0,
                               rope_scaling={"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 16,
                                             "beta_fast": 32, "beta_slow": 1, "mscale": 1.0, "mscale_all_dim": 0.0,
                                             "attention_factor": 1.0},
                               attention_bias=False, tie_word_embeddings=False, initializer_range=0.2,
                               pad_token_id=0, bos_token_id=1, eos_token_id=2, attn_implementation="eager",
                               scoring_func="softmax")
        model = DeepseekV2ForCausalLM(cfg)
    elif family == "deepseek3":
        from transformers import DeepseekV3Config, DeepseekV3ForCausalLM
        # keramos 1.6: DeepSeek-V3's router (sigmoid scores, a selection bias, 2 groups of which 1 is kept,
        # renormalised gates scaled by 2.5), a leading dense layer, the low-rank query, YaRN factor 4 over 16 with
        # mscale = mscale_all_dim = 1: cos and sin unscaled and the softmax temperature (0.1 ln 4 + 1)^2.
        cfg = DeepseekV3Config(vocab_size=VOCAB, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
                               num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4,
                               n_shared_experts=1, n_routed_experts=8, routed_scaling_factor=2.5, kv_lora_rank=20,
                               q_lora_rank=24, qk_rope_head_dim=8, v_head_dim=12, qk_nope_head_dim=16,
                               n_group=DS3_GROUPS[0], topk_group=DS3_GROUPS[1], num_experts_per_tok=2,
                               first_k_dense_replace=1, norm_topk_prob=True, max_position_embeddings=256,
                               rms_norm_eps=1e-6, rope_theta=10000.0,
                               rope_scaling={"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 16,
                                             "beta_fast": 32, "beta_slow": 1, "mscale": 1.0, "mscale_all_dim": 1.0},
                               rope_interleave=True, attention_bias=False, tie_word_embeddings=False,
                               initializer_range=0.2, pad_token_id=0, bos_token_id=1, eos_token_id=2,
                               attn_implementation="eager", scoring_func="sigmoid", topk_method="noaux_tc")
        model = DeepseekV3ForCausalLM(cfg)
        with torch.no_grad():
            for name_, buf in model.named_buffers():
                if name_.endswith("e_score_correction_bias"):
                    # Positive biases, so every selection score is positive: transformers fills dropped groups'
                    # experts with 0 where DeepSeek's own code (and keramos 5.16) excludes them, which picks a
                    # dropped group's expert whenever a kept group has fewer than top_k positive scores (with
                    # biases 0.3 N(0, 1): 7 of 40 sequences, every differing position such a case).
                    buf.copy_(0.5 + 0.1 * torch.randn_like(buf))
    else:
        raise SystemExit(f"thirdparty_synth: unknown family {family}")
    with torch.no_grad():
        for n, p in model.named_parameters():
            if not torch.isfinite(p).all():  # a parameter the model's init leaves empty (none expected)
                raise SystemExit(f"thirdparty_synth: {n} is not initialised")
            if "norm" in n:  # non-trivial norm weights (gemma's init is 0, llama's 1)
                p.add_(0.2 * torch.randn_like(p))
    return model.float().eval()


def convert(model, family, llamacpp, work):
    """save_pretrained, then llama.cpp's convert_hf_to_gguf.py --outtype f32, its set_vocab replaced
    by a placeholder vocabulary of VOCAB tokens (our tiny models have no tokenizer)."""
    hf_dir = os.path.join(work, "hf")
    model.save_pretrained(hf_dir, safe_serialization=True)
    out = os.path.join(work, f"{family}.gguf")
    sys.path.insert(0, llamacpp)
    sys.path.insert(0, os.path.join(llamacpp, "gguf-py"))
    import convert_hf_to_gguf as conv
    import gguf

    def placeholder_vocab(self):
        self.gguf_writer.add_tokenizer_model("llama")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list([f"<t{i}>" for i in range(VOCAB)])
        self.gguf_writer.add_token_scores([0.0] * VOCAB)
        self.gguf_writer.add_token_types([gguf.TokenType.NORMAL] * VOCAB)

    cls = conv.ModelBase.from_model_architecture(model.config.architectures[0])
    cls.set_vocab = placeholder_vocab
    # Gemma's modify_tensors trims token_embd to the SentencePiece vocabulary: the placeholder's size.
    cls._create_vocab_sentencepiece = lambda self: ([f"<t{i}>".encode() for i in range(VOCAB)], [0.0] * VOCAB,
                                                    [gguf.TokenType.NORMAL] * VOCAB)
    if family == "deepseek3":
        base3 = cls.set_gguf_parameters

        def with_groups(self):
            base3(self)
            self.gguf_writer.add_uint32("deepseek2.expert_group_count", DS3_GROUPS[0])
            self.gguf_writer.add_uint32("deepseek2.expert_group_used_count", DS3_GROUPS[1])
        cls.set_gguf_parameters = with_groups
    if family == "mistral_swa":
        # llama.cpp's convert drops Mistral's sliding_window; write it as <arch>.attention.sliding_window.
        base = cls.set_gguf_parameters

        def with_window(self):
            base(self)
            self.gguf_writer.add_sliding_window(self.hparams["sliding_window"])
        cls.set_gguf_parameters = with_window
    old = sys.argv
    sys.argv = ["convert_hf_to_gguf.py", hf_dir, "--outtype", "f32", "--outfile", out]
    try:
        conv.main()
    finally:
        sys.argv = old
    return out, hf_dir


def hf_greedy_full(model, ids, max_new):
    """Greedy argmax of HF's forward recomputed from position 0 at every step (no cache)."""
    import torch
    seq, new = list(ids), []
    for _ in range(max_new):
        t = int(torch.argmax(model(torch.tensor([seq])).logits[0, -1]))
        new.append(t)
        seq.append(t)
    return new


def compare(doc, W, model, seqs, max_new, full_recompute=False):
    import torch
    rows = []
    for k, ids in enumerate(seqs):
        ref = model(torch.tensor([ids])).logits[0].numpy()
        mine = ir_numpy.forward(doc, W, ids)
        rng = (ref.max(axis=1) - ref.min(axis=1))
        rel = float((np.abs(ref - mine).max(axis=1) / rng).max())
        top1 = int((ref.argmax(axis=1) == mine.argmax(axis=1)).sum())
        if full_recompute:
            hf_new = hf_greedy_full(model, ids, max_new)
        else:
            _, hf_new = thirdparty_parity.hf_greedy(model, ids, max_new, set())  # plain argmax, no generate
        seq, ir_new = list(ids), []
        for _ in range(max_new):
            t = int(ir_numpy.forward(doc, W, seq, last_only=True)[-1].argmax())
            ir_new.append(t)
            seq.append(t)
        rows.append({"i": k, "len": len(ids), "max_rel_all_positions": rel, "top1_same": top1, "positions": len(ids),
                     "greedy_same": hf_new == ir_new, "hf_new": hf_new,
                     "first_diff": next((j for j, (a, b) in enumerate(zip(hf_new, ir_new)) if a != b), None)})
        print(json.dumps({k2: v for k2, v in rows[-1].items() if k2 != "hf_new"}), flush=True)
    return rows


def rel_against(model, doc, W, ids, **kw):
    import torch
    ref = model(torch.tensor([ids])).logits[0].numpy()
    got = ir_numpy.forward(doc, W, ids, **kw)
    return float((np.abs(ref - got).max(axis=1) / (ref.max(axis=1) - ref.min(axis=1))).max())


def ablations(family, doc, W, model, seqs):
    """{name: rel}: the IR with the feature under test removed, against HF (it has to disagree)."""
    out = {}
    nodes = doc["block"]["nodes"]
    if any(n["op"] == "attention" and n.get("window") not in (0, None) for n in nodes):
        out["window_off_rel"] = rel_against(model, doc, W, seqs[0], no_window=True)
    if family == "gemma2":
        d2 = json.loads(json.dumps(doc))
        for n in d2["block"]["nodes"]:
            n.pop("softcap", None)
        out["attn_softcap_off_rel"] = rel_against(model, d2, W, seqs[0])
        d3 = json.loads(json.dumps(doc))
        d3["head"]["nodes"] = [n for n in d3["head"]["nodes"] if n["op"] != "expr"]
        d3["head"]["nodes"][-1]["out"] = "logits"
        out["final_softcap_off_rel"] = rel_against(model, d3, W, seqs[0])
    if family == "phi3_longrope":
        d2 = json.loads(json.dumps(doc))
        for n in d2["block"]["nodes"]:
            if n["op"] == "rope":
                ff = n["freq_factors"]
                n["freq_factors"] = dict(ff, short=ff["long"], long=ff["short"])
        short = next(s for s in seqs if len(s) <= LONGROPE_ORIG)
        long_ = next(s for s in seqs if len(s) > LONGROPE_ORIG)
        out["factors_swapped_rel_short"] = rel_against(model, d2, W, short)
        out["factors_swapped_rel_long"] = rel_against(model, d2, W, long_)
        d3 = json.loads(json.dumps(doc))
        for n in d3["block"]["nodes"]:
            if n["op"] == "rope":
                n.pop("attn_factor", None)
        out["attn_factor_off_rel"] = rel_against(model, d3, W, short)
    if any(n["op"] == "moe_ffn" for n in nodes):
        d2 = json.loads(json.dumps(doc))
        for n in d2["block"]["nodes"]:
            if n["op"] == "moe_ffn":
                n["gate_norm"] = "raw" if n["gate_norm"] == "renorm" else "renorm"
        out["gate_norm_swapped_rel"] = rel_against(model, d2, W, seqs[0])
    if any(n["op"] == "mul_col" for n in nodes):  # qwen2moe's shared-expert gate
        d2 = json.loads(json.dumps(doc))
        d2["block"]["nodes"] = [n for n in d2["block"]["nodes"] if n["op"] not in ("sigmoid", "mul_col")]
        out["shared_gate_off_rel"] = rel_against(model, d2, W, seqs[0])
    if any(n.get("apply") for n in nodes):  # keramos 1.6: the leading dense layer's FFN zeroed
        d2 = json.loads(json.dumps(doc))
        dense = [n["apply"] for n in d2["block"]["nodes"] if n.get("apply")][0]["per_layer"]  # the dense branch first
        d2["block"]["nodes"].insert(len(d2["block"]["nodes"]) - 1, {"op": "affine", "in": ["f"], "scale": 0.0,
                                                                     "shift": 0.0, "out": "f",
                                                                     "apply": {"per_layer": list(dense)}})
        out["dense_layer_zeroed_rel"] = rel_against(model, d2, W, seqs[0])
    rp = [n for n in nodes if n["op"] == "rope" and isinstance(n.get("freq_factors"), list)]
    if rp:  # YaRN's divisors and DeepSeek's softmax temperature
        d2 = json.loads(json.dumps(doc))
        for n in d2["block"]["nodes"]:
            if n["op"] == "rope":
                n.pop("freq_factors")
        out["yarn_factors_off_rel"] = rel_against(model, d2, W, seqs[0])
        mla = [n for n in nodes if n["op"] == "mla_attention"][0]
        plain = float(np.float32(1.0 / np.sqrt(mla["qk_nope"] + mla["qk_rope"])))
        if abs(mla["scale"] - plain) > 1e-7:
            d3 = json.loads(json.dumps(doc))
            for n in d3["block"]["nodes"]:
                if n["op"] == "mla_attention":
                    n["scale"] = plain
            out["softmax_temperature_off_rel"] = rel_against(model, d3, W, seqs[0])
    mo = [n for n in nodes if n["op"] == "moe_ffn"]
    if mo and (mo[0].get("scoring") or mo[0].get("selection_bias") or mo[0].get("groups")):
        for key, change in (("selection_bias_off_rel", lambda n: n.pop("selection_bias", None)),
                            ("groups_off_rel", lambda n: n.pop("groups", None)),
                            ("softmax_scoring_rel", lambda n: n.update(scoring="softmax"))):
            d2 = json.loads(json.dumps(doc))
            for n in d2["block"]["nodes"]:
                if n["op"] == "moe_ffn":
                    change(n)
            out[key] = rel_against(model, d2, W, seqs[0])
    if any(n["op"] == "mla_attention" for n in nodes):
        d2 = json.loads(json.dumps(doc))
        for n in d2["block"]["nodes"]:
            if n["op"] == "rope":
                n["pairing"] = "neox"
        out["rope_neox_rel"] = rel_against(model, d2, W, seqs[0])
        d3 = json.loads(json.dumps(doc))
        d3["block"]["nodes"] = [n for n in d3["block"]["nodes"] if n["op"] != "affine"]
        out["routed_scale_off_rel"] = rel_against(model, d3, W, seqs[0])
    return out


def export_cases(dirpath, family, gguf_path, hf_dir, seqs, rows, model):
    import torch
    from ir_conformance import enc
    os.makedirs(dirpath, exist_ok=True)
    shutil.copy2(gguf_path, os.path.join(dirpath, f"{family}.gguf"))
    dst = os.path.join(dirpath, f"{family}_hf")
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(hf_dir, dst)
    cases = []
    for ids, r in zip(seqs, rows):
        last = model(torch.tensor([ids])).logits[0, -1].numpy().astype(np.float32)
        cases.append({"ids": ids, "hf_greedy": r["hf_new"], "hf_last_logits": enc(last[None, :])})
    json.dump({"family": family, "vocab": VOCAB, "max_new": len(rows[0]["hf_new"]),
               "_comment": "thirdparty_synth.py --export: the gate's sequences, HF's greedy continuations (argmax "
                           "of HF's forward) and HF's f32 logits at each sequence's last position (hex, keramos "
                           "IR_SPEC 11.2)", "cases": cases},
              open(os.path.join(dirpath, f"{family}.cases.json"), "w"), indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--family", required=True, choices=FAMILIES)
    ap.add_argument("--llamacpp", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--len", type=int, default=48)
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spec-tools", default=None)
    ap.add_argument("--export", default=None)
    a = ap.parse_args(argv)
    import torch
    torch.set_grad_enabled(False)
    os.makedirs(a.work, exist_ok=True)
    model = hf_model(a.family, a.seed)
    path, hf_dir = convert(model, a.family, os.path.expanduser(a.llamacpp), a.work)
    meta, shapes = gguf_ir.read_meta(path)
    doc = gguf_ir.build(meta, shapes)
    json.dump(doc, open(a.out + ".ir.json", "w", encoding="utf-8"), indent=1)
    W = ir_numpy.load_weights(path)
    rng = np.random.default_rng(a.seed)
    lens = [LONGROPE_LENS[k % len(LONGROPE_LENS)] for k in range(a.n)] if a.family == "phi3_longrope" else [a.len] * a.n
    seqs = [rng.integers(0, VOCAB, size=n).tolist() for n in lens]
    rows = compare(doc, W, model, seqs, a.max_new, full_recompute=a.family == "phi3_longrope")
    for r in rows:
        r["variant"] = "gguf"
    summ = {"family": a.family, "gguf_arch": meta["general.architecture"], "sequences": len(rows),
            "len": sorted(set(lens)), "worst_rel_all_positions": max(r["max_rel_all_positions"] for r in rows),
            "top1_same": sum(r["top1_same"] for r in rows), "positions": sum(r["positions"] for r in rows),
            "greedy_same": sum(r["greedy_same"] for r in rows), "max_new": a.max_new, "seed": a.seed,
            "keramos_tools": os.path.relpath(os.path.dirname(gguf_ir.__file__)) if a.spec_tools else "spec/tools"}
    att = [n for n in doc["block"]["nodes"] if n["op"] in ("attention", "mla_attention")]
    summ["window"] = att[0].get("window") if att else None
    summ.update(ablations(a.family, doc, W, model, seqs))
    if a.family == "gemma3":
        W2, doc2 = dict(W), json.loads(json.dumps(doc))
        for part in ("block", "head"):
            for nd in doc2[part]["nodes"]:
                if nd["op"] == "rmsnorm":
                    nd["weight_offset"] = 1.0
        for k in W2:
            if k.endswith("norm.weight"):
                W2[k] = (W[k] - np.float32(1)).astype(np.float32)
        rows2 = compare(doc2, W2, model, seqs[:2], a.max_new)
        for r in rows2:
            r["variant"] = "raw_w_weight_offset_1"
        rows += rows2
        summ["weight_offset_rel"] = max(r["max_rel_all_positions"] for r in rows2)
        summ["weight_offset_greedy_same"] = sum(r["greedy_same"] for r in rows2)
    if a.export:
        export_cases(os.path.expanduser(a.export), a.family, path, hf_dir, seqs, [r for r in rows if r["variant"] == "gguf"],
                     model)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if k != "hf_new"}) + "\n")
    json.dump(summ, open(a.out + ".summary.json", "w"), indent=1)
    print(json.dumps(summ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
