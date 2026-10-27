#!/usr/bin/env python3
"""Run ONNX Runtime GenAI's model builder (onnxruntime_genai.models.builder) as its command line does, with
the named diagnostic patches applied in process. The adapter runs this in a subprocess per model, so the
builder's torch and transformers state and its printing stay out of the suite's process.

    python adapters/ortgenai/build_onnx.py --input HF_DIR_OR_GGUF [--config HF_DIR] --output DIR --cache DIR
                                           [--patch none|unfused_rope,no_gqa,gemma3_config]

  none            the builder as shipped: -p fp32 -e cpu, no extra options (the receipt).
  unfused_rope    Model.is_fused_rope_supported() -> False: GroupQueryAttention without do_rotary, RoPE by a
                  separate RotaryEmbedding node. ORT's CPU GQA refuses do_rotary when head_size % 16 != 0
                  (FINDINGS 1); this lifts that one precondition. It also passes the builder's own option
                  fuse_qkv=false: the separate-RoPE path is DML's, which never packs Q/K/V, and with packing
                  the builder wires the RotaryEmbedding nodes to an empty input (an invalid model).
  no_gqa          Model.is_gqa_supported() and is_packed_attn_supported() -> False: the builder's remaining
                  attention, the contrib MultiHeadAttention op, for head sizes CPU GQA refuses outright
                  (head_size % 8 != 0, FINDINGS 1). (The packed Attention op, the builder's second choice, gives
                  a graph without its present.N.key outputs on CPU: ONNX Runtime refuses to load it.)

--eos-fallback ID sets the configuration's eos_token_id when neither the configuration nor the tokenizer has
one (the suite's tp_ files declare none, and the builder stops with "Could not resolve eos_token_id"). The
suite only appends ids and reads logits, so the id is never acted on; the adapter passes the file's unknown
id.
  gemma3_config   for model_type gemma3_text, the configuration transformers 4.x gives the builder:
                  rope_parameters = the full-attention layers' (flat) and rope_local_base_freq = the sliding
                  layers' theta. Under transformers 5 rope_parameters is nested per layer type and the
                  builder's Gemma3Model stops with KeyError 'rope_type' (FINDINGS 2).

Two more patches exist only for probe.py's attribution of Gemma 3 (each is one candidate fix):
  local_rope_unscaled   Gemma3Model's local (sliding-layer) RoPE caches without the global layers' linear
                        factor (FINDINGS 3).
  window_minus_one      GroupQueryAttention's local_window_size = the configuration's sliding_window - 1
                        (FINDINGS 4).

--input a .gguf file uses the builder's own GGUF weight loader (loaders/gguf.py) with the configuration of
--config (the keramos export's config.json), as `builder -m CONFIG_DIR -i MODEL.gguf` does.

Exit status 0 on success; otherwise the last line of stderr is "BUILDER-REFUSED: <reason>" when the builder
declined the architecture by design (NotImplementedError), or "BUILDER-ERROR: <type>: <message>".
"""
from __future__ import annotations

import argparse
import sys
import traceback

PATCHES = {"none", "unfused_rope", "no_gqa", "gemma3_config", "local_rope_unscaled", "window_minus_one"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--config")
    ap.add_argument("--output", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--patch", default="none")
    ap.add_argument("--eos-fallback", type=int)
    a = ap.parse_args(argv)
    patches = set(a.patch.split(",")) - {"none", ""}
    if patches - PATCHES:
        ap.error(f"unknown patch {sorted(patches - PATCHES)}")

    import torch
    torch.set_num_threads(2)
    import onnxruntime_genai.models  # noqa: F401  (puts the builder's directory on sys.path)
    from onnxruntime_genai.models import builder as B
    from builders.base import Model  # the module object the builder's classes use

    if "unfused_rope" in patches:
        Model.is_fused_rope_supported = lambda self: False
    if "no_gqa" in patches:
        Model.is_gqa_supported = lambda self: False
        Model.is_packed_attn_supported = lambda self: False
    if "local_rope_unscaled" in patches:
        from builders.gemma import Gemma3Model

        def multi_cache(self):
            # Gemma3Model.make_rotary_embedding_multi_cache, with the linear factor reset for the local caches.
            self.cos_cache_global_name, self.sin_cache_global_name = "cos_cache_global", "sin_cache_global"
            super(Gemma3Model, self).make_rotary_embedding_caches(
                cos_cache_name=self.cos_cache_global_name, sin_cache_name=self.sin_cache_global_name)
            self.rope_attrs["create_caches"] = True
            self.rope_attrs["theta"] = self.rope_local_theta
            self.rope_attrs["rescale_factors"] = 1
            self.cos_cache_local_name, self.sin_cache_local_name = "cos_cache_local", "sin_cache_local"
            super(Gemma3Model, self).make_rotary_embedding_caches(
                cos_cache_name=self.cos_cache_local_name, sin_cache_name=self.sin_cache_local_name)

        Gemma3Model.make_rotary_embedding_multi_cache = multi_cache
    if "window_minus_one" in patches:
        orig_attrs = Model.get_attention_op_attributes

        def attrs(self, **kw):
            at = orig_attrs(self, **kw)
            if at.get("local_window_size", -1) > 0:
                at["local_window_size"] -= 1
            return at

        Model.get_attention_op_attributes = attrs

    model_name = a.config if a.input.endswith(".gguf") else None
    # The builder's separate-RoPE path is the DML one, which never packs Q/K/V (is_packed_matmul_supported is
    # false on DML): with packing on, the RotaryEmbedding nodes get an empty input. fuse_qkv=false is the
    # builder's own option for unpacked projections.
    extra = ["fuse_qkv=false"] if "unfused_rope" in patches else None
    try:
        opts = B.parse_extra_options(model_name, a.input, a.output, "fp32", "cpu", a.cache, extra)
        cfg = opts["hf_details"]["hf_config"]
        if a.eos_fallback is not None and getattr(cfg, "eos_token_id", None) is None:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(a.config or a.input)
            if tok.eos_token_id is None:
                cfg.eos_token_id = a.eos_fallback
                print(f"eos_token_id: none in the configuration or tokenizer; set to {a.eos_fallback}", flush=True)
        if "gemma3_config" in patches:
            rp = getattr(cfg, "rope_parameters", None)
            if getattr(cfg, "model_type", "") == "gemma3_text" and isinstance(rp, dict) and "full_attention" in rp:
                cfg.rope_parameters = dict(rp["full_attention"])
                cfg.rope_local_base_freq = rp["sliding_attention"]["rope_theta"]
                print(f"gemma3_config: rope_parameters {cfg.rope_parameters}, rope_local_base_freq "
                      f"{cfg.rope_local_base_freq}", flush=True)
        B.create_model(model_name, a.input, a.output, "fp32", "cpu", a.cache, **opts)
    except NotImplementedError as e:
        traceback.print_exc()
        print(f"BUILDER-REFUSED: {e}", file=sys.stderr)
        return 3
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        print(f"BUILDER-ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
