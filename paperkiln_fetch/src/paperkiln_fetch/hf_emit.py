# -*- coding: utf-8 -*-
"""Arch -> Hugging Face config.json, with provenance riding inside it.

The output is a config that ``transformers.AutoConfig.from_pretrained``
opens directly (unknown keys are preserved as attributes), where every
extracted value's evidence snippet travels WITH the config under the
``_paperkiln`` key, and every defaulted field is named there instead of
being silently guessed — the same never-guess-silently contract as the
extractor itself.

Family selection is flavor-driven and conservative:
  rmsnorm / swiglu-or-silu / rope    -> llama-family (LlamaConfig keys)
  otherwise                          -> gpt2-family  (GPT2Config keys)
Contested flavor verdicts are never auto-applied; they are treated as
unresolved and the default is declared.
"""
from __future__ import annotations

from ._fetch import Arch, refuse_unsupported_moe

GENERATOR = "paperkiln-fetch"


def _value(arch: Arch, key: str):
    """Extracted value for key, or None. Contested flavor verdicts are
    refused (reported via provenance, never applied)."""
    f = arch.fields.get(key)
    if f is None:
        return None
    if key in arch.unresolved or f.verdict not in (None, "used", "inherited"):
        return None
    return f.value


def _prov(arch: Arch) -> dict:
    ev = {}
    for k, f in arch.fields.items():
        entry = {"value": f.value, "evidence": f.evidence}
        if f.verdict:
            entry["verdict"] = f.verdict
        if f.runner_up:
            entry["runner_up"] = f.runner_up
        ev[k] = entry
    p = {
        "generator": GENERATOR,
        "arxiv_id": arch.arxiv_id,
        "title": arch.title,
        "fields": ev,
        "unresolved": list(arch.unresolved),
    }
    if arch.inherits:
        p["inherits"] = arch.inherits
    return p


def pick_family(arch: Arch) -> str:
    norm = _value(arch, "norm")
    act = _value(arch, "activation")
    pos = _value(arch, "positional")
    llama_votes = sum([
        norm == "rmsnorm",
        act in ("swiglu", "silu", "geglu"),
        pos == "rope",
    ])
    return "llama" if llama_votes >= 2 else "gpt2"


def build_hf_config(arch: Arch) -> tuple[dict, str, dict]:
    """Returns (config_dict, family, defaults) where defaults maps every
    field that had to fall back to a family default (name -> value)."""
    refuse_unsupported_moe(arch)
    family = pick_family(arch)
    defaults: dict = {}

    # These two skeletons cannot represent every mechanism the extractor
    # recognizes. Metadata extraction remains available when config emission
    # refuses; never replace an explicitly stated mechanism with a default.
    supported = {
        "llama": {"norm": {"rmsnorm"}, "positional": {"rope"},
                  "activation": {"swiglu", "geglu"}},
        "gpt2": {"norm": {"layernorm"}, "positional": {"learned"},
                 "activation": {"gelu", "relu"}},
    }
    for key, values in supported[family].items():
        value = _value(arch, key)
        if value is not None and value not in values:
            raise ValueError(f"{family} HF config cannot represent {key}={value}; use JSON metadata")
    for key, neutral in (("attention", {"exact", "dense"}),
                         ("residual", {"residual"})):
        value = _value(arch, key)
        if value is not None and value not in neutral:
            raise ValueError(f"{family} HF config cannot represent {key}={value}; use JSON metadata")
    for key in ("window", "n_sinks"):
        value = _value(arch, key)
        if value not in (None, 0):
            raise ValueError(f"{family} HF config cannot represent {key}={value}; use JSON metadata")

    def get(key, fallback):
        v = _value(arch, key)
        if v is None:
            defaults[key] = fallback
            return fallback
        return v

    def positive(key, fallback):
        value = get(key, fallback)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
        return value

    d = positive("d_model", 768)
    layers = positive("n_layers", 12)
    heads = positive("n_heads", 12)
    vocab = positive("vocab_size", 32000 if family == "llama" else 50257)
    ctx = positive("context_length", 2048 if family == "llama" else 1024)
    dff = positive("d_ff", 4 * d)
    if d % heads:
        raise ValueError("d_model must be divisible by n_heads")

    if family == "llama":
        kv = positive("n_kv_heads", heads)
        if heads % kv:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if (d // heads) % 2:
            raise ValueError("RoPE requires an even head dimension")
        activation = get("activation", "swiglu")
        cfg = {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": d,
            "intermediate_size": dff,
            "num_hidden_layers": layers,
            "num_attention_heads": heads,
            "num_key_value_heads": kv,
            "vocab_size": vocab,
            "max_position_embeddings": ctx,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "hidden_act": "gelu" if activation == "geglu" else "silu",
            "tie_word_embeddings": False,
            "attention_bias": False,
            "mlp_bias": False,
            "torch_dtype": "float32",
        }
    else:
        kv = _value(arch, "n_kv_heads")
        if kv is not None and (isinstance(kv, bool) or not isinstance(kv, int) or kv != heads):
            raise ValueError("GPT2 HF config cannot represent grouped-query attention")
        activation = get("activation", "gelu")
        cfg = {
            "architectures": ["GPT2LMHeadModel"],
            "model_type": "gpt2",
            "n_embd": d,
            "n_inner": dff,
            "n_layer": layers,
            "n_head": heads,
            "vocab_size": vocab,
            "n_positions": ctx,
            "n_ctx": ctx,
            "activation_function": "relu" if activation == "relu" else "gelu",
            "layer_norm_epsilon": 1e-5,
            "torch_dtype": "float32",
        }

    # Architecture assumptions and non-extracted config constants are visible
    # alongside missing dimensions, rather than being hidden inside HF defaults.
    for key, value in (("norm", "rmsnorm" if family == "llama" else "layernorm"),
                       ("positional", "rope" if family == "llama" else "learned")):
        get(key, value)
    constants = ("rms_norm_eps", "rope_theta", "tie_word_embeddings", "attention_bias", "mlp_bias") if family == "llama" else ("layer_norm_epsilon",)
    defaults.update({key: cfg[key] for key in constants})
    if activation in ("gelu", "geglu"):
        # The rule-based label does not determine an author's approximation.
        defaults["gelu_approximation"] = "exact"
    prov = _prov(arch)
    prov["family"] = family
    prov["scope"] = "Configuration skeleton; not weight conversion or verified paper reconstruction."
    prov["defaults_applied"] = defaults
    cfg["_paperkiln"] = prov
    return cfg, family, defaults
