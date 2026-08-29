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

from ._fetch import Arch

GENERATOR = "paperkiln-fetch"


def _value(arch: Arch, key: str):
    """Extracted value for key, or None. Contested flavor verdicts are
    refused (reported via provenance, never applied)."""
    f = arch.fields.get(key)
    if f is None:
        return None
    if f.verdict == "contested":
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
    family = pick_family(arch)
    defaults: dict = {}

    def get(key, fallback):
        v = _value(arch, key)
        if v is None:
            defaults[key] = fallback
            return fallback
        return v

    d = get("d_model", 768)
    layers = get("n_layers", 12)
    heads = get("n_heads", 12)
    vocab = get("vocab_size", 32000 if family == "llama" else 50257)
    ctx = get("context_length", 2048 if family == "llama" else 1024)

    if family == "llama":
        kv = _value(arch, "n_kv_heads")
        if kv is None:
            defaults["n_kv_heads"] = heads
            kv = heads
        dff = _value(arch, "d_ff")
        if dff is None:
            defaults["d_ff"] = 4 * d
            dff = 4 * d
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
            "hidden_act": "silu",
            "tie_word_embeddings": False,
            "attention_bias": False,
            "mlp_bias": False,
            "torch_dtype": "float32",
        }
    else:
        dff = _value(arch, "d_ff")
        if dff is None:
            defaults["d_ff"] = 4 * d
            dff = 4 * d
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
            "activation_function": "gelu_new",
            "layer_norm_epsilon": 1e-5,
            "torch_dtype": "float32",
        }

    prov = _prov(arch)
    prov["family"] = family
    prov["defaults_applied"] = defaults
    cfg["_paperkiln"] = prov
    return cfg, family, defaults
