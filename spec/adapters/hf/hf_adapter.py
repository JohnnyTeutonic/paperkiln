#!/usr/bin/env python3
"""Hugging Face transformers against the keramos conformance suite: an in-process adapter (IR_SPEC 11.3).

    python adapters/hf/hf_adapter.py --adapter hf --json OUT.json [--only tp_] [--hf-attn sdpa|eager]
                                     [--hf-positions implicit|explicit] [--hf-export-dir DIR]
                                     [--hf-loader export|gguf] [--hf-gguf-name NAME]

--hf-loader gguf measures transformers' own GGUF reader (from_pretrained(gguf_file=...)) in place of the
keramos exporter; tp_mistral is then given as llama.cpp's convert writes Mistral (architecture "llama",
general.name NAME, default "mistral").

Run from the keramos root with the pinned environment (requirements.txt beside this file): CPU torch,
float32, torch.set_num_threads(2). transformers never reads the IR document: each GGUF is exported to an
HF checkpoint by adapters/common/gguf_to_hf.py (cached by the file's sha256 under --hf-export-dir) and
loaded with AutoModelForCausalLM, as a user of a converted checkpoint would. So the suite measures what
transformers computes for the architecture the GGUF names, against the document's statement of it.

Methods (FINDINGS.txt has the reasoning):
  open      export + from_pretrained (float32, the attention implementation named); skip for architecture
            "paperkiln" (no transformers class: IR_SPEC 9.1), for a file whose own document carries semantics
            its architecture cannot (an op, window, sinks, mask or GELU form the HF config has no field
            for: pk_gpt2_swa, pk_kimi, pk_recurrence), for a replacement document (ir) and for evict true.
  forward   model(input_ids, past_key_values / cache_params, use_cache=True); positions from the cache
            (implicit, how a caller who passes only the cache gets them) or position_ids len .. len+n-1
            (explicit). logits [n, vocab]; kv_rows / kv_positions from each layer's cached keys (a sliding
            layer holds fewer: the suite does not check windowed slots); recurrent from Mamba's
            LinearAttentionLayer (conv: the last K-1 of HF's K cached inputs, transposed to the IR's
            [K-1, Di]; ssm: [Di, N]); depth [] (no depth_route).
  truncate  Cache.crop(-(len - n)) (transformers 5's signature: a negative count of tokens to remove);
            where transformers refuses (a sliding layer past its window, a recurrent layer: RuntimeError)
            the session re-prefills ids[:n], as IR_SPEC 6.9 prescribes. The result's "_how" says which.
  snapshot, restore  copy.deepcopy of the cache, with the ids.
  check, run_node, run_node_grad, digest, render_chat, dequantize: skip (transformers has no IR
            validation, single-node execution, digest or IR template records; its GGUF dequantisation is
            gguf-py's, not its own).
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "adapters" / "common"))
import ir_conformance as rc  # noqa: E402
import ir_validate  # noqa: E402
import gguf_to_hf as g2h  # noqa: E402

THREADS = 2

# What each architecture's HF class can express (the IR ops of tools/gguf_ir.py's graph for it).
OPS = {
    "llama": {"embedding", "rmsnorm", "linear", "rope", "attention", "silu", "mul", "add"},
    "mistral": {"embedding", "rmsnorm", "linear", "rope", "attention", "silu", "mul", "add"},
    "qwen3": {"embedding", "rmsnorm", "linear", "rope", "attention", "silu", "mul", "add"},
    "gemma3": {"embedding", "affine", "rmsnorm", "linear", "rope", "attention", "gelu", "mul", "add"},
    "gptneox": {"embedding", "layernorm", "linear", "split", "rope", "attention", "gelu", "add"},
    "gpt2": {"embedding", "pos_embedding", "layernorm", "linear", "split", "attention", "gelu", "add"},
    "mamba": {"embedding", "rmsnorm", "linear", "split", "conv1d_causal", "silu", "selective_scan", "mul", "add"},
}
GELU = {"gpt2": "tanh", "gemma3": "tanh", "gptneox": "none"}


def ir_only_reason(doc, meta):
    """Why the file's own document says more than its architecture's HF config can (None when it does not)."""
    arch = meta.get("general.architecture")
    ops = OPS[arch] | ({"moe_ffn"} if meta.get(f"{arch}.expert_count") else set())
    nodes = [n for sec in ("embed", "block", "head") for n in doc[sec]["nodes"]]
    extra = sorted({n["op"] for n in nodes} - ops)
    if extra:
        return f"its document has {', '.join(extra)}, which no {arch} configuration in transformers carries"
    for n in nodes:
        if n["op"] == "attention":
            if n.get("sinks"):
                return f"attention sinks {n['sinks']} (no {arch} configuration in transformers has sinks)"
            if n.get("mask", "causal") != "causal":
                return f"attention mask {n['mask']}"
            win = n.get("window") or 0
            want = meta.get(f"{arch}.attention.sliding_window", 0) if arch in ("mistral", "gemma3") else 0
            if isinstance(win, dict):
                win = max(win["per_layer"])
            if win != want:
                return f"attention window {win} where the {arch} architecture has {want or 'none'}"
        if n["op"] == "gelu" and n.get("approx") != GELU.get(arch, n.get("approx")):
            return f"gelu {n.get('approx')} where transformers' {arch} uses {GELU[arch]}"
        if n["op"] == "moe_ffn" and (n.get("gate_norm") != "renorm" or n.get("activation", "silu") != "silu"):
            return "a moe_ffn other than Mixtral's (renormalised top-k, SiLU experts)"
        if n["op"] == "rope" and n.get("rope_heads", "all") != "all":
            return "rope on the first head only"
    return None


class HFAdapter(rc.Adapter):
    name, backend, exact = "transformers", "cpu", False
    profiles = {"core", "thirdparty", "state"}

    def __init__(self, attn="sdpa", positions="implicit", export_dir=None, truncate="crop", loader="export",
                 gguf_name="mistral"):
        import torch
        import transformers
        torch.set_num_threads(THREADS)
        torch.manual_seed(0)
        transformers.logging.set_verbosity_error()
        transformers.utils.logging.disable_progress_bar()
        self.torch, self.tf = torch, transformers
        self.attn, self.positions, self.truncate_mode, self.loader = attn, positions, truncate, loader
        self.gguf_mistral_name = gguf_name
        self.export_dir = Path(export_dir or os.path.expanduser("~/hf_keramos/export"))
        self.models, self.sessions, self.n = {}, {}, 0
        self.load_notes = {}

    def hello(self):
        return {"engine": self.name, "version": f"transformers {self.tf.__version__}, torch {self.torch.__version__}",
                "backend": self.backend, "exact": self.exact, "profiles": sorted(self.profiles),
                "settings": {"attn_implementation": self.attn, "positions": self.positions, "dtype": "float32",
                             "threads": THREADS, "truncate": self.truncate_mode, "loader": self.loader,
                             "exporter": f"adapters/common/gguf_to_hf.py v{g2h.EXPORTER_VERSION}"},
                "methods": ["hello", "open", "forward", "truncate", "snapshot", "restore", "close"]}

    # ------------------------------------------------------------ loading

    def _model(self, gguf):
        data = Path(gguf).read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if sha in self.models:
            return self.models[sha]
        meta_raw, *_ = ir_validate.read_gguf(gguf)
        meta = {k: v[1] for k, v in meta_raw.items()}
        arch = meta.get("general.architecture")
        if arch not in g2h.FAMILIES:
            raise NotImplementedError(f"general.architecture {arch!r}: no transformers class (IR_SPEC 9.1: a file "
                                      f"no llama.cpp-architecture graph computes)")
        if "paperkiln.ir" in meta:
            why = ir_only_reason(json.loads(meta["paperkiln.ir"]), meta)
            if why:
                raise NotImplementedError(f"IR-only semantics: {why}; transformers would compute plain {arch} "
                                          f"(llama.cpp FINDINGS 7, K1)")
        torch = self.torch
        kw = {} if arch == "mamba" else {"attn_implementation": self.attn}  # Mamba has no attention to choose
        rewrites = []
        if self.loader == "gguf":
            # transformers' own GGUF reader (modeling_gguf_pytorch_utils): its config mapping and tensor
            # processors in place of the keramos exporter.
            src = Path(gguf)
            if arch == "mistral":
                # gguf-py has no architecture "mistral" (nor llama.cpp: its convert writes Mistral as "llama");
                # transformers maps a "llama" file whose general.name is "mistral" to its Mistral class.
                src = self._as_llama(gguf, sha)
                rewrites.append(f"general.architecture mistral -> llama, mistral.* -> llama.*, general.name "
                                f"{self.gguf_mistral_name!r} (a copy; tensors untouched)")
            out, kw["gguf_file"] = src.parent, src.name
            if arch == "gptneox":
                raise NotImplementedError("transformers' GGUF reader has no gptneox (GGUF_SUPPORTED_ARCHITECTURES)")
        else:
            out = self.export_dir / f"{Path(gguf).stem}-{sha[:12]}"
            if not (out / "config.json").is_file():
                g2h.export(gguf, out)
        model, info = self.tf.AutoModelForCausalLM.from_pretrained(
            str(out), dtype=torch.float32, output_loading_info=True, **kw)
        model.eval()
        bad = {k: sorted(v) for k, v in info.items() if v and k in ("missing_keys", "unexpected_keys", "mismatched_keys")}
        if bad:
            raise RuntimeError(f"{type(model).__name__} does not load cleanly from {self.loader}: {bad}")
        notes = {"loaded_from": str(out), "loader": self.loader, "class": type(model).__name__,
                 "attn_implementation": model.config._attn_implementation, "rewrites": rewrites}
        notes.update(self._effective_check(model, meta, gguf))
        entry = {"model": model, "arch": arch, "mamba": arch == "mamba", "notes": notes,
                 "vocab": int(model.config.vocab_size), "layers": int(model.config.num_hidden_layers)}
        self.models[sha] = entry
        self.load_notes[Path(gguf).stem] = notes
        return entry

    def _as_llama(self, gguf, sha):
        meta, tensors, data_start, raw, _ = ir_validate.read_gguf(gguf)
        W = {}
        for name, dims, typ, off in tensors:
            n = int(np.prod(dims))
            W[name] = np.frombuffer(raw, dtype="<f4", count=n, offset=data_start + off).reshape(list(reversed(dims)))
        m2 = {("llama" + k[len("mistral"):] if k.startswith("mistral.") else k): v for k, v in meta.items()}
        m2["general.architecture"] = (8, "llama")
        m2["general.name"] = (8, self.gguf_mistral_name)
        doc = json.loads(meta["paperkiln.ir"][1]) if "paperkiln.ir" in meta else None
        tag = "".join(ch if ch.isalnum() else "_" for ch in self.gguf_mistral_name)
        out = self.export_dir / f"{Path(gguf).stem}-{sha[:12]}-as-llama-{tag}" / Path(gguf).name
        if not out.is_file():
            out.parent.mkdir(parents=True, exist_ok=True)
            rc.write_gguf(out, m2, W, doc)
        return out

    def _effective_check(self, model, meta, gguf):
        """The weights transformers computes with against the GGUF's, where the export rewrote them:
        Gemma's 1 + w and Mamba's -exp(A_log), in torch float32."""
        torch = self.torch
        _, W = g2h.read(gguf)
        arch = meta["general.architecture"]
        out = {}
        if arch == "gemma3":
            diff = 0
            for n, p in model.named_parameters():
                if n.endswith("norm.weight"):
                    eff = (1.0 + p.detach().float()).numpy()
                    gname = _gemma_gguf_name(n)
                    diff += int(np.sum(eff.view(np.uint32) != W[gname].view(np.uint32)))
            out["gemma_norm_values_not_bit_identical"] = diff
        if arch == "mamba":
            diff, worst = 0, 0.0
            for i in range(model.config.num_hidden_layers):
                a = (-torch.exp(model.backbone.layers[i].mixer.A_log.detach().float())).numpy()
                ref = W[f"blk.{i}.ssm_a"]
                diff += int(np.sum(a.view(np.uint32) != ref.view(np.uint32)))
                worst = max(worst, float(np.max(np.abs(a - ref) / np.abs(ref))))
            out["mamba_A_values_not_bit_identical"] = diff
            out["mamba_A_worst_relative"] = worst
        return out

    # ------------------------------------------------------------ sessions

    def open(self, gguf, ir=None, evict=None):
        if ir is not None:
            raise NotImplementedError("a replacement document: transformers runs the GGUF's architecture, never a document")
        if evict is True:
            raise NotImplementedError("evict true: transformers has no state tokens or turn_state eviction")
        m = self._model(gguf)
        self.n += 1
        sid = f"s{self.n}"
        self.sessions[sid] = {"m": m, "ids": [], "cache": None, "snap": {}}
        return {"session": sid, "vocab": m["vocab"], "applications": m["layers"], "state_k": 0,
                "_notes": m["notes"]}

    def close(self, session):
        self.sessions.pop(session, None)

    def _run(self, s, ids, cache):
        """(logits [n, vocab] numpy, the cache after) of ids appended to a session holding len(s['ids']) tokens."""
        torch = self.torch
        m = s["m"]
        x = torch.tensor([list(ids)], dtype=torch.long)
        with torch.no_grad():
            if m["mamba"]:
                out = m["model"](input_ids=x, cache_params=cache, use_cache=True)
                cache = out.cache_params
            else:
                kw = {}
                if self.positions == "explicit":
                    L0 = len(s["ids"])
                    kw["position_ids"] = torch.arange(L0, L0 + len(ids), dtype=torch.long)[None]
                out = m["model"](input_ids=x, past_key_values=cache, use_cache=True, **kw)
                cache = out.past_key_values
        return out.logits[0].detach().float().numpy().astype(np.float32), cache

    def _report(self, s, report, logits=None):
        res = {"len": len(s["ids"])}
        if "logits" in report:
            res["logits"] = logits
        if "depth" in report:
            res["depth"] = []
        cache, m, L = s["cache"], s["m"], len(s["ids"])
        if "kv_rows" in report or "kv_positions" in report:
            if m["mamba"]:
                rows = None
            elif cache is None:
                rows = [0] * m["layers"]
            else:
                rows = [int(layer.keys.shape[-2]) if getattr(layer, "is_initialized", False) and layer.keys is not None
                        else 0 for layer in cache.layers]
            if "kv_rows" in report:
                res["kv_rows"] = {} if rows is None else {"kv": rows}
            if "kv_positions" in report:
                res["kv_positions"] = {} if rows is None else {"kv": [list(range(L - r, L)) for r in rows]}
        if "recurrent" in report:
            if not m["mamba"]:
                res["recurrent"] = {}
            else:
                res["recurrent"] = self._mamba_state(s)
        return res

    def _mamba_state(self, s):
        m, cache = s["m"], s["cache"]
        cfg = m["model"].config
        K, Di, N = cfg.conv_kernel, cfg.intermediate_size, cfg.state_size
        conv, ssm = [], []
        for i in range(m["layers"]):
            layer = cache.layers[i] if cache is not None else None
            if layer is None or not layer.is_conv_states_initialized[0]:
                conv.append({"x": np.zeros((K - 1, Di), np.float32)})
                ssm.append({"s": np.zeros((Di, N), np.float32)})
                continue
            c = layer.conv_states[0][0].detach().float().numpy()  # [Di, K] (more when recording the past)
            conv.append({"x": np.ascontiguousarray(c[:, -(K - 1):].T, np.float32)})
            ssm.append({"s": layer.recurrent_states[0][0].detach().float().numpy().astype(np.float32)})
        return {"conv": conv, "ssm": ssm}

    def forward(self, session, ids, report):
        s = self.sessions[session]
        if not ids:
            raise RuntimeError("forward: ids is empty (IR_SPEC 11.3)")
        before = copy.deepcopy(s["cache"])
        try:
            logits, cache = self._run(s, ids, s["cache"])
        except Exception as e:
            s["cache"] = before
            raise RuntimeError(f"transformers forward: {type(e).__name__}: {e}") from e
        s["cache"] = cache
        s["ids"] = s["ids"] + list(ids)
        return self._report(s, report, logits)

    def truncate(self, session, n, report):
        s = self.sessions[session]
        L = len(s["ids"])
        if not 0 <= n <= L:
            raise RuntimeError(f"truncate to {n}, outside 0 .. len {L} (IR_SPEC 11.3)")
        how = "nothing to remove"
        if n < L:
            cache = copy.deepcopy(s["cache"])
            how = None
            if self.truncate_mode == "crop" and n > 0 and cache is not None:
                try:
                    cache.crop(n - L)
                    how = f"Cache.crop({n - L})"
                except (RuntimeError, ValueError) as e:
                    how = f"re-prefill of {n} ids (Cache.crop refused: {str(e)[:90]})"
            if how is None or how.startswith("re-prefill"):
                cache = None
                if n > 0:
                    keep = {"ids": []}
                    keep["m"] = s["m"]
                    _, cache = self._run(keep, s["ids"][:n], None)
                how = how or (f"re-prefill of {n} ids" if n else "cleared")
            s["cache"] = cache
            s["ids"] = s["ids"][:n]
        res = self._report(s, [r for r in report if r != "logits"])
        res["_how"] = how
        return res

    def snapshot(self, session, tag):
        s = self.sessions[session]
        s["snap"][tag] = (list(s["ids"]), copy.deepcopy(s["cache"]))
        return {"len": len(s["ids"])}

    def restore(self, session, tag, report):
        s = self.sessions[session]
        if tag not in s["snap"]:
            raise RuntimeError(f"restore: no snapshot {tag!r} in this session")
        ids, cache = s["snap"][tag]
        s["ids"], s["cache"] = list(ids), copy.deepcopy(cache)
        return self._report(s, [r for r in report if r != "logits"])

    # ------------------------------------------------------------ what transformers has no path for

    def check(self, gguf, ir):
        raise NotImplementedError("transformers has no IR validation (it never reads the document)")

    def run_node(self, node, inputs, weights, state):
        raise NotImplementedError("transformers runs whole models, not single IR nodes")

    def run_node_grad(self, node, inputs, weights, out_grad):
        raise NotImplementedError("transformers runs whole models, not single IR nodes")

    def digest(self, text, prose=False):
        raise NotImplementedError("transformers has no IR digest")

    def render_chat(self, record, messages, generation_prompt=False, thinking=None):
        raise NotImplementedError("transformers renders Jinja chat templates, not IR template records")

    def dequantize(self, type_name, n, raw):
        raise NotImplementedError("transformers dequantises GGUF through gguf-py, not code of its own")


def _gemma_gguf_name(hf_name):
    if hf_name == "model.norm.weight":
        return "output_norm.weight"
    i = hf_name.split(".")[2]
    rest = hf_name.split(".", 3)[3]
    table = {"self_attn.q_norm.weight": "attn_q_norm.weight", "self_attn.k_norm.weight": "attn_k_norm.weight",
             **{h + ".weight": g + ".weight" for g, h in g2h.GEMMA3_NORMS.items()}}
    return f"blk.{i}.{table[rest]}"


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    opts = {"attn": "sdpa", "positions": "implicit", "export_dir": None, "truncate": "crop", "loader": "export",
            "gguf_name": "mistral"}
    for flag, key in (("--hf-attn", "attn"), ("--hf-positions", "positions"), ("--hf-export-dir", "export_dir"),
                      ("--hf-truncate", "truncate"), ("--hf-loader", "loader"),
                      ("--hf-gguf-name", "gguf_name")):
        if flag in argv:
            i = argv.index(flag)
            opts[key] = argv[i + 1]
            del argv[i:i + 2]
    return rc.main(argv, adapters={"hf": lambda: HFAdapter(**opts)})


if __name__ == "__main__":
    sys.exit(main())
